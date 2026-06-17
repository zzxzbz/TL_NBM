"""XGBoost-based ROP training script.

Supports three training modes (aligned with the three MLP modes in trainCL.py):

1. ``global`` —— Cross-well training.
   Multiple training wells are merged to train a single global XGBoost model,
   which then produces predictions on test wells in one shot.

2. ``cl`` —— In-well sliding window + continual learning.
   Within a single well, sliding-window splits are performed in the same way as
   trainCL.train_test_val_split_slide.
   **Each window**: fits a candidate model on ``train_df`` only; uses RMSE on
   ``val_df`` to **select the best model** (including a "cold start vs previous
   window warm start" two-way choice, and optional XGB early stopping for best
   rounds); **only the best model is used to predict on ``test_df``**.
   The tree-structure strategy is controlled by ``--cl-update-mode``:

     * ``refresh`` —— **NN-style**, aligned with the "tune parameters, not
       structure" approach of BPNN/NBM: does not add new trees, but uses
       XGBoost's ``process_type=update`` + ``updater=refresh`` to recompute
       leaf weights on top of the existing tree structures. Model size stays
       constant; new data only rewrites parameters (leaf values), while split
       features and thresholds remain unchanged. **This is the default.**
     * ``append`` —— Classic boosting style: each window appends
       ``cl_round_estimators`` new trees after the booster, leaving old trees
       untouched. Model size grows linearly with the number of windows.

   The first window is a cold start that builds ``cl_round_estimators`` initial
   trees to serve as the "skeleton" for subsequent refreshes.

3. ``ft`` —— Residual fine-tuning based on a cross-well model.
   First loads the cross-well XGBoost trained in ``global`` mode as the base
   model f_base; within each sliding window, fits a residual model on
   ``train_df``, **selects the best using residual RMSE on ``val_df``**
   (cold start vs warm start, and optional early stopping), then predicts on
   ``test_df`` as: ``y_hat = f_base(x) + f_res(x)``.

Output columns are aligned with the eval scripts in trainCL.py:
``DMEA, ROPA_Pre1, ROPA_Real``.

Usage examples:

    # 1) Cross-well training (produces a global base model + test-well predictions)
    python trainCL_XGB.py --mode global

    # 2) In-well sliding window continual learning (for SZ36-1-Q15)
    python trainCL_XGB.py --mode cl
        --well-csv ./data/ProcessedData-2/SZ36-1-Q15.csv
        --start-depth 1120
    python trainCL_XGB.py --mode cl --well-csv ./data/ProcessedData-2/USROP-A1.csv --start-depth 300

    # 3) Residual fine-tuning based on the cross-well model
    python trainCL_XGB.py --mode ft
        --well-csv ./data/ProcessedData-2/SZ36-1-Q15.csv
        --start-depth 1120
        --base-model-path ./saved_models-XGB/global_model.pkl
    python trainCL_XGB.py --mode ft --well-csv ./data/ProcessedData-2/USROP-A1.csv --start-depth 300 --exp-tag A1
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from create_scaler import load_scaler
from cost_profile import CostProfileLogger, StageTimer, add_cost_profile_args

# Input feature columns — kept fully consistent with trainCL.py / dataset.dataLoader
INPUT_COLS = ['DMEA', 'WOBA', 'RPMA', 'MFIA', 'MDIA',
              'SPPA', 'BIT_DIAMETER', 'ANGLE', 'TQA_ON_BIT']
TARGET_COL = 'ROPA'

SCALER, _FEATURE_INFO = load_scaler()


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize input features using the shared scaler, returning a DataFrame with column names."""
    X = df[INPUT_COLS]
    X_scaled = SCALER.transform(X)
    return pd.DataFrame(X_scaled, columns=INPUT_COLS, index=df.index)


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def train_test_val_split():
    """Cross-well split — fully consistent with trainCL.train_test_val_split."""
    '''
    well_ids = ['B2','B3','B4','B6','B7','B8','B9','B10',' B11','B13','B14',
                'B15','B16H','B17H','B18','B19','B20H','B22H','B24H'] 
    val_well_ids = ['B4','B15', 'B22H','B18H']
    test_well_ids = ['B2','B3']
    '''
    well_ids = ['V0', 'V1', 'V2', 'V3', 'V4',
                'V5', 'V6']
    val_well_ids = ['V3']
    test_well_ids = ['V2']
    train_ids = [w for w in well_ids if w not in val_well_ids and w not in test_well_ids]
    base = './data/ProcessedData-2'
    return (
        [f'{base}/{w}.csv' for w in train_ids],
        [f'{base}/{w}.csv' for w in val_well_ids],
        [f'{base}/{w}.csv' for w in test_well_ids],
    )


def _well_tag(path: str) -> str:
    """./data/.../SZ36-1-Q2.csv -> SZ36-1-Q2"""
    return os.path.splitext(os.path.basename(path))[0]


def _test_set_tag(test_paths) -> str:
    """Generate an experiment label from the current test-well list: ['SZ36-1-Q2','SZ36-1-Q3'] -> 'Q2-Q3'.

    Used as a subdirectory name under ``model_dir``: when test wells change,
    models/artifacts automatically land in separate directories and will not
    overwrite each other.
    """
    return '-'.join(_short_well(_well_tag(p)) for p in test_paths)


def _load_concat(file_paths) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in file_paths], ignore_index=True)


def train_test_val_split_slide(i: int, data: pd.DataFrame, start_depth: float,
                               train_len: float = 20.0,
                               val_len: float = 10.0,
                               test_len: float = 10.0,
                               step: float = 10.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sliding-window logic — consistent with trainCL.train_test_val_split_slide.

    Default per window: train 20 m, val 10 m, test 10 m, slide 10 m each time.
    The depth column is fixed to ``DMEA`` (CSV column 2), equivalent to
    ``data.columns[1]`` in trainCL.py.
    """
    train_end = start_depth + train_len
    val_end = train_end + val_len
    test_end = val_end + test_len

    offset = i * step
    train = data[(data['DMEA'] >= start_depth + offset) & (data['DMEA'] < train_end + offset)]
    val = data[(data['DMEA'] >= train_end + offset) & (data['DMEA'] < val_end + offset)]
    test = data[(data['DMEA'] >= val_end + offset) & (data['DMEA'] < test_end + offset)]

    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"window i={i} has empty split: train={len(train)}, val={len(val)}, test={len(test)}"
        )
    return train, val, test


# ---------------------------------------------------------------------------
# Evaluation & output
# ---------------------------------------------------------------------------

def _print_metrics(tag: str, y_true: np.ndarray, y_pred: np.ndarray):
    if len(y_true) == 0:
        print(f'[{tag}] empty')
        return
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    eps = 1e-1
    mape = float(np.mean(np.abs((y_pred - y_true) / (y_true + eps))) * 100)
    print(f'[{tag}] RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  MAPE={mape:.2f}%')


def _val_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Validation-set RMSE, used for model selection within a sliding window."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _save_predictions(out_path: str, dmea: np.ndarray,
                      y_pred: np.ndarray, y_true: np.ndarray):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    df = pd.DataFrame({
        'DMEA': dmea,
        'ROPA_Pre1': y_pred,
        'ROPA_Real': y_true,
    }).sort_values('DMEA').reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f'predictions saved -> {out_path} (rows={len(df)})')


# ---------------------------------------------------------------------------
# Three training modes
# ---------------------------------------------------------------------------

def _make_xgb(**override) -> xgb.XGBRegressor:
    """Unified XGBoost default hyperparameters, so all three modes stay consistent."""
    base = dict(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse',
    )
    base.update(override)
    return xgb.XGBRegressor(**base)


def _booster_n_trees(booster) -> int:
    """Compatible across xgboost versions: returns the number of trees in a booster."""
    try:
        return int(booster.num_boosted_rounds())
    except AttributeError:
        return len(booster.get_dump())


def _fit_warmstart_xgb(X_train, y_train, X_val, y_val, *,
                       prev_booster, mode: str, n_round: int,
                       lr: float,
                       early_stopping_rounds: int | None = None,
                       n_estimators_cold_cap: int | None = None) -> xgb.XGBRegressor:
    """Train a new XGBRegressor according to the continual-learning strategy (fit on train only, val goes into eval_set).

    - ``prev_booster=None`` —— cold start; ``n_estimators_cold_cap``, if given,
      serves as the maximum boosting rounds for this round (used with
      ``early_stopping_rounds`` to stop early on val), otherwise it builds
      ``n_round`` new trees;
    - ``mode='refresh'`` —— refresh leaf weights on top of ``prev_booster``'s
      existing structure (early stopping is not used, consistent with the
      QuantileDMatrix / refresh limitation);
    - ``mode='append'`` —— append ``n_round`` new trees after ``prev_booster``;
      ``early_stopping_rounds`` may be applied to this round's appended segment.
    """
    fit_kwargs: dict = dict(eval_set=[(X_val, y_val)], verbose=False)
    if early_stopping_rounds is not None and early_stopping_rounds > 0 and mode != 'refresh':
        fit_kwargs['early_stopping_rounds'] = int(early_stopping_rounds)

    if prev_booster is None:
        n_est = int(n_estimators_cold_cap) if n_estimators_cold_cap is not None else int(n_round)
        model = _make_xgb(n_estimators=n_est, learning_rate=lr)
    elif mode == 'refresh':
        fit_kwargs.pop('early_stopping_rounds', None)
        n_existing = _booster_n_trees(prev_booster)
        model = _make_xgb(
            n_estimators=n_existing,
            learning_rate=lr,
            tree_method='exact',
            process_type='update',
            updater='refresh',
            refresh_leaf=1,
        )
        fit_kwargs['xgb_model'] = prev_booster
    elif mode == 'append':
        model = _make_xgb(n_estimators=n_round, learning_rate=lr)
        fit_kwargs['xgb_model'] = prev_booster
    else:
        raise ValueError(f"unknown cl_update_mode: {mode!r} (valid values: 'refresh' / 'append')")

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_global(args) -> str:
    """Cross-well training: merge multiple training wells + output predictions and metrics per test well.

    Artifacts are grouped by test-well set (``{model_dir}/{tag}/``,
    ``{csv_dir}/{tag}/``), where ``tag`` can be explicitly specified via
    ``--exp-tag``, otherwise it is auto-derived from the test wells in
    ``train_test_val_split`` (e.g. ``Q2-Q3``). This way, when you switch test
    wells, all artifacts automatically land in separate directories for easy
    distinction and plotting.
    """
    train_paths, val_paths, test_paths = train_test_val_split()
    tag = args.exp_tag or _test_set_tag(test_paths)
    print(f'[exp-tag] {tag}  (test wells: {[_well_tag(p) for p in test_paths]})')

    train_df = _load_concat(train_paths)
    val_df = _load_concat(val_paths)

    X_train, y_train = scale_features(train_df), train_df[TARGET_COL].to_numpy()
    X_val, y_val = scale_features(val_df), val_df[TARGET_COL].to_numpy()

    cost = CostProfileLogger.from_args(args, model='XGB', well=tag, mode='global')
    model = _make_xgb(early_stopping_rounds=50)
    with StageTimer() as t_fit:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    t_train = t_fit.elapsed

    _print_metrics('train', y_train, model.predict(X_train))
    _print_metrics('val', y_val, model.predict(X_val))

    # Model grouped by test-well set: ./saved_models-XGB/<tag>/global_model.pkl
    model_dir = os.path.join(args.model_dir, tag)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'global_model.pkl')
    joblib.dump(model, model_path)
    print(f'global model saved -> {model_path}')

    # CSVs also placed under the same tag directory: ./csv1/<tag>/predictions_results-<well>-XGB-Global.csv
    out_dir = os.path.join(args.csv_dir, tag)

    overall_true, overall_pred = [], []
    for tp in test_paths:
        well_tag = _well_tag(tp)            # e.g. SZ36-1-Q2
        short = _short_well(well_tag)       # e.g. Q2

        df_w = pd.read_csv(tp)
        with StageTimer() as t_scale:
            X_w = scale_features(df_w)
        y_w = df_w[TARGET_COL].to_numpy()
        with StageTimer() as t_inf:
            pred_w = model.predict(X_w)
        if cost.enabled:
            cost.log_global(
                n_train=len(X_train),
                n_val=len(X_val),
                n_test=len(y_w),
                stages={
                    'train_cold_s': t_train,
                    'scale_s': t_scale.elapsed,
                    'infer_test_s': t_inf.elapsed,
                },
                extra={'well_infer': short},
            )

        _print_metrics(f'test[{short}]', y_w, pred_w)

        # Output filename: prefer user-supplied --output (supports {well} placeholder), else auto-generate
        if args.output:
            if '{well}' in args.output:
                out_path = args.output.format(well=short)
            elif len(test_paths) == 1:
                out_path = args.output
            else:
                base, ext = os.path.splitext(args.output)
                out_path = f'{base}-{short}{ext}'
                print(f'[warn] --output does not contain a {{well}} placeholder, auto-renamed to {out_path}')
        else:
            out_path = os.path.join(
                out_dir, f'predictions_results-{short}-XGB-Global.csv')

        _save_predictions(out_path, df_w['DMEA'].to_numpy(), pred_w, y_w)

        overall_true.append(y_w)
        overall_pred.append(pred_w)

    if overall_true:
        _print_metrics('test[overall]',
                       np.concatenate(overall_true),
                       np.concatenate(overall_pred))
    cost.flush_summary()
    return model_path


def train_cl(args):
    """In-well sliding window + true continual learning (XGB warm start).

    All artifacts are grouped by current well name:
      - per-window models: ``{model_dir}/{short_well}-CL/{well_tag}-CL-{i:03d}.pkl``
      - concatenated prediction CSV: ``{csv_dir}/{short_well}/predictions_results-{short_well}-XGB-CL.csv``
    """
    well_data = pd.read_csv(args.well_csv)
    well_tag = os.path.splitext(os.path.basename(args.well_csv))[0]
    short = _short_well(well_tag)

    cl_model_dir = os.path.join(args.model_dir, f'{short}-CL')
    cl_csv_dir = os.path.join(args.csv_dir, short)
    cost = CostProfileLogger.from_args(args, model='XGB', well=short, mode='cl')

    n_loop = max(1, args.end - args.start)
    print(
        f'[CL] well {well_tag} | sliding-window index i in [{args.start}, {args.end}) '
        f'({n_loop} cycles) | start_depth={args.start_depth}',
        flush=True,
    )

    pieces = []
    prev_booster = None
    used = 0

    for i in range(args.start, args.end):
        wall0 = time.perf_counter()
        stages: dict[str, float] = {}
        try:
            with StageTimer() as t_split:
                train_df, val_df, test_df = train_test_val_split_slide(
                    i, well_data, start_depth=args.start_depth)
            stages['split_s'] = t_split.elapsed
        except ValueError as e:
            cost.skip_window()
            if args.verbose:
                print(f'[Skip] i={i}: {e}')
            continue

        ith = i - args.start + 1
        d_tr = (float(train_df['DMEA'].min()), float(train_df['DMEA'].max()))
        d_va = (float(val_df['DMEA'].min()), float(val_df['DMEA'].max()))
        d_te = (float(test_df['DMEA'].min()), float(test_df['DMEA'].max()))
        print(
            f'[CL] progress {ith}/{n_loop}  |  i={i}  |  '
            f'DMEA train[{d_tr[0]:.1f},{d_tr[1]:.1f}]  '
            f'val[{d_va[0]:.1f},{d_va[1]:.1f}]  '
            f'test[{d_te[0]:.1f},{d_te[1]:.1f}] m',
            flush=True,
        )

        with StageTimer() as t_scale:
            X_train = scale_features(train_df)
            y_train = train_df[TARGET_COL].to_numpy()
            X_val = scale_features(val_df)
            y_val = val_df[TARGET_COL].to_numpy()
            X_test = scale_features(test_df)
            y_test = test_df[TARGET_COL].to_numpy()
        stages['scale_s'] = t_scale.elapsed

        es = args.cl_early_stopping_rounds or 0
        es = int(es) if es > 0 else None
        cold_cap = args.cl_max_estimators if es is not None else None

        with StageTimer() as t_cold:
            cold = _fit_warmstart_xgb(
                X_train, y_train, X_val, y_val,
                prev_booster=None,
                mode=args.cl_update_mode,
                n_round=args.cl_round_estimators,
                lr=args.cl_lr,
                early_stopping_rounds=es,
                n_estimators_cold_cap=cold_cap,
            )
        stages['train_cold_s'] = t_cold.elapsed
        rmse_cold = _val_rmse(y_val, cold.predict(X_val))

        model = cold
        choice = 'cold'
        rmse_warm: float | None = None
        if prev_booster is not None and args.use_warm_start:
            with StageTimer() as t_warm:
                warm = _fit_warmstart_xgb(
                    X_train, y_train, X_val, y_val,
                    prev_booster=prev_booster,
                    mode=args.cl_update_mode,
                    n_round=args.cl_round_estimators,
                    lr=args.cl_lr,
                    early_stopping_rounds=es if args.cl_update_mode == 'append' else None,
                    n_estimators_cold_cap=None,
                )
            stages['train_warm_s'] = t_warm.elapsed
            rmse_warm = _val_rmse(y_val, warm.predict(X_val))
            if rmse_warm + 1e-12 < rmse_cold:
                model, choice = warm, 'warm'
            else:
                model, choice = cold, 'cold'

        with StageTimer() as t_sel:
            prev_booster = model.get_booster()
            val_rmse_best = _val_rmse(y_val, model.predict(X_val))
        stages['select_s'] = t_sel.elapsed

        with StageTimer() as t_inf:
            y_pred = model.predict(X_test)
        stages['infer_test_s'] = t_inf.elapsed
        stages['wall_window_s'] = time.perf_counter() - wall0
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        cost.log_window(
            window_i=i,
            n_train=len(train_df),
            n_val=len(val_df),
            n_test=len(test_df),
            stages=stages,
            picked=choice,
            extra={
                'cl_update_mode': args.cl_update_mode,
                'n_trees': _booster_n_trees(prev_booster),
            },
        )
        pieces.append(pd.DataFrame({
            'DMEA': test_df['DMEA'].to_numpy(),
            'ROPA_Pre1': y_pred,
            'ROPA_Real': y_test,
        }))
        used += 1
        if args.verbose:
            extra = f'  warm_val_rmse={rmse_warm:.4f}' if rmse_warm is not None else ''
            print(f'[i={i:03d}] pick={choice}  mode={args.cl_update_mode}  '
                  f'trees={_booster_n_trees(prev_booster)}  '
                  f'train={len(train_df)} val={len(val_df)} test={len(test_df)}  '
                  f'cold_val_rmse={rmse_cold:.4f}{extra}  '
                  f'best_val_rmse={val_rmse_best:.4f}  test_rmse={test_rmse:.4f}')

        # Selective persistence: save a checkpoint for each window for later review
        if args.save_per_window:
            os.makedirs(cl_model_dir, exist_ok=True)
            joblib.dump(model, os.path.join(
                cl_model_dir, f'{well_tag}-CL-{i:03d}.pkl'))

    if not pieces:
        raise RuntimeError('no window produced any predictions; check start_depth / data range')

    result = pd.concat(pieces, ignore_index=True).sort_values('DMEA').reset_index(drop=True)
    out_path = args.output or os.path.join(
        cl_csv_dir, f'predictions_results-{short}-XGB-CL.csv')
    _save_predictions(out_path, result['DMEA'].to_numpy(),
                      result['ROPA_Pre1'].to_numpy(),
                      result['ROPA_Real'].to_numpy())
    print(f'CL: {used} windows used')
    _print_metrics('overall', result['ROPA_Real'].to_numpy(), result['ROPA_Pre1'].to_numpy())
    cost.flush_summary()


def train_ft(args):
    """Residual fine-tuning based on a pre-trained cross-well XGB (window-level + warm start).

    Outputs are grouped by current well name:
    ``{csv_dir}/{short_well}/predictions_results-{short_well}-XGB-FT.csv``.
    The global base model path defaults to
    ``{model_dir}/{exp_tag}/global_model.pkl``;
    use ``--exp-tag`` to switch to a global model trained on a different set of
    test wells, or supply ``--base-model-path`` explicitly.
    """
    base_model_path = args.base_model_path
    if base_model_path is None:
        if args.exp_tag is None:
            raise SystemExit(
                'FT mode requires either --base-model-path or --exp-tag to locate the global base model.')
        base_model_path = os.path.join(args.model_dir, args.exp_tag, 'global_model.pkl')

    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f'global base model not found: {base_model_path}\n'
            f'please run first: python trainCL_XGB.py --mode global')

    base_model: xgb.XGBRegressor = joblib.load(base_model_path)
    print(f'[FT] base model loaded -> {base_model_path}')

    well_data = pd.read_csv(args.well_csv)
    well_tag = os.path.splitext(os.path.basename(args.well_csv))[0]
    short = _short_well(well_tag)

    ft_csv_dir = os.path.join(args.csv_dir, short)
    cost = CostProfileLogger.from_args(args, model='XGB', well=short, mode='ft')

    pieces = []
    prev_booster = None
    used = 0

    for i in range(args.start, args.end):
        wall0 = time.perf_counter()
        stages: dict[str, float] = {}
        try:
            with StageTimer() as t_split:
                train_df, val_df, test_df = train_test_val_split_slide(
                    i, well_data, start_depth=args.start_depth)
            stages['split_s'] = t_split.elapsed
        except ValueError as e:
            cost.skip_window()
            if args.verbose:
                print(f'[Skip] i={i}: {e}')
            continue

        with StageTimer() as t_scale:
            X_train = scale_features(train_df)
            X_val = scale_features(val_df)
            X_test = scale_features(test_df)
            y_train = train_df[TARGET_COL].to_numpy()
            y_val = val_df[TARGET_COL].to_numpy()
            y_test = test_df[TARGET_COL].to_numpy()
        stages['scale_s'] = t_scale.elapsed

        with StageTimer() as t_base:
            base_train = base_model.predict(X_train)
            base_val = base_model.predict(X_val)
            base_test = base_model.predict(X_test)
        stages['infer_base_s'] = t_base.elapsed

        residual_train = y_train - base_train
        residual_val = y_val - base_val

        es = args.cl_early_stopping_rounds or 0
        es = int(es) if es > 0 else None
        cold_cap = args.cl_max_estimators if es is not None else None

        with StageTimer() as t_cold:
            cold = _fit_warmstart_xgb(
                X_train, residual_train, X_val, residual_val,
                prev_booster=None,
                mode=args.cl_update_mode,
                n_round=args.cl_round_estimators,
                lr=args.cl_lr,
                early_stopping_rounds=es,
                n_estimators_cold_cap=cold_cap,
            )
        stages['train_cold_s'] = t_cold.elapsed
        rmse_cold = _val_rmse(residual_val, cold.predict(X_val))

        residual_model = cold
        choice = 'cold'
        rmse_warm: float | None = None
        if prev_booster is not None and args.use_warm_start:
            with StageTimer() as t_warm:
                warm = _fit_warmstart_xgb(
                    X_train, residual_train, X_val, residual_val,
                    prev_booster=prev_booster,
                    mode=args.cl_update_mode,
                    n_round=args.cl_round_estimators,
                    lr=args.cl_lr,
                    early_stopping_rounds=es if args.cl_update_mode == 'append' else None,
                    n_estimators_cold_cap=None,
                )
            stages['train_warm_s'] = t_warm.elapsed
            rmse_warm = _val_rmse(residual_val, warm.predict(X_val))
            if rmse_warm + 1e-12 < rmse_cold:
                residual_model, choice = warm, 'warm'
            else:
                residual_model, choice = cold, 'cold'

        with StageTimer() as t_sel:
            prev_booster = residual_model.get_booster()
        stages['select_s'] = t_sel.elapsed

        with StageTimer() as t_inf:
            y_pred = base_test + residual_model.predict(X_test)
        stages['infer_test_s'] = t_inf.elapsed
        stages['wall_window_s'] = time.perf_counter() - wall0
        cost.log_window(
            window_i=i,
            n_train=len(train_df),
            n_val=len(val_df),
            n_test=len(test_df),
            stages=stages,
            picked=choice,
            extra={'ft': True, 'n_trees': _booster_n_trees(prev_booster)},
        )
        pieces.append(pd.DataFrame({
            'DMEA': test_df['DMEA'].to_numpy(),
            'ROPA_Pre1': y_pred,
            'ROPA_Real': y_test,
        }))
        used += 1
        if args.verbose:
            extra = f' warm_res_val_rmse={rmse_warm:.4f}' if rmse_warm is not None else ''
            print(f'[FT i={i:03d}] pick={choice}  base_rmse='
                  f'{np.sqrt(mean_squared_error(y_test, base_test)):.3f}  '
                  f'cold_res_val_rmse={rmse_cold:.4f}{extra}  '
                  f'ft_rmse={np.sqrt(mean_squared_error(y_test, y_pred)):.3f}')

    if not pieces:
        raise RuntimeError('no window produced any predictions; check start_depth / data range')

    result = pd.concat(pieces, ignore_index=True).sort_values('DMEA').reset_index(drop=True)
    out_path = args.output or os.path.join(
        ft_csv_dir, f'predictions_results-{short}-XGB-FT.csv')
    _save_predictions(out_path, result['DMEA'].to_numpy(),
                      result['ROPA_Pre1'].to_numpy(),
                      result['ROPA_Real'].to_numpy())
    print(f'FT: {used} windows used')
    _print_metrics('overall', result['ROPA_Real'].to_numpy(), result['ROPA_Pre1'].to_numpy())
    cost.flush_summary()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _short_well(well_tag: str) -> str:
    """SZ36-1-Q15 -> Q15; SZ36-1-Q18H -> Q18, consistent with existing CSV naming convention."""
    name = well_tag.split('-')[-1]
    return name.rstrip('H')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['global', 'cl', 'ft'], default='cl')
    p.add_argument('--well-csv', default='./data/ProcessedData-2/SZ36-1-Q15.csv')
    p.add_argument('--start-depth', type=float, default=1120.0,
                   help='starting depth for the sliding window (keep consistent with the corresponding well in trainCL.py)')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=400)
    p.add_argument('--cl-round-estimators', type=int, default=200,
                   help='number of trees built on cold start; in append mode, '
                        'the number of trees appended per window. In refresh '
                        'mode, subsequent windows do not use this value (tree '
                        'count stays the same as the first window).')
    p.add_argument('--cl-lr', type=float, default=0.05)
    p.add_argument('--cl-max-estimators', type=int, default=200,
                   help='maximum boosting rounds for CL/FT cold-start '
                        'candidates when used with early stopping (only '
                        'effective when --cl-early-stopping-rounds>0).')
    p.add_argument('--cl-early-stopping-rounds', type=int, default=30,
                   help='CL/FT: early_stopping_rounds passed to XGB for cold '
                        'start and append continue-training, selecting the '
                        'best iteration on val; 0 means disabled. The refresh '
                        'continue-training path does not use early stopping.')
    p.add_argument('--cl-update-mode', choices=['refresh', 'append'],
                   default='refresh',
                   help='CL/FT warm-start strategy between windows: '
                        'refresh = recompute leaf weights on top of existing '
                        'tree structures (aligned with NN fine-tuning, keeps '
                        'tree count constant, default); '
                        'append = append new trees after the booster (classic '
                        'boosting warm start).')
    p.add_argument('--use-warm-start', action='store_true', default=True)
    p.add_argument('--no-warm-start', dest='use_warm_start', action='store_false')
    p.add_argument('--save-per-window', action='store_true')
    p.add_argument('--exp-tag', default=None,
                   help='experiment tag (used as subdirectory name under '
                        'model-dir/csv-dir). In global mode defaults to the '
                        'test-well combination (e.g. Q2-Q3); in ft mode it is '
                        'used to locate the corresponding global base model.')
    p.add_argument('--base-model-path', default=None,
                   help='FT mode only; defaults to ./saved_models-XGB/<exp-tag>/global_model.pkl')
    p.add_argument('--model-dir', default='./saved_models-XGB',
                   help='model root directory; global lands at '
                        '{model-dir}/{exp-tag}/global_model.pkl, '
                        'CL per-window models land at {model-dir}/{well}-CL/.')
    p.add_argument('--csv-dir', default='./csv1',
                   help='prediction CSV root directory; subdirectories are '
                        'created per test well (global) or current well (cl/ft).')
    p.add_argument('--output', default=None,
                   help='directly specify the output CSV path for predictions; '
                        'if empty, auto-generate based on well + mode')
    p.add_argument('--verbose', action='store_true')
    add_cost_profile_args(p)
    return p


def main():
    args = build_parser().parse_args()
    if args.mode == 'global':
        train_global(args)
    elif args.mode == 'cl':
        train_cl(args)
    elif args.mode == 'ft':
        train_ft(args)


if __name__ == '__main__':
    main()
