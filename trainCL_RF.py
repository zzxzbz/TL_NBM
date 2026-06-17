"""RandomForest-based ROP training script.

Supports three training modes (aligned with the three MLP modes in trainCL.py):

1. ``global`` —— Cross-well training.
   Multiple training wells are merged to train a global RandomForest model,
   which produces predictions on test wells in one shot.

2. ``cl`` —— Intra-well sliding window + continual learning.
   Sliding window splits within a single well are created via
   trainCL.train_test_val_split_slide.
   **Each window**: a candidate is fitted on ``train_df``; the RMSE on ``val_df``
   is used to **select the best** between "cold start vs. warm start from
   previous window"; **only the best model** is applied to ``test_df``.
   The tree strategy is controlled by ``--cl-update-mode`` (aligned with
   ``trainCL_XGB.py``):

     * ``refresh`` —— **Fixed tree structure, leaf values only**: the first window
       trains an RF normally; each subsequent window preserves every tree's
       splits (features and thresholds unchanged), rewrites each leaf's prediction
       with the mean ``y`` of the current window's training samples that fall into
       that leaf; test samples landing in a leaf unseen during the training window
       fall back to the original sklearn leaf value. Forest size stays constant.
       **Default**.
     * ``append`` —— ``warm_start=True`` appends ``cl_round_estimators`` new trees
       to the end of the forest (classic sklearn continual learning).

3. ``ft`` —— Residual fine-tuning based on a cross-well model.
   First loads a cross-well RF trained in ``global`` mode as the base model
   f_base; within each sliding window, a residual RF is fitted on ``train_df``.
   **The full ROP RMSE on ``val_df`` is used to select** the best between
   "cold start vs. warm start" residual model, then the test set prediction is:
   ``y_hat = f_base(x) + f_res(x)``.

Output columns align with the companion eval scripts of trainCL.py:
``DMEA, ROPA_Pre1, ROPA_Real``.

Usage examples:

python trainCL_RF.py --mode global
python trainCL_RF.py --mode cl --well-csv ./data/ProcessedData-2/USROP-A1.csv --start-depth 300
python trainCL_RF.py --mode ft --well-csv ./data/ProcessedData-2/USROP-A1.csv --start-depth 300 --exp-tag A1

"""

from __future__ import annotations

import argparse
import copy
import os
import time
from typing import List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from create_scaler import load_scaler
from cost_profile import CostProfileLogger, StageTimer, add_cost_profile_args

INPUT_COLS = ['DMEA', 'WOBA', 'RPMA', 'MFIA', 'MDIA',
              'SPPA', 'BIT_DIAMETER', 'ANGLE', 'TQA_ON_BIT']
TARGET_COL = 'ROPA'

SCALER, _FEATURE_INFO = load_scaler()


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[INPUT_COLS]
    X_scaled = SCALER.transform(X)
    return pd.DataFrame(X_scaled, columns=INPUT_COLS, index=df.index)


def _as_tree_input_array(X) -> np.ndarray:
    """Convert feature matrix to a column-name-free float64 ndarray.

    When ``RefreshedRandomForest`` calls ``apply`` on a single
    ``DecisionTreeRegressor``, passing a DataFrame with ``feature_names``
    while the subtree was fitted internally with an ndarray causes sklearn to
    raise ``X has feature names, but DecisionTreeRegressor was fitted without``.
    """
    if isinstance(X, pd.DataFrame):
        return np.ascontiguousarray(X.to_numpy(dtype=np.float64, copy=False))
    return np.ascontiguousarray(np.asarray(X, dtype=np.float64))


# ---------------------------------------------------------------------------
# Data splitting (consistent with trainCL.py)
# ---------------------------------------------------------------------------

def train_test_val_split():
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
    """"./data/.../SZ36-1-Q2.csv -> SZ36-1-Q2" """
    return os.path.splitext(os.path.basename(path))[0]


def _test_set_tag(test_paths) -> str:
    """Generate an experiment tag from the current test well list,
    e.g. ['SZ36-1-Q7','SZ36-1-Q21H'] -> 'Q7-Q21'.

    Used as the subdirectory name under ``model_dir`` / ``csv_dir``:
    when switching test wells, models/artifacts automatically land in
    separate directories without overwriting each other.
    """
    return '-'.join(_short_well(_well_tag(p)) for p in test_paths)


def _load_concat(file_paths) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in file_paths], ignore_index=True)


def train_test_val_split_slide(i: int, data: pd.DataFrame, start_depth: float,
                               train_len: float = 20.0,
                               val_len: float = 10.0,
                               test_len: float = 10.0,
                               step: float = 10.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def _val_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """In sliding windows, use validation-set RMSE for model selection."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------------------------
# Three training modes
# ---------------------------------------------------------------------------

def _make_rf(n_estimators: int = 200, **override) -> RandomForestRegressor:
    base = dict(
        n_estimators=n_estimators,
        max_depth=40,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        warm_start=False,
    )
    base.update(override)
    return RandomForestRegressor(**base)


class RefreshedRandomForest:
    """Freeze each tree's splits in a RandomForest, only recompute leaf
    prediction values based on the current window's data.

    sklearn has no XGBoost ``updater=refresh`` equivalent; here we use
    ``tree.apply`` to obtain leaf ids, average ``y`` for each leaf from the
    training window as the new leaf value; at prediction time, unseen leaves
    fall back to the original ``tree_.value`` leaf values. Semantically
    aligned with ``trainCL_XGB``'s ``refresh`` mode.
    """

    def __init__(self, estimators: list, leaf_maps: List[dict]):
        self.estimators_ = estimators
        self._leaf_maps = leaf_maps
        self.n_estimators = len(estimators)

    def predict(self, X) -> np.ndarray:
        X_arr = _as_tree_input_array(X)
        n_samples = X_arr.shape[0]
        preds = np.zeros((n_samples, len(self.estimators_)), dtype=np.float64)
        for j, (tree, lmap) in enumerate(zip(self.estimators_, self._leaf_maps)):
            leaves = np.asarray(tree.apply(X_arr), dtype=np.intp)
            default_vals = tree.tree_.value[:, 0, 0]
            col = default_vals[leaves].astype(np.float64, copy=False)
            for lid, val in lmap.items():
                col[leaves == lid] = val
            preds[:, j] = col
        return preds.mean(axis=1)


def _rf_leaf_means(tree, X_train, y_train) -> dict:
    """Per-leaf mapping: leaf node id -> mean of y for that leaf
    under the current window.
    """
    leaves = tree.apply(_as_tree_input_array(X_train))
    y_arr = np.asarray(y_train, dtype=np.float64)
    out: dict = {}
    for lid in np.unique(leaves):
        lid = int(lid)
        mask = leaves == lid
        out[lid] = float(np.mean(y_arr[mask]))
    return out


def _rf_refresh_fit(prev: Union[RandomForestRegressor, RefreshedRandomForest],
                    X_train, y_train) -> RefreshedRandomForest:
    """Deep copy base tree structures, recompute leaf means with (X_train, y_train)."""
    trees = [copy.deepcopy(t) for t in prev.estimators_]
    leaf_maps = [_rf_leaf_means(t, X_train, y_train) for t in trees]
    return RefreshedRandomForest(trees, leaf_maps)


def _rf_n_trees(model) -> int:
    return len(model.estimators_)


def train_global(args) -> str:
    """Cross-well training: merge multiple wells for training + output
    predictions and metrics on each test well separately.

    Artifacts are grouped by test well set (``{model_dir}/{tag}/``,
    ``{csv_dir}/{tag}/``). ``tag`` is explicitly specified via ``--exp-tag``,
    otherwise auto-derived from ``train_test_val_split`` test wells
    (e.g. ``Q7-Q21``).
    """
    train_paths, val_paths, test_paths = train_test_val_split()
    tag = args.exp_tag or _test_set_tag(test_paths)
    print(f'[exp-tag] {tag}  (test wells: {[_well_tag(p) for p in test_paths]})')

    train_df = _load_concat(train_paths)
    val_df = _load_concat(val_paths)

    X_train, y_train = scale_features(train_df), train_df[TARGET_COL].to_numpy()
    X_val, y_val = scale_features(val_df), val_df[TARGET_COL].to_numpy()

    cost = CostProfileLogger.from_args(args, model='RF', well=tag, mode='global')
    model = _make_rf(n_estimators=args.global_estimators)
    with StageTimer() as t_fit:
        model.fit(X_train, y_train)
    t_train = t_fit.elapsed

    _print_metrics('train', y_train, model.predict(X_train))
    _print_metrics('val', y_val, model.predict(X_val))

    model_dir = os.path.join(args.model_dir, tag)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'global_model.pkl')
    joblib.dump(model, model_path)
    print(f'global model saved -> {model_path}')

    out_dir = os.path.join(args.csv_dir, tag)

    overall_true, overall_pred = [], []
    for tp in test_paths:
        well_tag = _well_tag(tp)
        short = _short_well(well_tag)

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
                extra={'well_infer': short, 'n_estimators': args.global_estimators},
            )

        _print_metrics(f'test[{short}]', y_w, pred_w)

        if args.output:
            if '{well}' in args.output:
                out_path = args.output.format(well=short)
            elif len(test_paths) == 1:
                out_path = args.output
            else:
                base, ext = os.path.splitext(args.output)
                out_path = f'{base}-{short}{ext}'
                print(f'[warn] --output missing {{well}} placeholder, auto-renamed to {out_path}')
        else:
            out_path = os.path.join(
                out_dir, f'predictions_results-{short}-RF-Global.csv')

        _save_predictions(out_path, df_w['DMEA'].to_numpy(), pred_w, y_w)

        overall_true.append(y_w)
        overall_pred.append(pred_w)

    if overall_true:
        _print_metrics('test[overall]',
                       np.concatenate(overall_true),
                       np.concatenate(overall_pred))
    cost.flush_summary()
    return model_path


def _warm_start_fit(prev_model: RandomForestRegressor, X_new, y_new,
                    add_estimators: int) -> RandomForestRegressor:
    """Append a batch of new trees to a trained RF model (continual learning).

    sklearn warm_start semantics: keep existing estimators, increment
    ``n_estimators`` to the new value, and train only the newly added
    estimators on the new data. We deep-copy the model object here to avoid
    sharing the estimator list across consecutive windows, which could cause
    unintended side effects.
    """
    model = copy.deepcopy(prev_model)
    model.warm_start = True
    model.n_estimators = len(model.estimators_) + add_estimators
    model.fit(X_new, y_new)
    return model


def train_cl(args):
    """Intra-well sliding window + continual learning (append extra trees
    or refresh fixed structure leaf values).

    Artifacts are grouped by current well name:
      - Per-window models: ``{model_dir}/{short_well}-CL/{well_tag}-CL-{i:03d}.pkl``
      - Concatenated prediction CSV: ``{csv_dir}/{short_well}/predictions_results-{short_well}-RF-CL.csv``
    """
    well_data = pd.read_csv(args.well_csv)
    well_tag = os.path.splitext(os.path.basename(args.well_csv))[0]
    short = _short_well(well_tag)

    cl_model_dir = os.path.join(args.model_dir, f'{short}-CL')
    cl_csv_dir = os.path.join(args.csv_dir, short)
    cost = CostProfileLogger.from_args(args, model='RF', well=short, mode='cl')

    pieces = []
    prev_model: Union[RandomForestRegressor, RefreshedRandomForest, None] = None
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
            y_train = train_df[TARGET_COL].to_numpy()
            X_val = scale_features(val_df)
            y_val = val_df[TARGET_COL].to_numpy()
            X_test = scale_features(test_df)
            y_test = test_df[TARGET_COL].to_numpy()
        stages['scale_s'] = t_scale.elapsed

        candidates: list[tuple[str, Union[RandomForestRegressor, RefreshedRandomForest]]] = []
        m_cold = _make_rf(n_estimators=args.cl_round_estimators)
        with StageTimer() as t_cold:
            m_cold.fit(X_train, y_train)
        stages['train_cold_s'] = t_cold.elapsed
        candidates.append(('cold', m_cold))

        if prev_model is not None and args.use_warm_start:
            with StageTimer() as t_warm:
                if args.cl_update_mode == 'append':
                    if isinstance(prev_model, RandomForestRegressor):
                        m_warm = _warm_start_fit(
                            prev_model, X_train, y_train,
                            add_estimators=args.cl_round_estimators)
                        candidates.append(('warm', m_warm))
                else:
                    candidates.append(
                        ('warm', _rf_refresh_fit(prev_model, X_train, y_train)))
            stages['train_warm_s'] = t_warm.elapsed

        with StageTimer() as t_sel:
            best_name, model = min(
                candidates,
                key=lambda t: _val_rmse(y_val, t[1].predict(X_val)),
            )
        stages['select_s'] = t_sel.elapsed
        prev_model = model

        val_rmse_best = _val_rmse(y_val, model.predict(X_val))
        with StageTimer() as t_inf:
            y_pred = model.predict(X_test)
        stages['infer_test_s'] = t_inf.elapsed
        stages['wall_window_s'] = time.perf_counter() - wall0
        test_rmse = _val_rmse(y_test, y_pred)
        cost.log_window(
            window_i=i,
            n_train=len(train_df),
            n_val=len(val_df),
            n_test=len(test_df),
            stages=stages,
            picked=best_name,
            extra={'cl_update_mode': args.cl_update_mode, 'n_trees': _rf_n_trees(model)},
        )
        pieces.append(pd.DataFrame({
            'DMEA': test_df['DMEA'].to_numpy(),
            'ROPA_Pre1': y_pred,
            'ROPA_Real': y_test,
        }))
        used += 1
        if args.verbose:
            parts = [
                f'{name}={_val_rmse(y_val, m.predict(X_val)):.4f}'
                for name, m in candidates
            ]
            joined = ' '.join(parts)
            print(f'[i={i:03d}] pick={best_name}  mode={args.cl_update_mode}  '
                  f'trees={_rf_n_trees(model)}  train={len(train_df)} val={len(val_df)} '
                  f'test={len(test_df)}  val_rmses=[{joined}]  '
                  f'best_val_rmse={val_rmse_best:.4f}  test_rmse={test_rmse:.4f}')

        if args.save_per_window:
            os.makedirs(cl_model_dir, exist_ok=True)
            joblib.dump(model, os.path.join(
                cl_model_dir, f'{well_tag}-CL-{i:03d}.pkl'))

    if not pieces:
        raise RuntimeError('No windows produced predictions; check start_depth / data range')

    result = pd.concat(pieces, ignore_index=True).sort_values('DMEA').reset_index(drop=True)
    out_path = args.output or os.path.join(
        cl_csv_dir, f'predictions_results-{short}-RF-CL.csv')
    _save_predictions(out_path, result['DMEA'].to_numpy(),
                      result['ROPA_Pre1'].to_numpy(),
                      result['ROPA_Real'].to_numpy())
    print(f'CL: {used} windows used')
    _print_metrics('overall', result['ROPA_Real'].to_numpy(), result['ROPA_Pre1'].to_numpy())
    cost.flush_summary()


def train_ft(args):
    """Residual fine-tuning based on a cross-well pretrained RF.

    Outputs are grouped by current well name:
    ``{csv_dir}/{short_well}/predictions_results-{short_well}-RF-FT.csv``.
    The global base model path defaults to
    ``{model_dir}/{exp_tag}/global_model.pkl``;
    use ``--exp-tag`` to switch to a global model trained on a different
    test well set, or ``--base-model-path`` to specify explicitly.
    """
    base_model_path = args.base_model_path
    if base_model_path is None:
        if args.exp_tag is None:
            raise SystemExit(
                'FT mode requires either --base-model-path or --exp-tag to locate the global base model.')
        base_model_path = os.path.join(args.model_dir, args.exp_tag, 'global_model.pkl')

    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f'Global base model not found: {base_model_path}\n'
            f'Please run first: python trainCL_RF.py --mode global')

    base_model: RandomForestRegressor = joblib.load(base_model_path)
    print(f'[FT] base model loaded -> {base_model_path}')

    well_data = pd.read_csv(args.well_csv)
    well_tag = os.path.splitext(os.path.basename(args.well_csv))[0]
    short = _short_well(well_tag)

    ft_csv_dir = os.path.join(args.csv_dir, short)
    cost = CostProfileLogger.from_args(args, model='RF', well=short, mode='ft')

    pieces = []
    prev_model: Union[RandomForestRegressor, RefreshedRandomForest, None] = None
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

        candidates: list[tuple[str, Union[RandomForestRegressor, RefreshedRandomForest]]] = []
        res_cold = _make_rf(n_estimators=args.cl_round_estimators)
        with StageTimer() as t_cold:
            res_cold.fit(X_train, residual_train)
        stages['train_cold_s'] = t_cold.elapsed
        candidates.append(('cold', res_cold))

        if prev_model is not None and args.use_warm_start:
            with StageTimer() as t_warm:
                if args.cl_update_mode == 'append' and isinstance(prev_model, RandomForestRegressor):
                    candidates.append(
                        ('warm', _warm_start_fit(
                            prev_model, X_train, residual_train,
                            add_estimators=args.cl_round_estimators)))
                elif args.cl_update_mode == 'refresh':
                    candidates.append(
                        ('warm', _rf_refresh_fit(prev_model, X_train, residual_train)))
            stages['train_warm_s'] = t_warm.elapsed

        def _ft_val_rmse(res_m):
            return _val_rmse(y_val, base_val + res_m.predict(X_val))

        with StageTimer() as t_sel:
            best_name, res_model = min(candidates, key=lambda t: _ft_val_rmse(t[1]))
        stages['select_s'] = t_sel.elapsed
        prev_model = res_model

        val_rmse_best = _ft_val_rmse(res_model)
        with StageTimer() as t_inf:
            y_pred = base_test + res_model.predict(X_test)
        stages['infer_test_s'] = t_inf.elapsed
        stages['wall_window_s'] = time.perf_counter() - wall0
        cost.log_window(
            window_i=i,
            n_train=len(train_df),
            n_val=len(val_df),
            n_test=len(test_df),
            stages=stages,
            picked=best_name,
            extra={'ft': True, 'n_trees': _rf_n_trees(res_model)},
        )
        pieces.append(pd.DataFrame({
            'DMEA': test_df['DMEA'].to_numpy(),
            'ROPA_Pre1': y_pred,
            'ROPA_Real': y_test,
        }))
        used += 1
        if args.verbose:
            parts = [f'{name}={_ft_val_rmse(m):.4f}' for name, m in candidates]
            joined = ' '.join(parts)
            print(f'[FT i={i:03d}] pick={best_name}  base_rmse='
                  f'{np.sqrt(mean_squared_error(y_test, base_test)):.3f}  '
                  f'val_rmses=[{joined}]  best_val_rmse={val_rmse_best:.4f}  '
                  f'ft_rmse={np.sqrt(mean_squared_error(y_test, y_pred)):.3f}')

    if not pieces:
        raise RuntimeError('No windows produced predictions; check start_depth / data range')

    result = pd.concat(pieces, ignore_index=True).sort_values('DMEA').reset_index(drop=True)
    out_path = args.output or os.path.join(
        ft_csv_dir, f'predictions_results-{short}-RF-FT.csv')
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
    name = well_tag.split('-')[-1]
    return name.rstrip('H')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['global', 'cl', 'ft'], default='cl')
    p.add_argument('--well-csv', default='./data/ProcessedData-2/SZ36-1-Q15.csv')
    p.add_argument('--start-depth', type=float, default=1120.0)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=400)
    p.add_argument('--global-estimators', type=int, default=200,
                   help='Total number of trees for cross-well RandomForest training')
    p.add_argument('--cl-round-estimators', type=int, default=200,
                   help='Number of trees for the first-window cold start; '
                        'under append mode, number of trees appended per window. '
                        'Under refresh mode, subsequent windows do not increase tree count.')
    p.add_argument('--cl-update-mode', choices=['refresh', 'append'],
                   default='refresh',
                   help='CL/FT inter-window strategy: refresh=freeze tree splits, '
                        'recompute leaf means from the current window '
                        '(aligned with trainCL_XGB refresh semantics; default); '
                        'append=warm_start appends new trees.')
    p.add_argument('--use-warm-start', action='store_true', default=True)
    p.add_argument('--no-warm-start', dest='use_warm_start', action='store_false')
    p.add_argument('--save-per-window', action='store_true')
    p.add_argument('--exp-tag', default=None,
                   help='Experiment tag (used as subdirectory name under model-dir/csv-dir). '
                        'In global mode defaults to the test well combination (e.g. Q7-Q21); '
                        'in ft mode used to locate the corresponding global base model.')
    p.add_argument('--base-model-path', default=None,
                   help='FT mode only; defaults to ./saved_models-RF/<exp-tag>/global_model.pkl')
    p.add_argument('--model-dir', default='./saved_models-RF',
                   help='Model root directory; global lands at '
                        '{model-dir}/{exp-tag}/global_model.pkl, '
                        'CL per-window models land at {model-dir}/{well}-CL/.')
    p.add_argument('--csv-dir', default='./csv1',
                   help='Prediction CSV root directory; subdirectories by test well '
                        '(global) or current well (cl/ft).')
    p.add_argument('--output', default=None)
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
