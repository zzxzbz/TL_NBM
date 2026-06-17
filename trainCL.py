# Training script for PTM-FT-BPNN model
import pytorch_lightning as pl
from models.CLROPModel import CLROPModel
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset.dataLoader import ContinuousLearningROPDataset, ContinuousLearningROPDataset_BE, AllForOneROPDataset
from models.simple_direct_difference_model import SimpleDirectDifferenceModel
from models.ft_residual_model import FTResidualModel
import  torch
import datetime
import  numpy as np
import  torch.nn as nn
import  torch.optim as optim
from    matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
import pandas as pd
import os
import sys
import time
from pathlib import Path
import glob
import re
from create_scaler import load_scaler
from cost_profile import CostProfileLogger, StageTimer, add_cost_profile_args, short_well_tag

torch.set_float32_matmul_precision('medium')

_scaler_cache = None
_feature_info_cache = None


def get_scaler():
    """Lazy-load scaler to avoid repeated import flushing from DataLoader subprocesses."""
    global _scaler_cache, _feature_info_cache
    if _scaler_cache is None:
        verbose = os.environ.get('LOAD_SCALER_VERBOSE', '1') == '1'
        _scaler_cache, _feature_info_cache = load_scaler(verbose=verbose)
    return _scaler_cache, _feature_info_cache


def _dataloader_num_workers() -> int:
    """On Windows, multiprocessing re-executes this script; recommend 0."""
    return 0 if sys.platform == 'win32' else 4


def _prepare_logger_dir(save_dir: str, name: str) -> None:
    """If save_dir/name is a file, remove it so Lightning Logger can create a subdirectory."""
    os.makedirs(save_dir, exist_ok=True)
    leaf = os.path.join(save_dir, name)
    if os.path.isfile(leaf):
        os.remove(leaf)

_VAL_LOSS_RE = re.compile(r'val_loss=(\d+(?:\.\d+)?)')

def find_best_ckpt_for_window(ckpt_dir, window_idx, model_name='CLROPModel'):
    """Return the checkpoint path with the smallest val_loss in the specified window.

    Uses ``model_name`` to distinguish checkpoint filename prefixes across different
    training pipelines, e.g., CL training uses ``CLROPModel``, FT residual training
    uses ``FTResidualModel``.
    """
    pattern = os.path.join(ckpt_dir, f'{window_idx:02d}-{model_name}-*.ckpt')
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def _val_loss(path):
        match = _VAL_LOSS_RE.search(os.path.basename(path))
        return float(match.group(1)) if match else float('inf')

    return min(candidates, key=_val_loss)


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

    train_well_ids = [well_id for well_id in well_ids if well_id not in val_well_ids and well_id not in test_well_ids]

    train_well_file_path_list = [f'./data/ProcessedData-2/{well_id}.csv' for well_id in train_well_ids]
    val_well_file_path_list = [f'./data/ProcessedData-2/{well_id}.csv' for well_id in val_well_ids]
    test_well_file_path_list = [f'./data/ProcessedData-2/{well_id}.csv' for well_id in test_well_ids]

    return train_well_file_path_list, val_well_file_path_list, test_well_file_path_list

def train_MLP_model(cost_logger: CostProfileLogger | None = None):
    scaler, _ = get_scaler()
    train_well_file_path_list, val_well_file_path_list, test_well_file_path_list = train_test_val_split()

    train_dataset = AllForOneROPDataset(train_well_file_path_list, scaler)
    val_dataset = AllForOneROPDataset(val_well_file_path_list, scaler)
    test_dataset = AllForOneROPDataset(test_well_file_path_list, scaler)

    nw = _dataloader_num_workers()
    loader_kw = dict(batch_size=32, num_workers=nw, pin_memory=torch.cuda.is_available(), drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kw)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_kw)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_kw)


    hparams = {
        'input_dim': len(train_dataset.input_col),
        'hidden_dim': 32,
        'learning_rate': 5e-4, 
        'dropout': 0.2,
        'weight_decay': 1e-4,
    }

    model = CLROPModel(hparams)

    # checkpoint_callback saves models, logger records training process
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints-MLP-C7',
        filename='CLROPModel-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    _prepare_logger_dir('./log/logsC7', 'CLROPModel')
    logger = CSVLogger('./log/logsC7', name='CLROPModel')

    trainer = pl.Trainer(
        accelerator="cuda",
        #devices=[0],
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=30,
    )

    stages: dict[str, float] = {}
    with StageTimer() as t_fit:
        trainer.fit(model, train_loader, val_loader)
    stages['train_fit_s'] = t_fit.elapsed
    with StageTimer() as t_test:
        test_result = trainer.test(model, test_loader)
    stages['train_test_s'] = t_test.elapsed
    if cost_logger is not None and cost_logger.enabled:
        cost_logger.log_global(
            n_train=len(train_dataset),
            n_val=len(val_dataset),
            n_test=len(test_dataset),
            stages=stages,
            extra={'max_epochs': 50, 'hidden_dim': 32},
        )
        cost_logger.flush_summary()

def train_test_val_split_slide(i, data=None):

    # This function splits the dataset via sliding window. The training set starts at a
    # depth that advances by 10m per training iteration. Train=20m, val=10m, test=10m.
    # data can be preloaded externally to avoid re-reading CSV for each i.
    # Assumes the column at index 1 is the depth column.
    depth_column = data.columns[1]

    # Define initial depth range
    start_depth = 455
    train_end_1 = 475
    val_end_1 = 485
    test_end_1 = 495

    # Generate three-way train/val/test split
    train_data = data[(data[depth_column] >= start_depth + i * 10) & (data[depth_column] < train_end_1 + i *10)]
    val_data = data[(data[depth_column] >= train_end_1 + i * 10) & (data[depth_column] < val_end_1 + i * 10)]
    test_data = data[(data[depth_column] >= val_end_1 + i * 10) & (data[depth_column] < test_end_1 + i *10)]

    # Some windows may be empty due to missing raw data; let caller skip these
    if train_data.empty or val_data.empty or test_data.empty:
        raise ValueError(
            f"window i={i} has empty split: "
            f"train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
        )

    return train_data, val_data, test_data

def train_CL_model(i, well_data=None, use_warm_start=True, cost_logger: CostProfileLogger | None = None):
    scaler, _ = get_scaler()
    wall0 = time.perf_counter()
    stages: dict[str, float] = {}
    with StageTimer() as t_split:
        train_data, val_data, test_data = train_test_val_split_slide(i, data=well_data)
    stages['split_s'] = t_split.elapsed

    train_dataset = ContinuousLearningROPDataset(train_data, scaler)
    val_dataset = ContinuousLearningROPDataset(val_data, scaler)
    test_dataset = ContinuousLearningROPDataset(test_data, scaler)

    # Aligned with train_NBM_CL_model: same batch_size, num_workers, drop_last, pin_memory
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    # Aligned with NBM optimizer config: lr=5e-2, weight_decay=1e-4
    params = {
        'input_dim': len(train_dataset.input_col),
        'hidden_dim': 8,
        'output_dim': 1,
        'dropout': 0.2,
        'learning_rate': 5e-2,
        'weight_decay': 1e-4,
        'num_time_steps' : 16
    }

    ckpt_dir = './checkpoints-MLP-Q2-Q3-cost/Q2-CL'
    prev_ckpt = None
    if use_warm_start and i > 0:
        prev_ckpt = find_best_ckpt_for_window(ckpt_dir, i - 1)

    picked = 'cold'
    with StageTimer() as t_init:
        if prev_ckpt is not None:
            model = CLROPModel.load_from_checkpoint(prev_ckpt, params=params)
            picked = 'warm'
            print(f"[WarmStart] i={i}, loaded: {prev_ckpt}")
        else:
            model = CLROPModel(params)
            if use_warm_start and i > 0:
                print(f"[WarmStart] i={i}, previous checkpoint not found, fallback to random init.")
    stages['init_load_s'] = t_init.elapsed

    modelName =  str(i).zfill(2) 
 
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f'{modelName}'+'-CLROPModel-{epoch:02d}-{val_loss:.2f}-1e-3',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    logger = CSVLogger('./log/logsQ2-Q3-cost/Q2-Q3-CL', name='Q2-Q3-CL')

    trainer = pl.Trainer(
        accelerator="gpu",
        #devices=[0],
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=100,
    )

    with StageTimer() as t_fit:
        trainer.fit(model, train_loader, val_loader)
    stages['train_fit_s'] = t_fit.elapsed
    with StageTimer() as t_test:
        trainer.test(model, test_loader)
    stages['train_test_s'] = t_test.elapsed
    stages['wall_window_s'] = time.perf_counter() - wall0
    if cost_logger is not None and cost_logger.enabled:
        cost_logger.log_window(
            window_i=i,
            n_train=len(train_data),
            n_val=len(val_data),
            n_test=len(test_data),
            stages=stages,
            picked=picked,
            extra={'max_epochs': 100, 'hidden_dim': 8},
        )

def train_FT_E_model(i, well_data=None, use_warm_start=True, cost_logger: CostProfileLogger | None = None):
    scaler, _ = get_scaler()
    wall0 = time.perf_counter()
    stages: dict[str, float] = {}
    with StageTimer() as t_split:
        train_data, val_data, test_data = train_test_val_split_slide(i, data=well_data)
    stages['split_s'] = t_split.elapsed

    train_dataset = ContinuousLearningROPDataset_BE(train_data, scaler)
    val_dataset = ContinuousLearningROPDataset_BE(val_data, scaler)
    test_dataset = ContinuousLearningROPDataset_BE(test_data, scaler)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    params = {
        'input_dim': len(train_dataset.input_col),
        'hidden_dim': 8,
        'output_dim': 1,
        'dropout': 0.2,
        'learning_rate': 5e-2,
        'weight_decay': 1e-4,
        'num_time_steps' : 16
    }

    ckpt_dir = './checkpoints-MLP-A1/A1-FT-E'
    prev_ckpt = None
    if use_warm_start and i > 0:
        prev_ckpt = find_best_ckpt_for_window(ckpt_dir, i - 1)

    picked = 'cold'
    with StageTimer() as t_init:
        if prev_ckpt is not None:
            model = CLROPModel.load_from_checkpoint(prev_ckpt, params=params)
            picked = 'warm'
            print(f"[WarmStart] i={i}, loaded: {prev_ckpt}")
        else:
            model = CLROPModel(params)
            if use_warm_start and i > 0:
                print(f"[WarmStart] i={i}, previous checkpoint not found, fallback to random init.")
    stages['init_load_s'] = t_init.elapsed

    modelName =  str(i).zfill(2) 
 
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f'{modelName}'+'-CLROPModel-{epoch:02d}-{val_loss:.2f}-1e-3',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    logger = CSVLogger('./log/logsA1FTModel-1e-2-2-Slide10', name='A2-stable')

    trainer = pl.Trainer(
        accelerator="cpu",
        #devices=[0],
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=100,
    )

    with StageTimer() as t_fit:
        trainer.fit(model, train_loader, val_loader)
    stages['train_fit_s'] = t_fit.elapsed
    with StageTimer() as t_test:
        trainer.test(model, test_loader)
    stages['train_test_s'] = t_test.elapsed
    stages['wall_window_s'] = time.perf_counter() - wall0
    if cost_logger is not None and cost_logger.enabled:
        cost_logger.log_window(
            window_i=i,
            n_train=len(train_data),
            n_val=len(val_data),
            n_test=len(test_data),
            stages=stages,
            picked=picked,
            extra={'mode': 'FT-E', 'max_epochs': 100},
        )

def train_FT_model(
    i,
    well_data=None,
    base_ckpt_path='./checkpoints-MLP-Q7-Q21/CLROPModel-epoch=29-val_loss=0.00.ckpt',
    ckpt_dir='./checkpoints-FT-Q10/Slide10-FT',
    log_dir='./log/logsFT-Q10',
    use_warm_start=True,
    cost_logger: CostProfileLogger | None = None,
    ):
    """Residual fine-tuning training.

    - The base model (cross-well model) is specified by ``base_ckpt_path``; parameters
      are fully frozen after loading and only provide base predictions.
    - The fine-tuning model (``FTResidualModel.residual_net``) fits MSE with
      ``y - f_base(x)`` as the label.
    - Sliding-window data split is identical to ``train_CL_model``; when
      ``use_warm_start=True`` and ``i>0``, the best FT checkpoint from window ``i-1``
      in the same ``ckpt_dir`` is loaded for warm start (continuous learning).
    """
    scaler, _ = get_scaler()
    wall0 = time.perf_counter()
    stages: dict[str, float] = {}
    with StageTimer() as t_split:
        train_data, val_data, test_data = train_test_val_split_slide(i, data=well_data)
    stages['split_s'] = t_split.elapsed

    train_dataset = ContinuousLearningROPDataset(train_data, scaler)
    val_dataset = ContinuousLearningROPDataset(val_data, scaler)
    test_dataset = ContinuousLearningROPDataset(test_data, scaler)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    params = {
        'input_dim': len(train_dataset.input_col),
        'hidden_dim': 8,
        'output_dim': 1,
        'dropout': 0.2,
        'learning_rate': 5e-2,
        'weight_decay': 1e-4,
        'num_time_steps': 16,
    }

    picked = 'cold'
    with StageTimer() as t_base:
        pretrained_base = CLROPModel.load_from_checkpoint(base_ckpt_path)
    stages['infer_base_s'] = t_base.elapsed

    prev_ckpt = None
    if use_warm_start and i > 0:
        prev_ckpt = find_best_ckpt_for_window(ckpt_dir, i - 1, model_name='FTResidualModel')

    with StageTimer() as t_init:
        if prev_ckpt is not None:
            model = FTResidualModel.load_from_checkpoint(
                prev_ckpt,
                backbone_model=pretrained_base,
                params=params,
            )
            picked = 'warm'
            print(f"[FT-WarmStart] i={i}, loaded: {prev_ckpt}")
        else:
            model = FTResidualModel(backbone_model=pretrained_base, params=params)
            if use_warm_start and i > 0:
                print(f"[FT-WarmStart] i={i}, previous checkpoint not found, fallback to fresh residual net.")
    stages['init_load_s'] = t_init.elapsed

    os.makedirs(ckpt_dir, exist_ok=True)
    modelName = str(i).zfill(2)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f'{modelName}' + '-FTResidualModel-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
    )
    logger = CSVLogger(log_dir, name='FT-Residual')

    trainer = pl.Trainer(
        accelerator="gpu",
        #devices=[0],
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=100,
    )

    with StageTimer() as t_fit:
        trainer.fit(model, train_loader, val_loader)
    stages['train_fit_s'] = t_fit.elapsed
    with StageTimer() as t_test:
        trainer.test(model, test_loader)
    stages['train_test_s'] = t_test.elapsed
    stages['wall_window_s'] = time.perf_counter() - wall0
    if cost_logger is not None and cost_logger.enabled:
        cost_logger.log_window(
            window_i=i,
            n_train=len(train_data),
            n_val=len(val_data),
            n_test=len(test_data),
            stages=stages,
            picked=picked,
            extra={'mode': 'FT-residual', 'max_epochs': 100},
        )


def build_parser():
    import argparse
    p = argparse.ArgumentParser(description='BPNN ROP training (supports --profile-cost)')
    p.add_argument('--mode', choices=['global', 'cl', 'ft', 'ft-e'], default='global')
    p.add_argument('--well-csv', default='./data/ProcessedData-2/SZ36-1-Q2.csv')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=400)
    p.add_argument('--base-ckpt', default='./checkpoints-MLP-Q2-Q3-cost\CLROPModel-epoch=08-val_loss=975.23.ckpt')
    p.add_argument('--ckpt-dir', default='./checkpoints-MLP-Q2-Q3-cost/Q2-FT')
    p.add_argument('--log-dir', default='./log/logsQ2-FT')
    add_cost_profile_args(p)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    get_scaler()
    well_path = args.well_csv
    short = short_well_tag(well_path)
    # python trainCL.py --mode global
    if args.mode == 'global':
        cost = CostProfileLogger.from_args(args, model='BPNN', well='global', mode='global')
        train_MLP_model(cost_logger=cost)
    else:
        pattern_mode = {'cl': 'cl', 'ft': 'ft', 'ft-e': 'ft'}[args.mode]
        cost = CostProfileLogger.from_args(args, model='BPNN', well=short, mode=pattern_mode)
        well_data = pd.read_csv(well_path)
        for i in range(args.start, args.end):
            try:
                if args.mode == 'cl':
                    train_CL_model(i, well_data=well_data, use_warm_start=True, cost_logger=cost)
                elif args.mode == 'ft-e':
                    train_FT_E_model(i, well_data=well_data, use_warm_start=True, cost_logger=cost)
                else:
                    train_FT_model(
                        i,
                        well_data=well_data,
                        base_ckpt_path=args.base_ckpt,
                        ckpt_dir=args.ckpt_dir,
                        log_dir=args.log_dir,
                        use_warm_start=True,
                        cost_logger=cost,
                    )
            except ValueError as e:
                cost.skip_window()
                print(f"[Skip] i={i}: {e}")
                continue
            except Exception as e:
                cost.skip_window()
                print(f"[Skip] i={i} failed: {e}")
                continue
        cost.flush_summary()
