"""Training script for NBM (Neural Basis Model).

Uses SparseNBMROPModel for ROP prediction with better interpretability.
"""
import pytorch_lightning as pl
from models.sparsenbm import SparseNBMROPModel
from models.nbm_ft_residual_model import NBMFTResidualModel
from dataset.dataLoader import AllForOneROPDataset,ContinuousLearningROPDataset,ContinuousLearningROPDataset_NE
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch 
import pandas as pd
import os
import glob
import re
import time
from create_scaler import load_scaler
from cost_profile import CostProfileLogger, StageTimer, add_cost_profile_args, short_well_tag

scaler, feature_info = load_scaler()
_VAL_LOSS_RE = re.compile(r'val_loss=(\d+(?:\.\d+)?)')

def find_best_ckpt_for_window(ckpt_dir, window_idx, model_name='CLROPModel'):
    """Return the checkpoint path with the smallest val_loss in the specified window."""
    pattern = os.path.join(ckpt_dir, f'{window_idx:02d}-{model_name}-*.ckpt')
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def _val_loss(path):
        match = _VAL_LOSS_RE.search(os.path.basename(path))
        return float(match.group(1)) if match else float('inf')

    return min(candidates, key=_val_loss)

torch.set_float32_matmul_precision('medium')

def train_test_val_split():
    """Split training, validation, and test wells"""

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

def train_NBM_model(cost_logger: CostProfileLogger | None = None):
    """Train NBM model"""
    print("=" * 80)
    print("Starting Neural Basis Model (NBM) training")
    print("=" * 80)
    
    train_well_file_path_list, val_well_file_path_list, test_well_file_path_list = train_test_val_split()

    # Create datasets
    train_dataset = AllForOneROPDataset(train_well_file_path_list, scaler)
    val_dataset = AllForOneROPDataset(val_well_file_path_list, scaler)
    test_dataset = AllForOneROPDataset(test_well_file_path_list, scaler)

    # Create data loaders
    # drop_last=True avoids BatchNorm errors when the last batch has only 1 sample
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=True)

    # NBM model hyperparameter configuration
    hparams = {
        'nbm_params': {
            'num_concepts': len(train_dataset.input_col),  # Number of input features (concept count)
            'num_classes': 1,  # ROP prediction is a regression task, output=1
            'nary': [1,2],  # None means use 1st and 2nd order interactions; can also use [1] or [1, 2] or [2, 3] etc.
            'num_bases': 32,  # Number of basis functions
            'hidden_dims': (32, 32),  # MLP hidden layer dimensions
            'num_subnets': 1,  # Number of neural networks for learning basis functions
            'dropout': 0.2,  # Dropout within MLP
            'bases_dropout': 0.2,  # Dropout at basis level
            'batchnorm': True,  # Whether to use batch normalization
            'output_penalty': 0.4,  # Output penalty coefficient
        },
        'is_sparse': True,
        'gumbel_layer_params': {
            'd': len(train_dataset.input_col),
            'k': 9,
            'tau': 0.1,
        },
        'optimizer': {
            'learning_rate': 5e-4,
            'weight_decay': 1e-4,
        }
    }

    print(f"\nModel configuration:")
    print(f"  - Input feature count: {hparams['nbm_params']['num_concepts']}")
    print(f"  - Number of basis functions: {hparams['nbm_params']['num_bases']}")
    print(f"  - Hidden layer dimensions: {hparams['nbm_params']['hidden_dims']}")
    print(f"  - N-ary interactions: {hparams['nbm_params']['nary']}")
    print(f"  - Learning rate: {hparams['optimizer']['learning_rate']}")
    print()

    # Create NBM model
    model = SparseNBMROPModel(hparams,input_data_range = (train_dataset.input_df_normalized_min, train_dataset.input_df_normalized_max))

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints-NB2M-GS-C7',
        filename='NBMROPModel-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    # CSV logger
    logger = CSVLogger('./log/logsC7-NB2M-GS/', name='NBMROPModel')

    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        #devices=[5],
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=30
    )

    print("Starting training...")
    stages: dict[str, float] = {}
    with StageTimer() as t_fit:
        trainer.fit(model, train_loader, val_loader)
    stages['train_fit_s'] = t_fit.elapsed
    print("\nStarting testing...")
    with StageTimer() as t_test:
        test_result = trainer.test(model, test_loader)
    stages['train_test_s'] = t_test.elapsed
    if cost_logger is not None and cost_logger.enabled:
        cost_logger.log_global(
            n_train=len(train_dataset),
            n_val=len(val_dataset),
            n_test=len(test_dataset),
            stages=stages,
            extra={'max_epochs': 50, 'num_bases': 32},
        )
        cost_logger.flush_summary()
    print("=" * 80)
    print("Training complete!")
    print(f"Test results: {test_result}")
    print("=" * 80)

def train_test_val_split_slide(i, data=None):

    # This function splits the dataset via sliding window. The training set start depth
    # advances by 10m per iteration. Train=20m, val=10m, test=10m.
    # data can be preloaded externally to avoid repeated CSV reads.
    # Assumes column at index 1 is the depth column.
    depth_column = data.columns[1]

    # Define initial depth range
    start_depth = 980
    train_end_1 = 1000
    val_end_1 = 1010
    test_end_1 = 1020
    
    # Generate three-way split
    train_data = data[(data[depth_column] >= start_depth + i * 10) & (data[depth_column] < train_end_1 + i *10)]
    val_data = data[(data[depth_column] >= train_end_1 + i * 10) & (data[depth_column] < val_end_1 + i * 10)]
    test_data = data[(data[depth_column] >= val_end_1 + i * 10) & (data[depth_column] < test_end_1 + i *10)]

    # Some windows may be empty due to missing raw data; let caller skip
    if train_data.empty or val_data.empty or test_data.empty:
        raise ValueError(
            f"window i={i} has empty split: "
            f"train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
        )

    return train_data, val_data, test_data

def train_NBM_CL_model(i, well_data=None, use_warm_start=True, cost_logger: CostProfileLogger | None = None):
    """Train NBM model with continuous learning."""
    wall0 = time.perf_counter()
    stages: dict[str, float] = {}
    with StageTimer() as t_split:
        train_data, val_data, test_data = train_test_val_split_slide(i, data=well_data)
    stages['split_s'] = t_split.elapsed

    train_dataset = ContinuousLearningROPDataset(train_data, scaler)
    val_dataset = ContinuousLearningROPDataset(val_data, scaler)
    test_dataset = ContinuousLearningROPDataset(test_data, scaler)

    # Data is already in memory with tiny samples; multiprocessing is counterproductive:
    # num_workers=0 + pin_memory=True is usually the fastest combination
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

    # NBM model hyperparameters
    hparams = {
        'nbm_params': {
            'num_concepts': len(train_dataset.input_col),
            'num_classes': 1,
            'nary': [1],
            'num_bases': 8,
            'hidden_dims': (8, 8),
            'num_subnets': 1,
            'dropout': 0.2,
            'bases_dropout': 0.2,
            'batchnorm': True,
            'output_penalty': 0.4,
        },
        'is_sparse': False,
        'gumbel_layer_params': {
            'd': len(train_dataset.input_col),
            'k': 9,
            'tau': 0.1,
        },
        'optimizer': {
            'learning_rate': 5e-2,
            'weight_decay': 1e-4,
        }
    }

    print(f"\nModel configuration:")
    print(f"  - Input feature count: {hparams['nbm_params']['num_concepts']}")
    print(f"  - Number of basis functions: {hparams['nbm_params']['num_bases']}")
    print(f"  - Hidden layer dimensions: {hparams['nbm_params']['hidden_dims']}")
    print(f"  - N-ary interactions: {hparams['nbm_params']['nary']}")
    print(f"  - Learning rate: {hparams['optimizer']['learning_rate']}")
    print()

    ckpt_dir = './checkpoints-NBM-Q2-Q3-cost/Q2-CL'
    input_data_range = (train_dataset.input_df_normalized_min, train_dataset.input_df_normalized_max)

    prev_ckpt = None
    if use_warm_start and i > 0:
        prev_ckpt = find_best_ckpt_for_window(ckpt_dir, i - 1)

    picked = 'cold'
    with StageTimer() as t_init:
        if prev_ckpt is not None:
            model = SparseNBMROPModel.load_from_checkpoint(
                prev_ckpt,
                input_data_range=input_data_range,
            )
            picked = 'warm'
            print(f"[WarmStart] i={i}, loaded: {prev_ckpt}")
        else:
            model = SparseNBMROPModel(hparams, input_data_range=input_data_range)
            if use_warm_start and i > 0:
                print(f"[WarmStart] i={i}, previous checkpoint not found, fallback to random init.")
    stages['init_load_s'] = t_init.elapsed

    modelName = str(i).zfill(2)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f'{modelName}'+'-CLROPModel-{epoch:02d}-{val_loss:.2f}-5e-2',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    logger = CSVLogger('./log/logsA1CLModel-1e-2-2-Slide10', name='A3-stable')

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
            extra={'num_bases': 8, 'max_epochs': 100},
        )


def train_NBM_FT_E_model(i, well_data=None, use_warm_start=True, cost_logger: CostProfileLogger | None = None):
    """Train NBM model (FT-E sliding window)."""
    wall0 = time.perf_counter()
    stages: dict[str, float] = {}
    with StageTimer() as t_split:
        train_data, val_data, test_data = train_test_val_split_slide(i, data=well_data)
    stages['split_s'] = t_split.elapsed

    train_dataset = ContinuousLearningROPDataset_NE(train_data, scaler)
    val_dataset = ContinuousLearningROPDataset_NE(val_data, scaler)
    test_dataset = ContinuousLearningROPDataset_NE(test_data, scaler)

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

    hparams = {
        'nbm_params': {
            'num_concepts': len(train_dataset.input_col),
            'num_classes': 1,
            'nary': [1],
            'num_bases': 8,
            'hidden_dims': (8, 8),
            'num_subnets': 1,
            'dropout': 0.2,
            'bases_dropout': 0.2,
            'batchnorm': True,
            'output_penalty': 0.4,
        },
        'is_sparse': False,
        'gumbel_layer_params': {
            'd': len(train_dataset.input_col),
            'k': 9,
            'tau': 0.1,
        },
        'optimizer': {
            'learning_rate': 5e-2,
            'weight_decay': 1e-4,
        }
    }

    print(f"\nModel configuration:")
    print(f"  - Input feature count: {hparams['nbm_params']['num_concepts']}")
    print(f"  - Number of basis functions: {hparams['nbm_params']['num_bases']}")
    print(f"  - Hidden layer dimensions: {hparams['nbm_params']['hidden_dims']}")
    print(f"  - N-ary interactions: {hparams['nbm_params']['nary']}")
    print(f"  - Learning rate: {hparams['optimizer']['learning_rate']}")
    print()

    ckpt_dir = './checkpoints-NBM-V2-trans/A2-FT-E'
    input_data_range = (train_dataset.input_df_normalized_min, train_dataset.input_df_normalized_max)

    prev_ckpt = None
    if use_warm_start and i > 0:
        prev_ckpt = find_best_ckpt_for_window(ckpt_dir, i - 1)

    picked = 'cold'
    with StageTimer() as t_init:
        if prev_ckpt is not None:
            model = SparseNBMROPModel.load_from_checkpoint(
                prev_ckpt,
                input_data_range=input_data_range,
            )
            picked = 'warm'
            print(f"[WarmStart] i={i}, loaded: {prev_ckpt}")
        else:
            model = SparseNBMROPModel(hparams, input_data_range=input_data_range)
            if use_warm_start and i > 0:
                print(f"[WarmStart] i={i}, previous checkpoint not found, fallback to random init.")
    stages['init_load_s'] = t_init.elapsed

    modelName = str(i).zfill(2)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f'{modelName}'+'-CLROPModel-{epoch:02d}-{val_loss:.2f}-5e-2',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    logger = CSVLogger('./log/logsA2FTModel-1e-2-2-Slide10', name='A2-stable')

    trainer = pl.Trainer(
        accelerator="gpu",
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
            extra={'mode': 'FT-E', 'num_bases': 8, 'max_epochs': 100},
        )


def train_NBM_FT_model(
    i,
    well_data=None,
    base_ckpt_path='./checkpoints-NBM-Q2-Q3-cost/NBMROPModel-epoch=00-val_loss=0.00.ckpt',
    ckpt_dir='./checkpoints-NBM-Q2-Q3-cost/Q2-FT',
    log_dir='./log/logsQ2-NBM-FT',
    use_warm_start=True,
    cost_logger: CostProfileLogger | None = None,
    ):
    """NBM residual fine-tuning: freeze cross-well NBM backbone, train residual MLP in sliding window.

    - ``base_ckpt_path``: All-for-one pretrained NBM weights.
    - When ``use_warm_start=True`` and ``i>0``, load the best FT residual weights from
      the previous window in ``ckpt_dir``.
    """
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

    input_data_range = (
        train_dataset.input_df_normalized_min,
        train_dataset.input_df_normalized_max,
    )

    params = {
        'input_dim': len(train_dataset.input_col),
        'hidden_dim': 8,
        'output_dim': 1,
        'dropout': 0.2,
        'learning_rate': 5e-2,
        'weight_decay': 1e-4,
        'input_data_range': input_data_range,
    }

    picked = 'cold'
    with StageTimer() as t_base:
        pretrained_base = SparseNBMROPModel.load_from_checkpoint(
            base_ckpt_path,
            input_data_range=input_data_range,
        )
    stages['infer_base_s'] = t_base.elapsed

    prev_ckpt = None
    if use_warm_start and i > 0:
        prev_ckpt = find_best_ckpt_for_window(
            ckpt_dir, i - 1, model_name='FTNBMResidualModel',
        )

    with StageTimer() as t_init:
        if prev_ckpt is not None:
            model = NBMFTResidualModel.load_from_checkpoint(
                prev_ckpt,
                backbone_model=pretrained_base,
                params=params,
                input_data_range=input_data_range,
            )
            picked = 'warm'
            print(f"[NBM-FT-WarmStart] i={i}, loaded: {prev_ckpt}")
        else:
            model = NBMFTResidualModel(
                backbone_model=pretrained_base,
                params=params,
                input_data_range=input_data_range,
            )
            if use_warm_start and i > 0:
                print(
                    f"[NBM-FT-WarmStart] i={i}, previous checkpoint not found, "
                    "fallback to fresh residual net.",
                )
    stages['init_load_s'] = t_init.elapsed

    os.makedirs(ckpt_dir, exist_ok=True)
    modelName = str(i).zfill(2)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f'{modelName}' + '-FTNBMResidualModel-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
    )
    logger = CSVLogger(log_dir, name='FT-NBM-Residual')

    trainer = pl.Trainer(
        accelerator="gpu",
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
    p = argparse.ArgumentParser(description='NBM ROP training (supports --profile-cost)')
    p.add_argument('--mode', choices=['global', 'cl', 'ft', 'ft-e'], default='cl')
    p.add_argument('--well-csv', default='./data/ProcessedData-2/SZ36-1-Q2.csv')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=400)
    p.add_argument(
        '--base-ckpt',
        default='./checkpoints-NBM-Q2-Q3-cost/NBMROPModel-epoch=12-val_loss=1683.66.ckpt',
        help='All-for-one pretrained NBM weights (frozen backbone for FT)',
    )
    p.add_argument('--ckpt-dir', default='./checkpoints-NBM-Q2-Q3-cost/Q2-FT')
    p.add_argument('--log-dir', default='./log/logsQ2-NBM-FT')
    add_cost_profile_args(p)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    short = short_well_tag(args.well_csv)
    # python train_nbm.py --mode global
    if args.mode == 'global':
        cost = CostProfileLogger.from_args(args, model='NBM5', well='global', mode='global')
        train_NBM_model(cost_logger=cost)
    else:
        pattern_mode = {'cl': 'cl', 'ft': 'ft', 'ft-e': 'ft'}[args.mode]
        cost = CostProfileLogger.from_args(args, model='NBM5', well=short, mode=pattern_mode)
        well_data = pd.read_csv(args.well_csv)
        for i in range(args.start, args.end):
            try:
                if args.mode == 'cl':
                    train_NBM_CL_model(
                        i, well_data=well_data, use_warm_start=True, cost_logger=cost,
                    )
                elif args.mode == 'ft-e':
                    train_NBM_FT_E_model(
                        i, well_data=well_data, use_warm_start=True, cost_logger=cost,
                    )
                else:
                    train_NBM_FT_model(
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
