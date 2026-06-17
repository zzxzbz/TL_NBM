"""Unified computational cost / wall-clock time recording (for GAM, RF, XGB, BPNN, NBM training scripts).

Usage:
    from cost_profile import CostProfileLogger, StageTimer, add_cost_profile_args

    logger = CostProfileLogger.from_args(args, model='XGB', pattern='CL', well='Q2')
    with StageTimer() as t:
        ...
    logger.log_window(i=0, n_train=..., stages={'train_cold_s': t.elapsed, ...})
    logger.flush_summary()
"""

from __future__ import annotations

import json
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

PATTERN_FROM_MODE = {
    'global': 'All-for-One',
    'cl': 'Continuous Learning',
    'ft': 'Transfer Learning',
}

WINDOW_COLUMNS = [
    'timestamp_utc',
    'hostname',
    'well',
    'model',
    'pattern',
    'window_i',
    'n_train',
    'n_val',
    'n_test',
    'split_s',
    'scale_s',
    'train_cold_s',
    'train_warm_s',
    'select_s',
    'infer_base_s',
    'infer_test_s',
    'init_load_s',
    'train_fit_s',
    'train_test_s',
    'wall_window_s',
    'picked',
    'extra_json',
]

SUMMARY_COLUMNS = [
    'timestamp_utc',
    'hostname',
    'well',
    'model',
    'pattern',
    'n_windows',
    'n_skipped',
    'total_train_cold_s',
    'total_train_warm_s',
    'total_select_s',
    'total_infer_base_s',
    'total_infer_test_s',
    'total_init_load_s',
    'total_train_fit_s',
    'total_train_test_s',
    'total_wall_s',
    'total_n_test_pred',
    'extra_json',
]


@dataclass
class StageTimer:
    """``with StageTimer() as t: ...; t.elapsed``"""

    start: float = field(default_factory=time.perf_counter)
    elapsed: float = 0.0

    def __enter__(self) -> 'StageTimer':
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self.start


class CostProfileLogger:
    """Append per-window CSV, write summary at end of training."""

    def __init__(
        self,
        *,
        enabled: bool,
        log_dir: str,
        model: str,
        pattern: str,
        well: str,
        mode: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.model = model
        self.pattern = pattern or PATTERN_FROM_MODE.get(mode or '', mode or '')
        self.well = well
        self.mode = mode
        self._rows: list[dict[str, Any]] = []
        self._skipped = 0
        self._hostname = socket.gethostname()

        if not enabled:
            self.window_csv = None
            self.summary_csv = None
            return

        os.makedirs(log_dir, exist_ok=True)
        tag = f'{well}_{model}_{self.pattern.replace(" ", "-")}'
        self.window_csv = os.path.join(log_dir, f'cost_windows_{tag}.csv')
        self.summary_csv = os.path.join(log_dir, f'cost_summary_{tag}.csv')

    @classmethod
    def from_args(cls, args, *, model: str, well: str, mode: str | None = None) -> 'CostProfileLogger':
        enabled = bool(getattr(args, 'profile_cost', False))
        log_dir = getattr(args, 'cost_log_dir', './cost_profile')
        pattern = PATTERN_FROM_MODE.get(mode or getattr(args, 'mode', ''), '')
        return cls(
            enabled=enabled,
            log_dir=log_dir,
            model=model,
            pattern=pattern,
            well=well,
            mode=mode or getattr(args, 'mode', None),
        )

    @contextmanager
    def disabled(self):
        """Temporarily disable recording."""
        prev = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = prev

    def skip_window(self) -> None:
        if self.enabled:
            self._skipped += 1

    def log_window(
        self,
        *,
        window_i: int,
        n_train: int = 0,
        n_val: int = 0,
        n_test: int = 0,
        stages: Optional[Dict[str, float]] = None,
        picked: str = '',
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        st = stages or {}
        wall = float(st.get('wall_window_s', 0.0))
        if wall <= 0:
            wall = sum(
                float(st.get(k, 0.0) or 0.0)
                for k in (
                    'split_s', 'scale_s', 'train_cold_s', 'train_warm_s',
                    'select_s', 'infer_base_s', 'infer_test_s',
                    'init_load_s', 'train_fit_s', 'train_test_s',
                )
            )

        row = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'hostname': self._hostname,
            'well': self.well,
            'model': self.model,
            'pattern': self.pattern,
            'window_i': int(window_i),
            'n_train': int(n_train),
            'n_val': int(n_val),
            'n_test': int(n_test),
            'split_s': st.get('split_s', 0.0),
            'scale_s': st.get('scale_s', 0.0),
            'train_cold_s': st.get('train_cold_s', 0.0),
            'train_warm_s': st.get('train_warm_s', 0.0),
            'select_s': st.get('select_s', 0.0),
            'infer_base_s': st.get('infer_base_s', 0.0),
            'infer_test_s': st.get('infer_test_s', 0.0),
            'init_load_s': st.get('init_load_s', 0.0),
            'train_fit_s': st.get('train_fit_s', 0.0),
            'train_test_s': st.get('train_test_s', 0.0),
            'wall_window_s': wall,
            'picked': picked,
            'extra_json': json.dumps(extra or {}, ensure_ascii=False),
        }
        self._rows.append(row)
        self._append_csv(self.window_csv, row, WINDOW_COLUMNS)

    def log_global(
        self,
        *,
        n_train: int = 0,
        n_val: int = 0,
        n_test: int = 0,
        stages: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cross-well / one-shot training (no sliding window index)."""
        self.log_window(
            window_i=-1,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            stages=stages,
            picked='global',
            extra=extra,
        )

    def flush_summary(self) -> None:
        if not self.enabled or not self._rows:
            return

        df = pd.DataFrame(self._rows)
        summary = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'hostname': self._hostname,
            'well': self.well,
            'model': self.model,
            'pattern': self.pattern,
            'n_windows': int((df['window_i'] >= 0).sum()),
            'n_skipped': self._skipped,
            'total_train_cold_s': float(df['train_cold_s'].sum()),
            'total_train_warm_s': float(df['train_warm_s'].sum()),
            'total_select_s': float(df['select_s'].sum()),
            'total_infer_base_s': float(df['infer_base_s'].sum()),
            'total_infer_test_s': float(df['infer_test_s'].sum()),
            'total_init_load_s': float(df['init_load_s'].sum()),
            'total_train_fit_s': float(df['train_fit_s'].sum()),
            'total_train_test_s': float(df['train_test_s'].sum()),
            'total_wall_s': float(df['wall_window_s'].sum()),
            'total_n_test_pred': int(df['n_test'].sum()),
            'extra_json': json.dumps({'mode': self.mode}, ensure_ascii=False),
        }
        self._append_csv(self.summary_csv, summary, SUMMARY_COLUMNS)
        print(
            f'[cost_profile] {self.model} {self.pattern} @ {self.well}: '
            f'{summary["n_windows"]} windows, wall={summary["total_wall_s"]:.2f}s '
            f'-> {self.summary_csv}',
            flush=True,
        )

    @staticmethod
    def _append_csv(path: str, row: dict, columns: list[str]) -> None:
        line = pd.DataFrame([row]).reindex(columns=columns)
        header = not os.path.exists(path)
        line.to_csv(path, mode='a', header=header, index=False)


def add_cost_profile_args(parser) -> None:
    parser.add_argument(
        '--profile-cost',
        action='store_true',
        help='Record per-stage wall-clock time to --cost-log-dir (for cost comparison)',
    )
    parser.add_argument(
        '--cost-log-dir',
        default='./cost_profile',
        help='Cost/profile CSV output directory (default ./cost_profile)',
    )


def short_well_tag(well_path_or_tag: str) -> str:
    """SZ36-1-Q2.csv / SZ36-1-Q2 -> Q2"""
    name = os.path.splitext(os.path.basename(well_path_or_tag))[0]
    return name.split('-')[-1].rstrip('H')
