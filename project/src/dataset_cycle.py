"""
dataset_cycle.py — Cycle-history dataset for LSTM/Transformer SOH prediction.

Instead of feeding each model a single RW step (301, 3), this dataset
builds sequences of block-level summaries so models can observe degradation
over time rather than sensor dynamics within one event.

One sample = window of W consecutive blocks from the same battery.
Each block is summarized into F=15 features computed from all RW steps in
that block. The label is the SOH at the last block in the window.

Block feature vector (15 features):
    Voltage stats   (4): mean/std/min/max of per-step voltage means
    Current stats   (4): mean/std/min/max of per-step current means
    Temp stats      (4): mean/std/min/max of per-step temperature means
    Step count      (1): number of RW steps in this block
    Charge ratio    (1): fraction of steps with positive mean current
    Norm block idx  (1): block_index / (total_blocks - 1), range [0, 1]

Input shape:  (W, 15)
Output shape: scalar SOH in [0, 1]

Usage:
    from src.dataset_cycle import CycleHistoryDataset

    train_ds = CycleHistoryDataset(h5_path, window=10, split='train')
    test_ds  = CycleHistoryDataset(h5_path, window=10, split='test',
                                   norm_mean=train_ds.norm_mean,
                                   norm_std=train_ds.norm_std)
"""

from __future__ import annotations

from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset import DEFAULT_TEST_BATTERIES

N_FEATURES: int = 15
WINDOW_DEFAULT: int = 10


# ---------------------------------------------------------------------------
# Block feature extraction
# ---------------------------------------------------------------------------


def _build_all_block_features(
    h5_path: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Read sequences.h5 and compute per-block summary features for every battery.

    Returns a dict mapping battery_id (str) to a tuple of:
        feats: float32 array (M, N_FEATURES)  — one row per block
        sohs:  float32 array (M,)             — SOH for each block
    where M is the number of blocks in that battery.

    Blocks are ordered by block_index (ascending).
    """
    battery_data: dict[str, dict] = {}

    with h5py.File(h5_path, "r") as f:
        X = f["X"][:]              # (N, 301, 3)
        y = f["y"][:]              # (N,)  SOH in %
        bids_raw = f["battery_id"][:]   # (N,) bytes
        blk_idx = f["block_index"][:]   # (N,) int32

    # Decode battery ids
    bids = np.array([b.decode("ascii") if isinstance(b, bytes) else b for b in bids_raw])

    for battery in np.unique(bids):
        mask = bids == battery
        Xb = X[mask]          # (Nb, 301, 3)
        yb = y[mask]          # (Nb,)
        blk = blk_idx[mask]   # (Nb,)
        battery_data[battery] = {"X": Xb, "y": yb, "block_index": blk}

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for battery, data in battery_data.items():
        Xb = data["X"]
        yb = data["y"]
        blk = data["block_index"]

        unique_blocks = np.sort(np.unique(blk))
        n_blocks = len(unique_blocks)

        feats_list: list[np.ndarray] = []
        soh_list: list[float] = []

        for bi, block_id in enumerate(unique_blocks):
            bmask = blk == block_id
            steps = Xb[bmask]    # (S, 301, 3)

            # Per-step channel means
            step_v_mean = steps[:, :, 0].mean(axis=1)   # (S,)
            step_c_mean = steps[:, :, 1].mean(axis=1)   # (S,)
            step_t_mean = steps[:, :, 2].mean(axis=1)   # (S,)

            v_mean = float(step_v_mean.mean())
            v_std  = float(step_v_mean.std()) if len(step_v_mean) > 1 else 0.0
            v_min  = float(step_v_mean.min())
            v_max  = float(step_v_mean.max())

            c_mean = float(step_c_mean.mean())
            c_std  = float(step_c_mean.std()) if len(step_c_mean) > 1 else 0.0
            c_min  = float(step_c_mean.min())
            c_max  = float(step_c_mean.max())

            t_mean = float(step_t_mean.mean())
            t_std  = float(step_t_mean.std()) if len(step_t_mean) > 1 else 0.0
            t_min  = float(step_t_mean.min())
            t_max  = float(step_t_mean.max())

            step_count = float(len(steps))
            charge_ratio = float((step_c_mean > 0).sum()) / max(len(step_c_mean), 1)
            norm_block = float(bi) / max(n_blocks - 1, 1)

            feat = np.array(
                [
                    v_mean, v_std, v_min, v_max,
                    c_mean, c_std, c_min, c_max,
                    t_mean, t_std, t_min, t_max,
                    step_count,
                    charge_ratio,
                    norm_block,
                ],
                dtype=np.float32,
            )
            feats_list.append(feat)
            soh_list.append(float(yb[bmask][0]))

        result[battery] = (
            np.stack(feats_list, axis=0),   # (M, 15)
            np.array(soh_list, dtype=np.float32),  # (M,)
        )

    return result


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class CycleHistoryDataset(Dataset):
    """
    Sliding-window dataset over block-level battery summaries.

    Each sample is a window of W consecutive blocks from the same battery,
    represented as (W, 15) float32 tensor. The label is the SOH at the
    last block in the window, in [0, 1].

    Args:
        h5_path:     Path to sequences.h5.
        window:      Number of consecutive blocks per sequence (default 10).
        split:       'train' or 'test'. Test batteries are fixed.
        test_batteries: Override the default test battery list.
        norm_mean:   If provided, use this mean for normalization instead of
                     computing from this split (pass train_ds.norm_mean to
                     the test dataset).
        norm_std:    Paired with norm_mean.
        normalize:   Whether to z-normalize features (default True).
    """

    FEATURE_NAMES: list[str] = [
        "v_mean", "v_std", "v_min", "v_max",
        "c_mean", "c_std", "c_min", "c_max",
        "t_mean", "t_std", "t_min", "t_max",
        "step_count", "charge_ratio", "norm_block_idx",
    ]

    def __init__(
        self,
        h5_path: str,
        window: int = WINDOW_DEFAULT,
        split: str = "train",
        test_batteries: Optional[list[str]] = None,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        normalize: bool = True,
    ) -> None:
        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split!r}"

        self.window = window
        self.split = split

        if test_batteries is None:
            test_batteries = DEFAULT_TEST_BATTERIES
        test_set = set(test_batteries)

        all_block_feats = _build_all_block_features(h5_path)

        # Partition batteries
        train_batteries = [b for b in all_block_feats if b not in test_set]
        test_batteries_present = [b for b in all_block_feats if b in test_set]
        active_batteries = test_batteries_present if split == "test" else train_batteries

        # Build sliding-window index: list of (feats_array, soh_array, start_idx)
        sequences_X: list[np.ndarray] = []
        sequences_y: list[float] = []

        for battery in sorted(active_batteries):
            feats, sohs = all_block_feats[battery]
            M = len(feats)
            if M < window:
                continue
            for i in range(window - 1, M):
                seq = feats[i - window + 1 : i + 1]   # (W, 15)
                label = sohs[i] / 100.0                # → [0, 1]
                sequences_X.append(seq)
                sequences_y.append(label)

        self._X = np.stack(sequences_X, axis=0)  # (N_seq, W, 15)
        self._y = np.array(sequences_y, dtype=np.float32)  # (N_seq,)

        # Normalization
        if normalize:
            if norm_mean is not None and norm_std is not None:
                self.norm_mean = norm_mean
                self.norm_std = norm_std
            else:
                # Compute from this split's data (should only be train)
                flat = self._X.reshape(-1, N_FEATURES)
                self.norm_mean = flat.mean(axis=0).astype(np.float32)
                self.norm_std = flat.std(axis=0).astype(np.float32)
            self._X = (self._X - self.norm_mean) / (self.norm_std + 1e-8)
        else:
            self.norm_mean = np.zeros(N_FEATURES, dtype=np.float32)
            self.norm_std = np.ones(N_FEATURES, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self._X[idx])   # (W, 15)
        y = torch.tensor(self._y[idx], dtype=torch.float32)
        return x, y

    def describe(self) -> None:
        """Print diagnostic stats about this split."""
        n = len(self._y)
        soh_pct = self._y * 100.0
        print(f"Split: {self.split}  |  Sequences: {n:,}  |  Window: {self.window}")
        print(f"SOH range: {soh_pct.min():.1f}% – {soh_pct.max():.1f}%  "
              f"(mean {soh_pct.mean():.1f}%)")
        print(f"Sample shape: {tuple(self._X[0].shape)}")

        print("\nWithin-sequence SOH variation (first position vs last, 10 random samples):")
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n, size=min(10, n), replace=False)
        print(f"  {'idx':>6}  {'SOH[0]%':>9}  {'SOH[-1]%':>10}  {'delta':>7}")
        for i in sample_idx:
            # Reconstruct un-normalized SOH from first/last positions is not
            # possible from X alone since we normalized. Use _y which is the
            # last block's SOH. For the first block's SOH we can't recover it
            # directly — instead print the norm_block_idx feature values.
            seq = self._X[i]   # (W, 15) normalized
            # norm_block_idx is feature index 14; just print its first and last
            nbi_first = float(seq[0, 14])
            nbi_last  = float(seq[-1, 14])
            print(f"  {i:>6}  block_norm[0]={nbi_first:+.3f}  block_norm[-1]={nbi_last:+.3f}  "
                  f"label_soh={self._y[i]*100:.1f}%")
