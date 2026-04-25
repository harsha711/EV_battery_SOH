"""
dataset_hybrid.py — Hybrid dataset: window of blocks, K raw steps per block.

Each sample is a window of W consecutive blocks from the same battery. Each
block is represented by K raw (301, 3) steps sampled from the steps that
belong to that block. The label is the SOH at the last block in the window.

Sample shape:  (W, K, 301, 3)
Label:         scalar SOH in [0, 1]

Step sampling: K steps are drawn uniformly with replacement from the block's
available steps each time __getitem__ is called. This acts as data
augmentation during training. For deterministic evaluation pass
`deterministic=True` so the first K steps (or all steps with cycling) are
returned instead.
"""

from __future__ import annotations

from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset import DEFAULT_TEST_BATTERIES, DEFAULT_VAL_BATTERIES

WINDOW_DEFAULT: int = 10
K_DEFAULT: int = 32
SEQ_LEN: int = 301
N_CHANNELS: int = 3


def _build_battery_block_index(
    h5_path: str,
) -> dict[str, list[tuple[int, np.ndarray, float]]]:
    """
    Read battery_id, block_index, and y arrays once and group HDF5 row indices
    by (battery, block).

    Returns dict:
        battery_id -> list of (block_id, row_indices, soh_label)
        where the list is sorted ascending by block_id.

    Raw X data is NOT loaded here — only the metadata, so this is fast and low
    memory. The caller (Dataset.__getitem__) reads X rows on demand.
    """
    with h5py.File(h5_path, "r") as f:
        bids_raw = f["battery_id"][:]
        blk_idx = f["block_index"][:]
        y = f["y"][:]

    bids = np.array(
        [b.decode("ascii") if isinstance(b, bytes) else b for b in bids_raw]
    )

    result: dict[str, list[tuple[int, np.ndarray, float]]] = {}
    for battery in np.unique(bids):
        bmask = bids == battery
        battery_rows = np.where(bmask)[0]
        battery_blocks = blk_idx[bmask]
        battery_y = y[bmask]

        unique_blocks = np.sort(np.unique(battery_blocks))
        block_list: list[tuple[int, np.ndarray, float]] = []
        for blk in unique_blocks:
            in_block = battery_blocks == blk
            row_indices = battery_rows[in_block]
            soh = float(battery_y[in_block][0])
            block_list.append((int(blk), row_indices, soh))
        result[battery] = block_list

    return result


class HybridDataset(Dataset):
    """
    Sliding-window dataset over (W blocks × K steps × 301 × 3) tensors.

    Args:
        h5_path:        Path to sequences.h5.
        window:         Number of consecutive blocks per sequence (default 10).
        k_steps:        Number of raw steps sampled per block (default 32).
        split:          'train', 'val', or 'test'. Battery sets follow the
                        defaults in src/dataset.py.
        val_batteries:  Override default validation batteries.
        test_batteries: Override default test batteries.
        norm_mean:      Per-channel mean (3,) for normalization. If None and
                        split == 'train', computed from training rows; otherwise
                        must be provided.
        norm_std:       Paired with norm_mean.
        normalize:      If True, z-normalize each channel (default True).
        deterministic:  If True, use a fixed seed per __getitem__ so step
                        sampling is reproducible (used for val/test).
        seed:           Base seed for deterministic sampling.
        preloaded_X:    Optional in-memory copy of the full X dataset
                        (numpy array, shape matching sequences.h5 X). When
                        provided, __getitem__ skips HDF5 reads entirely —
                        ~5–10× speedup. Use HybridDataset.preload_X() to
                        build it once and share across train/val/test.
    """

    def __init__(
        self,
        h5_path: str,
        window: int = WINDOW_DEFAULT,
        k_steps: int = K_DEFAULT,
        split: str = "train",
        val_batteries: Optional[list[str]] = None,
        test_batteries: Optional[list[str]] = None,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        normalize: bool = True,
        deterministic: bool = False,
        seed: int = 0,
        preloaded_X: Optional[np.ndarray] = None,
    ) -> None:
        assert split in ("train", "val", "test"), (
            f"split must be 'train', 'val', or 'test'; got {split!r}"
        )

        self.h5_path = h5_path
        self.window = window
        self.k_steps = k_steps
        self.split = split
        self.deterministic = deterministic
        self.seed = seed

        if val_batteries is None:
            val_batteries = DEFAULT_VAL_BATTERIES
        if test_batteries is None:
            test_batteries = DEFAULT_TEST_BATTERIES
        val_set, test_set = set(val_batteries), set(test_batteries)
        if val_set & test_set:
            raise ValueError(
                f"val and test battery sets overlap: {val_set & test_set}"
            )

        battery_blocks = _build_battery_block_index(h5_path)

        if split == "test":
            active = [b for b in battery_blocks if b in test_set]
        elif split == "val":
            active = [b for b in battery_blocks if b in val_set]
        else:
            active = [
                b for b in battery_blocks
                if b not in val_set and b not in test_set
            ]
        active = sorted(active)

        # Build per-sample index: list of (battery, [block_id, row_indices]_per_block, label)
        self._samples: list[tuple[str, list[np.ndarray], float]] = []
        for battery in active:
            blocks = battery_blocks[battery]
            M = len(blocks)
            if M < window:
                continue
            for end in range(window - 1, M):
                window_blocks = blocks[end - window + 1 : end + 1]
                row_lists = [b[1] for b in window_blocks]
                label = window_blocks[-1][2] / 100.0
                self._samples.append((battery, row_lists, label))

        # In-memory dataset (preferred) or lazy HDF5 fallback
        self._X_mem: Optional[np.ndarray] = preloaded_X
        self._file: Optional[h5py.File] = None

        # Normalization
        self.normalize = normalize
        if normalize:
            if norm_mean is not None and norm_std is not None:
                self.norm_mean = np.asarray(norm_mean, dtype=np.float32)
                self.norm_std = np.asarray(norm_std, dtype=np.float32)
            elif split == "train":
                self.norm_mean, self.norm_std = self._compute_norm_stats(active, battery_blocks)
            else:
                raise ValueError(
                    f"split={split!r} requires norm_mean/norm_std (compute "
                    f"them on the training split first)"
                )
        else:
            self.norm_mean = np.zeros(N_CHANNELS, dtype=np.float32)
            self.norm_std = np.ones(N_CHANNELS, dtype=np.float32)

    def _compute_norm_stats(
        self,
        active_batteries: list[str],
        battery_blocks: dict[str, list[tuple[int, np.ndarray, float]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Channel-wise mean/std over all rows in active batteries."""
        all_rows: list[np.ndarray] = []
        for battery in active_batteries:
            for _, rows, _ in battery_blocks[battery]:
                all_rows.append(rows)
        all_idx = np.sort(np.concatenate(all_rows))

        # Two-pass over a single open file
        n_channels = N_CHANNELS
        ch_sum = np.zeros(n_channels, dtype=np.float64)
        ch_sq = np.zeros(n_channels, dtype=np.float64)
        n_pts = 0
        chunk = 4096
        with h5py.File(self.h5_path, "r") as f:
            X = f["X"]
            for start in range(0, len(all_idx), chunk):
                idx_chunk = all_idx[start : start + chunk]
                data = X[idx_chunk.tolist()]   # (chunk, 301, 3)
                flat = data.reshape(-1, n_channels)
                ch_sum += flat.sum(axis=0)
                ch_sq += (flat ** 2).sum(axis=0)
                n_pts += flat.shape[0]
        mean = ch_sum / n_pts
        var = ch_sq / n_pts - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-12))
        return mean.astype(np.float32), std.astype(np.float32)

    def _open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    @staticmethod
    def preload_X(h5_path: str) -> np.ndarray:
        """Load the full X dataset into RAM once.

        ~1.52 GB for 450K × 301 × 3 × float32. Pass the returned array as
        `preloaded_X` to all three (train/val/test) Datasets to skip HDF5
        reads in the training loop.
        """
        with h5py.File(h5_path, "r") as f:
            return f["X"][:]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        battery, row_lists, label = self._samples[idx]

        if self.deterministic:
            rng = np.random.default_rng(self.seed + idx)
        else:
            rng = np.random.default_rng()

        K = self.k_steps
        out = np.empty((self.window, K, SEQ_LEN, N_CHANNELS), dtype=np.float32)

        if self._X_mem is not None:
            # Fast path: index directly into the in-memory tensor.
            X = self._X_mem
            for w, rows in enumerate(row_lists):
                n_avail = len(rows)
                if n_avail >= K:
                    pick = rng.choice(n_avail, size=K, replace=False)
                else:
                    pick = rng.integers(0, n_avail, size=K)
                out[w] = X[rows[pick]]
        else:
            # Lazy HDF5 path (slower; used when preloaded_X not provided).
            self._open()
            for w, rows in enumerate(row_lists):
                n_avail = len(rows)
                if n_avail >= K:
                    pick = rng.choice(n_avail, size=K, replace=False)
                    sampled_rows = np.sort(rows[pick])
                    out[w] = self._file["X"][sampled_rows.tolist()]
                else:
                    unique_rows = np.sort(rows)
                    unique_data = self._file["X"][unique_rows.tolist()]
                    pick = rng.integers(0, n_avail, size=K)
                    out[w] = unique_data[pick]

        if self.normalize:
            out = (out - self.norm_mean) / (self.norm_std + 1e-8)

        x = torch.from_numpy(out)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

    def describe(self) -> None:
        n = len(self._samples)
        labels = np.array([s[2] for s in self._samples]) * 100.0
        print(
            f"Split: {self.split}  |  Sequences: {n:,}  |  "
            f"W={self.window}, K={self.k_steps}"
        )
        if n:
            print(
                f"SOH range: {labels.min():.1f}% – {labels.max():.1f}%  "
                f"(mean {labels.mean():.1f}%)"
            )
        print(f"Sample shape: ({self.window}, {self.k_steps}, {SEQ_LEN}, {N_CHANNELS})")
