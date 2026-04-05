"""
dataset.py — PyTorch Dataset and DataLoader utilities.

Provides BatterySOHDataset: a lazy-loading HDF5 dataset that wraps the
preprocessed sequence data (sequences.h5). Supports battery-level train/test
splitting to prevent data leakage.

All 4 models (MLP, CNN, LSTM, Transformer) use this same dataset.
Each sample is a tuple (x, y) where:
    x: float32 tensor (301, 3)   — [voltage, current, temperature]
    y: float32 scalar            — SOH in [0, 1]  (original % / 100)
"""

import json
import os
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Default test battery IDs
# ---------------------------------------------------------------------------

DEFAULT_TEST_BATTERIES: list[str] = [
    "RW25", "RW26", "RW27", "RW28",  # skewed_high / 40C — OOD temperature
    "RW13", "RW14",                    # skewed_low / room_temp — OOD distribution
]
"""
Default test batteries (6 of 28):
  RW25–28: skewed_high load distribution, 40°C — tests temperature OOD generalization.
           None of the 40C skewed-high group appears in training.
  RW13–14: skewed_low load distribution, room_temp — tests distribution-shape OOD.
           The remaining skewed_low room_temp batteries (RW15–16) stay in training.
The remaining 40C group (RW21–24, skewed_low 40C) stays in training so the
model sees at least one elevated-temperature group during training.
"""


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class BatterySOHDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset backed by an HDF5 file.

    Opens the HDF5 file lazily inside __getitem__ to be safe with
    DataLoader multiprocessing (h5py.File is not fork-safe when opened
    in __init__).

    Args:
        h5_path: Path to sequences.h5 produced by preprocessing.py.
        indices: Optional 1-D integer array selecting a subset of the N
                 samples. If None, all samples are used.
        normalize: If True, z-score normalize X channel-wise using stats
                   loaded from stats_path.
        stats_path: JSON file containing normalization stats
                    {'mean': [v, c, t], 'std': [v, c, t]}.
                    Required when normalize=True.
    """

    def __init__(
        self,
        h5_path: str,
        indices: Optional[np.ndarray] = None,
        normalize: bool = True,
        stats_path: Optional[str] = None,
    ) -> None:
        self.h5_path = h5_path
        self._file: Optional[h5py.File] = None  # opened lazily — not fork-safe

        with h5py.File(h5_path, "r") as f:
            n_total = f["X"].shape[0]
            seq_len = f["X"].shape[1]
            n_channels = f["X"].shape[2]

        self.n_total = n_total
        self.seq_len = seq_len
        self.n_channels = n_channels

        if indices is None:
            self.indices = np.arange(n_total, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

        self.normalize = normalize
        self.mean: Optional[np.ndarray] = None  # (3,) channel means
        self.std: Optional[np.ndarray] = None   # (3,) channel stds

        if normalize:
            if stats_path is None:
                raise ValueError(
                    "stats_path must be provided when normalize=True. "
                    "Run compute_normalization_stats() on the training split first."
                )
            with open(stats_path, "r") as fp:
                stats = json.load(fp)
            self.mean = np.array(stats["mean"], dtype=np.float32)  # (3,)
            self.std = np.array(stats["std"], dtype=np.float32)    # (3,)

    def __len__(self) -> int:
        return len(self.indices)

    def _open(self) -> None:
        """Open the HDF5 file lazily (once per worker process)."""
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a single (x, y) pair.

        Args:
            idx: Index into self.indices (not the raw HDF5 row index).

        Returns:
            x: float32 tensor (seq_len, 3)   — normalized if normalize=True
            y: float32 scalar tensor          — SOH in [0, 1]
        """
        self._open()
        real_idx = int(self.indices[idx])

        x = self._file["X"][real_idx].astype(np.float32)  # (301, 3)
        y = float(self._file["y"][real_idx]) / 100.0       # scale % → [0,1]

        if self.normalize and self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-8)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    def compute_normalization_stats(self, stats_path: str, chunk_size: int = 4096) -> dict:
        """
        Compute channel-wise mean and std over self.indices and save to JSON.

        Iterates in chunks to avoid loading all data into RAM at once.
        Must be called on the training split only — never on the full dataset.

        Args:
            stats_path: Path to save the JSON stats file.
            chunk_size: Number of samples to process per chunk.

        Returns:
            Dict with keys 'mean' and 'std', each a list of 3 floats.
        """
        print("Computing normalization stats...", end=" ", flush=True)
        # Two-pass: first compute mean, then std
        n = len(self.indices)
        ch = self.n_channels

        # Pass 1: mean
        channel_sum = np.zeros(ch, dtype=np.float64)
        sample_count = 0

        with h5py.File(self.h5_path, "r") as f:
            for start in range(0, n, chunk_size):
                chunk_idx = self.indices[start : start + chunk_size]
                # HDF5 fancy indexing requires sorted indices
                sort_order = np.argsort(chunk_idx)
                sorted_idx = chunk_idx[sort_order]
                data = f["X"][sorted_idx.tolist()]  # (chunk, 301, 3)
                channel_sum += data.reshape(-1, ch).sum(axis=0)
                sample_count += data.shape[0] * data.shape[1]

        mean = (channel_sum / sample_count).astype(np.float32)

        # Pass 2: variance
        channel_sq_sum = np.zeros(ch, dtype=np.float64)
        with h5py.File(self.h5_path, "r") as f:
            for start in range(0, n, chunk_size):
                chunk_idx = self.indices[start : start + chunk_size]
                sort_order = np.argsort(chunk_idx)
                sorted_idx = chunk_idx[sort_order]
                data = f["X"][sorted_idx.tolist()].reshape(-1, ch)
                channel_sq_sum += ((data - mean) ** 2).sum(axis=0)

        std = np.sqrt(channel_sq_sum / sample_count).astype(np.float32)

        stats = {"mean": mean.tolist(), "std": std.tolist()}
        os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
        with open(stats_path, "w") as fp:
            json.dump(stats, fp, indent=2)

        print(f"done. mean={[f'{v:.4f}' for v in stats['mean']]}, "
              f"std={[f'{v:.4f}' for v in stats['std']]}")
        print(f"  Stats saved to {stats_path}")
        return stats

    @classmethod
    def create_splits(
        cls,
        h5_path: str,
        test_batteries: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create battery-level train/test index splits.

        Guarantees that no sample from a test battery appears in the training
        set. Splitting is at the battery level, not the step level, to prevent
        label leakage (steps from the same block share one SOH label).

        Args:
            h5_path: Path to sequences.h5.
            test_batteries: List of battery IDs to hold out for testing.
                            Defaults to DEFAULT_TEST_BATTERIES.

        Returns:
            (train_indices, test_indices) as int64 numpy arrays.
        """
        if test_batteries is None:
            test_batteries = DEFAULT_TEST_BATTERIES

        test_set = set(test_batteries)

        with h5py.File(h5_path, "r") as f:
            battery_ids = np.array(
                [b.decode("ascii") for b in f["battery_id"][:]]
            )
            n = len(battery_ids)

        all_indices = np.arange(n, dtype=np.int64)
        in_test = np.isin(battery_ids, list(test_set))

        train_indices = all_indices[~in_test]
        test_indices = all_indices[in_test]

        train_batteries = sorted(set(battery_ids[~in_test].tolist()))
        actual_test = sorted(set(battery_ids[in_test].tolist()))

        print(f"Train: {len(train_indices):,} steps from {len(train_batteries)} batteries: "
              f"{train_batteries}")
        print(f"Test:  {len(test_indices):,} steps from {len(actual_test)} batteries: "
              f"{actual_test}")

        return train_indices, test_indices


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def get_dataloaders(
    h5_path: str,
    stats_path: str,
    test_batteries: Optional[list[str]] = None,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders with battery-level splitting.

    Normalization stats are computed from the training split only and saved
    to stats_path. If stats_path already exists, stats are loaded from disk.

    Args:
        h5_path: Path to sequences.h5.
        stats_path: JSON path for normalization stats.
        test_batteries: Battery IDs to hold out. Defaults to DEFAULT_TEST_BATTERIES.
        batch_size: Batch size for both loaders (default 256).
        num_workers: DataLoader worker processes. Keep at 0 to avoid h5py
                     fork-safety issues unless using a worker_init_fn.

    Returns:
        (train_loader, test_loader)
    """
    train_idx, test_idx = BatterySOHDataset.create_splits(h5_path, test_batteries)

    # Compute stats from training split if not already cached
    if not os.path.exists(stats_path):
        tmp = BatterySOHDataset(h5_path, indices=train_idx, normalize=False)
        tmp.compute_normalization_stats(stats_path)

    train_ds = BatterySOHDataset(h5_path, indices=train_idx, normalize=True, stats_path=stats_path)
    test_ds = BatterySOHDataset(h5_path, indices=test_idx, normalize=True, stats_path=stats_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    return train_loader, test_loader

