"""
preprocessing.py — Raw .mat → HDF5 sequence dataset.

Extracts per-step (301, 3) voltage/current/temperature sequences from
charge and discharge random-walk steps, labels each step with the SOH
from the next reference discharge, and streams the result to HDF5.

Output HDF5 schema:
    X            float32  (N, 301, 3)   [voltage, current, temperature]
    y            float32  (N,)          SOH in [0, 100]%
    battery_id   |S6      (N,)          e.g. b'RW10'
    step_type    |S32     (N,)          'discharge (random walk)' etc.
    block_index  int32    (N,)          0-indexed RW block within battery
"""

import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from src.loaders import load_battery_mat

REF_DISCHARGE_COMMENT: str = "reference discharge"


def find_mat_files(data_dir: str) -> list[tuple[str, str]]:
    """Recursively find all .mat files under data_dir.

    Returns sorted list of (mat_path, battery_id) tuples,
    where battery_id is the filename stem (e.g. 'RW9' from 'RW9.mat').
    """
    mat_paths = sorted(Path(data_dir).rglob("*.mat"))
    return [(str(p), p.stem) for p in mat_paths]


def compute_discharge_capacity(step: dict) -> float:
    """Compute discharge capacity (Ah) via trapezoidal integration of current."""
    if len(step["relativeTime"]) < 2:
        return 0.0
    return abs(np.trapz(step["current"], step["relativeTime"]) / 3600.0)


def compute_soh_percentage(capacity: float, initial_capacity: float) -> float:
    """Compute SOH as a percentage of the initial (fresh) capacity."""
    if initial_capacity == 0:
        return 0.0
    return (capacity / initial_capacity) * 100.0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIVE_STEP_TYPES: frozenset[str] = frozenset(
    {"discharge (random walk)", "charge (random walk)"}
)
N_POINTS: int = 301
MIN_STEP_LENGTH: int = 10
CHANNELS: list[str] = ["voltage", "current", "temperature"]


# ---------------------------------------------------------------------------
# Step resampling
# ---------------------------------------------------------------------------


def resample_step(step: dict, n_points: int = N_POINTS) -> np.ndarray:
    """
    Resample a single step's time series to exactly n_points.

    Extracts voltage, current, and temperature arrays from the step dict,
    normalizes the relativeTime axis to [0, 1], and linearly interpolates
    each channel to n_points samples.

    If the step already has exactly n_points samples, no interpolation is
    performed (fast path).

    Args:
        step: Step dict with keys 'voltage', 'current', 'temperature',
              'relativeTime' — as returned by load_battery_mat().
        n_points: Target sequence length (default 301).

    Returns:
        float32 array of shape (n_points, 3) with columns
        [voltage, current, temperature].
    """
    v = np.asarray(step["voltage"], dtype=np.float32)
    c = np.asarray(step["current"], dtype=np.float32)
    t = np.asarray(step["temperature"], dtype=np.float32)
    n = len(v)

    if n == n_points:
        return np.stack([v, c, t], axis=1)

    # Normalize time axis to [0, 1] for interpolation
    t_orig = np.asarray(step["relativeTime"], dtype=np.float64)
    if t_orig[-1] > t_orig[0]:
        t_norm = (t_orig - t_orig[0]) / (t_orig[-1] - t_orig[0])
    else:
        t_norm = np.linspace(0.0, 1.0, n)

    t_new = np.linspace(0.0, 1.0, n_points)

    v_r = np.interp(t_new, t_norm, v).astype(np.float32)
    c_r = np.interp(t_new, t_norm, c).astype(np.float32)
    t_r = np.interp(t_new, t_norm, t).astype(np.float32)

    return np.stack([v_r, c_r, t_r], axis=1)  # (n_points, 3)


# ---------------------------------------------------------------------------
# Per-battery processing
# ---------------------------------------------------------------------------


def process_battery(
    mat_path: str,
    battery_id: str,
    n_points: int = N_POINTS,
    min_step_length: int = MIN_STEP_LENGTH,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Process one battery .mat file into sequence arrays.

    Identifies reference discharge cycles, computes SOH at each cycle, then
    iterates each RW block and extracts per-step sequences. Only
    'charge (random walk)' and 'discharge (random walk)' steps are kept;
    rest steps are excluded. Steps shorter than min_step_length samples are
    dropped.

    SOH label for all steps in block i comes from reference discharge i+1.
    initial_capacity is always from reference discharge 0.

    Args:
        mat_path: Path to the .mat file.
        battery_id: Battery identifier string (e.g. 'RW10').
        n_points: Target sequence length after resampling.
        min_step_length: Minimum number of time points required to keep a step.

    Returns:
        Tuple of (X, y, battery_ids, block_indices), each as numpy arrays:
            X:             float32 (N, n_points, 3)
            y:             float32 (N,)  — SOH in %
            battery_ids:   bytes   (N,)  — all equal to battery_id encoded
            block_indices: int32   (N,)  — 0-indexed block number
        Returns (None, None, None, None) if fewer than 2 reference discharges.
    """
    print(f"  Processing {battery_id}...", end=" ", flush=True)

    steps = load_battery_mat(mat_path)

    ref_indices = [
        i for i, s in enumerate(steps) if s["comment"] == REF_DISCHARGE_COMMENT
    ]

    if len(ref_indices) < 2:
        print(f"SKIP (only {len(ref_indices)} reference discharge cycles)")
        return None, None, None, None

    capacities = [compute_discharge_capacity(steps[i]) for i in ref_indices]
    initial_capacity = capacities[0]

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    bid_list: list[bytes] = []
    blk_list: list[int] = []

    bid_bytes = battery_id.encode("ascii")
    soh_vals = [
        compute_soh_percentage(cap, initial_capacity) for cap in capacities
    ]

    for block_idx in range(len(ref_indices) - 1):
        start = ref_indices[block_idx]
        end = ref_indices[block_idx + 1]
        soh_label = soh_vals[block_idx + 1]

        for j in range(start + 1, end):
            step = steps[j]
            if step["comment"] not in ACTIVE_STEP_TYPES:
                continue
            if len(step["voltage"]) < min_step_length:
                continue

            seq = resample_step(step, n_points)  # (n_points, 3)
            X_list.append(seq)
            y_list.append(soh_label)
            bid_list.append(bid_bytes)
            blk_list.append(block_idx)

    if len(X_list) == 0:
        print("SKIP (no valid steps)")
        return None, None, None, None

    X = np.stack(X_list, axis=0)                          # (N, 301, 3)
    y = np.array(y_list, dtype=np.float32)                # (N,)
    battery_ids = np.array(bid_list, dtype="S6")          # (N,)
    block_indices = np.array(blk_list, dtype=np.int32)    # (N,)

    print(f"{len(X_list)} steps, {len(ref_indices)-1} blocks, SOH {y.min():.1f}–{y.max():.1f}%")
    return X, y, battery_ids, block_indices


# ---------------------------------------------------------------------------
# HDF5 streaming writer
# ---------------------------------------------------------------------------


def build_sequence_dataset(
    data_dir: str,
    output_path: str,
    n_points: int = N_POINTS,
    min_step_length: int = MIN_STEP_LENGTH,
) -> None:
    """
    Process all battery .mat files and write sequences to an HDF5 file.

    Streams results one battery at a time to keep peak RAM usage low
    (one battery's data in memory at once). Resizes HDF5 datasets
    incrementally as each battery is processed.

    HDF5 output layout:
        X            float32  (N, n_points, 3)  chunks=(1024, n_points, 3)  gzip/4
        y            float32  (N,)
        battery_id   |S6      (N,)
        step_type    |S32     (N,)    placeholder (all zeros; field reserved)
        block_index  int32    (N,)

    Args:
        data_dir: Root directory containing .mat files (searched recursively).
        output_path: Path for the output HDF5 file.
        n_points: Sequence length (default 301).
        min_step_length: Minimum step length to include (default 10).
    """
    mat_files = find_mat_files(data_dir)
    if not mat_files:
        print(f"ERROR: No .mat files found in '{data_dir}'")
        return

    print(f"Found {len(mat_files)} battery files in '{data_dir}'")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total_written = 0
    batteries_processed = 0

    with h5py.File(output_path, "w") as f:
        # Create resizable datasets
        f.create_dataset(
            "X",
            shape=(0, n_points, 3),
            maxshape=(None, n_points, 3),
            dtype=np.float32,
            chunks=(1024, n_points, 3),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "y",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=(1024,),
        )
        f.create_dataset(
            "battery_id",
            shape=(0,),
            maxshape=(None,),
            dtype="S6",
            chunks=(1024,),
        )
        f.create_dataset(
            "step_type",
            shape=(0,),
            maxshape=(None,),
            dtype="S32",
            chunks=(1024,),
        )
        f.create_dataset(
            "block_index",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(1024,),
        )

        for mat_path, battery_id in mat_files:
            X, y, bids, blks = process_battery(
                mat_path, battery_id, n_points, min_step_length
            )
            if X is None:
                continue

            n_new = len(X)
            offset = total_written

            # Resize all datasets
            f["X"].resize((offset + n_new, n_points, 3))
            f["y"].resize((offset + n_new,))
            f["battery_id"].resize((offset + n_new,))
            f["step_type"].resize((offset + n_new,))
            f["block_index"].resize((offset + n_new,))

            # Write
            f["X"][offset : offset + n_new] = X
            f["y"][offset : offset + n_new] = y
            f["battery_id"][offset : offset + n_new] = bids
            f["block_index"][offset : offset + n_new] = blks
            # step_type left as zeros (empty bytes) — reserved for future use

            total_written += n_new
            batteries_processed += 1

        # Write metadata attributes
        f.attrs["n_samples"] = total_written
        f.attrs["seq_len"] = n_points
        f.attrs["n_channels"] = 3
        f.attrs["channels"] = ["voltage", "current", "temperature"]
        f.attrs["min_step_length"] = min_step_length
        f.attrs["soh_unit"] = "percent"
        f.attrs["created"] = datetime.now().isoformat()

    print(f"\nDone. {batteries_processed} batteries, {total_written:,} total steps.")
    print(f"Output: {output_path}")
    print(f"X shape: ({total_written}, {n_points}, 3)  "
          f"≈ {total_written * n_points * 3 * 4 / 1e9:.2f} GB uncompressed")
