"""
Dataset construction for the NASA Randomized Battery Usage Dataset.

Orchestrates file discovery, group metadata parsing, per-battery feature
extraction, and final DataFrame assembly.
"""

import warnings
from pathlib import Path

import pandas as pd

from battery_soh.constants import GROUP_METADATA, REF_DISCHARGE_COMMENT, RW_COMMENTS
from battery_soh.features import (
    compute_discharge_capacity,
    compute_soh_percentage,
    extract_rw_features,
)
from battery_soh.loaders import load_battery_mat


# ---------------------------------------------------------------------------
# Group metadata
# ---------------------------------------------------------------------------


def parse_group_metadata(mat_path: str) -> dict[str, str]:
    """
    Determine dataset group metadata from the path of a .mat file.

    Walks the path components and returns the entry from GROUP_METADATA
    whose key matches an inner folder name exactly. This relies on the
    dataset's fixed folder structure:

        <data_root>/<numbered_outer>/<GROUP_METADATA_KEY>/data/Matlab/RW*.mat

    Args:
        mat_path: Path to the .mat file (absolute or relative).

    Returns:
        Dict with keys 'distribution', 'temperature', 'charge_mode'.
        Falls back to ``{"distribution": "unknown", ...}`` with a warning
        if no match is found.
    """
    for part in Path(mat_path).parts:
        if part in GROUP_METADATA:
            return GROUP_METADATA[part].copy()

    warnings.warn(
        f"Could not determine group metadata for path: '{mat_path}'. "
        f"Known group folder names: {list(GROUP_METADATA.keys())}",
        stacklevel=2,
    )
    return {"distribution": "unknown", "temperature": "unknown", "charge_mode": "unknown"}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_mat_files(data_dir: str) -> list[tuple[str, str]]:
    """
    Recursively find all .mat files under data_dir.

    Args:
        data_dir: Root directory to search.

    Returns:
        Sorted list of (mat_path, battery_id) tuples, where battery_id is
        the filename stem (e.g. "RW9" from "RW9.mat").
    """
    mat_paths = sorted(Path(data_dir).rglob("*.mat"))
    return [(str(p), p.stem) for p in mat_paths]


# ---------------------------------------------------------------------------
# Per-battery dataset construction
# ---------------------------------------------------------------------------


def build_battery_dataset(mat_path: str, battery_id: str) -> pd.DataFrame:
    """
    Build a complete feature-label dataset from one battery .mat file.

    For each consecutive pair of reference discharge cycles, the random walk
    steps between them are used to extract features, and the SOH is computed
    from the capacity at the later reference discharge:

        RW block [i → i+1]  →  features
        reference discharge [i+1]  →  SOH label

    Metadata columns (distribution, temperature, charge_mode) are derived
    from the folder structure of mat_path.

    Args:
        mat_path: Path to the battery's .mat file.
        battery_id: Battery identifier string (e.g. "RW9").

    Returns:
        DataFrame with one row per RW block. Empty DataFrame if fewer than
        two reference discharge cycles are present.
    """
    print(f"\n{'='*60}")
    print(f"Processing battery: {battery_id}")
    print(f"File: {mat_path}")
    print(f"{'='*60}")

    steps = load_battery_mat(mat_path)
    print(f"  Total steps loaded: {len(steps)}")

    ref_discharge_indices = [
        i for i, s in enumerate(steps) if s["comment"] == REF_DISCHARGE_COMMENT
    ]
    print(f"  Reference discharge cycles found: {len(ref_discharge_indices)}")

    if len(ref_discharge_indices) < 2:
        print(f"  WARNING: Not enough reference cycles for {battery_id}, skipping.")
        return pd.DataFrame()

    capacities = [compute_discharge_capacity(steps[idx]) for idx in ref_discharge_indices]
    initial_capacity = capacities[0]
    final_capacity = capacities[-1]

    print(f"  Initial capacity: {initial_capacity:.4f} Ah")
    print(f"  Final capacity:   {final_capacity:.4f} Ah")
    capacity_fade_pct = (
        (initial_capacity - final_capacity) / initial_capacity * 100
        if initial_capacity > 0
        else 0.0
    )
    print(f"  Capacity fade:    {capacity_fade_pct:.1f}%")

    group_meta = parse_group_metadata(mat_path)

    dataset = []
    for i in range(len(ref_discharge_indices) - 1):
        start_idx = ref_discharge_indices[i]
        end_idx = ref_discharge_indices[i + 1]

        rw_steps = [
            steps[j]
            for j in range(start_idx + 1, end_idx)
            if steps[j]["comment"] in RW_COMMENTS
        ]

        if len(rw_steps) == 0:
            continue

        features = extract_rw_features(rw_steps)
        if features is None:
            continue

        features["soh"] = compute_soh_percentage(capacities[i + 1], initial_capacity)
        features["capacity_ah"] = capacities[i + 1]
        features["initial_capacity_ah"] = initial_capacity
        features["battery_id"] = battery_id
        features["cycle_index"] = i + 1
        features["distribution"] = group_meta["distribution"]
        features["temperature"] = group_meta["temperature"]
        features["charge_mode"] = group_meta["charge_mode"]

        dataset.append(features)

    df = pd.DataFrame(dataset)
    print(f"  Dataset samples built: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Full multi-battery dataset construction
# ---------------------------------------------------------------------------


def build_full_dataset(
    data_dir: str, output_path: str | None = None
) -> pd.DataFrame:
    """
    Process all battery .mat files in data_dir and combine into one DataFrame.

    Args:
        data_dir: Root directory containing the .mat files (searched recursively).
        output_path: Optional path to save the resulting CSV. If None, no file
                     is written.

    Returns:
        Combined DataFrame with all batteries. Empty DataFrame if no data
        could be extracted.
    """
    mat_files = find_mat_files(data_dir)

    if len(mat_files) == 0:
        print(f"ERROR: No .mat files found in '{data_dir}'")
        return pd.DataFrame()

    print(f"Found {len(mat_files)} battery files:")
    for path, bid in mat_files:
        print(f"  {bid}: {path}")

    all_dfs = []
    for mat_path, battery_id in mat_files:
        df = build_battery_dataset(mat_path, battery_id)
        if len(df) > 0:
            all_dfs.append(df)

    if len(all_dfs) == 0:
        print("ERROR: No data extracted from any battery.")
        return pd.DataFrame()

    full_df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(full_df)} total samples from {len(all_dfs)} batteries")
    print(f"{'='*60}")
    print(f"\nSOH range: {full_df['soh'].min():.1f}% — {full_df['soh'].max():.1f}%")
    print(f"Batteries: {sorted(full_df['battery_id'].unique().tolist())}")

    print("\nSamples per battery (SOH min/max):")
    print(
        full_df.groupby("battery_id")["soh"]
        .agg(["count", "min", "max"])
        .to_string()
    )

    print("\nSamples by group (distribution × temperature):")
    print(
        full_df.groupby(["distribution", "temperature"])["soh"]
        .agg(["count", "min", "max"])
        .to_string()
    )

    if output_path:
        full_df.to_csv(output_path, index=False)
        print(f"\nDataset saved to: {output_path}")

    return full_df
