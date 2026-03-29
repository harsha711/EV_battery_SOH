"""
CLI entry point — NASA Randomized Battery Usage Dataset preprocessing pipeline.

Processes all .mat files under DATA_DIR, extracts hand-crafted features from
random walk blocks, computes State-of-Health labels from reference discharge
cycles, and writes the result to a CSV.

Usage:
    python battery_processing.py --data_dir <path_to_dataset> [--output <csv_path>]

Dataset citation:
    B. Bole, C. Kulkarni, and M. Daigle, "Randomized Battery Usage Data Set",
    NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA

See battery_soh/ for the full implementation.
"""

import argparse

from battery_soh.dataset import build_full_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess NASA Randomized Battery Usage Dataset for SOH prediction"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the unzipped dataset directory containing .mat files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="battery_soh_dataset.csv",
        help="Output CSV path (default: battery_soh_dataset.csv)",
    )
    args = parser.parse_args()
    build_full_dataset(args.data_dir, args.output)
