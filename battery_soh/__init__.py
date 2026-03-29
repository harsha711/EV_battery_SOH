"""
battery_soh — preprocessing pipeline for the NASA Randomized Battery Usage Dataset.

Public API:

    from battery_soh import build_full_dataset          # full pipeline
    from battery_soh import build_battery_dataset       # single battery
    from battery_soh import load_battery_mat            # raw .mat loader
    from battery_soh import extract_rw_features         # feature extractor
    from battery_soh import compute_discharge_capacity  # capacity integration
    from battery_soh import compute_soh_percentage      # SOH computation
    from battery_soh import find_mat_files              # .mat file discovery
    from battery_soh import parse_group_metadata        # folder → metadata
    from battery_soh import GROUP_METADATA              # group metadata dict
    from battery_soh import RW_COMMENTS                 # step label constants
    from battery_soh import REF_DISCHARGE_COMMENT
"""

from battery_soh.constants import (
    GROUP_METADATA,
    MAT_STEP_FIELDS,
    REF_DISCHARGE_COMMENT,
    RW_COMMENTS,
    VOLTAGE_LOWER_BOUND,
    VOLTAGE_LOWER_HIT_THRESHOLD,
    VOLTAGE_UPPER_BOUND,
    VOLTAGE_UPPER_HIT_THRESHOLD,
)
from battery_soh.dataset import (
    build_battery_dataset,
    build_full_dataset,
    find_mat_files,
    parse_group_metadata,
)
from battery_soh.features import (
    compute_discharge_capacity,
    compute_soh_percentage,
    extract_rw_features,
)
from battery_soh.loaders import load_battery_mat

__version__ = "0.1.0"

__all__ = [
    # pipeline
    "build_full_dataset",
    "build_battery_dataset",
    "find_mat_files",
    "parse_group_metadata",
    # loaders
    "load_battery_mat",
    # features
    "extract_rw_features",
    "compute_discharge_capacity",
    "compute_soh_percentage",
    # constants
    "RW_COMMENTS",
    "REF_DISCHARGE_COMMENT",
    "GROUP_METADATA",
    "MAT_STEP_FIELDS",
    "VOLTAGE_LOWER_BOUND",
    "VOLTAGE_UPPER_BOUND",
    "VOLTAGE_LOWER_HIT_THRESHOLD",
    "VOLTAGE_UPPER_HIT_THRESHOLD",
]
