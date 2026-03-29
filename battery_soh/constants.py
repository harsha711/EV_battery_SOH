"""
Constants for the NASA Randomized Battery Usage Dataset pipeline.

All symbolic constants and dataset group metadata are defined here.
No other module in battery_soh should hardcode these values.
"""

# ---------------------------------------------------------------------------
# Step comment labels (from the dataset README)
# ---------------------------------------------------------------------------

RW_COMMENTS: frozenset[str] = frozenset(
    {
        "discharge (random walk)",
        "charge (random walk)",
        "rest (random walk)",
    }
)

REF_DISCHARGE_COMMENT: str = "reference discharge"

# ---------------------------------------------------------------------------
# MATLAB struct field names expected in each step
# ---------------------------------------------------------------------------

MAT_STEP_FIELDS: tuple[str, ...] = (
    "comment",
    "type",
    "relativeTime",
    "time",
    "voltage",
    "current",
    "temperature",
    "date",
)

# ---------------------------------------------------------------------------
# Voltage boundary thresholds for degradation feature extraction
#
# VOLTAGE_LOWER_BOUND / VOLTAGE_UPPER_BOUND are the nominal cell limits.
# The hit thresholds are offset inward by ~0.05 V to tolerate sensor noise
# while still detecting boundary contact events.
# ---------------------------------------------------------------------------

VOLTAGE_LOWER_BOUND: float = 3.2
VOLTAGE_UPPER_BOUND: float = 4.2

VOLTAGE_LOWER_HIT_THRESHOLD: float = 3.25
VOLTAGE_UPPER_HIT_THRESHOLD: float = 4.15

# ---------------------------------------------------------------------------
# Dataset group metadata
#
# Keys are the exact inner subfolder names as they appear on disk.
# The folder structure is:
#   <data_root>/<numbered_outer>/<GROUP_METADATA_KEY>/data/Matlab/RW*.mat
#
# Batteries per group:
#   uniform / charge_discharge  / room_temp  → RW9-RW12
#   uniform / discharge_only    / room_temp  → RW3-RW6
#   uniform / variable_charge   / room_temp  → RW1-RW2, RW7-RW8
#   skewed_high / standard      / room_temp  → RW17-RW20
#   skewed_high / standard      / 40c        → RW25-RW28
#   skewed_low  / standard      / room_temp  → RW13-RW16
#   skewed_low  / standard      / 40c        → RW21-RW24
# ---------------------------------------------------------------------------

GROUP_METADATA: dict[str, dict[str, str]] = {
    "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post": {
        "distribution": "uniform",
        "temperature": "room_temp",
        "charge_mode": "charge_discharge",
    },
    "Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post": {
        "distribution": "uniform",
        "temperature": "room_temp",
        "charge_mode": "discharge_only",
    },
    "Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post": {
        "distribution": "uniform",
        "temperature": "room_temp",
        "charge_mode": "variable_charge",
    },
    "RW_Skewed_High_40C_DataSet_2Post": {
        "distribution": "skewed_high",
        "temperature": "40c",
        "charge_mode": "standard",
    },
    "RW_Skewed_High_Room_Temp_DataSet_2Post": {
        "distribution": "skewed_high",
        "temperature": "room_temp",
        "charge_mode": "standard",
    },
    "RW_Skewed_Low_40C_DataSet_2Post": {
        "distribution": "skewed_low",
        "temperature": "40c",
        "charge_mode": "standard",
    },
    "RW_Skewed_Low_Room_Temp_DataSet_2Post": {
        "distribution": "skewed_low",
        "temperature": "room_temp",
        "charge_mode": "standard",
    },
}
