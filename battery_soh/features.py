"""
Feature and SOH computation functions for the NASA Battery dataset.

All functions here are pure (no I/O, no side effects). They transform raw
step dicts (as returned by loaders.load_battery_mat) into numeric features
and State-of-Health (SOH) values.
"""

import numpy as np

from battery_soh.constants import VOLTAGE_LOWER_HIT_THRESHOLD, VOLTAGE_UPPER_HIT_THRESHOLD


# ---------------------------------------------------------------------------
# SOH computation
# ---------------------------------------------------------------------------


def compute_discharge_capacity(step: dict) -> float:
    """
    Compute discharge capacity (Ah) from a reference discharge step.

    Uses trapezoidal integration of current over relative time:
        capacity = integral(current, time) / 3600

    abs() is applied because discharge current may be reported as positive
    or negative depending on sign convention, and capacity is always positive.

    Args:
        step: Step dict containing 'current' and 'relativeTime' arrays.

    Returns:
        Discharge capacity in ampere-hours (Ah).
    """
    if len(step["relativeTime"]) < 2:
        return 0.0
    capacity = np.trapz(step["current"], step["relativeTime"]) / 3600.0
    return abs(capacity)


def compute_soh_percentage(capacity: float, initial_capacity: float) -> float:
    """
    Compute State of Health as a percentage of the initial (fresh) capacity.

    Args:
        capacity: Current discharge capacity (Ah).
        initial_capacity: First measured discharge capacity of this battery (Ah).

    Returns:
        SOH in percent [0, 100]. Returns 0.0 if initial_capacity is zero.
    """
    if initial_capacity == 0:
        return 0.0
    return (capacity / initial_capacity) * 100.0


# ---------------------------------------------------------------------------
# Hand-crafted features from random walk blocks (MLP baseline)
# ---------------------------------------------------------------------------


def extract_rw_features(rw_steps: list[dict]) -> dict | None:
    """
    Extract summary statistics from a block of random walk (RW) steps.

    These features deliberately ignore temporal ordering — they are designed
    for the MLP baseline. Sequence models (LSTM, Transformer) should use
    raw time-series windows instead.

    Features extracted:
        Voltage    : mean, std, min, max, range
        Current    : mean, std, min, max, range
        Temperature: mean, std, min, max, range
        Voltage boundary hits:
            voltage_lower_hits — samples where voltage <= VOLTAGE_LOWER_HIT_THRESHOLD
            voltage_upper_hits — samples where voltage >= VOLTAGE_UPPER_HIT_THRESHOLD
            (boundary contact frequency correlates with degradation)
        n_rw_steps         — number of RW steps in this block
        avg_step_duration  — mean step duration in seconds
        energy_throughput  — total |power| integrated over time (Wh)

    Args:
        rw_steps: List of step dicts from a single between-reference block.

    Returns:
        Dict of feature name → scalar value, or None if rw_steps is empty.
    """
    if len(rw_steps) == 0:
        return None

    all_voltage = np.concatenate([s["voltage"] for s in rw_steps])
    all_current = np.concatenate([s["current"] for s in rw_steps])
    all_temperature = np.concatenate([s["temperature"] for s in rw_steps])

    features: dict = {}

    # Voltage statistics
    features["voltage_mean"] = np.mean(all_voltage)
    features["voltage_std"] = np.std(all_voltage)
    features["voltage_min"] = np.min(all_voltage)
    features["voltage_max"] = np.max(all_voltage)
    features["voltage_range"] = features["voltage_max"] - features["voltage_min"]

    # Current statistics
    features["current_mean"] = np.mean(all_current)
    features["current_std"] = np.std(all_current)
    features["current_min"] = np.min(all_current)
    features["current_max"] = np.max(all_current)
    features["current_range"] = features["current_max"] - features["current_min"]

    # Temperature statistics
    features["temp_mean"] = np.mean(all_temperature)
    features["temp_std"] = np.std(all_temperature)
    features["temp_min"] = np.min(all_temperature)
    features["temp_max"] = np.max(all_temperature)
    features["temp_range"] = features["temp_max"] - features["temp_min"]

    # Voltage boundary hits (degradation indicator)
    features["voltage_lower_hits"] = int(np.sum(all_voltage <= VOLTAGE_LOWER_HIT_THRESHOLD))
    features["voltage_upper_hits"] = int(np.sum(all_voltage >= VOLTAGE_UPPER_HIT_THRESHOLD))

    # Step count
    features["n_rw_steps"] = len(rw_steps)

    # Average step duration (seconds)
    durations = [
        s["relativeTime"][-1] - s["relativeTime"][0]
        for s in rw_steps
        if len(s["relativeTime"]) > 1
    ]
    features["avg_step_duration"] = np.mean(durations) if durations else 0.0

    # Energy throughput (Wh): integral of |power| = |current * voltage| over time
    total_energy = 0.0
    for s in rw_steps:
        if len(s["relativeTime"]) > 1:
            power = np.abs(s["current"] * s["voltage"])
            total_energy += np.trapz(power, s["relativeTime"]) / 3600.0
    features["energy_throughput"] = total_energy

    return features
