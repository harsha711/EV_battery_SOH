"""
loaders.py — MATLAB .mat file loaders for the NASA Randomized Battery Usage Dataset.

Supports both MATLAB v5/v7 (via scipy.io) and MATLAB v7.3 HDF5 format
(via h5py). The public entry point is load_battery_mat(), which tries
scipy first and falls back to h5py automatically.
"""

import numpy as np
import scipy.io as sio

_MAT_STEP_FIELDS: tuple[str, ...] = (
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
# scipy path (MATLAB v5 / v7)
# ---------------------------------------------------------------------------


def _load_mat_scipy(mat_path: str) -> list[dict]:
    """Load a .mat file using scipy.io (MATLAB v5/v7 format)."""
    mat = sio.loadmat(mat_path)
    data = mat["data"]
    steps_array = data["step"][0, 0]  # unwrap MATLAB struct nesting
    n_steps = steps_array.shape[1]

    steps = []
    for i in range(n_steps):
        step = steps_array[0, i]
        record = {
            "step_index": i,
            "comment": str(step["comment"][0]).strip(),
            "type": str(step["type"][0]).strip(),
            "voltage": step["voltage"].flatten().astype(np.float64),
            "current": step["current"].flatten().astype(np.float64),
            "temperature": step["temperature"].flatten().astype(np.float64),
            "relativeTime": step["relativeTime"].flatten().astype(np.float64),
            "time": step["time"].flatten().astype(np.float64),
            "date": str(step["date"][0]).strip(),
        }
        steps.append(record)

    return steps


# ---------------------------------------------------------------------------
# h5py path (MATLAB v7.3 / HDF5 format)
# ---------------------------------------------------------------------------


def _decode_hdf5_string(dataset) -> str:
    """
    Decode a string from an HDF5 MATLAB v7.3 dataset.

    MATLAB v7.3 stores character arrays as arrays of uint16 code points.
    Handles both that case and pre-decoded byte/str values.
    """
    raw = dataset[...]
    if raw.dtype.kind in ("U", "S"):
        return str(raw.flat[0]).strip()
    # uint16 char array — join code points
    return "".join(chr(int(c)) for c in raw.flatten()).strip()


def _load_mat_h5py(mat_path: str) -> list[dict]:
    """
    Load a .mat file using h5py (MATLAB v7.3 / HDF5 format).

    HDF5 MATLAB files differ from scipy-loaded files in two key ways:
      1. Strings are stored as uint16 char code arrays.
      2. Numeric arrays may be transposed (column-major storage).
    """
    import h5py

    steps = []

    with h5py.File(mat_path, "r") as f:
        try:
            step_refs = f["data"]["step"][0]  # array of HDF5 object references
        except KeyError as exc:
            raise ValueError(
                f"Unexpected HDF5 structure in '{mat_path}': missing 'data/step'. "
                f"Original error: {exc}"
            ) from exc

        for i, ref in enumerate(step_refs):
            step_group = f[ref]
            record: dict = {"step_index": i}

            for field in _MAT_STEP_FIELDS:
                if field not in step_group:
                    continue
                ds = step_group[field]

                if field in ("comment", "type", "date"):
                    # String fields stored as uint16 char arrays or HDF5 references
                    raw = ds[...]
                    if raw.dtype == object:
                        # Array of references — each element points to a char array
                        chars = []
                        for char_ref in raw.flatten():
                            chars.append(chr(int(f[char_ref][0])))
                        record[field] = "".join(chars).strip()
                    else:
                        record[field] = _decode_hdf5_string(ds)
                else:
                    # Numeric array: flatten and cast; HDF5 MATLAB stores
                    # column-major so a 1-D signal is typically shape (1, N) or (N,).
                    arr = np.array(ds[...]).flatten().astype(np.float64)
                    record[field] = arr

            steps.append(record)

    return steps


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def load_battery_mat(mat_path: str) -> list[dict]:
    """
    Load a single .mat file and return all cycling steps as a list of dicts.

    Each dict contains:
        step_index  : int
        comment     : str   (step label from the dataset README)
        type        : str   ('C', 'D', or 'R')
        voltage     : np.ndarray[float64]
        current     : np.ndarray[float64]
        temperature : np.ndarray[float64]
        relativeTime: np.ndarray[float64]  (seconds from step start)
        time        : np.ndarray[float64]  (seconds from experiment start)
        date        : str

    Tries scipy.io first (MATLAB v5/v7); falls back to h5py for
    MATLAB v7.3 HDF5 files. Raises ValueError if neither loader succeeds.

    Args:
        mat_path: Absolute or relative path to the .mat file.

    Returns:
        List of step dicts, one per cycling step.

    Raises:
        FileNotFoundError: If mat_path does not exist.
        ValueError: If the file format is unsupported or the data structure
                    does not match the expected NASA dataset layout.
    """
    import os

    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"No such file: '{mat_path}'")

    try:
        return _load_mat_scipy(mat_path)
    except NotImplementedError:
        # scipy raises NotImplementedError for MATLAB v7.3 (HDF5) files
        pass
    except Exception as exc:
        raise ValueError(
            f"scipy.io failed to load '{mat_path}': {exc}"
        ) from exc

    try:
        return _load_mat_h5py(mat_path)
    except Exception as exc:
        raise ValueError(
            f"h5py failed to load '{mat_path}' as MATLAB v7.3 HDF5: {exc}"
        ) from exc
