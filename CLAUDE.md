# EV Battery SOH Prediction

NCSU NN course project. Trains 4 architectures on raw Li-ion battery cycling data (NASA Randomized Battery Usage Dataset, 28 batteries) and compares SOH prediction performance.

**Env:** `conda activate torch_env` (Python 3.10, torch+CUDA 11.8)

**Preprocess (run once from a notebook cell):**
```python
from src.preprocessing import build_sequence_dataset
build_sequence_dataset(
    data_dir="../11. Randomized Battery Usage Data Set",
    output_path="data/processed/sequences.h5",
)
```

---

## What a "sequence sample" means

One sample = one **RW step** (charge or discharge random-walk step), resampled to exactly 301 time points. Labelled with SOH from the next reference discharge.

- ~450K samples across 28 batteries
- Rest steps excluded (always 2 pts)
- Steps shorter than 10 samples dropped

## The 4 models

All accept `(batch, 301, 3)` → `(batch, 1)`.

- **MLP (`mlp.py`):** Extracts 20 summary stats in `forward()`. In the notebook, features are precomputed once on GPU and a `TensorDataset` over the `(N, 20)` cache is used — avoids recomputing 450K times per epoch. `batch_size=4096`.
- **1D-CNN (`cnn1d.py`):** Permutes to `(batch, 3, 301)` inside `forward()`. 4 conv blocks, AdaptiveAvgPool(1). `batch_size=2048`.
- **LSTM (`lstm.py`):** Bidirectional, 2 layers, hidden=128. Concatenates `h_n[-2]` and `h_n[-1]` → `(batch, 256)`. Requires `clip_grad_norm=1.0`. `batch_size=1024`.
- **Transformer (`transformer.py`):** Learnable positional embeddings (not sinusoidal). Mean pools over 301 positions. `batch_size=256` — attention matrix is `(batch, 301, 301)`.

## GPU preload strategy (CNN, LSTM, Transformer)

Full dataset loaded into GPU VRAM once. `TensorDataset` over GPU tensors — no `.to(DEVICE)` in training loop, no HDF5 reads during training. RTX 3070 Ti Laptop has 7.7 GB; dataset is ~1.52 GB leaving ~6 GB headroom.

## Train/test split

Battery-level only. Test: `['RW25','RW26','RW27','RW28','RW13','RW14']`. RW25–28 = 40°C OOD temperature. RW13–14 = skewed-low distribution OOD. RW21–24 (40°C skewed-low) stays in train so model sees at least one elevated-temp group.

## Loader (`src/loaders.py`)

`load_battery_mat` tries scipy.io first (MATLAB v5/v7), falls back to h5py for v7.3. h5py path: strings stored as uint16 char-code arrays or HDF5 object references — both handled. Numeric arrays are column-major, `.flatten()` always required. `abs()` on capacity integral — discharge current sign is inconsistent in the dataset.

## Dual MATLAB format — why scipy catches `NotImplementedError`

scipy raises `NotImplementedError` (not `ValueError`) for v7.3 HDF5 files. The fallback catches that specific exception only; other scipy errors propagate as `ValueError`.

## `ReduceLROnPlateau` `verbose` removed in PyTorch 2.x

Passing `verbose=True` raises `TypeError` in PyTorch ≥ 2.4. Log LR manually from `optimizer.param_groups[0]['lr']`.

## Training (`src/trainer.py`)

`Trainer(model, device, clip_grad=None).fit(train_loader, val_loader, ...)` wraps the standard loop (Adam + ReduceLROnPlateau patience=5 + early stopping). Returns `{train_loss, val_mae, val_rmse, lr}` history dict. Pass `clip_grad=1.0` for LSTM. Checkpoint saved to `save_dir/<ModelClassName>_best.pt`.

`train.py` was deleted — all training is notebook-only. `evaluate.py`, `preprocessing.py`, and `dataset.py` are also notebook-only (no CLI entry points).

## Known issues

**Silent battery skip:** `process_battery` returns `None` if <2 reference discharge cycles found. `build_sequence_dataset` skips silently — no count reported.

**NotebookEdit silently no-ops:** Matches cells by `"id"` field. If the stored JSON lacks top-level `"id"` keys (common after Jupyter saves), it finds nothing and makes no change. Fix: overwrite notebooks via `python3 -c "import json; ..."` script rather than `NotebookEdit`.
