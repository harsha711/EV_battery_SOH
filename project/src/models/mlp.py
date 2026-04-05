"""
mlp.py — MLP baseline model for battery SOH prediction.

The MLP is the 'sanity check' model. It does not learn temporal patterns
from the raw sequence — instead it hand-crafts 20 summary statistics in the
forward pass, then passes them through a fully connected network.

Despite receiving the same (batch, 301, 3) input as all other models, its
predictions are based only on aggregate statistics (mean, min, max, std per
channel plus a few voltage-specific features), not on the ordering of data
points. If the deep sequence models can't beat the MLP, it suggests they are
not learning anything beyond what summary statistics already capture.

Feature set (20 total):
    Voltage (3 channels × 4 stats = 12):
        voltage: mean, std, min, max
        current: mean, std, min, max
        temperature: mean, std, min, max
    Voltage extras (4):
        voltage_range (max-min)
        voltage_start (value at timestep 0)
        voltage_end   (value at last timestep)
        voltage_delta (end - start)
    Current extras (2):
        current_range (max-min)
        current_abs_mean (mean of absolute values)
    Temperature extras (2):
        temp_range (max-min)
        temp_delta (last - first)

Standalone usage:
    python src/models/mlp.py   (runs a forward-pass shape check)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature extraction (pure torch, differentiable)
# ---------------------------------------------------------------------------


def extract_features(x: torch.Tensor) -> torch.Tensor:
    """
    Extract 20 hand-crafted summary statistics from raw sequence data.

    All operations are differentiable and run on the same device as x.
    No numpy is used — the entire feature extraction is part of the
    PyTorch compute graph (though gradients w.r.t. these stats are not
    meaningful for training; the feature extraction is fixed, not learned).

    Args:
        x: float32 tensor of shape (batch, seq_len, 3)
           Channels: [voltage, current, temperature]

    Returns:
        float32 tensor of shape (batch, 20)
    """
    v = x[:, :, 0]  # (batch, seq_len)
    c = x[:, :, 1]
    t = x[:, :, 2]

    # Per-channel stats (12 features)
    v_mean = v.mean(dim=1, keepdim=True)
    v_std  = v.std(dim=1, keepdim=True)
    v_min  = v.min(dim=1).values.unsqueeze(1)
    v_max  = v.max(dim=1).values.unsqueeze(1)

    c_mean = c.mean(dim=1, keepdim=True)
    c_std  = c.std(dim=1, keepdim=True)
    c_min  = c.min(dim=1).values.unsqueeze(1)
    c_max  = c.max(dim=1).values.unsqueeze(1)

    t_mean = t.mean(dim=1, keepdim=True)
    t_std  = t.std(dim=1, keepdim=True)
    t_min  = t.min(dim=1).values.unsqueeze(1)
    t_max  = t.max(dim=1).values.unsqueeze(1)

    # Voltage extras (4 features)
    v_range = v_max - v_min
    v_start = v[:, 0].unsqueeze(1)
    v_end   = v[:, -1].unsqueeze(1)
    v_delta = v_end - v_start

    # Current extras (2 features)
    c_range    = c_max - c_min
    c_abs_mean = c.abs().mean(dim=1, keepdim=True)

    # Temperature extras (2 features)
    t_range = t_max - t_min
    t_delta = t[:, -1].unsqueeze(1) - t[:, 0].unsqueeze(1)

    features = torch.cat(
        [
            v_mean, v_std, v_min, v_max,   # 4
            c_mean, c_std, c_min, c_max,   # 4
            t_mean, t_std, t_min, t_max,   # 4
            v_range, v_start, v_end, v_delta,  # 4
            c_range, c_abs_mean,               # 2
            t_range, t_delta,                  # 2
        ],
        dim=1,
    )  # (batch, 20)

    return features


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------


class SOHMLP(nn.Module):
    """
    MLP baseline for battery SOH prediction.

    Internally extracts 20 summary statistics from the raw (301, 3) input
    in forward(), then passes them through a fully connected network.

    Architecture:
        extract_features → (batch, 20)
        Linear(20 → 128) → BatchNorm1d → ReLU → Dropout(0.2)
        Linear(128 → 64) → BatchNorm1d → ReLU → Dropout(0.2)
        Linear(64 → 1)

    Args:
        input_dim: Number of hand-crafted features (default 20).
        dropout: Dropout probability applied after each hidden layer.
    """

    N_FEATURES: int = 20

    def __init__(self, input_dim: int = 20, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: float32 tensor (batch, 301, 3)

        Returns:
            float32 tensor (batch, 1) — predicted SOH in [0, 1]
        """
        features = extract_features(x)   # (batch, 20)
        return self.net(features)         # (batch, 1)


# ---------------------------------------------------------------------------
# Standalone shape check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== SOHMLP shape check ===")
    model = SOHMLP()
    model.eval()

    batch = torch.randn(4, 301, 3)
    print(f"Input:  {tuple(batch.shape)}")

    feats = extract_features(batch)
    print(f"Features after extract_features: {tuple(feats.shape)}")

    with torch.no_grad():
        out = model(batch)
    print(f"Output: {tuple(out.shape)}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print("OK")
