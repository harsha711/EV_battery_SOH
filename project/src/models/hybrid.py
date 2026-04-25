"""
hybrid.py — Hybrid CNN + GRU model for battery SOH prediction.

Combines the CNN's strong per-window distribution-summarisation (which won
the single-step k-fold) with a GRU's ability to model degradation trajectory
across blocks (which lifted LSTM from 10.88% → 8.24% on the cycle-history
task). The CNN encoder is shared across all (W × K) steps and produces a
128-dim embedding; embeddings are mean-pooled within each block to form a
block embedding, then a small bidirectional GRU integrates the W block
embeddings into a final state for regression.

Why GRU instead of LSTM: the sequence is only 10 blocks long, so a separate
cell state buys nothing. GRU has ~25% fewer parameters at the same hidden
size, trains faster, and is empirically better on short sequences with small
training sets — both of which apply here (~300 cycle-history training
sequences).

Architecture:
    Input: (batch, W, K, 301, 3)
        flatten to (batch * W * K, 301, 3)
    CNN encoder (shared, derived from cnn1d.py):
        Conv1d(3 → 32, k=7) → BN → ReLU → MaxPool(2)
        Conv1d(32 → 64, k=5) → BN → ReLU → MaxPool(2)
        Conv1d(64 → 128, k=3) → BN → ReLU → MaxPool(2)
        Conv1d(128 → 128, k=3) → BN → ReLU → AdaptiveAvgPool(1)
        Flatten → (batch * W * K, 128)
    Reshape to (batch, W, K, 128) → mean over K → (batch, W, 128)
    Bidirectional GRU (1 layer, hidden=64) over W blocks
        h_n: (2, batch, 64) → concat → (batch, 128)
    Head: Linear(128 → 64) → ReLU → Dropout(0.3) → Linear(64 → 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class CNNStepEncoder(nn.Module):
    """Encodes a single (301, 3) step into a 128-dim embedding.

    Same architecture as cnn1d.py but exposes the pooled embedding instead of
    a scalar. Operates on (batch, 301, 3) and returns (batch, 128).
    """

    EMBED_DIM: int = 128

    def __init__(self, in_channels: int = 3, base_filters: int = 32) -> None:
        super().__init__()
        f = base_filters
        self.conv1 = _conv_block(in_channels, f, kernel_size=7)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = _conv_block(f, f * 2, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = _conv_block(f * 2, f * 4, kernel_size=3)
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = _conv_block(f * 4, f * 4, kernel_size=3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 301, 3) → (batch, 3, 301)
        x = x.permute(0, 2, 1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.global_pool(self.conv4(x))   # (batch, 128, 1)
        return x.squeeze(-1)                   # (batch, 128)


class SOHHybrid(nn.Module):
    """Hybrid CNN encoder + GRU aggregator.

    Args:
        window:        Number of blocks per sample (W).
        k_steps:       Number of steps per block (K).
        cnn_filters:   Base filters in CNN encoder (default 32 → 128 embed).
        gru_hidden:    GRU hidden size per direction (default 64).
        gru_layers:    Number of stacked GRU layers (default 1).
        dropout:       Dropout in head and (if layers > 1) between GRU layers.
    """

    def __init__(
        self,
        window: int = 10,
        k_steps: int = 32,
        cnn_filters: int = 32,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.window = window
        self.k_steps = k_steps

        self.encoder = CNNStepEncoder(in_channels=3, base_filters=cnn_filters)
        embed_dim = self.encoder.EMBED_DIM   # 128

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        gru_out_dim = gru_hidden * 2
        self.head = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float32 tensor (batch, W, K, 301, 3)
        Returns:
            float32 tensor (batch, 1) — predicted SOH in [0, 1]
        """
        B, W, K, L, C = x.shape

        # Flatten W and K into the batch dimension for the shared CNN encoder.
        x = x.reshape(B * W * K, L, C)
        embeds = self.encoder(x)                          # (B*W*K, 128)
        embeds = embeds.view(B, W, K, -1)                 # (B, W, K, 128)

        # Mean-pool within each block to form block embedding.
        block_embeds = embeds.mean(dim=2)                 # (B, W, 128)

        # GRU over W blocks.
        # h_n: (num_layers * num_directions, B, hidden) = (2, B, 64) for 1-layer biGRU
        _, h_n = self.gru(block_embeds)
        h_fwd = h_n[-2]                                   # (B, 64)
        h_bwd = h_n[-1]                                   # (B, 64)
        h = torch.cat([h_fwd, h_bwd], dim=1)              # (B, 128)

        return self.head(h)                                # (B, 1)


if __name__ == "__main__":
    print("=== SOHHybrid shape check ===")
    model = SOHHybrid(window=10, k_steps=8)   # smaller K for quick CPU check
    model.eval()

    dummy = torch.randn(2, 10, 8, 301, 3)
    print(f"Input: {tuple(dummy.shape)}")
    with torch.no_grad():
        out = model(dummy)
    print(f"Output: {tuple(out.shape)}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print("OK")
