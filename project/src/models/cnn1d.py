"""
cnn1d.py — 1D-CNN model for battery SOH prediction.

The first model that actually learns from the raw sequence rather than
hand-crafted features. Treats the 3 sensor channels as feature maps and
applies successive 1D convolutions to capture local temporal patterns
(e.g. voltage knee, current plateaus).

Input is permuted from (batch, 301, 3) to (batch, 3, 301) inside forward()
so the external API matches all other models.

Architecture:
    Conv block 1: Conv1d(3 → 32, k=7, p=3) → BN → ReLU → MaxPool(2)   [301 → 150]
    Conv block 2: Conv1d(32 → 64, k=5, p=2) → BN → ReLU → MaxPool(2)  [150 → 75]
    Conv block 3: Conv1d(64 → 128, k=3, p=1) → BN → ReLU → MaxPool(2) [75 → 37]
    Conv block 4: Conv1d(128 → 128, k=3, p=1) → BN → ReLU → AdaptiveAvgPool(1) [→ 1]
    Flatten → Linear(128 → 64) → ReLU → Dropout(0.3) → Linear(64 → 1)

Standalone usage:
    python src/models/cnn1d.py   (runs a forward-pass shape check)
"""

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
    """
    Single conv block: Conv1d → BatchNorm1d → ReLU.

    Padding is computed to keep the sequence length unchanged (same padding)
    for odd kernel sizes.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        kernel_size: Convolution kernel size (must be odd for symmetric padding).

    Returns:
        Sequential module with Conv1d, BatchNorm1d, ReLU.
    """
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class SOHCNN1D(nn.Module):
    """
    1D Convolutional Network for battery SOH prediction.

    Processes the raw voltage/current/temperature sequence directly without
    any hand-crafted features. Learns hierarchical temporal patterns through
    4 convolutional blocks with progressively larger receptive fields.

    Args:
        in_channels: Number of input channels (default 3: V, I, T).
        base_filters: Number of filters in the first conv layer (default 32).
        dropout: Dropout rate in the classifier head (default 0.3).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        f = base_filters  # 32

        # Convolutional backbone
        # Each MaxPool(2) halves the sequence length.
        self.conv1 = _conv_block(in_channels, f, kernel_size=7)      # → (batch, 32, 301)
        self.pool1 = nn.MaxPool1d(kernel_size=2)                       # → (batch, 32, 150)

        self.conv2 = _conv_block(f, f * 2, kernel_size=5)            # → (batch, 64, 150)
        self.pool2 = nn.MaxPool1d(kernel_size=2)                       # → (batch, 64, 75)

        self.conv3 = _conv_block(f * 2, f * 4, kernel_size=3)        # → (batch, 128, 75)
        self.pool3 = nn.MaxPool1d(kernel_size=2)                       # → (batch, 128, 37)

        self.conv4 = _conv_block(f * 4, f * 4, kernel_size=3)        # → (batch, 128, 37)
        self.global_pool = nn.AdaptiveAvgPool1d(1)                     # → (batch, 128, 1)

        # Classifier head
        self.head = nn.Sequential(
            nn.Flatten(),                    # → (batch, 128)
            nn.Linear(f * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: float32 tensor (batch, 301, 3)
               Channels: [voltage, current, temperature]

        Returns:
            float32 tensor (batch, 1) — predicted SOH in [0, 1]
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)   # (batch, 3, 301)

        x = self.pool1(self.conv1(x))   # (batch, 32, 150)
        x = self.pool2(self.conv2(x))   # (batch, 64, 75)
        x = self.pool3(self.conv3(x))   # (batch, 128, 37)
        x = self.global_pool(self.conv4(x))  # (batch, 128, 1)

        return self.head(x)             # (batch, 1)


# ---------------------------------------------------------------------------
# Standalone shape check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== SOHCNN1D shape check ===")
    model = SOHCNN1D()
    model.eval()

    batch = torch.randn(4, 301, 3)
    print(f"Input:  {tuple(batch.shape)}")

    # Trace intermediate shapes
    x = batch.permute(0, 2, 1)
    print(f"After permute:  {tuple(x.shape)}")
    x = model.pool1(model.conv1(x))
    print(f"After conv1+pool1: {tuple(x.shape)}")
    x = model.pool2(model.conv2(x))
    print(f"After conv2+pool2: {tuple(x.shape)}")
    x = model.pool3(model.conv3(x))
    print(f"After conv3+pool3: {tuple(x.shape)}")
    x = model.global_pool(model.conv4(x))
    print(f"After conv4+global_pool: {tuple(x.shape)}")

    with torch.no_grad():
        out = model(batch)
    print(f"Output: {tuple(out.shape)}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print("OK")
