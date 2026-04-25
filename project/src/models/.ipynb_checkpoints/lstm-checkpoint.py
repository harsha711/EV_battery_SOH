"""
lstm.py — Bidirectional LSTM model for battery SOH prediction.

Processes the 301-timestep sequence in temporal order, capturing long-range
dependencies such as how early-cycle voltage behavior predicts end-of-step
capacity fade.

Uses a 2-layer bidirectional LSTM. The final hidden states from both
directions are concatenated and passed through a small classifier head.

Architecture:
    LSTM(input=3, hidden=128, layers=2, bidirectional=True, dropout=0.2)
    Last hidden state: concat h_n[-2] (forward) and h_n[-1] (backward)
        → (batch, 256)
    Linear(256 → 64) → ReLU → Dropout(0.3) → Linear(64 → 1)

Training note: Use gradient clipping (clip_grad_norm=1.0) to stabilize
training over 301 timesteps. train.py handles this automatically for LSTM.

Standalone usage:
    python src/models/lstm.py   (runs a forward-pass shape check)
"""

import torch
import torch.nn as nn


class SOHLSTM(nn.Module):
    """
    Bidirectional LSTM for battery SOH prediction.

    Processes the raw voltage/current/temperature sequence step-by-step.
    A bidirectional architecture lets the model learn from both forward
    (degradation trend) and backward (end-state) perspectives.

    Args:
        input_size: Number of input features per timestep (default 3: V, I, T).
        hidden_size: Number of hidden units per direction (default 128).
        num_layers: Number of stacked LSTM layers (default 2).
        bidirectional: If True, use bidirectional LSTM (default True).
        dropout: Dropout applied between LSTM layers and in classifier head
                 (default 0.2 for LSTM, 0.3 for head).
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # After concatenating both directions: hidden_size * 2
        lstm_out_dim = hidden_size * self.num_directions  # 256

        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: float32 tensor (batch, 301, 3)
               Already in (batch, seq_len, features) format for batch_first=True.

        Returns:
            float32 tensor (batch, 1) — predicted SOH in [0, 1]
        """
        # output: (batch, 301, hidden*directions)
        # h_n:    (num_layers * num_directions, batch, hidden)
        _, (h_n, _) = self.lstm(x)

        # Extract last layer's hidden states from both directions.
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # Layout: [layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd, ...]
        # Last layer forward:  h_n[-2]  (index num_layers*2 - 2)
        # Last layer backward: h_n[-1]  (index num_layers*2 - 1)
        h_fwd = h_n[-2]  # (batch, hidden_size)
        h_bwd = h_n[-1]  # (batch, hidden_size)

        # Concatenate both directions
        h_last = torch.cat([h_fwd, h_bwd], dim=1)  # (batch, hidden_size * 2)

        return self.head(h_last)  # (batch, 1)


# ---------------------------------------------------------------------------
# Standalone shape check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== SOHLSTM shape check ===")
    model = SOHLSTM()
    model.eval()

    batch = torch.randn(4, 301, 3)
    print(f"Input:  {tuple(batch.shape)}")

    with torch.no_grad():
        _, (h_n, _) = model.lstm(batch)
        print(f"h_n shape (num_layers*directions, batch, hidden): {tuple(h_n.shape)}")
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h_last = torch.cat([h_fwd, h_bwd], dim=1)
        print(f"Concatenated last hidden state: {tuple(h_last.shape)}")
        out = model(batch)

    print(f"Output: {tuple(out.shape)}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print("OK")
