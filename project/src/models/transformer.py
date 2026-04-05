"""
transformer.py — Transformer Encoder model for battery SOH prediction.

Attends to all 301 timesteps simultaneously using multi-head self-attention.
Unlike the LSTM which processes the sequence left-to-right, the Transformer
can directly relate any two timesteps (e.g. the voltage at t=0 and at t=300)
without information bottlenecks.

Uses PyTorch's built-in nn.TransformerEncoder — attention is not implemented
from scratch.

Architecture:
    Linear(3 → 64)          input projection to d_model
    nn.Embedding(301, 64)   learnable positional encoding (added to projection)
    TransformerEncoder(
        TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256,
                                dropout=0.1, batch_first=True),
        num_layers=3
    )
    Mean pool over seq dim: (batch, 301, 64) → (batch, 64)
    Linear(64 → 32) → ReLU → Dropout(0.2) → Linear(32 → 1)

Memory note: The attention matrix is (batch, 301, 301). At batch_size=256
this is ~92 MB per layer. Use batch_size=128 for training.

Standalone usage:
    python src/models/transformer.py   (runs a forward-pass shape check)
"""

import torch
import torch.nn as nn


class SOHTRANSFORMER_ALIASES:
    """Aliases kept for backwards compatibility in case registry uses old name."""
    pass


class SOHTransformer(nn.Module):
    """
    Transformer Encoder for battery SOH prediction.

    Projects the 3-channel input to d_model dimensions, adds learnable
    positional embeddings, runs through a stack of Transformer encoder layers,
    then mean-pools across the sequence dimension before the output head.

    Learnable positional embeddings (nn.Embedding) are used instead of
    sinusoidal because the sequence length is fixed at 301 and the model
    can learn task-specific position representations.

    Args:
        input_dim: Number of input channels (default 3: V, I, T).
        d_model: Transformer hidden dimension (default 64).
        nhead: Number of attention heads (default 4). Must divide d_model.
        num_encoder_layers: Number of stacked encoder layers (default 3).
        dim_feedforward: FFN hidden dimension inside each encoder layer (default 256).
        dropout: Dropout applied inside attention and FFN layers (default 0.1).
        max_seq_len: Maximum sequence length for positional embeddings (default 301).
    """

    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 301,
    ) -> None:
        super().__init__()

        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Project 3-channel input to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional encoding: one embedding per position
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        # Register position indices as a buffer so they move with the model's device
        self.register_buffer(
            "positions",
            torch.arange(max_seq_len).unsqueeze(0),  # (1, max_seq_len)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # expects (batch, seq, d_model)
            norm_first=False,   # Post-LN (standard)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,  # avoid warning when no padding mask
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
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
        batch_size, seq_len, _ = x.shape

        # Project input: (batch, seq_len, d_model)
        x = self.input_proj(x)

        # Add positional encoding
        pos = self.positions[:, :seq_len]            # (1, seq_len)
        x = x + self.pos_embedding(pos)              # (batch, seq_len, d_model)

        # Transformer encoder
        x = self.transformer(x)                      # (batch, seq_len, d_model)

        # Mean pool over sequence dimension
        x = x.mean(dim=1)                            # (batch, d_model)

        return self.head(x)                          # (batch, 1)

    def get_attention_weights(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract attention weight matrices from all encoder layers.

        Used by evaluate.py to visualize which timesteps the model focuses on.
        Runs a forward pass with hooks to capture attention weights.

        Args:
            x: float32 tensor (batch, 301, 3)

        Returns:
            List of attention weight tensors, one per encoder layer.
            Each has shape (batch, nhead, seq_len, seq_len).
        """
        attention_weights: list[torch.Tensor] = []

        def make_hook(attn_weights_list: list) -> callable:
            def hook(module, input, output):
                # TransformerEncoderLayer doesn't expose attn weights directly.
                # We re-run attn manually.
                pass
            return hook

        # Run a manual attention extraction using the built-in MHA
        self.eval()
        batch_size, seq_len, _ = x.shape

        with torch.no_grad():
            z = self.input_proj(x)
            pos = self.positions[:, :seq_len]
            z = z + self.pos_embedding(pos)

            for layer in self.transformer.layers:
                # Access the MultiheadAttention sub-module directly
                mha = layer.self_attn
                # norm1 is applied before (Pre-LN) or after (Post-LN) — we pass z as-is
                attn_out, attn_weights = mha(
                    z, z, z,
                    need_weights=True,
                    average_attn_weights=False,  # keep per-head weights
                )
                attention_weights.append(attn_weights.detach().cpu())
                # Continue the layer forward normally
                z = layer(z)

        return attention_weights


# ---------------------------------------------------------------------------
# Standalone shape check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== SOHTransformer shape check ===")
    model = SOHTransformer()
    model.eval()

    batch = torch.randn(4, 301, 3)
    print(f"Input:  {tuple(batch.shape)}")

    with torch.no_grad():
        # Trace intermediate shapes
        z = model.input_proj(batch)
        print(f"After input_proj: {tuple(z.shape)}")
        pos = model.positions[:, :301]
        z = z + model.pos_embedding(pos)
        print(f"After pos_embedding add: {tuple(z.shape)}")
        z = model.transformer(z)
        print(f"After transformer encoder: {tuple(z.shape)}")
        z_pooled = z.mean(dim=1)
        print(f"After mean pool: {tuple(z_pooled.shape)}")
        out = model(batch)

    print(f"Output: {tuple(out.shape)}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Test attention extraction
    attn = model.get_attention_weights(batch[:2])
    print(f"\nAttention weights per layer:")
    for i, a in enumerate(attn):
        print(f"  Layer {i}: {tuple(a.shape)}  (batch, heads, seq, seq)")

    print("OK")
