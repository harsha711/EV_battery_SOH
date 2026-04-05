"""
models/__init__.py — Model registry for the 4-model SOH prediction pipeline.

All models share the same interface:
    input:  (batch, 301, 3)  float32  — [voltage, current, temperature]
    output: (batch, 1)       float32  — predicted SOH in [0, 1]
"""

from .mlp import SOHMLP
from .cnn1d import SOHCNN1D
from .lstm import SOHLSTM
from .transformer import SOHTransformer

MODEL_REGISTRY: dict[str, type] = {
    "mlp": SOHMLP,
    "cnn": SOHCNN1D,
    "lstm": SOHLSTM,
    "transformer": SOHTransformer,
}

__all__ = ["SOHMLP", "SOHCNN1D", "SOHLSTM", "SOHTransformer", "MODEL_REGISTRY"]
