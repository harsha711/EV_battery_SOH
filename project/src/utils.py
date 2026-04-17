"""
utils.py — small shared helpers for the notebooks.

Keep this file tiny: anything bigger than a helper belongs in its own module.
"""

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed every RNG that PyTorch training touches.

    Args:
        seed: Integer seed applied to python, numpy, torch CPU + CUDA.
        deterministic: If True, force cuDNN into deterministic mode. Slightly
                       slower but results are reproducible run-to-run on the
                       same GPU. Set False for HPO trials where a small amount
                       of nondeterminism is acceptable for speed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
