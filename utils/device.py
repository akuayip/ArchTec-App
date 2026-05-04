"""
Hardware / device utilities shared across the project.
"""

import torch


def get_device() -> str:
    """Return the best available compute device string."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def print_device_info() -> None:
    """Print a brief hardware summary to stdout (useful for debugging)."""
    print(f"PyTorch version : {torch.__version__}")
    print(f"MPS available   : {torch.backends.mps.is_available()}")
    print(f"MPS built       : {torch.backends.mps.is_built()}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"Active device   : {get_device()}")
