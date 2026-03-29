"""
Centralized hardware-accelerator detection.

Every module that needs a device should do:

    from device import DEVICE
"""

import torch


def get_device() -> torch.device:
    """Return the best available accelerator: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE: torch.device = get_device()
