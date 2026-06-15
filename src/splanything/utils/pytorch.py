import torch

from jaxtyping import Bool, Integer
from torch import Tensor
from typing import Union, Sequence, Optional

# Useful type
TensorIndex1D = Union[
    int,
    slice,
    type(...),  # Ellipsis
    None,
    Bool[Tensor, "..."],
    Integer[Tensor, "..."],
    Sequence[int],
]


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the most performant compute device available.

    Checks in order of performance: CUDA (NVIDIA), HIP (AMD ROCm), MPS (Apple Silicon), CPU.

    Args:
        device: Optional device string override.

    Returns:
        torch.device: The most performant device available.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        return torch.device("hip")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch, "xla") and hasattr(torch, "get_xla_device"):
        try:
            devices = torch.xla.get_xla_device_list()
            if devices:
                return devices[0]
        except Exception:
            pass
    return torch.device("cpu")
