import torch
from jaxtyping import Float
from torch import Tensor


def soft_clamp(
    x: Float[Tensor, "..."],
    min_val: float = 0.0,
    max_val: float = 1.0,
    softness: float = 0.05,
) -> Float[Tensor, "..."]:
    """Softly clamp tensor values to range with smooth boundaries.

    Uses sigmoid-based transitions near boundaries for differentiable clamping.
    Outside the soft region, values are hard-clamped.

    Args:
        x: Input tensor of any shape.
        min_val: Lower bound of clamped range.
        max_val: Upper bound of clamped range.
        softness: Width of soft transition region near boundaries.

    Returns:
        Tensor of same shape as input with values clamped to [min_val, max_val].
    """
    i, a, s = min_val, max_val, softness

    lower_mask = x < i + s
    upper_mask = x > a - s

    x_lower = (2 * s) / (1 + torch.exp((2 / s) * (-x + i + s))) + i
    x_upper = (-2 * s) / (1 + torch.exp((-2 / s) * (-x + a - s))) + a

    x = torch.where(lower_mask, x_lower, x)
    x = torch.where(upper_mask, x_upper, x)
    return x
