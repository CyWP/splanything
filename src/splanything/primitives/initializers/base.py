"""Initializers creating initial parameter tensors for primitives."""

import torch
from torch import Tensor
from typing import Tuple, Optional
from jaxtyping import Float


class Initializer:
    """Default initializer with name-based heuristics.

    Maps parameter names to sensible random initializations: angles
    uniform in [0, 2*pi], centroids uniform in [0, 1], alphas near 1,
    scales/ranges/sigmas scaled by primitive count, colors uniform in
    [0, 1], unit-norm axes, anything else standard normal.
    """

    def init_param(
        self, name: str, param_shape: Tuple[int], batched: bool
    ) -> Float[Tensor, "N ..."] | Float[Tensor, "..."]:
        """Initialize a parameter tensor based on its name.

        Args:
            name: Parameter name; substring-matched against known patterns.
            param_shape: Shape of the parameter tensor.
            batched: Whether the parameter has a batch dimension.

        Returns:
            Initialized tensor (N, ...).
        """
        if any([c in name for c in ("theta", "angle")]):
            return torch.rand(param_shape) * 2 * torch.pi
        if any([c in name for c in ("centroid",)]):
            return torch.rand(param_shape)
        if any([c in name for c in ("alpha", "transparency")]):
            return 1 - (torch.rand(param_shape) * 2 / 3) ** 2
        if any([c in name for c in ("scale", "range", "sigma")]):
            return (torch.rand(param_shape) + 0.5) / param_shape[0] ** 0.5
            size = param_shape[0]
            area_factor = 1 / size**0.5
            rand = torch.randn(param_shape)
            pos_mask = rand > 0
            rand[pos_mask] += 0.25
            rand[~pos_mask] -= 0.25
            return rand * area_factor * 0.8
        if any([c in name for c in ("ref_axis", "reference_axis")]) and param_shape == (
            2,
        ):
            torch.tensor([-1.0, 0.0])
        if any([c in name for c in ("axis",)]):
            p = torch.rand(param_shape)
            return p / p.norm()
        if any([c in name for c in ("color", "red", "green", "blue")]):
            return torch.rand(param_shape)
        return torch.randn(param_shape)

    def __call__(
        self, name: str, size: int, channels: Optional[Tuple[int]] = None
    ) -> Float[Tensor, "Size ..."]:
        """Validate inputs, build the parameter shape, and initialize it.

        Args:
            name: Parameter name.
            size: Number of primitives; 0 for non-batched parameters.
            channels: Optional trailing shape of the parameter.

        Returns:
            Initialized tensor of shape (size, *channels), or (*channels,)
            if size == 0.

        Raises:
            ValueError: If ``size < 0`` or any channel is negative.
        """
        if channels is not None:
            if any([c < 0 for c in channels]):
                raise ValueError(
                    f"Value '{channels}' for channels is invalid. Must be >=0."
                )
        if size < 0:
            raise ValueError(
                f"Value '{size}' for size is invalid. Must be 0 for non batched params and >=1 for batched params."
            )
        if size == 0:
            param_shape = () if channels is None else channels
        else:
            param_shape = (size,) if channels is None else (size, *channels)
        return self.init_param(name, param_shape, size != 0)
