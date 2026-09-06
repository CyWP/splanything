"""Splitters defining parameter duplication for primitive splits."""

from __future__ import annotations
import torch
from torch import Tensor
from jaxtyping import Float, Bool
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...primitives.base import Primitive


class Splitter:
    """Default splitter producing two child parameter sets per split.

    Centroids are displaced in opposite random directions by up to a
    quarter of the primitive area's extent (sqrt(area) / 4); scales,
    ranges, and sigmas are shrunk by sqrt(0.5); all other parameters are
    copied unchanged.
    """

    def split_vals(
        self, name: str, primitive: Primitive, split_param: Float[Tensor, "N_split ..."]
    ) -> Tuple[Float[Tensor, "N_split ..."], Float[Tensor, "N_split ..."]]:
        """Compute child parameter values for instances being split.

        Args:
            name: Parameter name being split.
            primitive: Primitive being split, with the split instances
                masked in.
            split_param: Parameter values of the instances being split
                (N_split, ...).

        Returns:
            Tuple of two tensors (N_split, ...): values for the retained
            rows and the appended split rows.
        """
        if "centroid" in name:
            disp = torch.rand_like(split_param) * 2 - 1
            disp *= (
                (
                    primitive.areas
                    if hasattr(primitive, "areas")
                    else torch.full_like(split_param[:, 0], (1 / len(primitive)))
                )
                ** 0.5
                / (4 * disp.norm(dim=1))
            )[:, None]
            return split_param + disp, split_param - disp
        if any([n in name for n in ("scale", "range", "sigma")]):
            new = split_param / 2**0.5
            return new, new
        return split_param, split_param

    def __call__(
        self,
        primitive: Primitive,
        name: str,
        param: Float[Tensor, "N ..."],
        split_mask: Bool[Tensor, "N"],
    ) -> Float[Tensor, "N+N_split ..."]:
        """Split a parameter tensor, returning the expanded replacement.

        Runs ``split_vals`` inside a masked context on the instances being
        split, writes the first child values into the original rows, and
        appends the second child values as new rows.

        Args:
            primitive: Primitive being split.
            name: Parameter name being split.
            param: Full parameter tensor (N, ...).
            split_mask: Boolean mask (N,) marking instances to split.

        Returns:
            Expanded tensor (N + N_split, ...): original rows (split
            instances replaced by child 1) followed by child 2 rows.
        """
        with primitive.masked(split_mask):
            split_vals_1, split_vals_2 = self.split_vals(
                name, primitive, param[split_mask]
            )
        p = param.clone()
        p[split_mask] = split_vals_1
        return torch.cat([p, split_vals_2], dim=0)
