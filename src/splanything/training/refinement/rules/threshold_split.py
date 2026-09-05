"""Split rule thresholding a named primitive attribute."""

from __future__ import annotations

from typing import Literal, Union

from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ..base import RefinementRule, SplitRule


class ThresholdSplit(SplitRule):
    """Split primitives by thresholding a named attribute.

    Same logic as ``ThresholdFilter`` but as a ``SplitRule``:
    ``True`` in the output means SPLIT, ``False`` means IGNORE.

    Attributes:
        attr_name: Attribute name to read from the primitive (e.g. ``"alphas"``, ``"areas"``).
        threshold: Scalar or tensor threshold. For an attribute of shape ``(N, D)``,
            a tensor of shape ``(D,)`` applies per-dimension thresholds.
        comparison: ``"OVER"`` (criterion > threshold) or ``"UNDER"`` (criterion < threshold).
        method: ``"ALL"`` (all trailing dims must satisfy) or ``"ANY"`` (any trailing dim suffices).
    """

    def __init__(
        self,
        attr_name: str,
        threshold: Union[float, Float[Tensor, "..."]],
        comparison: Literal["OVER", "UNDER"] = "OVER",
        method: Literal["ALL", "ANY"] = "ALL",
        interval: int = 10,
    ):
        """Store the threshold configuration.

        Args:
            attr_name: Attribute name read from the primitive.
            threshold: Scalar or per-dimension threshold tensor.
            comparison: ``"OVER"`` or ``"UNDER"`` comparison direction.
            method: ``"ALL"`` or ``"ANY"`` reduction over trailing dims.
            interval: Fire every N invocations of ``__call__``.
        """
        super().__init__(interval=interval)
        self.attr_name = attr_name
        self.threshold = threshold
        self.comparison = comparison
        self.method = method

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N ..."]:
        """Read the named attribute from the primitive.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            Attribute values (N, ...).
        """
        return getattr(primitive, self.attr_name)

    def judge(self, criterion: Float[Tensor, "N ..."]) -> Bool[Tensor, "N"]:
        """Threshold the criterion and reduce trailing dimensions.

        Args:
            criterion: Per-primitive criterion (N, ...).

        Returns:
            split: Boolean mask (N,). True = SPLIT, False = IGNORE.
        """
        if self.comparison == "OVER":
            result: Bool[Tensor, "N ..."] = criterion > self.threshold
        else:
            result = criterion < self.threshold
        if result.ndim == 1:
            return result
        reduce_dims = tuple(range(1, result.ndim))
        if self.method == "ALL":
            return result.all(dim=reduce_dims)
        return result.any(dim=reduce_dims)

    apply = RefinementRule.apply
