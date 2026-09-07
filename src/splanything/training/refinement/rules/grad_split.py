"""Split rule based on aggregated parameter gradient magnitudes."""

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ..base import RefinementRule, SplitRule


class GradSplit(SplitRule):
    """Split primitives with high gradient magnitude.

    Aggregates the absolute gradient of one or more named attributes
    (summing over trailing dimensions per attribute, then summing across
    attributes), optionally multiplies by alpha, and splits primitives
    with large values (indicating high-detail regions).

    Attributes:
        threshold: Split threshold on the aggregated gradient.
        attr_names: Attribute(s) whose gradient is evaluated (default ``"alphas"``).
            A single string or a list of strings. When ``None``, aggregates
            gradients from all batched parameters.
        interval: Fire every N invocations.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        attr_names: str | list[str] | None = None,
        interval: int = 10,
    ):
        """Store the split configuration.

        Args:
            threshold: Split threshold on the aggregated gradient.
            attr_names: Attribute name(s) whose gradients are aggregated;
                ``None`` aggregates all batched parameter gradients.
            interval: Fire every N invocations of ``__call__``.
        """
        super().__init__(interval=interval)
        self.threshold = threshold
        if isinstance(attr_names, str):
            attr_names = [attr_names]
        self.attr_names = attr_names

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Aggregate absolute gradient magnitudes, scaled by alpha.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            Per-primitive aggregated gradient magnitude (N,), multiplied
            by ``alphas``.
        """
        if self.attr_names is not None:
            grad_mag = torch.zeros(
                (len(primitive),), device=primitive.device, dtype=primitive.dtype
            )
            for name in self.attr_names:
                param = getattr(primitive, name)
                if param.grad is None:
                    continue
                g = param.grad.abs()
                if g.ndim > 1:
                    g = g.sum(dim=tuple(range(1, g.ndim)))
                grad_mag += g
            return grad_mag * primitive.alphas
        areas = (
            primitive.areas
            if hasattr(primitive, "areas")
            else torch.ones(
                (len(primitive),), device=primitive.device, dtype=primitive.dtype
            )
        ) ** 0.5
        grad_mag = torch.zeros(
            (len(primitive),), device=areas.device, dtype=areas.dtype
        )
        for named, grad in primitive.batched_grads():
            g = grad.abs()
            if len(g.shape) > 1:
                g = g.sum(dim=tuple(range(1, len(g.shape))))
            grad_mag += g
        return grad_mag * primitive.alphas

    def judge(self, criterion: Float[Tensor, "N"]) -> Bool[Tensor, "N"]:
        """Split primitives whose aggregated gradient exceeds the threshold.

        Args:
            criterion: Per-primitive gradient magnitudes (N,).

        Returns:
            split: Boolean mask (N,). True = SPLIT, False = IGNORE.
        """
        return criterion > self.threshold
