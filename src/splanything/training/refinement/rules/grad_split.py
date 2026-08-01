import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ..base import RefinementRule, SplitRule


class GradSplit(SplitRule):
    """Split primitives with high gradient magnitude.

    Aggregates the absolute gradient of a named attribute
    (summing over trailing dimensions), optionally multiplies by alpha,
    and splits primitives with large values (indicating high-detail regions).

    Attributes:
        threshold: Split threshold on the aggregated gradient.
        attr_name: Attribute whose gradient is evaluated (default ``"alphas"``).
            When ``None``, aggregates gradients from all batched parameters.
        interval: Fire every N invocations.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        attr_name: str | None = "alphas",
        interval: int = 10,
    ):
        super().__init__(interval=interval)
        self.threshold = threshold
        self.attr_name = attr_name

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        if self.attr_name is not None:
            param = getattr(primitive, self.attr_name)
            if param.grad is None:
                return torch.zeros(
                    (len(primitive),), device=primitive.device, dtype=primitive.dtype
                )
            grad = param.grad.abs()
            if grad.ndim > 1:
                grad = grad.sum(dim=tuple(range(1, grad.ndim)))
            return grad * primitive.alphas
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
        return criterion > self.threshold
