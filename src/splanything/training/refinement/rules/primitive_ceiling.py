from __future__ import annotations

from typing import Literal, Optional

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ....utils.img import ImgUtils
from ..base import FilterRule, RefinementRule


class PrimitiveCeiling(FilterRule):
    """Cap the number of primitives, culling excess when above ``max_length``.

    When ``len(primitive) > max_length``, exactly ``N - max_length``
    primitives are culled. Below the cap, the rule is a no-op.

    Selection strategy for which primitives to drop:

    - ``"stochastic"``: uniform random culling.
    - ``"map"``: bilinearly sample ``map`` at centroids; drop primitives
      with lowest sampled values.
    - ``"map_stochastic"``: sampled value * uniform random; probabilistically
      weighted toward low-valued regions.
    - ``"rule"``: defer ranking to another ``FilterRule.criterion``.
      ``descending=True`` drops the highest scores; ``False`` drops the
      lowest.

    Attributes:
        max_length: Maximum allowed number of primitives.
        strategy: Selection strategy.
        map: Score map (B, 1, H, W) for ``"map"`` / ``"map_stochastic"``.
        rule: ``FilterRule`` for ``"rule"`` strategy.
        descending: For ``"rule"``, cull highest when True.
        interval: Fire every N invocations.

    Notes:
        - Only ``map[0]`` (first batch element) is sampled.
        - Centroids must be in [0, 1] UV coordinates.
    """

    STRATEGIES = ("stochastic", "map", "map_stochastic", "rule")

    def __init__(
        self,
        max_length: int,
        strategy: Literal["stochastic", "map", "map_stochastic", "rule"] = "stochastic",
        map: Optional[Float[Tensor, "B 1 H W"]] = None,
        rule: Optional[FilterRule] = None,
        descending: bool = False,
        coords_attr: str = "adjusted_coords",
        interval: int = 1,
    ):
        """Initialize the ceiling rule.

        Args:
            max_length: Maximum allowed number of primitives.
            strategy: Selection strategy (see class docstring).
            map: Probability map (B, 1, H, W). Required for ``"map"``
                and ``"map_stochastic"``.
            rule: ``FilterRule`` providing the ranking criterion.
                Required for ``"rule"``.
            descending: For ``"rule"``, cull highest scores when True,
                lowest scores when False.
            coords_attr: Attribute name for map sampling coordinates (default ``"centroids"``).
            interval: Fire every N invocations of ``__call__``.

        Raises:
            ValueError: If arguments are inconsistent with the chosen
                strategy or if ``max_length`` is negative.
        """
        super().__init__(interval=interval)
        if max_length < 0:
            raise ValueError(f"max_length must be non-negative, got {max_length}.")
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'; expected one of {self.STRATEGIES}."
            )
        if strategy in ("map", "map_stochastic") and map is None:
            raise ValueError(f"strategy '{strategy}' requires a `map` argument.")
        if strategy == "rule" and rule is None:
            raise ValueError("strategy 'rule' requires a `rule` argument.")

        self.max_length = max_length
        self.strategy = strategy
        self.map = map
        self.rule = rule
        self.descending = descending
        self.coords_attr = coords_attr

    def can_apply(self, primitive: Primitive, **kwargs) -> bool:
        """Only fire when the primitive actually exceeds the ceiling."""
        return len(primitive) > self.max_length

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Compute per-primitive score used to rank cull candidates.

        Returns a tensor of length ``N`` with one score per primitive.
        ``judge`` drops the ``k = N - max_length`` lowest scores
        (or, for ``strategy="rule"`` with ``descending=True``, the
        ``k`` highest scores).

        Args:
            primitive: Primitive being evaluated.

        Returns:
            Per-primitive scores (N,).
        """
        n = len(primitive)
        device = primitive.device
        dtype = primitive.dtype

        if self.strategy == "stochastic":
            return torch.rand(n, device=device, dtype=dtype)

        if self.strategy in ("map", "map_stochastic"):
            sampled = ImgUtils.uv_sample(self.map, getattr(primitive, self.coords_attr))
            values = sampled[0, :, 0]
            if self.strategy == "map_stochastic":
                values = values * torch.rand(n, device=device, dtype=dtype)
            return values

        if self.strategy == "rule":
            return self.rule.criterion(primitive, **kwargs)

        raise RuntimeError(f"Unhandled strategy '{self.strategy}'.")

    def judge(self, criterion: Float[Tensor, "N"]) -> Bool[Tensor, "N"]:
        """Drop the bottom (or top) ``k = N - max_length`` scores.

        Args:
            criterion: Per-primitive scores (N,) from ``criterion``.

        Returns:
            keep: Boolean mask (N,). True == KEEP, False == CULL.
        """
        n = criterion.shape[0]
        k = n - self.max_length
        if k <= 0:
            return torch.ones(n, dtype=torch.bool, device=criterion.device)
        largest = self.strategy == "rule" and self.descending
        _, drop_idx = torch.topk(criterion, k, largest=largest)
        keep = torch.ones(n, dtype=torch.bool, device=criterion.device)
        keep[drop_idx] = False
        return keep

    apply = RefinementRule.apply
