from __future__ import annotations

from typing import Literal, Optional

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ....utils.img import ImgUtils
from ..base import FilterRule, RefinementRule, SplitRule


class PrimitiveFloor(SplitRule):
    """Enforce a minimum primitive count, splitting when below ``min_length``.

    When ``len(primitive) < min_length``, exactly ``min_length - N``
    primitives are flagged for splitting. Above the floor, the rule is
    a no-op.

    Selection strategy for which primitives to split:

    - ``"stochastic"``: uniform random splitting.
    - ``"map"``: bilinearly sample ``map`` at centroids; split primitives
      with highest sampled values.
    - ``"map_stochastic"``: sampled value * uniform random; probabilistically
      weighted toward high-valued regions.
    - ``"rule"``: defer ranking to another ``RefinementRule.criterion``.
      ``descending=True`` splits the lowest scores; ``False`` (default)
      splits the highest.

    Attributes:
        min_length: Minimum allowed number of primitives.
        strategy: Selection strategy.
        map: Score map (B, 1, H, W) for ``"map"`` / ``"map_stochastic"``.
        rule: ``RefinementRule`` for ``"rule"`` strategy.
        descending: For ``"rule"``, split lowest when True.
        interval: Fire every N invocations.

    Notes:
        - Only ``map[0]`` (first batch element) is sampled.
        - Centroids must be in [0, 1] UV coordinates.
    """

    STRATEGIES = ("stochastic", "map", "map_stochastic", "rule")

    def __init__(
        self,
        min_length: int,
        strategy: Literal["stochastic", "map", "map_stochastic", "rule"] = "stochastic",
        map: Optional[Float[Tensor, "B 1 H W"]] = None,
        rule: Optional[FilterRule] = None,
        descending: bool = False,
        interval: int = 1,
    ):
        """Initialize the floor rule.

        Args:
            min_length: Minimum allowed number of primitives.
            strategy: Selection strategy (see class docstring).
            map: Score map (B, 1, H, W). Required for ``"map"`` and
                ``"map_stochastic"``.
            rule: ``RefinementRule`` providing the ranking criterion.
                Required for ``"rule"``.
            descending: For ``"rule"``, split lowest scores when True,
                highest scores when False.
            interval: Fire every N invocations of ``__call__``.

        Raises:
            ValueError: If arguments are inconsistent with the chosen
                strategy or if ``min_length`` is negative.
        """
        super().__init__(interval=interval)
        if min_length < 0:
            raise ValueError(f"min_length must be non-negative, got {min_length}.")
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'; expected one of {self.STRATEGIES}."
            )
        if strategy in ("map", "map_stochastic") and map is None:
            raise ValueError(f"strategy '{strategy}' requires a `map` argument.")
        if strategy == "rule" and rule is None:
            raise ValueError("strategy 'rule' requires a `rule` argument.")

        self.min_length = min_length
        self.strategy = strategy
        self.map = map
        self.rule = rule
        self.descending = descending

    def can_apply(self, primitive: Primitive, **kwargs) -> bool:
        """Only fire when the primitive actually falls below the floor."""
        return len(primitive) < self.min_length

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Compute per-primitive score used to rank split candidates.

        Returns a tensor of length ``N`` with one score per primitive.
        ``judge`` splits the ``k = min_length - N`` highest scores
        (or, for ``strategy="rule"`` with ``descending=True``, the
        ``k`` lowest scores).

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
            sampled = ImgUtils.uv_sample(self.map, primitive.centroids)
            values = sampled[0, :, 0]
            if self.strategy == "map_stochastic":
                values = values * torch.rand(n, device=device, dtype=dtype)
            return values

        if self.strategy == "rule":
            return self.rule.criterion(primitive, **kwargs)

        raise RuntimeError(f"Unhandled strategy '{self.strategy}'.")

    def judge(self, criterion: Float[Tensor, "N"]) -> Bool[Tensor, "N"]:
        """Split the top (or bottom) ``k = min_length - N`` scores.

        Args:
            criterion: Per-primitive scores (N,) from ``criterion``.

        Returns:
            split: Boolean mask (N,). True == SPLIT, False == IGNORE.
        """
        n = criterion.shape[0]
        k = self.min_length - n
        if k <= 0:
            return torch.zeros(n, dtype=torch.bool, device=criterion.device)
        largest = not (self.strategy == "rule" and self.descending)
        _, split_idx = torch.topk(criterion, k, largest=largest)
        split_mask = torch.zeros(n, dtype=torch.bool, device=criterion.device)
        split_mask[split_idx] = True
        return split_mask

    apply = RefinementRule.apply
