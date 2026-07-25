from __future__ import annotations

from typing import Literal, Optional

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ....utils.img import ImgUtils
from ..base import FilterRule, RefinementRule, SplitRule


class PrimitiveFloor(SplitRule):
    """Force splits when the primitive count drops below ``min_length``.

    A floor on the number of primitives a ``Primitive`` may contain.
    When the rule fires and ``len(primitive) < min_length``, it returns
    a boolean split mask that flags exactly ``min_length - len(primitive)``
    elements for splitting, raising the primitive back to the floor.
    Above the floor (``can_apply`` returns False) the rule is a no-op
    and the primitive is left untouched.

    The selection of which primitives to split is governed by
    ``strategy``:

    - ``"stochastic"``: draw one uniform random score per primitive and
      split the ``k`` highest. Equivalent to uniform random splitting.
    - ``"map"``: bilinearly sample ``map`` at each primitive's centroid
      and split the ``k`` primitives whose sampled values are highest.
      Primitives in high-valued regions of the map get split first.
    - ``"map_stochastic"``: same as ``"map"`` but the score is the
      sampled value multiplied by an independent uniform random draw.
      Primitives in high-valued regions are split on average more often,
      with stochastic variation per call.
    - ``"rule"``: defer ranking to another ``RefinementRule`` whose
      ``criterion(primitive)`` produces the per-primitive score.
      ``descending`` controls the cull direction: ``False`` (default)
      splits the ``k`` highest scores; ``True`` splits the ``k`` lowest
      scores.

    Attributes:
        min_length: Minimum allowed number of primitives.
        strategy: Selection strategy; one of ``"stochastic"``,
            ``"map"``, ``"map_stochastic"``, ``"rule"``.
        map: Score map (B, 1, H, W). Required for ``"map"`` and
            ``"map_stochastic"``. Centroids are assumed to be in [0, 1],
            matching ``gen_px_coords`` convention.
        rule: ``RefinementRule`` whose ``criterion`` ranks primitives.
            Required for ``"rule"``. Any rule with a ``criterion``
            method works (``FilterRule`` or ``SplitRule``).
        descending: For ``"rule"``, split the ``k`` lowest scores when
            True; split the ``k`` highest scores when False. Ignored
            for the other strategies (they always split the ``k``
            highest scores).
        interval: Fire every N invocations of ``__call__``.

    Notes:
        - Implements the ``SplitRule`` interface: ``judge`` returns
          True == SPLIT and False == IGNORE, matching the convention
          of the other split rules in this module.
        - Only the first batch element of ``map`` (``map[0]``) is
          sampled, consistent with ``MapFilter`` and ``MapSplit``.
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
