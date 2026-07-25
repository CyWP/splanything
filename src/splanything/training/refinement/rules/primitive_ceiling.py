from __future__ import annotations

from typing import Literal, Optional

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ....utils.img import ImgUtils
from ..base import FilterRule, RefinementRule


class PrimitiveCeiling(FilterRule):
    """Cull excess primitives when their count exceeds ``max_length``.

    A cap on the number of primitives a ``Primitive`` may contain. When
    the rule fires and ``len(primitive) > max_length``, it returns a
    boolean keep mask that drops exactly ``len(primitive) - max_length``
    elements, lowering the primitive back to the ceiling. Below the cap
    (``can_apply`` returns False) the rule is a no-op and the primitive
    is left untouched.

    The selection of which primitives to drop is governed by ``strategy``:

    - ``"stochastic"``: draw one uniform random score per primitive and
      drop the ``k`` lowest. Equivalent to uniform random culling.
    - ``"map"``: bilinearly sample ``map`` at each primitive's centroid
      and drop the ``k`` primitives whose sampled values are lowest.
      Primitives in low-valued regions of the map get culled first.
    - ``"map_stochastic"``: same as ``"map"`` but the score is the
      sampled value multiplied by an independent uniform random draw.
      Primitives in high-valued regions can still be dropped, with
      probability proportional to their sampled value.
    - ``"rule"``: defer ranking to another ``FilterRule`` whose
      ``criterion(primitive)`` produces the per-primitive score.
      ``descending`` controls the cull direction: ``False`` drops the
      ``k`` lowest scores (ascending cull); ``True`` drops the ``k``
      highest scores (descending cull).

    Attributes:
        max_length: Maximum allowed number of primitives.
        strategy: Selection strategy; one of ``"stochastic"``,
            ``"map"``, ``"map_stochastic"``, ``"rule"``.
        map: Probability map (B, 1, H, W) in [0, 1]. Required for
            ``"map"`` and ``"map_stochastic"``. Centroids are assumed
            to be in [0, 1], matching ``gen_px_coords`` convention.
        rule: ``FilterRule`` whose ``criterion`` ranks primitives.
            Required for ``"rule"``.
        descending: For ``"rule"``, drop the ``k`` highest scores when
            True; drop the ``k`` lowest scores when False. Ignored
            for the other strategies (they always drop the ``k`` lowest
            scores).
        interval: Fire every N invocations of ``__call__``.

    Notes:
        - Implements the ``FilterRule`` interface: ``judge`` returns
          True == KEEP and False == CULL, matching the convention of
          the other rules in this module.
        - Only the first batch element of ``map`` (``map[0]``) is
          sampled, consistent with ``MapFilter`` and ``MapSplit``.
    """

    STRATEGIES = ("stochastic", "map", "map_stochastic", "rule")

    def __init__(
        self,
        max_length: int,
        strategy: Literal["stochastic", "map", "map_stochastic", "rule"] = "stochastic",
        map: Optional[Float[Tensor, "B 1 H W"]] = None,
        rule: Optional[FilterRule] = None,
        descending: bool = False,
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
            sampled = ImgUtils.uv_sample(self.map, primitive.centroids)
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
