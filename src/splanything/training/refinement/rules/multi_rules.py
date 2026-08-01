from __future__ import annotations

import functools
import logging
from typing import Any, Callable, List, Literal, Optional

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ..base import FilterRule, RefinementRule, SplitRule

_logger = logging.getLogger(__name__)

_MODE_FNS: dict[str, Callable] = {
    "AND": lambda masks: functools.reduce(lambda a, b: a & b, masks),
    "OR": lambda masks: functools.reduce(lambda a, b: a | b, masks),
    "XOR": lambda masks: functools.reduce(lambda a, b: a ^ b, masks),
    "NAND": lambda masks: ~functools.reduce(lambda a, b: a & b, masks),
    "NOR": lambda masks: ~functools.reduce(lambda a, b: a | b, masks),
    "XNOR": lambda masks: ~functools.reduce(lambda a, b: a ^ b, masks),
}


class MultiFilterRule(FilterRule):
    """Combine multiple ``FilterRule`` s via a logical mode.

    Each child rule is invoked when the parent fires; their keep masks
    are combined using the chosen mode.

    Modes:
        ``"AND"``  — keep if ALL children keep.
        ``"OR"``   — keep if ANY child keeps.
        ``"XOR"``  — keep if an odd number of children keep.
        ``"NAND"`` — keep unless ALL children keep.
        ``"NOR"`` — keep unless ANY child keeps.
        ``"XNOR"`` — keep if an even number of children keep.

    Attributes:
        rules: Child filter rules.
        mode: Logical combination mode.
        interval: Fire every N invocations of ``__call__``.
    """

    def __init__(
        self,
        rules: List[FilterRule],
        mode: Literal["AND", "OR", "XOR", "NAND", "NOR", "XNOR"] = "AND",
        interval: int = 1,
    ):
        super().__init__(interval=interval)
        self.rules = list(rules)
        self._mode_fn = _MODE_FNS[mode]
        self.mode = mode

    def register(self, primitive: Primitive) -> None:
        super().register(primitive)
        for rule in self.rules:
            rule.register(primitive)

    def unregister(self, primitive: Primitive) -> None:
        super().unregister(primitive)
        for rule in self.rules:
            rule.unregister(primitive)

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        return torch.ones(len(primitive), device=primitive.device, dtype=torch.bool)

    def judge(self, criterion: Float[Tensor, "N"]) -> Optional[Bool[Tensor, "N"]]:
        return criterion

    def apply(self, primitive: Primitive, **kwargs) -> Optional[Bool[Tensor, "N"]]:
        masks: List[Bool[Tensor, "N"]] = []
        for rule in self.rules:
            mask = rule(primitive, **kwargs)
            if mask is not None:
                masks.append(mask)
        if not masks:
            return None
        return self._mode_fn(masks)


class MultiSplitRule(SplitRule):
    """Combine multiple ``SplitRule`` s via a logical mode.

    Each child rule is invoked when the parent fires; their split masks
    are combined using the chosen mode.

    Modes:
        ``"AND"``  — split if ALL children split.
        ``"OR"``   — split if ANY child splits.
        ``"XOR"``  — split if an odd number of children split.
        ``"NAND"`` — split unless ALL children split.
        ``"NOR"`` — split unless ANY child splits.
        ``"XNOR"`` — split if an even number of children split.

    Attributes:
        rules: Child split rules.
        mode: Logical combination mode.
        interval: Fire every N invocations of ``__call__``.
    """

    def __init__(
        self,
        rules: List[SplitRule],
        mode: Literal["AND", "OR", "XOR", "NAND", "NOR", "XNOR"] = "OR",
        interval: int = 1,
    ):
        super().__init__(interval=interval)
        self.rules = list(rules)
        self._mode_fn = _MODE_FNS[mode]
        self.mode = mode

    def register(self, primitive: Primitive) -> None:
        super().register(primitive)
        for rule in self.rules:
            rule.register(primitive)

    def unregister(self, primitive: Primitive) -> None:
        super().unregister(primitive)
        for rule in self.rules:
            rule.unregister(primitive)

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        return torch.ones(len(primitive), device=primitive.device, dtype=torch.bool)

    def judge(self, criterion: Float[Tensor, "N"]) -> Bool[Tensor, "N"]:
        return criterion > 0

    def apply(self, primitive: Primitive, **kwargs) -> Bool[Tensor, "N"]:
        masks: List[Bool[Tensor, "N"]] = []
        for rule in self.rules:
            mask = rule(primitive, **kwargs)
            if mask is not None:
                masks.append(mask)
        if not masks:
            return torch.zeros(
                len(primitive), dtype=torch.bool, device=primitive.device
            )
        return self._mode_fn(masks)
