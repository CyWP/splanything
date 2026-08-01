from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from typing import Dict, ItemsView, List, Optional, Any, TYPE_CHECKING

import math
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ..rendering.sample_output import SampleOutput
from .base import Primitive, nomask

if TYPE_CHECKING:
    from ..training.regularizers.base import Regularizer
    from ..training.refinement.base import SplitRule, FilterRule

_logger = logging.getLogger(__name__)


class MultiPrimitive(Primitive):
    def __init__(
        self,
        primitives: Dict[str, Primitive],
        filter_rules: Optional[List[FilterRule]] = None,
        split_rules: Optional[List[SplitRule]] = None,
        sample_processors: Optional[List[SampleProcessor]] = None,
        regularizers: Optional[Dict[str, Tuple[Regularizer, float]]] = None,
    ):
        nn.Module.__init__(self)
        self._context_masks: List[Bool[Tensor, "N"]] = []
        self.primitives = nn.ModuleDict(primitives)
        if filter_rules is not None:
            for f in filter_rules:
                self.add_filter_rule(f)
        if split_rules is not None:
            for s in split_rules:
                self.add_split_rule(s)
        if sample_processors is not None:
            for s in sample_processors:
                self.add_sample_processor(s)
        if regularizers is not None:
            for name, (r, weight) in regularizers.items():
                self.add_regularizer(name, r, weight)
        self._rescale_primitives()

    def _rescale_primitives(self):
        total = len(self)
        if total == 0:
            return
        for prim in self.primitives.values():
            factor = len(prim) / total
            if factor != 1.0:
                prim.scale(factor)

    @contextmanager
    def masked(self, mask: Bool[Tensor, "N"]):
        """Context manager for implicit masking of batched parameters.

        When active, accessing batched parameters returns masked versions.
        Supports nested contexts.

        Args:
            masks: Dict of Boolean tensor where True=keep, False=remove.

        Yields:
            None.
        """
        with ExitStack() as stack:
            tmp = []
            si = 0
            for prim in self.primitives.values():
                ei = si + len(prim)
                tmp.append(stack.enter_context(prim.masked(mask[si:ei])))
                si = ei
            yield

    @contextmanager
    def cache_properties(self):
        """Context manager enabling cached_property caching.

        While active, @cached_property-decorated properties compute once
        and return cached values on subsequent accesses. The cache is
        cleared on exit. Supports nesting (outermost exit clears).

        Yields:
            None.
        """
        with ExitStack() as stack:
            _ = [
                stack.enter_context(prim.cache_properties())
                for prim in self.primitives.values()
            ]
        yield

    def add_parameter(self, *args, **kwargs):
        raise NotImplementedError(
            f"This class ('{self.__class__}') is meant as a container for primitives only. Add parameters directly in the contained primitives."
        )

    def update_parameters(self, updates: Dict[str, Dict[str, Tensor]]):
        """Update parameters with new tensors.

        Replaces parameters with new tensors. After update, validates that
        all batched parameters have the same size in their first dimension.

        Args:
            updates: Dict of Dict mapping parameter names to new tensors.

        Raises:
            KeyError: If a parameter name is not found.
            ValueError: If batched parameters have inconsistent first-dimension sizes.
        """
        for name, updated_params in updates.items():
            prim = self.primitives[name]
            prim.update_parameters(updated_params)

    def clear_cache(self):
        """Manually clear the property cache."""
        for prim in self.primitives.values():
            prim._property_cache.clear()

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def __len__(self) -> int:
        """Number of primitives in this object."""
        return sum([len(prim) for prim in self.primitives.values()])

    @nomask
    def _validate_batched_sizes(self):
        """Ensure all batched parameters have the same first-dimension size."""
        for prim in self.primitives.values():
            prim._validate_batched_sizes()

    def __getitem__(self, key: str) -> Primitive:
        return self.primitives[key]

    @torch.no_grad()
    def patch_mask(
        self,
        centers: Float[Tensor, "P 2"],
        patch_sizes: Integer[Tensor, "P"],
        H: Integer[Tensor, "P"],
        W: Integer[Tensor, "P"],
    ) -> Bool[Tensor, "P N"]:
        return torch.cat(
            [
                prim.patch_mask(centers, patch_sizes, H, W)
                for prim in self.primitives.values()
            ],
            dim=1,
        )

    @nomask
    @torch.no_grad()
    def filter(self, keys: Dict[str,]) -> MultiPrimitive:
        """In-place index selection of batched elements.

        Filters primitive parameters to keep only elements matching key.
        Modifies the primitive in-place.

        Args:
            key: Boolean mask or integer indices to select.

        Notes:
            - Only applies to batched parameters (shape[0] == len(self)).
            - Used by refinement rules to cull primitives.
        """
        for name, idx in keys.items():
            self.primitives[name].filter(idx)
        return self

    @nomask
    @torch.no_grad()
    def split(self, idx: Dict[str, Bool[Tensor, "Ns"]]) -> MultiPrimitive:
        """Split instances at given indices"""
        for (
            name,
            i,
        ) in idx.items():
            self.primitives[name].split(i)
        return self

    def forward(self, co: Float[Tensor, "Nc 2"]) -> SampleOutput:
        """Sample primitive values at coordinates.

        Args:
            co: Coordinates to sample at (N, 2).
            rasterizer: Rasterizer Callable to aggregate rgb, a, weights.

        Returns:
            SampleOutput object.

        Notes:
            - Returns zeros if len(self) == 0.
            - Uses masked batched parameters if context is active.
        """
        return SampleOutput.cat(*[p(co) for p in self.primitives.values()])

    @nomask
    @torch.no_grad()
    def append(
        self,
        other: MultiPrimitive,
        weight: float | Dict[str, float] = 0.0,
        ignore_exclusive: bool = False,
    ) -> MultiPrimitive:
        for key, prim in other.primitives.items():
            if key in self.primitives:
                if isinstance(weight, Dict):
                    w = weight.get(key, 0.0)
                else:
                    w = weight
                self.primitives[key].append(prim, weight=w)
            elif not ignore_exclusive:
                self.primitives[key] = prim
        return self

    @nomask
    def param_groups(self) -> List[Dict[str, nn.Parameter | Any]]:
        groups = []
        for name, prim in self.primitives.items():
            pg = prim.param_groups()
            for g in pg:
                g["name"] = (
                    f"{name}$${g['name']}"  # Do not change '$$', needed for OptimizerWrapper (specified in optimizer.py)
                )
            groups.extend(pg)
        return groups

    def batched_parameters(self) -> ItemsView[str, Float[Tensor, "N ..."]]:
        raise NotImplementedError(
            f"This class ('{self.__class__}') is meant as a container for primitives only. Access parameters directly in the contained primitives."
        )

    def stable_parameters(self) -> ItemsView[str, Float[Tensor, "N ..."]]:
        raise NotImplementedError(
            f"This class ('{self.__class__}') is meant as a container for primitives only. Access parameters directly in the contained primitives."
        )

    def named_grads(self) -> ItemsView[str, Float[Tensor, "..."]]:
        raise NotImplementedError(
            f"This class ('{self.__class__}') is meant as a container for primitives only. Access parameters directly in the contained primitives."
        )

    def batched_grads(self) -> ItemsView[str, Float[Tensor, "..."]]:
        raise NotImplementedError(
            f"This class ('{self.__class__}') is meant as a container for primitives only. Access parameters directly in the contained primitives."
        )

    def stable_grads(self) -> ItemsView[str, Float[Tensor, "..."]]:
        raise NotImplementedError(
            f"This class ('{self.__class__}') is meant as a container for primitives only. Access parameters directly in the contained primitives."
        )

    def add_split_rule(self, rule: SplitRule):
        for prim in self.primitives.values():
            prim.add_split_rule(rule)

    def add_filter_rule(self, rule: FilterRule):
        for prim in self.primitives.values():
            prim.add_filter_rule(rule)

    def add_sample_processor(self, proc: SampleProcessor):
        for prim in self.primitives.values():
            prim.add_sample_processor(proc)

    def add_regularizer(self, name: str, reg: Regularizer, weight: float = 0.1):
        for prim in self.primitives.values():
            prim.add_regularizer(name, reg, weight)

    def compute_regularization(self) -> Dict[str, Float[Tensor, ""]]:
        regs = {}
        for p_name, prim in self.primitives.items():
            for r_name, r in prim.compute_regularization().items():
                regs[f"{p_name}_{r_name}"] = r
        return regs

    @nomask
    @torch.no_grad()
    def check_filter(self) -> Optional[Dict[str, Bool[Tensor, "N"]]]:
        filtered = {}
        for name, prim in self.primitives.items():
            f = prim.check_filter()
            if f is not None:
                filtered[name] = f
        if len(filtered) == 0:
            return None
        return filtered

    @nomask
    @torch.no_grad()
    def check_split(self) -> Optional[Dict[Bool[Tensor, "N"]]]:
        split = {}
        for name, prim in self.primitives.items():
            s = prim.check_split()
            if s is not None:
                split[name] = s
        if len(split) == 0:
            return None
        return split
