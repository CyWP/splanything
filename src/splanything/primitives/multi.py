from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from typing import Dict, ItemsView, List, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ..utils.pytorch import TensorIndex1D
from .base import Primitive

_logger = logging.getLogger(__name__)


class MultiPrimitive(Primitive):
    def __init__(self, primitives: Dict[str, Primitive] = {}):
        self.primitives = nn.ModuleDict(primitives)

    @contextmanager
    def masked(self, masks: Dict[Bool[Tensor, "N"]]):
        """Context manager for implicit masking of batched parameters.

        When active, accessing batched parameters returns masked versions.
        Supports nested contexts.

        Args:
            masks: Dict of Boolean tensor where True=keep, False=remove.

        Yields:
            None.
        """
        with ExitStack() as stack:
            _ = [
                stack.enter_context(self.primitives[key].masked(masks[key]))
                for key in masks.keys()
            ]
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
                stack.enter_context(prim.cache_properties)
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
        return object.__getattribute__(name)

    def __len__(self) -> int:
        """Number of primitives in this object."""
        return sum([len(prim) for prim in self.primitives.values()])

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
    ) -> Dict[str, Bool[Tensor, "P N"]]:
        """Compute mask for valid patches at given centers.

        Args:
            centers: Patch center coordinates (P, 2).
            patch_sizes: Size of patches (P,).
            H: Image heights (P,).
            W: Image widths (P,).

        Returns:
            Dict[str, Bool tensor (P, N)] indicating which primitives are valid for a given patch.
        """
        return {
            name: prim.patch_mask(centers, patch_sizes, H, W)
            for name, prim in self.primitives.items()
        }

    @torch.no_grad()
    def filter(self, keys: Dict[str, TensorIndex1D]) -> MultiPrimitive:
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

    @torch.no_grad()
    def split(self, idx: Dict[str, Bool[Tensor, "Ns"]]) -> MultiPrimitive:
        """Split instances at given indices"""
        for (
            name,
            i,
        ) in idx.items():
            self.primitives[name].split(i)
        return self

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc Np 3"]:
        return torch.cat(
            [
                prim.sample_rgb(co, **kwargs)
                for prim in self.primitives.values()
                if len(prim) > 0
            ],
            dim=1,
        )

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc Np"]:
        return torch.cat(
            [
                prim.sample_weights(co, **kwargs)
                for prim in self.primitives.values()
                if len(prim) > 0
            ],
            dim=1,
        )

    @torch.no_grad()
    def append(
        self,
        other: MultiPrimitive,
        weight: Union[float, Dict[str, float]] = 0.0,
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

    def param_groups(self) -> List[Dict[str, Union[nn.Parameter, str]]]:
        return [
            {"params": prim.parameters(), "name": name}
            for name, prim in self.primitives.items()
        ]

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

    def add_split_rule(self, rule: "SplitRule"):
        raise NotImplementedError(
            f"This class ('{self.__class__}') is meant as a container for primitives only. Edit rules directly in the contained primitives."
        )

    def add_filter_rule(self, rule: "FilterRule"):
        raise NotImplementedError(
            f"This class ('{self.__class__}') is meant as a container for primitives only. Edit rules directly in the contained primitives."
        )

    def add_finetune_rule(self, rule: "FinetuneRule"):
        raise NotImplementedError(
            f"This class ('{self.__class__}') is meant as a container for primitives only. Edit rules directly in the contained primitives."
        )

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

    @torch.no_grad()
    def check_finetune(self) -> bool:
        return any([prim.check_finetune() for prim in self.primitives.values()])
