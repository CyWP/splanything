"""MultiPrimitive container grouping named child primitives."""
from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from typing import Dict, ItemsView, List, Optional, Any, TYPE_CHECKING

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ..rendering.sample_output import SampleOutput
from .base import Primitive, ParamDef, nomask

if TYPE_CHECKING:
    from ..training.regularizers.base import Regularizer
    from ..training.refinement.base import SplitRule, FilterRule

_logger = logging.getLogger(__name__)


class MultiPrimitive(Primitive):
    """Container holding named child primitives.

    ``len()`` is the sum of the children's (mask-aware) lengths. The
    ``masked`` context slices a single boolean mask spanning all children
    (in ``self.primitives`` order) and enters each child's own ``masked``
    context. Refinement rules, regularizers and sample processors are
    broadcast to every child.

    Attributes:
        primitives (nn.ModuleDict): Child primitives by name.

    Notes:
        - ``forward`` concatenates child ``SampleOutput`` s along the
          primitive axis; ``sample_rgb``/``sample_weights`` do the same.
        - ``check_filter``/``check_split`` delegate to children and return
          per-child masks keyed by child name; like the single-primitive
          variants they are mask-independent (sized to each child's full
          parameter set).
    """

    def __init__(
        self,
        primitives: Dict[str, Primitive],
        filter_rules: Optional[List[FilterRule]] = None,
        split_rules: Optional[List[SplitRule]] = None,
        sample_processors: Optional[List[SampleProcessor]] = None,
        regularizers: Optional[Dict[str, Tuple[Regularizer, float]]] = None,
    ):
        """
        Args:
            primitives: Child primitives by name.
            filter_rules: Filter rules broadcast to every child.
            split_rules: Split rules broadcast to every child.
            sample_processors: Sample processors broadcast to every child.
            regularizers: Name -> (regularizer, weight) broadcast to every child.
        """
        nn.Module.__init__(self)
        self._context_masks: List[Bool[Tensor, "N"]] = []
        self._param_defs: Dict[str, ParamDef] = {}
        self.primitives = nn.ModuleDict(primitives)
        self.register_buffer("_aspect_ratio", torch.tensor(1.0))
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

        The mask is sliced by each child's (mask-aware) length, in
        ``self.primitives`` order, and each child enters its own ``masked``
        context. Supports nested contexts (inner masks compose with outer).

        Args:
            mask: Boolean tensor spanning all children, where True=keep,
                False=remove.

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
        """Unsupported: parameters live in the contained primitives."""
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
        """Child primitive by name."""
        return self.primitives[key]

    @torch.no_grad()
    def _raw_patch_mask(
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

        Returns:
            SampleOutput object.

        Notes:
            - Uses masked batched parameters if context is active.
        """
        return SampleOutput.cat(*[p(co) for p in self.primitives.values()])

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc Np 3"]:
        """Sample per-primitive RGB values at coordinates.

        Delegates to each contained primitive (each mask-aware under an
        active context) and concatenates along the primitive axis.

        Args:
            co: Coordinates to sample at (N, 2).

        Returns:
            RGB values (Nc, Np, 3), ``Np`` the (masked) primitive total.
        """
        return torch.cat(
            [p.sample_rgb(co, **kwargs) for p in self.primitives.values()], dim=1
        )

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc Np"]:
        """Sample per-primitive weights at coordinates.

        Delegates to each contained primitive (each mask-aware under an
        active context) and concatenates along the primitive axis.

        Args:
            co: Coordinates to sample at (N, 2).

        Returns:
            Weights (Nc, Np), ``Np`` the (masked) primitive total.
        """
        return torch.cat(
            [p.sample_weights(co, **kwargs) for p in self.primitives.values()], dim=1
        )

    @nomask
    @torch.no_grad()
    def append(
        self,
        other: MultiPrimitive,
        weight: float | Dict[str, float] = 0.0,
        ignore_exclusive: bool = False,
    ) -> MultiPrimitive:
        """Append another MultiPrimitive's children in-place.

        Children with matching keys are concatenated into the existing
        child; unmatched children are added by name.

        Args:
            other: MultiPrimitive to append.
            weight: Interpolation weight for non-batched (stable) params of
                matching children. 0 = keep original, 1 = use new. A dict
                maps child name to per-child weight.
            ignore_exclusive: Skip unmatched children instead of adding them.

        Returns:
            out: Self, modified in-place.
        """
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
    @torch.no_grad()
    def adjust_to_canvas(self, H: int, W: int) -> MultiPrimitive:
        """Adjust every child to the target aspect ratio (see ``Primitive.adjust_to_canvas``)."""
        for prim in self.primitives.values():
            prim.adjust_to_canvas(H, W)

    @nomask
    def param_groups(self) -> List[Dict[str, nn.Parameter | Any]]:
        """Collect children's optimizer param groups.

        Group names are prefixed with ``<child>$$`` so the optimizer can
        align state with the correct primitive (see OptimizerWrapper).

        Returns:
            out: Children's param groups with prefixed names.
        """
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
        """Unsupported: access parameters directly in the contained primitives."""
        raise NotImplementedError(
            f"This class ('{self.__class__}') is meant as a container for primitives only. Access parameters directly in the contained primitives."
        )

    def stable_parameters(self) -> ItemsView[str, Float[Tensor, "N ..."]]:
        """Unsupported: access parameters directly in the contained primitives."""
        raise NotImplementedError(
            f"This class ('{self.__class__}') is meant as a container for primitives only. Access parameters directly in the contained primitives."
        )

    def named_grads(self) -> ItemsView[str, Float[Tensor, "..."]]:
        """Unsupported: access gradients directly in the contained primitives."""
        raise NotImplementedError(
            f"This class ('{self.__class__}') is meant as a container for primitives only. Access parameters directly in the contained primitives."
        )

    def batched_grads(self) -> ItemsView[str, Float[Tensor, "..."]]:
        """Unsupported: access gradients directly in the contained primitives."""
        raise NotImplementedError(
            f"This class ('{self.__class__}') is meant as a container for primitives only. Access parameters directly in the contained primitives."
        )

    def stable_grads(self) -> ItemsView[str, Float[Tensor, "..."]]:
        """Unsupported: access gradients directly in the contained primitives."""
        raise NotImplementedError(
            f"This class ('{self.__class__}') is meant as a container for primitives only. Access parameters directly in the contained primitives."
        )

    def add_split_rule(self, rule: SplitRule):
        """Register the split rule with every child."""
        for prim in self.primitives.values():
            prim.add_split_rule(rule)

    def add_filter_rule(self, rule: FilterRule):
        """Register the filter rule with every child."""
        for prim in self.primitives.values():
            prim.add_filter_rule(rule)

    def add_sample_processor(self, proc: SampleProcessor):
        """Append the sample processor to every child."""
        for prim in self.primitives.values():
            prim.add_sample_processor(proc)

    def add_regularizer(self, name: str, reg: Regularizer, weight: float = 0.1):
        """Attach the regularizer to every child.

        Args:
            name: Regularizer key (prefixed per child in ``compute_regularization``).
            reg: Regularizer evaluated over each child.
            weight: Loss weight multiplied onto each child's term.
        """
        for prim in self.primitives.values():
            prim.add_regularizer(name, reg, weight)

    def compute_regularization(self) -> Dict[str, Float[Tensor, ""]]:
        """Evaluate every child's regularizers.

        Returns:
            out: Weighted scalar term per ``<child>_<regularizer>`` key.
        """
        regs = {}
        for p_name, prim in self.primitives.items():
            for r_name, r in prim.compute_regularization().items():
                regs[f"{p_name}_{r_name}"] = r
        return regs

    @nomask
    @torch.no_grad()
    def check_filter(self) -> Optional[Dict[str, Bool[Tensor, "N"]]]:
        """Run filter rules on every child.

        Mask-independent: children evaluate on their full parameter sets.

        Returns:
            out: Per-child keep masks keyed by child name. None if nothing
            to cull.
        """
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
        """Run split rules on every child.

        Mask-independent: children evaluate on their full parameter sets.

        Returns:
            out: Per-child split masks keyed by child name. None if nothing
            to split.
        """
        split = {}
        for name, prim in self.primitives.items():
            s = prim.check_split()
            if s is not None:
                split[name] = s
        if len(split) == 0:
            return None
        return split

    @nomask
    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """State dict extended with per-child ParamDef metadata.

        ParamDef fields are stored under ``<child>$$<param>`` keys so a
        reload can rebuild parameter trainability and channels.

        Returns:
            out: State dict including ParamDef metadata.
        """
        state = super().state_dict(*args, **kwargs)
        for name, prim in self.primitives.items():
            for p_name, p_def in prim._param_defs.items():
                state[f"{name}$${p_name}"] = p_def.to_dict()
        return state

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state_dict.pop(prefix + "_class", None)
        state_dict.pop(prefix + "_size", None)
        state_dict.pop(prefix + "_param_defs", None)

        child_pds: Dict[str, Dict[str, ParamDef]] = {n: {} for n in self.primitives}
        for key in list(state_dict.keys()):
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix) :]
            if "$$" not in suffix:
                continue
            child_name, p_name = suffix.split("$$", 1)
            data = state_dict.pop(key, None)
            if data is None:
                continue
            if child_name not in self.primitives:
                continue
            if p_name not in self.primitives[child_name]._param_defs:
                continue
            child_pds[child_name][p_name] = ParamDef(**data)

        for name, pds in child_pds.items():
            if pds:
                self.primitives[name].update_paramdefs(pds)

        for c_name, child in self.primitives.items():
            c_prefix = prefix + "primitives." + c_name + "."
            for p_name, param in list(child._parameters.items()) + list(
                child._buffers.items()
            ):
                if param is None:
                    continue
                key = c_prefix + p_name
                if key not in state_dict:
                    continue
                ckpt = state_dict[key]
                if param.shape == ckpt.shape:
                    continue
                if param.ndim != ckpt.ndim:
                    continue
                if param.shape[1:] != ckpt.shape[1:]:
                    continue
                new_tensor = torch.empty_like(ckpt)
                if p_name in child._parameters:
                    child._parameters[p_name] = nn.Parameter(
                        new_tensor, requires_grad=param.requires_grad
                    )
                else:
                    child._buffers[p_name] = new_tensor

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
