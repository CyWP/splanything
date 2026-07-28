from __future__ import annotations

import copy
import functools
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, ItemsView, List, Optional, Set, TYPE_CHECKING
from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer, Shaped
from torch import Tensor

from ..rendering.sample_output import SampleOutput
from ..rendering.processors.base import SampleProcessor
from ..utils.pytorch import TensorIndex1D
from .initializers.base import Initializer
from .splitters.base import Splitter

if TYPE_CHECKING:
    from ..training.regularizers.base import Regularizer
    from ..training.refinement.base import SplitRule, FilterRule

_logger = logging.getLogger(__name__)


class cached_property:
    """Descriptor for context-aware cached properties.

    Behaves like a normal @property outside a cache_properties() context
    (recomputed on every access). Inside an active cache_properties()
    context on the owning Primitive instance, the value is computed once
    and returned from _property_cache on subsequent accesses.

    Uses object.__getattribute__ internally to avoid recursion with
    Primitive.__getattribute__.
    """

    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, instance, owner):
        if instance is None:
            return self
        inst_dict = object.__getattribute__(instance, "__dict__")
        cache = inst_dict.get("_property_cache")
        active = inst_dict.get("_cache_active", False)
        if active and cache is not None and self.name in cache:
            return cache[self.name]
        value = self.func(instance)
        if active and cache is not None:
            cache[self.name] = value
        return value


class nomask:
    """Descriptor decorator running a Primitive method inside an ``unmasked`` context.

    Behaves like the wrapped method outside an ``unmasked()`` context.
    When accessed on a Primitive instance, returns a bound wrapper that
    enters ``instance.unmasked()`` before invoking the underlying method
    and restores the prior mask stack afterward.
    """

    def __init__(self, method):
        self.method = method
        self.name = method.__name__
        self.__doc__ = method.__doc__

    def __get__(self, instance, owner):
        if instance is None:
            return self
        method = self.method

        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            with instance.unmasked():
                return method(instance, *args, **kwargs)

        return wrapper


@dataclass
class ParamDef:
    batched: bool
    trainable: bool
    channels: Optional[Tuple[int]] = None
    lr_mod: float = 1.0


class Primitive(nn.Module):
    def __init__(
        self,
        size: int = 1,
        initializers: Optional[Dict[str, Initializer] | Initializer] = None,
        splitters: Optional[Dict[str, Splitter] | Splitter] = None,
        param_defs: Optional[Dict[str, ParamDef]] = None,
        filter_rules: Optional[List[FilterRule]] = None,
        split_rules: Optional[List[SplitRule]] = None,
        sample_processors: Optional[List[SampleProcessor]] = None,
        regularizers: Optional[Dict[str, Tuple[Regularizer, float]]] = None,
    ):
        super().__init__()
        if size < 1:
            raise ValueError("Size cannot be null or negative.")
        self.size = size
        self._batched_params: Set[str] = set()
        self._stable_params: Set[str] = set()
        self._context_masks: List[Bool[Tensor, "N"]] = []
        self._property_cache: Dict[str, Any] = {}
        self._cache_active: bool = False
        self._split_rules = []
        self._filter_rules = []
        self._lr_modifiers = {}
        self._sample_processors = []
        self._regularizers = {}
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
        self._register_initializers(initializers)
        self._register_params(self._initializers, param_defs)
        self._register_splitters(splitters)

    @property
    def default_params(self) -> Dict[str, ParamDef]:
        raise NotImplementedError()

    @property
    def default_initializers(self) -> Dict[str, Initializer] | Initializer:
        return {}

    @property
    def default_splitters(self) -> Dict[str, Splitter] | Splitter:
        return {}

    def _register_initializers(
        self, overrides: Optional[Dict[str, Initializer] | Initializer] = None
    ):
        if isinstance(overrides, Initializer):
            self._initializers = {
                name: overrides for name in self.default_params.keys()
            }
            return
        o_i = {} if overrides is None else overrides
        if isinstance(self.default_initializers, Initializer):
            i = self.default_initializers
            inits = {name: i for name in self.default_params.keys()}
            self._initializers = {**inits, **o_i}
            return
        i = Initializer()
        inits = {name: i for name in self.default_params.keys()}
        self._initializers = {
            **inits,
            **self.default_initializers,
            **o_i,
        }

    def _register_splitters(
        self, overrides: Optional[Dict[str, Splitter] | Splitter] = None
    ):
        if isinstance(overrides, Splitter):
            self._splitters = {name: overrides for name in self._default_params.keys()}
            return
        o_s = {} if overrides is None else overrides
        if isinstance(self.default_splitters, Splitter):
            s = self.default_splitters
            splits = {name: s for name in self.default_params.keys()}
            self._splitters = {**splits, **o_s}
            return
        s = Splitter()
        splits = {name: s for name in self.default_params.keys()}
        self._splitters = {**splits, **self.default_splitters, **o_s}

    def _register_params(
        self,
        initializers: Dict[str, Initializer],
        overrides: Optional[Dict[str, ParamDef]] = None,
    ):
        params = (
            self.default_params
            if overrides is None
            else {**self.default_params, **overrides}
        )
        for name, p_def in params.items():
            self.add_parameter(
                name,
                initializers[name](
                    name, self.size if p_def.batched else 0, p_def.channels
                ),
                batched=p_def.batched,
                trainable=p_def.trainable,
                lr_modifier=p_def.lr_mod,
            )

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        raise RuntimeError("Module has no parameters or buffers")

    @property
    def dtype(self) -> torch.dtype:
        for p in self.parameters():
            return p.dtype
        for b in self.buffers():
            return b.dtype
        raise RuntimeError("Module has no parameters or buffers")

    @contextmanager
    def masked(self, mask: Bool[Tensor, "N"]):
        """Context manager for implicit masking of batched parameters.

        When active, accessing batched parameters returns masked versions.
        Supports nested contexts.

        Args:
            mask: Boolean tensor where True=keep, False=remove.

        Yields:
            None.
        """
        old_mask = self._context_masks[-1] if len(self._context_masks) > 0 else None
        if old_mask is None:
            self._context_masks.append(mask)
        else:
            if mask.shape == old_mask.shape:
                self._context_masks.append(old_mask & mask)
            elif mask.shape[0] == old_mask.sum().item():
                m = old_mask.clone()
                m[old_mask] = m[old_mask] & mask
                self._context_masks.append(m)
            else:
                raise ValueError(
                    "Mask incompatible within current nested mask context."
                )
        try:
            yield
        finally:
            self._context_masks.pop()

    @contextmanager
    def unmasked(self):
        """Temporarily disables masking

        Yields:
            None.
        """
        old_masks = self._context_masks
        self._context_masks = []
        try:
            yield
        finally:
            self._context_masks = old_masks

    @contextmanager
    def cache_properties(self):
        """Context manager enabling cached_property caching.

        While active, @cached_property-decorated properties compute once
        and return cached values on subsequent accesses. The cache is
        cleared on exit. Supports nesting (outermost exit clears).

        Yields:
            None.
        """
        old_active = self._cache_active
        self._cache_active = True
        try:
            yield
        finally:
            self._cache_active = old_active
            self._property_cache.clear()

    def clear_cache(self):
        """Manually clear the property cache."""
        self._property_cache.clear()

    def _batched_param(self, name: str) -> Any:
        """Fetch a batched parameter, applying the active mask if any.

        Looks up ``name`` in ``_parameters`` / ``_buffers`` (where nn.Module
        stores registered params/buffers) and returns the masked slice when
        ``_context_mask`` is set, otherwise the raw tensor. Used by both
        ``__getattribute__`` and ``__len__`` so length and access agree.

        Args:
            name: Name of a registered batched parameter.

        Returns:
            The (possibly masked) parameter tensor.

        Raises:
            AttributeError: If ``name`` is not a registered batched
                parameter.
        """
        params = object.__getattribute__(self, "_parameters")
        buffers = object.__getattribute__(self, "_buffers")
        if name in params:
            p = params[name]
        elif name in buffers:
            p = buffers[name]
        else:
            raise AttributeError(
                f"{type(self).__name__!s} has no batched parameter {name!r}"
            )
        msk = self._context_masks[-1] if len(self._context_masks) > 0 else None
        if msk is None:
            return p
        if msk.shape[0] < p.shape[0]:
            # Only happens if a split was done in masked context
            msk = torch.cat(
                [
                    msk,
                    torch.ones(
                        p.shape[0] - msk.shape[0],
                        device=msk.device,
                        dtype=torch.bool,
                    ),
                ]
            )
        return p[msk]

    def __getattribute__(self, name: str) -> Any:
        batched = object.__getattribute__(self, "__dict__").get(
            "_batched_params", set()
        )
        if name in batched:
            return object.__getattribute__(self, "_batched_param")(name)
        return object.__getattribute__(self, name)

    def copy(self) -> Primitive:
        """Create a copy of this primitive.

        Returns:
            New Primitive instance with same state.
        """
        return copy.deepcopy(self)

    def add_parameter(
        self,
        name: str,
        param: Shaped[Tensor, ""],
        batched: bool = True,
        trainable: bool = True,
        lr_modifier: float = 1.0,
    ):
        if name in self._batched_params | self._stable_params:
            raise KeyError(
                f"Cannot register different parameters with same name: {name}."
            )
        if batched:
            if len(self._batched_params) == 0:
                self.size = param.shape[0]
            elif self.size != param.shape[0]:
                raise Exception(
                    f"Registered batched parameters must all have the same shape in dim 0: {self.size}."
                )
            self._batched_params.add(name)
        else:
            self._stable_params.add(name)
        if trainable:
            self._lr_modifiers[name] = lr_modifier
            self.register_parameter(name, nn.Parameter(param))
        else:
            self.register_buffer(name, param)

    def update_parameters(self, updates: Dict[str, Tensor]):
        """Update parameters with new tensors.

        Replaces parameters with new tensors. After update, validates that
        all batched parameters have the same size in their first dimension.

        Args:
            updates: Dict mapping parameter names to new tensors.

        Raises:
            KeyError: If a parameter name is not found.
            ValueError: If batched parameters have inconsistent first-dimension sizes.
        """
        for name, tensor in updates.items():
            if name not in self._batched_params and name not in self._stable_params:
                raise KeyError(f"Unknown parameter: {name}")
            if name in self._batched_params:
                param = nn.Parameter(tensor)
                # self.register_parameter(name, param)
                self.__setattr__(name, param)
            else:
                self.__setattr__(name, param)  # self.register_buffer(name, tensor)

        self._validate_batched_sizes()

    @nomask
    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Return state dict with class name for serialization."""
        state = super().state_dict(*args, **kwargs)
        state["_class"] = self.__class__.__name__.lower()
        state["_size"] = len(self)
        return state

    def __len__(self) -> int:
        """Number of primitives in this object.

        Under an active ``masked`` context, returns the masked length so
        callers that mix ``len(self)`` with batched-parameter accesses see
        consistent shapes.
        """
        batched = object.__getattribute__(self, "__dict__").get(
            "_batched_params", set()
        )
        if len(batched) == 0:
            return 0
        name = next(iter(batched))
        return object.__getattribute__(self, "_batched_param")(name).shape[0]

    @nomask
    def _validate_batched_sizes(self):
        """Ensure all batched parameters have the same first-dimension size."""
        sizes = set()
        for name in self._batched_params:
            param = self.__getattr__(name)
            sizes.add(param.shape[0])
        if len(sizes) > 1:
            raise ValueError(
                f"Batched parameters have inconsistent sizes: {sizes}. "
                f"All batched parameters must have the same size in dim 0."
            )

    def __getitem__(self, key: TensorIndex1D) -> Primitive:
        """Index retrieval returning new object.

        Creates a new primitive containing only elements matching key.
        Does not modify original.

        Args:
            key: Boolean mask or integer indices to select.

        Returns:
            New Primitive with selected elements.

        Notes:
            - Only applies to batched parameters (shape[0] == len(self)).
            - Non-batched parameters are copied as-is.
        """
        new = self.copy()

        old_state = self.state_dict()
        new_state = {}

        for name, param in old_state.items():
            if name in self._batched_params:
                new_state[name] = param[key].clone()
            else:
                new_state[name] = param.clone()

        new.load_state_dict(new_state)
        return new

    @torch.no_grad()
    def patch_mask(
        self,
        centers: Float[Tensor, "P 2"],
        patch_sizes: Integer[Tensor, "P"],
        H: Integer[Tensor, "P"],
        W: Integer[Tensor, "P"],
    ) -> Bool[Tensor, "P N"]:
        """Compute mask for valid patches at given centers.

        Args:
            centers: Patch center coordinates (P, 2).
            patch_sizes: Size of patches (P,).
            H: Image heights (P,).
            W: Image widths (P,).

        Returns:
            Bool tensor (P, N) indicating which primitives are valid for a given patch.
        """
        raise NotImplementedError()

    @nomask
    @torch.no_grad()
    def filter(self, idx: Bool[Tensor, "N"]) -> Primitive:
        """In-place index selection of batched elements.

        Filters primitive parameters to keep only elements matching key.
        Modifies the primitive in-place.

        Args:
            key: Boolean mask or integer indices to select.

        Notes:
            - Only applies to batched parameters (shape[0] == len(self)).
            - Used by refinement rules to cull primitives.
        """
        updates = dict()
        for name, param in self.batched_parameters():
            updates[name] = param[idx]
        self.update_parameters(updates)
        prev_msk = idx
        new_context = []
        for mask in self._context_masks[::-1]:
            if mask.shape[0] == prev_msk.shape[0]:
                new_context.insert(0, mask[prev_msk])
            elif mask.sum().item() == prev_msk.shape[0]:
                cull_idx = mask.nonzero()[prev_msk]
                new_prev = mask.clone()
                new_prev[cull_idx] = False
                keep_mask = torch.ones_like(mask)
                keep_mask[cull_idx] = False
                new_context.insert(0, mask[keep_mask])
                prev_msk = new_prev
        self._context_masks = new_context
        return self

    @nomask
    @torch.no_grad()
    def split(self, idx: Bool[Tensor, "N"]) -> Primitive:
        """Split instances at given indices"""
        updates = dict()
        for name, param in self.batched_parameters():
            updates[name] = self._splitters[name](self, name, param, idx)
        self.update_parameters(updates)
        return self

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc Np 3"]:
        raise NotImplementedError()

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc Np"]:
        raise NotImplementedError()

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
        if len(self) == 0:
            return SampleOutput(
                rgb=torch.zeros(
                    (co.shape[0], 1, 3), device=self.device, dtype=co.dtype
                ),
                weights=torch.zeros(
                    (co.shape[0], 1), device=self.device, dtype=co.dtype
                ),
                co=co,
            )
            return torch.zeros(co.shape[0], 4, device=self.device, dtype=self.dtype)
        with self.cache_properties():
            rgb = self.sample_rgb(co)
            weights = self.sample_weights(co)
        sample = SampleOutput(rgb=rgb, weights=weights, co=co)
        for proc in self._sample_processors:
            sample = proc(sample, self)
        return sample

    @nomask
    @torch.no_grad()
    def append(self, other: Primitive, weight: float = 0.0) -> Primitive:
        """Concatenate another primitive in-place.

        Appends batched parameters from other to self.

        Args:
            other: Primitive to concatenate.
            weight: importance to other's non batched params. 0 = keep original, 1 = use new.

        Notes:
            - Only concatenates batched parameters (shape[0] == len(self)).
            - Modifies self in-place.
        """
        np2 = dict(other.batched_parameters())
        for name, param in self.batched_parameters():
            self.__setattr__(name, torch.cat([param, np2[name]], dim=0))
        np2 = dict(other.stable_parameters())
        for name, param in self.stable_parameters():
            self.__setattr__(name, weight * np2[name] + (1 - weight) * param)
        return self

    @classmethod
    def cat(cls, primitives: List[Primitive]) -> Primitive:
        prim = cls()
        for p in primitives:
            prim.append(p)
        return prim

    @nomask
    def param_groups(self) -> List[Dict[str, nn.Parameter]]:
        groups = []
        params_dict = {
            **dict(self.batched_parameters()),
            **dict(self.stable_parameters()),
        }
        for name, param in params_dict.items():
            groups.append(
                {
                    "params": param,
                    "lr_modifier": self._lr_modifiers.get(name, 1.0),
                    "name": name,
                }
            )
        return groups

    def batched_parameters(self) -> ItemsView[str, Float[Tensor, "N ..."]]:
        """Get parameters with batch dimension.

        Returns:
            ItemsView of (name, param) for batched parameters.
        """
        return {name: self.__getattr__(name) for name in self._batched_params}.items()

    def stable_parameters(self) -> ItemsView[str, Float[Tensor, "N ..."]]:
        """Get parameters with batch dimension.

        Returns:
            ItemsView of (name, param) for batched parameters.
        """
        return {name: self.__getattr__(name) for name in self._stable_params}.items()

    def named_grads(self) -> ItemsView[str, Float[Tensor, "..."]]:
        """Get named gradients.

        Returns:
            ItemsView of (name, grad) for parameters with gradients.
        """
        grad = {
            name: param.grad
            for name, param in self.named_parameters()
            if param.grad is not None
        }
        return grad.items()

    def batched_grads(self) -> ItemsView[str, Float[Tensor, "..."]]:
        """Get gradients with batch dimension.

        Returns:
            ItemsView of (name, grad) for batched parameters with gradients.

        Notes:
            - Only returns gradients for batched parameters.
            - Uses masked gradients if context is active.
            - Used by refinement rules.
        """
        grads = {}
        for name in self._batched_params:
            param = self.__getattr__(name)
            if param.grad is not None:
                grad = param.grad
                if len(self._context_masks) > 0:
                    grad = grad[self._context_masks[-1]]
                grads[name] = grad
        return grads.items()

    def stable_grads(self) -> ItemsView[str, Float[Tensor, "..."]]:
        """Get gradients with batch dimension.

        Returns:
            ItemsView of (name, grad) for batched parameters with gradients.

        Notes:
            Only returns gradients for batched parameters.
            Used by refinement rules.
        """
        grads = {
            name: param.grad
            for name, param in self.stable_parameters()
            if param.grad is not None
        }
        return grads.items()

    def add_split_rule(self, rule: SplitRule):
        """Register a split rule with this primitive.

        Args:
            rule: Split rule to apply on ``check_split``.
        """
        self._split_rules.append(rule)
        rule.register(self)

    def add_filter_rule(self, rule: FilterRule):
        """Register a filter rule with this primitive.

        Args:
            rule: Filter rule to apply on ``check_filter``.
        """
        self._filter_rules.append(rule)
        rule.register(self)

    def add_sample_processor(self, processor: SampleProcessor):
        self._sample_processors.append(processor)

    def add_regularizer(self, name: str, reg: Regularizer, weight: float = 0.1):
        self._regularizers[name] = (reg, weight)

    def compute_regularization(self) -> Dict[str, Float[Tensor, ""]]:
        regs = {}
        for name, (regularizer, weight) in self._regularizers.items():
            regs[name] = weight * regularizer(self)
        return regs

    @nomask
    @torch.no_grad()
    def check_filter(self) -> Optional[Bool[Tensor, "N"]]:
        if len(self._filter_rules) == 0:
            return None
        combined_filter = torch.ones(len(self), dtype=torch.bool, device=self.device)
        for rule in self._filter_rules:
            mask = rule(self)
            if mask is not None:
                combined_filter &= mask
        if (~combined_filter).any():
            _logger.info(
                f"Combined filter: {(~combined_filter).sum().item()} marked for filtering."
            )
            return combined_filter
        return None

    @nomask
    @torch.no_grad()
    def check_split(self) -> Optional[Bool[Tensor, "N"]]:
        if len(self._split_rules) == 0:
            return None
        combined_split = torch.zeros(len(self), dtype=torch.bool, device=self.device)
        for rule in self._split_rules:
            split = rule(self)
            if split is not None:
                combined_split |= split
        if combined_split.any():
            _logger.info(
                f"Combined split: {combined_split.sum().item()} marked for splitting."
            )
            return combined_split
        return None

    @nomask
    def load(self, path: str | Path):
        self.load_state_dict(torch.load(path, weights_only=False), strict=False)

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
        # Resize parameters whose first dimension (number of primitives)
        # differs from the checkpoint. Any other shape mismatch is treated
        # as an error.
        for name, param in self._parameters.items():
            if param is None:
                continue

            key = prefix + name
            if key not in state_dict:
                continue

            ckpt = state_dict[key]

            # Shapes already match.
            if param.shape == ckpt.shape:
                continue

            # Rank must match.
            if param.ndim != ckpt.ndim:
                error_msgs.append(
                    f"size mismatch for {key}: copying a param with shape "
                    f"{tuple(ckpt.shape)} from checkpoint, "
                    f"the shape in current model is {tuple(param.shape)}."
                )
                continue

            # Only the first dimension may differ.
            if param.shape[1:] != ckpt.shape[1:]:
                error_msgs.append(
                    f"size mismatch for {key}: copying a param with shape "
                    f"{tuple(ckpt.shape)} from checkpoint, "
                    f"the shape in current model is {tuple(param.shape)}."
                )
                continue

            # Replace the parameter with one of the correct size.
            self._parameters[name] = nn.Parameter(
                torch.empty_like(ckpt),
                requires_grad=param.requires_grad,
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
