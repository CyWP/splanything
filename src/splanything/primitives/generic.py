from __future__ import annotations
import torch
import torch.nn as nn
import logging
import copy

from contextlib import contextmanager
from typing import Dict, Optional, Any, Sequence, ItemsView, Set
from jaxtyping import Float, Bool, Shaped, Integer
from torch import Tensor
from splanything.utils.pytorch import TensorIndex1D
from splanything.rasterizers import Rasterizer, SampleOutput

_logger = logging.getLogger(__name__)


class Primitive(nn.Module):
    """Base class for trainable geometric image primitives.

    A Primitive represents a learnable geometric representation that can be
    optimized to reconstruct a target image through gradient descent.

    Attributes:
        device: Computed device (torch.device).
        dtype: Computed dtype (torch.dtype).

    Notes:
        - Subclasses must implement `_sample()`, `__len__`, and `parameters` properties.
        - The instance method `sample()` calls `_sample()` with extracted parameters.
        - Uses lazy evaluation via `@lazy_tree` for property caching.
        - Uses tensor shaping [B, C, H, W], but image range remains [0, 1].
        - Supports refinement via `filter()`, `__getitem__()`, `cat()`, `combine()`.
        - Supports implicit masking via `masked()` contextmanager.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._batched_params: Set[str] = set()
        self._stable_params: Set[str] = set()
        self._context_mask: Optional[Bool[Tensor, "N"]] = None

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
        old_mask = self._context_mask
        if old_mask is None:
            self._context_mask = mask
        else:
            if mask.shape == old_mask.shape:
                self._context_mask = old_mask & mask
            elif mask.shape[0] == old_mask.sum().item():
                m = old_mask.clone()
                m[old_mask] = m[old_mask] & mask
            else:
                raise ValueError(
                    "Mask incompatible within current nested mask context."
                )
        try:
            yield
        finally:
            self._context_mask = old_mask

    def __getattribute__(self, name: str) -> Any:
        if name in object.__getattribute__(self, "__dict__").get(
            "_batched_parameters", set()
        ):
            p = object.__getattribute__(self, name)
            msk = object.__getattribute__(self, "_context_mask")
            if msk is not None:
                return p[msk]
            return p
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

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict with class name for serialization."""
        state = super().state_dict()
        state["_class"] = self.__class__.__name__.lower()
        state["_size"] = len(self)
        return state

    def __len__(self) -> int:
        """Number of primitives in this object."""
        if len(self._batched_params) == 0:
            return 0
        return self.__getattr__(next(iter(self._batched_params))).shape[0]

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

    def sample(
        cls,
        co: Float[Tensor, "Nc 2"],
        *args,
        **kwargs,
    ) -> SampleOutput:
        """Implementation of sampling logic. Subclasses must implement this.

        Args:
            co: Coordinates to sample (Nc, 2).

        Returns:
            SampleOutput object.

        Notes:
            - Assumes non-empty batched parameters (len(self) > 0).
            - Uses masked batched parameters if context is active.
        """
        raise NotImplementedError()

    @torch.no_grad()
    def filter(self, key: TensorIndex1D) -> Primitive:
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
            updates[name] = param[key]
        self.update_parameters(updates)
        return self

    def split_params(
        self, idx: TensorIndex1D
    ) -> Dict[str, Float[Tensor, "N_new N_Split ..."]]:
        """
        Returns a dict of params for which the first index of the first dimension
        are the parameters to update for the existing indices being split,
        and subsequent ones are parameters for the newly created primitives.
        """
        raise NotImplementedError()

    @torch.no_grad()
    def split(self, idx: TensorIndex1D) -> Primitive:
        """Split instances at given indices"""
        updates = dict()
        for name, params in self.split_params(idx).items():
            updates[name] = self.__getattr__(name)
            updates[name][idx] = params[0]
            updates[name] = torch.cat(
                [updates[name], *[params[i] for i in range(1, params.shape[0])]], dim=0
            )
        self.update_parameters(updates)
        return self

    def combined_params(
        self, collections: Integer[Tensor, "N_collections N_combined"]
    ) -> Dict[str, Float[Tensor, "N ..."]]:
        """
        Returns dict of edited params, which will be attributed at the indices of the first column of the collections.
        Indices of other columns will be filtered out afterwards.
        """
        raise NotImplementedError

    @torch.no_grad()
    def combine(
        self, collections: Integer[Tensor, "N_collections N_combined"]
    ) -> Primitive:
        collection_size = collections.shape[1]
        if collection_size <= 1:
            raise ValueError(
                "Must at least provide pairs of indices for primitive combination"
            )
        updates = dict()
        for name, param in self.combined_params(collections).items():
            updates[name] = self.__getattr__(name)
            updates[name][collections[:, 0]] = param
        self.update_parameters(updates)
        filter_mask = torch.ones((len(self),), device=self.device, dtype=torch.bool)
        for i in range(1, collection_size):
            filter_mask[collections[:, i]] = False
        return self.filter(filter_mask)

    def forward(
        self, co: Float[Tensor, "Nc 2"], rasterizer: Rasterizer
    ) -> Float[Tensor, "Nc 4"]:
        """Sample primitive values at coordinates.

        Args:
            co: Coordinates to sample at (N, 2).
            rasterizer: Rasterizer Callable to aggregate rgb, a, weights.

        Returns:
            Sampled RGBA values (N, 4).

        Notes:
            - Returns zeros if len(self) == 0.
            - Uses masked batched parameters if context is active.
        """
        if len(self) == 0:
            return torch.zeros(co.shape[0], 4, device=self.device, dtype=self.dtype)
        return rasterizer(self._sample(co))

    @torch.no_grad()
    def cat(self, other: Primitive, weight: float = 0.0):
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

    @classmethod
    @torch.no_grad()
    def combine(cls, primitives: Sequence[Primitive]) -> Primitive:
        """Combine multiple primitives into one.

        Concatenates all primitives in sequence into first primitive.

        Args:
            primitives: Sequence of primitives to combine.

        Returns:
            Combined primitive (first element with others concatenated).

        Raises:
            Exception: If primitives is empty.
        """
        if not primitives:
            raise Exception("Primitives empty or None.")
        p = primitives[0]
        for other in primitives[1:]:
            p.cat(other)
        return p

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
            - Used by refinement rules like GradSplit.
        """
        grads = {}
        for name in self._batched_params:
            param = self.__getattr__(name)
            if param.grad is not None:
                grad = param.grad
                if self._context_mask is not None:
                    grad = grad[self._context_mask]
                grads[name] = grad
        return grads.items()

    def stable_grads(self) -> ItemsView[str, Float[Tensor, "..."]]:
        """Get gradients with batch dimension.

        Returns:
            ItemsView of (name, grad) for batched parameters with gradients.

        Notes:
            Only returns gradients for batched parameters.
            Used by refinement rules like GradSplit.
        """
        grads = {
            name: param.grad
            for name, param in self.stable_parameters()
            if param.grad is not None
        }
        return grads.items()
