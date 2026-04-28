from __future__ import annotations
import torch
import torch.nn as nn
import logging
import copy

from contextlib import contextmanager
from typing import Dict, Optional, Any, Sequence, ItemsView, Set
from jaxtyping import Float, Bool, Shaped
from torch import Tensor
from utils.img import ImgUtils
from utils.pytorch import TensorIndex
from rasterizers import Rasterizer, WeightedRasterizer

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

    def filter(self, key: TensorIndex):
        """In-place index selection of batched elements.

        Filters primitive parameters to keep only elements matching key.
        Modifies the primitive in-place.

        Args:
            key: Boolean mask or integer indices to select.

        Notes:
            - Only applies to batched parameters (shape[0] == len(self)).
            - Used by refinement rules to cull primitives.
        """
        updates = {}
        for name, param in self.batched_parameters():
            updates[name] = param[key]
        self.update_parameters(updates)

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

    def __getitem__(self, key: TensorIndex) -> Primitive:
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

    def prepare_for_optimization(
        self,
        target: Float[Tensor, "..."],
        patch_size: Optional[int] = None,
        rasterizer: Optional[Rasterizer] = None,
    ):
        """Prepare buffers for optimization.

        Args:
            target: Target image tensor (C, H, W) or (B, C, H, W).
            patch_size: Optional patch size for patch-based rendering.
            rasterizer: Rasterizer for sample to image output (default WeightedRasterizer)
        """
        with torch.no_grad():
            H, W = target.shape[-2:]
            self._buffer_patches, self._buffer_centers = ImgUtils.get_patches(
                H, W, target.device, patch_size=patch_size
            )
            self._buffer_H = H
            self._buffer_W = W
            self._trained_H = H
            self._trained_W = W
            self._trained_aspect_ratio = W / H if H > 0 else 1.0
        self._buffer_rasterizer = (
            WeightedRasterizer() if rasterizer is None else rasterizer
        )
        self.train()

    def end_optimization(self):
        self.eval()

    @torch.no_grad()
    def patch_mask(
        self, center: Float[Tensor, "N 2"], patch_size: int, H: int, W: int
    ) -> Bool[Tensor, "N"]:
        """Compute mask for valid patches at given centers.

        Args:
            center: Patch center coordinates (N, 2).
            patch_size: Size of patch.

        Returns:
            Bool tensor (N,) indicating which patches are valid.
        """
        return torch.ones((len(self),), dtype=torch.bool, device=self.device)

    @classmethod
    def _sample(
        cls,
        co: Float[Tensor, "N 2"],
        *args,
        **kwargs,
    ) -> Float[Tensor, "N 4"]:
        """Implementation of sampling logic. Subclasses must implement this.

        Args:
            co: Coordinates to sample at (N, 2).

        Returns:
            Sampled RGBA values (N, 4).

        Notes:
            - Assumes non-empty batched parameters (len(self) > 0).
            - Uses masked batched parameters if context is active.
        """
        raise NotImplementedError()

    def sample(
        self, co: Float[Tensor, "N 2"], rasterizer: Rasterizer
    ) -> Float[Tensor, "N 4"]:
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

    def forward(
        self,
        H: int,
        W: int,
        patches: Float[Tensor, "P S 2"],
        centers: Float[Tensor, "P 2"],
        rasterizer: Rasterizer,
        max_batch: Optional[int] = None,
    ) -> Float[Tensor, "B C H W"]:
        """Render primitive to full image.

        Args:
            H: Output height.
            W: Output width.
            patches: Patch coordinates (P, S, 2) where P=num_patches, S=patch_size^2.
            centers: Patch centers (P, 2).
            max_batch: Optional integer value of max number of primitives to sample at once.
                Enables sampling multiple patches at once. Does not partially sample patches.

        Returns:
            Rendered image (B, C, H, W).
        """
        P, S, C = patches.shape
        patch_size = S if P == 1 else int(S**0.5)
        if max_batch is None:
            gen_patches = []
            for patch_idx in range(len(patches)):
                patch = patches[patch_idx]
                mask = self.patch_mask(
                    centers[patch_idx], patch_size=patch_size, H=H, W=W
                )
                with self.masked(mask):
                    gen_patches.append(self.sample(patch, rasterizer))
            return ImgUtils.assemble_patches(torch.stack(gen_patches, dim=0), H, W)
        patch_mask_sums = torch.empty((P,), device=patches.device, dtype=torch.long)
        i = 0
        current_batch_size = 0
        current_batch = None
        current_mask = torch.zeros(
            (len(self),), device=patches.device, dtype=torch.bool
        )
        for patch_idx in range(len(patches)):
            patch = patches[patch_idx]
            mask = self.patch_mask(centers[patch_idx], patch_size=patch_size, H=H, W=W)
            mask_sum = mask.sum()

    def optim_step(self) -> Float[Tensor, "B C H W"]:
        """Run one optimization step and return rendered output.

        Returns:
            Rendered image using cached buffers (B, C, H, W).
        """
        try:
            out = self(
                self._buffer_H,
                self._buffer_W,
                self._buffer_patches,
                self._buffer_centers,
                self._buffer_rasterizer,
            )
        except AttributeError as e:
            raise e from Exception(
                "Method 'prepare_for optimization()' may have not been called before running an optimization step."
            )
        return out

    def rasterize(
        self,
        H: int,
        W: int,
        patch_size: Optional[int] = None,
        rasterizer: Optional[Rasterizer] = None,
    ) -> Float[Tensor, "B H W C"]:
        """Convert rasterized output to displayable image.

        Args:
            H: Output height.
            W: Output width.
            patch_size: Optional patch size.
            rasterizer: Optional rasterizer override. Uses cached rasterizer if None.

        Returns:
            Image tensor (B, H, W, C) in [0, 1] range.
        """
        if rasterizer is None:
            rasterizer = self._buffer_rasterizer
        return ImgUtils.tensor2img(
            self(H, W, patch_size=patch_size, rasterizer=rasterizer),
            normalized=False,
            clamp=True,
        )

    # @torch.no_grad()
    # def cat(self, other: Primitive, weight: float = 0.0):
    #     """Concatenate another primitive in-place.

    #     Appends batched parameters from other to self.

    #     Args:
    #         other: Primitive to concatenate.
    #         weight: importance to other's non batched params. 0 = keep original, 1 = use new.

    #     Notes:
    #         - Only concatenates batched parameters (shape[0] == len(self)).
    #         - Modifies self in-place.
    #     """
    #     np2 = dict(other.batched_parameters())
    #     for name, param in self.batched_parameters():
    #         param = torch.cat([param, np2[name]], dim=0)

    # @classmethod
    # @torch.no_grad()
    # def combine(cls, primitives: Sequence[Primitive]) -> Primitive:
    #     """Combine multiple primitives into one.

    #     Concatenates all primitives in sequence into first primitive.

    #     Args:
    #         primitives: Sequence of primitives to combine.

    #     Returns:
    #         Combined primitive (first element with others concatenated).

    #     Raises:
    #         Exception: If primitives is empty.
    #     """
    #     if not primitives:
    #         raise Exception("Primitives empty or None.")
    #     p = primitives[0]
    #     for other in primitives[1:]:
    #         p.cat(other)
    #     return p

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
