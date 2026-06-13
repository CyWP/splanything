from __future__ import annotations
import torch
import torch.nn as nn
import logging
import copy

from contextlib import contextmanager
from typing import Dict, Optional, Any, Sequence, ItemsView, Set, Literal
from jaxtyping import Float, Bool, Shaped, Integer
from torch import Tensor
from utils.img import ImgUtils
from utils.pytorch import TensorIndex
from rasterizers import Rasterizer, SampleOutput

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

    @torch.no_grad()
    def patch_mask(
        self,
        centers: Float[Tensor, "P 2"],
        patch_sizes: Integer[Tensor, "P 2"],
        H: Integer[Tensor, "P"],
        W: Integer[Tensor, "P"],
    ) -> Bool[Tensor, "P N"]:
        """Compute mask for valid patches at given centers.

        Args:
            centers: Patch center coordinates (P, 2).
            patch_sizes: Size of patches (P, 2).

        Returns:
            Bool tensor (P, N) indicating which primitives are valid for a given patch.
        """
        return torch.ones((len(self),), dtype=torch.bool, device=self.device)

    @classmethod
    def _sample(
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

    def sample(
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

    def forward(
        self,
        H: int,
        W: int,
        patches: Float[Tensor, "P S 2"],
        centers: Float[Tensor, "P 2"],
        rasterizer: Rasterizer,
        max_batch: int = 100,
        low_vram: bool = False,
        format: Literal["tensor", "raster", "image"] = "tensor",
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
        patch_sizes = torch.full(
            (P,), S if P == 1 else int(S**0.5), dtype=torch.long, device=patches.device
        )
        H = torch.full((P,), H, dtype=torch.long, device=patches.device)
        W = torch.full((P,), W, dtype=torch.long, device=patches.device)
        patch_masks = self.patch_mask(centers, patch_sizes, H, W)  # [P, N]
        patch_mask_sums = patch_masks.sum(dim=1)  # [P,]
        gen = []
        i = 0
        mask = torch.empty((len(self),), dtype=torch.bool, device=patches.device)
        while i < P:
            acc_patches = []
            batch_size = 0
            mask.zero_()
            while (
                i < P
                and (len(acc_patches) + 1) * batch_size + patch_mask_sums[i] < max_batch
            ):
                mask = mask | patch_masks[i]
                batch_size = mask.sum()
                acc_patches.append(patches[i])
                i += 1
            # Only True if a single patch must be done in multiple passes
            if len(acc_patches) == 0:
                b_patches = torch.chunk(
                    patches[i], patch_mask_sums[i] // max_batch, dim=0
                )
                mask = patch_masks[i]
                for b in b_patches:
                    with self.masked(mask):
                        sample = self.sample(b, rasterizer)
                        if low_vram:
                            sample = sample.cpu()
                        gen.append(sample)
                i += 1
            else:
                with self.masked(mask):
                    co = torch.cat(acc_patches, dim=0)
                    sample = self.sample(co, rasterizer)
                    if low_vram:
                        sample = sample.cpu()
                    gen.append(sample)
        patch_gen = torch.cat(gen, dim=0).reshape(P, S, 4)
        out = ImgUtils.assemble_patches(patch_gen, H[0], W[0])
        if format == "tensor":
            return out
        elif format == "raster":
            return ImgUtils.tensor2img(out, normalized=False, clamp=True)
        elif format == "image":
            return ImgUtils.tensor2pil(out, normalized=False)
        else:
            raise ValueError(
                f"'{format}' is an invalid output format. Must be amongst 'tensor', 'raster', or 'image'."
            )

    def rasterize(
        self,
        H: int,
        W: int,
        rasterizer: Rasterizer,
        patch_size: Optional[int] = None,
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
        return ImgUtils.tensor2img(
            self(H, W, patch_size=patch_size, rasterizer=rasterizer),
            normalized=False,
            clamp=True,
        )

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
