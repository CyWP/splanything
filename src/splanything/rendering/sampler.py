"""Patch-grid rendering of primitives to full images."""
from __future__ import annotations

from typing import Iterator, Optional, Tuple, TYPE_CHECKING

import torch
from jaxtyping import Bool, Float
from torch import Tensor

if TYPE_CHECKING:
    from ..primitives import Primitive
from ..utils.img import ImgUtils, Splimage
from .rasterizers.base import Rasterizer
from .rasterizers.weighted import WeightedRasterizer


class Sampler:
    """Renders a primitive over a patch grid at a fixed resolution.

    Splits the (padded) image into square patches, samples the primitive
    per patch, and aggregates SampleOutputs to RGBA with the configured
    rasterizer.

    Attributes:
        H: Image height.
        W: Image width.
        patch_size: Side length of square sampling patches.
        max_batch: Coordinate-times-primitive budget per sample step;
            None samples one patch per step.
        rasterizer: Rasterizer aggregating SampleOutputs to RGBA.
        padding: Padding around the image as (top, bottom, left, right).
        low_vram: If True, ``rasterize`` offloads partial results to CPU.
        device: Device of the patch coordinate tensors.

    Notes:
        - Coordinates are in normalized [0, 1] space.
        - The sampler and the primitive must be on the same device.
    """

    def __init__(
        self,
        H: int,
        W: int,
        patch_size: int = 32,
        max_batch: Optional[int] = None,
        rasterizer: Optional[Rasterizer] = None,
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
        low_vram: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the sampler.

        Args:
            H: Image height.
            W: Image width.
            patch_size: Side length of square sampling patches.
            max_batch: Coordinate-times-primitive budget per sample step;
                None samples one patch per step.
            rasterizer: Rasterizer aggregating SampleOutputs; defaults to
                ``WeightedRasterizer``.
            padding: Padding as (top, bottom, left, right).
            low_vram: If True, ``rasterize`` offloads partial results to CPU.
            device: Device for the patch coordinate tensors.
        """
        self.device = device
        self.H = H
        self.W = W
        self.max_batch = max_batch
        self.low_vram = low_vram
        self.set_patch_size(patch_size, padding)
        self.rasterizer = WeightedRasterizer() if rasterizer is None else rasterizer
        self.padding = padding

    def set_patch_size(
        self, patch_size: int, padding: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ):
        """Set the patch size and rebuild the patch coordinate grids.

        Args:
            patch_size: Side length of square sampling patches.
            padding: Padding as (top, bottom, left, right) applied
                before patching.
        """
        self.patch_size = patch_size
        self.co_patches, self.co_centers = ImgUtils.get_patches(
            self.H, self.W, device=self.device, patch_size=patch_size, padding=padding
        )

    def set_padding(self, padding: Tuple[int, int, int, int]):
        """Change the padding and rebuild the patch coordinate grids.

        Args:
            padding: Padding as (top, bottom, left, right).
        """
        self.set_patch_size(self.patch_size, padding)
        self.padding = padding

    def to(self, device: torch.device) -> Sampler:
        """Move patch coordinate tensors to a device.

        Args:
            device: Target device.

        Returns:
            out: Self.
        """
        self.co_patches = self.co_patches.to(device)
        self.co_centers = self.co_centers.to(device)
        return self

    @property
    def padded_dims(self) -> Tuple[int, int]:
        """Image dimensions including padding.

        Returns:
            Tuple (H, W) with padding added.
        """
        t, b, l, r = self.padding
        return self.H + t + b, self.W + l + r

    @property
    def num_patches(self) -> int:
        """Number of sampling patches."""
        if self.patch_size is None:
            return 1
        return self.co_patches.shape[0]

    def samples(
        self, p: Primitive, verbose: bool = False
    ) -> Iterator[Tuple[Float[Tensor, "S C"], Float[Tensor, "S 2"]]]:
        """Yield rasterized sample batches with their coordinates.

        Args:
            p: Primitive to sample.
            verbose: If True, show a rich progress bar.

        Yields:
            Tuple of (RGBA tensor (S, 4), coordinates (S, 2)) per step,
            where S is the number of coordinates in the current patch,
            patch batch, or chunk.

        Notes:
            - Raises ValueError if the primitive is on a different device.
            - Per-patch primitive masks limit the active primitive set.
            - With ``max_batch`` set, multiple patches are batched per
              step; a single patch exceeding the budget is chunked by
              its pixels.
        """
        if p.device != self.device:
            raise ValueError(
                f"Sampler and primitive must be on same device. Currently: {self.device}, {p.device}."
            )
        patches = self.co_patches
        centers = self.co_centers
        rasterizer = self.rasterizer
        max_batch = self.max_batch
        P, S, C = patches.shape
        patch_sizes = torch.full(
            (P,), S if P == 1 else int(S**0.5), dtype=torch.long, device=patches.device
        )
        H = torch.full((P,), self.H, dtype=torch.long, device=patches.device)
        W = torch.full((P,), self.W, dtype=torch.long, device=patches.device)
        patch_masks = p.patch_mask(centers, patch_sizes, H, W)  # [P, N]

        def _compute_patch(batch: Float[Tensor, "S 2"], mask: Bool[Tensor, "N"]):
            nonlocal p
            nonlocal rasterizer
            with p.masked(mask):
                return rasterizer(p(batch))

        if verbose:
            from rich.progress import Progress

            progress = Progress()
            task = progress.add_task("[green]Sampling...", total=P)
            progress.start()

        # If there is no max batch size, just compute per patch.
        if max_batch is None:
            for i in range(P):
                if verbose:
                    progress.update(task, completed=i)
                yield _compute_patch(patches[i], patch_masks[i]), patches[i]
            if verbose:
                progress.stop()
            return
        patch_mask_sums = patch_masks.sum(dim=1)  # [P,]
        i = 0
        mask = torch.empty((len(p),), dtype=torch.bool, device=patches.device)
        while i < P:
            if verbose:
                progress.update(task, completed=i)
            acc_patches = []
            co_size = 0
            mask.zero_()
            while (
                i < P
                and (mask.sum() + patch_mask_sums[i]).item() * (co_size + S) < max_batch
            ):
                mask = mask | patch_masks[i]
                co_size += S
                acc_patches.append(patches[i])
                i += 1
            if len(acc_patches) == 0:
                # Single patch too large: chunk its pixels.
                chunk = max(1, max_batch // max(patch_mask_sums[i].item(), 1))
                mask = patch_masks[i]
                s = 0
                N = patches[i].shape[0]
                while s < N:
                    e = min(s + chunk, N)
                    yield _compute_patch(patches[i][s:e], mask), patches[i][s:e]
                    s = e
                i += 1
            else:
                batch_co = torch.cat(acc_patches, dim=0)
                yield _compute_patch(batch_co, mask), batch_co
        if verbose:
            progress.stop()

    def rasterize(
        self,
        p: Primitive,
        max_batch: Optional[int] = None,
        low_vram: Optional[bool] = None,
        verbose: bool = False,
    ) -> Float[Tensor, "B C H W"]:
        """Rasterize the primitive to a full (padded) image.

        Args:
            p: Primitive to render.
            max_batch: Optional one-shot override of ``self.max_batch``,
                restored afterwards.
            low_vram: Optional one-shot override of ``self.low_vram``.
            verbose: If True, show a rich progress bar.

        Returns:
            RGBA tensor (B, C, H', W') covering the padded image, where
            (H', W') = ``padded_dims``.

        Raises:
            ValueError: If the primitive is on a different device.
        """
        if p.device != self.device:
            raise ValueError(
                f"Sampler and primitive must be on same device. Currently: {self.device}, {p.device}."
            )
        lv = self.low_vram if low_vram is None else low_vram
        if max_batch is not None:
            saved = self.max_batch
            self.max_batch = max_batch
        else:
            saved = None
        try:
            P, S, _ = self.co_patches.shape
            gen = []
            for sample, _ in self.samples(p, verbose=verbose):
                if lv:
                    sample = sample.cpu()
                gen.append(sample)
            patch_gen = torch.cat(gen, dim=0).reshape(P, S, 4)
            H, W = self.padded_dims
            return ImgUtils.assemble_patches(patch_gen, H, W)
        finally:
            if saved is not None:
                self.max_batch = saved

    def render(
        self,
        p: Primitive,
        max_batch: Optional[int] = None,
        low_vram: Optional[bool] = None,
        verbose: bool = False,
    ) -> Splimage:
        """Rasterize the primitive and wrap the result in a Splimage.

        Args:
            p: Primitive to render.
            max_batch: Optional one-shot override of ``self.max_batch``.
            low_vram: Optional one-shot override of ``self.low_vram``.
            verbose: If True, show a rich progress bar.

        Returns:
            Splimage of the rendered image.
        """
        return Splimage(
            self.rasterize(p, max_batch=max_batch, low_vram=low_vram, verbose=verbose)
        )
