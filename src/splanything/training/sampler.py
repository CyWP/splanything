"""Training-time sampler with target patch extraction and per-pixel subsampling."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ..primitives.base import Primitive
from ..rendering.rasterizers.base import Rasterizer
from ..rendering.rasterizers.weighted import WeightedRasterizer
from ..rendering.sampler import Sampler
from ..utils.img import Splimage, ImgUtils


class TrainSampler(Sampler):
    """Sampler for training: subsamples per-pixel from a sampling map.

    Yields ``(output, target, batch_co)`` triples per batch. Each pixel in
    each patch is included independently with probability drawn from
    ``sampling_patches`` (per-patch, per-pixel). Aggregated batches respect
    ``max_batch`` (≈ primitive_count × coordinate_count budget).

    Attributes:
        sampling_map: Probability source map (B, 1, H, W) in [0, 1].
        sampling_patches: Per-patch per-pixel probabilities (P, S).
        max_batch: Compute budget for each yielded batch.
        epoch_size: Optional rescale of sampling_patches to a fixed expected
            number of selected pixels per epoch.
        low_vram: If True, intermediate render outputs are moved to CPU.

    Notes:
        - ``target_patches`` is batch-collapsed to ``(P, S, C)`` to match
          ``co_patches`` ``(P, S, 2)`` from the parent layout.
        - ``device`` is taken from the target tensor; the parent constructor
          is not called.
        - ``samples()`` does not mutate ``co_patches`` / ``co_centers``.
    """

    def __init__(
        self,
        target: Optional[Splimage] = None,
        H: Optional[int] = None,
        W: Optional[int] = None,
        patch_size: Optional[int] = None,
        max_batch: Optional[int] = None,
        sampling_map: Optional[Splimage] = None,
        rasterizer: Optional[Rasterizer] = None,
        low_vram: bool = False,
        epoch_size: Optional[int] = None,
        jitter_coords: bool = False,
    ):
        """Configure the training sampler.

        Does not call the parent constructor; state is set up directly
        from ``target`` when provided.

        Args:
            target: Target image (B, C, H, W); enables patch extraction
                when provided.
            H: Canvas height used when ``target`` is None.
            W: Canvas width used when ``target`` is None.
            patch_size: Side length of the extracted patches (S = patch_size²).
            max_batch: Compute budget per yielded batch
                (mask_size × co_size ≤ max_batch).
            sampling_map: Optional probability map (B, 1, H, W) in [0, 1]
                for per-pixel Bernoulli subsampling.
            rasterizer: Rasterizer aggregating per-primitive samples into
                RGBA; defaults to ``WeightedRasterizer``.
            low_vram: If True, intermediate render outputs are moved to CPU.
            epoch_size: Optional expected number of sampled pixels per
                epoch; rescales the sampling map probabilities.
            jitter_coords: If True, jitter patch coordinates and re-extract
                target patches on every ``samples()`` call.

        Raises:
            ValueError: If neither ``target`` nor both ``H`` and ``W`` are set.
        """
        if target is None and (H is None or W is None):
            raise ValueError(
                "Either 'target' or generation dimensions ('H', 'W') must be set upon initialization."
            )
        self.sampling_map = sampling_map
        self.max_batch = max_batch
        self.epoch_size = epoch_size
        self.low_vram = low_vram
        self.rasterizer = WeightedRasterizer() if rasterizer is None else rasterizer
        self.H = H
        self.W = W
        self.target_img = target
        self.jitter_coords = jitter_coords
        if target is not None:
            self.set_target(target, patch_size=patch_size)

    def set_target(self, target: Splimage, patch_size: Optional[int] = None):
        """Re-initialise target image and patch grids.

        Args:
            target: New target image (B, C, H, W).
            patch_size: Optional new patch size; keeps the old one if None.
        """
        self.device = target.device
        self.H, self.W = target.shape[-2:]
        if patch_size is not None:
            self.patch_size = patch_size
        self.target_img = target
        self.target_patches = target.extract_image_patches(
            patch_size, padding_mode="replicate"
        ).squeeze(0)  # (B, P, S, C) -> (P, S, C)
        self.co_patches, self.co_centers = ImgUtils.get_patches(
            self.H, self.W, device=self.device, patch_size=self.patch_size
        )
        if self.sampling_map is not None:
            self.set_sampling_map(self.sampling_map, patch_size=self.patch_size)

    def set_sampling_map(
        self,
        sampling_map: Splimage,
        patch_size: Optional[int] = None,
        epoch_size: Optional[int] = None,
    ):
        """Set the per-pixel sampling probabilities.

        Args:
            sampling_map: Probability map (B, 1, H, W) in [0, 1].
            patch_size: Patch size of the extractor; defaults to the
                sampler's own ``patch_size``.
            epoch_size: Optional expected number of selected pixels per
                epoch; the probabilities are rescaled so their sum equals
                ``epoch_size``.
        """
        if patch_size is None:
            patch_size = self.patch_size
        if not self.target_img.same_size(sampling_map):
            sampling_map = sampling_map.resize(self.H, self.W)
        e_size = self.epoch_size if epoch_size is None else epoch_size
        self.sampling_map = sampling_map
        self.sampling_patches = (
            self.sampling_map.extract_image_patches(
                patch_size=patch_size, padding_mode="constant"
            )
            .squeeze(-1)
            .squeeze(0)
        )
        if e_size is not None:
            total = self.sampling_patches.sum()
            self.sampling_patches = self.sampling_patches / total * e_size

    def jitter_target(
        self,
    ) -> Tuple[Float[Tensor[Float, "P S 2"]], Float[Tensor, "P S 4"]]:
        """Jitter patch coordinates and re-extract matching target patches.

        Returns:
            Jittered patch coordinates (P, S, 2) and target patches
            (P, S, 4) extracted at the jittered positions.
        """
        H_step = 1 / self.H
        W_step = 1 / self.W
        device = self.co_patches.device
        H_jitter = torch.rand(tuple(), device=device) * H_step - H_step / 2
        W_jitter = torch.rand(tuple(), device=device) * W_step - W_step / 2
        co_patches[:, :, 0] += H_jitter
        co_patches[:, :, 1] += W_jitter
        return co_patches, self.target_img.extract_image_patches(
            self.patch_size, jitter=(H_jitter, W_jitter)
        ).squeeze(0)

    def set_patch_size(self, patch_size: int):
        """Update the patch size.

        Re-extracts patch grids via ``set_target`` when a target image is
        set; otherwise only stores the new size.

        Args:
            patch_size: New patch side length.
        """
        if self.target_img is not None:
            self.set_target(self.target_img, patch_size)
        else:
            self.patch_size = patch_size

    @property
    def num_patches(self) -> int:
        """Number of patches (P) extracted from the target."""
        return self.target_patches.shape[0]

    def samples(
        self, p: Primitive
    ) -> Iterator[
        Tuple[Float[Tensor, "S C"], Float[Tensor, "S C"], Float[Tensor, "S 2"]]
    ]:
        """Yield (output, target, batch_co) batches.

        Per-pixel Bernoulli subsampling per patch is applied using
        ``sampling_patches``. Batches are accumulated patch-by-patch under
        the same ``mask_size × co_size ≤ max_batch`` budget contract as the
        parent, so the primitive-mask culling optimisation is preserved.

        Args:
            p: Primitive to sample.

        Yields:
            (output, target, batch_co) tuples. ``output`` is the rasteriser
            output (n, 4); ``target`` is (n, C); ``batch_co`` is (n, 2).
        """
        if p.device != self.device:
            raise ValueError(
                f"Sampler and primitive must be on same device. "
                f"Currently: {self.device}, {p.device}."
            )

        co_patches, target_patches = (
            self.jitter_target() if self.jitter_coords else self.co_patches,
            self.target_patches,
        )
        P, S, _ = co_patches.shape
        has_sampling_map = self.sampling_patches is not None

        with torch.no_grad():
            per_co = []  # list[(n_i, 2)]
            per_tgt = []  # list[(n_i, C)]
            counts = []  # list[int]
            for i in range(P):
                if has_sampling_map:
                    prob = self.sampling_patches[i]  # (S,)
                    keep = torch.bernoulli(prob).bool()  # (S,)
                else:
                    keep = torch.ones(S, dtype=torch.bool, device=p.device)
                n_i = int(keep.sum().item())
                if n_i == 0:
                    per_co.append(co_patches[i][0:0])
                    per_tgt.append(target_patches[i][0:0])
                    counts.append(0)
                    continue
                per_co.append(co_patches[i][keep])  # (n_i, 2)
                per_tgt.append(target_patches[i][keep])  # (n_i, C)
                counts.append(n_i)

            patch_size_int = int(S**0.5) if P != 1 else S
            patch_sizes = torch.full(
                (P,), patch_size_int, dtype=torch.long, device=p.device
            )
            Hs = torch.full((P,), self.H, dtype=torch.long, device=p.device)
            Ws = torch.full((P,), self.W, dtype=torch.long, device=p.device)
            patch_masks = p.patch_mask(
                self.co_centers, patch_sizes, Hs, Ws
            )  # (P, Nprims)

        rasterizer = self.rasterizer
        max_batch = self.max_batch

        def _run(
            co_batch: Float[Tensor, "n 2"], prim_mask: Bool[Tensor, "Nprims"]
        ) -> Float[Tensor, "Nc 4"]:
            with p.masked(prim_mask):
                return rasterizer(p(co_batch))

        if max_batch is None:
            co = (
                torch.cat(per_co, dim=0)
                if P > 0
                else torch.empty((0, 2), device=p.device)
            )
            tgt = (
                torch.cat(per_tgt, dim=0)
                if P > 0
                else torch.empty((0, self.target_patches.shape[-1]), device=p.device)
            )
            if co.shape[0] == 0:
                return
            prim_mask = (
                patch_masks.any(dim=0)
                if P > 0
                else torch.zeros(
                    patch_masks.shape[1] if P > 0 else 0,
                    dtype=torch.bool,
                    device=p.device,
                )
            )
            yield _run(co, prim_mask), tgt, co
            return

        i = 0
        n_prims = patch_masks.shape[1]
        while i < P:
            acc_co = []
            acc_tgt = []
            prim_mask = torch.zeros(n_prims, dtype=torch.bool, device=p.device)
            co_size = 0
            while i < P:
                if counts[i] == 0:
                    i += 1
                    continue
                new_prim_mask = prim_mask | patch_masks[i]
                new_mask_size = int(new_prim_mask.sum().item())
                new_co_size = co_size + counts[i]
                if len(acc_co) > 0 and new_mask_size * new_co_size > max_batch:
                    break
                prim_mask = new_prim_mask
                co_size = new_co_size
                acc_co.append(per_co[i])
                acc_tgt.append(per_tgt[i])
                i += 1
            if len(acc_co) == 0:
                # Single patch exceeds max_batch: chunk its pixels.
                pm = patch_masks[i]
                pm_count = int(pm.sum().item())
                chunk = max(1, max_batch // max(pm_count, 1))
                n = counts[i]
                s = 0
                while s < n:
                    e = min(s + chunk, n)
                    co_chunk = per_co[i][s:e]
                    tgt_chunk = per_tgt[i][s:e]
                    yield _run(co_chunk, pm), tgt_chunk, co_chunk
                    s = e
                i += 1
                continue
            co = torch.cat(acc_co, dim=0)
            tgt = torch.cat(acc_tgt, dim=0)
            yield _run(co, prim_mask), tgt, co

    def rasterize(
        self,
        p: Primitive,
        max_batch: Optional[int] = None,
    ) -> Float[Tensor, "B C H W"]:
        """Render the full primitive image over all patches (no subsampling).

        Args:
            p: Primitive to sample.
            max_batch: Compute budget for this render; defaults to
                ``self.max_batch``.

        Returns:
            Tuple of (assembled image (B, C, H, W), target image).
        """
        if p.device != self.device:
            raise ValueError(
                f"Sampler and primitive must be on same device. "
                f"Currently: {self.device}, {p.device}."
            )
        P, S, _ = self.co_patches.shape
        patch_size_int = int(S**0.5) if P != 1 else S
        patch_masks = p.patch_mask(
            self.co_centers,
            torch.full((P,), patch_size_int, dtype=torch.long, device=p.device),
            torch.full((P,), self.H, dtype=torch.long, device=p.device),
            torch.full((P,), self.W, dtype=torch.long, device=p.device),
        )  # (P, Nprims)
        budget = self.max_batch if max_batch is None else max_batch
        gen = [None] * P
        for i in range(P):
            pm = patch_masks[i]
            pm_count = int(pm.sum().item()) if budget is not None else 0
            co_i = self.co_patches[i]  # (S, 2)
            if budget is None or pm_count == 0 or pm_count * co_i.shape[0] <= budget:
                with p.masked(pm):
                    out = self.rasterizer(p(co_i))  # (S, 4)
                gen[i] = out
                continue
            chunk = max(1, budget // pm_count)
            parts = []
            s = 0
            N = co_i.shape[0]
            while s < N:
                e = min(s + chunk, N)
                with p.masked(pm):
                    part = self.rasterizer(p(co_i[s:e]))  # (n, 4)
                parts.append(part)
                s = e
            gen[i] = torch.cat(parts, dim=0)
        full = torch.stack(gen, dim=0)  # (P, S, 4)
        return ImgUtils.assemble_patches(full, self.H, self.W), self.target_img
