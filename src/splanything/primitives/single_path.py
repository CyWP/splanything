from __future__ import annotations
from typing import Tuple, Dict, Optional, List, TYPE_CHECKING

import math
import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ..rendering.sample_output import SampleOutput
from .base import Primitive, cached_property, ParamDef, nomask
from .initializers.base import Initializer

if TYPE_CHECKING:
    from ..training.regularizers.base import Regularizer
    from ..training.refinement.base import SplitRule, FilterRule
    from .splitters.base import Splitter


class SinglePathInitializer(Initializer):
    def init_param(
        self, name: str, param_shape: Tuple[int], batched: bool
    ) -> Float[Tensor, "N ..."]:
        if name == "width":
            return torch.rand(param_shape) * 0.05 + 0.005
        return super().init_param(name, param_shape, batched)


class SinglePathPrimitive(Primitive):
    def __init__(
        self,
        size: int = 4,
        initializers: Optional[Dict[str, Initializer] | Initializer] = None,
        splitters: Optional[Dict[str, Splitter] | Splitter] = None,
        param_defs: Optional[Dict[str, ParamDef]] = None,
        filter_rules: Optional[List[FilterRule]] = None,
        split_rules: Optional[List[SplitRule]] = None,
        sample_processors: Optional[List[SampleProcessor]] = None,
        regularizers: Optional[Dict[str, Tuple[Regularizer, float]]] = None,
    ):
        super().__init__(
            size=size,
            initializers=initializers,
            splitters=splitters,
            param_defs=param_defs,
            filter_rules=filter_rules,
            split_rules=split_rules,
            sample_processors=sample_processors,
            regularizers=regularizers,
        )

    @property
    def default_params(self) -> Dict[str, ParamDef]:
        return dict(
            centroids=ParamDef(True, True, (2,)),
            width=ParamDef(False, True, None),
            color_1=ParamDef(False, True, (3,)),
            color_2=ParamDef(False, True, (3,)),
            alphas=ParamDef(False, True, None),
        )

    @property
    def default_initializers(self) -> Dict[str, Initializer] | Initializer:
        return SinglePathInitializer()

    @cached_property
    def segment_a(self) -> Float[Tensor, "S 2"]:
        return self.centroids[:-1, :]

    @cached_property
    def segment_b(self) -> Float[Tensor, "S 2"]:
        return self.centroids[1:, :]

    @cached_property
    def segment_vec(self) -> Float[Tensor, "S 2"]:
        return self.segment_b - self.segment_a

    @cached_property
    def segment_len_sq(self) -> Float[Tensor, "S"]:
        return (self.segment_vec**2).sum(dim=-1).clamp(min=1e-8)

    @cached_property
    def areas(self) -> Float[Tensor, "1"]:
        total_len = self.segment_len_sq.sqrt().sum()
        area = 2 * total_len * self.width + math.pi * self.width**2
        return area.unsqueeze(0)

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "1"], Float[Tensor, "1"]]:
        w = self.width.unsqueeze(0)
        return (w, w)

    @torch.no_grad()
    def patch_mask(
        self,
        centers: Float[Tensor, "P 2"],
        patch_sizes: Integer[Tensor, "P"],
        H: Integer[Tensor, "P"],
        W: Integer[Tensor, "P"],
    ) -> Bool[Tensor, "P 1"]:
        V = len(self)
        if V < 2:
            return torch.zeros(
                centers.shape[0], 1, device=centers.device, dtype=torch.bool
            )
        total_len = self.segment_len_sq.sqrt().sum()
        radius = total_len / (V - 1) + self.width
        c = self.centroids.mean(dim=0)
        unit_patches = patch_sizes / torch.minimum(H, W)
        dists = (centers - c[None, :]).norm(dim=-1)
        return (dists - unit_patches < radius).unsqueeze(-1)

    def _segment_distances(
        self, co: Float[Tensor, "Nc 2"]
    ) -> Tuple[
        Float[Tensor, "Nc"],
        Float[Tensor, "Nc"],
    ]:
        a = self.segment_a
        v = self.segment_vec
        v_len_sq = self.segment_len_sq

        d = co[:, None, :] - a[None, :, :]
        t = (d * v[None, :, :]).sum(dim=-1) / v_len_sq[None, :]
        t_clamped = t.clamp(0, 1)

        closest = a[None, :, :] + t_clamped[:, :, None] * v[None, :, :]
        dist_sq = ((co[:, None, :] - closest) ** 2).sum(dim=-1)
        best_dist_sq, best_seg = dist_sq.min(dim=-1)

        cross_val = d[..., 0] * v[None, :, 1] - d[..., 1] * v[None, :, 0]
        best_cross = cross_val.gather(-1, best_seg[:, None]).squeeze(-1)

        return best_dist_sq, best_cross

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc 1 3"]:
        _, best_cross = self._segment_distances(co)
        side = (best_cross > 0).float()[:, None, None]
        return (
            side * self.color_1[None, None, :]
            + (1 - side) * self.color_2[None, None, :]
        )

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc 1"]:
        best_dist_sq, _ = self._segment_distances(co)
        w = torch.exp(-best_dist_sq / (2 * self.width**2 + 1e-8))
        return w[:, None]

    def forward(self, co: Float[Tensor, "Nc 2"]) -> SampleOutput:
        if len(self) < 2:
            return SampleOutput(
                rgb=torch.zeros(co.shape[0], 1, 3, device=co.device, dtype=co.dtype),
                weights=torch.zeros(co.shape[0], 1, device=co.device, dtype=co.dtype),
                co=co,
            )
        with self.cache_properties():
            rgb = self.sample_rgb(co)
            weights = self.sample_weights(co)
        sample = SampleOutput(rgb=rgb, weights=weights, co=co)
        for proc in self._sample_processors:
            sample = proc(sample, self)
        return sample

    @nomask
    @torch.no_grad()
    def append(
        self, other: SinglePathPrimitive, weight: float = 0.0
    ) -> SinglePathPrimitive:
        np2 = dict(other.batched_parameters())
        for name, param in self.batched_parameters():
            self.__setattr__(name, torch.cat([param, np2[name]], dim=0))
        np2 = dict(other.stable_parameters())
        for name, param in self.stable_parameters():
            self.__setattr__(name, weight * np2[name] + (1 - weight) * param)
        return self

    @classmethod
    def cat(cls, primitives: List[SinglePathPrimitive]) -> SinglePathPrimitive:
        if not primitives:
            return cls(size=2)
        prim = primitives[0].copy()
        for p in primitives[1:]:
            prim.append(p)
        return prim
