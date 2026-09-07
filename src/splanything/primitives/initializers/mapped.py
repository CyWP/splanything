"""Image-map-guided parameter initializer."""

import torch
from torch import Tensor
from typing import Tuple, Optional, Callable
from jaxtyping import Float

from .base import Initializer
from ...utils.img import Splimage


class MappedInitializer(Initializer):
    """Initializer placing parameters according to a Splimage map.

    The coordinate parameter is sampled as (noisy) pixel positions of the
    map; other parameters are initialized normally and then combined with
    the map's sampled value at the corresponding coordinate.
    """

    def __init__(
        self,
        map: Splimage,
        coordinate_key: str = "centroids",
        initializer: Optional[Initializer] = None,
        apply_func: Optional[
            Callable[
                [str, Float[Tensor, "N"], Float[Tensor, "N ..."]],
                Float[Tensor, "N ..."],
            ]
        ] = None,
    ):
        """Initialize the initializer.

        Args:
            map: Splimage mask sampled at the generated coordinates.
            coordinate_key: Name of the coordinate parameter (e.g.
                "centroids") that receives sampled coordinates.
            initializer: Fallback initializer for other parameters;
                defaults to ``Initializer()``.
            apply_func: Optional custom (name, sampled, feat) -> tensor;
                defaults to elementwise multiplication with the sampled
                map values.
        """
        self.map = map
        self.initializer = Initializer() if initializer is None else initializer
        self.coordinate_key = coordinate_key
        self._sampled_cache = None
        self.apply_func = apply_func
        self._feats_cache = {}

    def init_param(
        self, name: str, param_shape: Tuple[int], batched: bool
    ) -> Float[Tensor, "Size ..."]:
        """Initialize a parameter tensor.

        For the coordinate parameter, samples coordinates from the map and
        primes the sampled-value cache. Other parameters are initialized
        by the fallback initializer and combined with the cached map
        values; if coordinates are not initialized yet, features are
        cached and re-processed once coordinates arrive.

        Args:
            name: Parameter name.
            param_shape: Shape of the parameter tensor.
            batched: Whether the parameter has a batch dimension.

        Returns:
            Initialized tensor.
        """
        if not batched:
            return self.initializer.init_param(name, param_shape, batched)
        if name == self.coordinate_key:
            assert param_shape == (param_shape[0], 2), (
                f"The coordinate key '{name}' must represent 2D Coordinates."
            )
            co = self.generate_coords(param_shape[0])
            self._sampled_cache = self.map.mask_sample(co)
            self.process_feats_cache()
            return co
        feat = self.initializer.init_param(name, param_shape, batched)
        if self._sampled_cache is None:
            self._feats_cache[name] = feat
            return feat
        else:
            return self.apply_map(name, self._sampled_cache, feat)

    def generate_coords(self, N: int) -> Float[Tensor, "N 2"]:
        """Sample ``N`` noisy pixel coordinates from the map.

        Args:
            N: Number of coordinates to sample.

        Returns:
            Coordinates (N, 2).
        """
        return self.map.sample_px_coords(N, noise=True)

    def process_feats_cache(self):
        """Apply the map in-place to features initialized before coordinates.

        Features initialized prior to the coordinate parameter are cached;
        once coordinates are sampled, this updates them in-place with
        their map-scaled values and clears the cache.
        """
        # Reference to previously intialized tensors are kept,
        # so we can update in place once coordinates are computed/sampled.
        sampled = self._sampled_cache
        with torch.no_grad():
            for name, feat in self._feats_cache.items():
                feat.copy_(self.apply_map(name, sampled, feat))
        self._feats_cache = {}

    def apply_map(
        self, name: str, sampled: Float[Tensor, "N"], feat: Float[Tensor, "N ..."]
    ) -> Float[Tensor, "N ..."]:
        """Combine a feature tensor with sampled map values.

        Args:
            name: Parameter name.
            sampled: Per-primitive sampled map values (N,).
            feat: Feature tensor (N, ...).

        Returns:
            Combined tensor (N, ...); elementwise product by default.
        """
        if self.apply_func is None:
            return sampled * feat
        return self.apply_func(name, sampled, feat)
