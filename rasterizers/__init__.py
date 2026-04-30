"""Rasterizers for converting sample outputs to final RGBA images.

Exposes:
- Rasterizer: Base class for rasterization strategies
- SampleOutput: Container for per-coordinate per-primitive sampling data
- WeightedRasterizer: Weight-normalized aggregation (default)
- InverseWeightedRasterizer: Inverse-weight normalization aggregation
- ProbabilisticRasterizer: Weighted random selection aggregation
- InverseProbabilisticRasterizer: Inverse-weight random selection aggregation
- UniformRasterizer: Uniform averaging aggregation
- ExponentialWeightedRasterizer: Exponent-powered weight aggregation
- MaxWeightedRasterizer: Maximum weight selection aggregation
- get_rasterizer: Factory function to instantiate a rasterizer from config

Internal:
- generic.py: Rasterizer base class
- sample_out.py: SampleOutput container
"""

from .generic import Rasterizer
from .sample_out import SampleOutput
from .weighted import WeightedRasterizer
from .inv_weighted import InverseWeightedRasterizer
from .probabilistic import ProbabilisticRasterizer
from .inv_probabilistic import InverseProbabilisticRasterizer
from .uniform import UniformRasterizer
from .exponential import ExponentialWeightedRasterizer
from .inv_exponential import InverseExponentialWeightedRasterizer
from .inv_exp_prob import InverseExponentialProbabilisticRasterizer
from .exp_prob import ExponentialProbabilisticRasterizer
from .max import MaxWeightedRasterizer

CLASSES = [
    WeightedRasterizer,
    InverseWeightedRasterizer,
    ProbabilisticRasterizer,
    InverseProbabilisticRasterizer,
    UniformRasterizer,
    ExponentialWeightedRasterizer,
    InverseExponentialWeightedRasterizer,
    MaxWeightedRasterizer,
    InverseExponentialProbabilisticRasterizer,
    ExponentialProbabilisticRasterizer,
]

RASTERIZERS = {c.__name__.lower(): c for c in CLASSES}


def get_rasterizer(**kwargs) -> Rasterizer:
    """Instantiate a rasterizer from name and kwargs.

    Args:
        name: Rasterizer class name (e.g., "Weighted", "Probabilistic").
        kwargs: Constructor arguments.

    Returns:
        Rasterizer instance.

    Raises:
        KeyError: If name is not a valid rasterizer class.
    """
    name = kwargs.pop("name", None)
    if name is None:
        return WeightedRasterizer()
    rcls = RASTERIZERS.get(name.lower(), None)
    if rcls is None:
        raise KeyError(
            f"'{name}' is an invalid rasterizer class.\n"
            f"Valid classes: {list(RASTERIZERS.keys())}"
        )
    return rcls(**kwargs)
