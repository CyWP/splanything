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
- InverseExponentialWeightedRasterizer: Inverse exponent-powered weight aggregation
- ExponentialProbabilisticRasterizer: Exponent-powered probabilistic selection
- InverseExponentialProbabilisticRasterizer: Inverse exponent-powered probabilistic selection
- MaxWeightedRasterizer: Maximum weight selection aggregation

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

__all__ = [
    "Rasterizer",
    "SampleOutput",
    "WeightedRasterizer",
    "InverseWeightedRasterizer",
    "ProbabilisticRasterizer",
    "InverseProbabilisticRasterizer",
    "UniformRasterizer",
    "ExponentialWeightedRasterizer",
    "InverseExponentialWeightedRasterizer",
    "ExponentialProbabilisticRasterizer",
    "InverseExponentialProbabilisticRasterizer",
    "MaxWeightedRasterizer",
]
