"""Rasterizers for converting sample outputs to final RGBA images.

Exposes:
- Rasterizer: Base class for rasterization strategies
- SampleOutput: Container for per-coordinate per-primitive sampling data
- WeightedRasterizer: Weight-based aggregation rasterizer (default)
- ProbabilisticRasterizer: Probabilistic selection rasterizer
- get_rasterizer: Factory function to instantiate a rasterizer from config
"""

from typing import Dict, Any, Optional

from .generic import Rasterizer
from .sample_out import SampleOutput
from .weighted import WeightedRasterizer
from .probabilistic import ProbabilisticRasterizer

CLASSES = [WeightedRasterizer, ProbabilisticRasterizer]

RASTERIZERS = {c.__name__.lower(): c for c in CLASSES}


def get_rasterizer(
    name: Optional[str] = None, kwargs: Dict[str, Any] = None
) -> Rasterizer:
    """Instantiate a rasterizer from name and kwargs.

    Args:
        name: Rasterizer class name (e.g., "Weighted", "Probabilistic").
        kwargs: Constructor arguments.

    Returns:
        Rasterizer instance.

    Raises:
        KeyError: If name is not a valid rasterizer class.
    """
    if name is None:
        return WeightedRasterizer()
    if kwargs is None:
        kwargs = {}
    rcls = RASTERIZERS.get(name.lower(), None)
    if rcls is None:
        raise KeyError(
            f"'{name}' is an invalid rasterizer class.\n"
            f"Valid classes: {list(RASTERIZERS.keys())}"
        )
    return rcls(**kwargs)
