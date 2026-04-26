import torch

from abc import ABC, abstractmethod
from jaxtyping import Float
from torch import Tensor

from .sample_out import SampleOutput


class Rasterizer(ABC):
    """
    General base class for rasterizing function
    """

    @abstractmethod
    def __call__(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "N 4"]:
        pass
