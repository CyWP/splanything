from __future__ import annotations

import logging

import torch

from .base import Primitive

_logger = logging.getLogger(__name__)


class MetaPrimitive(Primitive):
    def __init__(
        self,
        primitive: Primitive,
        size: int = 1,
        location: bool = True,
        rotation: bool = True,
        scale: bool = True,
        color: bool = True,
        primitive_trainable: bool = False,
    ):
        super().__init__()
        self.primitive = primitive
        self.add_parameter(
            "centroids", torch.rand((size, 2)), batched=True, trainable=True
        )
        self.add_parameter("thetas", torch.rand((size,)), batched=True, trainable=True)
