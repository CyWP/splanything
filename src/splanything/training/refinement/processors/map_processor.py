from torch import Tensor
from jaxtyping import Float
from typing import Callable, Optional

from ..base import CriterionProcessor, RefinementRule
from ....primitives.base import Primitive
from ....utils.img import ImgUtils


class MapCriterionProcessor(CriterionProcessor):
    def __init__(
        self,
        map: Float[Tensor, "B 1 H W"],
        proc_fn: Optional[
            Callable[Float[Tensor, "N ..."], Float[Tensor, "N ..."]]
        ] = None,
    ):
        self.map = map
        self.proc_fn = lambda x, y: x * y if proc_fn is None else proc_fn

    def apply(
        self,
        primitive: Primitive,
        rule: RefinementRule,
        criterion: Float[Tensor, "N ..."],
        **kwargs,
    ) -> Float[Tensor, "N ..."]:
        sampled = ImgUtils.uv_sample(self.map, primitive.centroids).view(
            criterion.shape
        )
        return self.proc_fn(sampled, criterion)
