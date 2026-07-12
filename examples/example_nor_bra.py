import torch
from torch.optim import AdamW

from PIL import Image, ImageFilter
from splanything.training import Trainer, TrainSampler, OptimizerWrapper
from splanything.primitives import (
    MultiPrimitive,
    MetaPrimitive,
    RadialFreqSplat,
    CubicFanPrimitive,
    GaussianSinePrimitive,
)
from splanything.training.callbacks import PreviewWindow, PrimitiveCheckpoint
from splanything.training.refinement.rules import (
    AlphaCull,
    GradSplit,
    AreaSplit,
    IsoSplit,
)
from splanything.training.losses import L2Loss
from splanything.training.refinement.processors import MapCriterionProcessor
from splanything.utils import ImgUtils


def run():
    device = torch.device("cuda:0")
    radial = RadialFreqSplat(size=10000).to(device)
    # prim = MetaPrimitive(
    #     primitive=MultiPrimitive(
    #         {
    #             "radial": RadialFreqSplat(size=3),
    #             "fan": CubicFanPrimitive(size=3),
    #             "sine": GaussianSinePrimitive(size=3),
    #         }
    #     ),
    #     size=20,
    #     primitive_trainable=True,
    # )
    tgt = ImgUtils.pil2tensor(
        Image.open("../assets/bra_nor_offside.png").convert("RGBA")
    ).to(device)
    tgt_mask = ImgUtils.pil2map(
        Image.open("../assets/bra_nor_offside_masked.png").filter(
            ImageFilter.GaussianBlur(radius=40)
        ),
        mode="A",
    ).to(device)
    prim = radial
    sampler = TrainSampler(
        target=tgt,
        patch_size=128,
        max_batch=1000000,
        sampling_map=tgt_mask * 0.2 + 0.05,
        low_vram=True,
    )
    train_callbacks = [PreviewWindow(frequency=3, show_target=True)]
    optimizer = OptimizerWrapper(prim, AdamW, lr=0.0002)
    trainer = Trainer(
        "NorVBra",
        prim,
        sampler=sampler,
        optimizer=optimizer,
        losses={"L2": L2Loss()},
        callbacks=train_callbacks,
    )
    print("train")
    for _ in trainer.train():
        print(trainer.epoch)
    print("stop")


if __name__ == "__main__":
    run()
