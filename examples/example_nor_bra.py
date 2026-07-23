import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from PIL import Image, ImageFilter
from splanything.training import Trainer, TrainSampler, OptimizerWrapper
from splanything.primitives import (
    MultiPrimitive,
    MetaPrimitive,
    RadialFreqSplat,
    CubicFanPrimitive,
)
from splanything.training.callbacks import (
    PreviewWindow,
    PrimitiveCheckpoint,
    PrimitiveSave,
    StatsPanel,
)
from splanything.training.refinement.rules import (
    AlphaCull,
    GradSplit,
    AreaSplit,
    IsoSplit,
)
from splanything.training.losses import L2Loss
from splanything.training.refinement.processors import MapCriterionProcessor
from splanything.utils.img import ImgUtils


def run():
    # primitives
    device = torch.device("cuda:0")
    radial = RadialFreqSplat(size=5, scale_factor=1.0).to(device)
    cubic = CubicFanPrimitive(size=3, scale_factor=1.0).to(device)
    multi = MultiPrimitive({"radial": radial, "cubic": cubic})
    prim = MetaPrimitive(
        primitive=multi, size=200, primitive_trainable=False, scale_factor=1.0
    ).to(device)
    # rules
    # prim = multi
    alpha_cull = AlphaCull(threshold=0.1, interval=242)
    grad_split = GradSplit(threshold=0.2, interval=407)
    # radial.add_filter_rule(alpha_cull)
    # radial.add_split_rule(grad_split)
    # cubic.add_filter_rule(alpha_cull)
    # cubic.add_split_rule(grad_split)
    prim.add_filter_rule(alpha_cull)
    prim.add_split_rule(grad_split)

    tgt = ImgUtils.pil2tensor(
        Image.open("../assets/bra_nor_offside.png").convert("RGBA")
    ).to(device)
    tgt_mask = ImgUtils.pil2map(
        Image.open("../assets/bra_nor_offside_masked.png").filter(
            ImageFilter.GaussianBlur(radius=40)
        ),
        mode="A",
    ).to(device)
    map_proc = MapCriterionProcessor(tgt_mask)
    grad_split.add_processor(map_proc)
    sampler = TrainSampler(
        target=tgt,
        patch_size=128,
        max_batch=1000000,
        sampling_map=tgt_mask * 0.2 + 0.05,
        low_vram=True,
    )
    train_callbacks = [
        PreviewWindow(frequency=5, show_target=True),
        StatsPanel(),
        PrimitiveSave("./prim.pth"),
    ]
    optimizer = OptimizerWrapper(prim, AdamW, lr=0.001)
    scheduler = CosineAnnealingWarmRestarts(optimizer._optimizer, T_0=200)
    trainer = Trainer(
        "NorVBra",
        prim,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        losses={"L2": L2Loss()},
        callbacks=train_callbacks,
        base_folder="../test_runs",
    )
    for _ in trainer.train():
        pass


if __name__ == "__main__":
    run()
