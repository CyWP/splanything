import argparse
import torch
import math
from typing import Tuple
from jaxtyping import Float
from torch import Tensor
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path

from splanything.training import Trainer, TrainSampler, OptimizerWrapper
from splanything.primitives import (
    MultiPrimitive,
    MetaPrimitive,
    RadialFreqPrimitive,
    CubicFanPrimitive,
    StarPrimitive,
    ParamDef,
)
from splanything.primitives.initializers import Initializer, MappedInitializer
from splanything.training.callbacks import (
    PreviewWindow,
    PrimitiveCheckpoint,
    PrimitiveSave,
    StatsPanel,
)
from splanything.training.refinement.rules import (
    ThresholdFilter,
    ThresholdSplit,
    GradSplit,
    IsoSplit,
    MapSplit,
    BoundsFilter,
    PrimitiveCeiling,
    PrimitiveFloor,
)
from splanything.training.losses import L2ImageLoss, SSIMImageLoss
from splanything.training.regularizers import (
    AttributeProximity,
    AttributeRange,
    AttributeAttractor,
    AttributeMap,
)
from splanything.training.refinement.processors import MapCriterionProcessor
from splanything.utils.img import ImgUtils, Splimage
from splanything.rendering import Sampler, SampleOutput
from splanything.rendering.processors import (
    FlexibleSampleProcessor,
    DistanceSampleProcessor,
    MultiSampleProcessor,
    ColorSkewSampleProcessor,
    VecSampleProcessor,
)
from splanything.rendering.rasterizers import (
    ProbabilisticRasterizer,
    WeightedRasterizer,
    MultiRasterizer,
)

device = torch.device("cuda:0")
run_name = "PorVCro"
base_folder = Path("../test_runs").resolve()
run_folder = base_folder / run_name
seed = None
if seed is not None:
    torch.manual_seed(seed)


class ThetaSet(Initializer):
    def __init__(self, val: float):
        self.val = val

    def init_param(
        self, name: str, param_shape: Tuple[int], batched: bool
    ) -> Float[Tensor, "N ..."] | Float[Tensor, "..."]:
        if any([c in name for c in ("theta", "angle")]):
            return torch.full(param_shape, self.val)
        return super().init_param(name, param_shape, batched)


def get_primitive():
    # cubic = CubicFanPrimitive(size=10).to(device)
    msk = Splimage(
        "../assets/por_cro_offside_masked.png", mask_mode="A", as_mask=True
    ).to(device)
    # msk_body = Splimage(
    #     "../assets/por_cro_offside_masked_body.png", mask_mode="A", as_mask=True
    # ).to(device)
    cubic = CubicFanPrimitive(
        size=80,
        initializers={
            # "thetas": ThetaSet(0.0),
            "centroids": MappedInitializer(msk.expand(100)),
        },
        # param_defs={"thetas": ParamDef(batched=True, trainable=False)},
    ).to(device)
    # radial = RadialFreqPrimitive(
    #     size=50,
    #     initializers={
    #         "thetas": ThetaSet(0.0),
    #         "centroids": MappedInitializer(msk_body),
    #     },
    #     param_defs={"thetas": ParamDef(batched=True, trainable=False)},
    # ).to(device)
    # cubic.scale(0.4)
    # radial.scale(0.05)
    # prim = MultiPrimitive({"cubic": cubic, "radial": radial})
    # return prim
    return cubic


def train():
    prev_H = 1080
    prev_W = 1080
    # Images
    tgt = Splimage("../assets/por_cro_offside.png").to(device)
    msk = Splimage(
        "../assets/por_cro_offside_masked.png", mask_mode="A", as_mask=True
    ).to(device)
    theta_msk = Splimage(
        "../assets/por_cro_offside_theta_mask.png", mask_mode="mean", as_mask=True
    ).to(device)
    # primitive
    prim = get_primitive()

    # Rules
    alpha_cull = ThresholdFilter(
        attr_name="alphas", threshold=0.1, interval=52, comparison="OVER"
    )
    # area_limit = ThresholdFilter(
    #     threshold=0.0005, interval=73, attr_name="areas", comparison="OVER"
    # )
    area_split = ThresholdSplit("areas", 0.03, interval=83, comparison="OVER")
    grad_split_lo = GradSplit(threshold=0.05, interval=201, attr_names=["centroids"])
    grad_split_hi = GradSplit(threshold=0.02, interval=173, attr_names=["centroids"])
    map_split = MapSplit(msk.blur(10) * 0.1 + 0.02, interval=307)
    ceiling = PrimitiveCeiling(2500)
    # prim.add_split_rule(area_limit)
    prim.add_split_rule(map_split)
    prim.add_filter_rule(alpha_cull)
    prim.add_filter_rule(ceiling)
    prim.add_split_rule(grad_split_lo)
    prim.add_split_rule(area_split)
    # prim["radial"].add_split_rule(grad_split_hi)

    # Rule processors
    map_proc = MapCriterionProcessor(msk.expand(20) * 0.6 + 0.4)
    area_split.add_processor(map_proc)
    # grad_split_lo.add_processor(map_proc)
    # grad_split_hi.add_processor(map_proc)

    # Sampler
    sampler = TrainSampler(
        target=tgt.resize(200, 360),
        patch_size=64,
        max_batch=500000,
        sampling_map=msk.blur(30) * 0.2 + 1e-5,
        low_vram=True,
    )

    # Callbacks
    H_pad = (prev_H - tgt.H) // 2
    W_pad = (prev_W - tgt.W) // 2
    vis_sampler = Sampler(
        H=800,
        W=1440,
        patch_size=256,
        max_batch=1000000,
        padding=(H_pad, H_pad, W_pad, W_pad),
        device=device,
    )
    train_callbacks = [
        PreviewWindow(
            frequency=10,
            show_target=True,
            sampler=vis_sampler,
            # save_folder=run_folder / "train_preview",
        ),
        StatsPanel(),
    ]

    # Optim
    optimizer = OptimizerWrapper(prim, AdamW, lr=0.002)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer._optimizer, T_0=1000, eta_min=0.001
    )
    losses = {
        "L2": (L2ImageLoss(msk.expand(150) * 2), 1.0),
        "SSIM": (SSIMImageLoss(1 - msk.expand(50)), 0.05),
    }
    # Regularizers
    prim.add_regularizer(
        "Color Push",
        AttributeProximity(["color_1", "color_2"], mode="PUSH"),
        weight=0.05,
    )
    prim.add_regularizer("Area penalty", AttributeRange("areas", max=0.08), weight=10.0)
    # prim.add_regularizer("Freqmaxx", AttributeRange("freq", min=4.0), weight=10.0)
    prim.add_regularizer("Alpha Target", AttributeRange("alphas", min=0.6), weight=12.0)
    prim.add_regularizer(
        "Theta Map",
        AttributeMap(theta_msk, "thetas"),
        weight=1000.0,
    )

    # prim["star"].add_regularizer(
    #     "Vertical skew",
    #     AttributeProximity(["scales_1", "scales_2"], mode="RATIO", ratio=0.5),
    #     weight=5.0,
    # )

    # Trainer
    trainer = Trainer(
        run_name,
        prim,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        losses=losses,
        callbacks=train_callbacks,
        base_folder=base_folder,
        adjust_prim=True,
    )
    for _ in trainer.train():
        pass


def generate():
    # Load primitive
    gen_H = 3072
    gen_W = 1024
    gen_padding = (1536, 1024, 1024, 1024)
    full_H = gen_H + gen_padding[0] + gen_padding[1]
    full_W = gen_W + gen_padding[2] + gen_padding[3]
    prim = get_primitive()
    prim.load(run_folder / "primitive.pt")
    prim.requires_grad_(False)
    prim = prim.to(device)
    prim.adjust_to_canvas(full_H, full_W)
    msk = (
        Splimage("../assets/por_cro_offside_masked.png", mask_mode="A", as_mask=True)
        .to(device)
        .resize(gen_H, gen_W)
    )

    # Sample processor
    sample_proc = FlexibleSampleProcessor(
        lambda s, p: SampleOutput(s.rgb, s.weights**1.08, s.co)
    )
    dist_proc = DistanceSampleProcessor(
        sample_proc,
        proc_fn=lambda s, p, d: SampleOutput(
            s.rgb, s.weights * (torch.sin(d * 10000) * 0.2 + 0.8), s.co
        ),
    )
    diag_proc = VecSampleProcessor(
        sample_proc,
        lambda s, p, x, y: SampleOutput(
            s.rgb,
            s.weights * (torch.cos(x * 1000) * torch.sin(y * 7000) * 0.5 + 0.5) ** 2,
            s.co,
        ),
    )
    vert_proc = VecSampleProcessor(
        sample_proc,
        lambda s, p, x, y: SampleOutput(
            s.rgb,
            s.weights * (torch.sin(y * 7000) * 0.5 + 0.5) ** 2,
            s.co,
        ),
    )
    multi_proc = MultiSampleProcessor(
        [
            (sample_proc, msk.blur(40)),
            (dist_proc, 1 - msk.blur(40)),
            (diag_proc, msk.blur(100) * 3),
            (vert_proc, (1 - (msk.blur(100))) * 3),
        ],
        normalize_weights=True,
    )
    prim.add_sample_processor(multi_proc)
    color_proc = ColorSkewSampleProcessor(
        torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.9, 0.6], [0.0, 0.0, 0.0]]).to(device),
        sigma=3.0,
        reduction="MIN",
        rescale=True,
    )
    prim.add_sample_processor(color_proc)

    # Rasterizer
    rast = MultiRasterizer(
        [
            (WeightedRasterizer(), msk.blur(700)),
            (ProbabilisticRasterizer(top_k=5), 1 - msk.blur(700)),
        ]
    )

    # Sampler
    sampler = Sampler(
        gen_H,
        gen_W,
        patch_size=382,
        max_batch=10000000,
        rasterizer=rast,
        padding=gen_padding,
        device=device,
        low_vram=False,
    )

    # Output
    out = sampler.rasterize(prim, verbose=True)
    img = ImgUtils.tensor2pil(out)
    img.save(run_folder / "output.png")
    print(f"Saved output to {run_folder}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true", help="Train")
    parser.add_argument("-g", "--generate", action="store_true", help="Generate")
    args = parser.parse_args()
    if args.train:
        train()
    if args.generate:
        generate()


if __name__ == "__main__":
    main()
