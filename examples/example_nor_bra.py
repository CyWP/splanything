import argparse
import torch
from typing import Tuple
from jaxtyping import Float
from torch import Tensor
from torch.optim import AdamW
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
    GradSplit,
    IsoSplit,
    MapSplit,
    BoundsFilter,
    PrimitiveCeiling,
    PrimitiveFloor,
)
from splanything.training.losses import L2Loss
from splanything.training.regularizers import (
    AttributeProximity,
    AttributeRange,
    AttributeAttractor,
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
run_name = "NorVBra"
base_folder = Path("../test_runs").resolve()
run_folder = base_folder / run_name


class ThetaZero(Initializer):
    def init_param(
        self, name: str, param_shape: Tuple[int], batched: bool
    ) -> Float[Tensor, "N ..."] | Float[Tensor, "..."]:
        if any([c in name for c in ("theta", "angle")]):
            return torch.zeros(param_shape)
        return super().init_param(name, param_shape, batched)


def get_primitive():
    # cubic = CubicFanPrimitive(size=10).to(device)
    msk = Splimage("../assets/bra_nor_offside_masked.png", mask_mode="A", as_mask=True)
    radial = RadialFreqPrimitive(
        size=1500,
        initializers={
            "thetas": ThetaZero(),
            "centroids": MappedInitializer(msk.blur(40) * 0.5 + 0.1),
        },
        param_defs={"thetas": ParamDef(batched=True, trainable=False)},
    ).to(device)
    star = StarPrimitive(
        size=800,
        n_axes=2,
        initializers={
            "thetas": ThetaZero(),
            "centroids": MappedInitializer(msk.blur(100) * 0.3 + 0.2),
        },
        param_defs={"thetas": ParamDef(batched=True, trainable=False)},
    ).to(device)
    multi = MultiPrimitive({"star": star, "radial": radial})
    return multi


def train():
    prev_H = 1080
    prev_W = 1080
    # Images
    tgt = Splimage("../assets/bra_nor_offside.png").to(device)
    msk = Splimage(
        "../assets/bra_nor_offside_masked.png", mask_mode="A", as_mask=True
    ).to(device)
    msk_blur10 = msk.blur(10)
    # primitive
    prim = get_primitive()

    # Rules
    alpha_cull = ThresholdFilter(attr_name="alphas", threshold=0.1, interval=17)
    grad_split_lo = GradSplit(threshold=0.0005, interval=201, attr_names=["centroids"])
    grad_split_hi = GradSplit(threshold=0.02, interval=173, attr_names=["centroids"])
    area_split = ThresholdFilter(attr_name="areas", threshold=0.1, interval=87)
    area_filter = ThresholdFilter(attr_name="areas", threshold=2e-4, interval=69)
    map_split = MapSplit(msk_blur10 * 0.1 + 0.02, interval=307)
    ceiling = PrimitiveCeiling(10000)
    prim.add_split_rule(map_split)
    prim.add_filter_rule(alpha_cull)
    prim.add_filter_rule(area_filter)
    prim["star"].add_split_rule(grad_split_lo)
    prim["radial"].add_split_rule(grad_split_hi)
    prim.add_split_rule(area_split)
    prim.add_filter_rule(ceiling)

    # Rule processors
    map_proc = MapCriterionProcessor(msk_blur10 * 0.6 + 0.4)
    grad_split_lo.add_processor(map_proc)
    grad_split_hi.add_processor(map_proc)

    # Sampler
    sampler = TrainSampler(
        target=tgt,
        patch_size=128,
        max_batch=1000000,
        sampling_map=msk_blur10 * 0.2 + 0.05,
        low_vram=True,
    )

    # Callbacks
    H_pad = (prev_H - tgt.H) // 2
    W_pad = (prev_W - tgt.W) // 2
    vis_sampler = Sampler(
        H=tgt.H,
        W=tgt.W,
        patch_size=256,
        max_batch=1000000,
        padding=(H_pad, H_pad, W_pad, W_pad),
        device=device,
    )
    train_callbacks = [
        PreviewWindow(
            frequency=5,
            show_target=False,
            sampler=vis_sampler,
            save_folder=run_folder / "train_preview",
        ),
        StatsPanel(),
    ]

    # Optim
    optimizer = OptimizerWrapper(prim, AdamW, lr=0.002)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer._optimizer, T_0=1000, eta_min=0.001
    )
    losses = {
        "L2": (L2Loss(), 1.0),
    }
    # Regularizers
    prim["radial"].add_regularizer(
        "Color Push",
        AttributeProximity(["color_1", "color_2"], mode="PUSH"),
        weight=0.05,
    )
    prim.add_regularizer("Area Range", AttributeRange("areas", min=0.001), weight=12.0)
    prim.add_regularizer(
        "Alpha Target", AttributeRange("alphas", min=0.95), weight=12.0
    )
    prim["star"].add_regularizer(
        "Range Ratio",
        AttributeProximity(["rng_1", "rng_2"], mode="RATIO", ratio=0.5),
        weight=5.0,
    )
    # prim["star"].add_regularizer(
    #     "Vertical skew",
    #     AttributeProximity(["scales_1", "scales_2"], mode="RATIO", ratio=0.5),
    #     weight=5.0,
    # )
    prim["radial"].add_regularizer(
        "Dark Test 1",
        AttributeRange("color_1", target=0.0, weight_map=msk_blur10),
        weight=1000000,
    )
    prim["radial"].add_regularizer(
        "Dark Test 2",
        AttributeRange("color_2", target=0.0, weight_map=msk_blur10),
        weight=1000000,
    )

    prim["star"].add_regularizer(
        "Dark Test 3",
        AttributeRange("color", target=0.0, weight_map=msk_blur10),
        weight=1000000,
    )

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
        adjust_prim=False,
    )
    for _ in trainer.train():
        pass


def generate():
    # Load primitive
    gen_H = 3072
    gen_W = 1024
    gen_padding = (1024, 1536, 1536, 1536)
    prim = get_primitive()
    prim.load(run_folder / "primitive.pt")
    prim.requires_grad_(False)
    prim = prim.to(device)
    msk = Splimage(
        "../assets/bra_nor_offside_masked.png", mask_mode="A", as_mask=True
    ).resize(gen_H, gen_W)

    # Sample processor
    sample_proc = FlexibleSampleProcessor(
        lambda s, p: SampleOutput(s.rgb, s.weights, s.co)
    )
    dist_proc = DistanceSampleProcessor(
        sample_proc,
        proc_fn=lambda s, p, d: SampleOutput(
            s.rgb, s.weights * (torch.sin(d * 10000) * 0.2 + 0.8), s.co
        ),
    )
    vec_proc = VecSampleProcessor(
        sample_proc,
        lambda s, p, x, y: SampleOutput(
            s.rgb, s.weights * torch.sin(y * 7000) ** 2, s.co
        ),
    )
    multi_proc = MultiSampleProcessor(
        [
            (sample_proc, msk.blur(40)),
            (dist_proc, 1 - msk.blur(40)),
            (vec_proc, 3),
        ],
        normalize_weights=True,
    )
    prim.add_sample_processor(multi_proc)
    color_proc = ColorSkewSampleProcessor(
        torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.9, 0.6], [0.0, 0.0, 0.0]]).to(device),
        sigma=0.5,
        reduction="MIN",
        rescale=True,
    )
    prim.add_sample_processor(color_proc)

    # Rasterizer
    rast = MultiRasterizer(
        [
            (WeightedRasterizer(), msk.blur(100)),
            (ProbabilisticRasterizer(top_k=5), 1 - msk.blur(100)),
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
