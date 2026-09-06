"""Fit a CubicFanPrimitive to the Portugal/Croatia target image, then
re-render the trained primitive at high resolution.

Run with ``-t/--train`` to train and ``-g/--generate`` to re-render.
"""

import argparse
import torch
import math
from typing import Tuple
from jaxtyping import Float
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path

from splanything.training import Trainer, TrainSampler, OptimizerWrapper
from splanything.primitives import (
    CubicFanPrimitive,
)
from splanything.primitives.initializers import Initializer, MappedInitializer
from splanything.training.callbacks import (
    PreviewWindow,
    StatsPanel,
)
from splanything.training.refinement.rules import (
    ThresholdFilter,
    ThresholdSplit,
    GradSplit,
    MapSplit,
    PrimitiveCeiling,
)
from splanything.training.losses import L2ImageLoss, SSIMImageLoss
from splanything.training.regularizers import (
    AttributeProximity,
    AttributeRange,
    AttributeMap,
)
from splanything.training.refinement.processors import MapCriterionProcessor
from splanything.utils.img import ImgUtils, Splimage
from splanything.rendering import Sampler, SampleOutput
from splanything.rendering.processors import (
    FlexibleSampleProcessor,
    MultiSampleProcessor,
    ColorSkewSampleProcessor,
    VecSampleProcessor,
)
from splanything.rendering.rasterizers import (
    ProbabilisticRasterizer,
    WeightedRasterizer,
    MultiRasterizer,
)

# Device and output folder: checkpoints, previews and the final render are
# saved under base_folder / run_name.
device = torch.device("cuda:0")
run_name = "PorVCro"
base_folder = Path("../test_runs").resolve()
run_folder = base_folder / run_name
# Set a seed to make runs reproducible; None keeps random initialization.
seed = None
if seed is not None:
    torch.manual_seed(seed)


# Custom initializer: pin every theta (fan rotation) to one value; all other
# parameters fall back to the name-based default initializer.
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
    # Cubic-fan splats seeded inside the masked flag region: the
    # MappedInitializer samples each centroid from the (slightly lifted)
    # mask density, so all splats start on the flag.
    msk = Splimage(
        "../assets/por_cro_offside_masked.png", mask_mode="A", as_mask=True
    ).to(device)
    cubic = CubicFanPrimitive(
        size=20,
        initializers={
            "centroids": MappedInitializer(msk.expand(200) + 1e-3),
        },
    ).to(device)
    return cubic


def train():
    prev_H = 1080
    prev_W = 1080
    # Target image, alpha mask (train and render only inside the flag), and
    # a map of target fan angles used by the theta regularizer and nudge.
    tgt = Splimage("../assets/por_cro_offside.png").to(device)
    msk = Splimage(
        "../assets/por_cro_offside_masked.png", mask_mode="A", as_mask=True
    ).to(device)
    theta_msk = (
        Splimage(
            "../assets/por_cro_offside_theta_mask.png", mask_mode="mean", as_mask=True
        ).to(device)
        * math.pi
        / 2
        + math.pi / 6
    )
    # Primitive
    prim = get_primitive()
    # Refinement rules, applied by the trainer around each optimizer step:
    # cull faded splats, split oversized areas and high-gradient regions,
    # split by position via a blurred mask, and cap the total splat count.
    # Staggered intervals make the rules fire on different epochs.
    alpha_cull = ThresholdFilter(
        attr_name="alphas", threshold=0.1, interval=52, comparison="OVER"
    )
    area_split = ThresholdSplit("areas", 0.03, interval=83, comparison="OVER")
    grad_split_lo = GradSplit(threshold=0.04, interval=201, attr_names=["centroids"])
    map_split = MapSplit(msk.blur(10) * 0.02 + 0.005, interval=137)
    ceiling = PrimitiveCeiling(1000)
    prim.add_split_rule(map_split)
    prim.add_filter_rule(alpha_cull)
    prim.add_filter_rule(ceiling)
    prim.add_split_rule(grad_split_lo)
    prim.add_split_rule(area_split)

    # Rule processor: scale the area-split criterion by the mask so
    # splitting concentrates inside the flag.
    map_proc = MapCriterionProcessor(msk.expand(20) * 0.6 + 0.4)
    area_split.add_processor(map_proc)

    # Training sampler: subsamples pixels per patch using the blurred mask
    # as probability map; max_batch bounds the per-step compute budget.
    sampler = TrainSampler(
        target=tgt.resize(200, 360),
        patch_size=64,
        max_batch=100000,
        sampling_map=msk.blur(30) * 0.2 + 1e-5,
        low_vram=True,
        jitter_coords=False,
    )

    # Callbacks: live preview rendered at the display resolution (with
    # padding to match the previous viewport) plus a console stats panel.
    H_pad = int(prev_H - tgt.H)
    W_pad = int(prev_W - tgt.W)
    vis_sampler = Sampler(
        H=prev_H - H_pad,
        W=prev_W - W_pad,
        patch_size=256,
        max_batch=1000000,
        padding=(H_pad // 2, H_pad // 2, W_pad // 3, W_pad * 2 // 3),
        device=device,
    )
    train_callbacks = [
        PreviewWindow(
            frequency=1,
            show_target=False,
            sampler=vis_sampler,
            save_folder=run_folder / "train_preview",
        ),
        StatsPanel(),
    ]

    # Optimizer with per-parameter learning-rate modifiers and a cosine
    # schedule with warm restarts; the 100 pre-steps start training
    # mid-cycle instead of at the peak learning rate.
    optimizer = OptimizerWrapper(prim, AdamW, lr=0.005)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer._optimizer, T_0=200, eta_min=0.001
    )
    for _ in range(100):
        scheduler.step()
    # Image-level losses on full renders; both use blurred masks as
    # per-pixel weights so the background does not dominate the loss.
    losses = {
        "L2": (L2ImageLoss(msk.blur(40)), 1.0),
        "SSIM": (SSIMImageLoss(msk.blur(200)), 0.1),
    }
    # Regularizers: push the two fan colors apart, keep alphas high, pull
    # thetas toward the angle map, and keep splat areas in a sane range.
    prim.add_regularizer(
        "Color Push",
        AttributeProximity(["color_1", "color_2"], mode="PUSH"),
        weight=1e-8,
    )
    prim.add_regularizer("Alpha Target", AttributeRange("alphas", min=0.6), weight=12.0)
    prim.add_regularizer(
        "Theta Map",
        AttributeMap(theta_msk, "thetas"),
        weight=1.0,
    )
    prim.add_regularizer(
        "Area_floor", AttributeRange("areas", min=1e-6, max=0.25), weight=1.0
    )

    # Trainer drives epochs; the loop body runs after every epoch and
    # nudges thetas toward the angle map (momentum 0.995).
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
        with torch.no_grad():
            prim.update_parameters(
                {
                    "thetas": 0.005
                    * theta_msk.mask_sample(prim.centroids)[0].squeeze(-1)
                    + 0.995 * prim.thetas
                }
            )


def generate():
    """Re-render the trained primitive at high resolution with decorative
    sample processors and a blended rasterizer."""
    gen_H = 2040
    gen_W = 3600
    gen_padding = (1536, 1536, 712, 800)
    # Load the trained checkpoint; adapt splat size to the larger canvas.
    prim = get_primitive()
    prim.load(run_folder / "primitive.pt")
    prim.requires_grad_(False)
    prim = prim.to(device)
    prim.adjust_to_canvas(gen_H, gen_W)
    msk = (
        Splimage("../assets/por_cro_offside_masked.png", mask_mode="A", as_mask=True)
        .to(device)
        .resize(gen_H, gen_W)
    )
    theta_msk = (
        Splimage(
            "../assets/por_cro_offside_theta_mask.png", mask_mode="mean", as_mask=True
        ).to(device)
        * math.pi
        / 2
        + math.pi / 6
    )
    # Pin fan angles from the angle map and fade small splats so they do
    # not dominate the high-res render.
    prim.thetas.weight = theta_msk.mask_sample(prim.centroids)[0].squeeze(-1)
    areas = prim.areas
    areas_weight = 1 - ((areas - areas.min()) / (areas.max() - areas.min())) * 0.5 + 0.5
    prim.alphas.weight = prim.alphas * areas_weight

    # Sample processors for the final look: a slight weight sharpening
    # inside the flag, plain weights outside, and a color skew toward a
    # fixed palette; MultiSampleProcessor blends them by mask weight.
    exp_proc = FlexibleSampleProcessor(
        lambda s, p: SampleOutput(s.rgb, s.weights**1.1, s.co)
    )
    reg_proc = FlexibleSampleProcessor(lambda s, p: s)
    color_proc = ColorSkewSampleProcessor(
        torch.tensor(
            [[1.0, 0.65, 0.0], [0.25, 0.0, 1.0], [0.95, 0.8, 0.65], [0.0, 0.0, 0.0]]
        ).to(device),
        sigma=4.0,
        reduction="MIN",
        rescale=True,
    )

    proc = MultiSampleProcessor(
        [(exp_proc, msk.blur(40)), (reg_proc, 1 - msk.blur(40)), (color_proc, 2.0)],
        normalize_weights=True,
    )

    # Rays: modulate weights along each splat's dominant axis to draw
    # radial rays at high frequency.
    def _radius_proc(s, p, x, y):
        ax_1, ax_2 = p.axes
        ax = torch.where((p.range_1 > p.range_2)[:, None], ax_1, ax_2)  # [N, 2]
        delta = torch.stack([x, y], dim=-1)  # [Nc, N, 2]
        proj = (delta * ax).sum(dim=-1)  # [Nc, N]
        W = s.weights * (torch.cos(proj * 3000) * 0.4 + 0.6)
        return SampleOutput(s.rgb, W, s.co)

    radius_proc = VecSampleProcessor(
        proc,
        _radius_proc,
    )
    prim.add_sample_processor(radius_proc)

    # Rasterizer blend: weighted aggregation inside the flag, Monte Carlo
    # sampling outside.
    rast = MultiRasterizer(
        [
            (WeightedRasterizer(), msk.blur(200)),
            (ProbabilisticRasterizer(top_k=50), 1 - msk.blur(200)),
        ]
    )

    # Inference sampler over the large canvas, then render and save.
    sampler = Sampler(
        gen_H,
        gen_W,
        patch_size=756,
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
