import argparse
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path

from PIL import Image, ImageFilter
from splanything.training import Trainer, TrainSampler, OptimizerWrapper
from splanything.primitives import (
    MultiPrimitive,
    MetaPrimitive,
    RadialFreqPrimitive,
    CubicFanPrimitive,
    GaussianPrimitive,
)
from splanything.training.callbacks import (
    PreviewWindow,
    PrimitiveCheckpoint,
    PrimitiveSave,
    StatsPanel,
)
from splanything.training.refinement.rules import (
    AlphaFilter,
    GradSplit,
    AreaSplit,
    IsoSplit,
    MapSplit,
    BoundsFilter,
    PrimitiveCeiling,
    PrimitiveFloor,
)
from splanything.training.losses import L2Loss
from splanything.training.regularizers import AttributeProximity, AttributeRange
from splanything.training.refinement.processors import MapCriterionProcessor
from splanything.utils.img import ImgUtils
from splanything.rendering import Sampler, SampleOutput
from splanything.rendering.processors import FlexibleSampleProcessor
from splanything.rendering.rasterizers import (
    ProbabilisticRasterizer,
    WeightedRasterizer,
)

device = torch.device("cuda:0")
run_name = "NorVBra"
base_folder = Path("../test_runs").resolve()
run_folder = base_folder / run_name


def get_primitive():
    cubic = CubicFanPrimitive(size=100).to(device)
    radial = RadialFreqPrimitive(size=200, scale_factor=1.0).to(device)
    multi = MultiPrimitive({"cubic": cubic, "radial": radial})
    return multi


def train():
    # Images
    tgt = ImgUtils.pil2tensor(
        Image.open("../assets/bra_nor_offside.png").convert("RGBA")
    ).to(device)
    msk_img = Image.open("../assets/bra_nor_offside_masked.png")
    msk_img_blur40 = msk_img.filter(ImageFilter.GaussianBlur(radius=40))
    msk_img_blur10 = msk_img.filter(ImageFilter.GaussianBlur(radius=10))
    msk_tensor_blur40 = ImgUtils.pil2map(msk_img_blur40, mode="A").to(device)
    msk_tensor_blur10 = ImgUtils.pil2map(msk_img_blur10, mode="A").to(device)

    # primitive
    prim = get_primitive()

    # Rules
    alpha_cull = AlphaFilter(threshold=0.5, interval=87)
    grad_split = GradSplit(threshold=0.002, interval=223)
    # area_split = AreaSplit(threshold=0.1, interval=87)
    # map_split = MapSplit(msk_tensor_blur10 * 0.6 + 0.05, interval=333)
    ceiling = PrimitiveCeiling(10000)
    # bounds_cull = BoundsFilter(interval=200, margin=0.0, use_areas=False)
    prim.add_filter_rule(alpha_cull)
    prim.add_split_rule(grad_split)
    # prim.add_split_rule(map_split)
    # prim.add_split_rule(area_split)
    prim.add_filter_rule(ceiling)
    # prim.add_filter_rule(bounds_cull)

    # Rule processors
    map_proc = MapCriterionProcessor(msk_tensor_blur10 * 0.6 + 0.4)
    grad_split.add_processor(map_proc)

    # Sampler
    sampler = TrainSampler(
        target=tgt,
        patch_size=128,
        max_batch=1000000,
        sampling_map=msk_tensor_blur10 * 0.2 + 0.05,
        low_vram=True,
    )

    # Callbacks
    train_callbacks = [PreviewWindow(frequency=5, show_target=True), StatsPanel()]

    # Optim
    optimizer = OptimizerWrapper(prim, AdamW, lr=0.002)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer._optimizer, T_0=1000, eta_min=0.001
    )
    losses = {
        "L2": (L2Loss(), 1.0),
    }
    # Regularizers
    # cubic = prim.primitives["cubic"]
    # radial = prim.primitives["radial"]
    prim.add_regularizer(
        "Color Push",
        AttributeProximity(["color_1", "color_2"], mode="PUSH"),
        weight=0.25,
    )
    prim.add_regularizer(
        "Area Range", AttributeRange("areas", min=0.1, max=0.25), weight=12.0
    )
    prim.add_regularizer(
        "Alpha Target", AttributeRange("alphas", min=0.95), weight=12.0
    )
    prim.add_regularizer("Verticality", AttributeRange("thetas", target=0), weight=50.0)

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
    )
    for _ in trainer.train():
        pass


def generate():
    # Load primitive
    prim = get_primitive()
    prim.load(run_folder / "primitive.pt")
    prim.requires_grad_(False)
    prim = prim.to(device)

    # Sample processor
    def proc_fn(sample, primitive):
        sample.weights *= ((sample.rgb - 0.5) ** 2).sum(dim=2)
        # sample.weights = sample.weights**2
        # sample.weights = sample.weights.max(dim=1).values.unsqueeze(-1) - sample.weights
        co = sample.co
        centroids = primitive.centroids
        dists = ((centroids[None, :, :] - co[:, None, :]) ** 2).sum(dim=-1)
        # sample.weights = sample.weights * torch.exp(-(dists))
        sample.weights = sample.weights * (torch.sin(dists * 10000) * 0.4 + 0.7)
        return sample

    sample_proc = FlexibleSampleProcessor(proc_fn)
    prim.add_sample_processor(sample_proc)

    # Sampler
    sampler = Sampler(
        3072,
        1024,
        patch_size=32,
        max_batch=10000000,
        rasterizer=None,  # ProbabilisticRasterizer(top_k=100),
        padding=(1536, 1536, 1024, 1024),
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
