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
    ThresholdFilter,
    GradSplit,
    IsoSplit,
    MapSplit,
    BoundsFilter,
    PrimitiveCeiling,
    PrimitiveFloor,
)
from splanything.training.losses import L2Loss, Anisotropy
from splanything.training.refinement.processors import MapCriterionProcessor
from splanything.utils.img import ImgUtils
from splanything.rendering import Sampler, SampleOutput
from splanything.rendering.processors import FlexibleSampleProcessor
from splanything.rendering.rasterizers import ProbabilisticRasterizer

device = torch.device("cuda:0")
run_name = "PorVCro"
base_folder = Path("../test_runs").resolve()
run_folder = base_folder / run_name


def get_primitive():
    radial = RadialFreqPrimitive(size=5).to(device)
    cubic = CubicFanPrimitive(size=3).to(device)
    gauss = GaussianPrimitive(size=1).to(device)
    multi = MultiPrimitive({"radial": radial, "cubic": cubic})  # , "gaussian": gauss})
    prim = MetaPrimitive(
        primitive=multi, size=100, primitive_trainable=False, scale_factor=1.0
    ).to(device)
    return prim


def train():
    # Images
    tgt = ImgUtils.pil2tensor(
        Image.open("../assets/por_cro_offside.png").convert("RGBA")
    ).to(device)
    msk_img = Image.open("../assets/por_cro_offside_masked.png")
    msk_img_blur40 = msk_img.filter(ImageFilter.GaussianBlur(radius=40))
    msk_img_blur10 = msk_img.filter(ImageFilter.GaussianBlur(radius=10))
    msk_tensor_blur40 = ImgUtils.pil2mask(msk_img_blur40, mode="A").to(device)
    msk_tensor_blur10 = ImgUtils.pil2mask(msk_img_blur10, mode="A").to(device)

    # primitive
    prim = get_primitive()

    # Rules
    alpha_cull = ThresholdFilter(attr_name="alphas", threshold=0.4, interval=47)
    grad_split = GradSplit(threshold=0.01, interval=103)
    area_split = ThresholdFilter(attr_name="areas", threshold=0.2, interval=15)
    map_split = MapSplit(msk_tensor_blur10 * 0.6 + 0.05, interval=187)
    ceiling = PrimitiveCeiling(800)
    bounds_cull = BoundsFilter(interval=10)
    # radial.add_filter_rule(alpha_cull)
    # radial.add_split_rule(grad_split)
    # cubic.add_filter_rule(alpha_cull)
    # cubic.add_split_rule(grad_split)
    prim.add_filter_rule(alpha_cull)
    prim.add_split_rule(grad_split)
    prim.add_split_rule(map_split)
    prim.add_split_rule(area_split)
    prim.add_filter_rule(ceiling)
    prim.add_filter_rule(bounds_cull)

    # Rule processors
    map_proc = MapCriterionProcessor(msk_tensor_blur10 * 0.6 + 0.4)
    grad_split.add_processor(map_proc)

    # Sampler
    sampler = TrainSampler(
        target=tgt,
        patch_size=128,
        max_batch=1000000,
        sampling_map=msk_tensor_blur40 * 0.2 + 0.05,
        low_vram=True,
    )

    # Callbacks
    train_callbacks = [PreviewWindow(frequency=5, show_target=True), StatsPanel()]

    # Optim
    optimizer = OptimizerWrapper(prim, AdamW, lr=0.01)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer._optimizer, T_0=1000, eta_min=0.001
    )

    # Trainer
    trainer = Trainer(
        run_name,
        prim,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        losses={"L2": L2Loss(), "Anisotropy": Anisotropy(weight=0.5)},
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
    def proc_fn(sample, processor):
        sample.weights = sample.weights**2
        return sample

    sample_proc = FlexibleSampleProcessor(proc_fn)
    # prim.add_sample_processor(sample_proc)

    # Sampler
    sampler = Sampler(
        1536,
        2048,
        patch_size=128,  # 512,
        max_batch=10000000,
        rasterizer=None,  # ProbabilisticRasterizer(),
        padding=(1024, 1024, 1024, 1024),
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
