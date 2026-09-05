# AGENTS.md - splanything

NOTE: Use your skills as much as possible, they include a lot of information on how to program for such a project.

## Project Overview

`splanything` is a PyPI library for reconstructing images using trainable geometric primitives. It fits mathematical primitives (e.g., cubic gradients, Gaussians) to a target image through gradient descent optimization.

## Project Structure

```
splanything/
├── src/
│   └── splanything/       # Main package
│       ├── primitives/    # Geometric primitives for image reconstruction
│       │   ├── base.py       # Base Primitive class
│       │   ├── cubic_fan.py  # CubicFanPrimitive
│       │   ├── gaussian.py   # GaussianPrimitive
│       │   ├── radial_freq.py
│       │   ├── star.py
│       │   ├── bezier.py
│       │   ├── meta.py       # MetaPrimitive
│       │   ├── multi.py      # MultiPrimitive (container)
│       │   ├── initializers/ # Primitive initialization strategies
│       │   └── splitters/    # Parameter splitting strategies
│       ├── rendering/     # Rendering pipeline
│       │   ├── sampler.py       # Base Sampler (rasterize / render)
│       │   ├── sample_output.py # SampleOutput dataclass
│       │   ├── rasterizers/     # Sample aggregation strategies
│       │   │   ├── base.py
│       │   │   ├── weighted.py
│       │   │   ├── probabilistic.py
│       │   │   └── multi.py
│       │   └── processors/      # Sample transform processors
│       │       ├── base.py
│       │       ├── vec.py
│       │       ├── dist.py
│       │       ├── color_skew.py
│       │       ├── mapped.py
│       │       ├── multi.py
│       │       └── flex.py
│       ├── training/      # Training orchestration
│       │   ├── trainer.py       # Trainer + TrainerLogHandler/Formatter
│       │   ├── sampler.py       # TrainSampler
│       │   ├── optimizer.py     # OptimizerWrapper
│       │   ├── stages.py        # Stage name constants
│       │   ├── losses/          # Loss functions
│       │   │   ├── base.py      # Loss / ImageLoss bases
│       │   │   ├── l1.py, l2.py
│       │   │   ├── l1_image.py, l2_image.py
│       │   │   └── ssim.py
│       │   ├── callbacks/       # Training callbacks
│       │   │   ├── base.py
│       │   │   ├── loop_control.py
│       │   │   ├── preview_window.py
│       │   │   ├── primitive_checkpoint.py
│       │   │   ├── primitive_save.py
│       │   │   └── panel.py
│       │   ├── refinement/      # Refinement rules
│       │   │   ├── base.py      # RefinementRule / FilterRule / SplitRule / CriterionProcessor
│       │   │   ├── rules/       # Threshold*, GradSplit, IsoSplit, Map*, BoundsFilter, ...
│       │   │   └── processors/  # Criterion processors
│       │   └── regularizers/    # Attribute regularizers
│       │       ├── base.py
│       │       ├── attr_attractor.py
│       │       ├── attr_map.py
│       │       ├── attr_proximity.py
│       │       └── attr_range.py
│       ├── utils/         # Utilities
│       │   └── img.py      # ImgUtils + Splimage image wrapper
│       └── __init__.py
├── examples/               # Full usage scripts
├── assets/                 # Images used by examples/tests
├── tests/                  # Pytest tests
├── pyproject.toml
├── README.md
├── LICENSE
└── MANIFEST.in
```

## Scope

- **Core Framework**: `Trainer` class managing optimization loops with callbacks
- **Primitives**: Geometric image representations (`CubicFanPrimitive`, `GaussianPrimitive`, `RadialFreqPrimitive`, `StarPrimitive`, `MultiPrimitive`, `MetaPrimitive`) as trainable modules
- **Rendering**: `Sampler` (patch grid + batching), `Rasterizer` strategies for aggregating per-primitive samples into RGBA, `SampleOutput` data flow, sample `Processor` transforms
- **Loss Functions**: per-sample (`L1Loss`, `L2Loss`) and image-level (`L1ImageLoss`, `L2ImageLoss`, `SSIMImageLoss`) losses
- **Callbacks**: Loop management, checkpoints, previews, logging
- **Refinement Rules**: Adaptive optimization (`FilterRule`/`SplitRule` subclasses such as `ThresholdFilter`, `GradSplit`, `IsoSplit`, `BoundsFilter`)
- **Regularizers**: Attribute-based parameter regularization (`AttributeProximity`, `AttributeRange`, `AttributeMap`, `AttributeAttractor`)
- **Utilities**: `Splimage` cached image wrapper, `ImgUtils` image ops, optimizer wrapper

## Goals

1. Provide a clean, composable Python API for primitive-based image reconstruction
2. Enable direct class instantiation without factory functions or YAML configs
3. Support extensible training pipelines with modular losses and callbacks
4. Package and distribute via PyPI

## Implementation Notes

### Type Annotations
- Use `jaxtyping` for tensor type annotations
- Format: `Float[Tensor, "B C H W"]` (short uppercase dims, `Tensor` imported from `torch`)
- See `python-jaxtyping` skill for details

### Package API
- No factory functions or class registries
- Users import classes directly and compose them explicitly:
  ```python
  from splanything.primitives import CubicFanPrimitive
  from splanything.rendering.rasterizers import WeightedRasterizer
  from splanything.training import Trainer, TrainSampler, OptimizerWrapper
  from splanything.training.losses import L2Loss
  from splanything.utils.img import ImgUtils, Splimage
  ```
- Top-level `splanything.__init__.py` exposes only the subpackages (`primitives`, `rendering`, `training`) plus `ImgUtils`; import classes from their subpackage modules

### Images
- `Splimage` (in `utils/img.py`) is the canonical image wrapper: BCHW torch tensor in [0, 1], lazy conversion to numpy/PIL/mask, cached and invalidated on mutation
- `ImgUtils` provides static image ops (`extract_image_patches`, `assemble_patches`, `get_patches`, `gen_px_coords`, ...)
- `TrainSampler` / `MapFilter` / `MapSplit` expect `Splimage` instances, not raw tensors

### Training
- `Trainer` takes `name`, `primitive`, `sampler`, `optimizer`, `losses`, `callbacks`, `base_folder`, optional `scheduler`
- `losses` is a `Dict[str, Tuple[Callable, float]]` mapping a name to `(loss_fn, weight)`
- `Loss` (per-sample) and `ImageLoss` (full BCHW image) are distinct bases; the trainer dispatches per-sample patches vs. full-image rendering based on which types are present
- `TrainSampler(target=Splimage, patch_size=...)` handles patch creation and feeds `(output, target, batch_co)` batches to the trainer; optional `sampling_map` (Splimage) does per-pixel Bernoulli subsampling
- `OptimizerWrapper(primitive, OptimizerClass, **kwargs)` wraps a torch optimizer; `filter`/`split` keep its state aligned with the primitive
- Saves trainer state, primitive state, and logs at end of training
- Handles `KeyboardInterrupt` gracefully
- Callbacks triggered at: `TRAIN_START`, `TRAIN_END`, `EPOCH_START`, `EPOCH_END`, `PRE_STEP`, `BATCH_START`, `BATCH_END`
- Regularizers are attached to a primitive via `primitive.add_regularizer(name, regularizer, weight=...)` (`AttributeProximity`, `AttributeRange`, `AttributeMap`, `AttributeAttractor`)

### Rendering
- `Sampler` (in `rendering/sampler.py`) renders a primitive over a patch grid: `rasterize()` returns a BCHW tensor, `render()` returns a `Splimage`
- Requires `H`, `W`, `patch_size` for rendering at a specified resolution
- `TrainSampler` subclasses it for training with target extraction and subsampling

### Callbacks
- Base `Callback` ABC: subclasses define `_stages` (class attribute) and implement `run(trainer, stage)`; `__call__` dispatches only when the stage is in `_stages`
- `PrimitiveCheckpoint` saves at interval using `trainer.save_checkpoint(epoch)`

### Refinement Rules
- Inherit from `FilterRule` (culling) or `SplitRule` (repopulation), both under `RefinementRule`; `CriterionProcessor` transforms criteria before judgement
- Attached to a primitive via `primitive.add_filter_rule(rule)` / `primitive.add_split_rule(rule)`
- Applied by the trainer around the optimizer step: `check_filter()` first, then `check_split()` on the filtered primitive; optimizer state is kept in sync
- Rely on duck-typing for required primitive behaviors (e.g., `areas`, `alphas`, `scales`, `split()`)
- String literals for method names must be fully uppercase: `"ALL"`, `"ANY"`, `"OVER"`, `"UNDER"`
- `BoundsFilter` margin always defines the **outer** cull zone: primitives within `margin` of the image border are culled

### No CLI / No Config Loader
- The package is a library; users write Python scripts
- Factory functions and YAML config loaders have been removed
- ComfyUI-specific code has been removed

### Development
```bash
pip install -e ".[dev]"
pytest
python -m build
```
