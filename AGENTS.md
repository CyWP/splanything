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
│       │   ├── generic.py   # Base Primitive class
│       │   ├── cubic_grad.py
│       │   ├── gaussian.py
│       │   └── protocols.py
│       ├── rasterizers/   # Sample aggregation strategies
│       │   ├── generic.py
│       │   ├── sample_out.py
│       │   ├── weighted.py
│       │   ├── probabilistic.py
│       │   └── ...
│       ├── losses/        # Loss functions
│       │   ├── generic.py
│       │   ├── l1.py
│       │   ├── l2.py
│       │   └── ssim.py
│       ├── callbacks/     # Training callbacks
│       │   ├── generic.py
│       │   ├── loop_control.py
│       │   ├── preview_window.py
│       │   ├── primitive_checkpoint.py
│       │   └── losses_log.py
│       ├── refinement/    # Refinement rules
│       │   ├── generic.py
│       │   ├── grad_split.py
│       │   ├── area_split.py
│       │   ├── alpha_cull.py
│       │   └── iso_split.py
│       ├── training/      # Training orchestration
│       │   ├── trainer.py
│       │   └── train_sampler.py
│       ├── generators/    # Image generation
│       │   ├── __init__.py
│       │   └── generator.py
│       ├── utils/         # Utilities
│       │   ├── img.py
│       │   ├── pytorch.py
│       │   ├── lazy/
│       │   ├── math.py
│       │   ├── tkinter.py
│       │   └── types.py
│       └── __init__.py
├── tests/                 # Pytest tests and example usage scripts
├── pyproject.toml
├── README.md
├── LICENSE
└── MANIFEST.in
```

## Scope

- **Core Framework**: `Trainer` class managing optimization loops with callbacks
- **Primitives**: Geometric image representations (`CubicGrad`, `Gaussian`) as trainable modules
- **Rasterizers**: Strategies for aggregating per-primitive samples into RGBA output
- **Loss Functions**: L1 and L2 losses for image comparison
- **Callbacks**: Loop management, checkpoints, previews, logging
- **Refinement Rules**: Adaptive optimization (`GradSplit`, `AreaSplit`, `AlphaCull`, `IsoSplit`)
- **Generators**: Image generation from pretrained primitives at arbitrary resolutions
- **Utilities**: Image processing, optimizer wrapper, optional lazy property evaluation (external/user preference)

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
  from splanything import CubicGrad, WeightedRasterizer
  from splanything.training import Trainer, TrainSampler
  from splanything.losses import L2Loss
  ```
- Top-level `splanything.__init__.py` exposes the main public API

### Training
- `Trainer` takes `name`, `sampler`, `optimizer`, `losses`, `callbacks`
- `TrainSampler` handles patch creation and feeds `(output, target)` pairs to the trainer
- Saves trainer state, primitive state, and logs at end of training
- Handles `KeyboardInterrupt` gracefully
- Callbacks triggered at: `TRAIN_START`, `TRAIN_END`, `EPOCH_START`, `EPOCH_END`, `PRE_STEP`

### Generation
- `Generator` renders a pretrained primitive at a specified resolution
- Requires `H`, `W`, `patch_size` for generation

### Callbacks
- Base `Callback` class with `run(trainer, stage)` method
- `LoopControl` uses tqdm for progress display
- `PrimitiveCheckpoint` saves at interval using `trainer.save_checkpoint(epoch)`

### Refinement Rules
- Inherit from `RefinementRule` (which also inherits `Callback`)
- Run at `EPOCH_END`, modify primitive in-place
- Require specific behaviors via protocols (e.g., `Splittable`, `HasAlphas`)

### Lazy Evaluation (external / optional)
- `utils.lazy` provides `@lazy_tree` for users who want cached properties with dependency tracking
- Core primitives do not use `@lazy_tree`; properties compute on every access
- Use `clear_all_caches()` to invalidate all registered lazy_tree instances

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
