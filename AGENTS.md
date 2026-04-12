# AGENTS.md - splanything

NOTE: Use your skills as much as possible, they include a lot of information on how to program for such a project.

## Project Overview

`splanything` is a ComfyUI custom node that provides a training framework for reconstructing images using trainable geometric primitives. It fits mathematical primitives (e.g., cubic gradients) to a target image through gradient descent optimization.

## Project Structure

```
splanything/
├── primitives/          # Geometric primitives for image reconstruction
│   ├── generic.py        # Base Primitive class
│   ├── cubic_grad.py    # CubicGrad implementation
│   └── protocols.py     # Splittable, HasAreas, HasAlphas protocols
├── losses/              # Loss functions
│   ├── l1.py           # L1 (MAE) loss
│   ├── l2.py           # L2 (MSE) loss
│   └── ssim.py         # SSIM loss
├── callbacks/          # Training callbacks
│   ├── generic.py              # Base Callback class
│   ├── loop_control.py         # Base loop control (tqdm)
│   ├── comfy_control.py         # ComfyUI-specific loop control
│   ├── img_update.py           # NodePreview callback
│   ├── preview_window.py       # PreviewWindow callback
│   ├── primitive_checkpoint.py  # Checkpoint saving callback
│   └── losses_log.py          # Loss logging callback
├── refinement/         # Refinement rules
│   ├── generic.py       # Base RefinementRule class
│   ├── grad_split.py    # Gradient-based splitting
│   └── alpha_cull.py   # Alpha-based culling
├── trainers/           # Training orchestration
│   ├── trainer.py       # Trainer class
│   └── train_config.py  # load_train_config factory
├── generators/          # Image generation
│   └── generator.py    # Generator class
├── utils/              # Utilities
│   ├── img.py          # Image processing
│   ├── pytorch.py      # Optimizer/scheduler init
│   ├── comfy.py        # ComfyUI integration
│   ├── lazy/           # Lazy property evaluation
│   ├── math.py         # Math utilities
│   └── types.py        # Type utilities
├── examples/           # Example configs
├── cli.py              # CLI entry point
└── __init__.py        # Main module
```

## Scope

- **Core Framework**: Trainer class managing optimization loops with callbacks
- **Primitives**: Geometric image representations (e.g., `CubicGrad`) as trainable modules
- **Loss Functions**: L1, L2, and SSIM losses for image comparison
- **Callbacks**: Loop management, checkpoints, previews, logging
- **Refinement Rules**: Adaptive optimization (GradSplit, AlphaCull)
- **Generators**: Image generation from pretrained primitives
- **Utilities**: Image processing, PyTorch optimizers/schedulers, ComfyUI integration, lazy property evaluation

## Goals

1. Enable primitive-based image reconstruction in ComfyUI workflow
2. Provide extensible training pipeline with modular losses and callbacks
3. Support real-time image preview during training
4. Integrate seamlessly with ComfyUI's execution model
5. CLI for training and generation tasks

## Implementation Notes

### Type Annotations
- Use `jaxtyping` for tensor type annotations
- Format: `Float[Tensor, "B C H W"]` (short uppercase dims, Tensor imported from torch)
- See `python-jaxtyping` skill for details

### Factory Functions
- Each module has factory function that takes `(name: str, kwargs: dict)` and returns single instance
- Keys are compared case-insensitively (lowercase)
- Config format uses lists for multiple items:
  ```yaml
  losses:
    - L1:
        weight: 1.0
    - L2:
        weight: 0.5
  ```

### Training
- `Trainer` takes `name`, `base_folder`, creates `run_folder` for outputs
- Saves trainer state, primitive state, and logs at end of training
- Handles `KeyboardInterrupt` gracefully
- Callbacks triggered at: `TRAIN_START`, `TRAIN_END`, `EPOCH_START`, `EPOCH_END`, `PRE_STEP`

### Generation
- `Generator` loads pretrained checkpoint and rasterizes at specified resolution
- Requires `H`, `W`, `patch_size` for generation

### Callbacks
- Base `Callback` class with `run(trainer, stage)` method
- `LoopControl` uses tqdm, `ComfyUIControl` inherits for ComfyUI integration
- `PrimitiveCheckpoint` saves at interval using `trainer.save_checkpoint(epoch)`

### Refinement Rules
- Inherit from `RefinementRule` (which also inherits `Callback`)
- Run at `EPOCH_END`, modify primitive in-place
- Require specific behaviors via protocols (e.g., `Splittable`, `HasAlphas`)

### Lazy Evaluation
- `@lazy_tree` decorator caches properties and auto-invalidates on dependency changes
- Use `clear_all_caches()` to invalidate all registered instances

### CLI Usage
```bash
# Training
python -m splanything train --config config.yaml
python -m splanything train --config config.yaml --epochs 100
python -m splanything train --config config.yaml --optimizer.lr 0.001

# Generation
python -m splanything generate --config gen.yaml
python -m splanything generate --config gen.yaml --output out.png
```

### Config Format
```yaml
primitive:
  CubicGrad:
    size: 1000

optimizer:
  name: adam
  lr: 0.01

scheduler:
  name: cosine
  T_max: 100

losses:
  - L1:
      weight: 1.0
  - L2:
      weight: 0.5

callbacks:
  - LoopControl:
      epochs: 100
  - PrimitiveCheckpoint:
      interval: 10
  - LossLogger: {}

refinement:
  - GradSplit:
      threshold: 0.05
      interval: 10

trainer:
  patch_size: 32

target:
  path: image.png
  mode: RGBA
  normalize: false
```
