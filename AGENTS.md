# AGENTS.md - splanything

NOTE: Use yor skills as much as possible, they include a lot of information on how to program for such a project.

## Project Overview

`splanything` is a ComfyUI custom node that provides a training framework for reconstructing images using trainable geometric primitives. It fits mathematical primitives (e.g., cubic gradients) to a target image through gradient descent optimization.

## Scope

- **Core Framework**: Trainer class managing optimization loops with callbacks
- **Primitives**: Geometric image representations (e.g., `CubicGrad`) as trainable modules
- **Loss Functions**: L1, L2, and SSIM losses for image comparison
- **Callbacks**: LoopControl (epoch management), ImgUpdate (preview), LossLogger
- **Utilities**: Image processing, PyTorch optimizers/schedulers, ComfyUI integration, lazy property evaluation

## Goals

1. Enable primitive-based image reconstruction in ComfyUI workflow
2. Provide extensible training pipeline with modular losses and callbacks
3. Support real-time image preview during training
4. Integrate seamlessly with ComfyUI's execution model

## Implementation Notes

- Uses `torch` for all tensor operations
- Depends on `comfy.model_management` for device/interrupt handling
- Primitives implement `sample()` to generate patches from coordinates
- Lazy evaluation via `@lazy_tree` decorator for caching properties