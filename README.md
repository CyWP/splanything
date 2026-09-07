# splanything

*Splat any differentiable function.*

`splanything` reconstructs images by fitting trainable geometric primitives — cubic fans,
Gaussians, radial-frequency splats, stars — to a target image through gradient descent.
Each primitive is a batch of splat instances with trainable parameters (position, shape,
rotation, color, opacity); the fitted primitive can then be re-rendered at any resolution.

## How it works

- **Primitive** — a trainable `nn.Module` holding one row of parameters per splat instance.
  Sampling it at coordinates yields per-primitive RGB values and weights. Primitives can be
  grouped in a `MultiPrimitive` or wrapped by a `MetaPrimitive` that applies per-splat
  affine transforms and color/alpha modulation to a child primitive.
- **Sampler** — covers the canvas with a patch grid and feeds coordinate batches to the
  primitive, keeping memory bounded via a coordinate-times-primitive budget. Batched
  parameter access is mask-aware: inside an active mask context, primitives only see the
  splats relevant to the current patch.
- **Sample processors** — optional transforms applied to the per-primitive samples before
  aggregation (distance/axis projections, color skewing, map-modulated weighting, ...).
- **Rasterizer** — aggregates the samples into RGBA (`WeightedRasterizer`,
  `ProbabilisticRasterizer`, or a weighted `MultiRasterizer` blend).
- **Trainer** — orchestrates the optimization loop: losses (per-sample L1/L2 or image-level
  L2/SSIM), attribute regularizers, refinement rules that cull faded splats (`FilterRule`)
  and split oversized ones (`SplitRule`) with the optimizer state kept in sync, an
  `OptimizerWrapper` supporting per-parameter learning-rate modifiers, and callbacks for
  live previews, statistics panels, and checkpointing.
- **Splimage** — the canonical image wrapper (BCHW tensor in [0, 1]) with lazy, cached
  conversion to numpy/PIL and mask extraction. `ImgUtils` provides the underlying image
  operations (patch extraction/assembly, coordinate grids, resampling).

## Installation

Requires Python >= 3.10 with PyTorch available.

```bash
pip install .
```

For development (tests, linting, building):

```bash
pip install -e ".[dev]"
```

## Examples

Complete usage scripts live in [`examples/`](examples/):

- `example_por_cro.py` — single `CubicFanPrimitive`, image-level losses, theta-map
  regularizer, and a high-resolution re-render.
- `example_nor_bra.py` — `MultiPrimitive` (star + radial frequency), per-sample losses,
  per-child refinement rules, and a high-resolution re-render.

Each script trains a primitive against an image from `assets/` (`-t/--train`) and then
loads the checkpoint to re-render it at a higher resolution with decorative sample
processors (`-g/--generate`).

## Modules

| Module | Contents |
| --- | --- |
| `splanything.primitives` | `Primitive` base class, `CubicFanPrimitive`, `GaussianPrimitive`, `RadialFreqPrimitive`, `StarPrimitive`, `MetaPrimitive`, `MultiPrimitive`; `initializers/` and `splitters/` |
| `splanything.rendering` | `Sampler`, `SampleOutput`; `rasterizers/`, `processors/` |
| `splanything.training` | `Trainer`, `TrainSampler`, `OptimizerWrapper`; `losses/`, `callbacks/`, `refinement/`, `regularizers/` |
| `splanything.utils` | `ImgUtils` static image ops and the `Splimage` image wrapper |

The top-level package exposes only these subpackages plus `ImgUtils`; import classes from
their subpackages.

## License

MIT
