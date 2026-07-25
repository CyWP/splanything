# Splanything

Primitive-based image reconstruction library in PyTorch.

Splanything fits trainable geometric primitives (cubic gradients, Gaussians) to a target image through gradient descent optimization.

## Installation

```bash
pip install splanything
```

## Quickstart

```python
import torch
from splanything import CubicFanPrimitive, WeightedRasterizer
from splanything.training import Trainer
from splanything.training.train_sampler import TrainSampler
from splanything.losses import L2Loss
from splanything.callbacks import LoopControl

# Target image
H, W = 128, 128
target = torch.rand(1, 3, H, W)

# Primitive and rasterizer
primitive = CubicFanPrimitive(size=64)
rasterizer = WeightedRasterizer()

# Sampler feeds patches to the trainer
sampler = TrainSampler(
    primitive=primitive,
    target=target,
    patch_size=16,
    rasterizer=rasterizer,
)

optimizer = torch.optim.Adam(primitive.parameters(), lr=0.01)

# Train for 50 epochs
trainer = Trainer(
    name="my_run",
    sampler=sampler,
    optimizer=optimizer,
    losses={"l2": L2Loss(weight=1.0)},
    callbacks=[LoopControl(epochs=50)],
)

for state in trainer.train():
    print(f"epoch {state['epoch']}: loss={state['loss'].item():.4f}")
```

## Modules

- `splanything.primitives` — `CubicFanPrimitive`, `Gaussian`, `Primitive` base class
- `splanything.rasterizers` — aggregation strategies such as `WeightedRasterizer`, `ProbabilisticRasterizer`
- `splanything.losses` — `L1Loss`, `L2Loss`
- `splanything.callbacks` — training callbacks like `LoopControl`
- `splanything.refinement` — adaptive rules like `GradSplit`, `AlphaFilter`
- `splanything.training` — `Trainer`, `TrainSampler`
- `splanything.generators` — `Generator` for rendering at arbitrary resolutions
- `splanything.utils` — image utilities, optimizer wrapper, lazy evaluation

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
