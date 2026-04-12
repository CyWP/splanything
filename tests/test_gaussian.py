import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from primitives.gaussian import Gaussian

g = Gaussian(size=5)
print(f"len: {len(g)}")
print(f"thetas shape: {g.thetas.shape}")
print(f"R shape: {g.R.shape}")
print(f"axes shapes: {g.axes[0].shape}, {g.axes[1].shape}")
print(f"areas shape: {g.areas.shape}")

# Test sample
co = torch.rand((100, 2))
result = g.sample(co)
print(f"sample result shape: {result.shape}")
