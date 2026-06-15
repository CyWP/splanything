import os
import torch

LOW_VRAM = bool(os.environ.get("SPLANYTHING_LOW_VRAM", False))
DEVICE = torch.device(os.environ.get("SPLANYTHING_DEVICE", "cpu"))
