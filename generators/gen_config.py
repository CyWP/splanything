"""Generator configuration from JSON/YAML.

Exposes:
- load_gen_config: Load and instantiate generator from config file or dict
"""

from __future__ import annotations

import torch
import json
import yaml
import os

from typing import Dict, Any, Optional, Union
from jaxtyping import Float
from torch import Tensor

from primitives import Primitive, get_primitive
from .generator import Generator
from rasterizers import get_rasterizer, Rasterizer
from utils.pytorch import get_device


GEN_START = "gen_start"
GEN_END = "gen_end"
GEN_STAGES = [GEN_START, GEN_END]


def load_gen_config(
    config: Union[Dict[str, Any], str],
    checkpoint: Union[str, Dict[str, Any], None] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Load generator configuration and instantiate generator.

    Takes a config dict or file path (JSON/YAML) and returns a dictionary
    containing an instantiated generator.

    Config Structure:
        primitive:
            name: CubicGrad
            path: blabla.pt

        H: 1024
        W: 1024

        patch_size: 64

        rasterizer:
            name: probabilisticrasterizer

    Args:
        config: Config dict or path to JSON/YAML file.
        checkpoint: Optional checkpoint path or state dict. Can also be
            specified as "checkpoint.path" in config.
        device: Optional device string. Uses get_device() fallback if None.

    Returns:
        Dict with keys:
            - generator: Generator instance
            - primitive: Loaded primitive

    Raises:
        ValueError: If config is invalid or checkpoint not found.
    """
    if isinstance(config, str):
        config = _load_config_from_file(config)
    device = get_device(config.get("device", device))
    primitive = get_primitive(**config["primitive"]).to(device).eval()
    H, W = config.get("H", 1024), config.get("W", 1024)
    patch_size = config.get("patch_size", 64)

    # Build rasterizer
    rasterizer_cfg = config.get("rasterizer", {"name": "weightedrasterizer"})
    rasterizer = get_rasterizer(**rasterizer_cfg)

    generator = Generator(
        H=H,
        W=W,
        patch_size=patch_size,
        rasterizer=rasterizer,
    )

    return {
        "generator": generator,
        "primitive": primitive,
    }


def _load_config_from_file(path: str) -> Dict[str, Any]:
    """Load config from JSON or YAML file.

    Args:
        path: Path to config file.

    Returns:
        Config dictionary.

    Raises:
        ValueError: If file extension is not .json or .yaml/.yml.
    """
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    elif path.endswith((".yaml", ".yml")):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Config file must be .json, .yaml, or .yml: {path}")
