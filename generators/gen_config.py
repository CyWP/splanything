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
            CubicGrad:
                path: /path/to/primitive.pt
            # OR
            CubicGrad:
                size: 1000  # Only for generating random init

        resolution:
            H: 1024  # explicit height
            W: 512   # explicit width
            # OR
            scale: 2.0  # scale relative to trained aspect ratio
            # OR
            scale_to: 2048  # scale so max dimension is 2048

        patch_size: 64

        checkpoint:
            path: /path/to/checkpoint.pt
            # OR loaded externally and passed as dict

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

    _validate_config(config)

    # Resolve device
    device = config.get("device", device)
    device = torch.device(device) if device is not None else get_device()

    # Build primitive
    primitive_cfg = config.get("primitive", {})
    primitive = _build_primitive(primitive_cfg)

    # Load checkpoint if provided
    checkpoint = _resolve_checkpoint(config, checkpoint, primitive)
    if checkpoint is not None:
        primitive.load_state_dict(checkpoint)

    # Resolve resolution
    H, W = _resolve_resolution(config, primitive)

    patch_size = config.get("patch_size", 64)

    generator = Generator(
        primitive=primitive,
        H=H,
        W=W,
        patch_size=patch_size,
        device=device,
    )

    return {
        "generator": generator,
        "primitive": primitive,
    }


def _build_primitive(cfg: Dict[str, Any]) -> Primitive:
    """Build primitive from config.

    Args:
        cfg: Dict with single key being class name and value being kwargs.

    Returns:
        Primitive instance.
    """
    if len(cfg) != 1:
        raise ValueError("Primitive config must have exactly one entry.")
    name, kwargs = next(iter(cfg.items()))
    return get_primitive(name, kwargs)


def _resolve_resolution(
    config: Dict[str, Any], primitive: Primitive
) -> tuple[int, int]:
    """Resolve output resolution from config.

    Supports three modes:
    1. Explicit H, W
    2. Scale relative to trained aspect ratio
    3. Scale so max dimension equals a value

    Args:
        config: Config dictionary.
        primitive: Primitive to get trained aspect ratio from.

    Returns:
        Tuple of (H, W).
    """
    resolution = config.get("resolution", {})

    # Explicit H, W
    if "H" in resolution and "W" in resolution:
        return resolution["H"], resolution["W"]

    # Scale factor
    if "scale" in resolution:
        scale = resolution["scale"]
        H = primitive._trained_H
        W = primitive._trained_W
        return int(H * scale), int(W * scale)

    # Scale to max dimension
    if "scale_to" in resolution:
        max_dim = resolution["scale_to"]
        H = primitive._trained_H
        W = primitive._trained_W
        if H >= W:
            new_H = max_dim
            new_W = int(W * (max_dim / H))
        else:
            new_W = max_dim
            new_H = int(H * (max_dim / W))
        return new_H, new_W

    # Fallback to trained resolution
    H = getattr(primitive, "_trained_H", 512)
    W = getattr(primitive, "_trained_W", 512)
    return H, W


def _resolve_checkpoint(
    config: Dict[str, Any],
    checkpoint: Union[str, Dict[str, Any], None],
    primitive: Primitive,
) -> Optional[Dict[str, Any]]:
    """Resolve checkpoint from config or argument.

    Args:
        config: Config dictionary.
        checkpoint: Directly passed checkpoint path or state dict.
        primitive: Primitive to load checkpoint into.

    Returns:
        Checkpoint state dict or None if not available.
    """
    if checkpoint is None:
        checkpoint_cfg = config.get("checkpoint")
        if checkpoint_cfg is None:
            return None
        if isinstance(checkpoint_cfg, str):
            checkpoint_path = checkpoint_cfg
        elif isinstance(checkpoint_cfg, dict):
            return checkpoint_cfg
        else:
            checkpoint_path = checkpoint_cfg.get("path")
        if checkpoint_path is None:
            return None
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    elif isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location="cpu")

    if isinstance(checkpoint, dict):
        return checkpoint

    return None


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


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate required config sections.

    Args:
        config: Config dictionary to validate.

    Raises:
        ValueError: If required sections are missing.
    """
    if "primitive" not in config:
        raise ValueError("Config must contain 'primitive' section.")
