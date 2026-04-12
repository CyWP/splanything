"""Full training configuration from JSON/YAML.

Exposes:
- load_train_config: Load and instantiate full training setup from config file or dict
"""

from __future__ import annotations

import torch
import json
import yaml

from typing import Dict, Any, Optional, Union, List
from jaxtyping import Float
from torch import Tensor

from primitives import Primitive, get_primitive
from losses import Loss, get_loss
from callbacks import Callback, get_callback
from refinement import RefinementRule, get_refinement_rule
from .trainer import Trainer
from utils.pytorch import init_optimizer, init_scheduler, get_device
from utils.img import ImgUtils


PathLike = Union[str, None]


def load_train_config(
    config: Union[Dict[str, Any], str],
    target: Optional[Union[str, Float[Tensor, "B C H W"]]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Load training configuration and instantiate all objects.

    Takes a config dict or file path (JSON/YAML) and returns a dictionary
    containing instantiated primitives, losses, callbacks, optimizer, scheduler,
    and optionally a Trainer instance.

    Config Structure:
        primitive:
            CubicGrad:
                size: 1000

        optimizer:
            name: adam
            lr: 0.01
            weight_decay: 0.0001

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
            - ImgUpdate:
                frequency: 10
            - LossLogger: {}

        refinement:
            - GradSplit:
                threshold: 0.05
                interval: 10

        trainer:
            patch_size: 32

        target:
            path: /path/to/image.png

    Args:
        config: Config dict or path to JSON/YAML file.
        target: Optional target image tensor (B, C, H, W). Can also be
            specified as "target.path" in config.
        device: Optional device string (e.g., "cuda", "mps", "cpu"). If None,
            uses get_device() as fallback.

    Returns:
        Dict with keys:
            - primitive: Instantiated Primitive
            - losses: Dict[str, Loss]
            - callbacks: List[Callback]
            - optimizer: torch.optim.Optimizer
            - scheduler: Optional scheduler
            - trainer: Trainer instance (if trainer config specified)

    Raises:
        ValueError: If target is not provided and not in config, or if
            file extension is invalid.
    """
    if isinstance(config, str):
        config = _load_config_from_file(config)

    _validate_config(config)

    target = _resolve_target(config, target)

    # Resolve device: config takes priority, otherwise use function argument
    device = config.get("device", device)
    device = torch.device(device) if device is not None else get_device()

    primitive = config.get("primitive", {})
    primitive = _build_primitive(primitive)

    losses = config.get("losses", {})
    losses = _build_losses(losses)

    callbacks = config.get("callbacks", {})
    callbacks = _build_callbacks(callbacks)

    refinement = config.get("refinement", {})
    refinement_rules = _build_refinement(refinement, primitive)

    params = list(primitive.parameters())
    optimizer = config.get("optimizer", {})
    optimizer_name = optimizer.pop("name", "adam")
    optimizer = init_optimizer(optimizer_name, params, **optimizer)

    scheduler = None
    scheduler = config.get("scheduler")
    if scheduler:
        scheduler_name = scheduler.pop("name")
        scheduler = init_scheduler(scheduler_name, optimizer, **scheduler)

    trainer = None
    trainer = config.get("trainer")
    if trainer and target is not None:
        trainer = Trainer(
            target=target,
            primitive=primitive,
            optimizer=optimizer,
            losses=losses,
            callbacks=callbacks,
            scheduler=scheduler,
            refinements=refinement_rules,
            device=device,
            **trainer,
        )

    return {
        "primitive": primitive,
        "losses": losses,
        "callbacks": callbacks,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "trainer": trainer,
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


def _build_losses(cfg: List[Dict[str, Any]]) -> Dict[str, Loss]:
    """Build losses from config list.

    Args:
        cfg: List of dicts, each with single key being loss name and value being kwargs.

    Returns:
        Dict mapping name to Loss instance.
    """
    losses = {}
    for item in cfg:
        if len(item) != 1:
            raise ValueError("Each loss entry must have exactly one key-value pair.")
        name, kwargs = next(iter(item.items()))
        losses[name] = get_loss(name, kwargs)
    return losses


def _build_callbacks(cfg: List[Dict[str, Any]]) -> List[Callback]:
    """Build callbacks from config list.

    Args:
        cfg: List of dicts, each with single key being callback name and value being kwargs.

    Returns:
        List of Callback instances.
    """
    callbacks = []
    for item in cfg:
        if len(item) != 1:
            raise ValueError(
                "Each callback entry must have exactly one key-value pair."
            )
        name, kwargs = next(iter(item.items()))
        callbacks.append(get_callback(name, kwargs))
    return callbacks


def _build_refinement(
    cfg: List[Dict[str, Any]], primitive: Primitive
) -> List[RefinementRule]:
    """Build refinement rules from config list.

    Args:
        cfg: List of dicts, each with single key being rule name and value being kwargs.
        primitive: The primitive to apply rules to.

    Returns:
        List of RefinementRule instances.
    """
    rules = []
    for item in cfg:
        if len(item) != 1:
            raise ValueError(
                "Each refinement entry must have exactly one key-value pair."
            )
        name, kwargs = next(iter(item.items()))
        rules.append(get_refinement_rule(name, kwargs, primitive))
    return rules


def _resolve_target(
    config: Dict[str, Any],
    target: Optional[Union[str, Float[Tensor, "B C H W"]]],
) -> Optional[Float[Tensor, "B C H W"]]:
    """Resolve target from config or argument.

    Args:
        config: Config dictionary.
        target: Directly passed target tensor.

    Returns:
        Resolved target tensor or None if not available.

    Raises:
        ValueError: If target path is invalid.
    """
    if isinstance(target, Tensor):
        return target

    if target is None:
        target = config.get("target")
        if target is None:
            return None

    if isinstance(target, str):
        return ImgUtils.load_image(target)

    if isinstance(target, dict):
        path = target.get("path")
        mode = target.get("mode", "RGBA")
        normalize = target.get("normalize", False)
        if path is None:
            raise ValueError("target config must contain 'path' key.")
        return ImgUtils.load_image(path, mode=mode, normalize=normalize)

    raise ValueError(f"Invalid target config: {target}")


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
    if "optimizer" not in config:
        raise ValueError("Config must contain 'optimizer' section.")
    if "losses" not in config:
        raise ValueError("Config must contain 'losses' section.")
