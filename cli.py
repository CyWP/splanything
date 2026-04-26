#!/usr/bin/env python
"""CLI for training and generation tasks.

Usage:
    # Train
    python -m splanything train --config config.yaml
    python -m splanything train --config config.yaml --epochs 100
    python -m splanything train --config config.yaml --optimizer.lr 0.001

    # Generate
    python -m splanything generate --config gen.yaml
    python -m splanything generate --config gen.yaml --output out.png
    python -m splanything generate --config gen.yaml --resolution.H 512 --resolution.W 512
    python -m splanything generate --config gen.yaml --resolution.scale 2.0
"""

import argparse
import json
import os
import sys
import tempfile
import torch
import yaml


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base config."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_config(path: str) -> dict:
    """Load config from JSON or YAML file."""
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    elif path.endswith((".yaml", ".yml")):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Config must be .json, .yaml, or .yml: {path}")


def _apply_overrides(config_path: str, overrides: dict) -> str:
    """Apply CLI overrides to config file and return temp path."""
    config = _load_config(config_path)
    config = _deep_merge(config, overrides)

    ext = os.path.splitext(config_path)[1]
    with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
        if ext == ".json":
            json.dump(config, f)
        else:
            yaml.dump(config, f)
        return f.name


def train(args):
    """Run training task."""
    from trainers import Trainer
    from trainers.train_config import load_train_config

    config = {}
    if args.config:
        config = _load_config(args.config)

    if args.overrides:
        config = _deep_merge(config, args.overrides)

    target = config.pop("target", None)
    if target is not None and isinstance(target, dict):
        target = target.get("path")

    loaded = load_train_config(config, target=target)
    trainer = loaded["trainer"]
    if trainer is None:
        print(
            "Error: trainer not created. Check config has 'trainer' section.",
            file=sys.stderr,
        )
        return 1

    for state in trainer.train():
        pass
    return 0


def generate(args):
    """Run generation task."""
    from generators import Generator, load_gen_config
    from utils.img import ImgUtils

    config = {}
    if args.config:
        config = _load_config(args.config)

    if args.overrides:
        config = _deep_merge(config, args.overrides)

    loaded = load_gen_config(config)
    generator = loaded["generator"]
    primitive = loaded["primitive"]
    output_path = args.output or "output.png"
    with torch.no_grad():
        img = ImgUtils.tensor2pil(generator(primitive), normalized=False)
    img.save(output_path)
    print(f"Saved to {output_path}")
    return 0


def _parse_overrides(argv: list) -> dict:
    """Parse --key.subkey value arguments into nested dict."""
    overrides = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--"):
            key = arg[2:]
            if "=" in key:
                key, value = key.split("=", 1)
                value = _parse_value(value)
            else:
                i += 1
                if i >= len(argv) or argv[i].startswith("--"):
                    value = True
                else:
                    value = _parse_value(argv[i])
            parts = key.split(".")
            d = overrides
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
        i += 1
    return overrides


def _parse_value(value: str) -> any:
    """Parse string value to appropriate type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def main(argv: list = None):
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]

    known = {"train", "generate", "config", "output"}
    override_args = [
        a for i, a in enumerate(argv) if a.startswith("--") and _get_key(a) not in known
    ]
    if argv and not argv[0].startswith("--"):
        cmd_idx = next((i for i, a in enumerate(argv) if a in ("train", "generate")), 0)
        override_args = [
            a
            for a in argv[cmd_idx + 1 :]
            if a.startswith("--") and _get_key(a) not in known
        ]
    else:
        override_args = [
            a for a in argv if a.startswith("--") and _get_key(a) not in known
        ]

    parser = argparse.ArgumentParser(description="Splanything CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--config", help="Config file (JSON or YAML)")

    gen_parser = subparsers.add_parser("generate", help="Run generation")
    gen_parser.add_argument("--config", help="Config file (JSON or YAML)")
    gen_parser.add_argument("--output", help="Output image path")

    filtered_argv = [
        a
        for i, a in enumerate(argv)
        if not (a.startswith("--") and _get_key(a) not in known)
    ]
    args = parser.parse_args(filtered_argv)
    args.overrides = _parse_overrides(override_args) if override_args else {}

    if args.command == "train":
        return train(args)
    elif args.command == "generate":
        return generate(args)
    return 1


def _get_key(arg: str) -> str:
    """Extract key from --key or --key=value."""
    key = arg[2:]
    if "=" in key:
        key = key.split("=")[0]
    return key


if __name__ == "__main__":
    sys.exit(main())
