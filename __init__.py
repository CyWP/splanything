"""Splanything - ComfyUI custom node for primitive-based image reconstruction.

Modules:
- primitives: Geometric primitives for image reconstruction
- losses: Loss functions for optimization
- callbacks: Training callbacks
- refinement: Refinement rules for adaptive optimization
- trainers: Training orchestration
- generators: Image generation from pretrained primitives
- utils: Utility functions

Usage:
    # Training
    python -m splanything train --config config.yaml

    # Generation
    python -m splanything generate --config gen.yaml
"""

__version__ = "0.1.0"

__all__ = [
    "main",
]


def main():
    """Entry point for CLI (python -m splanything)."""
    from cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    import sys

    sys.exit(main())
