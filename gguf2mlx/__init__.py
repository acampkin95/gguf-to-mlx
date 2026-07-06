"""
GGUF to MLX Converter — vendored internal module.

Upstream: https://github.com/barrontang/gguf2mlx (MIT, Barron Tang)
Vendored into this project for self-contained maintenance.

Public API mirrors the upstream package:
    convert()            — full GGUF → MLX conversion
    detect_architecture() — identify model architecture from GGUF metadata
    build_config()       — build MLX-compatible config.json
    extract_tokenizer()   — extract and save tokenizer files
"""

__version__ = "2.0.2+vendored"

from .core import convert, detect_architecture, build_config, extract_tokenizer, main

__all__ = ["convert", "detect_architecture", "build_config", "extract_tokenizer", "main"]
