# GGUF → MLX Converter

**Easy one-step pipeline for converting GGUF models to Apple MLX format with dynamic quantization.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-orange)](https://github.com/acampkin95/gguf2mlx)

---

## Overview

This tool wraps the powerful `gguf2mlx` conversion engine with a user-friendly CLI that handles:

- **Auto-detection** of model architecture (Llama, Gemma, Mistral, etc.)
- **Smart quantization** recommendations based on available RAM
- **Apple Silicon optimization** for M-series Macs
- **One-command operation** with sensible defaults

## Quick Start

```bash
# Guided mode (prompts for input)
python3 convert.py

# Direct mode
python3 convert.py model.gguf

# Custom output directory
python3 convert.py model.gguf ./output/

# Use specific quantization
python3 convert.py model.gguf --bits 4

# Float16 only (no quantization)
python3 convert.py model.gguf --no-quantize

# Inspect model metadata
python3 convert.py model.gguf --inspect
```

## Features

| Feature | Description |
|---------|-------------|
| 🔍 **Smart Defaults** | Auto-detects architecture and recommends optimal quantization |
| 💾 **RAM-Based Suggestions** | Recommends bit-depth based on available system memory |
| 📊 **Progress Tracking** | Real-time conversion progress with ETA |
| 🎛️ **Presets** | Quality, balanced, speed, or custom configurations |
| 🛡️ **Safety Checks** | Validates disk space before conversion |
| ⚡ **mlx_lm Integration** | Direct generation after conversion |

## Quantization Options

| Bits | Quality | Best For |
|------|---------|----------|
| 16 | Float16 (no quant) | Maximum quality, ≥32GB RAM |
| 8 | Q8_0 | High quality, ≥16GB RAM |
| 4 | Q4_K_M | Balanced, ≥8GB RAM |
| 2 | Q2_K | Low memory, minimal RAM |

## Supported Architectures

- **Llama** (including MoE variants)
- **Gemma** (2, 3, 4 including MoE)
- **Mistral** & **Mixtral**
- **Qwen** (2, 3, including MoE)
- **DeepSeek** (2, 3)
- **Phi** (2, 3)
- **And 40+ more...**

## Installation

```bash
# No external conversion engine needed — gguf2mlx is bundled
# Just install the dependencies:
pip install -e .

# Or install dependencies manually:
pip install gguf mlx mlx-lm safetensors transformers rich tqdm psutil requests

# Run the converter
python3 convert.py model.gguf
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GGUF_TO_MLX_DEFAULT_BITS` | 4 | Default quantization bits |
| `GGUF_TO_MLX_MAX_SHARD_SIZE` | 4.5GB | Maximum safetensor shard size |

### Presets

```bash
--preset quality     # 8-bit quantization, quality focus
--preset balanced    # 4-bit quantization, balanced
--preset speed       # 2-bit quantization, speed focus
--preset auto        # Smart detection based on RAM
```

## Examples

```bash
# Convert a Llama model with 4-bit quantization
python3 convert.py Llama-3.2-3B-Q4_K_M.gguf

# Convert Gemma4 to float16 (bypasses quantization issues)
python3 convert.py gemma-4-26B.gguf --no-quantize

# Inspect model before converting
python3 convert.py model.gguf --inspect

# Custom output with 8-bit quantization
python3 convert.py model.gguf ./my-model --bits 8
```

## Troubleshooting

### Gemma4 MoE Models

Gemma4 MoE models have architectural differences that may prevent 4-bit quantization. Workarounds:

```bash
# Option 1: Float16 conversion (recommended)
python3 convert.py gemma4.gguf --no-quantize

# Option 2: Use Ollama (native support)
brew install ollama && ollama run gemma4:27b

# Option 3: Use llama.cpp directly
llama-cli -m gemma4.gguf -p "Hello"
```

### Out of Memory

```bash
# Reduce quantization
python3 convert.py model.gguf --bits 2

# Or use float16 with memory mapping
python3 convert.py model.gguf --no-quantize
```

## Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  model.gguf   │ ──▶ │  convert.py     │ ──▶ │  output/         │
│  (quantized) │     │  + gguf2mlx     │     │  ├ config.json   │
│  Q4_K, Q8... │     │  • detect arch  │     │  ├ tokenizer.json │
└──────────────┘     │  • dequantize   │     │  └ model-*.safetensors
                     │  • remap names  │     └──────────────────┘
                     │  • quantize MLX │
                     └─────────────────┘
```

## Development

```bash
# Run tests
python3 -m pytest test_convert.py -v

# Run type checking
python3 -m mypy convert.py

# Format code
python3 -m black convert.py
```

## License

See [LICENSE](LICENSE) for details.

## Credits

See [CREDITS.md](CREDITS.md) for full attribution.