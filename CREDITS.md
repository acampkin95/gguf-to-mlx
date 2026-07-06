# Credits & Attribution

This project builds upon the work of many contributors and open-source projects.

---

## Core Conversion Engine

### gguf2mlx (Vendored Internally)

**Repository:** https://github.com/barrontang/gguf2mlx  
**License:** MIT  
**Author:** Barron Tang

The `gguf2mlx` library (v2.0.2) is vendored into the `gguf2mlx/` package directory
for self-contained maintenance. It handles:

- GGUF metadata parsing and architecture detection
- Tensor dequantization (Q2_K through F16)
- Weight transposition (GGUF ↔ HuggingFace layout)
- Tensor name remapping for 44+ architectures
- Smart sharding for large models
- Tokenizer extraction and conversion

### Key Contributors to gguf2mlx

| Contributor | Contribution |
|-------------|--------------|
| [barrontang](https://github.com/barrontang) | Original author, core architecture |
| [acampkin95](https://github.com/acampkin95) | Gemma4 MoE support, bug fixes |

---

## This Project

### Authors

| Name | GitHub | Contribution |
|------|--------|--------------|
| Alex | [@alex](https://github.com) | CLI enhancements, Apple Silicon optimization, Gemma4 support, integration |

### Enhancements Added

This project (`gguf-to-mlx`) adds the following enhancements to `gguf2mlx`:

1. **User-Friendly CLI** - Interactive guided mode with prompts
2. **Smart Defaults** - RAM-based quantization recommendations
3. **Hardware Detection** - Apple Silicon M-series chip detection
4. **Preset System** - Quality, balanced, speed configurations
5. **Progress UI** - Rich console output with progress bars
6. **Safety Checks** - Disk space validation and error handling
7. **Gemma4 Support** - MoE architecture fixes and workarounds
8. **Error Recovery** - Clear error messages with troubleshooting hints

---

## Dependencies

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| [gguf](https://pypi.org/project/gguf/) | latest | GGUF file parsing | MIT |
| [mlx](https://mlc.ai/mlx/) | latest | Apple MLX framework | Apache 2.0 |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | latest | MLX language models | Apache 2.0 |
| [safetensors](https://pypi.org/project/safetensors/) | latest | Safe tensor I/O | Apache 2.0 |
| [transformers](https://huggingface.co/docs/transformers) | latest | Tokenizer handling | Apache 2.0 |
| [Rich](https://rich.readthedocs.io/) | latest | Terminal UI | MIT |
| [tqdm](https://tqdm.github.io/) | latest | Progress bars | MIT |
| [psutil](https://psutil.readthedocs.io/) | latest | System info | BSD |

---

## Acknowledge

### Apple

Apple's MLX framework and the `mlx-lm` library enable efficient language model inference on Apple Silicon.

### Hugging Face

Hugging Face's transformers library and safetensors format are the industry standard for model distribution.

### llama.cpp

The llama.cpp project defined the GGUF format and provides the quantization algorithms used by `gguf2mlx`.

---

## Contributing

Contributions are welcome! Please see the main [README.md](README.md) for development setup.

### Adding New Architectures

1. Edit the vendored source in `gguf2mlx/core.py`
2. Add architecture mapping to the `ARCH_MAP` and/or tensor name remapping functions
3. Test with a representative GGUF file
4. Run the test suite: `python3 -m pytest test_convert.py -v`

### Bug Reports

- **Conversion engine issues:** Fix directly in `gguf2mlx/core.py` — consider upstream PR
- **CLI issues:** Report to this project's issue tracker

---

## Changelog

### 2025-05-26

- Added Gemma4 MoE support with tensor remapping fixes
- Integrated user's gguf2mlx fork with bug fixes
- Enhanced error handling with architecture-specific workarounds
- Added Ollama as alternative for Gemma4 models

### 2025-05-25

- Initial CLI wrapper created
- Smart quantization defaults based on available RAM
- Apple Silicon hardware detection
- Preset system (quality, balanced, speed, auto)

---

*Last updated: 2025-05-26*