# GGUF-to-MLX Quick Reference

## Installation

### One-liner (Recommended)
```bash
curl -sSL https://raw.githubusercontent.com/acampkin95/gguf-to-mlx/main/install.py | python3
```

### Manual Install
```bash
git clone https://github.com/acampkin95/gguf-to-mlx.git
cd gguf-to-mlx
python3 install.py
```

## Usage

### Basic Conversion
```bash
cpmm model.gguf                        # Guided mode
cpmm model.gguf ./output/              # Custom output
cpmm model.gguf --bits 4              # 4-bit quantization
```

### Advanced Options
```bash
cpmm model.gguf --preset quality       # 8-bit, high quality
cpmm model.gguf --preset balanced     # 4-bit, balanced
cpmm model.gguf --preset speed        # 2-bit, fast
cpmm model.gguf --no-quantize         # Float16 only
cpmm model.gguf --inspect             # Show metadata
```

### Gemma4 Models
```bash
cpmm gemma4.gguf --no-quantize        # Float16 (MoE workaround)
```

## Alias Commands

| Alias | Description |
|-------|-------------|
| `cpmm` | Main converter (default) |
| `convertgguf` | Alternative name |

To change alias:
```bash
# Edit ~/.zshrc and change the alias line
alias convertgguf="python3 ~/Projects/gguf-to-mlx/convert.py"
```

## Ollama Alternative

For Gemma4 with native 4-bit support:
```bash
brew install ollama
ollama run gemma4:31b
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Use `--bits 2` or `--no-quantize` |
| Gemma4 fails | Use `--no-quantize` or Ollama |
| Unknown architecture | Report issue on GitHub |

## Files

| File | Purpose |
|------|---------|
| `convert.py` | Main CLI |
| `install.py` | Installer |
| `Makefile` | Build automation |
| `README.md` | Documentation |
| `CREDITS.md` | Attribution |
| `COMPARISON.md` | Technical comparison |

## Repository

- **Main**: https://github.com/acampkin95/gguf-to-mlx
- **Fork**: https://github.com/acampkin95/gguf2mlx
- **Upstream**: https://github.com/barrontang/gguf2mlx

## Developer Commands

```bash
cd ~/Projects/gguf-to-mlx

make test      # Run tests
make verify    # Check installation
make clean     # Clean artifacts
make update    # Update from upstream
```