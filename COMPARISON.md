# GGUF-TO-MLX vs GGUF2MLX: Technical Comparison

## Overview

| Aspect | GGUF-TO-MLX | GGUF2MLX |
|--------|-------------|----------|
| **Type** | User wrapper | Core engine |
| **Lines of code** | ~1,800 | ~1,300 |
| **Functions** | 31 | 18 |
| **Entry point** | `convert.py` | `gguf2mlx` CLI |

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER                                            │
│                         "Convert my model"                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GGUF-TO-MLX (convert.py)                            │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ • Guided CLI with prompts                                             │  │
│  │ • Smart defaults (RAM-based)                                          │  │
│  │ • Hardware detection (M-series chips)                                 │  │
│  │ • Rich progress UI (colors, tables, panels)                            │  │
│  │ • Known issues database                                               │  │
│  │ • Architecture-specific workarounds                                   │  │
│  │ • Pre-flight checks (disk, dependencies)                               │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌───────────────────────────┐       ┌───────────────────────────┐
│          gguf-py          │       │          mlx_lm            │
│    (GGUF metadata read)    │       │   (Quantization pipeline)   │
└───────────────────────────┘       └───────────────────────────┘
                    │                               │
                    └───────────┬───────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GGUF2MLX (core engine)                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ • GGUFReader - low-level file parsing                                 │  │
│  │ • Tensor name mapping (44+ architectures)                             │  │
│  │ • NumPy - tensor dequantization                                       │  │
│  │ • Safetensors - output generation                                     │  │
│  │ • Tokenizer extraction                                                │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT                                             │
│                    MLX-compatible model                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Feature Matrix

| Feature | GGUF-TO-MLX | GGUF2MLX |
|---------|:-----------:|:--------:|
| **User Interface** | | |
| Guided CLI (prompts) | ✅ | ❌ |
| Direct CLI | ✅ | ✅ |
| Rich colors/tables | ✅ | ❌ |
| Progress bars | ✅ | ✅ |
| | | |
| **Smart Defaults** | | |
| RAM-based quantization | ✅ | ❌ |
| Apple Silicon detection | ✅ | ❌ |
| Preset system | ✅ | ❌ |
| Architecture auto-detect | ✅ | ✅ |
| | | |
| **Error Handling** | | |
| Known issues database | ✅ | ❌ |
| Workaround suggestions | ✅ | ❌ |
| Pre-flight checks | ✅ | ❌ |
| Detailed error messages | ✅ | ❌ |
| | | |
| **Core Conversion** | | |
| GGUF parsing | ❌ | ✅ |
| Tensor dequantization | ❌ | ✅ |
| Tensor name remapping | ❌ | ✅ |
| Tokenizer extraction | ❌ | ✅ |
| Safetensor output | ❌ | ✅ |
| | | |
| **Special Support** | | |
| Gemma4 MoE fixes | ✅ | ✅ |
| List/tuple metadata handling | ✅ | ✅ |
| Expert tensor naming | ✅ | ✅ |

## Code Comparison

### GGUF-TO-MLX: `smart_defaults()`

```python
def smart_defaults(
    ram_gb: float,
    file_type: int,
    model_size_gb: float,
    is_known_issue: bool = False,
) -> dict:
    """Select quantization based on available RAM."""
    if ram_gb >= 64:
        bits = 8 if not is_known_issue else 4
    elif ram_gb >= 32:
        bits = 4
    elif ram_gb >= 16:
        bits = 4
    elif ram_gb >= 8:
        bits = 4
    else:
        bits = 2
    return {"bits": bits, "dtype": "float16"}
```

### GGUF2MLX: `detect_architecture()`

```python
def detect_architecture(reader: GGUFReader) -> str:
    """Detect architecture from GGUF metadata."""
    arch = get_metadata_str(reader, "general.architecture")
    if arch:
        return arch
    
    # Try model name fallback
    model_name = get_metadata_str(reader, "general.name", "")
    # ... model name parsing
```

## Gemma4 Handling Comparison

### GGUF-TO-MLX: Post-Conversion Fix

```python
def fix_gemma4_tensor_names(intermediate_dir: Path) -> bool:
    """Fix tensor names AFTER gguf2mlx conversion."""
    # Rename expert tensors for mlx_lm compatibility
    for sf_file in safetensor_files:
        for key in list(weights.keys()):
            if "experts.gate_up_proj.weight" in key:
                # Rename for mlx_lm sanitize()
```

### GGUF2MLX: Inline Mapping

```python
def _map_gemma4_tensor_name(gguf_name: str) -> str:
    """Map GGUF names to MLX names during conversion."""
    if "ffn_gate.weight" in gguf_name:
        return f"{prefix}experts.gate_up_proj"  # No .weight suffix!
    # ... other mappings
```

## Integration Points

| Operation | GGUF-TO-MLX | GGUF2MLX |
|-----------|-------------|----------|
| Parse GGUF metadata | `gguf.GGUFReader` | `gguf.GGUFReader` |
| Run conversion | `subprocess.run("gguf2mlx ...", shell=True)` | Direct function call |
| Quantize | `mlx_lm.convert()` | N/A |
| Test inference | `mlx_lm.generate()` | N/A |

## Summary

**GGUF-TO-MLX** is a **user experience layer** that:
- Makes the conversion process accessible to non-technical users
- Provides smart defaults based on hardware
- Offers architecture-specific troubleshooting
- Shows beautiful progress and results

**GGUF2MLX** is the **core engine** that:
- Performs the actual GGUF → MLX conversion
- Handles low-level tensor manipulation
- Supports 44+ architectures natively
- Provides a simple CLI for power users

Both systems complement each other - GGUF-TO-MLX invokes GGUF2MLX for the heavy lifting while adding user-friendly features on top.
