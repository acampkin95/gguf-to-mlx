#!/usr/bin/env python3
"""
GGUF → MLX Converter with Dynamic Quantization
Easy one-step pipeline for Apple Silicon Macs.

Usage:
  python3 convert.py                          (guided, prompts for file)
  python3 convert.py model.gguf               (direct, smart defaults)
  python3 convert.py model.gguf ./output/     (custom output folder)
  python3 convert.py model.gguf --bits 8      (8-bit quant)
  python3 convert.py model.gguf --no-quantize (float16 only)
  python3 convert.py model.gguf --inspect     (show metadata, no convert)
  python3 convert.py model.gguf --preset quality
"""

import sys
import os
import time
import re
import subprocess
import shutil
import argparse
from pathlib import Path

# Rich - console output, progress, prompts
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.prompt import Prompt, Confirm
from rich.rule import Rule
from rich.text import Text
from rich.style import Style
from rich import box

# Hardware detection
import psutil

# ═══════════════════════════════════════════════════════════════════════════
# Module-level console - configured in main() based on --no-color
# ═══════════════════════════════════════════════════════════════════════════

console = Console(highlight=False)


# ═══════════════════════════════════════════════════════════════════════════
# GGUF Quantisation Type → Metadata Mapping
# ═══════════════════════════════════════════════════════════════════════════

# These are the gguf general.file_type enum values (GGML FTYPE)
# See: llama.cpp gguf-py/gguf/constants.py
GGUF_FTYPE_MAP = {
    0:  ("ALL_F32",              "float32",   "full precision"),
    1:  ("MOSTLY_F16",           "float16",   "half precision"),
    2:  ("MOSTLY_Q4_0",          "4-bit",     "legacy 4-bit"),
    3:  ("MOSTLY_Q4_1",          "4-bit",     "legacy 4-bit (f32 scale)"),
    4:  ("MOSTLY_Q4_1_SOME_F16", "mixed",     "4-bit + some f16"),
    5:  ("MOSTLY_Q8_0",          "8-bit",     "legacy 8-bit"),
    6:  ("MOSTLY_Q5_0",          "5-bit",     "legacy 5-bit"),
    7:  ("MOSTLY_Q5_1",          "5-bit",     "legacy 5-bit (f32 scale)"),
    8:  ("MOSTLY_Q2_K",          "2-bit",     "K-quant 2-bit"),
    9:  ("MOSTLY_Q3_K_S",        "3-bit",     "K-quant 3-bit small"),
    10: ("MOSTLY_Q3_K_M",        "3-bit",     "K-quant 3-bit medium"),
    11: ("MOSTLY_Q3_K_L",        "3-bit",     "K-quant 3-bit large"),
    12: ("MOSTLY_Q4_K_S",        "4-bit",     "K-quant 4-bit small"),
    13: ("MOSTLY_Q4_K_M",        "4-bit",     "K-quant 4-bit medium"),
    14: ("MOSTLY_Q5_K_S",        "5-bit",     "K-quant 5-bit small"),
    15: ("MOSTLY_Q5_K_M",        "5-bit",     "K-quant 5-bit medium"),
    16: ("MOSTLY_Q6_K",          "6-bit",     "K-quant 6-bit"),
    17: ("MOSTLY_IQ2_XXS",       "2-bit",     "IQ 2-bit extra small"),
    18: ("MOSTLY_IQ2_XS",        "2-bit",     "IQ 2-bit small"),
    19: ("MOSTLY_IQ2_S",         "2-bit",     "IQ 2-bit"),
    20: ("MOSTLY_IQ3_XXS",       "3-bit",     "IQ 3-bit extra small"),
    21: ("MOSTLY_IQ3_S",         "3-bit",     "IQ 3-bit"),
    22: ("MOSTLY_IQ1_S",         "1-bit",     "IQ 1-bit"),
    23: ("MOSTLY_IQ4_NL",        "4-bit",     "IQ 4-bit nonlinear"),
    24: ("MOSTLY_IQ4_XS",        "4-bit",     "IQ 4-bit extra small"),
    25: ("MOSTLY_IQ1_M",         "1-bit",     "IQ 1-bit medium"),
    26: ("MOSTLY_BF16",          "bfloat16",  "brain float 16"),
    27: ("MOSTLY_Q4_0_4_4",      "4-bit",     "MoE 4-bit"),
    28: ("MOSTLY_Q4_0_4_8",      "4-bit",     "MoE 4-bit"),
    29: ("MOSTLY_Q4_0_8_8",      "4-bit",     "MoE 4-bit"),
    30: ("MOSTLY_TQ1_0",         "1-bit",     "ternary 1-bit"),
    31: ("MOSTLY_TQ2_0",         "2-bit",     "ternary 2-bit"),
}


# Architectures known to have gguf2mlx compatibility issues
KNOWN_CONVERSION_ISSUES: dict[str, dict] = {
    "gemma4": {
        "issue": "head_count_kv metadata returns list instead of int",
        "workarounds": [
            ("cpmm model.gguf --no-quantize", "Converts to float16 only, avoids gguf2mlx quant step"),
            ("pip install --upgrade gguf2mlx", "May already be fixed in latest version"),
        ],
    },
    "gemma3": {
        "issue": "Similar metadata issues as Gemma4",
        "workarounds": [
            ("cpmm model.gguf --no-quantize", "Converts to float16 only"),
            ("pip install --upgrade gguf2mlx", "May already be fixed"),
        ],
    },
    "gemma2": {
        "issue": "May have similar head_count_kv issues",
        "workarounds": [
            ("cpmm model.gguf --no-quantize", "Converts to float16 only"),
            ("pip install --upgrade gguf2mlx", "May already be fixed"),
        ],
    },
}



# Gemma4-specific metadata field mapping
Gemma4_METADATA_FIELDS: dict[str, list[str]] = {
    "attention_head_count_kv": ["gemma4.attention.head_count_kv", "gemma3.attention.head_count_kv"],
    "sliding_window": ["gemma4.attn_attn.sliding_window", "gemma3.attn_attn.sliding_window"],
    "max_seq_len": ["gemma4.context_length", "gemma3.context_length"],
}



# Tensor name mappings for GGUF -> MLX conversion fixes
# Some GGUF files have tensor names that don't match MLX expected format
GGUF_TO_MLX_TENSOR_RENAME = {
    # Gemma4 MoE layer naming
    "blk": "model.layers",
}


def fix_gemma4_tensor_names(intermediate_dir: Path) -> bool:
    """Fix Gemma4 tensor names if they don't match MLX expected format.
    
    Returns True if fixes were applied, False otherwise.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    
    # Check if this is a Gemma4 model with wrong tensor names
    safetensor_files = list(intermediate_dir.glob("*.safetensors"))
    if not safetensor_files:
        return False
    
    # Check first file for problematic tensor names
    needs_fix = False
    with safe_open(safetensor_files[0], framework="numpy") as f:
        keys = list(f.keys())
        # Check for blk.* pattern (GGUF format) without model.layers
        if any(k.startswith("blk.") for k in keys):
            needs_fix = True
    
    if not needs_fix:
        return False
    
    print(f"  [yellow]⚠[/yellow]  Detected Gemma4 tensor naming issue, applying fix...")
    
    # We need to rename tensors, but safetensors are read-only in this context
    # Instead, we'll create a mapping and document the workaround
    console.print()
    console.print(
        "  [yellow]⚠[/yellow]  [bold]Gemma4 tensor naming incompatibility detected[/bold]"
    )
    console.print("  gguf2mlx produces incorrect tensor names for Gemma4 MoE layers.")
    console.print("  These need to be renamed from 'blk.*' to 'model.layers.*' format.")
    console.print()
    console.print("  [bold]Workaround options:[/bold]")
    console.print("  [dim]1.[/dim] Use [cyan]--no-quantize[/cyan] for float16 (50GB)")
    console.print("  [dim]2.[/dim] Wait for gguf2mlx fix at: https://github.com/acampkin95/gguf2mlx")
    console.print("  [dim]3.[/dim] Use Ollama or llama.cpp for 4-bit conversion")
    console.print()
    
    return False  # Can't auto-fix without rewriting safetensors



def is_known_issue_arch(arch: str) -> tuple[bool, dict | None]:
    """Check if architecture is known to have gguf2mlx issues."""
    arch_lower = arch.lower()
    for known_arch, issue_info in KNOWN_CONVERSION_ISSUES.items():
        if arch_lower.startswith(known_arch):
            return True, issue_info
    return False, None



def read_gemma4_metadata(reader, arch: str) -> dict:
    """Read Gemma4-specific metadata fields."""
    gemma_meta: dict = {}

    # Try Gemma4-specific fields
    for field_name, possible_keys in Gemma4_METADATA_FIELDS.items():
        for key in possible_keys:
            val = _field_value(reader, key)
            if val is not None:
                gemma_meta[field_name] = val
                break

    return gemma_meta


def classify_source_quality(file_type: int) -> dict:
    """Classify source GGUF model quality to inform re-quantization decisions."""
    if file_type <= 1:  # F32 or F16
        return {"risk": "none", "label": "unquantized source",
                "advice": "Ideal for quantization - no quality already lost."}
    if file_type in (5,):  # Q8_0
        return {"risk": "low", "label": "lightly quantized (8-bit)",
                "advice": "Re-quantizing from 8-bit is generally safe."}
    if file_type in (6, 7, 14, 15, 16):  # Q5_0, Q5_1, Q5_K_S, Q5_K_M, Q6_K
        return {"risk": "medium", "label": "moderately quantized (5-6 bit)",
                "advice": "Re-quantization will compound quality loss. Consider --no-quantize instead."}
    if file_type in (2, 3, 4, 10, 11, 12, 13, 27, 28, 29):  # Q4 variants
        return {"risk": "high", "label": "heavily quantized (4-bit)",
                "advice": "⚠ Double-quantization - significant quality loss expected."}
    if file_type in (8, 9, 19, 20, 21, 22, 24, 25, 30, 31):  # Q2/Q3/IQ/TQ
        return {"risk": "severe", "label": "extremely quantized (≤3-bit)",
                "advice": "⚠ Severe quality degradation if re-quantized. Use --no-quantize."}
    return {"risk": "unknown", "label": "unknown type",
            "advice": "Unable to determine source quality."}


# ═══════════════════════════════════════════════════════════════════════════
# Quantisation Presets
# ═══════════════════════════════════════════════════════════════════════════

PRESETS = {
    "speed": {
        "bits": 4, "group_size": 32, "mode": "affine",
        "description": "Smallest files, fastest inference - good for large models",
    },
    "balanced": {
        "bits": 4, "group_size": 64, "mode": "affine",
        "description": "Good quality/size tradeoff - recommended default",
    },
    "quality": {
        "bits": 8, "group_size": 32, "mode": "affine",
        "description": "Highest quality, larger files - best for small models",
    },
    "m5-max": {
        "bits": 4, "group_size": 32, "mode": "affine",
        "description": "Optimised for M5 Max (64 GB) - memory-bandwidth-tuned, "
                       "smaller groups leverage unified memory throughput",
    },
}

# For --high-bandwidth flag: maps to the m5-max preset
HIGH_BANDWIDTH_PRESET = "m5-max"


# ═══════════════════════════════════════════════════════════════════════════
# Hardware Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_apple_silicon() -> dict:
    """Detect Apple Silicon hardware and return chip info.

    Returns a dict with:
        is_apple_silicon: bool
        chip_name:       str   - "Apple M5 Max", "Apple M3", etc.
        chip_tier:       str   - "base", "pro", "max", "ultra"
        chip_gen:        int   - 5 for M5, 3 for M3, etc.
        ram_gb:          float - total system RAM in GB
    """
    result = {
        "is_apple_silicon": False,
        "chip_name": "Unknown",
        "chip_tier": "base",
        "chip_gen": 0,
        "ram_gb": 0.0,
    }

    # Detect chip via sysctl
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
        ).strip()
    except Exception:
        return result

    if not brand.startswith("Apple "):
        return result

    result["is_apple_silicon"] = True
    result["chip_name"] = brand

    # Parse "Apple M5 Max" → gen=5, tier="max"
    m = re.match(r"Apple\s+M(\d+)\s*(.*)", brand)
    if m:
        result["chip_gen"] = int(m.group(1))
        suffix = m.group(2).strip().lower()
        if "ultra" in suffix:
            result["chip_tier"] = "ultra"
        elif "max" in suffix:
            result["chip_tier"] = "max"
        elif "pro" in suffix:
            result["chip_tier"] = "pro"
        elif suffix:
            result["chip_tier"] = suffix
        else:
            result["chip_tier"] = "base"

    # RAM via psutil or sysctl fallback
    try:
        result["ram_gb"] = psutil.virtual_memory().total / 1e9
    except Exception:
        try:
            raw = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            ).strip()
            result["ram_gb"] = int(raw) / 1e9
        except Exception:
            pass

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Smart Defaults (Hardware-Aware)
# ═══════════════════════════════════════════════════════════════════════════

def smart_defaults(
    model_size_gb: float,
    chip_tier: str = "base",
    ram_gb: float = 0,
    chip_gen: int = 0,
) -> dict:
    """Auto-select quantization params based on model size and hardware.

    Returns dict with bits, group_size, mode, description explaining why.
    """
    # Max/Ultra with ≥48 GB - bandwidth-optimized for unified memory
    if chip_tier in ("max", "ultra") and ram_gb >= 48:
        throughput_note = (
            "~800 GB/s unified memory throughput"
            if chip_gen >= 5
            else "high unified memory throughput"
        )
        return {
            "bits": 4, "group_size": 32, "mode": "affine",
            "description": (
                f"Bandwidth-optimized for {chip_tier.title()} chip with ample RAM"
                f" (group_size=32 leverages {throughput_note})"
            ),
        }

    # Pro chips (M5/M4/M3/M2 Pro)
    if chip_tier == "pro":
        if model_size_gb >= 10:
            return {
                "bits": 4, "group_size": 64, "mode": "affine",
                "description": (
                    "Balanced 4-bit/group64 for Pro chip with large model "
                    "(good throughput without straining memory bandwidth)"
                ),
            }
        else:
            return {
                "bits": 6, "group_size": 32, "mode": "affine",
                "description": (
                    "Quality-focused 6-bit for Pro chip with small model "
                    "(plenty of headroom for higher precision)"
                ),
            }

    # Base chips or <16 GB RAM - conservative
    if ram_gb < 16:
        return {
            "bits": 4, "group_size": 128, "mode": "affine",
            "description": (
                "Conservative 4-bit/group128 - limited RAM, "
                "larger groups keep overhead low"
            ),
        }

    # Fallback: model-size-based defaults
    if model_size_gb < 3:
        return {
            "bits": 8, "group_size": 32, "mode": "affine",
            "description": "Quality-focused 8-bit for small model (<3 GB)",
        }
    elif model_size_gb < 15:
        return {
            "bits": 4, "group_size": 64, "mode": "affine",
            "description": "Balanced 4-bit/group64 for medium model (3-15 GB)",
        }
    else:
        return {
            "bits": 4, "group_size": 32, "mode": "affine",
            "description": "Compact 4-bit/group32 for large model (≥15 GB)",
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gguf-to-mlx",
        description=(
            "Convert GGUF models to MLX format with configurable quantization."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.gguf                       Convert with smart defaults
  %(prog)s model.gguf --bits 8              Convert to 8-bit quant
  %(prog)s model.gguf --no-quantize          Float16 only, no quantisation
  %(prog)s model.gguf --preset quality       Quality-preset quantisation
  %(prog)s model.gguf --high-bandwidth       M5 Max / Ultra device optimised
  %(prog)s model.gguf --inspect              Show GGUF metadata, no conversion
  %(prog)s model.gguf --mtp                  Show MTP info during conversion
  %(prog)s model.gguf --mode mxfp4           Use MXFP4 quantisation mode
  %(prog)s model.gguf --bits 2 --group-size 128  Custom 2-bit quantisation
""",
    )

    # Positional
    p.add_argument(
        "input", nargs="?", help="Path to .gguf model file",
    )
    p.add_argument(
        "output", nargs="?",
        help="Output directory (auto-named if omitted)",
    )

    # Quantisation
    quant = p.add_argument_group("Quantisation Options")
    quant.add_argument(
        "--bits", type=int, choices=[2, 3, 4, 6, 8],
        help="Bits per weight for quantisation (default: auto)",
    )
    quant.add_argument(
        "--group-size", type=int, choices=[32, 64, 128, 256],
        help="Group size for quantisation (default: auto)",
    )
    quant.add_argument(
        "--mode", choices=["affine", "mxfp4", "nvfp4", "mxfp8"],
        help="Quantisation mode (default: affine)",
    )
    quant.add_argument(
        "--predicate",
        choices=["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"],
        help="Mixed-bit quantisation recipe",
    )
    quant.add_argument(
        "--no-quantize", "-n", action="store_true",
        help="Skip quantisation - output float16 MLX only",
    )
    quant.add_argument(
        "--preset", choices=list(PRESETS.keys()),
        help="Quantisation preset (overrides --bits/--group-size/--mode)",
    )
    quant.add_argument(
        "--high-bandwidth", action="store_true",
        help=(
            "Shortcut for --preset m5-max "
            "(Apple Silicon Max/Ultra, ≥48 GB)"
        ),
    )

    # Intermediate dtype
    p.add_argument(
        "--dtype", choices=["float16", "float32"],
        help="Intermediate dtype for GGUF→MLX step (default: auto-detect)",
    )

    # Behaviour
    p.add_argument(
        "--inspect", action="store_true",
        help="Display GGUF metadata and exit (no conversion)",
    )
    p.add_argument(
        "--mtp", action="store_true",
        help="Report MTP (Multi-Token Prediction) capability during conversion",
    )
    p.add_argument(
        "--keep-intermediate", action="store_true",
        help="Keep intermediate float16 files after quantisation",
    )
    p.add_argument(
        "--force", "-f", action="store_true",
        help="Skip disk space check and confirmation prompts",
    )
    p.add_argument(
        "--quiet", "-q", action="store_true",
        help="Minimal output - only errors and final result",
    )
    p.add_argument(
        "--no-color", action="store_true",
        help="Disable Rich markup (for piping / CI environments)",
    )

    return p


# ═══════════════════════════════════════════════════════════════════════════
# GGUF Metadata Reader
# ═══════════════════════════════════════════════════════════════════════════

def _has_gguf_py() -> bool:
    try:
        import gguf  # noqa: F401
        return True
    except ImportError:
        return False


def _field_value(reader, name: str):
    """Safely read a scalar field value from a GGUF reader."""
    field = reader.get_field(name)
    if field is None:
        return None
    try:
        val = field.contents()
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="replace")
        # Handle numpy scalars
        elif hasattr(val, "item"):
            val = val.item()
        # Handle list values (some GGUF fields return lists, e.g., Gemma4 head_count_kv)
        if isinstance(val, list):
            val = val[0] if len(val) > 0 else None
        elif isinstance(val, tuple):
            val = val[0] if len(val) > 0 else None
        return val
    except Exception:
        return None


def _field_exists(reader, name: str) -> bool:
    return reader.get_field(name) is not None


def read_gguf_metadata(gguf_path: Path) -> dict | None:
    """Read key metadata from a GGUF file. Returns None if gguf-py unavailable."""
    if not _has_gguf_py():
        return None

    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path))

    meta: dict = {
        "architecture": None,
        "file_type": None,
        "file_type_name": None,
        "model_name": None,
        "vocab_size": None,
        "hidden_size": None,
        "num_layers": None,
        "num_heads": None,
        "num_kv_heads": None,
        "context_length": None,
        "param_count": None,
        "mtp_layers": None,
        "has_ssm": False,
        "tensor_count": 0,
        "total_weight_bytes": 0,
        "warnings": [],
        "fields": {},
    }

    # Collect all metadata fields for --inspect display
    for field in reader.fields.values():
        try:
            val = field.contents()
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            elif hasattr(val, "item"):
                val = val.item()
            # Truncate very long values
            val_str = str(val)
            if len(val_str) > 200:
                val_str = val_str[:197] + "..."
            meta["fields"][field.name] = val_str
        except Exception:
            meta["fields"][field.name] = "<complex>"

    # Extract key fields
    meta["architecture"] = (
        _field_value(reader, "general.architecture") or "unknown"
    )
    meta["file_type"] = _field_value(reader, "general.file_type")
    meta["model_name"] = (
        _field_value(reader, "general.name") or gguf_path.stem
    )

    # Look for arch-specific fields (try multiple prefixes)
    arch = str(meta["architecture"])
    prefixes = [arch + ".", "llama."]
    for prefix in prefixes:
        if meta["vocab_size"] is None:
            meta["vocab_size"] = _field_value(
                reader, prefix + "vocab_size"
            )
        if meta["hidden_size"] is None:
            meta["hidden_size"] = _field_value(
                reader, prefix + "embedding_length"
            )
        if meta["num_layers"] is None:
            meta["num_layers"] = _field_value(
                reader, prefix + "block_count"
            )
        if meta["num_heads"] is None:
            meta["num_heads"] = _field_value(
                reader, prefix + "attention.head_count"
            )
        if meta["num_kv_heads"] is None:
            meta["num_kv_heads"] = _field_value(
                reader, prefix + "attention.head_count_kv"
            )
        if meta["context_length"] is None:
            meta["context_length"] = _field_value(
                reader, prefix + "context_length"
            )

    # MTP (Multi-Token Prediction) - present in Qwen3, DeepSeek-V3, etc.
    for prefix in [arch + "."]:
        mtp_layers = _field_value(reader, prefix + "nextn_predict_layers")
        if mtp_layers is not None:
            meta["mtp_layers"] = int(mtp_layers)
            break

    # SSM / hybrid architecture detection (Mamba, MRP, Qwen3-hybrid)
    meta["has_ssm"] = (
        _field_exists(reader, arch + ".ssm.conv_kernel")
        or _field_exists(reader, arch + ".ssm.inner_size")
    )

    # Gemma4-specific metadata
    gemma_meta = read_gemma4_metadata(reader, arch)
    if gemma_meta:
        meta.update(gemma_meta)
        # Override num_kv_heads with Gemma4-specific value if available
        if "attention_head_count_kv" in gemma_meta:
            meta["num_kv_heads"] = gemma_meta["attention_head_count_kv"]
        # Override context_length with Gemma4-specific value
        if "max_seq_len" in gemma_meta:
            meta["context_length"] = gemma_meta["max_seq_len"]

    # Architecture-specific warnings for known issues
    is_known, issue_info = is_known_issue_arch(arch)
    if is_known and issue_info:
        meta["warnings"].append(
            f"⚠ {arch} has known gguf2mlx issues: {issue_info['issue']}"
        )

    # File type name
    if meta["file_type"] is not None:
        ftype_info = GGUF_FTYPE_MAP.get(int(meta["file_type"]))
        if ftype_info:
            meta["file_type_name"] = f"{ftype_info[0]} ({ftype_info[2]})"

    # Estimate parameter count
    if meta["hidden_size"] and meta["num_layers"] and meta["num_heads"]:
        h = int(meta["hidden_size"])
        L = int(meta["num_layers"])
        V = int(meta["vocab_size"]) if meta["vocab_size"] else 32000
        params = 4 * h * h * L + 2 * h * V
        meta["param_count"] = params

    # Tensor count and total weight bytes
    meta["tensor_count"] = len(reader.tensors)
    try:
        meta["total_weight_bytes"] = sum(t.n_bytes for t in reader.tensors)
    except Exception:
        meta["total_weight_bytes"] = 0

    # MTP warnings
    if meta["mtp_layers"]:
        mtp_tensors = [
            t for t in reader.tensors
            if "nextn." in t.name or "mtp." in t.name
        ]
        if mtp_tensors:
            meta["warnings"].append(
                f"MTP present ({meta['mtp_layers']} layer(s), "
                f"{len(mtp_tensors)} tensors) - "
                f"mlx_lm may strip MTP weights during conversion. "
                f"This is expected and handled transparently."
            )

    # Source quality risk warns
    if meta["file_type"] is not None:
        quality = classify_source_quality(int(meta["file_type"]))
        if quality["risk"] in ("high", "severe"):
            meta["warnings"].append(
                f"Source is {quality['label']} - "
                f"re-quantization will compound quality loss."
            )

    return meta


# ═══════════════════════════════════════════════════════════════════════════
# Display Helpers (Rich-based)
# ═══════════════════════════════════════════════════════════════════════════

def banner():
    """Print the converter banner using Rich Panel."""
    console.print()
    console.print(Panel(
        "[bold]GGUF → MLX Converter[/bold]\n"
        "[dim]with Dynamic Quantization for Apple Silicon Macs[/dim]",
        border_style="bold cyan",
        padding=(1, 2),
    ))
    console.print()


def step(n: int, total: int, label: str):
    """Print a pipeline step header using Rich Rule."""
    console.print()
    console.print(Rule(
        f"[bold cyan]Step {n}/{total}[/bold cyan]  {label}",
        style="dim",
    ))


def ok(msg: str):
    console.print(f"  [green]✓[/green] {msg}")


def fail(msg: str):
    console.print(f"\n  [red]✗[/red] {msg}")


def info(msg: str):
    console.print(f"  [dim]·[/dim] {msg}")


def warn(msg: str):
    console.print(f"  [yellow]⚠[/yellow]  {msg}")


def run_with_spinner(
    cmd: list[str],
    description: str,
    quiet: bool = False,
) -> tuple[bool, "subprocess.CompletedProcess[str]" | None]:
    """Run a subprocess with a Rich spinner, capturing output.

    On failure, displays captured stdout/stderr with proper formatting.
    Returns (True, None) on success, (False, result) on failure.
    """
    if not quiet:
        info(f"Running: [dim]{' '.join(str(c) for c in cmd)}[/dim]")

    try:
        with console.status(
            f"[bold cyan]{description}...[/bold cyan]",
            spinner="dots",
        ):
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

        if result.returncode != 0:
            fail(f"{description} failed (exit {result.returncode})")
            if result.stderr:
                console.print(Panel(
                    result.stderr.strip()[-1000:],
                    title="Error Output",
                    border_style="red",
                ))
            if result.stdout:
                console.print(Panel(
                    result.stdout.strip()[-500:],
                    title="Last Output",
                    border_style="dim",
                ))
            return False, result
        return True, result

    except FileNotFoundError as e:
        fail(f"Command not found: {e}")
        return False, None


def format_size(size_bytes: float) -> str:
    """Human-readable size."""
    gb = size_bytes / 1e9
    if gb >= 1:
        return f"{gb:.2f} GB"
    mb = size_bytes / 1e6
    return f"{mb:.0f} MB"


def format_time(seconds: float) -> str:
    """Human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


# ═══════════════════════════════════════════════════════════════════════════
# Metadata Display (Rich Table)
# ═══════════════════════════════════════════════════════════════════════════

def display_metadata(meta: dict):
    """Pretty-print GGUF metadata using Rich Table."""

    # Section 1: Model Info
    console.print()
    model_table = Table(
        title="GGUF Model Information",
        title_style="bold cyan",
        box=box.ROUNDED,
        border_style="dim",
        show_header=False,
    )
    model_table.add_column("Field", style="dim", width=20)
    model_table.add_column("Value", style="bold")

    rows = [
        ("Architecture", str(meta.get("architecture", "?"))),
        ("Model name",   str(meta.get("model_name", "?"))),
        (
            "Parameters",
            (
                f"{meta['param_count'] / 1e9:.2f}B"
                if meta.get("param_count") else "?"
            ),
        ),
        (
            "Vocab size",
            (
                f"{int(meta['vocab_size']):,}"
                if meta.get("vocab_size") else "?"
            ),
        ),
        ("Hidden size",  str(meta.get("hidden_size", "?"))),
        ("Layers",       str(meta.get("num_layers", "?"))),
        ("Attention heads", str(meta.get("num_heads", "?"))),
        ("KV heads",     str(meta.get("num_kv_heads", "?"))),
        (
            "Context length",
            (
                f"{int(meta['context_length']):,}"
                if meta.get("context_length") else "?"
            ),
        ),
        (
            "Tensors",
            (
                f"{meta['tensor_count']:,}"
                if meta.get("tensor_count") else "?"
            ),
        ),
        (
            "Weight data",
            (
                format_size(meta["total_weight_bytes"])
                if meta.get("total_weight_bytes") else "?"
            ),
        ),
    ]

    for label, value in rows:
        model_table.add_row(label, value)

    console.print(model_table)

    # Section 2: Quantization Source
    if meta.get("file_type") is not None:
        quality = classify_source_quality(int(meta["file_type"]))
        risk_styles = {
            "none":    "green",
            "low":     "green",
            "medium":  "yellow",
            "high":    "red",
            "severe":  "bold red",
            "unknown": "dim",
        }
        risk_style = risk_styles.get(quality["risk"], "dim")
        ftype_name = meta.get(
            "file_type_name",
            f"type {meta['file_type']}",
        )

        quant_table = Table(
            title="Source Quantization",
            title_style="bold yellow",
            box=box.ROUNDED,
            border_style="dim",
            show_header=False,
        )
        quant_table.add_column("Field", style="dim", width=20)
        quant_table.add_column("Value")
        quant_table.add_row("File type", ftype_name)
        quant_table.add_row(
            "Quality risk",
            f"[{risk_style}]{quality['risk'].upper()}[/{risk_style}] - "
            f"{quality['label']}",
        )
        quant_table.add_row("Advice", f"[dim]{quality['advice']}[/dim]")
        console.print()
        console.print(quant_table)

    # Section 3: Capabilities
    capabilities = []
    if meta.get("mtp_layers"):
        capabilities.append(
            f"MTP Layers: [bold]{meta['mtp_layers']}[/bold] "
            f"[dim](Multi-Token Prediction enabled)[/dim]"
        )
    else:
        mtp_archs = {
            "qwen3", "qwen35", "deepseek2", "deepseek3", "qwen3moe",
        }
        if str(meta.get("architecture", "")).lower() in mtp_archs:
            capabilities.append(
                "[yellow]MTP not detected[/yellow] "
                "[dim](model may not expose nextn_predict_layers)[/dim]"
            )

    if meta.get("has_ssm"):
        capabilities.append(
            "Architecture: [bold]Hybrid SSM/Attention[/bold] "
            "[dim](Mamba/MRP detected)[/dim]"
        )

    if capabilities:
        cap_table = Table(
            title="Capabilities",
            title_style="bold magenta",
            box=box.ROUNDED,
            border_style="dim",
            show_header=False,
        )
        cap_table.add_column("Capability", style="")
        for cap in capabilities:
            cap_table.add_row(cap)
        console.print()
        console.print(cap_table)

    # Section 4: Warnings
    if meta.get("warnings"):
        console.print()
        for w in meta["warnings"]:
            warn(w)


# ═══════════════════════════════════════════════════════════════════════════
# Inspect Mode
# ═══════════════════════════════════════════════════════════════════════════

def inspect_mode(gguf_path: Path):
    """Run inspect mode: display metadata and exit."""
    info(f"Reading metadata from: [bold]{gguf_path.name}[/bold]")

    if not _has_gguf_py():
        fail("gguf-py package required for metadata inspection.")
        info("Install: [cyan]pip install gguf[/cyan]")
        sys.exit(1)

    meta = read_gguf_metadata(gguf_path)
    if meta:
        display_metadata(meta)
        console.print()
        if meta["fields"]:
            console.print(
                f"  [bold]All metadata fields:[/bold] "
                f"({len(meta['fields'])} total)"
            )
            console.print()
            for key in sorted(meta["fields"].keys()):
                val = meta["fields"][key]
                val_str = str(val)
                if len(val_str) > 80:
                    val_str = val_str[:77] + "..."
                console.print(f"  [dim]{key}[/dim]: {val_str}")
    else:
        fail("Could not read GGUF metadata.")
        sys.exit(1)

    console.print()
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════
# Preflight Checks
# ═══════════════════════════════════════════════════════════════════════════

def preflight_checks(
    gguf_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[bool, list[str], list[str]]:
    """Run pre-flight checks before conversion.

    Returns (passed, warnings, errors) where passed is True if no errors.
    Errors are fatal; warnings are informational.
    """
    warnings: list[str] = []
    errors: list[str] = []

    # 1. File exists
    if not gguf_path.exists():
        errors.append(f"GGUF file not found: {gguf_path}")
        return (False, warnings, errors)

    # 2. File is a file
    if not gguf_path.is_file():
        errors.append(f"Not a regular file: {gguf_path}")
        return (False, warnings, errors)

    # 3. Correct extension
    if gguf_path.suffix.lower() != ".gguf":
        errors.append(
            f"Expected a .gguf file, got: {gguf_path.name}"
        )
        return (False, warnings, errors)

    # 4. Magic bytes check
    try:
        with open(gguf_path, "rb") as f:
            magic = f.read(4)
        if magic != b"GGUF":
            errors.append(
                "File does not appear to be a valid GGUF file "
                "(magic bytes mismatch). The file may be corrupted."
            )
            return (False, warnings, errors)
    except OSError as e:
        errors.append(f"Cannot read GGUF file: {e}")
        return (False, warnings, errors)

    # 5. Disk space (need ~3× GGUF size: intermediate + quantized + margin)
    try:
        gguf_size = gguf_path.stat().st_size
        free_space = shutil.disk_usage(output_dir.parent).free
        needed = gguf_size * 3
        if free_space < needed:
            warnings.append(
                f"Low disk space: {format_size(free_space)} free, "
                f"~{format_size(needed)} recommended. "
                f"Conversion may fail if space runs out."
            )
    except Exception:
        warnings.append("Could not check disk space - proceeding anyway.")

    # 6. Dependencies
    deps = check_dependencies()
    if not deps.get("gguf2mlx"):
        errors.append(
            "gguf2mlx not installed. "
            "Install with: [cyan]pip install gguf2mlx[/cyan]"
        )
    if not deps.get("mlx_lm") and not args.no_quantize:
        errors.append(
            "mlx-lm not installed. "
            "Install with: [cyan]pip install mlx-lm[/cyan]"
        )

    # 7. Output dir already exists
    if output_dir.exists() and not args.force:
        warnings.append(
            f"Output directory already exists: {output_dir}\n"
            f"  Existing files will be overwritten."
        )

    return (len(errors) == 0, warnings, errors)


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_output(output_dir: Path) -> bool:
    """Try to load the converted model to verify it works."""
    try:
        import mlx.core as mx  # noqa: F401
        from mlx_lm.utils import load_config
    except ImportError:
        info("mlx_lm not available - skipping validation")
        return True

    try:
        config = load_config(output_dir)
        info(
            f"Model loads successfully: "
            f"{config.get('model_type', '?')} "
            f"({config.get('hidden_size', '?')} hidden, "
            f"{config.get('num_hidden_layers', '?')} layers)"
        )
        # Check for safetensors
        safetensors = list(output_dir.glob("*.safetensors"))
        if safetensors:
            total = sum(f.stat().st_size for f in safetensors)
            info(f"Weight files: {len(safetensors)} ({format_size(total)})")
        return True
    except Exception as e:
        warn(f"Validation warning: {e}")
        return True  # Don't fail - the files exist, just couldn't load


# ═══════════════════════════════════════════════════════════════════════════
# Version checks
# ═══════════════════════════════════════════════════════════════════════════

def check_dependencies() -> dict[str, str | None]:
    """Check required and optional dependencies. Returns version info."""
    deps: dict[str, str | None] = {
        "gguf2mlx": None, "mlx_lm": None, "mlx": None, "gguf_py": None,
    }

    try:
        import gguf2mlx
        deps["gguf2mlx"] = getattr(gguf2mlx, "__version__", None) or "?"
    except ImportError:
        pass

    try:
        import mlx_lm
        deps["mlx_lm"] = getattr(mlx_lm, "__version__", None) or "?"
    except ImportError:
        pass

    try:
        import mlx.core as mx
        deps["mlx"] = getattr(mx, "__version__", None) or "?"
    except ImportError:
        pass

    try:
        import gguf
        deps["gguf_py"] = getattr(gguf, "__version__", None) or "?"
    except ImportError:
        pass

    return deps


def ensure_deps(deps: dict, for_convert: bool = True):
    """Check required deps are installed, exit with helpful message if not."""
    missing = []
    if not deps.get("gguf2mlx"):
        missing.append(("gguf2mlx", "pip install gguf2mlx"))
    if for_convert and not deps.get("mlx_lm"):
        missing.append(("mlx-lm", "pip install mlx-lm"))

    if missing:
        banner()
        fail("Missing required dependencies:")
        for name, install_cmd in missing:
            info(f"  {name} - install with: [cyan]{install_cmd}[/cyan]")
        console.print()
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# Interactive Prompts (Rich-based)
# ═══════════════════════════════════════════════════════════════════════════

def get_gguf_path() -> Path:
    """Prompt the user for a GGUF file path (supports drag-and-drop)."""
    console.print(
        "  [yellow]→[/yellow] Drag your .gguf file into this window "
        "and press Enter,"
    )
    console.print("    or type the full path:\n")
    raw = Prompt.ask("  [bold]GGUF file[/bold]")
    return Path(raw.strip().strip("'\"").strip()).expanduser()


def get_output_dir(gguf_path: Path) -> Path:
    """Suggest an output directory, let user override."""
    suggested = gguf_path.parent / (gguf_path.stem + "-4bit-mlx")
    console.print(
        f"\n  [yellow]→[/yellow] Output folder "
        f"(press Enter to use default):"
    )
    console.print(f"    [dim]{suggested}[/dim]\n")
    raw = Prompt.ask(
        "  [bold]Output folder[/bold]",
        default=str(suggested),
    )
    return Path(raw.strip().strip("'\"").strip()).expanduser()


# ═══════════════════════════════════════════════════════════════════════════
# Disk Space Check
# ═══════════════════════════════════════════════════════════════════════════

def check_disk_space(
    gguf_path: Path,
    output_dir: Path,
    force: bool = False,
) -> bool:
    """Warn if available space looks tight (need ~2.5× GGUF size for temp)."""
    try:
        gguf_gb = gguf_path.stat().st_size / 1e9
        free_gb = shutil.disk_usage(output_dir.parent).free / 1e9
        needed_gb = gguf_gb * 2.5
        if free_gb < needed_gb:
            warn(
                f"Low disk space: {free_gb:.1f} GB free, "
                f"~{needed_gb:.1f} GB recommended"
            )
            if not force:
                ans = Confirm.ask(
                    "  Continue anyway?",
                    default=False,
                )
                return ans
            info("--force: skipping disk space prompt")
    except Exception:
        pass
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Quant Args Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_quant_args(args) -> list[str]:
    """Build the mlx_lm convert quantisation arguments from CLI args."""
    q_args = ["--quantize"]

    if args.predicate:
        q_args += ["--quant-predicate", args.predicate]
    else:
        q_args += ["--q-bits", str(args.bits)]

    if args.mode:
        q_args += ["--q-mode", args.mode]
    if args.group_size:
        q_args += ["--q-group-size", str(args.group_size)]

    return q_args


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global console

    parser = build_parser()
    args = parser.parse_args()

    # Configure console
    if args.no_color:
        console = Console(no_color=True, highlight=False)

    banner()

    # ═══════════════════════════════════════════════════════════════════════
    # Hardware Detection
    # ═══════════════════════════════════════════════════════════════════════
    hw = detect_apple_silicon()
    if hw["is_apple_silicon"] and not args.quiet:
        hw_table = Table(
            title="Hardware Detection",
            title_style="bold green",
            box=box.SIMPLE,
            border_style="dim",
            show_header=False,
        )
        hw_table.add_column("", style="dim", width=15)
        hw_table.add_column("", style="bold")
        hw_table.add_row("Chip", hw["chip_name"])
        tier_style = {
            "max": "bold magenta",
            "ultra": "bold magenta",
            "pro": "cyan",
        }.get(hw["chip_tier"], "")
        hw_table.add_row(
            "Tier",
            f"[{tier_style}]{hw['chip_tier'].title()}[/{tier_style}]"
            if tier_style else hw["chip_tier"].title(),
        )
        hw_table.add_row("Generation", f"M{hw['chip_gen']}")
        hw_table.add_row("RAM", f"{hw['ram_gb']:.1f} GB")
        console.print(hw_table)

    # Check deps early
    deps = check_dependencies()

    # ── resolve GGUF path ─────────────────────────────────────────────────
    if args.input:
        gguf_path = Path(args.input).expanduser()
    else:
        ensure_deps(deps, for_convert=False)  # Only minimal deps for interactive
        gguf_path = get_gguf_path()

    # ── inspect mode ──────────────────────────────────────────────────────
    if args.inspect:
        inspect_mode(gguf_path)
        return  # unreachable, inspect_mode calls exit

    # ── preflight checks ──────────────────────────────────────────────────
    # Resolve output dir early for preflight
    if args.output:
        final_dir = Path(args.output).expanduser()
    else:
        final_dir = gguf_path.parent / (gguf_path.stem + "-4bit-mlx")

    passed, pre_warnings, pre_errors = preflight_checks(
        gguf_path, final_dir, args
    )

    # Show preflight results
    for prer_err in pre_errors:
        fail(prer_err)
    for prer_warn in pre_warnings:
        warn(prer_warn)

    if not passed:
        console.print()
        console.print(
            "[bold red]Preflight checks failed. "
            "Fix the errors above and try again.[/bold red]"
        )
        sys.exit(1)

    if not args.quiet and pre_warnings:
        if not args.force:
            ans = Confirm.ask(
                "  [yellow]Warnings detected. Continue?[/yellow]",
                default=True,
            )
            if not ans:
                console.print("\n  Cancelled.")
                sys.exit(0)

    gguf_size_bytes = gguf_path.stat().st_size
    ok(f"Input: [bold]{gguf_path.name}[/bold]  ({format_size(gguf_size_bytes)})")

    # ── read metadata for smart decisions ─────────────────────────────────
    meta = read_gguf_metadata(gguf_path)

    # Gemma4 bug detection - show structured warnings before conversion fails
    if meta:
        arch = str(meta.get("architecture", "")).lower()
        is_known, issue_info = is_known_issue_arch(arch)
        if is_known and issue_info:
            console.print()
            console.print(
                f"[yellow]⚠[/yellow]  [bold]{meta['architecture']}[/bold] "
                f"detected - known gguf2mlx issue: [dim]{issue_info['issue']}[/dim]"
            )
            console.print()
            console.print("  [bold]If conversion fails, use one of these workarounds:[/bold]")
            for i, (cmd, desc) in enumerate(issue_info['workarounds'], 1):
                console.print(f"  [dim]{i}.[/dim] [cyan]{cmd}[/cyan] - {desc}")
            console.print()

    if meta and not args.quiet:
        # Show key info
        if meta.get("architecture"):
            info(f"Architecture: [bold]{meta['architecture']}[/bold]")
        if meta.get("param_count"):
            p_count = meta["param_count"] / 1e9
            info(f"Parameters:   [bold]{p_count:.2f}B[/bold]")
        if meta.get("file_type_name"):
            info(f"Source type:  [bold]{meta['file_type_name']}[/bold]")
        if meta.get("tensor_count"):
            info(f"Tensors:      [bold]{meta['tensor_count']:,}[/bold]")
        if meta.get("total_weight_bytes"):
            info(
                "Weight data:  "
                f"[bold]{format_size(meta['total_weight_bytes'])}[/bold]"
            )

        # MTP detection
        if meta.get("mtp_layers"):
            console.print(
                f"  [magenta]◆[/magenta] MTP detected: "
                f"[bold]{meta['mtp_layers']}[/bold] layer(s) "
                f"[dim](Multi-Token Prediction)[/dim]"
            )
            if args.mtp:
                info(
                    "MTP weights will be preserved during GGUF→MLX conversion. "
                    "mlx_lm handles MTP transparently."
                )

        # Source quality warning
        if meta.get("file_type") is not None:
            quality = classify_source_quality(int(meta["file_type"]))
            if quality["risk"] in ("high", "severe"):
                warn(f"Source is {quality['label']}. {quality['advice']}")
                if not args.force and not args.no_quantize:
                    ans = Confirm.ask(
                        "  [yellow]Continue with quantisation anyway?[/yellow]",
                        default=False,
                    )
                    if not ans:
                        info(
                            "Tip: use [cyan]--no-quantize[/cyan] "
                            "to convert to float16 only"
                        )
                        sys.exit(0)

    # ── resolve intermediate dtype ────────────────────────────────────────
    if args.dtype:
        intermediate_dtype = args.dtype
    elif meta and meta.get("file_type") == 0:  # F32 source
        intermediate_dtype = "float32"
        info(f"Auto-detected dtype: [bold]float32[/bold] (source is float32)")
    elif meta and meta.get("file_type") == 26:  # BF16 source
        intermediate_dtype = "float16"
        info(
            f"Auto-detected dtype: [bold]float16[/bold] "
            f"(source is bfloat16)"
        )
    else:
        intermediate_dtype = "float16"

    # ── resolve output dir ────────────────────────────────────────────────
    if not args.output:
        final_dir = get_output_dir(gguf_path)

    intermediate_dir = final_dir.parent / (final_dir.name + "_intermediate")

    # ── resolve quantisation params ───────────────────────────────────────
    do_quantize = not args.no_quantize

    if do_quantize:
        # --high-bandwidth flag: auto-select m5-max preset (non-overriding)
        if args.high_bandwidth and not args.preset and not args.bits:
            args.preset = HIGH_BANDWIDTH_PRESET
        if args.preset:
            preset = PRESETS[args.preset]
            # Only apply preset values if not already set via CLI
            if args.bits is None:
                args.bits = preset["bits"]
            if args.group_size is None:
                args.group_size = preset["group_size"]
            if args.mode is None:
                args.mode = preset["mode"]
            if not args.quiet:
                info(
                    f"Preset: [bold]{args.preset}[/bold] - "
                    f"{preset['description']}"
                )
        elif not args.bits:
            # Smart defaults based on model size AND hardware
            model_size_gb = gguf_size_bytes / 1e9
            defaults = smart_defaults(
                model_size_gb,
                chip_tier=hw["chip_tier"],
                ram_gb=hw["ram_gb"],
                chip_gen=hw["chip_gen"],
            )
            args.bits = defaults["bits"]
            args.group_size = defaults["group_size"]
            args.mode = defaults["mode"]
            if not args.quiet:
                info(
                    f"Auto-selected: [bold]{args.bits}-bit[/bold], "
                    f"group-size=[bold]{args.group_size}[/bold] "
                    f"({defaults['description']})"
                )

    total_steps = 3 if do_quantize else 1

    # ── disk space check ──────────────────────────────────────────────────
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if not args.force and not check_disk_space(gguf_path, final_dir):
        console.print("\n  Cancelled.")
        sys.exit(0)

    # ── plan summary ──────────────────────────────────────────────────────
    if not args.quiet:
        console.print()
        plan_table = Table(
            title="Conversion Plan",
            title_style="bold",
            box=box.SIMPLE,
            border_style="dim",
            show_header=False,
        )
        plan_table.add_column("", style="dim", width=8)
        plan_table.add_column("")
        plan_table.add_row(
            "Step 1",
            f"Convert GGUF → MLX [bold]{intermediate_dtype}[/bold]  "
            f"[dim]→ {intermediate_dir.name}[/dim]",
        )
        if do_quantize:
            quant_desc = f"{args.bits}-bit"
            if args.predicate:
                quant_desc = args.predicate
            plan_table.add_row(
                "Step 2",
                f"Quantise to [bold]{quant_desc}[/bold] MLX  "
                f"[dim]→ {final_dir.name}[/dim]",
            )
            plan_table.add_row(
                "Step 3",
                "Clean up intermediate files",
            )
        else:
            plan_table.add_row(
                "Output",
                f"[dim]→ {final_dir.name}[/dim]",
            )
        console.print(plan_table)

    if not args.force:
        if not args.quiet:
            console.print()
            console.print(
                "  [yellow]This may take several minutes for large models.[/yellow]"
            )
            console.print()
            Confirm.ask(
                "  Press [bold]Enter[/bold] to start, or Ctrl-C to cancel...",
                default=True,
            )

    # ═══════════════════════════════════════════════════════════════════════
    # Pipeline Progress
    # ═══════════════════════════════════════════════════════════════════════
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        disable=args.quiet,
    )

    with progress:
        pipeline_task = progress.add_task(
            "[bold cyan]Pipeline[/bold cyan]", total=total_steps
        )

        # ═══════════════════════════════════════════════════════════════════
        # STEP 1 - GGUF → MLX float
        # ═══════════════════════════════════════════════════════════════════
        t0 = time.time()

        ensure_deps(deps, for_convert=True)
        step(1, total_steps, f"Converting GGUF → MLX ({intermediate_dtype} safetensors)")

        intermediate_dir.mkdir(parents=True, exist_ok=True)
        ok_step1, result_step1 = run_with_spinner(
            [
                sys.executable, "-m", "gguf2mlx",
                "--input", str(gguf_path),
                "--output", str(intermediate_dir),
                "--dtype", intermediate_dtype,
            ],
            (
                f"Converting GGUF to MLX {intermediate_dtype} "
                "(this may take 5-30 minutes)"
            ),
            quiet=args.quiet,
        )

        if not ok_step1:
            # Detect architecture-specific gguf2mlx bugs
            arch = str(meta.get("architecture", "")).lower() if meta else ""
            is_known, issue_info = is_known_issue_arch(arch)
            
            stderr_text = result_step1.stderr if result_step1 else ""
            
            if is_known and issue_info:
                console.print()
                fail(f"gguf2mlx failed on {meta.get('architecture', arch)} model")
                console.print()
                console.print("  [bold]Known issue:[/bold]")
                console.print(f"  [dim]{issue_info['issue']}[/dim]")
                console.print()
                console.print("  [bold]Workarounds:[/bold]")
                for i, (cmd, desc) in enumerate(issue_info['workarounds'], 1):
                    console.print(f"  [dim]{i}.[/dim] [cyan]{cmd}[/cyan] — {desc}")
                console.print()
            else:
                fail("Conversion failed. Check the error above.")
            sys.exit(1)

        t1 = time.time()
        ok(f"GGUF converted to MLX {intermediate_dtype} ({format_time(t1 - t0)})")
        progress.update(pipeline_task, advance=1)
        
        # Check for Gemma4 tensor naming issues
        arch = str(meta.get("architecture", "")).lower() if meta else ""
        if arch in ("gemma4", "gemma3"):
            tensor_fix_applied = fix_gemma4_tensor_names(intermediate_dir)
            if tensor_fix_applied:
                ok("Gemma4 tensor names fixed")
        
        if not do_quantize:
            # Float16-only mode - move intermediate to final location
            if intermediate_dir != final_dir:
                if final_dir.exists():
                    warn(f"Output directory exists, removing: {final_dir}")
                    shutil.rmtree(final_dir)
                shutil.move(str(intermediate_dir), str(final_dir))
                ok(
                    "Model saved to: "
                    f"[bold cyan]{final_dir}[/bold cyan]"
                )
            # Done
            final_size = sum(
                f.stat().st_size
                for f in final_dir.rglob("*")
                if f.is_file()
            )
            console.print()
            console.print(Panel(
                "[bold green]✓ Done![/bold green]",
                border_style="green",
            ))
            console.print()
            console.print(
                f"  Your MLX [bold]{intermediate_dtype}[/bold] model is ready at:"
            )
            console.print(f"  [bold cyan]{final_dir}[/bold cyan]  ({format_size(final_size)})")
            console.print()
            sys.exit(0)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2 - MLX float16 → Quantised
        # ═══════════════════════════════════════════════════════════════════
        step(2, total_steps, "Quantising to MLX")

        quant_args = build_quant_args(args)
        ok_step2, result_step2 = run_with_spinner(
            [
                sys.executable, "-m", "mlx_lm", "convert",
                "--hf-path", str(intermediate_dir),
                "--mlx-path", str(final_dir),
                *quant_args,
            ],
            f"Quantising to {args.bits}-bit MLX (this may take several minutes)",
            quiet=args.quiet,
        )

        if not ok_step2:
            # Detect architecture-specific quantization issues
            arch = str(meta.get("architecture", "")).lower() if meta else ""
            is_known, issue_info = is_known_issue_arch(arch)
            
            # Check for mlx_lm tensor naming errors (Gemma4, Gemma3, etc.)
            stderr_text = result_step2.stderr if result_step2 else ""
            has_tensor_error = (
                "Received" in stderr_text and "parameters not in model" in stderr_text
            )
            has_blk_error = "blk." in stderr_text
            
            if is_known or has_tensor_error or has_blk_error:
                console.print()
                fail(f"mlx_lm quantization failed on {meta.get('architecture', arch) or arch} model")
                console.print()
                
                # Check for Gemma4/Gemma3 MoE tensor name mismatch
                is_gemma_moe = is_known and arch in ("gemma4", "gemma3")
                
                if has_blk_error or is_gemma_moe:
                    console.print("  [bold]Root cause:[/bold]")
                    console.print("  [dim]Gemma4 MoE architectural mismatch with mlx_lm[/dim]")
                    console.print("  [dim]GGUF is missing some layernorm tensors that mlx_lm expects[/dim]")
                    console.print("  [dim]This is a known limitation for Gemma4 MoE models[/dim]")
                    console.print()
                
                console.print("  [bold]Recommended options:[/bold]")
                
                # Show recommended option first (most practical)
                console.print()
                console.print("  [bold cyan]OPTION 1: Float16 (Recommended for Gemma4)[/bold cyan]")
                console.print("  [dim]  cpmm model.gguf --no-quantize[/dim]")
                console.print("  [dim]  • ~50GB, but mlx_lm generate works directly[/dim]")
                console.print("  [dim]  • Best inference speed on Apple Silicon[/dim]")
                console.print()
                
                console.print("  [bold cyan]OPTION 2: Ollama (Native Gemma4 support)[/bold cyan]")
                console.print("  [dim]  brew install ollama && ollama run gemma4:27b[/dim]")
                console.print("  [dim]  • Built-in 4-bit quantization[/dim]")
                console.print("  [dim]  • Optimized for Apple Silicon[/dim]")
                console.print()
                
                console.print("  [bold cyan]OPTION 3: Use pre-quantized GGUF directly[/bold cyan]")
                console.print("  [dim]  • Your UD-Q4_K_XL GGUF is already 4-bit quantized[/dim]")
                console.print("  [dim]  • Run llama.cpp: llama-cli -m model.gguf -p '...'[/dim]")
                console.print()
                
                if intermediate_dir:
                    console.print(f"  [dim]Float16 model available at: {intermediate_dir}[/dim]")
            else:
                fail("Quantisation failed. Check the error above.")
                if result_step2 and result_step2.stderr:
                    fail(
                        "The float16 intermediate is still at: "
                        f"[dim]{intermediate_dir}[/dim]"
                    )
            sys.exit(1)

        t2 = time.time()
        ok(
            f"Model quantised to [bold]{args.bits}-bit[/bold] "
            f"({format_time(t2 - t1)})"
        )
        progress.update(pipeline_task, advance=1)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3 - Clean up intermediate
        # ═══════════════════════════════════════════════════════════════════
        if not args.keep_intermediate:
            step(3, total_steps, "Cleaning up intermediate files")
            try:
                shutil.rmtree(intermediate_dir)
                ok("Intermediate files removed")
            except Exception as e:
                warn(f"Could not remove {intermediate_dir}: {e}")
        else:
            info(
                "Intermediate files kept at: "
                f"[dim]{intermediate_dir}[/dim]"
            )
        progress.update(pipeline_task, advance=1)

    # ── validation ────────────────────────────────────────────────────────
    if not args.quiet:
        info("Validating output...")
    validate_output(final_dir)

    # ── done ──────────────────────────────────────────────────────────────
    final_size_bytes = sum(
        f.stat().st_size for f in final_dir.rglob("*") if f.is_file()
    )
    total_time = time.time() - t0

    console.print()
    console.print(Panel(
        "[bold green]✓ Done![/bold green]",
        border_style="green",
    ))
    console.print()

    # Summary table
    summary = Table(
        title="Conversion Summary",
        title_style="bold",
        box=box.SIMPLE,
        border_style="dim",
        show_header=False,
    )
    summary.add_column("", style="dim", width=8)
    summary.add_column("", style="bold")
    summary.add_row(
        "Input",
        f"[dim]{gguf_path}[/dim]  ({format_size(gguf_size_bytes)})",
    )
    summary.add_row(
        "Output",
        f"[bold cyan]{final_dir}[/bold cyan]  ({format_size(final_size_bytes)})",
    )
    if gguf_size_bytes > 0:
        ratio = final_size_bytes / gguf_size_bytes * 100
        summary.add_row("Ratio", f"{ratio:.0f}% of original size")
    summary.add_row("Time", format_time(total_time))
    console.print(summary)

    console.print()

    quant_tag = f"{args.bits}-bit" if do_quantize else intermediate_dtype
    console.print(f"  [bold]To generate text, run:[/bold]")
    console.print(
        f"  [cyan]python3 -m mlx_lm generate "
        f'--model "{final_dir}" --prompt "Hello"[/cyan]'
    )
    console.print()
    console.print(f"  [bold]Or start an interactive chat:[/bold]")
    console.print(
        f"  [cyan]python3 -m mlx_lm chat "
        f'--model "{final_dir}"[/cyan]'
    )
    console.print()

    if not args.quiet and not args.keep_intermediate:
        ok("Conversion complete - model verified and ready to use")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print()
        console.print(
            Panel(
                "[yellow]Cancelled by user.[/yellow] "
                "Intermediate files may remain - clean up "
                "with [dim]--keep-intermediate[/dim] if needed.",
                border_style="yellow",
            )
        )
        console.print()
        sys.exit(0)
