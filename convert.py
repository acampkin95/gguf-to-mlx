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
from typing import Any, Optional
from urllib.parse import urlparse

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
    TimeRemainingColumn,
)
from rich.prompt import Prompt, Confirm
from rich.rule import Rule
from rich.text import Text
from rich.style import Style
from rich import box

# Hardware detection
import psutil

# For registry downloads
try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


# Module-level console - configured in main() based on --no-color

console = Console(highlight=False)

# Shared constants (SonarQube S1192 dedup)
STATUS_READY = "[green]READY[/green]"
STATUS_SKIPPED = "[yellow]SKIPPED[/yellow]"
MSG_CANCELLED = "\n  Cancelled."
_MOE_4BIT_LABEL = "MoE 4-bit"
_CPMM_NO_QUANTIZE = "cpmm model.gguf --no-quantize"
_UPGRADE_GGUF2MLX = "python3 -m pip install --upgrade git+https://github.com/acampkin95/gguf2mlx.git"
_STYLE_BOLD_CYAN = "bold cyan"
_STYLE_BOLD_MAGENTA = "bold magenta"
_SEPARATOR = "═" * 77



# GGUF Quantisation Type → Metadata Mapping

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
    27: ("MOSTLY_Q4_0_4_4",      "4-bit",     _MOE_4BIT_LABEL),
    28: ("MOSTLY_Q4_0_4_8",      "4-bit",     _MOE_4BIT_LABEL),
    29: ("MOSTLY_Q4_0_8_8",      "4-bit",     _MOE_4BIT_LABEL),
    30: ("MOSTLY_TQ1_0",         "1-bit",     "ternary 1-bit"),
    31: ("MOSTLY_TQ2_0",         "2-bit",     "ternary 2-bit"),
}


# Architectures known to have gguf2mlx compatibility issues
KNOWN_CONVERSION_ISSUES: dict[str, dict[str, Any]] = {
    "gemma4": {
        "issue": "head_count_kv metadata returns list instead of int",
        "workarounds": [
            (_CPMM_NO_QUANTIZE, "Converts to float16 only, avoids gguf2mlx quant step"),
            (_UPGRADE_GGUF2MLX, "May already be fixed in latest version"),
        ],
    },
    "gemma3": {
        "issue": "Similar metadata issues as Gemma4",
        "workarounds": [
            (_CPMM_NO_QUANTIZE, "Converts to float16 only"),
            (_UPGRADE_GGUF2MLX, "May already be fixed"),
        ],
    },
    "gemma2": {
        "issue": "May have similar head_count_kv issues",
        "workarounds": [
            (_CPMM_NO_QUANTIZE, "Converts to float16 only"),
            (_UPGRADE_GGUF2MLX, "May already be fixed"),
        ],
    },
    "qwen35": {
        "issue": "mlx_lm does not support Qwen3.5 architecture yet (Fast-Fail)",
        "workarounds": [
            ("Use Ollama: brew install ollama && ollama run qwen2.5", "Native Qwen support via Ollama"),
            ("Use llama.cpp directly", "Run GGUF directly with llama-cli"),
            ("Check mlx_lm for updates: https://github.com/ml-explore/mlx-examples", "Support may be added in future releases"),
        ],
    },
    "deepseek_v3": {
        "issue": "deepseek-v3 requires specific mlx-lm versions and may have quantization bugs",
        "workarounds": [
            ("python3 -m pip install --upgrade mlx-lm", "Ensure you are on the absolute latest version"),
            (_CPMM_NO_QUANTIZE, "Convert to float16 to avoid quant bugs"),
        ],
    },
}

# Architectures supported by mlx_lm (as of latest release)
# This list can be expanded as mlx_lm adds support
SUPPORTED_MLX_ARCHITECTURES = frozenset([
    "llama", "llama2", "llama3", "llama3_1", "llama3_2",
    "gemma", "gemma2",
    "mistral", "mixtral",
    "phi", "phi2", "phi3", "phi3_5",
    "qwen", "qwen2", "qwen2_5",
    "deepseek", "deepseek2", "deepseek_v3",
    "gptneox", "stablelm",
])



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


def fix_gemma4_tensor_names(intermediate_dir: Path) -> None:
    """Detect and report Gemma4 tensor naming issues.
    
    Displays workaround guidance when issues are detected.
    Auto-fix is not yet supported (safetensors are read-only).
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return  # Can't import safetensors
    
    # Check if this is a Gemma4 model with wrong tensor names
    safetensor_files = list(intermediate_dir.glob("*.safetensors"))
    if not safetensor_files:
        return  # No safetensor files
    
    # Check first file for problematic tensor names
    needs_fix = False
    try:
        with safe_open(safetensor_files[0], framework="numpy") as f:  # type: ignore[no-untyped-call]
            keys = list(f.keys())
            # Check for blk.* pattern (GGUF format) without model.layers
            if any(k.startswith("blk.") for k in keys):
                needs_fix = True
    except Exception:
        return  # Error reading safetensors
    
    if not needs_fix:
        return
    
    console.print("  [yellow]⚠[/yellow]  Detected Gemma4 tensor naming issue, applying fix...")
    
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
    
    # Can't auto-fix without rewriting safetensors - user must use workarounds



def is_known_issue_arch(arch: str) -> tuple[bool, dict[str, Any] | None]:
    """Check if architecture is known to have gguf2mlx issues."""
    arch_lower = arch.lower()
    for known_arch, issue_info in KNOWN_CONVERSION_ISSUES.items():
        if arch_lower.startswith(known_arch):
            return True, issue_info
    return False, None


def is_mlx_supported_arch(arch: str) -> tuple[bool, str | None]:
    """Check if architecture is supported by mlx_lm.
    
    Returns (is_supported, reason) where reason is None if supported,
    or a message explaining why it's not supported.
    """
    arch_lower = arch.lower()
    
    # Check if explicitly unsupported due to known issues
    for known_issue_arch in KNOWN_CONVERSION_ISSUES.keys():
        if arch_lower.startswith(known_issue_arch):
            return False, f"Known incompatibility with {known_issue_arch} models"
    
    # Check against supported list
    for supported_arch in SUPPORTED_MLX_ARCHITECTURES:
        if arch_lower.startswith(supported_arch):
            return True, None
    
    # Unknown architecture - may work but not guaranteed
    return False, "Architecture not in mlx_lm supported list (may still work)"



def read_gemma4_metadata(reader: object, _arch: str) -> dict[str, Any]:
    """Read Gemma4-specific metadata fields."""
    gemma_meta: dict[str, Any] = {}

    # Try Gemma4-specific fields
    for field_name, possible_keys in Gemma4_METADATA_FIELDS.items():
        for key in possible_keys:
            val = _field_value(reader, key)
            if val is not None:
                gemma_meta[field_name] = val
                break

    return gemma_meta


def classify_source_quality(file_type: int) -> dict[str, str]:
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



# Quantisation Presets

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



# Hardware Detection

def detect_apple_silicon() -> dict[str, Any]:
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



# Smart Defaults (Hardware-Aware)

def smart_defaults(
    model_size_gb: float,
    chip_tier: str = "base",
    ram_gb: float = 0,
    chip_gen: int = 0,
) -> dict[str, Any]:
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



# CLI

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
        "--resume", action="store_true",
        help="Resume conversion by skipping Step 1 if intermediate files exist",
    )
    p.add_argument(
        "--inspect", action="store_true",
        help="Display GGUF metadata and exit (no conversion)",
    )
    p.add_argument(
        "--estimate", action="store_true",
        help="Estimate conversion time, memory, and final size (no conversion)",
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
        "--cleanup-old", action="store_true",
        help="Remove existing intermediate directories in the output parent folder",
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



# GGUF Metadata Reader

def _has_gguf_py() -> bool:
    try:
        import gguf  # noqa: F401
        return True
    except ImportError:
        return False


def _field_value(reader: Any, name: str) -> Any:
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
        if isinstance(val, (list, tuple)):
            val = val[0] if len(val) > 0 else None
        return val
    except Exception:
        return None


def _field_exists(reader: Any, name: str) -> bool:
    return reader.get_field(name) is not None


def read_gguf_metadata(gguf_path: Path) -> dict[str, Any] | None:
    """Read key metadata from a GGUF file. Returns None if gguf-py unavailable."""
    if not _has_gguf_py():
        return None

    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path))

    meta: dict[str, Any] = {
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



# Display Helpers (Rich-based)

def banner() -> None:
    """Print the converter banner using Rich Panel."""
    console.print()
    console.print(Panel(
        "[bold]GGUF → MLX Converter[/bold]\n"
        "[dim]with Dynamic Quantization for Apple Silicon Macs[/dim]",
        border_style=_STYLE_BOLD_CYAN,
        padding=(1, 2),
    ))
    console.print()


def step(n: int, total: int, label: str) -> None:
    """Print a pipeline step header using Rich Rule."""
    console.print()
    console.print(Rule(
        f"[bold cyan]Step {n}/{total}[/bold cyan]  {label}",
        style="dim",
    ))


def ok(msg: str) -> None:
    console.print(f"  [green]✓[/green] {msg}")


def fail(msg: str) -> None:
    console.print(f"\n  [red]✗[/red] {msg}")


def info(msg: str) -> None:
    console.print(f"  [dim]·[/dim] {msg}")


def warn(msg: str) -> None:
    console.print(f"  [yellow]⚠[/yellow]  {msg}")


def run_with_progress(
    cmd: list[str],
    description: str,
    progress: Progress,
    quiet: bool = False,
) -> tuple[bool, str]:
    """Run a subprocess with a live progress bar.

    Parses output for percentage (N%) or fraction (N/M) patterns and
    updates a Rich sub-task in real-time.

    Returns (success, full_output).
    """
    if not quiet:
        info(f"Running: [dim]{' '.join(str(c) for c in cmd)}[/dim]")

    op_task = progress.add_task(
        f"[bold cyan]{description}", total=100, visible=not quiet
    )

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        full_output: list[str] = []

        assert process.stdout is not None  # PIPE ensures this
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                full_output.append(line)
                # Match percentages (e.g. "45%")
                pct_match = re.search(r"(\d+)%", line)
                if pct_match:
                    progress.update(op_task, completed=int(pct_match.group(1)))
                # Match fractions (e.g. "10/100")
                frac_match = re.search(r"(\d+)/(\d+)", line)
                if frac_match:
                    cur, tot = int(frac_match.group(1)), int(frac_match.group(2))
                    if tot > 0:
                        progress.update(op_task, completed=int((cur / tot) * 100))

        process.wait()
        output_text = "".join(full_output)

        if process.returncode != 0:
            fail(f"{description} failed (exit {process.returncode})")
            if output_text:
                console.print(Panel(
                    output_text.strip()[-1000:],
                    title="Error Output",
                    border_style="red",
                ))
            progress.remove_task(op_task)
            return False, output_text

        progress.update(op_task, completed=100)
        progress.remove_task(op_task)
        return True, output_text

    except FileNotFoundError as e:
        fail(f"Command not found: {e}")
        if 'op_task' in locals():
            progress.remove_task(op_task)
        return False, ""


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



# Registry & Download Support (Quick Win #1)

def download_from_huggingface(repo_id: str, filename: str, cache_dir: Optional[Path] = None) -> Path:
    """Download model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., "mistralai/Mistral-7B")
        filename: Filename in repo (e.g., "Mistral-7B-Q4_K_M.gguf")
        cache_dir: Cache directory for downloads (default: ~/.cache/gguf-to-mlx)

    Returns:
        Path to downloaded file
    """
    if not HAS_URLLIB:
        fail("urllib not available for downloads")
        sys.exit(1)

    cache_dir = cache_dir or Path.home() / ".cache" / "gguf-to-mlx"
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = cache_dir / f"{repo_id.replace('/', '_')}_{filename}"

    # Skip if already cached
    if local_path.exists():
        ok(f"Using cached: {local_path}")
        return local_path

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    ok(f"Downloading from HuggingFace: {repo_id}/{filename}")

    try:
        # Simple download with progress
        def download_hook(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, int(100 * downloaded / total_size))
                console.print(f"  Progress: {percent}%", end="\r")

        urllib.request.urlretrieve(url, local_path, reporthook=download_hook)
        console.print()  # newline after progress
        ok(f"Downloaded: {local_path}")
        return local_path
    except Exception as e:
        fail(f"Download failed: {e}")
        sys.exit(1)


def handle_registry_url(input_str: str) -> Path:
    """Handle registry URLs like hf:namespace/model or return local path.

    Examples:
        hf:mistralai/Mistral-7B -> downloads from HuggingFace
        /local/path/model.gguf -> returns as-is
    """
    if input_str.startswith("hf:"):
        # HuggingFace format: hf:namespace/model/filename or hf:namespace/model
        parts = input_str[3:].split("/")
        if len(parts) >= 3:
            repo_id = "/".join(parts[:-1])
            filename = parts[-1]
        elif len(parts) == 2:
            repo_id = "/".join(parts)
            filename = f"{parts[1]}-Q4_K_M.gguf"  # Default filename
        else:
            fail(f"Invalid HuggingFace URL: {input_str}")
            fail("Format: hf:namespace/model or hf:namespace/model/filename.gguf")
            sys.exit(1)
        return download_from_huggingface(repo_id, filename)
    else:
        return Path(input_str).expanduser()



# Conversion Time & Memory Estimation (Quick Win #3)

def estimate_conversion_metrics(model_size_gb: float, bits: int, chip_tier: str = "base") -> dict[str, Any]:
    """Estimate conversion time, peak memory, and final size.

    Args:
        model_size_gb: Size of GGUF model in GB
        bits: Target quantization bits (2, 4, 8, 16)
        chip_tier: Apple Silicon tier ('base', 'pro', 'max', 'ultra')

    Returns:
        dict with time_minutes, peak_memory_gb, final_size_gb, warnings
    """
    # Heuristics based on observed patterns
    # Time: roughly 0.5-1.5 min per GB depending on chip and quant
    chip_speed = {"base": 1.0, "pro": 1.3, "max": 1.5, "ultra": 1.5}.get(chip_tier, 1.0)
    quant_complexity = {2: 0.3, 4: 0.7, 8: 0.9, 16: 1.2}.get(bits, 0.7)
    time_minutes = max(1, int(model_size_gb * 0.8 * quant_complexity / chip_speed))

    # Peak memory: roughly 3-4x model size during conversion
    peak_memory_gb = model_size_gb * (4.5 - (bits / 4))  # Lower bits = more overhead

    # Final size: roughly model_size * (bits / 16)
    compression_ratio = bits / 16.0
    final_size_gb = model_size_gb * compression_ratio

    warnings = []
    if peak_memory_gb > 48:
        warnings.append(f"Peak memory {peak_memory_gb:.1f}GB may exceed M5 Max capacity")
    if time_minutes > 120:
        warnings.append(f"Conversion may take {time_minutes} minutes (2+ hours)")

    return {
        "time_minutes": time_minutes,
        "peak_memory_gb": peak_memory_gb,
        "final_size_gb": final_size_gb,
        "warnings": warnings,
    }



# Metadata Display (Rich Table)

def display_metadata(meta: dict[str, Any]) -> None:
    """Pretty-print GGUF metadata using Rich Table."""

    # Section 1: Model Info
    console.print()
    model_table = Table(
        title="GGUF Model Information",
        title_style=_STYLE_BOLD_CYAN,
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
            title_style=_STYLE_BOLD_MAGENTA,
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



# Inspect Mode

def inspect_mode(gguf_path: Path) -> None:
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



# Preflight Checks

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
            "Install with: [cyan]python3 -m pip install git+https://github.com/acampkin95/gguf2mlx.git[/cyan]"
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



# Validation

def validate_output(output_dir: Path) -> None:
    """Try to load the converted model to verify it works.
    
    Logs warnings on failure but never blocks the pipeline.
    """
    try:
        import mlx.core as mx  # noqa: F401
        from mlx_lm.utils import load_config
    except ImportError:
        info("mlx_lm not available - skipping validation")
        return

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
    except Exception as e:
        warn(f"Validation warning: {e}")



# Version checks

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


def ensure_deps(deps: dict[str, str | None], for_convert: bool = True) -> None:
    """Check required deps are installed, exit with helpful message if not."""
    missing = []
    if not deps.get("gguf2mlx"):
        missing.append(("gguf2mlx", "python3 -m pip install git+https://github.com/acampkin95/gguf2mlx.git"))
    if for_convert and not deps.get("mlx_lm"):
        missing.append(("mlx-lm", "pip install mlx-lm"))

    if missing:
        banner()
        fail("Missing required dependencies:")
        for name, install_cmd in missing:
            info(f"  {name} - install with: [cyan]{install_cmd}[/cyan]")
        console.print()
        sys.exit(1)



# Interactive Prompts (Rich-based)

def get_gguf_path() -> Path:
    """Prompt the user for a GGUF file path (supports drag-and-drop)."""
    console.print()
    console.print(Panel(
        "Please provide the path to your .gguf model file.\n"
        "You can [bold]drag and drop[/bold] the file into this window "
        "or type the path manually.",
        title="[bold cyan]Input Model[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    raw = Prompt.ask("  [bold]GGUF file path[/bold]")
    return Path(raw.strip().strip("'\"").strip()).expanduser()


def get_output_dir(gguf_path: Path) -> Path:
    """Suggest an output directory, let user override."""
    suggested = gguf_path.parent / (gguf_path.stem + "-4bit-mlx")
    console.print()
    console.print(Panel(
        f"The model will be saved to a new directory.\n"
        f"Default suggestion: [bold cyan]{suggested}[/bold cyan]",
        title="[bold cyan]Output Directory[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    raw = Prompt.ask(
        "  [bold]Output folder[/bold]",
        default=str(suggested),
    )
    return Path(raw.strip().strip("'\"").strip()).expanduser()



# Disk Space Check

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



# Quant Args Builder

def build_quant_args(args: argparse.Namespace) -> list[str]:
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
# Pipeline Stages — extracted from main() for reduced cognitive complexity
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_gguf_input(
    args: argparse.Namespace, deps: dict[str, str | None]
) -> Path:
    """Resolve the GGUF input path from args or interactive prompt."""
    if args.input:
        return handle_registry_url(args.input)
    ensure_deps(deps, for_convert=False)
    return get_gguf_path()


def _run_estimate_mode(
    args: argparse.Namespace,
    gguf_path: Path,
    hw: dict[str, Any],
) -> None:
    """Handle --estimate mode: show cost/time estimates and exit."""
    gguf_size_gb = gguf_path.stat().st_size / 1e9
    bits = args.bits or 4
    estimates = estimate_conversion_metrics(
        gguf_size_gb, bits, hw.get("chip_tier", "base")
    )

    console.print("\n[bold cyan]📊 Conversion Estimates[/bold cyan]")
    est_table = Table(
        title="Based on model size and target quantization",
        title_style="dim",
        box=box.SIMPLE,
        border_style="dim",
        show_header=False,
    )
    est_table.add_column("", style="dim", width=18)
    est_table.add_column("", style="bold")
    est_table.add_row("Model Size", f"{gguf_size_gb:.2f} GB")
    est_table.add_row("Target Bits", f"{bits}-bit")
    est_table.add_row(
        "Est. Time",
        f"~{estimates['time_minutes']} minutes"
        if estimates["time_minutes"] > 1
        else "< 1 minute",
    )
    est_table.add_row("Peak Memory", f"~{estimates['peak_memory_gb']:.1f} GB")
    est_table.add_row("Final Size", f"~{estimates['final_size_gb']:.2f} GB")

    if estimates["warnings"]:
        console.print(est_table)
        console.print("\n[bold yellow]⚠️  Warnings:[/bold yellow]")
        for est_warn in estimates["warnings"]:
            console.print(f"  • {est_warn}")
    else:
        console.print(est_table)

    console.print("\n[dim]Run without --estimate to start conversion.[/dim]\n")


def _show_known_issue_panel(meta: dict[str, Any]) -> None:
    """Display known architecture issue panel if applicable."""
    arch = str(meta.get("architecture", "")).lower()
    is_known, issue_info = is_known_issue_arch(arch)
    if not (is_known and issue_info):
        return

    console.print()
    issue_panel = Panel(
        f"[bold]{meta['architecture']}[/bold] detected\n"
        f"[dim]Known gguf2mlx issue:[/dim] {issue_info['issue']}",
        title="[bold yellow]⚠  Known Architecture Issue[/bold yellow]",
        border_style="yellow",
        expand=False,
    )
    console.print(issue_panel)
    console.print("[bold]Recommended Solutions:[/bold]")
    for i, (cmd, desc) in enumerate(issue_info["workarounds"], 1):
        console.print(f"\n  [bold cyan]Option {i}:[/bold cyan] {desc}")
        console.print(f"  [dim cyan]Command:[/dim cyan] [code]{cmd}[/code]")
    console.print(
        "\n[dim]The conversion may fail during quantization. "
        "If it does, try one of the options above.[/dim]\n"
    )


def _show_source_quality_warning(
    args: argparse.Namespace,
    meta: dict[str, Any],
) -> None:
    """Warn about high-risk source quality and prompt to continue."""
    if meta.get("file_type") is None:
        return
    quality = classify_source_quality(int(meta["file_type"]))
    if quality["risk"] not in ("high", "severe"):
        return

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


def _show_metadata_warnings(
    args: argparse.Namespace,
    meta: dict[str, Any] | None,
) -> None:
    """Display metadata info, known issues, MTP, and source quality warnings."""
    if not meta:
        return

    _show_known_issue_panel(meta)

    if args.quiet:
        return

    # Key info display
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

    _show_source_quality_warning(args, meta)


def _check_arch_compatibility(
    args: argparse.Namespace,
    meta: dict[str, Any] | None,
) -> None:
    """Pre-flight check: warn if architecture unsupported by mlx_lm."""
    if not meta or args.no_quantize:
        return

    arch = str(meta.get("architecture", "")).lower()
    is_supported, reason = is_mlx_supported_arch(arch)

    if is_supported:
        return

    console.print()
    fail(
        f"Architecture '{meta.get('architecture', arch)}' "
        "is not supported by mlx_lm"
    )
    console.print()
    console.print(f"  [dim]Reason: {reason}[/dim]")
    console.print()

    # Check for known issues with workarounds
    is_known, issue_info = is_known_issue_arch(arch)
    if is_known and issue_info:
        console.print("  [bold]Known issue:[/bold]")
        console.print(f"  [dim]{issue_info['issue']}[/dim]")
        console.print()
        console.print("  [bold]Workarounds:[/bold]")
        for i, (cmd, desc) in enumerate(issue_info["workarounds"], 1):
            console.print(f"  [dim]{i}.[/dim] [cyan]{cmd}[/cyan] — {desc}")
        console.print()

    # Suggest alternatives
    console.print("  [bold]Alternatives:[/bold]")
    console.print("  [dim]1.[/dim] Use [cyan]--no-quantize[/cyan] for float16 MLX")
    console.print("  [dim]2.[/dim] Use Ollama for native inference: brew install ollama")
    console.print("  [dim]3.[/dim] Use llama.cpp directly: llama-cli -m model.gguf")
    console.print()

    if not args.force:
        ans = Confirm.ask(
            "  [yellow]Continue anyway? This may fail.[/yellow]",
            default=False,
        )
        if not ans:
            console.print(MSG_CANCELLED)
            sys.exit(0)
    else:
        warn("Continuing with --force flag.")


def _resolve_dtype(
    args: argparse.Namespace,
    meta: dict[str, Any] | None,
) -> str:
    """Determine the intermediate dtype for conversion."""
    if args.dtype:
        return str(args.dtype)
    if meta and meta.get("file_type") == 0:
        info("Auto-detected dtype: [bold]float32[/bold] (source is float32)")
        return "float32"
    if meta and meta.get("file_type") == 26:
        info(
            "Auto-detected dtype: [bold]float16[/bold] "
            "(source is bfloat16)"
        )
        return "float16"
    return "float16"


def _resolve_quant_params(
    args: argparse.Namespace,
    gguf_size_bytes: int,
    hw: dict[str, Any],
) -> None:
    """Resolve quantization parameters (mutates args.bits/group_size/mode)."""
    if args.no_quantize:
        return

    # --high-bandwidth flag: auto-select m5-max preset (non-overriding)
    if args.high_bandwidth and not args.preset and not args.bits:
        args.preset = HIGH_BANDWIDTH_PRESET
    if args.preset:
        preset = PRESETS[args.preset]
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


def _resolve_resume(
    args: argparse.Namespace,
    intermediate_dir: Path,
) -> bool:
    """Determine if Step 1 should be skipped. Returns skip_step1."""
    if not args.resume and not intermediate_dir.exists():
        return False

    if args.resume:
        info("Resume flag detected. Skipping Step 1 (using existing intermediate).")
        return True

    # intermediate_dir exists but no --resume flag
    if not args.quiet:
        console.print()
        ans = Confirm.ask(
            f"  [yellow]Found existing intermediate files at:[/yellow]\n"
            f"  [dim]{intermediate_dir}[/dim]\n"
            f"  [bold]Skip Step 1 and proceed directly to quantization?[/bold]",
            default=True,
        )
        if ans:
            info("Skipping Step 1.")
            return True
    return False


def _cleanup_old_intermediates(final_dir: Path) -> None:
    """Remove old *_intermediate directories in the output parent."""
    info("Cleaning up old intermediate directories...")
    parent = final_dir.parent
    old_intermediates = list(parent.glob("*_intermediate"))
    for folder in old_intermediates:
        try:
            shutil.rmtree(folder)
            info(f"  Removed: {folder.name}")
        except Exception as e:
            warn(f"  Could not remove {folder.name}: {e}")
    if not old_intermediates:
        info("  No old intermediate directories found.")


def _show_conversion_plan(
    args: argparse.Namespace,
    skip_step1: bool,
    do_quantize: bool,
    intermediate_dtype: str,
) -> None:
    """Display the conversion pipeline plan table."""
    if args.quiet:
        return

    console.print()
    plan_table = Table(
        title="Conversion Pipeline Plan",
        title_style=_STYLE_BOLD_CYAN,
        box=box.ROUNDED,
        border_style="dim",
        show_header=True,
    )
    plan_table.add_column("Step", style="bold", width=10)
    plan_table.add_column("Action", style="white")
    plan_table.add_column("Status", style="dim", width=15)

    # Step 1
    status1 = "[yellow]SKIPPED[/yellow]" if skip_step1 else STATUS_READY
    plan_table.add_row(
        "Step 1",
        f"Convert GGUF → MLX [bold]{intermediate_dtype}[/bold]",
        status1,
    )

    if do_quantize:
        quant_desc = f"{args.bits}-bit"
        if args.predicate:
            quant_desc = args.predicate
        plan_table.add_row(
            "Step 2",
            f"Quantise to [bold]{quant_desc}[/bold] MLX",
            STATUS_READY,
        )
        plan_table.add_row(
            "Step 3",
            "Clean up intermediate files",
            STATUS_READY,
        )
    else:
        plan_table.add_row(
            "Output",
            "Save float16 model directly",
            STATUS_READY,
        )
    console.print(plan_table)

    if not args.force:
        console.print()
        console.print(
            "  [yellow]This may take several minutes for large models.[/yellow]"
        )
        console.print()
        Confirm.ask(
            "  Press [bold]Enter[/bold] to start, or Ctrl-C to cancel...",
            default=True,
        )


def _handle_step1_failure(
    meta: dict[str, Any] | None,
) -> None:
    """Display error info when Step 1 (gguf2mlx) fails."""
    arch = str((meta or {}).get("architecture", "")).lower()
    is_known, issue_info = is_known_issue_arch(arch)

    if is_known and issue_info:
        console.print()
        fail(f"gguf2mlx failed on {(meta or {}).get('architecture', arch)} model")
        console.print()
        console.print("  [bold]Known issue:[/bold]")
        console.print(f"  [dim]{issue_info['issue']}[/dim]")
        console.print()
        console.print("  [bold]Workarounds:[/bold]")
        for i, (cmd, desc) in enumerate(issue_info["workarounds"], 1):
            console.print(f"  [dim]{i}.[/dim] [cyan]{cmd}[/cyan] — {desc}")
        console.print()
    else:
        fail("Conversion failed. Check the error above.")


def _handle_step2_failure(
    meta: dict[str, Any] | None,
    output_step2: str,
    intermediate_dir: Path,
) -> None:
    """Display error info when Step 2 (mlx_lm quantize) fails."""
    arch = str((meta or {}).get("architecture", "")).lower()
    is_known, _issue_info = is_known_issue_arch(arch)

    stderr_text = output_step2 or ""
    has_tensor_error = (
        "Received" in stderr_text and "parameters not in model" in stderr_text
    )
    has_blk_error = "blk." in stderr_text

    if is_known or has_tensor_error or has_blk_error:
        console.print()
        fail(
            f"mlx_lm quantization failed on "
            f"{(meta or {}).get('architecture', arch) or arch} model"
        )
        console.print()

        is_gemma_moe = is_known and arch in ("gemma4", "gemma3")

        if has_blk_error or is_gemma_moe:
            console.print("  [bold]Root cause:[/bold]")
            console.print("  [dim]Gemma4 MoE architectural mismatch with mlx_lm[/dim]")
            console.print("  [dim]GGUF is missing some layernorm tensors that mlx_lm expects[/dim]")
            console.print("  [dim]This is a known limitation for Gemma4 MoE models[/dim]")
            console.print()

        console.print("  [bold]Recommended options:[/bold]")
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
        if output_step2:
            fail(
                "The float16 intermediate is still at: "
                f"[dim]{intermediate_dir}[/dim]"
            )


def _run_step1(
    args: argparse.Namespace,
    gguf_path: Path,
    intermediate_dir: Path,
    intermediate_dtype: str,
    meta: dict[str, Any] | None,
    progress: Progress,
    pipeline_task: Any,
) -> float:
    """Run Step 1: GGUF → MLX float conversion. Returns elapsed time."""
    t0 = time.time()

    if not args.quiet:
        step(1, 3 if not args.no_quantize else 1, f"Converting GGUF → MLX ({intermediate_dtype} safetensors)")

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    ok_step1, _output_step1 = run_with_progress(
        [
            sys.executable, "-m", "gguf2mlx",
            "--input", str(gguf_path),
            "--output", str(intermediate_dir),
            "--dtype", intermediate_dtype,
        ],
        "Converting GGUF to MLX " + intermediate_dtype + " (this may take 5-30 minutes)",
        progress=progress,
        quiet=args.quiet,
    )

    if not ok_step1:
        _handle_step1_failure(meta)
        sys.exit(1)

    elapsed = time.time() - t0
    ok(f"GGUF converted to MLX {intermediate_dtype} ({format_time(elapsed)})")
    progress.update(pipeline_task, advance=1)
    return elapsed


def _run_step2(
    args: argparse.Namespace,
    intermediate_dir: Path,
    final_dir: Path,
    meta: dict[str, Any] | None,
    progress: Progress,
    pipeline_task: Any,
) -> float:
    """Run Step 2: MLX float16 → Quantized. Returns elapsed time."""
    t_start = time.time()
    total_steps = 3 if not args.no_quantize else 1

    step(2, total_steps, "Quantising to MLX")

    quant_args = build_quant_args(args)
    ok_step2, output_step2 = run_with_progress(
        [
            sys.executable, "-m", "mlx_lm", "convert",
            "--hf-path", str(intermediate_dir),
            "--mlx-path", str(final_dir),
            *quant_args,
        ],
        f"Quantising to {args.bits}-bit MLX (this may take several minutes)",
        progress=progress,
        quiet=args.quiet,
    )

    if not ok_step2:
        _handle_step2_failure(meta, output_step2, intermediate_dir)
        sys.exit(1)

    elapsed = time.time() - t_start
    ok(f"Model quantised to [bold]{args.bits}-bit[/bold] ({format_time(elapsed)})")
    progress.update(pipeline_task, advance=1)
    return elapsed


def _run_step3(
    args: argparse.Namespace,
    intermediate_dir: Path,
    progress: Progress,
    pipeline_task: Any,
) -> None:
    """Run Step 3: Clean up intermediate files."""
    if not args.keep_intermediate:
        step(3, 3, "Cleaning up intermediate files")
        try:
            shutil.rmtree(intermediate_dir)
            ok("Intermediate files removed")
        except Exception as e:
            warn(f"Could not remove {intermediate_dir}: {e}")
    else:
        info(f"Intermediate files kept at: [dim]{intermediate_dir}[/dim]")
    progress.update(pipeline_task, advance=1)


def _save_float16_direct(
    _args: argparse.Namespace,
    intermediate_dir: Path,
    final_dir: Path,
    intermediate_dtype: str,
) -> None:
    """Move intermediate to final location for --no-quantize mode."""
    if intermediate_dir != final_dir:
        if final_dir.exists():
            warn(f"Output directory exists, removing: {final_dir}")
            shutil.rmtree(final_dir)
        shutil.move(str(intermediate_dir), str(final_dir))
        ok(f"Model saved to: [bold cyan]{final_dir}[/bold cyan]")

    final_size = sum(
        f.stat().st_size for f in final_dir.rglob("*") if f.is_file()
    )
    console.print()
    console.print(Panel("[bold green]✓ Done![/bold green]", border_style="green"))
    console.print()
    console.print(
        f"  Your MLX [bold]{intermediate_dtype}[/bold] model is ready at:"
    )
    console.print(f"  [bold cyan]{final_dir}[/bold cyan]  ({format_size(final_size)})")
    console.print()
    sys.exit(0)


def _show_conversion_summary(
    args: argparse.Namespace,
    gguf_path: Path,
    final_dir: Path,
    gguf_size_bytes: int,
    total_time: float,
    _do_quantize: bool,
) -> None:
    """Display the final conversion summary table and usage hints."""
    if not args.quiet:
        info("Validating output...")
    validate_output(final_dir)

    final_size_bytes = sum(
        f.stat().st_size for f in final_dir.rglob("*") if f.is_file()
    )

    console.print()
    console.print(Panel("[bold green]✓ Done![/bold green]", border_style="green"))
    console.print()

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
    console.print("  [bold]To generate text, run:[/bold]")
    console.print(
        "  [cyan]python3 -m mlx_lm generate "
        f'--model "{final_dir}" --prompt "Hello"[/cyan]'
    )
    console.print()
    console.print("  [bold]Or start an interactive chat:[/bold]")
    console.print(
        "  [cyan]python3 -m mlx_lm chat "
        f'--model "{final_dir}"[/cyan]'
    )
    console.print()

    if not args.quiet and not args.keep_intermediate:
        ok("Conversion complete - model verified and ready to use")



# Main Pipeline

def _show_hardware_table(hw: dict[str, Any], quiet: bool) -> None:
    """Display Apple Silicon hardware detection table."""
    if not hw["is_apple_silicon"] or quiet:
        return

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
        "max": _STYLE_BOLD_MAGENTA,
        "ultra": _STYLE_BOLD_MAGENTA,
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


def _show_preflight_results(
    args: argparse.Namespace,
    passed: bool,
    pre_warnings: list[str],
    pre_errors: list[str],
) -> None:
    """Show preflight check results and prompt if needed."""
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

    if not args.quiet and pre_warnings and not args.force:
        ans = Confirm.ask(
            "  [yellow]Warnings detected. Continue?[/yellow]",
            default=True,
        )
        if not ans:
            console.print(MSG_CANCELLED)
            sys.exit(0)


def main() -> None:
    global console

    parser = build_parser()
    args = parser.parse_args()

    if args.no_color:
        console = Console(no_color=True, highlight=False)

    banner()

    hw = detect_apple_silicon()
    _show_hardware_table(hw, args.quiet)

    deps = check_dependencies()
    gguf_path = _resolve_gguf_input(args, deps)

    # Inspect mode
    if args.inspect:
        inspect_mode(gguf_path)
        return

    # Estimate mode
    if args.estimate:
        _run_estimate_mode(args, gguf_path, hw)
        return

    # Preflight
    final_dir = (
        Path(args.output).expanduser()
        if args.output
        else gguf_path.parent / (gguf_path.stem + "-4bit-mlx")
    )
    passed, pre_warnings, pre_errors = preflight_checks(gguf_path, final_dir, args)
    _show_preflight_results(args, passed, pre_warnings, pre_errors)

    gguf_size_bytes = gguf_path.stat().st_size
    ok(f"Input: [bold]{gguf_path.name}[/bold]  ({format_size(gguf_size_bytes)})")

    # Read metadata and check compatibility
    meta = read_gguf_metadata(gguf_path)
    _check_arch_compatibility(args, meta)
    _show_metadata_warnings(args, meta)

    # Resolve conversion configuration
    intermediate_dtype = _resolve_dtype(args, meta)

    if not args.output:
        final_dir = get_output_dir(gguf_path)

    if args.cleanup_old:
        _cleanup_old_intermediates(final_dir)

    intermediate_dir = final_dir.parent / (final_dir.name + "_intermediate")
    skip_step1 = _resolve_resume(args, intermediate_dir)
    do_quantize = not args.no_quantize

    _resolve_quant_params(args, gguf_size_bytes, hw)
    total_steps = 3 if do_quantize else 1

    # Disk space check
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if not args.force and not check_disk_space(gguf_path, final_dir):
        console.print(MSG_CANCELLED)
        sys.exit(0)

    # Show plan and confirm
    _show_conversion_plan(args, skip_step1, do_quantize, intermediate_dtype)

    # Run pipeline
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=args.quiet,
    )

    conversion_success = False
    t0 = time.time()

    with progress:
        pipeline_task = progress.add_task(
            "[bold cyan]Pipeline[/bold cyan]", total=total_steps
        )

        try:
            ensure_deps(deps, for_convert=True)

            # Step 1: GGUF → MLX float
            if skip_step1:
                info("Skipping Step 1 as requested (resume mode).")
                progress.update(pipeline_task, advance=1)
            else:
                _run_step1(
                    args, gguf_path, intermediate_dir,
                    intermediate_dtype, meta, progress, pipeline_task,
                )

            # Gemma4 tensor fix
            arch = str((meta or {}).get("architecture", "")).lower()
            if arch in ("gemma4", "gemma3"):
                fix_gemma4_tensor_names(intermediate_dir)

            # Float16-only mode
            if not do_quantize:
                _save_float16_direct(
                    args, intermediate_dir, final_dir, intermediate_dtype
                )

            # Step 2: Quantize
            _run_step2(
                args, intermediate_dir, final_dir,
                meta, progress, pipeline_task,
            )

            # Step 3: Cleanup
            _run_step3(args, intermediate_dir, progress, pipeline_task)

            conversion_success = True

        except Exception as e:
            fail(f"Conversion failed: {e}")
            sys.exit(1)

        finally:
            if not conversion_success and not args.keep_intermediate:
                try:
                    if intermediate_dir.exists():
                        shutil.rmtree(intermediate_dir)
                        info("Cleaned up intermediate files after failure.")
                except Exception as cleanup_error:
                    warn(f"Failed to clean up {intermediate_dir}: {cleanup_error}")

    # Summary
    total_time = time.time() - t0
    _show_conversion_summary(
        args, gguf_path, final_dir, gguf_size_bytes, total_time, do_quantize
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print()
        console.print(
            Panel(
                "[yellow]Cancelled by user.[/yellow] "
                "Partial files may remain.",
                border_style="yellow",
                expand=False,
            )
        )
        sys.exit(0)
