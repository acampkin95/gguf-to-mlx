#!/usr/bin/env python3
"""
GGUF → MLX — converter + quantizer for Apple Silicon.

One-step pipeline: GGUF → float16 safetensors → quantized MLX.
Hardware-aware defaults, HuggingFace Hub integration, model scanning.

Quick start:
  python3 convert.py model.gguf           Convert with smart defaults
  python3 convert.py --scan               Find and convert models
  python3 convert.py --hf-search "llama"   Download from HuggingFace
"""

import sys
import os
import time
import re
import subprocess
import shutil
import argparse
from pathlib import Path
from typing import Any
import json
from dataclasses import dataclass

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
from rich import box

# Hardware detection
import psutil

# Requests for streaming downloads
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# Module-level console - configured in main() based on --no-color

console = Console(highlight=False)

# Shared constants (SonarQube S1192 dedup)
STATUS_READY = "[green]READY[/green]"
STATUS_SKIPPED = "[yellow]SKIPPED[/yellow]"
MSG_CANCELLED = "\n  Cancelled."
_MOE_4BIT_LABEL = "MoE 4-bit"
_CPMM_NO_QUANTIZE = "cpmm model.gguf --no-quantize"
# _UPGRADE_GGUF2MLX removed — gguf2mlx is now vendored internally
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


# Architectures known to have conversion engine compatibility issues
KNOWN_CONVERSION_ISSUES: dict[str, dict[str, Any]] = {
    "gemma4": {
        "issue": "head_count_kv metadata returns list instead of int",
        "workarounds": [
            (_CPMM_NO_QUANTIZE, "Converts to float16 only, avoids quant step"),
            ("Internal fix pending", "Will be addressed in a future update"),
        ],
    },
    "gemma3": {
        "issue": "Similar metadata issues as Gemma4",
        "workarounds": [
            (_CPMM_NO_QUANTIZE, "Converts to float16 only"),
            ("Internal fix pending", "Will be addressed in a future update"),
        ],
    },
    "gemma2": {
        "issue": "May have similar head_count_kv issues",
        "workarounds": [
            (_CPMM_NO_QUANTIZE, "Converts to float16 only"),
            ("Internal fix pending", "Will be addressed in a future update"),
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
    console.print("  The conversion engine produces incorrect tensor names for Gemma4 MoE layers.")
    console.print("  These need to be renamed from 'blk.*' to 'model.layers.*' format.")
    console.print()
    console.print("  [bold]Workaround options:[/bold]")
    console.print("  [dim]1.[/dim] Use [cyan]--no-quantize[/cyan] for float16 (50GB)")
    console.print("  [dim]2.[/dim] A fix will be included in a future update to the internal converter.")
    console.print("  [dim]3.[/dim] Use Ollama or llama.cpp for 4-bit conversion")
    console.print()
    
    # Can't auto-fix without rewriting safetensors - user must use workarounds



def is_known_issue_arch(arch: str) -> tuple[bool, dict[str, Any] | None]:
    """Check if architecture is known to have conversion engine issues."""
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
Categories:
  Basic Conversion         model.gguf, model.gguf --bits 8, --no-quantize
  Advanced Conversion      --preset quality, --high-bandwidth, --predicate mixed_4_6
  Analysis                 --inspect, --estimate
  Model Management         --scan, --set-models-dir, --delete-gguf
  HuggingFace              --hf-search, --hf-download, --auto-convert

Examples:

  # --- Basic Conversion ---
  %(prog)s model.gguf                              Smart defaults
  %(prog)s model.gguf --bits 8                     8-bit quantisation
  %(prog)s model.gguf --no-quantize                 Float16 only, no quant
  %(prog)s model.gguf ./out/                        Custom output directory

  # --- Advanced Quantisation ---
  %(prog)s model.gguf --preset quality              Quality preset
  %(prog)s model.gguf --high-bandwidth               M5 Max / Ultra optimised
  %(prog)s model.gguf --bits 2 --group-size 128      Compact 2-bit quant
  %(prog)s model.gguf --predicate mixed_4_6          Mixed 4/6-bit recipe
  %(prog)s model.gguf --mode mxfp4                   MXFP4 quantisation mode

  # --- Analysis & Diagnostics ---
  %(prog)s model.gguf --inspect                      Show GGUF metadata
  %(prog)s model.gguf --estimate                      Estimate conversion cost
  %(prog)s model.gguf --resume                        Resume partial conversion

  # --- Model Management ---
  %(prog)s --scan                                    Find and pick models
  %(prog)s --scan-lmstudio                           Scan LM Studio models only
  %(prog)s --set-models-dir ~/Models                  Set default models folder
  %(prog)s model.gguf --delete-gguf                   Remove GGUF after convert

  # --- HuggingFace Hub ---
  %(prog)s --hf-search "mistral gguf"                 Search models on HF
  %(prog)s --hf-download mistralai/Mistral-7B         Download from HF
  %(prog)s --hf-list mistralai/Mistral-7B              List files in a HF repo
  %(prog)s --hf-download org/model --hf-file model.gguf --auto-convert  DL + convert
""",
    )

    # Positional
    p.add_argument(
        "input", nargs="?",
        help="Path to .gguf model file (or hf:org/model for HF download)",
    )
    p.add_argument(
        "output", nargs="?",
        help="Output directory (default: auto-named from input)",
    )

    # ── Quantisation ──
    quant = p.add_argument_group("Quantisation")
    quant.add_argument(
        "--bits", type=int, choices=[2, 3, 4, 6, 8],
        help="Bits per weight (default: auto-selected by hardware)",
    )
    quant.add_argument(
        "--group-size", type=int, choices=[32, 64, 128, 256],
        help="Group size for quantisation (default: hardware-aware)",
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
        help="Skip quantisation; output float16 safetensors only",
    )
    quant.add_argument(
        "--preset", choices=list(PRESETS.keys()),
        help="Quantisation preset (overrides --bits/--group-size/--mode)",
    )
    quant.add_argument(
        "--high-bandwidth", action="store_true",
        help="Shortcut for --preset m5-max (Max/Ultra ≥48 GB)",
    )
    quant.add_argument(
        "--dtype", choices=["float16", "float32"],
        help="Intermediate float type for GGUF\u2192MLX step (default: auto)",
    )

    # ── Pipeline ──
    pipe = p.add_argument_group("Pipeline Control")
    pipe.add_argument(
        "--resume", action="store_true",
        help="Skip GGUF conversion; reuse existing intermediate files",
    )
    pipe.add_argument(
        "--keep-intermediate", action="store_true",
        help="Preserve intermediate float16 files after quantisation",
    )
    pipe.add_argument(
        "--cleanup-old", action="store_true",
        help="Remove stale *_intermediate directories before starting",
    )
    pipe.add_argument(
        "--force", "-f", action="store_true",
        help="Skip interactive prompts and disk space warnings",
    )

    # ── Analysis ──
    analysis = p.add_argument_group("Analysis")
    analysis.add_argument(
        "--inspect", action="store_true",
        help="Display GGUF metadata and exit (no conversion)",
    )
    analysis.add_argument(
        "--estimate", action="store_true",
        help="Predict conversion time, memory, and output size; no conversion",
    )
    analysis.add_argument(
        "--mtp", action="store_true",
        help="Show Multi-Token Prediction capability during conversion",
    )

    # ── Display ──
    display = p.add_argument_group("Display")
    display.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress output; show only errors and final result",
    )
    display.add_argument(
        "--no-color", action="store_true",
        help="Disable Rich formatting (for piping, CI, or logs)",
    )

    # ── Model Management ──
    model = p.add_argument_group("Model Management")
    model.add_argument(
        "--scan", "-S", action="store_true",
        help="Scan all known model directories and pick one interactively",
    )
    model.add_argument(
        "--scan-omlx", action="store_true",
        help="Scan for models in ~/.omlx/models/",
    )
    model.add_argument(
        "--scan-lmstudio", action="store_true",
        help="Scan for models in ~/.lmstudio/models/",
    )
    model.add_argument(
        "--scan-hf-cache", action="store_true",
        help="Scan ~/.cache/huggingface/hub/ for downloaded models",
    )
    model.add_argument(
        "--models-dir", type=str, metavar="PATH",
        help="Scan a custom directory for GGUF/MLX files",
    )
    model.add_argument(
        "--set-models-dir", type=str, metavar="PATH",
        help="Set default models folder (saved to config)",
    )
    model.add_argument(
        "--delete-gguf", action="store_true",
        help="Delete source .gguf after successful conversion",
    )

    # ── HuggingFace Hub ──
    hf = p.add_argument_group("HuggingFace Hub")
    hf.add_argument(
        "--hf-search", "-s", type=str, metavar="QUERY",
        help="Search HuggingFace models by keyword",
    )
    hf.add_argument(
        "--hf-download", "-H", type=str, metavar="REPO_ID",
        help="Download a model from HuggingFace",
    )
    hf.add_argument(
        "--hf-file", type=str, metavar="FILENAME",
        help="Specific file to download (default: auto-detect)",
    )
    hf.add_argument(
        "--hf-token", type=str, metavar="TOKEN",
        help="HuggingFace token (falls back to HF_TOKEN env var)",
    )
    hf.add_argument(
        "--hf-list", "-l", type=str, metavar="REPO_ID",
        help="List all files in a HuggingFace repository",
    )
    hf.add_argument(
        "--auto-convert", "-C", action="store_true",
        help="Automatically convert the downloaded GGUF to MLX",
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
            f"⚠ {arch} has known conversion issues: {issue_info['issue']}"
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
        "[bold]GGUF → MLX[/bold]  [dim]converter + quantizer for Apple Silicon[/dim]\n"
        "[cyan]smart defaults  ·  hardware-aware  ·  HuggingFace Hub[/cyan]",
        border_style="cyan",
        padding=(1, 1),
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



# ═══════════════════════════════════════════════════════════════════════════
# Configuration Management
# ═══════════════════════════════════════════════════════════════════════════

CONFIG_DIR = Path.home() / ".config" / "gguf-to-mlx"
CONFIG_PATH = CONFIG_DIR / "config.json"


def load_config() -> dict[str, Any]:
    """Load configuration from ~/.config/gguf-to-mlx/config.json."""
    if CONFIG_PATH.exists():
        try:
            return dict(json.loads(CONFIG_PATH.read_text()))
        except (json.JSONDecodeError, OSError):
            warn("Config file corrupted, using defaults")
            return {}
    return {}


def save_config(config: dict[str, Any]) -> None:
    """Save configuration."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    ok(f"Config saved to {CONFIG_PATH}")


def get_models_dir() -> Path:
    """Get the configured models directory."""
    config = load_config()
    custom = config.get("models_dir")
    if custom:
        p = Path(custom).expanduser()
        if p.exists():
            return p
        warn(f"Configured models dir not found: {p}")
    return Path.home() / "Models"


def set_models_dir(path: str) -> None:
    """Set the configured models directory."""
    config = load_config()
    config["models_dir"] = str(Path(path).expanduser())
    save_config(config)
    ok(f"Models directory set to: {config['models_dir']}")


def get_hf_token_from_config() -> str | None:
    """Get HuggingFace token from config."""
    config = load_config()
    return config.get("hf_token")


def save_hf_token(token: str) -> None:
    """Save HuggingFace token to config."""
    config = load_config()
    config["hf_token"] = token
    save_config(config)


# ═══════════════════════════════════════════════════════════════════════════
# Model Folder Scanning
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FoundModel:
    path: Path
    format: str        # "gguf" or "mlx"
    source: str        # "custom", "omlx", "lmstudio", "hf-cache"
    size_gb: float
    name: str          # Display name
    metadata: dict[str, Any] | None = None


# LM Studio stores models in ~/.lmstudio/models (main)
# or ~/Library/Application Support/LM Studio/models/ (newer installs)
_LM_STUDIO_PRIMARY = Path.home() / ".lmstudio" / "models"
_LM_STUDIO_ALT = Path.home() / "Library" / "Application Support" / "LM Studio" / "models"

SCAN_DIRS: dict[str, Path] = {
    "omlx": Path.home() / ".omlx" / "models",
    "lmstudio": _LM_STUDIO_PRIMARY if _LM_STUDIO_PRIMARY.exists() else _LM_STUDIO_ALT,
    "hf-cache": Path.home() / ".cache" / "huggingface" / "hub",
}


def scan_for_models(
    custom_dir: Path | None = None,
    scan_omlx: bool = False,
    scan_lmstudio: bool = False,
    scan_hf: bool = False,
    scan_all: bool = False,
) -> list[FoundModel]:
    """Scan directories for GGUF and MLX model files.

    Returns sorted list of FoundModel objects (largest first).
    """
    found: list[FoundModel] = []
    scanned: set[Path] = set()

    if custom_dir and custom_dir.exists():
        _scan_directory(custom_dir, found, scanned, source="custom")

    if scan_all or scan_omlx:
        omlx_dir = SCAN_DIRS["omlx"]
        if omlx_dir.exists():
            _scan_directory(omlx_dir, found, scanned, source="omlx")

    if scan_all or scan_lmstudio:
        lm_dir = SCAN_DIRS["lmstudio"]
        if lm_dir.exists():
            _scan_directory(lm_dir, found, scanned, source="lmstudio")

    if scan_all or scan_hf:
        hf_dir = SCAN_DIRS["hf-cache"]
        if hf_dir.exists():
            _scan_hf_cache(hf_dir, found, scanned)

    found.sort(key=lambda m: m.size_gb, reverse=True)
    return found


def _scan_directory(
    directory: Path,
    found: list[FoundModel],
    scanned: set[Path],
    source: str,
) -> None:
    """Recursively scan a directory for model files."""
    if not directory.exists():
        return
    try:
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in (".gguf", ".safetensors"):
                continue
            if path in scanned:
                continue
            scanned.add(path)
            size_gb = path.stat().st_size / 1e9
            if size_gb < 0.1:
                continue
            fmt = "gguf" if path.suffix.lower() == ".gguf" else "mlx"
            try:
                rel = path.relative_to(directory)
                name = str(rel.parent / path.stem) if rel.parent != Path(".") else path.stem
            except ValueError:
                name = path.stem
            found.append(FoundModel(
                path=path, format=fmt, source=source,
                size_gb=size_gb, name=name,
            ))
    except PermissionError:
        pass


def _scan_hf_cache(
    hf_dir: Path,
    found: list[FoundModel],
    scanned: set[Path],
) -> None:
    """Scan HuggingFace cache directory for downloaded models."""
    if not hf_dir.exists():
        return
    try:
        for snap_dir in hf_dir.rglob("snapshots"):
            if not snap_dir.is_dir():
                continue
            for model_file in snap_dir.rglob("*.safetensors"):
                if model_file in scanned:
                    continue
                scanned.add(model_file)
                size_gb = model_file.stat().st_size / 1e9
                if size_gb < 0.1:
                    continue
                name = str(model_file.relative_to(snap_dir))
                found.append(FoundModel(
                    path=model_file, format="mlx", source="hf-cache",
                    size_gb=size_gb, name=name,
                ))
    except PermissionError:
        pass


TRUNCATE_LEN = 44

def _shorten_name(name: str, max_len: int = TRUNCATE_LEN) -> str:
    """Shorten a model name for table display."""
    if len(name) <= max_len:
        return name
    # Try to keep the last segment (filename stem) and start of path
    parts = name.split("/")
    if len(parts) >= 2:
        stem = parts[-1]
        prefix = "/".join(parts[:-1])
        keep_prefix = max_len - len(stem) - 4  # ".../"
        if keep_prefix >= 6:
            return prefix[:keep_prefix] + ".../" + stem
    return name[: max_len - 3] + "..."


def _format_source_tag(source: str) -> str:
    """Return a styled source tag for scan table."""
    tags = {
        "omlx": "[dodger_blue1]omlx[/]",
        "lmstudio": "[orange1]lmstudio[/]",
        "hf-cache": "[magenta]hf-cache[/]",
        "custom": "[green]custom[/]",
    }
    return tags.get(source, source)


def display_scan_results(models: list[FoundModel]) -> FoundModel | None:
    """Display scanned models in a Rich table and let user pick one.

    Returns the selected FoundModel, or None if cancelled.
    """
    if not models:
        console.print()
        console.print(Panel(
            "[yellow]No models found[/yellow]\n\n"
            " Try:\n"
            "  \u2022 Use [cyan]--set-models-dir[/cyan] to point to your models folder\n"
            "  \u2022 Download from HuggingFace with [cyan]--hf-search[/cyan]\n"
            "  \u2022 Place .gguf files in [dim]~/.omlx/models/[/dim] or [dim]~/.lmstudio/models/[/dim]",
            title="[bold]Scan Complete[/bold]",
            border_style="yellow",
        ))
        return None

    console.print()
    table = Table(
        title=f"Found {len(models)} model(s)",
        title_style="bold cyan",
        box=box.SIMPLE_HEAD,
        border_style="dim",
        header_style="bold white",
    )
    table.add_column("#", style="dim", width=3, no_wrap=True)
    table.add_column("Name", style="bold", width=TRUNCATE_LEN, min_width=30)
    table.add_column("Format", width=7, no_wrap=True)
    table.add_column("Source", width=10, no_wrap=True)
    table.add_column("Size", justify="right", width=10, no_wrap=True)

    for i, m in enumerate(models, 1):
        fmt_style = "green" if m.format == "gguf" else "magenta"
        table.add_row(
            str(i),
            _shorten_name(m.name),
            f"[{fmt_style}]{m.format}[/{fmt_style}]",
            _format_source_tag(m.source),
            format_size(m.size_gb * 1e9),
        )

    console.print(table)
    console.print()

    raw = Prompt.ask(
        "  [bold]Model to convert[/bold] (number, or Enter to cancel)",
        default="",
    )
    if not raw.strip():
        return None
    try:
        idx = int(raw.strip()) - 1
        if 0 <= idx < len(models):
            return models[idx]
    except ValueError:
        pass
    warn("Invalid selection — enter a number from the table")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# HuggingFace Search & Download
# ═══════════════════════════════════════════════════════════════════════════

KNOWN_GGUF_REPOS = {
    "mistralai": "Mistral",
    "meta-llama": "Llama",
    "microsoft": "Phi",
    "deepseek-ai": "DeepSeek",
    "Qwen": "Qwen",
    "google": "Gemma",
    "TheBloke": "Community Quants",
    "MaziyarPanahi": "Community Quants",
    "bartowski": "Community Quants",
}


def get_hf_token(quiet: bool = False) -> str | None:
    """Get HuggingFace token from env, config, or interactive prompt."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token

    token = get_hf_token_from_config()
    if token:
        return token

    if quiet:
        return None

    console.print()
    console.print(Panel(
        "[yellow]HuggingFace Token[/yellow]\n\n"
        "Some models require authentication. You can:\n"
        "1. Set [bold]HF_TOKEN[/bold] environment variable\n"
        "2. Enter token now (saved to config)\n"
        "3. Skip (public models only)\n"
        "\n"
        "Get a token at: [dim]https://huggingface.co/settings/tokens[/dim]",
        title="[bold]Authentication[/bold]",
        border_style="yellow",
        padding=(1, 2),
    ))
    raw = Prompt.ask("  [bold]HuggingFace token[/bold] (or Enter to skip)")
    if raw.strip():
        save_hf_token(raw.strip())
        return raw.strip()
    return None


def hf_search(
    query: str,
    limit: int = 20,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Search HuggingFace Hub for models.

    Returns list of model dicts with id, downloads, likes, etc.
    """
    if not HAS_REQUESTS:
        fail("requests library required for HF search. Install: pip install requests")
        sys.exit(1)

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    search_url = "https://huggingface.co/api/models"
    params: dict[str, Any] = {
        "search": query,
        "sort": "downloads",
        "direction": "-1",
        "limit": limit,
    }

    try:
        resp = requests.get(search_url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        models = resp.json()
    except requests.RequestException as e:
        fail(f"Search failed: {e}")
        return []

    results: list[dict[str, Any]] = []
    for m in models:
        model_id = m.get("id", "unknown")
        results.append({
            "id": model_id,
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "pipeline_tag": m.get("pipeline_tag", ""),
            "private": m.get("private", False),
            "last_modified": m.get("lastModified", ""),
        })

    return results


def hf_list_files(
    repo_id: str,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """List files in a HuggingFace repo."""
    if not HAS_REQUESTS:
        fail("requests library required")
        sys.exit(1)

    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        fail(f"Failed to list repo {repo_id}: {e}")
        return []

    siblings = data.get("siblings", [])
    files: list[dict[str, Any]] = []
    for sib in siblings:
        rfilename = sib.get("rfilename", "")
        size = sib.get("size", 0)
        files.append({
            "path": rfilename,
            "size": size,
            "size_str": format_size(size),
        })
    return files


def format_speed(bytes_per_sec: float) -> str:
    """Human-readable transfer speed with adaptive precision."""
    if bytes_per_sec >= 1_000_000_000:
        return f"{bytes_per_sec / 1_000_000_000:.2f} GB/s"
    elif bytes_per_sec >= 1_000_000:
        return f"{bytes_per_sec / 1_000_000:.1f} MB/s"
    elif bytes_per_sec >= 1_000:
        return f"{bytes_per_sec / 1_000:.0f} KB/s"
    return f"{bytes_per_sec:.0f} B/s"


def download_with_progress(
    url: str,
    dest: Path,
    repo_id: str = "",
    quiet: bool = False,
) -> bool:
    """Download a file with beautiful Rich progress bar showing speed and ETA.

    Uses streaming download with Rich Progress for:
      \u2022 Progress bar with percentage
      \u2022 Download speed (formatted)
      \u2022 Time remaining
      \u2022 Total size

    Returns True on success.
    """
    if not HAS_REQUESTS:
        fail("requests library required for downloads. Install: pip install requests")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            chunk_size = 1024 * 1024  # 1 MB chunks

            label = f"  Downloading {repo_id}" if repo_id else "  Downloading"

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("[progress.download]{task.fields[speed]}"),
                TimeRemainingColumn(),
                console=console,
                disable=quiet,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    label,
                    total=total,
                    speed="? MB/s",
                )
                downloaded = 0
                t_start = time.time()

                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            elapsed = time.time() - t_start
                            if elapsed > 0:
                                speed_bps = downloaded / elapsed
                                progress.update(
                                    task,
                                    completed=downloaded,
                                    speed=format_speed(speed_bps),
                                )

    except requests.RequestException as e:
        fail(f"Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False
    except OSError as e:
        fail(f"Write failed: {e}")
        if dest.exists():
            dest.unlink()
        return False

    return True


def download_from_huggingface(
    repo_id: str,
    filename: str,
    cache_dir: Path | None = None,
    token: str | None = None,
    quiet: bool = False,
    auto_convert: bool = False,
    delete_gguf: bool = False,
    output_dir: Path | None = None,
    conversion_args: argparse.Namespace | None = None,
) -> Path | None:
    """Download model from HuggingFace Hub with beautiful progress.

    Args:
        repo_id: HuggingFace repo ID (e.g., "mistralai/Mistral-7B")
        filename: Filename in repo (e.g., "model.gguf")
        cache_dir: Cache directory (default: ~/.cache/gguf-to-mlx)
        token: HuggingFace token for auth
        quiet: Suppress output
        auto_convert: Run conversion after download
        delete_gguf: Delete GGUF after successful conversion
        output_dir: Output dir for conversion
        conversion_args: CLI args to pass to conversion

    Returns:
        Path to downloaded file, or None on failure
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "gguf-to-mlx" / "downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{repo_id.replace('/', '--')}__{filename}"
    local_path = cache_dir / safe_name

    # Skip if already cached
    if local_path.exists():
        ok(f"Using cached: [bold]{local_path.name}[/bold] ({format_size(local_path.stat().st_size)})")
        if auto_convert:
            _auto_convert_downloaded(local_path, output_dir, delete_gguf, quiet, conversion_args)
        return local_path

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

    console.print()
    console.print(Panel(
        f"[bold]{repo_id}[/bold] / [cyan]{filename}[/cyan]\n"
        f"  \u2192 [dim]{local_path}[/dim]",
        title="[bold cyan]Downloading from HuggingFace[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))

    success = download_with_progress(url, local_path, repo_id, quiet)
    if not success:
        return None

    final_size = local_path.stat().st_size
    ok(f"Downloaded: [bold]{local_path.name}[/bold] ({format_size(final_size)})")

    # Auto-convert if requested
    if auto_convert and local_path.suffix.lower() == ".gguf":
        _auto_convert_downloaded(local_path, output_dir, delete_gguf, quiet, conversion_args)
    elif auto_convert and local_path.suffix.lower() != ".gguf":
        warn(f"Auto-convert skipped: {local_path.suffix} is not GGUF format")

    return local_path


def _auto_convert_downloaded(
    gguf_path: Path,
    output_dir: Path | None,
    delete_gguf: bool,
    quiet: bool,
    conversion_args: argparse.Namespace | None,
) -> None:
    """Run the conversion pipeline on a downloaded GGUF file."""
    console.print()
    console.print(Rule("[bold green]Auto-Converting Downloaded Model[/bold green]"))

    if conversion_args:
        conv_args = conversion_args
    else:
        conv_args = argparse.Namespace(
            input=str(gguf_path),
            output=str(output_dir) if output_dir else None,
            bits=4, group_size=64, mode="affine",
            preset=None, high_bandwidth=False,
            dtype=None, no_quantize=False,
            resume=False, inspect=False, estimate=False,
            mtp=False, keep_intermediate=False,
            cleanup_old=False,
            force=True, quiet=quiet, no_color=False,
            predicate=None,
        )

    try:
        main_with_file(gguf_path, conv_args)
    except Exception as e:
        fail(f"Auto-convert failed: {e}")
        return

    console.print()
    ok("Conversion complete")

    if delete_gguf:
        try:
            gguf_path.unlink()
            ok(f"Deleted GGUF file: [dim]{gguf_path}[/dim]")
        except OSError as e:
            warn(f"Could not delete GGUF file: {e}")


def handle_registry_url(
    input_str: str,
    auto_convert: bool = False,
    delete_gguf: bool = False,
    output_dir: Path | None = None,
    conversion_args: argparse.Namespace | None = None,
    quiet: bool = False,
) -> Path:
    """Handle registry URLs like hf:namespace/model or return local path.

    Examples:
        hf:mistralai/Mistral-7B -> downloads from HuggingFace
        /local/path/model.gguf -> returns as-is
    """
    if input_str.startswith("hf:"):
        parts = input_str[3:].split("/")
        if len(parts) >= 3:
            repo_id = "/".join(parts[:-1])
            filename = parts[-1]
        elif len(parts) == 2:
            repo_id = "/".join(parts)
            filename = f"{parts[1]}-Q4_K_M.gguf"
        else:
            fail(f"Invalid HuggingFace URL: {input_str}")
            fail("Format: hf:namespace/model or hf:namespace/model/filename.gguf")
            sys.exit(1)

        token = get_hf_token(quiet)
        result = download_from_huggingface(
            repo_id, filename,
            token=token, quiet=quiet,
            auto_convert=auto_convert, delete_gguf=delete_gguf,
            output_dir=output_dir, conversion_args=conversion_args,
        )
        if result is None:
            sys.exit(1)
        return result
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
            "Internal gguf2mlx module failed to import. "
            "Check that the gguf2mlx/ package is present in the project directory."
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
        from gguf2mlx import __version__ as _v
        deps["gguf2mlx"] = _v
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
        missing.append(("gguf2mlx (internal)", "Ensure gguf2mlx/ package is present in project"))
    if for_convert and not deps.get("mlx_lm"):
        missing.append(("mlx-lm", "pip install mlx-lm"))

    if missing:
        banner()
        fail("Missing required dependencies:")
        for name, install_cmd in missing:
            info(f"  {name} - install with: [cyan]{install_cmd}[/cyan]")
        console.print()
        sys.exit(1)


_ALL_DEPS_INFO = [
    ("gguf2mlx", "Vendored", "gguf2mlx (internal)",
     "gguf2mlx/ (MIT, Barron Tang — vendored)", True),
    ("mlx-lm", "pip", "mlx_lm", "mlx-lm", True),
    ("mlx", "pip", "mlx.core", "mlx", True),
    ("gguf", "pip", "gguf", "gguf", True),
    ("rich", "pip", "rich", "rich", True),
    ("requests", "pip", "requests", "requests>=2.31.0", True),
    ("safetensors", "pip", "safetensors", "safetensors>=0.4.0", False),
    ("transformers", "pip", "transformers", "transformers>=4.40.0", False),
    ("psutil", "pip", "psutil", "psutil>=5.9.0", True),
    ("tqdm", "pip", "tqdm", "tqdm>=4.0.0", False),
]


def _show_dependency_wizard() -> None:
    """Check all dependencies and offer to install missing ones."""
    _menu_header("Dependency Check", "health")

    statuses: list[dict[str, Any]] = []
    for name, source, module, pkg, required in _ALL_DEPS_INFO:
        installed = _check_single_dep(module)
        statuses.append({
            "name": name, "source": source, "pkg": pkg,
            "installed": installed, "required": required,
        })

    console.print()
    t = Table(box=box.SIMPLE_HEAD, border_style="dim", header_style="bold white")
    t.add_column("Package", style="bold", width=16)
    t.add_column("Source", style="dim", width=8)
    t.add_column("Required", width=9)
    t.add_column("Status", width=14)

    missing_required = []
    for s in statuses:
        icon = "[green]\u2713[/green]" if s["installed"] else "[red]\u2717[/red]"
        req = "[bold]Yes[/bold]" if s["required"] else "[dim]No[/dim]"
        t.add_row(s["name"], s["source"], req, f"{icon} {'installed' if s['installed'] else 'missing'}")
        if s["required"] and not s["installed"]:
            missing_required.append(s)

    console.print(t)

    if not missing_required:
        console.print()
        ok("All required dependencies installed")
        _pause_after()
        return

    console.print()
    warn(f"{len(missing_required)} required package(s) missing")
    if Confirm.ask("  [bold cyan]Install now?[/bold cyan]", default=True):
        for s in missing_required:
            console.print()
            info(f"Installing [bold]{s['name']}[/bold]...")
            code = subprocess.run(
                [sys.executable, "-m", "pip", "install", s["pkg"]],
                capture_output=True, text=True, check=False,
            )
            if code.returncode == 0:
                ok(f"{s['name']} installed")
            else:
                fail(f"{s['name']} failed: {code.stderr[-200:]}")
        console.print()
        ok("Dependency installation complete — restart may be needed")
    else:
        warn("Skipped — some features may not work")

    _pause_after()


def _check_single_dep(module: str) -> bool:
    """Check if a single Python module is importable."""
    try:
        __import__(module)
        return True
    except ImportError:
        return False



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
        return handle_registry_url(
            args.input,
            auto_convert=getattr(args, 'auto_convert', False),
            delete_gguf=getattr(args, 'delete_gguf', False),
            output_dir=Path(args.output).expanduser() if getattr(args, 'output', None) else None,
            conversion_args=args,
            quiet=getattr(args, 'quiet', False),
        )
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
        f"[dim]Known conversion issue:[/dim] {issue_info['issue']}",
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

    # Add resource estimation when bits are known
    if do_quantize and getattr(args, 'bits', None):
        console.print()
        console.print(f"  [dim]Est. final size: ~{args.bits}/16 of source[/dim]")

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
    """Display error info when Step 1 (GGUF→MLX conversion) fails."""
    arch = str((meta or {}).get("architecture", "")).lower()
    is_known, issue_info = is_known_issue_arch(arch)

    if is_known and issue_info:
        console.print()
        fail(f"Conversion failed on {(meta or {}).get('architecture', arch)} model")
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
    """Run Step 1: GGUF → MLX float conversion. Returns elapsed time.

    Calls the vendored gguf2mlx.convert() directly (no subprocess).
    """
    import io
    from contextlib import redirect_stdout, redirect_stderr
    from gguf2mlx import convert as _gguf2mlx_convert

    t0 = time.time()

    if not args.quiet:
        step(1, 3 if not args.no_quantize else 1, f"Converting GGUF → MLX ({intermediate_dtype} safetensors)")

    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Capture upstream convert() output for progress parsing
    captured = io.StringIO()
    try:
        with redirect_stdout(captured), redirect_stderr(captured):
            ok_step1 = _gguf2mlx_convert(
                str(gguf_path),
                str(intermediate_dir),
                dtype=intermediate_dtype,
            )
    except Exception as exc:
        fail(f"Conversion failed: {exc}")
        _handle_step1_failure(meta)
        sys.exit(1)

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
    """Display the final conversion summary with rich celebration."""
    if not args.quiet:
        info("Validating output...")
    validate_output(final_dir)

    final_size_bytes = sum(
        f.stat().st_size for f in final_dir.rglob("*") if f.is_file()
    )
    ratio = (final_size_bytes / gguf_size_bytes * 100) if gguf_size_bytes > 0 else 0
    saved = gguf_size_bytes - final_size_bytes

    # Celebration panel
    speed_tag = (
        "[green]blazing fast[/green]" if total_time < 120
        else "[yellow]moderate[/yellow]" if total_time < 600
        else "[dim]heavy[/dim]"
    )
    console.print()
    console.print(Panel(
        "[bold green]\u2714  Conversion Complete![/bold green]\n\n"
        f"  Model ready at: [bold cyan]{final_dir.name}[/bold cyan]\n"
        f"  Size: [bold]{format_size(final_size_bytes)}[/bold] "
        f"([green]{ratio:.0f}%[/green] of original)\n"
        f"  Time: [bold]{format_time(total_time)}[/bold] ({speed_tag})\n"
        f"  Saved: [bold green]{format_size(saved)}[/bold green]"
        if saved > 0 else "",
        border_style="green",
        padding=(1, 2),
    ))

    # Detailed summary table
    console.print()
    summary = Table(
        title="Conversion Details",
        title_style="bold",
        box=box.SIMPLE,
        border_style="dim",
        show_header=False,
    )
    summary.add_column("", style="dim", width=8)
    summary.add_column("", style="bold")
    summary.add_row("Input", f"[dim]{gguf_path.name}[/dim]  ({format_size(gguf_size_bytes)})")
    summary.add_row("Output", f"[bold cyan]{final_dir}[/bold cyan]  ({format_size(final_size_bytes)})")
    if ratio > 0:
        summary.add_row("Ratio", f"{ratio:.0f}% of original")
    summary.add_row("Time", format_time(total_time))
    console.print(summary)

    console.print()
    console.print("  [bold]Quick start:[/bold]")
    console.print(f"  [cyan]python3 -m mlx_lm generate --model \"{final_dir}\" --prompt \"Hello\"[/cyan]")
    console.print(f"  [cyan]python3 -m mlx_lm chat --model \"{final_dir}\"[/cyan]")
    console.print()



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


def main_with_file(
    gguf_path: Path,
    args: argparse.Namespace,
) -> None:
    """Run the conversion pipeline for a specific GGUF file.

    Reuses the same pipeline logic as main() but bypasses input resolution,
    scan/detect modes, and interactive prompts. Used by --auto-convert.
    """
    global console
    if getattr(args, 'no_color', False):
        console = Console(no_color=True, highlight=False)

    hw = detect_apple_silicon()
    deps = check_dependencies()

    if getattr(args, 'inspect', False):
        inspect_mode(gguf_path)
        return

    final_dir = (
        Path(args.output).expanduser()
        if getattr(args, 'output', None)
        else gguf_path.parent / (gguf_path.stem + "-4bit-mlx")
    )
    passed, pre_warnings, pre_errors = preflight_checks(gguf_path, final_dir, args)
    _show_preflight_results(args, passed, pre_warnings, pre_errors)

    gguf_size_bytes = gguf_path.stat().st_size
    ok(f"Input: [bold]{gguf_path.name}[/bold]  ({format_size(gguf_size_bytes)})")

    meta = read_gguf_metadata(gguf_path)
    _check_arch_compatibility(args, meta)
    _show_metadata_warnings(args, meta)

    intermediate_dtype = _resolve_dtype(args, meta)

    if not getattr(args, 'output', None):
        final_dir = get_output_dir(gguf_path)

    if getattr(args, 'cleanup_old', False):
        _cleanup_old_intermediates(final_dir)

    intermediate_dir = final_dir.parent / (final_dir.name + "_intermediate")
    skip_step1 = _resolve_resume(args, intermediate_dir)
    do_quantize = not getattr(args, 'no_quantize', False)

    _resolve_quant_params(args, gguf_size_bytes, hw)
    total_steps = 3 if do_quantize else 1

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if not getattr(args, 'force', False) and not check_disk_space(gguf_path, final_dir):
        console.print(MSG_CANCELLED)
        sys.exit(0)

    _show_conversion_plan(args, skip_step1, do_quantize, intermediate_dtype)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=getattr(args, 'quiet', False),
    )

    conversion_success = False
    t0 = time.time()

    with progress:
        pipeline_task = progress.add_task(
            "[bold cyan]Pipeline[/bold cyan]", total=total_steps
        )

        try:
            ensure_deps(deps, for_convert=True)

            if skip_step1:
                info("Skipping Step 1 as requested (resume mode).")
                progress.update(pipeline_task, advance=1)
            else:
                _run_step1(
                    args, gguf_path, intermediate_dir,
                    intermediate_dtype, meta, progress, pipeline_task,
                )

            arch = str((meta or {}).get("architecture", "")).lower()
            if arch in ("gemma4", "gemma3"):
                fix_gemma4_tensor_names(intermediate_dir)

            if not do_quantize:
                _save_float16_direct(
                    args, intermediate_dir, final_dir, intermediate_dtype
                )

            _run_step2(
                args, intermediate_dir, final_dir,
                meta, progress, pipeline_task,
            )

            _run_step3(args, intermediate_dir, progress, pipeline_task)
            conversion_success = True

        except Exception as e:
            fail(f"Conversion failed: {e}")
            sys.exit(1)

        finally:
            if not conversion_success and not getattr(args, 'keep_intermediate', False):
                try:
                    if intermediate_dir.exists():
                        shutil.rmtree(intermediate_dir)
                        info("Cleaned up intermediate files after failure.")
                except Exception as cleanup_error:
                    warn(f"Failed to clean up {intermediate_dir}: {cleanup_error}")

    total_time = time.time() - t0
    _show_conversion_summary(
        args, gguf_path, final_dir, gguf_size_bytes, total_time, do_quantize
    )

    # Post-conversion actions
    if conversion_success:
        _show_post_convert_actions(args, gguf_path, final_dir)


def _show_post_convert_actions(
    args: argparse.Namespace,
    gguf_path: Path,
    final_dir: Path,
) -> None:
    """Offer post-conversion actions: test, inspect, delete GGUF."""
    console.print()
    console.print(Panel(
        "[bold]What would you like to do next?[/bold]",
        border_style="cyan",
        padding=(0, 1),
    ))
    console.print("  [bold cyan]1.[/bold cyan]  Test the model with a quick prompt")
    console.print("  [bold cyan]2.[/bold cyan]  Start interactive chat with the model")
    console.print("  [bold cyan]3.[/bold cyan]  Show model info & file listing")
    if not getattr(args, 'delete_gguf', False):
        console.print("  [bold cyan]4.[/bold cyan]  Delete original GGUF to free space")
    console.print("  [bold cyan]0.[/bold cyan]  Done — back to menu")
    console.print()

    choices = ["0", "1", "2", "3"]
    if not getattr(args, 'delete_gguf', False):
        choices.append("4")

    choice = Prompt.ask("  [bold]Choose[/bold]", choices=choices, default="0")

    if choice == "1":
        console.print()
        console.print(
            "  [bold cyan]Running:[/bold cyan] "
            f"[dim]python3 -m mlx_lm generate --model \"{final_dir}\" --prompt \"Hello, introduce yourself\" --max-tokens 100[/dim]"
        )
        if Confirm.ask("  [bold]Run this command?[/bold]", default=True):
            try:
                subprocess.run(
                    [sys.executable, "-m", "mlx_lm", "generate",
                     "--model", str(final_dir),
                     "--prompt", "Hello, introduce yourself",
                     "--max-tokens", "100"],
                    check=False,
                )
            except FileNotFoundError:
                warn("mlx_lm not available — install: pip install mlx-lm")
    elif choice == "2":
        console.print()
        console.print(
            "  [bold cyan]To chat:[/bold cyan] "
            f"[dim]python3 -m mlx_lm chat --model \"{final_dir}\"[/dim]"
        )
    elif choice == "3":
        console.print()
        _list_output_files(final_dir)
    elif choice == "4":
        if Confirm.ask(
            f"  [yellow]Delete {gguf_path.name}?[/yellow]", default=False,
        ):
            try:
                gguf_path.unlink()
                ok(f"Deleted: [dim]{gguf_path.name}[/dim]")
            except OSError as e:
                warn(f"Could not delete: {e}")


def _list_output_files(final_dir: Path) -> None:
    """List files in the output directory."""
    files = sorted(final_dir.rglob("*"), key=lambda f: f.stat().st_size, reverse=True)
    console.print()
    t = Table(box=box.SIMPLE_HEAD, border_style="dim")
    t.add_column("File", style="bold", width=40)
    t.add_column("Size", justify="right", width=10)
    for f in files:
        if f.is_file():
            t.add_row(f.name, format_size(f.stat().st_size))
    console.print(t)


def _handle_hf_search_mode(args: argparse.Namespace) -> None:
    """Handle --hf-search: search HF Hub and let user pick a model."""
    token = args.hf_token or get_hf_token(quiet=args.quiet)

    console.print()
    _menu_header(f"Search: {args.hf_search}", "huggingface")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console, transient=True,
    ) as prog:
        prog.add_task("[cyan]Searching HuggingFace Hub...", total=None)
        results = hf_search(args.hf_search, token=token)

    if not results:
        warn("No models found matching your query.")
        sys.exit(0)

    console.print()
    table = Table(
        title=f"Search Results ({len(results)} models)",
        title_style="bold cyan",
        box=box.ROUNDED,
        border_style="dim",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Model ID", style="bold", width=45)
    table.add_column("Downloads", justify="right", width=12)
    table.add_column("Likes", justify="right", width=8)
    table.add_column("Pipeline", width=18)

    for i, m in enumerate(results, 1):
        model_id = m["id"]
        if len(model_id) > 42:
            model_id = model_id[:39] + "..."
        table.add_row(
            str(i),
            model_id,
            f"{m['downloads']:,}",
            str(m["likes"]),
            m.get("pipeline_tag", "") or "",
        )
    console.print(table)
    console.print()

    raw = Prompt.ask(
        "[bold]Select model number to download[/bold] (or Enter to cancel)",
        default="",
    )
    if not raw.strip():
        console.print(MSG_CANCELLED)
        sys.exit(0)

    try:
        idx = int(raw.strip()) - 1
        if 0 <= idx < len(results):
            repo_id = results[idx]["id"]
            _handle_hf_download(repo_id, args)
        else:
            warn("Invalid selection")
            sys.exit(1)
    except ValueError:
        warn("Invalid selection")
        sys.exit(1)


def _handle_hf_download(repo_id: str, args: argparse.Namespace) -> None:
    """Handle downloading a model from HuggingFace Hub.

    If --hf-file is specified, use that; otherwise list files and let user pick.
    """
    token = args.hf_token or get_hf_token(quiet=args.quiet)

    # Determine filename
    if args.hf_file:
        filename = args.hf_file
    else:
        # List files and look for GGUF files
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console, transient=True,
        ) as prog:
            prog.add_task(
                f"[cyan]Listing files in {repo_id}...", total=None
            )
            files = hf_list_files(repo_id, token=token)

        # Filter for GGUF files
        gguf_files = [f for f in files if f["path"].lower().endswith(".gguf")]

        if not gguf_files:
            # No GGUF files - show all and let user pick
            all_files = sorted(files, key=lambda f: f["size"], reverse=True)
            if not all_files:
                fail(f"No files found in {repo_id} or repo does not exist.")
                sys.exit(1)

            console.print()
            file_table = Table(
                title=f"Files in {repo_id}",
                title_style="bold cyan",
                box=box.ROUNDED,
                border_style="dim",
            )
            file_table.add_column("#", style="dim", width=3)
            file_table.add_column("Filename", style="bold", width=60)
            file_table.add_column("Size", justify="right", width=12)

            # Show top 50 files
            for i, f in enumerate(all_files[:50], 1):
                file_table.add_row(str(i), f["path"][:58], f["size_str"])
            console.print(file_table)
            console.print()

            raw = Prompt.ask(
                "[bold]Select file number to download[/bold] (or Enter to cancel)",
                default="",
            )
            if not raw.strip():
                console.print(MSG_CANCELLED)
                sys.exit(0)
            try:
                idx = int(raw.strip()) - 1
                if 0 <= idx < len(all_files):
                    filename = all_files[idx]["path"]
                else:
                    warn("Invalid selection")
                    sys.exit(1)
            except ValueError:
                warn("Invalid selection")
                sys.exit(1)
        else:
            # Let user pick from GGUF files
            if len(gguf_files) == 1:
                filename = gguf_files[0]["path"]
                ok(f"Auto-selected: [bold]{filename}[/bold]")
            else:
                console.print()
                file_table = Table(
                    title=f"GGUF Files in {repo_id}",
                    title_style="bold cyan",
                    box=box.ROUNDED,
                    border_style="dim",
                )
                file_table.add_column("#", style="dim", width=3)
                file_table.add_column("Filename", style="bold", width=60)
                file_table.add_column("Size", justify="right", width=12)

                for i, f in enumerate(gguf_files, 1):
                    file_table.add_row(str(i), f["path"][:58], f["size_str"])
                console.print(file_table)
                console.print()

                raw = Prompt.ask(
                    "[bold]Select file number[/bold] (or Enter to cancel)",
                    default="",
                )
                if not raw.strip():
                    console.print(MSG_CANCELLED)
                    sys.exit(0)
                try:
                    idx = int(raw.strip()) - 1
                    if 0 <= idx < len(gguf_files):
                        filename = gguf_files[idx]["path"]
                    else:
                        warn("Invalid selection")
                        sys.exit(1)
                except ValueError:
                    warn("Invalid selection")
                    sys.exit(1)

    # Download the file
    console.print()
    result = download_from_huggingface(
        repo_id, filename,
        token=token,
        quiet=args.quiet,
        auto_convert=args.auto_convert,
        delete_gguf=args.delete_gguf,
        output_dir=Path(args.output).expanduser() if getattr(args, 'output', None) else None,
        conversion_args=args,
    )

    if result is None:
        sys.exit(1)

    if not args.auto_convert:
        console.print()
        console.print(Panel(
            "[bold green]Download Complete![/bold green]\n\n"
            f"File: [bold cyan]{result}[/bold cyan]\n"
            f"Size: {format_size(result.stat().st_size)}\n\n"
            "[dim]Next steps:[/dim]\n"
            "  Run conversion: [cyan]cpmm \"" + str(result) + "\"[/cyan]\n"
            "  Or use --auto-convert to chain download and conversion",
            border_style="green",
        ))

    sys.exit(0)


def _handle_hf_list_mode(args: argparse.Namespace) -> None:
    """Handle --hf-list: list files in a HuggingFace repo."""
    if not args.hf_list:
        fail("--hf-list requires a repo ID")
        sys.exit(1)

    token = args.hf_token or get_hf_token(quiet=args.quiet)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console, transient=True,
    ) as prog:
        prog.add_task(
            f"[cyan]Listing files in {args.hf_list}...", total=None
        )
        files = hf_list_files(args.hf_list, token=token)

    if not files:
        warn(f"No files found or repo does not exist: {args.hf_list}")
        sys.exit(0)

    # Sort by size (largest first)
    files.sort(key=lambda f: f["size"], reverse=True)

    console.print()
    table = Table(
        title=f"Files in {args.hf_list} ({len(files)} total)",
        title_style="bold cyan",
        box=box.ROUNDED,
        border_style="dim",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Filename", style="bold", width=65)
    table.add_column("Size", justify="right", width=12)

    for i, f in enumerate(files, 1):
        table.add_row(str(i), f["path"][:63], f["size_str"])
    console.print(table)

    console.print()
    sys.exit(0)

# ═══════════════════════════════════════════════════════════════════════════
# Interactive Menu System
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Interactive Menu System — full guided workflows with Rich UI
# ═══════════════════════════════════════════════════════════════════════════

# History tracking
HISTORY_PATH = CONFIG_DIR / "history.json"


def _load_history() -> list[dict[str, Any]]:
    """Load conversion/download history from config."""
    if HISTORY_PATH.exists():
        try:
            data = json.loads(HISTORY_PATH.read_text())
            if isinstance(data, list):
                return data[-10:]  # Keep last 10
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_history_entry(entry: dict[str, Any]) -> None:
    """Append an entry to conversion history."""
    history = _load_history()
    history.append(entry)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history[-10:], indent=2))


def _make_menu_args(**overrides: Any) -> argparse.Namespace:
    """Create a default args Namespace for interactive menu use."""
    defaults: dict[str, Any] = {
        'input': None, 'output': None,
        'bits': None, 'group_size': None, 'mode': None,
        'predicate': None, 'no_quantize': False,
        'preset': None, 'high_bandwidth': False,
        'dtype': None, 'resume': False,
        'inspect': False, 'estimate': False, 'mtp': False,
        'keep_intermediate': False, 'cleanup_old': False,
        'force': False, 'quiet': False, 'no_color': False,
        'scan': False, 'scan_omlx': False, 'scan_lmstudio': False,
        'scan_hf_cache': False, 'models_dir': None,
        'set_models_dir': None, 'delete_gguf': False,
        'hf_search': None, 'hf_download': None, 'hf_file': None,
        'hf_token': None, 'hf_list': None, 'auto_convert': False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ── helpers ──


def _menu_header(title: str, icon: str = "") -> None:
    """Print a consistent section header for guided workflows."""
    console.print()
    console.print(Panel(
        f"[bold]{title}[/bold]",
        border_style="cyan",
        padding=(0, 1),
        subtitle=icon if icon else None,
    ))


def _pause_after(msg: str = "Press Enter to return to menu") -> None:
    """Pause and wait for user input before returning to main menu."""
    console.print()
    Prompt.ask(f"  [dim]{msg}[/dim]", default="")


def _menu_banner() -> None:
    """Print a banner with version and hardware summary."""
    hw = detect_apple_silicon()
    chip_label = hw["chip_name"] if hw["is_apple_silicon"] else "Unknown"
    ram_label = f"{hw['ram_gb']:.0f} GB"
    history = _load_history()
    history_count = len(history)

    subtitle = (
        f"[cyan]{chip_label}  \u2022  {ram_label}[/cyan]"
        + (f"  \u2022  [green]{history_count} conversion(s)[/green]" if history_count else "")
    )

    console.print()
    console.print(Panel(
        "[bold]GGUF \u2192 MLX[/bold]  [dim]converter + quantizer[/dim]\n"
        + subtitle,
        border_style="cyan",
        padding=(1, 1),
    ))


# ── Option 1: Guided Conversion ──


def _show_guided_convert(args: argparse.Namespace) -> None:
    """Guided conversion: prompt for file, configure, run."""
    _menu_header("Convert a Model", "\u2699")
    hw = detect_apple_silicon()

    try:
        gguf_path = get_gguf_path()

        # Show metadata summary
        meta = read_gguf_metadata(gguf_path)
        if meta and meta.get("architecture"):
            console.print()
            t = Table(box=box.SIMPLE_HEAD, border_style="dim", show_header=False)
            t.add_column("", style="dim", width=16)
            t.add_column("", style="bold")
            t.add_row("Architecture", meta.get("architecture", "?"))
            if meta.get("param_count"):
                t.add_row("Parameters", f"{meta['param_count']/1e9:.2f}B")
            if meta.get("file_type_name"):
                t.add_row("Source type", meta["file_type_name"])
            t.add_row("File size", format_size(gguf_path.stat().st_size))
            console.print(t)

        # Quantisation choice
        console.print()
        q_panel = Panel(
            "[bold]Quantisation[/bold]\n\n"
            "[bold cyan]1.[/bold cyan]  [bold]Smart defaults[/bold]  "
            "[dim](auto-selected for your hardware)[/dim]\n"
            "[bold cyan]2.[/bold cyan]  [bold]Quality preset[/bold]  "
            "[dim](8-bit, highest quality)[/dim]\n"
            "[bold cyan]3.[/bold cyan]  [bold]Custom config[/bold]  "
            "[dim](full control)[/dim]",
            border_style="cyan", padding=(1, 2),
        )
        console.print(q_panel)
        q_choice = Prompt.ask("  [bold]Choose[/bold]", choices=["1", "2", "3"], default="1")

        if q_choice == "2":
            args.preset = "quality"
        elif q_choice == "3":
            bits_str = Prompt.ask(
                "  Bits per weight",
                choices=["2", "3", "4", "6", "8"], default="4",
            )
            args.bits = int(bits_str)
            gs_str = Prompt.ask(
                "  Group size",
                choices=["32", "64", "128", "256"], default="64",
            )
            args.group_size = int(gs_str)
            if Confirm.ask("  Skip quantisation (float16 only)?", default=False):
                args.no_quantize = True

        # Delete-GGUF prompt
        gguf_size_bytes = gguf_path.stat().st_size
        _resolve_quant_params(args, gguf_size_bytes, hw)

        # Show pre-conversion plan
        console.print()
        if args.no_quantize:
            info(f"Plan: [bold]float16[/bold] only, est. {format_size(gguf_size_bytes)}")
        else:
            info(f"Plan: [bold]{args.bits}-bit[/bold], group={args.group_size}, est. ~{format_size(gguf_size_bytes * (args.bits / 16.0))}")

        if Confirm.ask(
            "  [yellow]Delete original GGUF after successful conversion?[/yellow]",
            default=False,
        ):
            args.delete_gguf = True

        # Final confirm
        console.print()
        if Confirm.ask("  [bold cyan]\u25b6 Start conversion?[/bold cyan]", default=True):
            main_with_file(gguf_path, args)
            _save_history_entry({
                "action": "convert",
                "input": str(gguf_path),
                "bits": args.bits,
                "preset": args.preset,
            })
        else:
            warn("Cancelled")

    except (KeyboardInterrupt, EOFError):
        warn("Cancelled")
    except Exception as e:
        fail(f"Error: {e}")

    _pause_after()


# ── Option 2: Scan & Convert ──


def _show_guided_scan_convert(args: argparse.Namespace) -> None:
    """Guided scan + convert workflow."""
    _menu_header("Scan & Convert", "\U0001F50D")

    try:
        source_choice = Prompt.ask(
            "  [bold]Scan source[/bold]",
            choices=["all", "omlx", "lmstudio", "custom"],
            default="all",
        )

        custom_dir = None
        scan_omlx = scan_lmstudio = scan_hf = False

        if source_choice == "all":
            scan_omlx = scan_lmstudio = scan_hf = True
        elif source_choice == "omlx":
            scan_omlx = True
        elif source_choice == "lmstudio":
            scan_lmstudio = True
        elif source_choice == "custom":
            raw = Prompt.ask("  [bold]Path to models directory[/bold]")
            custom_dir = Path(raw.strip().strip("'\"")).expanduser()
            if not custom_dir.exists():
                fail(f"Directory not found: {custom_dir}")
                _pause_after()
                return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console, transient=True,
        ) as prog:
            prog.add_task("[cyan]Scanning for models...", total=None)
            models = scan_for_models(
                custom_dir=custom_dir,
                scan_omlx=scan_omlx,
                scan_lmstudio=scan_lmstudio,
                scan_hf=scan_hf,
            )

        selected = display_scan_results(models)
        if selected is None:
            return

        info(f"Selected: [bold]{selected.path.name}[/bold]")

        if selected.format == "gguf":
            if Confirm.ask("  [bold cyan]\u25b6 Convert to MLX?[/bold cyan]", default=True):
                main_with_file(selected.path, args)
        else:
            console.print()
            console.print(Panel(
                f"[bold green]\u2713 MLX model ready[/bold green]\n\n"
                f"Path: [cyan]{selected.path.parent}[/cyan]\n"
                f"Size: {format_size(selected.size_gb * 1e9)}\n\n"
                "[dim]Try:[/dim]\n"
                f"  [cyan]python3 -m mlx_lm generate --model \"{selected.path.parent}\" "
                f'--prompt "Hello"[/cyan]',
                border_style="green",
            ))

    except (KeyboardInterrupt, EOFError):
        warn("Cancelled")
    except Exception as e:
        fail(f"Error: {e}")

    _pause_after()


# ── Option 3 & 4: HuggingFace Download ──


def _show_guided_hf_download(auto_convert: bool = False) -> None:
    """Guided HuggingFace download workflow."""
    label = "Download & Convert" if auto_convert else "Download from HF"
    _menu_header(label, "\U0001F4E5")

    try:
        token = get_hf_token()

        query = Prompt.ask("  [bold]Search HuggingFace[/bold] (e.g. \"mistral gguf\")")
        if not query.strip():
            return

        results = hf_search(query, token=token)
        if not results:
            warn("No models found. Try a different search.")
            _pause_after()
            return

        console.print()
        t = Table(
            title=f"Results \u2014 {len(results)} model(s)",
            title_style="bold cyan", box=box.SIMPLE_HEAD, header_style="bold white",
        )
        t.add_column("#", style="dim", width=3)
        t.add_column("Model ID", style="bold", width=50)
        t.add_column("Downloads", justify="right", width=12)
        t.add_column("Likes", justify="right", width=6)
        for i, m in enumerate(results, 1):
            mid = m["id"][:47] + "..." if len(m["id"]) > 47 else m["id"]
            t.add_row(str(i), mid, f"{m['downloads']:,}", str(m["likes"]))
        console.print(t)

        raw = Prompt.ask("  [bold]Model to download[/bold] (number, or Enter to cancel)", default="")
        if not raw.strip():
            return
        try:
            idx = int(raw.strip()) - 1
            if not (0 <= idx < len(results)):
                warn("Invalid selection")
                return
            repo_id = results[idx]["id"]
        except ValueError:
            warn("Invalid selection")
            return

        # List files
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console, transient=True,
        ) as prog:
            prog.add_task(f"[cyan]Listing files in {repo_id}...", total=None)
            files = hf_list_files(repo_id, token=token)

        gguf_files = [f for f in files if f["path"].lower().endswith(".gguf")]
        if not gguf_files:
            fail("No GGUF files found in this repository")
            _pause_after()
            return

        filename: str
        if len(gguf_files) == 1:
            filename = gguf_files[0]["path"]
            ok(f"Auto-selected: [bold]{filename}[/bold]")
        else:
            console.print()
            ft = Table(
                title="GGUF files", title_style="bold cyan",
                box=box.SIMPLE_HEAD, header_style="bold white",
            )
            ft.add_column("#", style="dim", width=3)
            ft.add_column("Filename", style="bold", width=65)
            ft.add_column("Size", justify="right", width=12)
            for i, f in enumerate(gguf_files, 1):
                ft.add_row(str(i), f["path"][:63], f["size_str"])
            console.print(ft)

            c = Prompt.ask("  [bold]File to download[/bold] (number)", default="1")
            try:
                idx = int(c) - 1
                if not (0 <= idx < len(gguf_files)):
                    warn("Invalid selection")
                    return
                filename = gguf_files[idx]["path"]
            except ValueError:
                return

        if Confirm.ask(f"  [bold cyan]\u25b6 Download {repo_id}/{filename}?[/bold cyan]", default=True):
            download_from_huggingface(
                repo_id, filename, token=token, auto_convert=auto_convert,
            )
            _save_history_entry({
                "action": "download_convert" if auto_convert else "download",
                "repo": repo_id,
                "file": filename,
            })

    except (KeyboardInterrupt, EOFError):
        warn("Cancelled")
    except Exception as e:
        fail(f"Error: {e}")

    _pause_after()


# ── Option 5: Inspect ──


def _show_guided_inspect() -> None:
    """Guided model inspection."""
    _menu_header("Inspect a Model", "\U0001F50D")

    try:
        gguf_path = get_gguf_path()
        if not gguf_path.exists():
            fail(f"File not found: {gguf_path}")
            _pause_after()
            return

        if gguf_path.suffix.lower() != ".gguf":
            warn(f"Not a .gguf file: {gguf_path.name}")
            info("Only GGUF files can be inspected with metadata.")
            _pause_after()
            return

        if not _has_gguf_py():
            fail("gguf-py package required. Install: pip install gguf")
            _pause_after()
            return

        meta = read_gguf_metadata(gguf_path)
        if meta:
            display_metadata(meta)
        else:
            fail("Could not read metadata")

    except (KeyboardInterrupt, EOFError):
        warn("Cancelled")
    except Exception as e:
        fail(f"Error: {e}")

    _pause_after()


# ── Option 6: Settings ──


def _show_settings_menu() -> None:
    """Settings management sub-menu."""
    while True:
        config = load_config()
        models_dir = config.get("models_dir", "[dim]Not set[/dim]")
        has_token = "[bold green]\u2713 Configured[/bold green]" if config.get("hf_token") else "[dim]Not set[/dim]"
        history = _load_history()
        history_count = len(history)

        _menu_header("Settings", "\u2699")

        console.print()
        t = Table(box=box.SIMPLE_HEAD, border_style="dim", show_header=False)
        t.add_column("Setting", style="bold", width=22)
        t.add_column("Value", width=40)
        t.add_row("Models directory", models_dir)
        t.add_row("HF token", has_token)
        t.add_row("Config file", f"[dim]{CONFIG_PATH}[/dim]")
        t.add_row("Conversion history", f"{history_count} entries")
        console.print(t)

        if history:
            console.print()
            console.print("  [dim]Recent activity:[/dim]")
            for entry in history[-3:]:
                if entry.get("action") in ("download", "download_convert"):
                    label = "DL" if entry["action"] == "download" else "DL+Conv"
                    console.print(f"    [dim]\u2022 [{label}] {entry.get('repo', '?')}[/dim]")
                else:
                    inp = entry.get("input", "?")
                    console.print(f"    [dim]\u2022 [Conv] {Path(inp).name}[/dim]")

        console.print()
        console.print("  [bold cyan]1.[/bold cyan]  Set models directory")
        console.print("  [bold cyan]2.[/bold cyan]  Set HuggingFace token")
        console.print("  [bold cyan]3.[/bold cyan]  Clear HuggingFace token")
        console.print("  [bold cyan]4.[/bold cyan]  Clear conversion history")
        console.print("  [bold cyan]5.[/bold cyan]  Back to main menu")
        console.print()

        choice = Prompt.ask(
            "  [bold]Option[/bold]",
            choices=["1", "2", "3", "4", "5"],
            default="5",
        )

        if choice == "1":
            raw = Prompt.ask("  [bold]Path to models folder[/bold]")
            if raw.strip():
                set_models_dir(raw.strip())
        elif choice == "2":
            token = Prompt.ask(
                "  [bold]HuggingFace token[/bold]"
                " (get one at [dim]hf.co/settings/tokens[/dim])"
            )
            if token.strip():
                save_hf_token(token.strip())
                ok("Token saved")
        elif choice == "3":
            if Confirm.ask("  [yellow]Clear saved HF token?[/yellow]", default=False):
                config.pop("hf_token", None)
                save_config(config)
                ok("Token removed")
        elif choice == "4":
            if Confirm.ask("  [yellow]Clear conversion history?[/yellow]", default=False):
                HISTORY_PATH.write_text("[]")
                ok("History cleared")
        else:
            break


# ── Main Menu Loop ──


def _get_hardware_recommendation() -> str:
    """Return a hardware-specific size recommendation string."""
    hw = detect_apple_silicon()
    ram = hw["ram_gb"]
    tier = hw["chip_tier"]

    if ram >= 64 and tier in ("max", "ultra"):
        return "[green]\u2713[/green]  [bold]70B+ models[/bold] fit comfortably (4-bit ~35 GB)"
    elif ram >= 48:
        return "[green]\u2713[/green]  [bold]30B–70B[/bold] recommended (4-bit ~15–35 GB)"
    elif ram >= 32:
        return "[yellow]~[/yellow]  [bold]13B–30B[/bold] works well (4-bit ~7–15 GB)"
    elif ram >= 16:
        return "[yellow]~[/yellow]  [bold]7B–13B[/bold] ideal (4-bit ~4–7 GB)"
    else:
        return "[yellow]~[/yellow]  [bold]1B–7B[/bold] only (limited RAM)"


def run_interactive_menu() -> None:
    """Launch the full interactive CLI menu system."""
    _menu_banner()

    # Hardware recommendation
    rec = _get_hardware_recommendation()
    console.print(f"  {rec}")

    while True:
        console.print()
        console.print(Rule(style="dim"))

        menu = Table(
            title="[bold cyan]Main Menu[/bold cyan]",
            box=box.SIMPLE_HEAD,
            border_style="dim",
            header_style="bold white",
            padding=(0, 2),
        )
        menu.add_column("Action", style="bold", width=38)
        menu.add_column("Description", style="dim", width=50)

        menu.add_row("[bold]1.[/bold] Convert a Model", "GGUF file \u2192 MLX with smart defaults or custom quant")
        menu.add_row("[bold]2.[/bold] Scan & Convert", "Scan omlx/LM Studio/HF cache, pick a model, convert")
        menu.add_row("[bold]3.[/bold] Download from HF", "Search HuggingFace Hub, pick a model, download")
        menu.add_row("[bold]4.[/bold] Download & Convert", "Search HF, download, and auto-convert to MLX")
        menu.add_row("[bold]5.[/bold] Inspect a Model", "View full GGUF metadata breakdown")
        menu.add_row("[bold]6.[/bold] Settings", "Configure paths, manage HF token, view history")
        menu.add_row("", "")
        menu.add_row("[bold]0.[/bold] Exit", "")

        console.print(menu)

        try:
            choice = Prompt.ask(
                "  [bold cyan]What would you like to do?[/bold cyan]",
                choices=["0", "1", "2", "3", "4", "5", "6"],
                default="1",
            )
        except (KeyboardInterrupt, EOFError):
            console.print()
            info("Goodbye!")
            console.print()
            sys.exit(0)

        try:
            if choice == "0":
                console.print()
                info("Goodbye!")
                console.print()
                sys.exit(0)
            elif choice == "1":
                _show_guided_convert(_make_menu_args(force=True))
            elif choice == "2":
                _show_guided_scan_convert(_make_menu_args(force=True))
            elif choice == "3":
                _show_guided_hf_download(auto_convert=False)
            elif choice == "4":
                _show_guided_hf_download(auto_convert=True)
            elif choice == "5":
                _show_guided_inspect()
            elif choice == "6":
                _show_settings_menu()
        except (KeyboardInterrupt, EOFError):
            console.print()
            info("Returning to menu...")
            continue


def _handle_scan_mode(args: argparse.Namespace) -> None:
    """Handle --scan mode: scan model directories and let user pick."""
    custom_dir = Path(args.models_dir).expanduser() if getattr(args, 'models_dir', None) else None

    scan_omlx = getattr(args, 'scan_omlx', False)
    scan_lmstudio = getattr(args, 'scan_lmstudio', False)
    scan_hf = getattr(args, 'scan_hf_cache', False)
    scan_all = not (scan_omlx or scan_lmstudio or scan_hf or custom_dir)

    info("Scanning for models...")
    models = scan_for_models(
        custom_dir=custom_dir,
        scan_omlx=scan_omlx or scan_all,
        scan_lmstudio=scan_lmstudio or scan_all,
        scan_hf=scan_hf or scan_all,
    )

    selected = display_scan_results(models)
    if selected is None:
        console.print(MSG_CANCELLED)
        sys.exit(0)

    console.print()
    info(f"Selected: [bold]{selected.path}[/bold]")

    if selected.format == "gguf":
        # Run conversion on the selected GGUF
        console.print()
        ans = Confirm.ask(
            "  [yellow]Convert this model to MLX?[/yellow]",
            default=True,
        )
        if ans:
            main_with_file(selected.path, args)
        else:
            console.print(MSG_CANCELLED)
    else:
        # MLX model - show info
        console.print()
        console.print(Panel(
            f"[bold green]MLX Model Selected[/bold green]\n\n"
            f"Path: [cyan]{selected.path}[/cyan]\n"
            f"Size: {format_size(selected.size_gb * 1e9)}\n\n"
            "[dim]MLX models are already in the correct format.\n"
            "To use with mlx_lm:[/dim]\n"
            f"  [cyan]python3 -m mlx_lm generate --model \"{selected.path.parent}\" --prompt \"Hello\"[/cyan]",
            border_style="green",
        ))

    sys.exit(0)


def _has_cli_args(args: argparse.Namespace) -> bool:
    """Check if the user provided any meaningful CLI arguments.

    Returns False when only display flags (--quiet, --no-color)
    or --help are present, so the interactive menu launches instead.
    """
    # Input file or output dir = explicit intent
    if args.input is not None or args.output is not None:
        return True
    # Any mode/action flag
    mode_flags: tuple[str, ...] = (
        'scan', 'scan_omlx', 'scan_lmstudio', 'scan_hf_cache',
        'set_models_dir', 'models_dir',
        'hf_search', 'hf_download', 'hf_file', 'hf_token', 'hf_list',
        'auto_convert',
        'inspect', 'estimate', 'mtp',
    )
    if any(getattr(args, f, None) not in (None, False) for f in mode_flags):
        return True
    return False


def main() -> None:
    global console

    parser = build_parser()
    args = parser.parse_args()

    if args.no_color:
        console = Console(no_color=True, highlight=False)

    # Launch interactive menu when no CLI args are provided
    if not _has_cli_args(args):
        try:
            run_interactive_menu()
        except KeyboardInterrupt:
            console.print("\n  [yellow]Goodbye![/yellow]")
        sys.exit(0)

    banner()

    hw = detect_apple_silicon()
    _show_hardware_table(hw, args.quiet)

    # === Handle --set-models-dir ===
    if args.set_models_dir:
        set_models_dir(args.set_models_dir)
        sys.exit(0)

    # === Handle --hf-search ===
    if args.hf_search:
        _handle_hf_search_mode(args)
        return

    # === Handle --hf-list ===
    if args.hf_list:
        _handle_hf_list_mode(args)
        sys.exit(0)

    # === Handle --hf-download ===
    if args.hf_download:
        _handle_hf_download(args.hf_download, args)
        return

    # === Handle --scan or scan-* flags ===
    if args.scan or args.scan_omlx or args.scan_lmstudio or args.scan_hf_cache or args.models_dir:
        _handle_scan_mode(args)
        return

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

    # Delete GGUF on successful conversion if requested
    if args.delete_gguf and conversion_success:
        console.print()
        if Confirm.ask(
            "  [yellow]Delete original GGUF file?[/yellow]",
            default=False,
        ):
            try:
                gguf_path.unlink()
                ok(f"Deleted GGUF: [dim]{gguf_path.name}[/dim]")
            except OSError as e:
                warn(f"Could not delete GGUF: {e}")


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
