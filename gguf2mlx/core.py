#!/usr/bin/env python3
"""
GGUF to MLX Converter v2.0
Converts GGUF models to MLX format (safetensors) for Apple Silicon inference.

Phase 1: Real weight extraction, safetensors output, architecture detection,
         real tokenizer extraction.
"""

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm

import numpy as np

# ---------------------------------------------------------------------------
# Required imports with friendly error messages
# ---------------------------------------------------------------------------

try:
    from gguf import GGUFReader
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    print("❌ gguf library required. Install: pip install gguf>=0.18.0")
    sys.exit(1)

try:
    from safetensors.numpy import save_file as save_safetensors
    from safetensors import safe_open

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("❌ safetensors library required. Install: pip install safetensors")
    sys.exit(1)

# ---------------------------------------------------------------------------
# GGUF metadata helpers
# ---------------------------------------------------------------------------


def get_metadata_str(reader: GGUFReader, key: str) -> Optional[str]:
    """Extract a string metadata value from GGUF fields."""
    field = reader.get_field(key)
    if field is None:
        return None
    val = field.contents()
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val) if val is not None else None


def get_metadata_int(reader: GGUFReader, key: str) -> Optional[int]:
    """Extract an integer metadata value from GGUF fields."""
    field = reader.get_field(key)
    if field is None:
        return None
    val = field.contents()
    if val is None:
        return None
    # Handle numpy arrays
    if isinstance(val, np.ndarray):
        return int(val.flat[0]) if val.size > 0 else None
    # Handle Python lists/tuples (some GGUF fields return these, e.g., Gemma4 head_count_kv)
    if isinstance(val, (list, tuple)):
        return int(val[0]) if len(val) > 0 else None
    return int(val)


def get_metadata_float(reader: GGUFReader, key: str) -> Optional[float]:
    """Extract a float metadata value from GGUF fields."""
    field = reader.get_field(key)
    if field is None:
        return None
    val = field.contents()
    if val is None:
        return None
    # Handle numpy arrays
    if isinstance(val, np.ndarray):
        return float(val.flat[0]) if val.size > 0 else None
    # Handle Python lists/tuples
    if isinstance(val, (list, tuple)):
        return float(val[0]) if len(val) > 0 else None
    return float(val)


def get_metadata_array_str(reader: GGUFReader, key: str) -> list[str]:
    """Extract a string array from GGUF fields (e.g., tokenizer tokens)."""
    field = reader.get_field(key)
    if field is None:
        return []
    try:
        vals = field.contents()
        if isinstance(vals, (list, np.ndarray)):
            return [
                v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
                for v in vals
            ]
        return []
    except Exception:
        return []


def get_metadata_array_int(reader: GGUFReader, key: str) -> list[int]:
    """Extract an integer array from GGUF fields (e.g., token types)."""
    field = reader.get_field(key)
    if field is None:
        return []
    try:
        vals = field.contents()
        if isinstance(vals, (list, np.ndarray)):
            return [int(v) for v in vals]
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Architecture detection & config building
# ---------------------------------------------------------------------------

# Map GGUF architecture names to HuggingFace model types
ARCH_MAP: dict[str, str] = {
    "llama": "llama",
    "mistral": "mistral",
    "falcon": "falcon",
    "mpt": "mpt",
    "gptneox": "gpt_neox",
    "gpt2": "gpt2",
    "bert": "bert",
    "bloom": "bloom",
    "starcoder": "gpt_bigcode",
    "refact": "refact",
    "command-r": "cohere",
    "command-r-plus": "cohere",
    "qwen2": "qwen2",
    "qwen2moe": "qwen2_moe",
    "qwen3moe": "qwen3_moe",
    "phi3": "phi3",
    "phi": "phi",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "gemma3": "gemma3",
    "stablelm": "stablelm",
    "deepseek2": "deepseek_v2",
    "deepseek3": "deepseek_v3",
    "chatglm": "chatglm",
    "baichuan": "baichuan",
    "xverse": "xverse",
    "orion": "orion",
    "bitnet": "bitnet",
    "plamo": "plamo",
    "codeshell": "codeshell",
    "minicpm": "minicpm",
    "minicpm3": "minicpm3",
    "t5": "t5",
    "jais": "jais",
    "olmo": "olmo",
    "olmo2": "olmo2",
    "openelm": "openelm",
    "dbrx": "dbrx",
    "grok-1": "grok",
    "arctic": "arctic",
    "nemotron": "nemotron",
    "exaone": "exaone",
    "granite": "granite",
    "smolm": "smolm",
    "chameleon": "chameleon",
}


def detect_architecture(reader: GGUFReader) -> str:
    """Detect model architecture from GGUF metadata."""
    arch = get_metadata_str(reader, "general.architecture")
    if arch:
        return arch
    # Try fallback based on model name
    name = get_metadata_str(reader, "general.name")
    if name:
        name_lower = name.lower()
        for gguf_arch, hf_name in ARCH_MAP.items():
            if gguf_arch in name_lower:
                return gguf_arch
    return "llama"  # Safe default


def build_config(reader: GGUFReader, arch: str) -> dict[str, Any]:
    """Build MLX-compatible config.json from GGUF metadata."""

    def _warn(key: str, value: Any) -> None:
        warnings.warn(f"  ⚠ '{key}' not found in GGUF metadata, using default: {value}")

    # --- Basic params ---
    vocab_size = get_metadata_int(reader, "llama.vocab_size") or get_metadata_int(
        reader, f"{arch}.vocab_size"
    )

    # If vocab_size not in metadata, infer from tokenizer tokens
    if vocab_size is None:
        tokens = get_metadata_array_str(reader, "tokenizer.ggml.tokens")
        if tokens:
            vocab_size = len(tokens)
        else:
            vocab_size = 32000
            _warn("vocab_size", 32000)

    hidden_size = get_metadata_int(reader, "llama.embedding_length") or get_metadata_int(
        reader, f"{arch}.embedding_length"
    )
    if hidden_size is None:
        hidden_size = 4096
        _warn("embedding_length", 4096)

    num_layers = get_metadata_int(reader, "llama.block_count") or get_metadata_int(
        reader, f"{arch}.block_count"
    )
    if num_layers is None:
        num_layers = 32
        _warn("block_count", 32)

    num_heads = get_metadata_int(reader, "llama.attention.head_count") or get_metadata_int(
        reader, f"{arch}.attention.head_count"
    )
    if num_heads is None:
        num_heads = 32
        _warn("head_count", 32)

    num_kv_heads = get_metadata_int(
        reader, "llama.attention.head_count_kv"
    ) or get_metadata_int(reader, f"{arch}.attention.head_count_kv") or num_heads

    # MoE: expert feed-forward length may differ from shared FFN
    if arch in ("qwen2moe", "qwen3moe", "deepseek2", "deepseek3", "dbrx", "grok-1"):
        ffn_size = get_metadata_int(
            reader, f"{arch}.expert_feed_forward_length"
        ) or get_metadata_int(reader, f"{arch}.feed_forward_length") or (hidden_size * 4)
        # Shared expert FFN (if present)
        shared_ffn_size = get_metadata_int(
            reader, f"{arch}.expert_shared_feed_forward_length"
        ) or ffn_size
    else:
        ffn_size = get_metadata_int(reader, "llama.feed_forward_length") or get_metadata_int(
            reader, f"{arch}.feed_forward_length"
        )
        if ffn_size is None:
            ffn_size = hidden_size * 4
        shared_ffn_size = ffn_size

    ctx_length = get_metadata_int(reader, "llama.context_length") or get_metadata_int(
        reader, f"{arch}.context_length"
    )
    if ctx_length is None:
        ctx_length = 4096
        _warn("context_length", 4096)

    rope_theta = get_metadata_float(reader, "llama.rope.freq_base") or get_metadata_float(
        reader, f"{arch}.rope.freq_base"
    )
    if rope_theta is None:
        rope_theta = 10000.0

    norm_eps = get_metadata_float(
        reader, "llama.attention.layer_norm_rms_epsilon"
    ) or get_metadata_float(reader, f"{arch}.attention.layer_norm_rms_epsilon")
    if norm_eps is None:
        norm_eps = 1e-6

    file_type = get_metadata_int(reader, "general.file_type") or 1
    model_name = get_metadata_str(reader, "general.name") or "unknown"
    bos_id = get_metadata_int(reader, "tokenizer.ggml.bos_token_id") or 1
    eos_id = get_metadata_int(reader, "tokenizer.ggml.eos_token_id") or 2

    hf_model_type = ARCH_MAP.get(arch, arch)

    # --- Build config ---
    # Detect tied embeddings (many Qwen, Gemma, etc. models have this)
    tie_embeddings = arch in (
        "qwen2", "qwen2moe", "gemma", "gemma2", "gemma3",
        "olmo", "olmo2", "openelm",
    )

    # Detect attention bias (Qwen, Gemma, etc.)
    attention_bias = arch in (
        "qwen2", "qwen2moe", "qwen3moe",
        "gemma", "gemma2", "gemma3",
    )

    # Convert model_type to CamelCase architecture class name
    arch_class = "".join(part.capitalize() for part in hf_model_type.split("_")) + "ForCausalLM"

    config = {
        "architectures": [arch_class],
        "model_type": hf_model_type,
        "hidden_size": hidden_size,
        "intermediate_size": ffn_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "max_position_embeddings": ctx_length,
        "rms_norm_eps": norm_eps,
        "rope_theta": rope_theta,
        "vocab_size": vocab_size,
        "hidden_act": "silu",
        "tie_word_embeddings": tie_embeddings,
        "attention_bias": attention_bias,
        "torch_dtype": "float16",
        "transformers_version": "4.50.0",
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        # Extra metadata from source GGUF
        "_gguf_architecture": arch,
        "_gguf_file_type": file_type,
        "_original_name": model_name,
    }

    # --- Architecture-specific overrides ---
    if arch in ("qwen2moe", "deepseek2", "deepseek3", "qwen3moe", "dbrx", "grok-1"):
        num_experts = get_metadata_int(reader, f"{arch}.expert_count") or 8
        num_experts_per_tok = get_metadata_int(
            reader, f"{arch}.expert_used_count"
        ) or 2
        config["num_experts"] = num_experts
        config["num_experts_per_tok"] = num_experts_per_tok
        config["model_type"] = {
            "qwen2moe": "qwen2_moe",
            "qwen3moe": "qwen3_moe",
            "deepseek2": "deepseek_v2",
            "deepseek3": "deepseek_v3",
        }.get(arch, hf_model_type)

        # MoE-specific config fields
        config["moe_intermediate_size"] = ffn_size
        config["norm_topk_prob"] = arch in ("qwen3moe",)
        config["decoder_sparse_step"] = 1
        config["mlp_only_layers"] = []

        # Head dim (Qwen3, DeepSeek-V3 style)
        head_dim = get_metadata_int(
            reader, f"{arch}.attention.key_length"
        ) or get_metadata_int(reader, f"{arch}.attention.value_length") or (hidden_size // num_heads)
        if arch in ("qwen3moe", "deepseek3"):
            config["head_dim"] = head_dim

        # Shared expert config (Qwen3MoE, DeepSeek-V3)
        if arch in ("qwen3moe", "deepseek3"):
            config["shared_expert_intermediate_size"] = shared_ffn_size
            config["output_router_logits"] = False
            config["router_aux_loss_coef"] = 0.001

    # --- Gemma4 MoE Configuration ---
    if arch in ("gemma4", "gemma3"):
        # Gemma4 has MoE with 128 experts, using 8 per token
        num_experts = get_metadata_int(reader, f"{arch}.expert_count") or 8
        num_experts_per_tok = get_metadata_int(reader, f"{arch}.expert_used_count") or 2
        moe_ffn_size = get_metadata_int(reader, f"{arch}.expert_feed_forward_length") or ffn_size
        
        # Head dimension from key_length
        head_dim = get_metadata_int(reader, f"{arch}.attention.key_length") or (hidden_size // num_heads)
        sliding_window = get_metadata_int(reader, f"{arch}.attention.sliding_window") or 0
        sliding_window_pattern = get_metadata_int(reader, f"{arch}.attention.sliding_window_pattern") or 4
        shared_kv_layers = get_metadata_int(reader, f"{arch}.attention.shared_kv_layers") or 0
        final_logit_softcapping = get_metadata_float(reader, f"{arch}.final_logit_softcapping") or 0.0
        
        config["enable_moe_block"] = True
        config["num_experts"] = num_experts
        config["num_experts_per_tok"] = num_experts_per_tok
        config["top_k_experts"] = num_experts_per_tok
        config["moe_intermediate_size"] = moe_ffn_size
        config["hidden_size_per_layer_input"] = 0  # Gemma4 doesn't use per-layer input
        
        # Head dimension config
        config["head_dim"] = head_dim
        config["global_head_dim"] = head_dim
        config["attention_k_eq_v"] = False
        
        # Layer types (sliding window pattern)
        config["sliding_window"] = sliding_window
        config["sliding_window_pattern"] = sliding_window_pattern
        config["num_kv_shared_layers"] = shared_kv_layers
        config["num_global_key_value_heads"] = num_kv_heads  # Default, may vary per layer
        
        # Rope config - use format expected by Llama3RoPE
        config["rope_traditional"] = False
        config["partial_rotary_factor"] = 0.5
        config["global_partial_rotary_factor"] = 0.5
        # Note: rope_parameters format must be compatible with mlx_lm's Llama3RoPE
        # which expects: factor, low_freq_factor, high_freq_factor, original_max_position_embeddings
        config["rope_parameters"] = {
            "factor": 4,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 4096,
        }
        
        # Other Gemma4-specific config
        config["final_logit_softcapping"] = final_logit_softcapping
        config["use_double_wide_mlp"] = False  # Gemma4 uses standard MLP
        config["tie_word_embeddings"] = False
        config["pad_token_id"] = 0
        config["vocab_size_per_layer_input"] = 0
        
        # Model type for Gemma4
        config["model_type"] = "gemma4"
        
        # Gemma4 uses tied weights (lm_head = token_embd.T)
        config["tie_word_embeddings"] = True
    
    # Create lm_head as transpose of embeddings for tied weights architectures
    if arch in ("gemma4", "gemma3", "gemma2"):
        # Gemma models use tied embeddings - lm_head = token_embd.weight.T
        # mlx_lm will handle the tying at load time, but we need to provide
        # a lm_head.weight that can be used
        pass

    return config


# ---------------------------------------------------------------------------
# Tensor name mapping: GGUF → MLX/HuggingFace
# ---------------------------------------------------------------------------


def _map_gemma4_tensor_name(gguf_name: str) -> str:
    """Map a Gemma4-architecture GGUF tensor name to MLX format.
    
    mlx_lm Gemma4 Model expects weights with 'language_model.model.' prefix:
    - language_model.lm_head.weight
    - language_model.model.embed_tokens.weight
    - language_model.model.layers.X.xxx
    
    The sanitize function preserves this prefix.
    """
    # Embedding (language_model.model. prefix)
    if gguf_name == "token_embd.weight":
        return "language_model.model.embed_tokens.weight"

    # Output
    if gguf_name == "output.weight":
        return "language_model.lm_head.weight"
    if gguf_name == "output_norm.weight":
        return "language_model.model.norm.weight"

    # Blocks: blk.N.xxx → language_model.model.layers.N.xxx
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".", 2)
        if len(parts) < 3:
            return gguf_name
        layer_idx = parts[1]
        rest = parts[2]

        # === Gemma4 Attention ===
        if rest == "attn_q.weight":
            return f"language_model.model.layers.{layer_idx}.self_attn.q_proj.weight"
        if rest == "attn_k.weight":
            return f"language_model.model.layers.{layer_idx}.self_attn.k_proj.weight"
        if rest == "attn_v.weight":
            return f"language_model.model.layers.{layer_idx}.self_attn.v_proj.weight"
        if rest == "attn_output.weight":
            return f"language_model.model.layers.{layer_idx}.self_attn.o_proj.weight"
        if rest == "attn_q_norm.weight":
            return f"language_model.model.layers.{layer_idx}.self_attn.q_norm.weight"
        if rest == "attn_k_norm.weight":
            return f"language_model.model.layers.{layer_idx}.self_attn.k_norm.weight"

        # === Gemma4 Layer Norms ===
        if rest == "attn_norm.weight":
            return f"language_model.model.layers.{layer_idx}.input_layernorm.weight"
        if rest == "attn_norm_2.weight":
            return f"language_model.model.layers.{layer_idx}.input_layernorm.weight"
        if rest == "post_attention_norm.weight":
            return f"language_model.model.layers.{layer_idx}.input_layernorm.weight"
        if rest == "post_attention_norm_1.weight":
            return f"language_model.model.layers.{layer_idx}.input_layernorm.weight"
        if rest == "ffn_norm.weight":
            return f"language_model.model.layers.{layer_idx}.post_attention_layernorm.weight"
        if rest == "post_ffw_norm.weight":
            return f"language_model.model.layers.{layer_idx}.post_attention_layernorm.weight"
        if rest == "post_ffw_norm_1.weight":
            return f"language_model.model.layers.{layer_idx}.post_feedforward_layernorm_1.weight"
        if rest == "post_ffw_norm_2.weight":
            return f"language_model.model.layers.{layer_idx}.post_feedforward_layernorm_2.weight"
        if rest == "pre_ffw_norm.weight":
            return f"language_language.model.layers.{layer_idx}.pre_feedforward_layernorm.weight"
        if rest == "pre_ffw_norm_2.weight":
            return f"language_model.model.layers.{layer_idx}.pre_feedforward_layernorm_2.weight"

        # === Gemma4 MoE FFN (Standard MLP) ===
        if rest == "ffn_gate.weight":
            return f"language_model.model.layers.{layer_idx}.mlp.gate_proj.weight"
        if rest == "ffn_up.weight":
            return f"language_model.model.layers.{layer_idx}.mlp.up_proj.weight"
        if rest == "ffn_down.weight":
            return f"language_model.model.layers.{layer_idx}.mlp.down_proj.weight"

        # === Gemma4 MoE FFN Layernorms ===
        if rest == "pre_feedforward_layernorm.weight":
            return f"language_model.model.layers.{layer_idx}.pre_feedforward_layernorm.weight"
        if rest == "post_feedforward_layernorm.weight":
            return f"language_model.model.layers.{layer_idx}.post_feedforward_layernorm.weight"

        # === Gemma4 MoE Layer Scalar ===
        if rest == "layer_scalar" or rest.endswith(".layer_scalar"):
            return f"language_model.model.layers.{layer_idx}.layer_scalar"

        # === Gemma4 MoE Router Scale ===
        if rest == "router.scale" or rest.endswith(".router.scale"):
            return f"language_model.model.layers.{layer_idx}.router.scale"
        if rest == "router.per_expert_scale" or rest.endswith(".router.per_expert_scale"):
            return f"language_model.model.layers.{layer_idx}.router.per_expert_scale"

        # === Gemma4 MoE Router ===
        if rest == "ffn_gate_inp.weight":
            return f"language_model.model.layers.{layer_idx}.router.proj.weight"

        # === Gemma4 MoE Experts (SwitchGLU) ===
        if rest == "ffn_gate_up_exps.weight":
            return f"language_model.model.layers.{layer_idx}.experts.gate_up_proj"
        if rest == "ffn_gate_exps.weight":
            return f"language_model.model.layers.{layer_idx}.experts.gate_up_proj"
        if rest == "ffn_up_exps.weight":
            return None  # Skip - merged into gate_up_proj by mlx_lm sanitize
        if rest == "ffn_down_exps.weight":
            return f"language_model.model.layers.{layer_idx}.experts.down_proj"

        # === Gemma4 Layer Scalar ===
        if rest == "layer_scalar" or rest.endswith(".layer_scalar"):
            return f"language_language.model.layers.{layer_idx}.layer_scalar"

        # === Skip scale factors ===
        if any(rest.startswith(x) or rest == x for x in [
            "ffn_down_exps.scale", "ffn_up_exps.scale", "ffn_gate_exps.scale",
            "layer_output_scale", "layer_output_scale.weight", "ffn_gate_inp.scale"
        ]):
            return None

        # === Rope Frequencies ===
        if rest == "rope_freqs.weight":
            return "rope_freqs.weight"

    return gguf_name


def _map_llama_tensor_name(gguf_name: str) -> str:
    """Map a Llama-architecture GGUF tensor name to HuggingFace format."""
    # Embedding
    if gguf_name == "token_embd.weight":
        return "model.embed_tokens.weight"

    # Output
    if gguf_name == "output.weight":
        return "lm_head.weight"
    if gguf_name == "output_norm.weight":
        return "model.norm.weight"

    # Blocks: blk.N.xxx → model.layers.N.xxx
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".", 2)
        if len(parts) < 3:
            return gguf_name
        layer_idx = parts[1]
        rest = parts[2]

        # Attention weights
        if rest == "attn_q.weight":
            return f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        if rest == "attn_k.weight":
            return f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        if rest == "attn_v.weight":
            return f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        if rest == "attn_output.weight":
            return f"model.layers.{layer_idx}.self_attn.o_proj.weight"

        # Attention biases (some architectures have these)
        if rest == "attn_q.bias":
            return f"model.layers.{layer_idx}.self_attn.q_proj.bias"
        if rest == "attn_k.bias":
            return f"model.layers.{layer_idx}.self_attn.k_proj.bias"
        if rest == "attn_v.bias":
            return f"model.layers.{layer_idx}.self_attn.v_proj.bias"
        if rest == "attn_output.bias":
            return f"model.layers.{layer_idx}.self_attn.o_proj.bias"

        # FFN
        if rest == "ffn_gate.weight":
            return f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        if rest == "ffn_up.weight":
            return f"model.layers.{layer_idx}.mlp.up_proj.weight"
        if rest == "ffn_down.weight":
            return f"model.layers.{layer_idx}.mlp.down_proj.weight"

        # MoE: expert router gate
        if rest == "ffn_gate_inp.weight":
            return f"model.layers.{layer_idx}.mlp.gate.weight"

        # MoE: stacked expert weights (3D: [num_experts, out, in])
        # mlx-lm expects switch_mlp.*.weight for stacked format
        if rest == "ffn_gate_exps.weight":
            return f"model.layers.{layer_idx}.mlp.switch_mlp.gate_proj.weight"
        if rest == "ffn_down_exps.weight":
            return f"model.layers.{layer_idx}.mlp.switch_mlp.down_proj.weight"
        if rest == "ffn_up_exps.weight":
            return f"model.layers.{layer_idx}.mlp.switch_mlp.up_proj.weight"

        # QK normalization (Qwen3, Qwen3MoE)
        if rest == "attn_q_norm.weight":
            return f"model.layers.{layer_idx}.self_attn.q_norm.weight"
        if rest == "attn_k_norm.weight":
            return f"model.layers.{layer_idx}.self_attn.k_norm.weight"

        # Norms
        if rest == "attn_norm.weight":
            return f"model.layers.{layer_idx}.input_layernorm.weight"
        if rest == "ffn_norm.weight":
            return f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        if rest == "attn_norm_2.weight":
            return f"model.layers.{layer_idx}.input_layernorm.weight"

        # Norm biases (rare but possible)
        if rest == "attn_norm.bias":
            return f"model.layers.{layer_idx}.input_layernorm.bias"
        if rest == "ffn_norm.bias":
            return f"model.layers.{layer_idx}.post_attention_layernorm.bias"

    return gguf_name


def _map_tensor_name(gguf_name: str, arch: str) -> str:
    """Map a GGUF tensor name to HuggingFace format based on architecture."""
    arch_lower = arch.lower()
    
    # Gemma4 has different tensor naming conventions (MoE)
    if arch_lower in ("gemma4", "gemma3"):
        return _map_gemma4_tensor_name(gguf_name)
    
    # Most other architectures follow llama naming
    return _map_llama_tensor_name(gguf_name)


# ---------------------------------------------------------------------------
# Tokenizer extraction
# ---------------------------------------------------------------------------


def extract_tokenizer(reader: GGUFReader, output_dir: Path) -> None:
    """Extract tokenizer from GGUF metadata and save standard files."""
    model_type = get_metadata_str(reader, "tokenizer.ggml.model") or "bpe"
    bos_id = get_metadata_int(reader, "tokenizer.ggml.bos_token_id") or 1
    eos_id = get_metadata_int(reader, "tokenizer.ggml.eos_token_id") or 2
    pad_id = get_metadata_int(reader, "tokenizer.ggml.padding_token_id") or 0

    tokens = get_metadata_array_str(reader, "tokenizer.ggml.tokens")
    token_types = get_metadata_array_int(reader, "tokenizer.ggml.token_type")
    merges = get_metadata_array_str(reader, "tokenizer.ggml.merges")
    scores = [
        float(s)
        for s in get_metadata_array_str(reader, "tokenizer.ggml.scores")
    ] if reader.get_field("tokenizer.ggml.scores") else []

    if not tokens:
        print("  ⚠ No tokenizer tokens found in GGUF — creating minimal tokenizer")
        tokens = ["<unk>", "<s>", "</s>", "<pad>"]
        token_types = [0, 3, 3, 3]
        bos_id, eos_id, pad_id = 1, 2, 3

    # --- Fix BOS/EOS for Qwen, DeepSeek, and similar families ---
    # These models use special tokens like <|endoftext|> (BOS) and <|im_end|> (EOS)
    # but GGUF files often omit bos_token_id / eos_token_id, or set them to wrong defaults (1, 2).
    #
    # Known special token names by role:
    SPECIAL_BOS_CANDIDATES = [
        "<|endoftext|>", "<s>", "<|begin_of_text|>", "<|startoftext|>",
    ]
    SPECIAL_EOS_CANDIDATES = [
        "<|im_end|>", "</s>", "<|end_of_text|>", "<|eot_id|>", "<|end|>",
    ]

    if tokens and (bos_id in (0, 1, 2, 3)):
        found = False
        for candidate in SPECIAL_BOS_CANDIDATES:
            if candidate in tokens:
                bos_id = tokens.index(candidate)
                print(f"  ✓ Fixed bos_token_id: {bos_id} ({candidate})")
                found = True
                break
        if not found:
            # Try uppercase versions
            for candidate in SPECIAL_BOS_CANDIDATES:
                for i, tok in enumerate(tokens):
                    if tok.upper() == candidate.upper():
                        bos_id = i
                        print(f"  ✓ Fixed bos_token_id: {bos_id} ({tok})")
                        found = True
                        break
                if found:
                    break

    if tokens and (eos_id in (0, 1, 2, 3)):
        found = False
        for candidate in SPECIAL_EOS_CANDIDATES:
            if candidate in tokens:
                eos_id = tokens.index(candidate)
                print(f"  ✓ Fixed eos_token_id: {eos_id} ({candidate})")
                found = True
                break
        if not found:
            for candidate in SPECIAL_EOS_CANDIDATES:
                for i, tok in enumerate(tokens):
                    if tok.upper() == candidate.upper():
                        eos_id = i
                        print(f"  ✓ Fixed eos_token_id: {eos_id} ({tok})")
                        found = True
                        break
                if found:
                    break

    vocab_size = len(tokens)
    print(f"  Extracted tokenizer: {vocab_size} tokens, model={model_type}")

    # --- tokenizer_config.json ---
    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": tokens[bos_id] if bos_id < vocab_size else "<s>",
        "eos_token": tokens[eos_id] if eos_id < vocab_size else "</s>",
        "unk_token": tokens[0] if tokens else "<unk>",
        "pad_token": tokens[pad_id] if pad_id < vocab_size else "<pad>",
        "model_max_length": 131072,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": False,
    }

    if model_type == "llama" or model_type == "bpe":
        tokenizer_config.update(
            {
                "model_type": "bpe",
                "tokenizer_class": "LlamaTokenizerFast"
                if "llama" in model_type
                else "PreTrainedTokenizerFast",
            }
        )

    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    print("  ✓ Saved tokenizer_config.json")

    # --- special_tokens_map.json ---
    special_tokens = {
        "bos_token": tokens[bos_id] if bos_id < vocab_size else "<s>",
        "eos_token": tokens[eos_id] if eos_id < vocab_size else "</s>",
        "unk_token": tokens[0] if tokens else "<unk>",
    }
    if pad_id < vocab_size and tokens[pad_id]:
        special_tokens["pad_token"] = tokens[pad_id]

    with open(output_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens, f, indent=2, ensure_ascii=False)
    print("  ✓ Saved special_tokens_map.json")

    # --- vocab.json (word → id mapping) ---
    vocab = {}
    for i, token in enumerate(tokens):
        if i < vocab_size:
            # Normalize token: GGUF stores bytes, HF expects strings
            if isinstance(token, str):
                vocab[token] = i

    with open(output_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved vocab.json ({len(vocab)} entries)")

    # --- merges.txt (for BPE tokenizers) ---
    if model_type in ("bpe", "gpt2") and merges:
        # GGUF stores merges with space characters
        merges_path = output_dir / "merges.txt"
        with open(merges_path, "w") as f:
            # No version header — HuggingFace GPT-2 tokenizer expects raw merges
            for merge in merges:
                if isinstance(merge, bytes):
                    merge = merge.decode("utf-8", errors="replace")
                f.write(merge + "\n")
        print(f"  ✓ Saved merges.txt ({len(merges)} merges)")

    # --- tokenizer.json (for fast tokenizers) ---
    tokenizer_json = _build_tokenizer_json(
        tokens, token_types, merges, scores, model_type, bos_id, eos_id, pad_id
    )
    if tokenizer_json:
        with open(output_dir / "tokenizer.json", "w") as f:
            json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
        print("  ✓ Saved tokenizer.json")


def _build_tokenizer_json(
    tokens: list[str],
    token_types: list[int],
    merges: list[str],
    scores: list[float],
    model_type: str,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> dict:
    """Build a complete tokenizer.json for HuggingFace tokenizers."""
    vocab = {}
    for i, token in enumerate(tokens):
        vocab[token] = i

    # Token type: 1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte

    added_tokens = []
    normal_tokens = []
    for i, token in enumerate(tokens):
        tt = token_types[i] if i < len(token_types) else 1
        # Control/special tokens go in added_tokens
        if tt in (3,) or i in (bos_id, eos_id, pad_id):
            special = i in (bos_id, eos_id, pad_id)
            added_tokens.append(
                {
                    "id": i,
                    "content": token,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": special,
                }
            )
        else:
            normal_tokens.append(token)

    # Build model block
    if model_type in ("bpe", "gpt2"):
        model_block = {
            "type": "BPE",
            "dropout": None,
            "unk_token": tokens[0] if tokens else "<unk>",
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges if merges else [],
        }
    elif model_type == "llama":
        model_block = {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "▁",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges if merges else [],
        }
    else:
        # Generic fallback
        model_block = {
            "type": "BPE",
            "vocab": vocab,
            "merges": merges if merges else [],
        }

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {
                        "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    },
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": False,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "model": model_block,
    }

    return tokenizer_json


# ---------------------------------------------------------------------------
# Weight extraction & conversion
# ---------------------------------------------------------------------------


def extract_and_convert_weights(
    reader: GGUFReader, arch: str, output_dir: Path, dtype: str = "float16"
) -> dict[str, np.ndarray]:
    """Extract GGUF tensors, dequantize, rename, and save as safetensors."""

    print(f"\n  Converting {len(reader.tensors)} tensors...")
    print(f"  Output dtype: {dtype}")

    np_dtype = np.float16 if dtype == "float16" else np.float32
    weights: dict[str, np.ndarray] = {}
    all_keys: list[str] = []

    skipped = 0
    total_bytes_in = 0
    total_bytes_out = 0
    shard_idx = 1
    current_shard_bytes = 0
    max_shard_bytes = int(4.5 * 1e9)  # 4.5 GB per shard max for safetensors

    # Progress bar
    pbar = tqdm(total=len(reader.tensors), desc="  Converting", unit="tensor")

    def _shard_filename(idx: int, total_final: int | None = None) -> str:
        """Generate shard filename. When total is unknown, use NNNNN placeholder."""
        if total_final is None:
            return f"model-{idx:05d}-of-NNNNN.safetensors"
        return f"model-{idx:05d}-of-{total_final:05d}.safetensors"

    def _flush_shard(
        shard_weights: dict[str, np.ndarray], shard_idx: int, total_shards: int | None
    ) -> int:
        """Write current shard to disk, clear dict, return bytes written. Returns byte count."""
        if not shard_weights:
            return 0
        path = output_dir / _shard_filename(shard_idx, total_shards)
        save_safetensors(shard_weights, str(path))
        n_bytes = sum(arr.nbytes for arr in shard_weights.values())
        shard_weights.clear()
        gc.collect()
        return n_bytes

    for i, tensor in enumerate(reader.tensors):
        gguf_name = tensor.name
        hf_name = _map_tensor_name(gguf_name, arch)
        
        # Skip tensors that are not needed for the target format
        if hf_name is None:
            skipped += 1
            continue
        
        qtype = tensor.tensor_type
        logical_shape = tuple(tensor.shape)
        n_bytes = tensor.n_bytes
        total_bytes_in += n_bytes

        # Progress indicator
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    [{i + 1}/{len(reader.tensors)}] Processing...")

        try:
            # Dequantize if needed
            qtype_val = int(qtype) if hasattr(qtype, "value") else int(qtype)
            raw_data = tensor.data

            if qtype_val == 0:  # F32
                arr = np.array(raw_data, dtype=np.float32).reshape(logical_shape)
                if dtype == "float16":
                    arr = arr.astype(np.float16)
                # F32/F16 tensors use GGUF layout [in_features, out_features],
                # need transpose to HF layout [out_features, in_features]
                if arr.ndim == 2:
                    arr = arr.T
            elif qtype_val == 1:  # F16
                arr = np.array(raw_data, dtype=np.float16).reshape(logical_shape)
                if arr.ndim == 2:
                    arr = arr.T
            elif qtype_val == 28:  # F64
                arr = np.array(raw_data, dtype=np.float64).reshape(logical_shape)
                if arr.ndim == 2:
                    arr = arr.T
                arr = arr.astype(np_dtype)
            elif qtype_val in (24, 25, 26, 27):  # I8, I16, I32, I64
                int_dtype_map = {24: np.int8, 25: np.int16, 26: np.int32, 27: np.int64}
                arr = np.array(raw_data, dtype=int_dtype_map.get(qtype_val, np.int32))
                arr = arr.reshape(logical_shape).astype(np_dtype)
                if arr.ndim == 2:
                    arr = arr.T
            else:
                # Quantized: gguf's dequantize already returns [out_features, in_features]
                # (HF layout), so don't reshape or transpose
                try:
                    ggml_qtype = (
                        qtype if isinstance(qtype, GGMLQuantizationType) else GGMLQuantizationType(qtype_val)
                    )
                    arr = dequantize(raw_data, ggml_qtype)
                    # dequantize already returns correct shape, no .reshape needed
                    arr = arr.astype(np_dtype)
                except Exception as e:
                    print(f"    ⚠ Failed to dequantize {gguf_name} ({qtype}): {e}")
                    skipped += 1
                    continue

            weights[hf_name] = arr
            all_keys.append(hf_name)
            total_bytes_out += arr.nbytes
            current_shard_bytes += arr.nbytes
            pbar.update(1)

            # Shard when approaching the per-shard byte limit
            if current_shard_bytes >= max_shard_bytes:
                n_bytes = _flush_shard(weights, shard_idx, None)
                print(f"\n    ✓ Shard {shard_idx}: {len(all_keys)} tensors so far, {n_bytes / 1e9:.2f} GB")
                shard_idx += 1
                current_shard_bytes = 0
                weights = {}

        except Exception as e:
            print(f"    ⚠ Error processing {gguf_name}: {e}")
            skipped += 1
            continue

    pbar.close()

    # --- Save remaining tensors as final shard ---
    if weights:
        n_bytes = _flush_shard(weights, shard_idx, None)
        if n_bytes:
            print(f"    ✓ Shard {shard_idx}: {len(all_keys)} tensors total, {n_bytes / 1e9:.2f} GB")
            shard_idx += 1

    if not all_keys:
        raise RuntimeError("No weights extracted!")

    # --- Rename shards with correct total-count filenames ---
    shard_files = sorted(
        output_dir.glob("model-*-of-NNNNN.safetensors"),
        key=lambda p: int(p.stem.split("-")[1])
    )
    total_shards = len(shard_files)
    weight_map: dict[str, str] = {}

    for i, old_path in enumerate(shard_files, 1):
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        new_path = output_dir / new_name
        old_path.rename(new_path)

        # Read keys from shard to build weight map
        with safe_open(str(new_path), framework="np") as f:
            for key in f.keys():
                weight_map[key] = new_name

    # --- Add tied weights for Gemma models ---
    if arch in ("gemma4", "gemma3", "gemma2"):
        # Get hidden_size for creating missing parameters
        hidden_size = get_metadata_int(reader, f"{arch}.embedding_length") or 2816
        
        # Gemma models use tied embeddings - lm_head = token_embd.weight.T
        # This is required by mlx_lm even though weights are tied at runtime
        # Note: language_model.model. prefix for embed_tokens
        if "language_model.model.embed_tokens.weight" in weight_map:
            # Get the shard that contains embed_tokens
            embed_shard = weight_map["language_model.model.embed_tokens.weight"]
            embed_path = output_dir / embed_shard
            
            # Read embed_tokens and create lm_head
            with safe_open(str(embed_path), framework="np") as f:
                embed_weights = f.get_tensor("language_model.model.embed_tokens.weight")
                lm_head_weights = embed_weights.T.astype(np.float16)
            
            # Save lm_head to a new shard
            lm_head_shard = {"language_model.lm_head.weight": lm_head_weights}
            new_total = total_shards + 1
            lm_shard_name = f"model-{new_total:05d}-of-{new_total:05d}.safetensors"
            save_safetensors(lm_head_shard, str(output_dir / lm_shard_name))
            
            # Update weight map
            weight_map["language_model.lm_head.weight"] = lm_shard_name
            all_keys.append("language_model.lm_head.weight")
            total_bytes_out += lm_head_weights.nbytes
            total_shards = new_total
            print(f"    ✓ Added tied lm_head.weight from embed_tokens ({lm_head_weights.nbytes / 1e6:.1f} MB)")
        
        # Add router scale parameters (initialized to ones, as in mlx_lm)
        num_experts = get_metadata_int(reader, f"{arch}.expert_count") or 128
        router_scale_shard: dict[str, np.ndarray] = {}
        
        # Create scale tensors for each layer (language_model.model. prefix)
        num_layers = get_metadata_int(reader, f"{arch}.block_count") or 30
        for layer_idx in range(num_layers):
            router_scale_shard[f"language_model.model.layers.{layer_idx}.router.scale"] = np.ones(int(hidden_size), dtype=np.float16)
            router_scale_shard[f"language_model.model.layers.{layer_idx}.router.per_expert_scale"] = np.ones(num_experts, dtype=np.float16)
            router_scale_shard[f"language_model.model.layers.{layer_idx}.layer_scalar"] = np.ones(1, dtype=np.float16)
        
        if router_scale_shard:
            new_total = total_shards + 1
            scale_shard_name = f"model-{new_total:05d}-of-{new_total:05d}.safetensors"
            save_safetensors(router_scale_shard, str(output_dir / scale_shard_name))
            
            for key in router_scale_shard:
                weight_map[key] = scale_shard_name
                all_keys.append(key)
            total_bytes_out += sum(arr.nbytes for arr in router_scale_shard.values())
            total_shards = new_total
            print(f"    ✓ Added {len(router_scale_shard)} router scale tensors ({sum(arr.nbytes for arr in router_scale_shard.values()) / 1e6:.1f} MB)")

    # --- Save index ---
    index_json = {
        "metadata": {"total_size": total_bytes_out},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index_json, f, indent=2)

    print(f"\n  ✓ Saved {len(all_keys)} weight tensors ({total_shards} shards)")
    print(f"    Total input:  {total_bytes_in / 1e9:.2f} GB (GGUF)")
    print(f"    Total output: {total_bytes_out / 1e9:.2f} GB (safetensors)")
    if skipped:
        print(f"    ⚠ Skipped {skipped} tensors due to errors")

    return weights


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(gguf_path: str, output_dir: str, dtype: str = "float16") -> bool:
    """Convert a GGUF file to MLX-compatible safetensors format."""

    gguf_file = Path(gguf_path)
    if not gguf_file.exists():
        print(f"❌ GGUF file not found: {gguf_path}")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = gguf_file.stem

    print("=" * 60)
    print("GGUF → MLX Converter v2.0")
    print(f"  Model: {model_name}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    # Step 1: Open GGUF
    print("\n[1/5] Reading GGUF file...")
    reader = GGUFReader(str(gguf_path))
    print(
        f"  ✓ GGUF version {reader.fields['GGUF.version'].contents()}, "
        f"{len(reader.tensors)} tensors, "
        f"{len(reader.fields)} metadata fields"
    )
    print(f"  File size: {gguf_file.stat().st_size / 1e9:.2f} GB")

    # Step 2: Detect architecture & build config
    print("\n[2/5] Detecting architecture...")
    arch = detect_architecture(reader)
    hf_type = ARCH_MAP.get(arch, arch)
    model_name_full = get_metadata_str(reader, "general.name") or model_name
    print(f"  Architecture: {arch} (HF type: {hf_type})")
    print(f"  Model name:   {model_name_full}")

    config = build_config(reader, arch)
    print(
        f"  Config: {config['num_hidden_layers']} layers, "
        f"{config['hidden_size']} hidden, "
        f"{config['num_attention_heads']} heads, "
        f"{config['vocab_size']} vocab"
    )
    if "num_experts" in config:
        print(f"  MoE: {config['num_experts']} experts, top-{config['num_experts_per_tok']}")

    try:
        file_type = reader.get_field("general.file_type")
        if file_type:
            ft = file_type.contents()
            ft_names = {
                1: "F16", 2: "Q4_0", 3: "Q4_1",
                7: "Q8_0", 10: "Q2_K", 12: "Q4_K",
                13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS",
                17: "IQ2_XS", 19: "IQ1_S", 20: "IQ4_NL",
            }
            print(f"  Source quantization: {ft_names.get(int(ft), f'unknown({ft})')}")
    except Exception:
        pass

    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  ✓ Saved config.json")

    # Step 3: Extract tokenizer
    print("\n[3/5] Extracting tokenizer...")
    extract_tokenizer(reader, output_path)

    # Step 4: Extract, dequantize, and convert weights
    print("\n[4/5] Extracting and converting weights...")
    try:
        extract_and_convert_weights(reader, arch, output_path, dtype)
    except Exception as e:
        print(f"❌ Weight extraction failed: {e}")
        return False

    # Step 5: Verify output
    print("\n[5/5] Finalizing...")
    # Index file already created by extract_and_convert_weights
    index_path = output_path / "model.safetensors.index.json"
    if not index_path.exists():
        print("❌ model.safetensors.index.json was not created")
        return False

    with open(index_path) as f:
        index_data = json.load(f)
    num_keys = len(index_data.get("weight_map", {}))
    print(f"  ✓ Index file: {num_keys} keys across {len(set(index_data['weight_map'].values()))} shards")

    # Summary
    print("\n" + "=" * 60)
    print("✅ Conversion complete!")
    print(f"  Output directory: {output_path}")
    print(f"  Architecture:     {arch} → {hf_type}")
    print("  Files generated:")
    for f_path in sorted(output_path.iterdir()):
        size = f_path.stat().st_size
        if size > 1_000_000_000:
            size_str = f"{size / 1e9:.2f} GB"
        elif size > 1_000_000:
            size_str = f"{size / 1e6:.1f} MB"
        elif size > 1000:
            size_str = f"{size / 1000:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"    - {f_path.name} ({size_str})")
    print("=" * 60)

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GGUF to MLX Converter — Convert GGUF models to MLX safetensors format"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input GGUF file path"
    )
    parser.add_argument(
        "--output", "-o", help="Output MLX directory"
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Output data type (default: float16)",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip weight extraction (metadata + tokenizer only, for inspection)",
    )

    args = parser.parse_args()

    # Auto-derive output directory from input filename if not specified
    if args.output is None:
        args.output = Path(args.input).stem + "-mlx"

    if args.skip_weights:
        # Just dump info
        reader = GGUFReader(args.input)
        print(f"Architecture: {detect_architecture(reader)}")
        print(f"Tensors: {len(reader.tensors)}")
        print(f"Fields: {len(reader.fields)}")
        for name in sorted(reader.fields.keys()):
            print(f"  {name}")
        return

    success = convert(args.input, args.output, args.dtype)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
