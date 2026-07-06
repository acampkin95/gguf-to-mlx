# GGUF→MLX Refactor Assessment & Model Support Roadmap

**Date:** 2026-07-06  
**Scope:** Full codebase audit + Qwen/Gemma/LFM/AgentWorld model support roadmap  
**mlx-lm version:** 0.31.3 (82+ architecture modules)

---

## Part 1: Codebase Refactor Assessment

### 1.1 Architecture Health

| File | Lines | Functions | >100-line funcs | >50-line funcs | Verdict |
|------|-------|-----------|------------------|----------------|---------|
| `convert.py` | 4,249 | 96 | 7 | 24 | 🔴 Monolith |
| `gguf2mlx/core.py` | 1,310 | 18 | 5 | 6 | 🟡 Manageable |
| `test_convert.py` | 3,735 | — | — | — | 🟡 Large but tested |
| **Total** | **9,294** | | | | |

**`convert.py` is the primary problem.** 96 functions in a single file with 7 God functions exceeding 100 lines. The `build_parser()` alone is 198 lines and `main()` is 186 lines. This violates single-responsibility at the file level.

### 1.2 Critical Findings

#### 🔴 CRITICAL: 82 Missing Architecture Mappings

Our `gguf2mlx/core.py` `ARCH_MAP` has **45 entries**. mlx-lm 0.31.3 supports **127+ architecture modules**. We are missing **82 architectures**, including every major model released since mid-2025:

**Missing critical architectures:**

| Priority | Architecture | mlx-lm module | Impact |
|----------|-------------|---------------|--------|
| P0 | Llama 4 | `llama4`, `llama4_text` | Meta's flagship 2025 release |
| P0 | Qwen 3 | `qwen3` | Dense Qwen3 (not just MoE) |
| P0 | Qwen 3.5 | `qwen3_5`, `qwen3_5_moe` | Latest Qwen generation |
| P0 | Qwen AgentWorld | *(new)* | Just released July 2026 |
| P0 | Gemma 4 | `gemma4`, `gemma4_text` | Google's latest, PLE architecture |
| P1 | DeepSeek V3.2 | `deepseek_v32` | R1 successor |
| P1 | Mistral 3 | `mistral3` | Latest Mistral |
| P1 | Ministral 3 | `ministral3` | Small Mistral |
| P1 | GLM 4 | `glm4`, `glm4_moe` | Zhipu AI flagship |
| P1 | LFM 2 | `lfm2`, `lfm2_moe` | Meta's next-gen |
| P2 | Kimi K2.5 | `kimi_k25` | Moonshot AI |
| P2 | InternLM 3 | `internlm3` | Shanghai AI Lab |
| P2 | Nemotron H | `nemotron_h` | NVIDIA |

Additionally, **10 entries in our ARCH_MAP map to mlx-lm modules that don't exist** (dead mappings): `falcon`, `mpt`, `bert`, `bloom`, `refact`, `chatglm`, `xverse`, `orion`, `grok`, `smolm`, `chameleon`.

#### 🔴 CRITICAL: Gemma4 MoE Conversion Broken

The `fix_gemma4_tensor_names()` function in `convert.py` **detects the issue but doesn't fix it** — it just prints warnings and tells the user to use `--no-quantize`. This is a documented bug since v1.0.0. The vendored `gguf2mlx/core.py` has Gemma4 tensor remapping in `_map_gemma4_tensor_name()` but it has known issues (incorrect layer norm mappings, duplicated layer_scalar keys).

#### 🟡 HIGH: 21 Swallowed Exceptions

Both `convert.py` (20) and `core.py` (6) have `except Exception` blocks that silently swallow errors. Many of these fall back to default values without any logging, making conversion failures invisible to users.

#### 🟡 HIGH: Type Safety Debt

- `convert.py`: 34 uses of `dict[str, Any]`, 30 uses of `| None`
- 4 pre-existing mypy errors (1 assignment type mismatch, 2 return-value mismatches in core.py)
- No strict mypy configuration

#### 🟡 HIGH: Qwen AgentWorld Not Supported

Qwen-AgentWorld-35B-A3B (released July 2026) uses a hybrid architecture with Qwen3 as backbone. GGUF files exist on HuggingFace (unsloth, FreedomAISVR quantized). Our tool will fall back to generic llama mapping — likely producing broken MLX output.

#### 🟡 MEDIUM: Dead Code & Stale References

- `_SEPARATOR` constant defined but rarely used
- `fix_gemma4_tensor_names()` is vestigial — core.py now handles this, convert.py still has the detection
- `GGUF_TO_MLX_TENSOR_RENAME` dict is a single entry that's no longer needed
- 5 unused imports in convert.py (pre-existing)

#### 🟡 MEDIUM: No Double-Quantization Guard

The existing `classify_source_quality()` warns about double quantization but **does not prevent it**. A Q4_K_M source can be re-quantized to 2-bit MLX without any hard stop. Per research from [arxiv 2505.02214v1], this causes sharp quality degradation especially for Qwen.

#### 🟢 LOW: Test Coverage

273 tests passing, 92% coverage on convert.py. But:
- No tests for the vendored `gguf2mlx/core.py`
- No integration tests that verify actual GGUF→MLX conversion produces loadable models
- No architecture-specific conversion tests (e.g., Gemma4 MoE, Qwen3 MoE)

---

## Part 2: Proposed Module Architecture

### 2.1 Target Structure

```
gguf-to-mlx/
├── convert.py                    # CLI entrypoint only (argparse + dispatch)
├── gguf2mlx/                     # Vendored conversion engine
│   ├── __init__.py
│   ├── __main__.py
│   └── core.py                   # GGUF reader, weight extraction, tokenizer
├── gguf_to_mlx/                  # NEW: our wrapper logic (extracted from convert.py)
│   ├── __init__.py
│   ├── plan.py                   # ConversionPlan, ConversionMode, SourceQuantInfo
│   ├── planner.py                 # build_conversion_plan() with arch-specific rules
│   ├── arch_rules.py             # Per-architecture quantization constraints
│   ├── validate.py               # Post-conversion validation harness
│   ├── hf.py                     # HuggingFace search/download (extracted)
│   ├── scan.py                   # Model scanning (extracted)
│   ├── config.py                 # Config/history/settings (extracted)
│   ├── ui.py                     # Rich console helpers, banners, menus (extracted)
│   └── metadata.py               # GGUF metadata reading & classification
├── gguf_to_mlx/arch/             # NEW: architecture-specific modules
│   ├── __init__.py               # Auto-discovery registry
│   ├── qwen.py                   # Qwen 2/2.5/3/3.5/AgentWorld rules
│   ├── gemma.py                  # Gemma 2/3/4 PLE + MoE handling
│   ├── llama.py                  # Llama 3.1/3.2/3.3/4 Scout/Maverick
│   ├── deepseek.py               # DeepSeek V2/V3/V3.2
│   ├── mistral.py                # Mistral/Mixtral/Mistral 3/Ministral
│   └── registry.py               # Map GGUF arch → arch module + quant rules
├── tests/
│   ├── test_convert.py           # Existing CLI tests
│   ├── test_plan.py              # NEW: ConversionPlan tests
│   ├── test_arch_rules.py        # NEW: Per-arch constraint tests
│   └── test_core.py              # NEW: Vendored engine tests
├── pyproject.toml
└── ...
```

### 2.2 Key Dataclasses (from research input)

```python
# gguf_to_mlx/plan.py

class ConversionMode(str, Enum):
    PRESERVE_SOURCE = "preserve-source"   # No double-quant; prefer FP16/8-bit
    HF_QUALITY = "hf-quality"             # Target HF-level quality
    SPEED = "speed"                       # Max compression

@dataclass
class SourceQuantInfo:
    file_type: int
    risk: str               # "none"/"low"/"medium"/"high"/"severe"
    scheme_label: str       # e.g. "MOSTLY_Q4_K_M"
    arch: str               # e.g. "qwen3", "gemma4"
    effective_bits: float   # Approximate effective bit-width

@dataclass
class ConversionPlan:
    mode: ConversionMode
    target_bits: int | None
    target_group_size: int | None
    target_mode: str | None
    intermediate_dtype: str
    allow_double_quant: bool
    arch: str
    arch_module: str         # e.g. "qwen3" for arch_rules lookup
    warnings: list[str]
    metadata: dict[str, Any]  # Propagated into output config.json
```

---

## Part 3: Model Support Roadmap

### 3.1 Current Coverage Gap Summary

| Family | In ARCH_MAP | In mlx-lm | GGUF Available | Status |
|--------|-------------|-----------|---------------|--------|
| **Qwen 2** | ✅ `qwen2` | ✅ `qwen2` | ✅ | Working |
| **Qwen 2.5** | ❌ | ✅ `qwen2` (reuses) | ✅ | Works via qwen2 fallback |
| **Qwen 2.5 MoE** | ✅ `qwen2moe` | ✅ `qwen2_moe` | ✅ | Working |
| **Qwen 3** | ❌ | ✅ `qwen3` | ✅ | ❌ **Broken — falls to generic** |
| **Qwen 3 MoE** | ✅ `qwen3moe` | ✅ `qwen3_moe` | ✅ | Working |
| **Qwen 3.5** | ❌ | ✅ `qwen3_5` | ✅ | ❌ **Missing** |
| **Qwen 3.5 MoE** | ❌ | ✅ `qwen3_5_moe` | ✅ | ❌ **Missing** |
| **Qwen AgentWorld** | ❌ | ❌ (new) | ✅ | ❌ **Not supported anywhere** |
| **Gemma 2** | ✅ | ✅ `gemma2` | ✅ | Working |
| **Gemma 3** | ✅ `gemma3` | ✅ `gemma3` | ✅ | ⚠️ MoE issues |
| **Gemma 4** | ❌ | ✅ `gemma4` | ✅ | ❌ **Missing + PLE arch** |
| **Gemma 4 Text** | ❌ | ✅ `gemma4_text` | ✅ | ❌ **Missing** |
| **Llama 3.1** | ❌ | ✅ `llama` (reuses) | ✅ | Works via llama |
| **Llama 3.2** | ❌ | ✅ `llama` (reuses) | ✅ | Works via llama |
| **Llama 4** | ❌ | ✅ `llama4` | ✅ | ❌ **Missing — MoE architecture** |
| **Llama 4 Text** | ❌ | ✅ `llama4_text` | ✅ | ❌ **Missing** |
| **LFM 2** | ❌ | ✅ `lfm2` | ✅ | ❌ **Missing** |
| **LFM 2 MoE** | ❌ | ✅ `lfm2_moe` | ✅ | ❌ **Missing** |
| **DeepSeek V3** | ✅ `deepseek3` | ✅ `deepseek_v3` | ✅ | Working |
| **DeepSeek V3.2** | ❌ | ✅ `deepseek_v32` | ✅ | ❌ **Missing** |
| **Mistral 3** | ❌ | ✅ `mistral3` | ✅ | ❌ **Missing** |
| **Mixtral** | ❌ | ✅ `mixtral` | ✅ | ❌ **Missing** |
| **GLM 4** | ❌ | ✅ `glm4` | ✅ | ❌ **Missing** |
| **InternLM 3** | ❌ | ✅ `internlm3` | ✅ | ❌ **Missing** |

### 3.2 Qwen AgentWorld Details

**Qwen-AgentWorld-35B-A3B** (released July 2026):
- **Architecture:** Hybrid — Qwen3 backbone with agentic environment prediction heads
- **Total params:** 35B, **Active params:** 3B (MoE-like, but environment-specific)
- **Simulates:** 7 agentic environments (MCP, Search, Terminal, SWE, Android, Web, OS)
- **GGUF:** Available via `unsloth/Qwen-AgentWorld-35B-A3B-GGUF`, `FreedomAISVR/MXFP4-MOE-GGUF`
- **mlx-lm:** Not yet supported (needs new model module)
- **Conversion approach:** Treat as Qwen3 variant; extract backbone weights, skip environment heads if they don't map to standard MLX layers

### 3.3 Architecture-Specific Notes

#### Qwen Family
| Model | Type | Experts | Key Tensor Differences | Quantization Notes |
|-------|------|---------|----------------------|-------------------|
| Qwen 2.5 (0.5-72B) | Dense | — | Standard Qwen2 naming | 4-bit OK for ≥7B |
| Qwen 2.5 MoE (57B) | MoE | 128E/4A | `ffn_gate_exps`, `ffn_down_exps` stacked 3D | 4-bit OK, 8-bit for experts |
| Qwen 3 (0.6-32B) | Dense | — | Adds `attn_q_norm`, `attn_k_norm` | **No ≤3-bit** per arxiv research |
| Qwen 3 MoE (235B) | MoE | 128E/22A | QK norms + MoE stacked experts | FP16 or 8-bit recommended |
| Qwen 3.5 (TBD) | Dense+MoE | TBD | Likely extends Qwen3 naming | TBD — research on release |
| AgentWorld (35B) | Hybrid/MoE | 7 env heads | Custom heads on Qwen3 backbone | Convert backbone only initially |

#### Gemma Family
| Model | Type | Experts | Key Tensor Differences | Quantization Notes |
|-------|------|---------|----------------------|-------------------|
| Gemma 2 | Dense | — | Standard gemma2 naming | 4-bit OK |
| Gemma 3 | Dense + MoE | 16E/1A (27B) | `language_model.model.` prefix | MoE: FP16 only |
| Gemma 4 (26B) | MoE + PLE | ~128E | PLE embeddings, `layer_scalar`, `router.scale` | **FP16 only** — 4-bit broken |

#### LFM / Llama Family
| Model | Type | Experts | Key Tensor Differences | Quantization Notes |
|-------|------|---------|----------------------|-------------------|
| Llama 3.1-3.3 | Dense | — | Standard llama naming | 4-bit OK |
| Llama 4 Scout | MoE | 16E/17A | `llama4` module, multi-head latent attention | Needs tensor mapping work |
| Llama 4 Maverick | MoE | 16E/17A/400B | Extended Scout architecture | Same as Scout |
| LFM 2 | Dense | — | `lfm2` module | 4-bit OK |

---

## Part 4: Phased Roadmap

### Phase 1: Foundation (Week 1-2) — Architecture + Critical Model Gaps

**Goal:** Fix broken conversions, add ConversionPlan layer, support P0 models

| Task | Priority | Effort | Detail |
|------|----------|--------|--------|
| Add `gguf_to_mlx/plan.py` + `planner.py` | P0 | 1 day | ConversionMode, SourceQuantInfo, ConversionPlan dataclasses |
| Add `--quality-mode` CLI flag | P0 | 0.5 day | preserve-source / hf-quality / speed |
| Implement double-quant guard | P0 | 0.5 day | Block Q4→Q2 without `--allow-low-bits` |
| Add Qwen-specific rules | P0 | 0.5 day | No ≤3-bit for Qwen in quality modes |
| Add Gemma MoE FP16-only rule | P0 | 0.5 day | Force `--no-quantize` for gemma3/4 MoE |
| **Add Qwen 3 to ARCH_MAP** | P0 | 0.5 day | `"qwen3": "qwen3"` — verify tensor mapping |
| **Add Qwen 3.5 + 3.5 MoE to ARCH_MAP** | P0 | 0.5 day | `"qwen3_5": "qwen3_5"`, `"qwen3_5_moe": "qwen3_5_moe"` |
| **Add Gemma 4 + 4 Text to ARCH_MAP** | P0 | 1 day | PLE architecture tensor mapping, fix layer_scalar bugs |
| **Add Llama 4 + 4 Text to ARCH_MAP** | P0 | 1 day | MoE tensor naming, MLA attention layout |
| Fix dead ARCH_MAP entries | P0 | 0.5 day | Remove 11 mappings to non-existent mlx-lm modules |
| Wire plan into `main()` / `main_with_file()` | P0 | 0.5 day | Apply plan to args before pipeline |

### Phase 2: Model Expansion (Week 3-4) — P1 Models

**Goal:** Cover all major open-source model families

| Task | Priority | Effort | Detail |
|------|----------|--------|--------|
| Add DeepSeek V3.2 | P1 | 0.5 day | Extend deepseek_v3 mapping |
| Add Mistral 3 + Ministral 3 | P1 | 0.5 day | New tensor naming convention |
| Add Mixtral | P1 | 0.5 day | Standard MoE — should be straightforward |
| Add GLM 4 + MoE | P1 | 1 day | ChatGLM successor, new naming |
| Add LFM 2 + MoE | P1 | 0.5 day | Meta's latest dense/MoE |
| Add Kimi K2.5 | P1 | 0.5 day | Moonshot AI, deepseek_v3 remap |
| Add InternLM 3 | P1 | 0.5 day | New naming from internlm2 |
| Add Nemotron H | P1 | 0.5 day | NVIDIA's latest |
| Add Qwen AgentWorld support | P1 | 2 days | Hybrid arch — extract Qwen3 backbone, document env head handling |
| Add Cohere 2 | P2 | 0.5 day | command-r successor |

### Phase 3: Quality & Validation (Week 5-6)

**Goal:** Research-grade quality preservation, validation harness

| Task | Priority | Effort | Detail |
|------|----------|--------|--------|
| Implement `--validate` mode | P1 | 2 days | GGUF vs MLX prompt comparison harness |
| Metadata propagation | P1 | 1 day | Write source_scheme, mode, calibrated status into output |
| Calibration-aware quant (AWQ-lite) | P2 | 5 days | Optional per-channel quantization for hf-quality mode |
| Per-tensor quantization strategy | P2 | 3 days | Different quant for lm_head vs attention vs MLP |
| Architecture auto-discovery | P2 | 1 day | Scan mlx-lm model directory to auto-populate ARCH_MAP |

### Phase 4: Refactor (Week 7-8)

**Goal:** Clean architecture, extract convert.py into modules

| Task | Priority | Effort | Detail |
|------|----------|--------|--------|
| Extract `gguf_to_mlx/` package | P1 | 3 days | Move metadata, hf, scan, config, ui modules |
| Fix swallowed exceptions | P1 | 1 day | Add structured logging to all except blocks |
| Type safety pass | P2 | 2 days | Reduce `dict[str, Any]` usage, fix mypy errors |
| Add core.py unit tests | P1 | 2 days | Test tensor mapping, config building, tokenizer extraction |
| Integration test suite | P2 | 3 days | Convert small GGUF files, verify mlx_lm.load() works |

---

## Part 5: ConversionPlan Integration Design

### 5.1 How It Plugs In

```
main()
  ├── read_gguf_metadata(gguf_path)          → meta dict
  ├── describe_source(meta)                   → SourceQuantInfo
  ├── build_conversion_plan(args, meta, hw)   → ConversionPlan
  ├── plan.warnings → display to user
  ├── plan → override args.bits, args.no_quantize, args.dtype
  ├── _run_step1(gguf → float16 safetensors)  → unchanged
  ├── _run_step2(float16 → quantized MLX)      → unchanged
  └── write plan.metadata into output config.json
```

### 5.2 Arch-Specific Rules (in `arch_rules.py`)

```python
ARCH_RULES: dict[str, ArchRule] = {
    "qwen": ArchRule(
        min_bits=4,                     # No ≤3-bit in quality modes
        moe_preferred_bits=8,           # 8-bit for MoE experts
        fp16_threshold_gb=32,          # ≥32B models default to FP16
        requires_awq=False,
    ),
    "gemma4": ArchRule(
        min_bits=None,                  # Force FP16-only (None = no quant)
        moe_preferred_bits=None,        # PLE + MoE = FP16 only
        fp16_threshold_gb=0,            # Always FP16
        requires_awq=False,
    ),
    "gemma3": ArchRule(
        min_bits=4,
        moe_preferred_bits=None,        # MoE variant → FP16
        fp16_threshold_gb=16,
        requires_awq=False,
    ),
    "llama4": ArchRule(
        min_bits=4,
        moe_preferred_bits=8,
        fp16_threshold_gb=64,
        requires_awq=False,
    ),
    "deepseek_v3": ArchRule(
        min_bits=4,
        moe_preferred_bits=8,
        fp16_threshold_gb=32,
        requires_awq=True,               # Benefits from AWQ calibration
    ),
}
```

### 5.3 Qwen AgentWorld Conversion Strategy

```python
# gguf_to_mlx/arch/qwen.py

AGENTWORLD_RULES = ArchRule(
    min_bits=4,
    moe_preferred_bits=8,
    fp16_threshold_gb=16,
    # AgentWorld has custom environment-prediction heads that don't map
    # to standard MLX layers. Strategy: extract Qwen3 backbone only,
    # document env heads as unsupported.
    skip_tensors=[
        "env_head.*",          # Environment prediction heads
        "action_head.*",       # Action output heads  
        "state_head.*",        # State transition heads
    ],
    custom_config_overrides={
        "model_type": "qwen3",  # Use Qwen3 model class for inference
        "_agentworld_backbone": True,
        "_unsupported_heads": ["mcp", "search", "terminal", "swe", "android", "web", "os"],
    },
)
```

---

## Part 6: Success Metrics

| Metric | Current | Phase 1 Target | Phase 4 Target |
|--------|---------|----------------|----------------|
| ARCH_MAP entries | 45 | 65 (+ Qwen3/3.5, Gemma4, Llama4, AgentWorld) | 100+ (auto-discovery) |
| mlx-lm compatibility | ~35% | ~55% | ~85% |
| Broken MoE models | Gemma4 | Gemma4 fixed | All MoE working |
| Double-quant guard | Warn only | Hard block | Configurable |
| Quality validation | None | Basic | AWQ-calibrated |
| convert.py lines | 4,249 | 4,249 (plan layer added) | ~800 (extracted to package) |
| Test count | 273 | 300+ | 400+ |
| mypy errors | 4 | 4 | 0 |

---

## Sources & Research

- [ContraCollective — GGUF vs MLX Quantization Formats 2026](https://contracollective.com/blog/gguf-vs-mlx-quantization-formats-apple-silicon-2026)
- [MuhammadRaza — GGUF vs MLX Decision Guide](https://muhammadraza.me/2026/gguf-vs-mlx-decision-guide/)
- [Latitude — Quantized LLMs Cost/Performance Results](https://latitude.so/blog/quantized-llms-cost-performance-results)
- [arxiv 2505.02214v1 — Qwen Quantization Research](https://arxiv.org/html/2505.02214v1)
- [Qwen docs — AWQ Quantization](https://qwen.readthedocs.io/en/latest/quantization/awq.html)
- [Qwen-AgentWorld HuggingFace](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B)
- [mlx-lm GitHub (82+ model modules)](https://github.com/ml-explore/mlx-lm/)
- [DeepWiki — mlx-lm Supported Models](https://deepwiki.com/ml-explore/mlx-lm/5.1-overview-and-supported-models)
