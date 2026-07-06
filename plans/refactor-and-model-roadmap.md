# GGUF→MLX Refactor Assessment & Model Support Roadmap

**Date:** 2026-07-06  
**Updated:** 2026-07-06 — gap review: dead count corrected (11→16), MODEL_REMAPPING documented, VLM scope, unified dataclasses, phase reordering, test strategy, rollback plan  
**Scope:** Full codebase audit + Qwen/Gemma/LFM/AgentWorld model support roadmap  
**mlx-lm version:** 0.31.3 (107 real model modules, 11 utility/shared modules)

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

#### 🔴 CRITICAL: 16 Dead ARCH_MAP Entries

Our `gguf2mlx/core.py` `ARCH_MAP` has **45 entries**. After applying mlx-lm's `MODEL_REMAPPING` (which remaps e.g. `mistral→llama`, `kimi_k2→deepseek_v3`), **16 entries map to modules that don't exist in mlx-lm 0.31.3**:  
`falcon`, `mpt`, `bert`, `bloom`, `refact`, `chatglm`, `baichuan`, `xverse`, `orion`, `codeshell`, `t5`, `jais`, `grok-1→grok`, `arctic`, `smolm`, `chameleon`.

**Replacement candidates** (where a successor module exists):  
| Dead Entry | Successor in mlx-lm | Action |  
|------------|-------------------|--------|  
| `chatglm→chatglm` | `glm`, `glm4`, `glm4_moe` | Remap to `glm4` for recent models |  
| `baichuan→baichuan` | `baichuan_m1` | Remap to `baichuan_m1` |  
| `smolm→smolm` | `smollm3` | Remap to `smollm3` |  
| `falcon→falcon` | `falcon_h1` | Remap to `falcon_h1` |  
| `mistral→mistral` | remapped by mlx-lm to `llama` | **Works** — keep as-is |  
| mpt, bert, bloom, refact, xverse, orion, codeshell, t5, jais, grok, arctic, chameleon | No replacement | Remove from ARCH_MAP |  

> **Note:** `mistral` is NOT dead — mlx-lm's `MODEL_REMAPPING` remaps `"mistral"` → `"llama"` transparently, so mistral GGUF files convert correctly via the llama module.

#### 🔴 CRITICAL: 80 Missing Architecture Mappings

mlx-lm 0.31.3 has **107 real model modules** (excluding 11 utility/shared modules like `base`, `cache`, `pipeline`, `activations`, etc.). Our ARCH_MAP covers **29 working entries**. We are missing **80 architectures**:

| Priority | Count | Key Architectures |
|----------|-------|-------------------|
| **P0** | 7 | `qwen3`, `qwen3_5`, `qwen3_5_moe`, `gemma4`, `gemma4_text`, `llama4`, `llama4_text` |
| **P1** | 16 | `deepseek_v32`, `mistral3`, `ministral3`, `mixtral`, `glm4`/`glm4_moe`/`glm4_moe_lite`, `lfm2`/`lfm2_moe`, `kimi_k25`, `internlm3`, `nemotron_h`, `cohere2`, `qwen2_vl`, `gemma3_text`, `gemma3n` |
| **P2** | 49 | `falcon_h1`, `hunyuan`, `jamba`, `olmo3`, `olmoe`, `mamba`, `mamba2`, `rwkv7`, `exaone_moe`/`exaone4`, `phi3small`, `phimoe`, `pixtral`, `plamo2`, `smollm3`, `qwen3_next`, `telechat3`, +30 more |

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

**Missing from roadmap: GGUF file_type enum.** `_resolve_dtype()` checks `file_type==0` (F32) and `file_type==26` (BF16) but GGUF defines many more: 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1, 8=Q8_0, 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K. These are critical for `classify_source_quality()` and the double-quant guard. Must be documented and referenced in the planner.

#### 🟢 LOW: Test Coverage

273 tests passing, 92% coverage on convert.py. But:
- No tests for the vendored `gguf2mlx/core.py`
- No integration tests that verify actual GGUF→MLX conversion produces loadable models
- No architecture-specific conversion tests (e.g., Gemma4 MoE, Qwen3 MoE)
- **No mock GGUF metadata tests** — can't test new ARCH_MAP entries without 10-50GB files. Must create lightweight mock GGUF fixtures (minimal metadata + tiny weight tensors) to validate mapping logic without real models.

#### 🟢 LOW: Vision-Language Models (Out of Scope)

mlx-lm supports several vision-language architectures: `qwen2_vl`, `qwen3_vl`, `qwen3_vl_moe`, `kimi_vl`, `lfm2-vl`, `pixtral`. These have additional vision encoder tensors (image embeddings, vision projection layers) that complicate GGUF→MLX conversion significantly. **VLMs are explicitly out of scope for this roadmap.** Rationale:
- VLM GGUF files are extremely large (often 50-100GB+)
- Vision encoder tensors require separate quantization strategies
- mlx-lm's VLM modules have different weight layout expectations
- Worth revisiting as a separate effort after core text-only support is solid

#### 🟢 LOW: mlx-lm MODEL_REMAPPING Interaction

mlx-lm 0.31.3 has a `MODEL_REMAPPING` dict in `utils.py` that transparently remaps certain `model_type` values before module lookup:

```python
MODEL_REMAPPING = {
    "mistral": "llama",        # Mistral 7B uses llama module
    "llava": "mistral3",       # LLaVA uses mistral3 module
    "phi-msft": "phixtral",    # Microsoft Phi → PhiXtral
    "falcon_mamba": "mamba",    # Falcon Mamba → Mamba module
    "joyai_llm_flash": "deepseek_v3",  # JoyAI → DeepSeek V3
    "kimi_k2": "deepseek_v3",  # Kimi K2 → DeepSeek V3
    "qwen2_5_vl": "qwen2_vl",  # Qwen 2.5 VL → Qwen 2 VL
    "minimax_m2": "minimax",
    "iquestcoder": "llama",
}
```

Our `ARCH_MAP` should target the **pre-remap** model_type (i.e., the value that mlx-lm's `load_model()` receives). Since `gguf2mlx/core.py` writes `config["model_type"]` from `ARCH_MAP` value, and `mlx_lm.utils.load_model()` then applies `MODEL_REMAPPING`, our ARCH_MAP values must match what `load_model()` expects *before* remapping. Currently this works correctly — our mapping produces `"mistral"` and mlx-lm remaps it to `"llama"`. But new entries must account for this chain.

#### 🟢 LOW: Unknown Architecture Handling

`gguf2mlx/core.py` `detect_architecture()` returns `"llama"` as safe default when the GGUF metadata doesn't contain a recognized architecture. This means truly unknown architectures silently produce llama-format output that's likely broken. The roadmap should add explicit unknown-arch warnings and a `--arch-override` flag for manual specification.

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
    file_type: int              # GGUF file_type enum (0=F32, 2=Q4_0, ..., 26=BF16)
    risk: str                    # "none"/"low"/"medium"/"high"/"severe"
    scheme_label: str            # e.g. "MOSTLY_Q4_K_M"
    arch: str                    # e.g. "qwen3", "gemma4"
    effective_bits: float        # Approximate effective bit-width

class ArchRule:
    """Per-architecture quantization constraints."""
    min_bits: int | None           # None = force FP16-only (no quant)
    moe_preferred_bits: int | None # None = FP16-only for MoE experts
    fp16_threshold_gb: float       # Models above this get FP16 default
    requires_awq: bool            # Benefits from AWQ calibration
    skip_tensors: list[str]        # Globs of tensors to skip during conversion
    custom_config_overrides: dict[str, Any]  # Extra config.json fields

@dataclass
class ConversionPlan:
    # Core conversion params
    mode: ConversionMode
    target_bits: int | None
    target_group_size: int | None
    target_mode: str | None
    intermediate_dtype: str        # float16 or bfloat16 (chip-aware)
    allow_double_quant: bool
    # Architecture
    arch: str                     # GGUF architecture string (e.g. "qwen3")
    arch_module: str               # mlx-lm module name (e.g. "qwen3")
    arch_rule: ArchRule | None    # Applied arch-specific constraints
    # Hardware awareness
    bandwidth_tier: str            # "low"/"mid"/"high"/"ultra"
    chip_gen: int                 # 1-5, for bf16/fp16 decisions
    # Diagnostics
    warnings: list[str]
    metadata: dict[str, Any]       # Propagated into output config.json
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

**Design principle:** Extract the package shell first (Phase 0), then add features into the extracted modules (Phases 1–3). This avoids the double-movement anti-pattern of adding features to convert.py then refactoring them out.

### Phase 0: Package Shell Extraction (Week 1, first half) — "Move First"

**Goal:** Create `gguf_to_mlx/` package skeleton, extract utilities from convert.py, no behavior change

| Task | Priority | Effort | Detail |
|------|----------|--------|--------|
| Create `gguf_to_mlx/__init__.py` | P0 | 0.5 day | Package init with re-exports |
| Extract `gguf_to_mlx/metadata.py` | P0 | 0.5 day | `read_gguf_metadata()`, `classify_source_quality()` from convert.py |
| Extract `gguf_to_mlx/hardware.py` | P0 | 0.5 day | `detect_apple_silicon()`, `classify_bandwidth()`, `smart_defaults()` |
| Extract `gguf_to_mlx/ui.py` | P0 | 0.5 day | `info()`, `warn()`, `success()`, Rich console setup |
| Extract `gguf_to_mlx/hf.py` | P0 | 0.5 day | HuggingFace search/download helpers |
| Wire imports in convert.py | P0 | 0.5 day | Replace local functions with package imports, all 273 tests still pass |

### Phase 1: Foundation (Week 1 second half – Week 2) — Architecture + Critical Model Gaps

**Goal:** Fix broken conversions, add ConversionPlan layer, support P0 models

| Task | Priority | Effort | Detail |
|------|----------|--------|--------|
| Add `classify_bandwidth()` to `hardware.py` | P0 | 0.5 day | 4-tier system: low/mid/high/ultra |
| Update `_resolve_dtype()` with bf16/fp16 chip awareness | P0 | 0.5 day | M1/M2 → force fp16; M3+ → allow bf16 |
| Update `smart_defaults()` to use `bandwidth_tier` | P0 | 0.5 day | Replace ram_gb/tier with tier-based rules |
| Add `gguf_to_mlx/plan.py` + `planner.py` | P0 | 1 day | ConversionMode, SourceQuantInfo, ArchRule, ConversionPlan |
| Add `--quality-mode` CLI flag | P0 | 0.5 day | preserve-source / hf-quality / speed |
| Implement double-quant guard | P0 | 0.5 day | Block Q4→Q2 without `--allow-low-bits` |
| Add Qwen-specific rules | P0 | 0.5 day | No ≤3-bit for Qwen in quality modes |
| Add Gemma MoE FP16-only rule | P0 | 0.5 day | Force `--no-quantize` for gemma3/4 MoE |
| **Add Qwen 3 to ARCH_MAP** | P0 | 0.5 day | `"qwen3": "qwen3"` — verify tensor mapping |
| **Add Qwen 3.5 + 3.5 MoE to ARCH_MAP** | P0 | 0.5 day | `"qwen3_5": "qwen3_5"`, `"qwen3_5_moe": "qwen3_5_moe"` |
| **Add Gemma 4 + 4 Text to ARCH_MAP** | P0 | 1 day | PLE architecture tensor mapping, fix layer_scalar bugs |
| **Add Llama 4 + 4 Text to ARCH_MAP** | P0 | 1 day | MoE tensor naming, MLA attention layout |
| Fix dead ARCH_MAP entries | P0 | 0.5 day | Remove 16 dead mappings, remap 4 to successors |
| Wire plan into `main()` / `main_with_file()` | P0 | 0.5 day | Apply plan to args before pipeline |
| Add mock GGUF test fixtures | P0 | 1 day | Minimal metadata + tiny tensors for ARCH_MAP testing without real models |
| Add `--arch-override` CLI flag | P1 | 0.25 day | Manual arch specification for unknown GGUF files |

### Phase 2: Model Expansion (Week 3-4) — P1 Models

**Goal:** Cover all major open-source model families

| Task | Priority | Effort | Detail |
|------|----------|--------|--------|
| Add DeepSeek V3.2 | P1 | 0.5 day | Extend deepseek_v3 mapping |
| Add Mistral 3 + Ministral 3 | P1 | 0.5 day | New tensor naming convention |
| Add Mixtral | P1 | 0.5 day | Standard MoE — should be straightforward |
| Add GLM 4 + MoE + MoE Lite | P1 | 1 day | ChatGLM successor, new naming, `glm_moe_dsa` |
| Add LFM 2 + MoE | P1 | 0.5 day | Meta's latest dense/MoE |
| Add Kimi K2.5 | P1 | 0.5 day | Moonshot AI, deepseek_v3 remap |
| Add InternLM 3 | P1 | 0.5 day | New naming from internlm2 |
| Add Nemotron H | P1 | 0.5 day | NVIDIA's latest |
| Add Qwen AgentWorld support | P1 | 2 days | Hybrid arch — extract Qwen3 backbone, document env head handling |
| Add Cohere 2 | P1 | 0.5 day | command-r successor |
| Add Gemma 3 Text + Gemma 3n | P1 | 0.5 day | Text-only and nano variants |
| Add Qwen 2 VL | P1 | 0.5 day | Vision-language (minimal — document as partial support) |
| Add PhiXtral + Phi3Small | P2 | 0.5 day | Microsoft Phi mixtral variants |

### Phase 3: Quality & Validation (Week 5-6)

**Goal:** Research-grade quality preservation, validation harness

| Task | Priority | Effort | Detail |
|------|----------|--------|--------|
| Implement `--validate` mode | P1 | 2 days | GGUF vs MLX prompt comparison, 3 metrics, chip-aware budget |
| Generalize validation beyond Qwen | P1 | 0.5 day | Arch-agnostic prompt suite, not Qwen-only |
| Bandwidth-adaptive validation prompts | P2 | 0.5 day | Short prompts M1/M2, long-context on M4/M5 |
| `--validate-perplexity` MLX-only proxy | P2 | 1 day | Small held-out text shard, flag >20% perplexity delta |
| Metadata propagation | P1 | 1 day | Write source_scheme, mode, calibrated status into output |
| Calibration-aware quant (AWQ-lite) | P2 | 5 days | Optional per-channel quantization for hf-quality mode |
| Per-tensor quantization strategy | P2 | 3 days | Different quant for lm_head vs attention vs MLP |
| Architecture auto-discovery | P2 | 1 day | Scan mlx-lm model directory to auto-populate ARCH_MAP |

**Performance budget:** The planning stage (hardware detect + plan + warnings) must add **<5% overhead** to total conversion time. Validation runs are opt-in (`--validate`) so excluded from this budget.

### Phase 4: Quality Polish & Remaining Refactor (Week 7-8)

**Goal:** Clean architecture, fix remaining debt, production readiness

| Task | Priority | Effort | Detail |
|------|----------|--------|--------|
| Fix swallowed exceptions | P1 | 1 day | Add structured logging to all except blocks |
| Type safety pass | P2 | 2 days | Reduce `dict[str, Any]` usage, fix mypy errors |
| Add core.py unit tests | P1 | 2 days | Test tensor mapping, config building, tokenizer extraction |
| Integration test suite | P2 | 3 days | Convert small GGUF files, verify `mlx_lm.load()` works |
| Remaining P2 ARCH_MAP entries | P2 | 3 days | Falcon H1, Hunyuan, Jamba, Olmo3, Mamba, +batch |

**Rollback strategy:** Each ARCH_MAP entry and arch rule is independently testable via mock GGUF fixtures (Phase 1). If a new entry breaks existing conversions, the specific entry can be reverted without affecting others. No feature flags needed — the modular `arch_rules.py` design means per-arch rules are isolated by definition.

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

Rules use the unified `ArchRule` dataclass from §2.2. Example entries:
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

> **Note:** `ArchRule` fields (`min_bits`, `moe_preferred_bits`, `fp16_threshold_gb`, `requires_awq`, `skip_tensors`, `custom_config_overrides`) are defined in §2.2 `plan.py`. See §5.3 for an `ArchRule` with `skip_tensors` and `custom_config_overrides`.

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

## Part 6: Validation Harness Design

### 6.1 Goals

- **Deterministic and reproducible** — fixed sampling (temp=0.0, seed=42)
- **Three simple metrics** — keyword hit-rate, token overlap, length ratio (not just one score)
- **Chip-aware prompt budget** — bounded runtime per bandwidth tier
- **Graceful degradation** — skip GGUF side if `llama-cli` absent, skip MLX if `mlx_lm` absent
- **Arch-agnostic** — initial Qwen-focused prompt suite, but `--validate-prompts` JSONL allows any architecture. Phase 3 adds arch-agnostic defaults.

### 6.2 CLI Flags

```python
analysis.add_argument("--validate", action="store_true",
    help="Run lightweight GGUF vs MLX validation on a prompt suite")
analysis.add_argument("--validate-prompts", type=str, metavar="PATH",
    help="JSONL of prompts: {\"id\",\"task\",\"prompt\",\"expected_keywords\":[...]}")
analysis.add_argument("--validate-max-tokens", type=int, default=128,
    help="Generation cap per prompt (default: 128)")
analysis.add_argument("--validate-temp", type=float, default=0.0,
    help="Sampling temperature (default: 0.0 for deterministic)")
```

### 6.3 Chip-Aware Prompt Budget

```python
def qwen_validation_budget(hw: dict, gguf_size_bytes: int) -> int:
    tier = hw.get("bandwidth_tier", "mid")
    gen = hw.get("chip_gen", 0)
    big_model = gguf_size_bytes > 15e9
    if gen <= 1 and tier == "low":   return 4 if big_model else 6
    if tier == "low":                return 6 if big_model else 8
    if tier == "mid":                return 8 if big_model else 12
    if tier == "high":               return 10 if big_model else 14
    if tier == "ultra":              return 14 if big_model else 20
    return 10
```

| Bandwidth Tier | Big Model (>15GB) | Small Model | Rationale |
|----------------|-------------------|--------------|-----------|
| `low` (M1 base) | 4 prompts | 6 | Minimal — avoid timeout |
| `mid` (M1 Pro, M2–M5 base) | 8 | 12 | Good coverage, bounded runtime |
| `high` (M3/M4 Pro/Max, M5 Pro) | 10 | 14 | Strong coverage for throughput chips |
| `ultra` (M4/M5 Max) | 14 | 20 | Full suite — bandwidth is plentiful |

### 6.4 Three Metrics

```python
def keyword_hit_rate(output: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in output (0–1)."""
    if not keywords: return 0.0
    text = output.lower()
    return sum(1 for kw in keywords if kw.lower() in text) / len(keywords)

def token_overlap(a: str, b: str) -> float:
    """Symmetric token overlap between two outputs (0–1)."""
    a_t, b_t = a.split(), b.split()
    if not a_t or not b_t: return 0.0
    sa, sb = set(a_t), set(b_t)
    return len(sa & sb) / ((len(sa) + len(sb)) / 2.0)
```

| Metric | What it detects | Good threshold |
|--------|----------------|----------------|
| `keyword_hit_rate` | Task-level accuracy loss from quantization | MLX ≥ GGUF - 0.05 |
| `token_overlap` | Output divergence between GGUF and MLX | ≥ 0.4 average |
| `length_ratio` | Truncation (ratio << 1) or verbosity (ratio >> 1) | 0.7–1.5 |

**Regression rule:** If `avg_overlap < 0.4` AND `avg_kh_mlx + 0.05 < avg_kh_gguf`, emit a warning that quantization may be too aggressive.

### 6.5 Runner Design

```python
def run_llama_cli(gguf_path, prompt, max_tokens, temp) -> str:
    try:
        proc = subprocess.run(
            ["llama-cli", "-m", str(gguf_path), "-p", prompt,
             "-n", str(max_tokens), "--temp", str(temp), "--seed", "42"],
            capture_output=True, text=True, check=False)
        return proc.stdout.strip()
    except FileNotFoundError:
        warn("llama-cli not found; skipping GGUF validation")
        return ""

def run_mlx_generate(model_dir, prompt, max_tokens, temp) -> str:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "mlx_lm", "generate",
             "--model", str(model_dir), "--prompt", prompt,
             "--max-tokens", str(max_tokens), "--temperature", str(temp),
             "--seed", "42"],
            capture_output=True, text=True, check=False)
        return proc.stdout.strip()
    except FileNotFoundError:
        warn("mlx_lm generate not available; skipping MLX validation")
        return ""
```

Both runners use **identical sampling params** (temp, max_tokens, seed=42) for fair comparison.

### 6.6 Output Format

Rich table with per-prompt rows + aggregate summary:

```
╭─ Qwen GGUF vs MLX Validation ─╮
│ ID         Task    KH GGUF  KH MLX  Overlap  GGUF len  MLX len │
│ q-code-01  code    1.00     1.00    0.82     45        47       │
│ q-reason-02 reason  0.67     0.50    0.55     98        87       │
│ q-fact-03  fact    1.00     1.00    0.91     32        34       │
│ ...                                                             │
╰─────────────────────────────────────────────────────────────────╯
Validation summary: avg overlap=0.73, avg KH GGUF=0.89, avg KH MLX=0.83
```

### 6.7 Future: `--validate-perplexity`

Optional MLX-only perplexity proxy for when no GGUF baseline exists:
- Compute perplexity on a small held-out text shard (~500 tokens)
- Compare across different MLX quantization configs
- Flag if perplexity delta > 20% between configs

---

## Part 7: Success Metrics

| Metric | Current | Phase 1 Target | Phase 4 Target |
| ARCH_MAP entries (working) | 29 | 45 (+ P0 models, remap successors) | 90+ (auto-discovery) |
| Dead ARCH_MAP entries | 16 | 0 (removed or remapped) | 0 |
| mlx-lm compatibility | ~27% | ~42% | ~84% |
| Broken MoE models | Gemma4 | Gemma4 fixed | All MoE working |
| Double-quant guard | Warn only | Hard block | Configurable |
| Quality validation | None | Basic | AWQ-calibrated |
| convert.py lines | 4,249 | ~3,200 (extracted to package) | ~800 (thin CLI) |
| Test count | 273 | 310+ (mock GGUF fixtures) | 400+ |
| mypy errors | 4 | 4 | 0 |
| Planning overhead | N/A | <5% of convert time | <5% |

---

## Part 8: M1–M5 Hardware Bandwidth Tier System

### 8.1 Problem Statement

The current `smart_defaults()` uses only `chip_tier` and `ram_gb` — it doesn't account for:
- **Memory bandwidth** (the strongest predictor of LLM tok/s on Apple Silicon)
- **bf16 vs fp16** hardware support (M1/M2 emulate bf16 in software, M3+ has native support)
- **Generation-specific capabilities** (M5 Neural Accelerators, M3+ bf16 native)

### 8.2 Bandwidth Tier Classification

Extend `detect_apple_silicon()` to add a `bandwidth_tier` field:

```python
# In convert.py — extend detect_apple_silicon() return dict
def classify_bandwidth(hw: dict[str, Any]) -> str:
    gen = hw["chip_gen"]
    tier = hw["chip_tier"]
    if gen == 1:
        if tier in ("max", "ultra"): return "high"    # ~400–800 GB/s
        elif tier == "pro":                return "mid"     # ~200 GB/s
        else:                              return "low"     # ~68 GB/s
    if gen == 2:
        return "mid" if tier in ("pro", "base") else "high"
    if gen == 3:
        return "mid" if tier == "base" else "high"
    if gen == 4:
        return "mid" if tier == "base" else "high"
    if gen >= 5:
        if tier in ("max", "ultra"):  return "ultra"   # ~614 GB/s
        elif tier == "pro":              return "high"    # ~307 GB/s
        else:                            return "mid"     # ~153 GB/s
    return "mid"
```

**Bandwidth tiers and their implications:**

| Tier | Chips | Approx BW | Best Use | Quant Guidance |
|------|-------|-----------|----------|---------------|
| `low` | M1/M2 base | ~68 GB/s | 1–7B models, 4-bit | Clamp to 4-bit for >3GB, group_size≥128 |
| `mid` | M1 Pro, M2 Pro/base, M3/M4/M5 base | ~150–200 GB/s | 3–14B models, 4–6-bit | 4-bit/group64 for large, 6-bit/group32 for small |
| `high` | M1/M2 Max/Ultra, M3/M4 Pro/Max, M5 Pro | ~307–546 GB/s | 14–70B models, 4-bit | 4-bit/group32, 6–8-bit for <7B in hf-quality |
| `ultra` | M4 Max, M5 Max/Ultra | ~614+ GB/s | 30–400B MoE, any | 4-bit/group32 aggressive, 8-bit for supervisor models |

### 8.3 bf16 vs fp16 Handling

**Critical difference:** M1/M2 emulate bf16 in software → 40–70% prefill penalty. M3+ has native bf16.

Update `_resolve_dtype()` to accept `hw` parameter:

```python
def _resolve_dtype(args, meta, hw) -> str:
    if args.dtype:
        return str(args.dtype)
    file_type = meta.get("file_type") if meta else None
    # M1/M2: always use float16 (bf16 is emulated)
    if hw.get("chip_gen", 0) <= 2:
        if file_type == 0:
            return "float32"
        return "float16"  # Force fp16 even if source is bf16
    # M3+: bf16 is native
    if file_type == 26:  # MOSTLY_BF16
        return "bfloat16"
    return "float16"
```

### 8.4 Per-Generation Planner Rules

#### M1 (low bandwidth, 8–16GB)
- Clamp `target_bits ≤ 4` for models >6GB
- Force `group_size ≥ 128` to reduce memory overhead
- Always `intermediate_dtype = "float16"`

#### M1 Pro/Max/Ultra (mid/high, more RAM)
- Allow 8-bit for <3GB models in hf-quality mode
- 4-bit/group64 for 3–15GB models
- Keep bf16→fp16 conversion

#### M2 (incremental over M1)
- Same bf16 handling as M1 (still emulated)
- Slightly more headroom; existing smart_defaults suffice

#### M3 (bf16 native)
- `intermediate_dtype = "bfloat16"` when source is BF16 (file_type 26)
- No bf16→fp16 conversion needed
- Pro/Max can handle 30B at 4-bit

#### M4 (high bandwidth for Max)
- M4 Max: ~546 GB/s, good for 30–70B at 4-bit
- Annotate plan: "4-bit/group32 tuned for high-bandwidth; good for 30–70B Qwen"

#### M5 (ultra bandwidth, Neural Accelerators)
- M5 Max: ~614 GB/s, 128GB — ideal for large models
- M5 base: ~153 GB/s, 32GB — great for dense 14B or MoE 30B
- **Auto-upgrade speed→hf-quality for large models** on M5 Max/Ultra
- Bias toward MLX hf-quality for long-context generation
- Validation harness: longer prompts on M4/M5 to probe long-context quality

### 8.5 Integration Into ConversionPlan

The `ConversionPlan` dataclass (§2.2) already includes `bandwidth_tier` and `chip_gen` fields. No additional dataclass changes needed.

The planner function applies bandwidth-tier rules **after** arch-specific rules, so Qwen's "no ≤3-bit" constraint takes priority over M1's bandwidth constraints, but both are enforced.

---

## Sources & Research

### Quality & Quantization
- [ContraCollective — GGUF vs MLX Quantization Formats 2026](https://contracollective.com/blog/gguf-vs-mlx-quantization-formats-apple-silicon-2026)
- [MuhammadRaza — GGUF vs MLX Decision Guide](https://muhammadraza.me/2026/gguf-vs-mlx-decision-guide/)
- [Latitude — Quantized LLMs Cost/Performance Results](https://latitude.so/blog/quantized-llms-cost-performance-results)
- [arxiv 2505.02214v1 — Qwen Quantization Research](https://arxiv.org/html/2505.02214v1)
- [Qwen docs — AWQ Quantization](https://qwen.readthedocs.io/en/latest/quantization/awq.html)
- [GenerativeAI — Practical Guide to LLM Quantization](https://generativeai.pub/practical-guide-of-llm-quantization-gptq-awq-bitsandbytes-and-unsloth-bdeaa2c0bbf6)
- [JarvisLabs — vLLM Quantization Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks)

### Hardware & Bandwidth
- [Apple — M5 Pro and M5 Max announcement](https://www.apple.com/au/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)
- [Apple — M5 launch](https://www.apple.com/au/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)
- [Apple ML Research — Exploring LLMs with MLX on M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [LLMCheck — Apple Silicon Chips for AI comparison](https://llmcheck.net/compare/apple-silicon-chips-for-ai/)
- [Parallels — Apple M Chips Guide](https://www.parallels.com/blogs/apple-m-chips-guide/)
- [Markus Schall — MLX on Apple Silicon compared with Ollama](https://www.markus-schall.de/en/2025/09/mlx-on-apple-silicon-as-local-ki-compared-with-ollama-co/)
- [FamStack — MLX vs GGUF isolating variables](https://famstack.dev/guides/mlx-vs-gguf-part-2-isolating-variables/)
- [Cho.sh — M-series LLM benchmarking](https://cho.sh/E5B180)

### Validation & Evaluation
- [APXML — Evaluating Deployed Quantized LLMs](https://apxml.com/courses/practical-llm-quantization/chapter-6-evaluating-deploying-quantized-llms/evaluating-quantized-models)
- [DeepChecks — Top LLM Quantization Methods Impact on Quality](https://deepchecks.com/top-llm-quantization-methods-impact-on-model-quality/)
- [arxiv 2405.06001v1 — Quantization Evaluation Methods](https://arxiv.org/html/2405.06001v1)
- [Cast.ai — Demystifying Quantization for LLMs](https://cast.ai/blog/demystifying-quantizations-llms/)

### Models & Architecture
- [Qwen-AgentWorld HuggingFace](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B)
- [mlx-lm GitHub (82+ model modules)](https://github.com/ml-explore/mlx-lm/)
- [DeepWiki — mlx-lm Supported Models](https://deepwiki.com/ml-explore/mlx-lm/5.1-overview-and-supported-models)
- [Qwen docs — GGUF quantization](https://qwen.readthedocs.io/en/v1.5/quantization/gguf.html)
- [Atomic — GGUF vs MLX](https://atomic.chat/blog/guides/gguf-vs-mlx)
