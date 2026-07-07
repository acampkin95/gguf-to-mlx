# GGUF→MLX Roadmap

**Vision:** Make GGUF→MLX conversion seamless, reliable, and *quality-preserving* for all Apple Silicon (M1–M5) users — treating GGUF as a transport format and recomputing MLX-native quantization where it improves fidelity.

**Status:** v1.4.0 stable — vendored engine, 326 tests, ~78 % coverage, ruff clean, `convert.py` mypy clean.

> **Companion planning docs (authoritative detail):**
> - [`plans/refactor-and-model-roadmap.md`](plans/refactor-and-model-roadmap.md) — full 8-part design: ConversionPlan layer, M1–M5 bandwidth tiers, validation harness, Qwen/AgentWorld strategy, phased schedule.
> - [`plans/MACOS_GUI_MASTERPLAN.md`](plans/MACOS_GUI_MASTERPLAN.md) — the **macOS GUI track, run as a separate workstream** (v1.5.0). Not in the timeline below; tracked independently.

---

## Current State — v1.4.0 (2026-07-06)

### ✅ Shipping
- **Vendored `gguf2mlx` v2.0.2** — conversion engine lives in `gguf2mlx/`; `convert.py` calls `gguf2mlx.convert()` **directly** (no subprocess). Bugs fixable in-tree.
- **Smart defaults** from Apple Silicon chip + RAM (M1–M5, all tiers).
- **Full CLI** — guided menu + direct modes (Rich panels, spinners, progress with ETAs). 30 distinct flags.
- **Quantization control** — `--bits {2,3,4,6,8}`, `--group-size {32,64,128,256}`, `--mode {affine,mxfp4,nvfp4,mxfp8}`, `--predicate`, `--dtype {float16,float32}`, `--no-quantize`, `--preset {speed,balanced,quality,m5-max}`, `--high-bandwidth`.
- **Architecture auto-detection** + pre-flight compatibility checks; known-issues DB (`KNOWN_CONVERSION_ISSUES`) for Gemma2/3/4, Qwen3.5, DeepSeek-V3.
- **HuggingFace hub** — `hf:org/model` registry URLs, `--hf-search/-s`, `--hf-download/-H`, `--hf-list/-l`, `--hf-file`, `--hf-token`, `--auto-convert/-C` (download → convert in one step, via streaming `requests`).
- **Model management** — `--scan/-S` (omlx / LM Studio / HF cache / custom), `--models-dir`, `--set-models-dir`, `--delete-gguf`, persistent config at `~/.config/gguf-to-mlx/config.json`, last-10 history.
- **Pipeline control** — `--resume`, `--keep-intermediate`, `--cleanup-old`, `--force`, intermediate-dir safety.
- **Analysis** — `--inspect`, `--estimate`, `--mtp`.
- **326 tests pass (exit 0)**; ruff clean; `convert.py` mypy clean (non-strict).

### ⚠️ Known Limitations (honest, refreshed)
- **Coverage dropped to ~78 %** (was 91.6 % at v1.1.0) — v1.4.0 added the vendored engine wrapper, streaming download, auto-convert, and the full interactive-menu system faster than tests grew. Recovery is an explicit roadmap goal.
- **3 mypy errors remain in vendored `gguf2mlx/core.py`** (out of scope for `convert.py`; see ADR). Makefile runs non-strict `mypy convert.py`.
- **No `ConversionPlan` layer yet** — quantization decisions are ad-hoc; no architecture-aware "no-double-quant" guard, no AWQ calibration, no `--validate` harness. *(This is the v1.5 focus — see below.)*
- **Qwen3 / Qwen3.5 are in `KNOWN_CONVERSION_ISSUES` (warn/fast-fail), not in `SUPPORTED_MLX_ARCHITECTURES`** — the supported set today has 21 entries (`qwen, qwen2, qwen2_5` plus llama/gemma/mistral/phi/deepseek families).
- **`convert.py` is 4,200 LOC single-file** (97 functions/classes) — `main()` itself is now a lean 187 lines (split into 16 helpers), but the file is a module-extraction candidate.
- **No batch / parallel conversion** — one model per invocation.
- **No structured file logging** — console-only.
- **SonarQube dashboard last scanned at v1.1.0** — needs a re-scan against current code before its "0 issues" claim can be trusted.

---

## Release History

| Version | Date | Focus | Status |
|---|---|---|---|
| **v1.0.0** | 2026-05-30 | MVP — single-file, smart defaults | ✅ |
| **v1.1.0** | 2026-06-11 | 91 % coverage, SonarQube (28→4 issues), stability | ✅ |
| **v1.2.0** | 2026-06-12 | Model management + HF hub (search/download/list/scan) + config persistence | ✅ |
| **v1.3.0** | 2026-06-12 | Interactive menu system, guided workflows, presets, history | ✅ |
| **v1.4.0** | 2026-07-06 | **Vendor gguf2mlx** (direct calls, no subprocess), requests-based download, auto-convert | ✅ |

---

## Forward Roadmap

The forward work is organized into **tracks** grounded in the technical research (quality-preserving conversion, M1–M5 optimization, validation). Full designs live in `plans/refactor-and-model-roadmap.md`; this section is the executive view. All items below are **not yet implemented** unless marked ✅.

### Track 1 — ConversionPlan layer & quality-preserving conversion *(highest leverage)*
The central architectural addition: an explicit, architecture-aware plan that decides quantization and prevents the **double-quantization** problem (re-compressing already-K-quantized GGUF weights degrades quality).

- `ConversionMode` enum: `preserve-source` (dequantize GGUF K-quant → FP16, no re-quant), `hf-quality` (target HF-equivalent FP16/8-bit), `speed` (max compression).
- `ConversionPlan` dataclass + `build_conversion_plan(args, meta, hw)` planner (see plan §5).
- **No-double-quant guard**: detect source scheme (e.g. `Q4_K_M`) and refuse/emwarn re-quantization to ≤ source bits unless `--allow-low-bits`.
- **Bit-width guidance to encode** (research-backed): 8-bit ≈ 99–100 %, 4-bit ≈ 98–99 %, **2-bit = quality cliff**; 6-bit is the sweet spot for small coding models.
- **CLI**: `--quality-mode {preserve-source,hf-quality,speed}`, `--allow-low-bits`.

### Track 2 — Architecture expansion (the 82-arch gap)
Broaden `SUPPORTED_MLX_ARCHITECTURES` and add arch-specific rules.

- **P1**: Qwen3, Qwen3.5, Qwen3-AgentWorld, Gemma4 (MoE), Llama4, DeepSeek-V3/V3.2, Mistral3, LFM2.
- **Qwen strategy** (plan §3.2/§5.3): `qwen3moe` expert routing, avoid ≤3-bit defaults (bump to 4-bit + warn), first-class MLX targets.
- **Gemma3/4 MoE**: default to FP16-only in quality modes (quant needs arch-specific handling).
- Per-arch dispatch table replacing the current `read_gguf_metadata()` complexity hotspot.

### Track 3 — M1–M5 hardware optimization *(research: bandwidth-tiers)*
Memory bandwidth is the strongest predictor of token throughput; encode it into the planner (plan §8).

- **Bandwidth tiers** from `detect_apple_silicon()`: low (~68 GB/s, M1 base) → mid (~150–200) → high (~300, Max/Pro) → ultra (~307–614, Ultra/M5 Max).
- **bf16 vs fp16**: M1/M2 prefer `float16` intermediates (bf16 emulated → 40–70 % prefill penalty recovered); M3+ keep `bfloat16` natively when source is `MOSTLY_BF16`.
- **Per-generation planner rules**: tier × model-size → default bits/group-size (e.g. ultra + 30–70B → 4-bit/group32; low + 1–7B → 4–6-bit/group128).
- **Supervisor vs micro-agent** defaults (plan §8.4): high-RAM supervisors → FP16/8-bit; small helpers → 4–6-bit.

### Track 4 — Validation harness `--validate` *(regression guard)*
Compare GGUF vs MLX outputs on a deterministic prompt suite to catch quality regressions from aggressive quant settings (plan §6).

- `--validate`, `--validate-prompts PATH` (JSONL), `--validate-max-tokens` (default 128), `--validate-temp` (default 0.0, deterministic).
- **Chip-aware prompt budget** (`qwen_validation_budget`) — fewer prompts on low-bandwidth/big-model combos.
- **Runners**: `run_llama_cli(gguf)` vs `run_mlx_generate(model_dir)`.
- **Metrics**: keyword hit-rate, symmetric token overlap, length ratio (truncation/verbosity).
- CI-integrable; future `--validate-perplexity`.

### Track 5 — AWQ calibration & per-tensor quantization *(stretch)*
- `--calibrate-awq`: forward-pass calibration over representative prompts (Qwen code, Mistral long-context, Llama chat) → per-channel AWQ scales before final quant.
- **Per-tensor quant**: keep `lm_head`/`output_projection` at 8-bit/FP16; AWQ `q/k/v_proj`; generic 4/6-bit for the rest.
- Disabled in `speed`, opt-in for `hf-quality`.

### Track 6 — Metadata propagation
Write provenance into MLX output (`config.json` or `metadata.json`) for oMLX/LM Studio routing:
```json
{ "gguf_to_mlx": { "source_scheme": "Q4_K_M", "source_bits": 4.5,
  "target_bits": 8, "mode": "hf-quality", "calibrated": true,
  "arch": "qwen3", "hf_origin": "Qwen/Qwen3-8B-Instruct" } }
```

### Track 7 — Codebase refactor & module extraction
`convert.py` → `gguf_to_mlx/` package (plan Parts 1–2): extract `plan.py`, `arch/`, `validate.py`, `quant.py`, `cli.py`. Recover coverage to 90 %+ and re-enable `mypy --strict`.

### Track 8 — Batch, CI/CD, integration tests
`--batch-dir`, `--batch-list`, `--parallel N`; GitHub Actions on macOS M-series runners (mypy, pytest --cov, sonar-scanner); E2E suite with small real GGUF models (Phi-2, Gemma2-2B, Llama3.2-1B) across 5–6 archs × 2–3 quants.

> **Strategic note (research §1.1):** for models with canonical HF repos, `mlx_lm.convert --hf-path` (HF→MLX direct) remains the *primary* quality path; GGUF→MLX is the pragmatic fallback when only a GGUF is available. The roadmap invests in making the fallback fidelity match the primary.

---

## Phased Schedule (from plan Part 4)

| Phase | Window | Scope |
|---|---|---|
| **0 — Package shell** | Wk 1a | Extract `gguf_to_mlx/` skeleton, move-only refactor, tests green |
| **1 — Foundation** | Wk 1b–2 | ConversionPlan (T1) + critical model gaps: Qwen3/3.5, Gemma4 (T2) |
| **2 — Expansion** | Wk 3–4 | P1 architectures (T2), per-arch dispatch |
| **3 — Quality & validation** | Wk 5–6 | M1–M5 tiers (T3), `--validate` (T4), bf16/fp16 handling |
| **4 — Polish** | Wk 7–8 | AWQ/per-tensor (T5), metadata (T6), coverage → 90 %+, `mypy --strict` |

---

## Success Metrics

| Metric | v1.1 (prior) | v1.4 (now) | Target |
|---|---|---|---|
| Tests | 209 | **326** | 380+ |
| Coverage (`convert.py`) | 91.6 % | **~78 %** | 90 %+ |
| `mypy` | "strict clean" (stale) | `convert.py` clean (non-strict); 3 errs in vendored `core.py` | `--strict` clean end-to-end |
| Supported archs | 18 | 21 (qwen3/3.5 *known-issue* only) | +12 P1 archs |
| Quality guard | — | none | `--validate` + no-double-quant |
| ruff | — | ✅ clean | ✅ clean |
| Batch / CI | — | — | `--parallel`, GitHub Actions |

---

## Architecture Decision Log

### Why vendor `gguf2mlx` and call it directly? (v1.4.0) — *supersedes the old "subprocess for isolation" ADR*
- ✅ No subprocess overhead; engine bugs fixed in-tree without waiting on upstream.
- ✅ Direct `gguf2mlx.convert()` call; cleaner error interception than streamed stdout.
- ⚠️ Trade-off: we now own `gguf2mlx/core.py` (3 mypy errors, vendored quirks) — acceptable, documented as known debt.
- `mlx_lm` quantization (`mlx_lm.convert`) **still runs as a subprocess** (Popen at L930) — intentional: it's an external tool whose progress stream we parse.

### Why single-file `convert.py`? (revisit condition updated)
- ✅ Easy install (one file + `pip` deps); clear user-visible flow.
- The original "revisit when main() > 300 lines" trigger **already fired and was handled** — `main()` was split to 187 lines (16 helpers, `cbb0416`).
- **New revisit trigger:** total file exceeded 4,000 LOC → Track 7 package extraction now warranted, not just main() splitting.

### Why did coverage drop to ~78 %?
- v1.4.0 added vendored-engine wrapping, streaming download, auto-convert, and the full interactive menu faster than tests.
- Not a "ceiling" — recoverable via the package refactor (Track 7) + integration tests (Track 8). The old "91 % ceiling because of real GGUF files" rationale is retired.

### Quality strategy: GGUF as transport, not truth
- GGUF K-quants are mixed-precision; naive re-quantization to MLX 4-bit compounds error. The ConversionPlan `preserve-source`/`hf-quality` modes (Track 1) dequantize to FP16 first and recompute quantization, optionally AWQ-calibrated (Track 5).

---

## Acknowledgments

Built on [gguf2mlx](https://github.com/barrontang/gguf2mlx) (vendored engine), [mlx-lm](https://github.com/ml-explore/mlx-community) (quantization), [Rich](https://rich.readthedocs.io/)), [Apple MLX](https://github.com/ml-explore/mlx). Forward tracks are grounded in technical research on Qwen GGUF→MLX conversion, M1–M5 bandwidth optimization, and AWQ-style calibration — detailed and sourced in [`plans/refactor-and-model-roadmap.md`](plans/refactor-and-model-roadmap.md) §"Sources & Research".
