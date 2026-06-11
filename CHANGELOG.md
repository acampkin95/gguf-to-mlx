# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-06-11

### Stability & Fast-Fail
- **Auto-cleanup on Failure** — Intermediate directories automatically removed on failure unless `--keep-intermediate`
- **Conversion Resume** — `--resume` flag skips Step 1 when intermediate files exist
- **Pre-flight Compatibility Check** — Early failure for unsupported mlx_lm architectures with fallback guidance
- **Cleanup Old Intermediates** — `--cleanup-old` removes stale `*_intermediate` directories
- **Known Issues Database** — Expanded: Qwen3.5, DeepSeek-V3, Gemma2/3/4 workarounds

### CLI Improvements
- **Rich UI** — Progress bars with ETAs, plan tables with status badges, summary tables
- **Full Parameter Control** — `--bits`, `--group-size`, `--mode`, `--dtype`, `--predicate`
- **`--estimate` mode** — Resource estimation without conversion
- **`--inspect` mode** — Display GGUF metadata without conversion
- **`--high-bandwidth` preset** — M5 Max / Ultra device optimisation
- **HuggingFace download** — `hf:org/model` registry URLs
- **Smart Defaults** — Hardware-aware quantization selection (M1–M5, all tiers)

### Testing & Quality
- **209 tests** (up from 28), **92% coverage** (up from ~10%)
- **mypy --strict** — 0 errors
- **SonarQube** — 0 bugs, 0 vulnerabilities, 0% duplication
- SonarQube code smells: 28 → 5 (all remaining are Cognitive Complexity)
- SonarQube sqale debt: 369 → 78 minutes (79% reduction)
- Bug fix: variable shadowing (`warn` function overwritten by loop variable)

### Architecture
- **Refactored `main()`** — Extracted 16 focused helper functions, complexity 209 → 21
- **Pipeline stages** — `_run_step1/2/3()`, `_show_conversion_plan()`, `_show_conversion_summary()`
- **Validation stages** — `_check_arch_compatibility()`, `_show_metadata_warnings()`, `_show_source_quality_warning()`
- **Config resolution** — `_resolve_dtype()`, `_resolve_quant_params()`, `_resolve_resume()`

### Expanded Architecture Support
- Added `llama3_2`, `qwen2_5`, `deepseek_v3` to supported MLX architectures
- Gemma4 tensor naming detection and workaround guidance

---

## [1.0.0] - 2026-05-30

### Initial Release
- Smart defaults based on Apple Silicon chip & RAM
- CLI with guided + direct modes
- Quantization presets (quality, balanced, speed, auto)
- Architecture auto-detection
- Gemma2/3/4 workarounds
- 28 passing tests + full mypy compliance
