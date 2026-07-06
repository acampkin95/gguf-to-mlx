# Changelog

All notable changes to this project will be documented in this file.

## [1.4.0] - 2026-07-06

### Vendor gguf2mlx Internally
- **Vendored gguf2mlx v2.0.2** into `gguf2mlx/` package — no more external dependency
- **Direct function calls** — Step 1 now calls `gguf2mlx.convert()` directly instead of spawning a subprocess
- **Removed** `gguf2mlx @ git+https://github.com/acampkin95/gguf2mlx.git` from `pyproject.toml` dependencies
- **Self-contained maintenance** — conversion engine bugs can be fixed directly in `gguf2mlx/core.py`
- Updated LICENSE, CREDITS.md, README.md to reflect vendored status

---

## [1.3.0] - 2026-06-12

### Interactive Menu System
- **Full guided menu** — Launches when no CLI args provided (6 options + exit)
- **Guided Convert** — Step-by-step: pick file, choose quant (smart/quality/custom), confirm plan, convert
- **Guided Scan & Convert** — Choose scan source (all/omlx/lmstudio/custom), spinner while scanning, pick model
- **Guided HF Download** — Search → pick model → pick file → download with progress (option 3 & 4)
- **Guided Inspect** — Drag-and-drop GGUF file for full metadata view
- **Settings sub-menu** — Manage models dir, HF token, clear history, view config path
- **Hardware recommendations** — Menu shows size recommendation based on your chip/RAM (e.g. "70B+ models fit comfortably")
- **Post-convert actions** — After conversion: test with prompt, start chat, list output files, delete GGUF

### Polish & Visual Quality
- **Richer completion celebration** — Green panel with size ratio, time, space saved, speed rating
- **Spinner indicators** — HF search, file listing, and model scanning show animated spinners
- **Conversion plan estimates** — Guided convert shows pre-conversion plan with estimated output size
- **History tracking** — Last 10 conversions/downloads saved to `~/.config/gguf-to-mlx/history.json`
- **Pause after actions** — Each workflow pauses with "Press Enter to return to menu" so output is readable
- **Error resilience** — All guided workflows wrapped in try/except; errors return to menu instead of crashing
- **Consistent section headers** — Panel-style headers with subtitle labels for each workflow

### CLI Polish
- Reorganized arg groups: Quantisation | Pipeline Control | Analysis | Display | Model Management | HuggingFace Hub
- Short flags: `-S` (scan), `-s` (hf-search), `-H` (hf-download), `-l` (hf-list), `-C` (auto-convert)
- Categorized epilog examples: Basic Conversion, Advanced Quantisation, Analysis, Model Management, HuggingFace Hub
- Cleaner banner with chip name, RAM, and conversion count

### Infrastructure
- Added `requests>=2.31.0` to dependencies
- Config system at `~/.config/gguf-to-mlx/config.json` with models_dir and hf_token
- History system at `~/.config/gguf-to-mlx/history.json`
- LM Studio dual-path support (`~/.lmstudio/models/` and `~/Library/Application Support/LM Studio/models/`)

---

## [1.2.0] - 2026-06-12

### Model Management
- **Config System** — Persistent configuration via `~/.config/gguf-to-mlx/config.json`
- **Models Directory** — `--set-models-dir PATH` saves default models folder; `--models-dir PATH` to scan custom location
- **Model Scanning** — `--scan` finds GGUF/MLX models in known directories with interactive picker
- **LM Studio Scan** — `--scan-lmstudio` scans `~/.lmstudio/models/`
- **omlx Scan** — `--scan-omlx` scans `~/.omlx/models/`
- **HF Cache Scan** — `--scan-hf-cache` scans `~/.cache/huggingface/hub/`
- **Delete GGUF on Completion** — `--delete-gguf` prompts to remove source GGUF after successful conversion

### HuggingFace Hub Integration
- **`--hf-search QUERY`** — Search HuggingFace Hub for models by keyword, sorted by downloads
- **`--hf-download REPO_ID`** — Download models from HuggingFace with file selection
- **`--hf-file FILENAME`** — Specify exact filename for download
- **`--hf-list REPO_ID`** — List all files in a HuggingFace repository
- **`--hf-token TOKEN`** — Provide token inline (or uses `HF_TOKEN` env var)
- **Token Prompt** — Interactive token prompt if `HF_TOKEN` and config token are both missing; saves to config
- **Beautiful Download Progress** — Rich progress bar with: percentage, download speed (GB/s/MB/s/KB/s), ETA
- **`--auto-convert`** — Chain download → conversion in one command
- **Resume Downloads** — Automatic caching; re-downloads skip if file exists

### CLI
- Updated `epilog` with all new command examples
- All new modes terminate cleanly with `sys.exit(0)`

### Maintenance
- Added `requests>=2.31.0` to dependencies

---

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
