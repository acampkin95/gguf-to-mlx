# GGUF-to-MLX Roadmap

**Vision:** Make GGUF→MLX conversion seamless, reliable, and accessible for all Apple Silicon users.

**Status:** v1.1.0 stable — 91.6% coverage, 209 tests, SonarQube clean

---

## Current State: v1.1.0

### ✅ Shipping Features
- **Smart defaults** based on Apple Silicon chip & RAM (M1–M5, all tiers)
- **CLI with guided + direct modes** (Rich panels, colors, spinners)
- **Quantization presets** (quality, balanced, speed, high-bandwidth, auto)
- **Full parameter control** — `--bits`, `--group-size`, `--mode`, `--dtype`, `--predicate`
- **Architecture auto-detection** with pre-flight compatibility checks
- **Gemma2/3/4 + Qwen3.5 + DeepSeek-V3 workarounds** (known issues database)
- **Rich UI** — progress bars with ETAs, plan tables with status badges, summary tables
- **Conversion resume** — `--resume` flag skips completed steps
- **Auto-cleanup** — removes intermediate dirs on failure, `--keep-intermediate` for debug
- **`--cleanup-old`** — removes stale `*_intermediate` directories
- **`--estimate`** mode — resource estimation without conversion
- **`--inspect`** mode — display GGUF metadata without conversion
- **`--high-bandwidth`** preset for M5 Max / Ultra devices
- **HuggingFace download** — `hf:org/model` registry URLs
- **Cost estimation** — RAM, disk, duration estimates before conversion
- **209 passing tests**, 91.6% coverage, mypy --strict clean
- **SonarQube** — 0 bugs, 0 vulnerabilities, 0% duplication

### ⚠️ Known Limitations
- No batch conversion → One model per CLI invocation
- No structured logging to file → Console-only output
- 4 Cognitive Complexity warnings (S3776) in large functions — requires architecture refactor
- 2 regex security hotspots (controlled input, acceptable risk)
- `main()` is 700 lines — needs splitting into sub-commands

---

## Release History

| Version | Focus | Status |
|---------|-------|--------|
| **v1.0.0** | MVP, single-file, smart defaults | ✅ Stable |
| **v1.1.0** | 91% coverage, SonarQube, stability fixes | ✅ Released |
| **v1.2** | Architecture refactor, logging | 📅 Next |
| **v2.0** | Batch, integration tests, CI/CD | 📅 Planned |

---

## Roadmap: v1.2 → v2.0

### **v1.2: Architecture Refactor & Logging** (Target: 1-2 weeks)

**Goal:** Reduce cognitive complexity; add structured logging; improve maintainability.

- [ ] **Refactor `main()` into sub-commands**
  - Extract pipeline stages into `Pipeline` class with `run()` method
  - Sub-command pattern: `convert`, `inspect`, `estimate`, `resume`
  - Target: reduce `main()` complexity from 209 → <50
  - **Test:** Existing 209 tests must still pass unchanged

- [ ] **Refactor `read_gguf_metadata()` complexity (50 → <15)**
  - Extract arch-specific metadata readers into dispatch table
  - Separate SSM detection, MTP detection, quality classification
  - **Test:** Existing mocked tests cover all paths

- [ ] **Refactor `display_metadata()` complexity (18 → <15)**
  - Extract panel builders for each metadata section
  - **Test:** Existing display tests cover all paths

- [ ] **Structured logging to file**
  - Add `--log-file` for conversion history
  - Add `--verbose` / `-v` flag
  - JSON structured logging option
  - Log: timestamp, step, status, command, duration
  - **Test:** Parse logs, verify format

- [ ] **Fix remaining SonarQube issues**
  - Resolve 4 Cognitive Complexity warnings (S3776)
  - Target: 0 open issues on SonarQube dashboard

**Estimated effort:** ~1-1.5 weeks
**Impact:** Maintainable codebase; enables future batch features

---

### **v2.0: Batch & Integration Tests** (Target: 2-3 weeks)

**Goal:** Production-grade reliability; enable bulk conversions.

- [ ] **Integration test suite with real GGUF models**
  - Download small test models (Phi-2 2.7B, Gemma2-2B, Llama3.2-1B)
  - E2E tests: convert → verify output → load in mlx_lm
  - Test matrix: 5-6 architectures × 2-3 quantization levels
  - Target: push coverage from 91% → 95%+
  - **Test:** 15-20 new e2e tests (~5-10 min runtime)

- [ ] **Batch conversion**
  - Add `--batch-dir` to scan for all `.gguf` files
  - Add `--batch-list` to read paths from text file
  - Add `--parallel N` for concurrent conversions
  - Resource checks: disk, memory per task
  - **Test:** e2e with 5+ models, verify all succeed

- [ ] **Post-conversion validation hardening**
  - Verify tokenizer.json loads in transformers
  - Verify config.json matches mlx_lm schema
  - Check safetensors file integrity (headers, tensor counts)
  - Optional: single inference pass to verify model loads
  - **Test:** Unit tests for validation rules

- [ ] **CI/CD Pipeline**
  - GitHub Actions on macOS M-series runner
  - Run: mypy --strict, pytest --cov, sonar-scanner
  - Quality gate: 90%+ coverage, 0 type errors
  - **Test:** CI config validated

- [ ] **Documentation**
  - Add TROUBLESHOOTING.md for common errors
  - Add ADVANCED.md for power users (custom params, batch, logging)
  - Expand README with examples and architecture table

**Estimated effort:** ~2-2.5 weeks
**Impact:** Production-ready; catch regressions automatically

---

### **v2.1+: Premium Features** (Optional, lower priority)

- [ ] **Performance benchmarking**
  - Add `--benchmark` flag
  - Run inference on test dataset (GSM8K or MMLU)
  - Measure: tokens/sec, memory peak, quality score
  - **Estimated effort:** 1-2 weeks

- [ ] **Model metadata enrichment**
  - Scrape HuggingFace for recommended quantization
  - Display recommendations pre-conversion
  - **Estimated effort:** ~1 week

- [ ] **Advanced features**
  - Shell completion (bash/zsh/fish)
  - Config file support (`.gguf-to-mlx.toml`)
  - Non-local models (HuggingFace URLs, download + convert)
  - Version auto-update checks
  - **Estimated effort:** 2-3 weeks total

---

## Success Metrics (Updated)

| Metric | v1.0 Target | v1.1 Actual | v2.0 Target |
|--------|-------------|-------------|-------------|
| **Test coverage** | 40+ unit tests | 209 tests, 91.6% | 230+ tests, 95%+ |
| **Type safety** | mypy clean | ✅ mypy --strict | ✅ mypy --strict |
| **SonarQube** | — | 0 bugs, 0 vulns | 0 issues total |
| **Unsupported arch** | Pre-flight check | ✅ Done + known issues DB | Same |
| **Error recovery** | Resume + cleanup | ✅ Done | Same |
| **User customization** | `--group-size`, `--mode` | ✅ Done + `--dtype`, `--predicate` | Same |
| **Batch support** | `--batch-dir`, `--parallel` | — | ✅ Target |
| **CI/CD** | GitHub Actions | — | ✅ Target |
| **Documentation** | ROADMAP | ✅ + SonarQube | TROUBLESHOOTING, ADVANCED |

---

## Architecture Decision Log

### Why single-file CLI? (Not modular)
- ✅ Easy to install (copy 1 file, `pip install` deps)
- ✅ No dependency management complexity
- ✅ Fast startup, low overhead
- ✅ Clear for users to understand flow

**Revisit when:** `main()` exceeds 300 lines after refactor → split into `cli/` package

### Why subprocess for gguf2mlx + mlx_lm?
- ✅ Isolation: one tool failure doesn't crash CLI
- ✅ Progress tracking: stream stdout for % and fraction patterns
- ✅ Simplicity: no direct Python dependency on internals
- ⚠️ Trade-off: harder to intercept errors

**Alternative for v2.0:** Direct Python function calls with error wrapping

### Why 91% coverage ceiling?
- Remaining 9% requires real GGUF files (integration tests)
- `main()` edge paths with interactive prompts (stdin blocking)
- Defensive error handlers for corrupted safetensors
- **v2.0 integration tests** will close this gap

---

## Acknowledgments

Built on:
- [gguf2mlx](https://github.com/barrontang/gguf2mlx) (core engine)
- [mlx-lm](https://github.com/ml-explore/mlx-community) (quantization)
- [Rich](https://rich.readthedocs.io/) (CLI UI)
- [Apple MLX](https://github.com/ml-explore/mlx) (framework)
