# GGUF-to-MLX Roadmap

**Vision:** Make GGUF→MLX conversion seamless, reliable, and accessible for all Apple Silicon users.

**Status:** v1.0.0 stable (production-ready for supported architectures)

---

## Current State: v1.0.0

### ✅ Shipping Features
- **Smart defaults** based on Apple Silicon chip & RAM
- **CLI with guided + direct modes**
- **Quantization presets** (quality, balanced, speed, auto)
- **Architecture auto-detection**
- **Gemma2/3/4 workarounds** (known issues + post-conversion fixes)
- **Rich UI** with progress bars, tables, error messages
- **28 passing tests** + full mypy compliance

### ⚠️ Known Limitations
- No pre-flight check for mlx_lm architecture support → Qwen3.5 fails after 1min
- No conversion resume/caching → Must restart if Step 2 fails
- Limited quantization control → Can't customize `group_size`, `mode`, `dtype`
- No batch conversion → One model per CLI invocation
- No detailed logging → Hard to debug failures
- No intermediate cleanup on error → Clutter left behind

---

## Roadmap: v1.1 → v2.0

### **v1.1: Stability & Fast-Fail** (Target: 2-3 weeks)

**Goal:** Prevent wasted computation on unsupported architectures; enable graceful fallbacks.

- [ ] **Pre-flight mlx_lm compatibility check**
  - Scrape supported architectures from mlx_lm source
  - Fail before Step 1 if unsupported
  - Suggest fallbacks (Ollama, llama.cpp, web services)
  - **Test:** Unit test for Qwen3.5 detection, e2e with unsupported arch

- [ ] **Auto-cleanup intermediate dirs on error**
  - Detect and remove failed intermediate directories
  - Add `--keep-intermediate` flag for debugging
  - Add `--cleanup-old` to remove old intermediates
  - **Test:** Unit + e2e failure scenarios

- [ ] **Conversion resume / skip completed steps**
  - Add `--resume` flag to restart from specific step
  - Detect existing intermediate directories
  - Skip Step 1 if intermediate already present
  - **Test:** Interrupt mid-conversion, resume, verify success

- [ ] **Expand known issues database**
  - Document Qwen3.5 → unsupported + workarounds
  - Add Mixtral MoE edge cases
  - Add bfloat16 compatibility notes
  - **Test:** Unit test for issue lookup

**Estimated effort:** ~1 week  
**Impact:** Prevents user time waste on doomed conversions

---

### **v1.2: User Customization** (Target: 2-3 weeks)

**Goal:** Unlock power-user fine-tuning; improve observability.

- [ ] **Quantization parameter controls**
  - Add `--group-size` flag (8, 16, 32, 64, 128)
  - Add `--mode` flag (affine, block_sparse)
  - Add `--dtype` flag (float16, bfloat16, float32)
  - Update help, docs, examples
  - **Test:** Parameter validation, e2e with custom values

- [ ] **Detailed logging & debug mode**
  - Add `--log-file` for conversion history
  - Add `--verbose` / `-v` flag
  - Structured logging (JSON option)
  - Log: timestamp, step, status, command, duration
  - **Test:** Parse logs, verify format

- [ ] **Cost estimation before conversion**
  - Estimate peak RAM (model + buffers)
  - Estimate disk space (source + intermediate + final + 20% margin)
  - Estimate duration (model_size / throughput lookup)
  - Fail early if insufficient resources
  - **Test:** Edge cases (low RAM, low disk)

**Estimated effort:** ~1.5 weeks  
**Impact:** Enable advanced workflows; better diagnostics

---

### **v2.0: Batch & Hardening** (Target: 3-4 weeks)

**Goal:** Production-grade reliability; enable bulk conversions.

- [ ] **Batch conversion**
  - Add `--batch-dir` to scan for all `.gguf` files
  - Add `--batch-list` to read paths from text file
  - Add `--parallel N` for concurrent conversions
  - Resource checks: disk, memory per task
  - **Test:** e2e with 5+ models, verify all succeed

- [ ] **Integration test suite**
  - Download small test models (Phi-2, Gemma2-2B, Llama2-7B)
  - E2E tests: convert → verify output → load in mlx_lm
  - Test matrix: 5-6 architectures × 2-3 quantization levels
  - CI/CD: GitHub Actions (macOS M-series runner)
  - **Test:** 15-20 new e2e tests (~5-10 min runtime)

- [ ] **Post-conversion validation**
  - Verify tokenizer.json loads in transformers
  - Verify config.json matches mlx_lm schema
  - Check safetensors file integrity (headers, tensor counts)
  - Optional: single inference pass to verify model loads
  - **Test:** Unit tests for validation rules

- [ ] **Documentation**
  - Add TROUBLESHOOTING.md for common errors
  - Add ADVANCED.md for power users (custom params, batch, logging)
  - Add ARCHITECTURE.md (design decisions)
  - Expand README with examples

**Estimated effort:** ~2-2.5 weeks  
**Impact:** Production-ready; catch regressions automatically

---

### **v2.1+: Premium Features** (Optional, lower priority)

- [ ] **Performance benchmarking**
  - Add `--benchmark` flag
  - Run inference on test dataset (GSM8K or MMLU)
  - Measure: tokens/sec, memory peak, quality score
  - Compare quantized vs. float16 reference
  - **Estimated effort:** 1-2 weeks

- [ ] **Model metadata enrichment**
  - Scrape HuggingFace for recommended quantization
  - Cache locally (optional)
  - Display recommendations pre-conversion
  - **Estimated effort:** ~1 week

- [ ] **Advanced features**
  - Shell completion (bash/zsh)
  - Config file support (`.gguf-to-mlx.rc`)
  - Non-local models (HuggingFace URLs, download + convert)
  - Version auto-update checks
  - **Estimated effort:** 2-3 weeks total

---

## Release Timeline

| Version | Timeline | Focus | Status |
|---------|----------|-------|--------|
| **v1.0.0** | Now | MVP, single-file, smart defaults | ✅ Stable |
| **v1.1** | +2-3w | Stability, fast-fail, resume | 🔄 Next |
| **v1.2** | +4-6w | Power users, logging, costs | 📅 Planned |
| **v2.0** | +7-11w | Batch, integration tests, CI/CD | 📅 Planned |
| **v2.1+** | +12w+ | Premium features (optional) | 💡 Backlog |

---

## Contributing

Interested in helping? See issues labeled:
- `good-first-issue` for newcomers
- `help-wanted` for priority features
- `stability` for Phase 1 work

See [CONTRIBUTING.md](CONTRIBUTING.md) (planned) for guidelines.

---

## Feedback

Have a model that fails? Found a gap?
- Open an issue with error log and model name
- Tag with `unsupported-arch` or `bug` as appropriate
- Include: GGUF file size, quantization type, M-chip variant, error message

---

## Success Metrics (3 months)

| Metric | Target |
|--------|--------|
| **Test coverage** | 40+ unit + 15+ e2e tests |
| **Unsupported architecture handling** | Pre-flight check 100% coverage |
| **Error recovery** | Resume, cleanup, cost estimation implemented |
| **User customization** | `--group-size`, `--mode`, `--dtype` flags |
| **Batch support** | `--batch-dir`, `--batch-list`, `--parallel` |
| **CI/CD** | E2E tests running on GitHub Actions |
| **Documentation** | ROADMAP, TROUBLESHOOTING, ADVANCED guides |

---

## Architecture Decision Log

### Why single-file CLI? (Not modular)
- ✅ Easy to install (copy 1 file)
- ✅ No dependency management complexity
- ✅ Fast startup, low overhead
- ✅ Clear for users to understand flow

Revisit if:
- Grows beyond ~2,500 lines
- Multiple entry points needed (daemon, API)
- Complex state management required

### Why subprocess for gguf2mlx + mlx_lm?
- ✅ Isolation: one tool failure doesn't crash CLI
- ✅ Progress tracking: can stream output
- ✅ Simplicity: no direct Python dependency on internals
- ⚠️ Trade-off: harder to intercept errors

Alternative: Direct Python function calls (Phase 2+)

---

## Acknowledgments

Built on:
- [gguf2mlx](https://github.com/barrontang/gguf2mlx) (core engine)
- [mlx-lm](https://github.com/ml-explore/mlx-community) (quantization)
- [Rich](https://rich.readthedocs.io/) (CLI UI)
- [Apple MLX](https://github.com/ml-explore/mlx) (framework)
