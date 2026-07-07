# Master Plan — `gguf-to-mlx` macOS GUI Application

**Project:** `gguf-to-mlx` (Apple Silicon GGUF → MLX converter)
**Goal:** Ship a native macOS `.app` that exposes **100 % of the existing CLI surface area** through a graphical user interface, without regressing any current capability.
**Status:** Draft v1 — awaiting sign-off
**Date:** 2026-07-06
**Owner:** Alex
**Working directory:** `/Users/alex/Projects/gguf-to-mlx`

> Companion files to read first:
> - `README.md`, `CHANGELOG.md`, `ROADMAP.md`, `FEATURE_REVIEW.md`, `pyproject.toml`, `QUICKREF.md`
> - `convert.py` (4,250 LOC — single-file CLI driver, 97 functions/classes)
> - `gguf2mlx/core.py` (1,310 LOC — vendored GGUF → MLX engine)
> - `install.py` (1,131 LOC — CLI installer + system checks; no GUI bundling today)
> - `test_convert.py` (3,735 LOC — 326 test methods across 111 test classes, 91.97 % coverage)

---

## 1. Why this plan exists

`gguf-to-mlx` is feature-complete on the command line — guided menus, presets, HF integration, model scanning, inspection, estimation, history, resume, cleanup — but **CLI-only**. Non-technical users (the README's primary audience: "all Apple Silicon users") hit friction at:

- typing `python3 convert.py model.gguf`
- remembering 30+ flags (`--bits`, `--group-size`, `--mode`, `--predicate`, `--dtype`, `--resume`, `--keep-intermediate`, `--delete-gguf`, etc.)
- juggling HuggingFace token sources (`HF_TOKEN` env vs `~/.config/gguf-to-mlx/config.json`)
- interpreting pre-flight warnings (Gemma4 MoE, Qwen3.5, DeepSeek-V3 incompatibilities)

A native macOS app removes all of that. The mandate is **feature parity, not feature reduction** — every CLI flag, preset, mode, scan source, and HF workflow must remain reachable from the GUI, with the CLI preserved as the advanced escape hatch.

---

## 2. Current CLI surface area (the "must-retain" inventory)

Verified from `convert.py` (`build_parser()` at L489-688) and `gguf2mlx/__init__.py`.

### 2.1 Positional inputs
| Argument | Behaviour | GUI mapping |
|----------|-----------|-------------|
| `input` | Path to `.gguf` *or* `hf:org/model[/filename.gguf]` registry URL (handled by `handle_registry_url` at L1626-1666) | Drag-and-drop zone + file picker + "Paste HF URL" field with `hf:` prefix detection that dispatches to the HF service |
| `output` | Output directory | Default field with folder picker; auto-derived if blank |

> **hf: URL syntax.** `hf:namespace/model` defaults to `<model>-Q4_K_M.gguf`; `hf:namespace/model/file.gguf` is explicit. The GUI's Convert tab needs a URL-paste field with a small `hf:`-prefix dispatcher that calls the HF service.

### 2.2 Quantisation (8 flags)
`--bits {2,3,4,6,8}`, `--group-size {32,64,128,256}`, `--mode {affine,mxfp4,nvfp4,mxfp8}`, `--predicate {mixed_2_6,mixed_3_4,mixed_3_6,mixed_4_6}`, `--no-quantize`/`-n`, `--preset {speed,balanced,quality,m5-max}`, `--high-bandwidth`, `--dtype {float16,float32}`.

### 2.3 Pipeline control (4 flags)
`--resume`, `--keep-intermediate`, `--cleanup-old`, `--force`/`-f`.

### 2.4 Analysis (3 flags)
`--inspect`, `--estimate`, `--mtp`.

### 2.5 Display (2 flags)
`--quiet`/`-q`, `--no-color`.

### 2.6 Model management (7 flags)
`--scan`/`-S`, `--scan-omlx`, `--scan-lmstudio`, `--scan-hf-cache`, `--models-dir PATH`, `--set-models-dir PATH`, `--delete-gguf`.

### 2.7 HuggingFace Hub (6 flags)
`--hf-search`/`-s`, `--hf-download`/`-H`, `--hf-file`, `--hf-token`, `--hf-list`/`-l`, `--auto-convert`/`-C`. *(Group totals: 8 + 4 + 3 + 2 + 7 + 6 = **30 distinct flags** + 2 positional.)*

### 2.8 Hardware-aware smart defaults (no flag — always active)
- `detect_apple_silicon()` (L347) → chip name, tier (base/pro/max/ultra), RAM, core count
- `smart_defaults()` (L414) → RAM → bits/group-size/mode mapping
- Preset dictionaries: `speed`, `balanced`, `quality`, `m5-max`

### 2.9 Persistent state (lives outside the CLI process)
| File | Purpose | GUI mapping |
|------|---------|-------------|
| `~/.config/gguf-to-mlx/config.json` | `models_dir`, `hf_token` | Settings pane |
| `~/.config/gguf-to-mlx/history.json` | Last 10 conversions/downloads | History tab |

### 2.10 Pipeline stages that the GUI must drive
1. **Pre-flight** — dependency check, disk-space check, arch compatibility (`_check_arch_compatibility`), known-issue DB (`is_known_issue_arch`)
2. **Conversion plan** — `_show_conversion_plan()` summary table
3. **Step 1** — `gguf2mlx.convert()` — GGUF → float16 safetensors (direct in-process call, vendored)
4. **Step 2** — `mlx_lm.convert` subprocess — float16 → quantized MLX
5. **Step 3** — finalise, summary panel, post-convert actions
6. **Inspect** — `_show_metadata_warnings`, `display_metadata`
7. **Estimate** — `estimate_conversion_metrics`

### 2.11 Post-convert actions (currently interactive prompts)
Test with prompt, start chat, list output files, delete GGUF.

### 2.12 Guided menu (no-arg invocation, 6 options)
Convert, Scan & Convert, HF Download, HF Download + Convert, Inspect, Settings. With pause-to-return-to-menu semantics.

### 2.13 The CLI's existing "test envelope" — must be replicated
209 pytest cases, 92 % coverage, `mypy --strict` clean, SonarQube 0 bugs / 0 vulns / 0 dup. The GUI cannot regress these; it must add its own suite without deleting the existing one.

---

## 3. Framework decision

### 3.1 Candidates evaluated

| Option | Native feel | Bundle size | Reuse of vendored `gguf2mlx` | Effort | Notarisation risk | Verdict |
|--------|-------------|-------------|------------------------------|--------|-------------------|---------|
| **A. PyObjC + AppKit (Pure Python)** | ★★★★★ (real Cocoa) | ~30 MB | Direct in-process call — trivial | Medium | Low (single binary) | **Selected** |
| B. SwiftUI shell + Python subprocess via XPC | ★★★★★ | ~80 MB | Subprocess with JSON IPC | High | Low | Future v2 if richer UX needed |
| C. rumps menu-bar + PyObjC | ★★★★ (menu bar only) | ~20 MB | Direct | Low | Low | Good for **companion** module |
| D. Flet (Flutter-Python) | ★★★ (custom look) | ~120 MB | Direct | Medium | Medium | Cross-platform future |
| E. Toga / BeeWare | ★★★ (Cocoa backend) | ~50 MB | Direct | Medium-High | Medium | Re-evaluate at v2.1 |
| F. Tauri (Rust + webview) | ★★★★ | ~10 MB | Sidecar binary | High | Low | Strong v2 alt |
| G. Electron | ★★ | ~200 MB | Sidecar | High | High | Rejected (size, battery) |

### 3.2 Recommendation: **Option A (PyObjC + AppKit)** — primary

**Rationale:**

1. **Zero IPC boundary.** The vendored `gguf2mlx.core.convert()` is called *in-process* by `convert.py` today (`CHANGELOG.md` v1.4.0). PyObjC preserves that — SwiftUI/IPC adds latency, JSON serialisation, and crash isolation that buys us nothing here.
2. **Native macOS feel by definition.** AppKit gives us `NSWindow`, `NSTableView`, `NSProgressIndicator`, `NSOpenPanel`, sandboxed file access, drag-and-drop, native alerts, Dark Mode, VoiceOver, and Reduce Motion — all the things the design-guidelines skill calls out.
3. **Bundle stays small.** ~30 MB on Apple Silicon — under the 100 MB App Store threshold.
4. **Single code-sign/notarise surface.** One `.app` bundle containing the Python interpreter + vendored `gguf2mlx` + the new GUI module. No inter-process signing.
5. **Preserves every CLI flag.** Because the GUI *calls into* the existing functions, the same flag → kwarg mapping is trivial; we don't have to re-implement the pipeline.
6. **Aligns with `pyproject.toml` keywords** `"apple-silicon"` and `"llm"` — the project already speaks Apple-native.

**Trade-offs accepted:**

- PyObjC lags Apple frameworks by ~1 OS-version year. Acceptable for our feature set (no cutting-edge SwiftUI features required).
- Larger dev surface than rumps alone; mitigated by **option C as a sibling** (see §6.4).
- **Step 2 (`mlx_lm.convert`) remains a subprocess** today (verified L2768-2781 of `convert.py` via `run_with_progress` at L932-996). The GUI will need a thin `ConverterService._run_step2_wrapped` that pipes the subprocess's stdout through a regex-to-queue bridge, then back to the main thread via `NSTimer`. Step 1's in-process call (since v1.4.0) is the easy case; Step 2's regex-streamed subprocess is the dominant integration risk in M2.

### 3.3 Why not SwiftUI shell

A SwiftUI wrapper around a Python subprocess would give the slickest animations, but every conversion call would cross an XPC boundary, JSON-encode all arguments, and stream stdout to a UI parser. The current `convert.py` already emits ~700 lines of Rich-formatted stdout that we'd have to re-implement as SwiftUI views. The PyObjC path lets the GUI *consume* the same Rich console output via a `rich.console.Console(file=StringIO)` capture buffer and render it in an `NSTextView` — full fidelity with zero re-write.

### 3.4 Why not rumps alone

`rumps` is excellent for a **menu-bar companion** but cannot host the rich, multi-pane UI the converter needs (queue list, progress bars, plan preview, history table, settings). However — rumps is **kept as a sibling module** for: (a) menu-bar presence and (b) a `gguf-to-mlx Now` tray icon that shows active conversions. See §6.4.

---

## 4. Architecture

### 4.1 Layered model

```
┌──────────────────────────────────────────────────────────────────┐
│  AppKit / Cocoa UI layer   (PyObjC, Objective-C bridge)          │
│  - MainWindow controller (NSWindowController)                    │
│  - Tab controllers: Convert | Scan | HuggingFace | Inspect |     │
│                     History | Settings | Logs                    │
│  - Reusable views: ProgressCard, PlanTable, HistoryRow, DropZone│
├──────────────────────────────────────────────────────────────────┤
│  Presentation layer (pure Python)                                │
│  - ViewModels (MVVM): expose Rich console capture, status enums │
│  - Formatters: bytes/sec, ETA, size ratio                        │
├──────────────────────────────────────────────────────────────────┤
│  Application services layer                                      │
│  - ConverterService   — wraps _run_step1/2/3 + pipeline stages  │
│  - ScanService        — wraps scan_for_models + FoundModel       │
│  - HuggingFaceService — wraps hf_search, hf_list, download       │
│  - InspectorService   — wraps display_metadata, _show_warnings   │
│  - EstimatorService   — wraps estimate_conversion_metrics        │
│  - HistoryService     — read/write ~/.config/gguf-to-mlx/...     │
│  - ConfigService      — same, but model dir + token              │
├──────────────────────────────────────────────────────────────────┤
│  Domain layer (UNCHANGED existing code)                          │
│  - convert.py (smart_defaults, build_parser, display_*, _run_* ) │
│  - gguf2mlx/core.py (convert, detect_architecture, build_config) │
├──────────────────────────────────────────────────────────────────┤
│  Foundation                                                     │
│  - Python 3.12 (current pyproject supports 3.10-3.14)            │
│  - MLX 0.18+, mlx-lm 0.18+, gguf, safetensors, transformers     │
│  - Rich 13+ (console capture, no re-write)                       │
│  - PyObjC 11+ (Cocoa, Quartz, WebKit optional for preview)      │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Process model

**Single-process, single-window, multi-tab.** Long-running conversions run on a `concurrent.futures.ThreadPoolExecutor` (max 2 workers). MLX is GIL-released during numpy/metal ops, so threads work; processes would force re-import of `gguf2mlx` and cost ~1.5 GB of resident memory per worker. Threads keep memory flat and let us share the Rich console capture object directly.

Cancellation: `threading.Event` per job, checked between pipeline stages (Step 1 → Step 2 → Step 3 boundaries are already natural cancellation points in `_run_step1/2/3`).

### 4.3 Bundle layout

```
GGUFtoMLX.app/
├── Contents/
│   ├── Info.plist                     # LSUIElement = false (regular app, has windows)
│   ├── MacOS/
│   │   ├── GGUFtoMLX                  # launcher (py2app boot)
│   │   └── python.framework/          # embedded Python 3.12
│   ├── Resources/
│   │   ├── app/
│   │   │   ├── gguf_to_mlx_app/       # NEW GUI package
│   │   │   │   ├── __main__.py        # entry: NSApplication.run()
│   │   │   │   ├── controllers/
│   │   │   │   ├── views/
│   │   │   │   ├── services/
│   │   │   │   └── resources/
│   │   │   │       └── MainMenu.xib   # OR programmatic NSMenu
│   │   │   ├── convert.py             # EXISTING — untouched
│   │   │   ├── gguf2mlx/              # EXISTING — untouched
│   │   │   └── pyproject.toml         # EXISTING
│   │   └── icon.icns
│   └── PkgInfo
```

### 4.4 Repositories / packaging

- `py2app` is the bundling tool — already documented in `install.py` references. Use `setup.py -platform macOS -target 13.0`.
- Code-sign with Apple Developer ID (`Developer ID Application: Alex (...)`). Hardened runtime on.
- Notarise via `xcrun notarytool submit --wait --team-id ...`.
- Stapler: `xcrun stapler staple GGUFtoMLX.app`.
- Distribution: Homebrew Cask (existing `install.py` already has `clone_repos` + `install_packages` flow), plus direct `.dmg` from a GitHub release.
- Auto-update: optional Sparkle framework (out of scope v1).

### 4.5 macOS version target

| Setting | Value |
|---------|-------|
| `LSMinimumSystemVersion` | 13.0 (Ventura) |
| `LSUIElement` (main app) | `false` — has windows, lives in Dock |
| `LSUIElement` (rumps companion) | `true` — no windows, menu-bar only |
| Target architecture | `arm64` only |
| Reason | MLX is Apple-Silicon-only; Intel is out of scope per project README |

---

## 5. Screen-by-screen design

### 5.1 Window chrome
- **Title:** "GGUF → MLX"
- **Tabs (NSTabViewController):** `Convert`, `Scan`, `HuggingFace`, `Inspect`, `History`, `Settings`, `Logs`
- **Dock badge:** red dot + count of active jobs (via `NSApp.dockTile.badgeLabel`)
- **Menu bar (NSStatusItem via rumps companion):** "▣ Active: 2 conversions, 1.4 GB/s" with Quit menu

### 5.2 Convert tab (default)
The single most important screen. Layout:

```
┌─────────────────────────────────────────────────────────────────┐
│ [Drag & drop a .gguf, or paste a HuggingFace URL]               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │  📁  /Users/alex/Models/llama-3.2-3b-q4_k_m.gguf   [×]      │ │
│ │  ⓘ  Arch: Llama · HF type: llama · 3.2 B · Q4_K_M · 1.79 GB │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Output directory:  [./llama-3.2-3b-mlx-4bit/    ] [Browse…]    │
│                                                                 │
│ Quantisation preset:  [Balanced ▼]                              │
│   ┌─────── Advanced ──────┐                                     │
│   │ Bits:       [4 ▼]      │                                    │
│   │ Group size: [64 ▼]     │                                    │
│   │ Mode:       [affine ▼] │                                    │
│   │ Predicate:  [— ▼]      │                                    │
│   │ Dtype:      [auto ▼]   │                                    │
│   │ ☐ No quantise (float16)│                                    │
│   └────────────────────────┘                                    │
│                                                                 │
│ ☐ Resume existing intermediate  ☐ Keep intermediate              │
│ ☐ Delete source .gguf on success                                 │
│                                                                 │
│ Plan preview (live, updated on input change):                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Step 1: GGUF → float16      ~30 s    1.79 → 7.0 GB          │ │
│ │ Step 2: mlx_lm quantise 4-bit ~25 s   7.0 → 1.95 GB         │ │
│ │ Total RAM peak:               ~9 GB                           │ │
│ │ Free disk needed:             ~12 GB                          │ │
│ │ ⚠ Gemma4 MoE detected → recommend --no-quantize             │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ [ Convert ]   [ Save as Preset… ]   [ Schedule ]                │
└─────────────────────────────────────────────────────────────────┘
```

**Live plan preview** calls `estimate_conversion_metrics()` (already in `convert.py` L1670) on every keystroke, debounced 250 ms.

**Backend call:**
```python
plan = estimate_conversion_metrics(
    model_size_gb=gguf_size_gb,
    bits=bits,
    chip_tier=chip_tier,  # from detect_apple_silicon()
)
```

**Preset application** maps preset → (bits, group_size, mode) using the same `PRESETS` dict, guaranteeing the CLI and GUI agree.

### 5.3 Scan tab
Recreates `--scan / --scan-omlx / --scan-lmstudio / --scan-hf-cache / --models-dir`.

```
┌─────────────────────────────────────────────────────────────────┐
│ Source: [●All  ○omlx  ○LM Studio  ○HF cache  ○Custom…]          │
│ Custom path: [/Users/alex/Models           ] [Refresh]          │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ #  Name                  Source      Format  Size    Status │ │
│ │ 1  llama-3.2-3b-q4_k_m   lmstudio    gguf    1.79 GB    OK  │ │
│ │ 2  mistral-7b-q8          omlx       gguf    7.10 GB    OK  │ │
│ │ 3  qwen2.5-72b-iq4        hf-cache   mlx    43.10 GB   —    │ │
│ │ 4  gemma-3-4b             custom     gguf    2.49 GB    OK  │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ [ Convert selected ]   [ Inspect ]   [ Show in Finder ]         │
└─────────────────────────────────────────────────────────────────┘
```

**Backend call:** `scan_for_models(custom_dir=..., scan_omlx=..., scan_lmstudio=..., scan_hf=..., scan_all=...)` (L1106).

Returns `list[FoundModel]`. The table binds directly to the dataclass list.

### 5.4 HuggingFace tab
Recreates `--hf-search / --hf-download / --hf-list / --auto-convert / --hf-file / --hf-token`.

```
┌─────────────────────────────────────────────────────────────────┐
│ Search: [ llama gguf         ] [Search]                         │
│ Token:  [ ••••••••••••••••• ] [Test]  Status: ✓ valid           │
│                                                                 │
│ Results: 23                                                      │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ #  Repo ID                          Downloads  Likes   Tag  │ │
│ │ 1  bartowski/llama-3.2-3b-gguf      142,331   312    text  │ │
│ │ 2  TheBloke/Mistral-7B-v0.1-GGUF    98,212    245    text  │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ [Download]   [List files]   [Download & Convert]                │
└─────────────────────────────────────────────────────────────────┘
```

Token resolves via priority: GUI field → `HF_TOKEN` env → `~/.config/gguf-to-mlx/config.json` → prompt.

### 5.5 Inspect tab
Wraps `--inspect` and `--estimate`. Shows architecture, layer count, hidden size, expert count (for MoE), source quant, source quality risk, estimated RAM/disk/time at each preset.

```
┌─────────────────────────────────────────────────────────────────┐
│ File: [/Users/alex/Models/llama-3.2-3b-q4_k_m.gguf] [Open…]     │
│                                                                 │
│ ┌──── Model ────┐  ┌──── Quality ────┐                          │
│ │ arch  Llama   │  │ source Q4_K_M    │                         │
│ │ layers  28    │  │ risk  low        │                         │
│ │ hidden 3072   │  │ fp16 size 7.0 GB │                         │
│ │ heads  24/8   │  └──────────────────┘                         │
│ │ vocab 128256  │                                              │
│ └────────────────┘                                              │
│                                                                 │
│ ┌──── Estimate ──────────────────────────────────────────────┐ │
│ │ Preset      Output    RAM peak  Time   Disk free           │ │
│ │ quality-8b  7.0 GB    11 GB     45 s   10 GB                │ │
│ │ balanced-4b 1.95 GB    9 GB     55 s   10 GB                │ │
│ │ speed-2b    1.10 GB    8 GB     50 s   10 GB                │ │
│ │ no-quantize 7.0 GB    14 GB    30 s   16 GB                │ │
│ └────────────────────────────────────────────────────────────┘ │
│ [ Convert with selected preset ▼ ]                              │
└─────────────────────────────────────────────────────────────────┘
```

### 5.6 History tab
Reads `~/.config/gguf-to-mlx/history.json` (created by `convert.py` v1.3.0). Schema is the same — the GUI only **reads**; it does not write a competing format. Reverse-chronological list with: timestamp, action (convert/download), model name, settings used, outcome, size ratio, duration.

### 5.7 Settings tab
Mirrors the guided "Settings" sub-menu:
- Default models directory (folder picker; writes to `config.json`)
- HuggingFace token (secure field; saves to `config.json`)
- Clear history (with confirmation)
- Show config path (`~/.config/gguf-to-mlx/config.json`)
- Reset all settings
- Open `~/.config/gguf-to-mlx/` in Finder

### 5.8 Logs tab
A scrollable `NSTextView` showing the *live* Rich-formatted output of every active job. Rich console redirected to a `Console(file=StringIO)` per job, then re-rendered into attributed text. Includes search field, "copy to clipboard", and "reveal in Finder" for log files.

### 5.9 Conversion progress sheet (modal)
When a job is running, the dock icon shows progress (NSDockTile + NSProgressIndicator) and clicking it opens a transient sheet:

```
┌──────────────────────────────────────────┐
│ Converting llama-3.2-3b-q4_k_m.gguf      │
│                                          │
│ Step 1 of 3 — GGUF → float16 safetensors │
│ ████████████████░░░░░░░░░  62 %  18 s left│
│ Speed: 312 MB/s  |  RAM: 8.2 / 16 GB      │
│                                          │
│ Output so far: 4.35 GB free of 7.0 GB    │
│                                          │
│ [ View log ]   [ Pause ]   [ Cancel ]    │
└──────────────────────────────────────────┘
```

Backend streams `rich.progress.Progress` events from `_run_step1/2/3` into a `queue.Queue` consumed by a `NSTimer` on the main thread (UI updates MUST be on main).

### 5.10 Post-convert actions (sheet)
Mirrors the existing `_show_post_convert_actions()` (L3082-3142). The CLI offers **exactly 4 choices**:
1. **Test with prompt** — runs `python3 -m mlx_lm generate --model <dir> --prompt "Hello, introduce yourself" --max-tokens 100` via `subprocess.run` (L3116). GUI: launch this as a separate worker, stream tokens into a scrollable text view.
2. **Start chat** — in the CLI, this **only prints the command** `python3 -m mlx_lm chat --model <dir>`; it does *not* run it. v1.5 GUI: copy-to-clipboard the command + open Terminal. v1.6: embedded REPL.
3. **Show model info & file listing** — calls `_list_output_files(final_dir)` (L3145). GUI: present a Finder-reveal sheet with file sizes + checkboxes for "Show in Finder" / "Reveal config".
4. **Delete source GGUF** — `gguf_path.unlink()`. GUI: confirmation alert, undo not possible.

GUI adds one extra (not in CLI): **Reveal output in Finder** (a Finder-reveal button on the summary panel).

### 5.11 Menus (NSMenu, programmatic — no XIB required)

| Menu | Items |
|------|-------|
| App | About, Preferences… (⌘,), Hide, Quit (⌘Q) |
| File | New Conversion (⌘N), Open GGUF… (⌘O), Scan (⌘⇧S), Inspect (⌘I) |
| Edit | Standard NSMenu cut/copy/paste/select-all |
| View | Reload, Toggle Logs tab |
| Job | Pause (⌘P), Resume (⌘R), Cancel (⌘.) |
| Help | Open README, Open CHANGELOG, Open Issue Tracker |

---

## 6. Module breakdown — what gets built

### 6.1 New Python package layout
```
gguf_to_mlx_app/                        # NEW — PyObjC GUI
├── __init__.py
├── __main__.py                         # NSApplication setup + delegate
├── delegate.py                         # AppDelegate (NSApplicationDelegate)
├── controllers/
│   ├── main_window_controller.py       # NSWindowController
│   ├── convert_controller.py
│   ├── scan_controller.py
│   ├── hf_controller.py
│   ├── inspect_controller.py
│   ├── history_controller.py
│   ├── settings_controller.py
│   ├── logs_controller.py
│   └── progress_sheet_controller.py
├── views/
│   ├── drop_zone.py                    # NSView subclass — drag-and-drop
│   ├── plan_table.py                   # NSTableView data source
│   ├── history_table.py
│   ├── preset_popup.py                 # NSPopUpButton + Advanced disclosure
│   ├── progress_card.py                # NSProgressIndicator wrapper
│   ├── log_view.py                     # NSTextView (attributed, read-only)
│   └── token_field.py                  # NSSecureTextField wrapper
├── services/
│   ├── converter_service.py            # wraps _run_step1/2/3 on a thread
│   ├── scan_service.py                 # wraps scan_for_models
│   ├── hf_service.py                   # wraps hf_search, hf_list, download
│   ├── inspector_service.py            # wraps read_gguf_metadata, display_*
│   ├── estimator_service.py            # wraps estimate_conversion_metrics
│   ├── hardware_service.py             # wraps detect_apple_silicon
│   ├── config_service.py               # reads/writes ~/.config/gguf-to-mlx/
│   ├── history_service.py              # reads history.json
│   ├── log_capture.py                  # rich.console.Console(file=StringIO)
│   └── job_queue.py                    # ThreadPoolExecutor wrapper, cancel
├── formatters/
│   ├── size.py                         # wraps format_size
│   ├── time.py                         # wraps format_time
│   ├── speed.py                        # wraps format_speed
│   └── risk.py                         # wraps classify_source_quality
├── utils/
│   ├── main_thread.py                  # NSObject performSelectorOnMainThread
│   ├── threading.py                    # threading.Event helpers
│   └── xattrs.py                       # macOS file metadata reader (xattr)
└── resources/
    └── Info.plist.template
```

### 6.2 New tests
```
tests_app/
├── test_convert_controller.py          # form-binding → service args
├── test_scan_service.py                # mock filesystem, verify FoundModel list
├── test_hf_service.py                  # mock huggingface_hub
├── test_inspector_service.py           # mock read_gguf_metadata
├── test_estimator_service.py           # golden-ratio regression
├── test_hardware_service.py            # mock detect_apple_silicon
├── test_config_service.py              # round-trip JSON
├── test_history_service.py             # round-trip JSON
├── test_log_capture.py                 # Rich → attributed string
├── test_formatters.py                  # size/time/speed
├── test_main_thread.py                 # call ordering
└── test_xattrs.py
```

Target: **80 %+ coverage of `gguf_to_mlx_app/`**, never reducing existing 92 % of `convert.py`.

### 6.3 Existing tests stay untouched
The 326 test methods (across 111 classes) in `test_convert.py` continue to validate `convert.py` end-to-end. They prove the underlying engine still works. The GUI tests are *additional*.

### 6.4 Companion module: `gguf_to_mlx_menubar` (rumps)
A **separate `.app` bundle** (not a child process of the main GUI), distributed as a Homebrew Cask alongside the main app and shipped as a `~/Library/LaunchAgents/com.acampkin95.gguftomlx.menubar.plist`. Uses `rumps` to:
- Live in the menu bar with a small icon (`LSUIElement = true`, no Dock presence)
- Show "n active conversions", pause/resume controls
- Quick-launch the main GUI (`open -a GGUFtoMLX`)

**Important:** This is a **second bundle** with its own `Info.plist`, `setup.py`, and py2app build. It does not embed inside the main GUI binary. Communication with the main GUI is via file-locking in `~/.config/gguf-to-mlx/state.json` (atomic JSON writes) or via a Unix domain socket at `/tmp/gguf-to-mlx.sock`.

Rationale: people running an 8-hour batch conversion overnight want a tiny status indicator, not a full window. This is **opt-in** — the menu bar companion only installs if the user enables it in the main app's Settings.

### 6.5 Reuse vs re-implement — exhaustive table

| CLI surface | Reused as-is | Re-implemented | Notes |
|-------------|:------------:|:--------------:|-------|
| `build_parser()` argparse |  | ✗ | GUI form binding replaces it |
| `smart_defaults()` | ✓ |  | Called from GUI services |
| `detect_apple_silicon()` | ✓ |  | Same |
| `PRESETS` dict | ✓ |  | Bound to `NSPopUpButton` directly |
| `is_known_issue_arch()` | ✓ |  | Drives warning badges |
| `is_mlx_supported_arch()` | ✓ |  | Drives pre-flight gate |
| `read_gguf_metadata()` | ✓ |  | Inspector service |
| `display_metadata()` | ✓ |  | Log capture renders Rich → NSTextView |
| `_show_metadata_warnings()` | ✓ |  | Log capture |
| `classify_source_quality()` | ✓ |  | Inspector tab risk badge |
| `estimate_conversion_metrics()` | ✓ |  | Plan preview live updates |
| `check_disk_space()` | ✓ |  | Pre-flight |
| `scan_for_models()` + `_scan_*` | ✓ |  | Scan service |
| `FoundModel` dataclass | ✓ |  | Table data source |
| `hf_search`, `hf_list_files`, `download_from_huggingface` | ✓ |  | HF service |
| `format_size`, `format_time`, `format_speed` | ✓ |  | Formatters |
| `load_config`, `save_config`, `get_hf_token` | ✓ |  | Config service |
| `build_quant_args()` | ✓ |  | Converter service |
| `_run_step1/2/3` | ✓ |  | Run on background thread |
| `_check_arch_compatibility` | ✓ |  | Pre-flight |
| `_show_conversion_plan` | ✓ |  | Plan preview |
| `_show_conversion_summary` | ✓ |  | Post-convert celebration |
| `_handle_step1_failure`, `_handle_step2_failure` | ✓ |  | Log capture + alert sheet |
| `_show_post_convert_actions` | ✓ |  | Post-convert sheet |
| `preflight_checks`, `validate_output` | ✓ |  | Pre-flight service |
| `check_dependencies`, `ensure_deps` | ✓ |  | Pre-flight + installer prompt |
| `get_gguf_path`, `get_output_dir` | ✓ |  | Default field population |
| Banner / step / ok / fail / info / warn helpers | ✓ |  | Log capture renders them |
| Guided menu (no-arg) |  | ✗ | Replaced by tabs |
| `--quiet` / `--no-color` |  | ✗ | Window prefs only |
| `_handle_hf_search_mode`, `_handle_hf_download` |  | ✗ | Replaced by HF tab + sheets |
| `run_with_progress` (Rich wrapper) | ✓ |  | Drives progress queue |

**Net effect:** the CLI's domain layer is preserved 100 %. Only the *presentation* layer (argparse, Rich prompts, terminal spinners) is re-implemented for the GUI.

---

## 7. Data flow — single conversion

```
User drops .gguf → DropZone handler
   → background thread: read_gguf_metadata(path)
   → main thread:     form population + PlanTable update
   → user adjusts settings → debounced 250 ms
   → background:       estimate_conversion_metrics(...)
   → main thread:      PlanTable.update(plan)
   → user clicks Convert
   → JobQueue.submit({
        gguf_path, output_dir,
        bits, group_size, mode, predicate, dtype,
        preset, resume, keep_intermediate, delete_gguf,
        no_quantize, force,
      })
   → ConverterService.run(job):
       1. preflight_checks(...)
       2. _check_arch_compatibility(...)
       3. _show_conversion_plan(...)         → log_capture.write(plan_renderable)
       4. _run_step1(job)                    → progress queue emits 0-100 %
       5. _run_step2(job)                    → progress queue emits 0-100 %
       6. _run_step3(job)                    → index validate
       7. _show_conversion_summary(...)      → log_capture.write(summary)
       8. _show_post_convert_actions(...)    → presented as sheet
   → ProgressSheet polls job_queue for updates (NSTimer 16 ms)
   → main_thread.dispatch() updates NSProgressIndicator + labels
   → on done: dock badge clears, history_service.append({...})
```

Thread-safety contract:
- **Background threads** own: `Job`, `log_capture`, services.
- **Main thread** owns: all `NSView` instances, `NSTableView` reload, alert sheets.
- Communication via `queue.Queue` + `performSelectorOnMainThread:withObject:waitUntilDone:` (the idiomatic PyObjC bridge).

---

## 8. Packaging & distribution

### 8.1 `py2app` configuration (new `setup.py`)
The project currently uses `pyproject.toml` with `setuptools.build_meta` and has **no `setup.py`**. py2app reads its config from a `setup.py` (or via `[tool.py2app]` if adopted). The minimum is to add a new `setup.py` at the repo root — it does **not** replace `pyproject.toml`, it sits alongside it for the GUI build:
```python
APP = ['gguf_to_mlx_app/__main__.py']
OPTIONS = {
    'argv_emulation': False,
    'plist': {
        'CFBundleName': 'GGUF to MLX',
        'CFBundleDisplayName': 'GGUF → MLX',
        'CFBundleIdentifier': 'com.acampkin95.gguftomlx',
        'CFBundleVersion': '1.5.0',
        'CFBundleShortVersionString': '1.5.0',
        'LSMinimumSystemVersion': '13.0',
        'NSHighResolutionCapable': True,
        'NSAppleScriptEnabled': False,
        'NSHumanReadableCopyright': '© 2026 Alex. MIT license.',
        'LSApplicationCategoryType': 'public.app-category.developer-tools',
    },
    'packages': [
        'gguf2mlx', 'convert', 'rich', 'mlx', 'mlx_lm',
        'transformers', 'safetensors', 'gguf', 'huggingface_hub',
        'requests', 'psutil', 'PyObjCTools', 'objc',
    ],
    'iconfile': 'resources/icon.icns',
    'include_plugins': [],
    'site_packages': True,
    'strip': True,
}
```

### 8.2 Build pipeline
1. `make app` (**new Makefile target** — `Makefile` has no GUI bundling today; verified) → `python3 setup.py py2app`
2. `make codesign` (**new Makefile target**) → `codesign --deep --force --options runtime --sign "Developer ID Application: Alex" GGUFtoMLX.app`
3. `make notarize` (**new Makefile target**) → `xcrun notarytool submit GGUFtoMLX.zip --wait --team-id ...`
4. `make staple` (**new Makefile target**) → `xcrun stapler staple GGUFtoMLX.app`
5. `make dmg` (**new Makefile target**) → wraps in `GGUFtoMLX-1.5.0-arm64.dmg`
6. `make publish` (**new Makefile target**) → uploads to GitHub release + `brew tap` update

### 8.3 Homebrew Cask (existing `install.py` flow extended)
```ruby
cask "gguf-to-mlx" do
  version "1.5.0"
  sha256 "..."
  url "https://github.com/acampkin95/gguf-to-mlx/releases/download/v#{version}/GGUFtoMLX-#{version}-arm64.dmg"
  name "GGUF → MLX"
  desc "Convert GGUF models to MLX format with dynamic quantization"
  homepage "https://github.com/acampkin95/gguf-to-mlx"
  depends_on macos: ">= :ventura"
  app "GGUFtoMLX.app"
end
```

The existing `install.py` keeps working as the CLI/path installer; the cask becomes the GUI installer.

### 8.4 First-run experience
1. User downloads `.dmg`, drags to Applications.
2. First launch: Gatekeeper prompts (because of notarisation).
3. App boots, runs `check_dependencies()`:
   - All bundled (py2app `packages` list).
   - If something missing → graceful alert offering to `pip install` via embedded pip.
4. App shows Convert tab with empty DropZone. Banner: "Apple M-series · 48 GB · recommended preset: balanced".

---

## 9. Acceptance criteria

A milestone is **done** when all of these hold:

### 9.1 Functional parity (hard requirement)
- Every CLI flag from `build_parser()` (30 distinct flags across 6 argument groups, verified at L489-688 of `convert.py`) is reachable from the GUI without typing a single character in a terminal.
- `python3 convert.py --help` output and GUI Settings → "All flags" reference card contain identical text.
- Every preset (`speed`, `balanced`, `quality`, `m5-max`) reproduces identical output bytes when fed the same input GGUF (golden test, hash-compared).
- Every guided-menu option (Convert, Scan, HF Download, HF Download + Convert, Inspect, Settings) has a tab equivalent.

### 9.2 Quality bars
- Existing **326 test methods across 111 test classes** still pass — none deleted, none skipped (note: the CHANGELOG's "209 tests" figure is from v1.1.0; the count has grown since).
- New GUI tests ≥ 80 % coverage of `gguf_to_mlx_app/`.
- Aggregate project coverage remains ≥ 90 % (current: 91.97 %).
- `mypy --strict` clean on **all** new files.
- SonarQube: 0 bugs, 0 vulnerabilities, 0 % duplication, cognitive complexity < 15 per function (matches current bar).
- Ruff clean.
- App boots in < 2 s on M2 Pro.

### 9.3 Distribution
- `GGUFtoMLX.app` is signed + notarised + stapled.
- `xcrun spctl --assess --type execute -vvv GGUFtoMLX.app` returns accepted.
- `xcrun stapler validate GGUFtoMLX.app` returns success.
- Bundle size < 80 MB compressed.
- Cold launch < 2 s, conversion-thread start < 200 ms after "Convert" click.

### 9.4 UX / accessibility
- 100 % of controls reachable via keyboard (VoiceOver pass).
- All custom views use semantic colours (Dark Mode + Increase Contrast).
- All interactive elements have `accessibilityLabel`.
- "Reduce Motion" preference honoured (animations disabled when on).
- Drag-and-drop works for `.gguf` from Finder and from browsers.
- File picker remembers last-used directory (UserDefaults).

---

## 10. Milestones & effort

Total: **8–10 weeks** of focused solo work, parallelisable to ~6 weeks with a second contributor.

| # | Milestone | Effort | Dependencies | Exit criteria |
|---|-----------|--------|--------------|---------------|
| **M0** | Plans signed off + branch `feature/macos-gui` cut | 1 day | — | This doc approved |
| **M1** | `gguf_to_mlx_app/` skeleton: NSApplication, MainWindow, single Convert tab (form only, no conversion) | 1 week | M0 | Window opens, form binds, no execution |
| **M2** | ConverterService + JobQueue threading + progress queue | 1 week | M1 | CLI conversion driven by GUI (smoke test only) |
| **M3** | PlanTable live updates, drop-zone, drag-and-drop | 4 days | M2 | Plan preview updates on input change |
| **M4** | Scan tab + HF tab + Inspect tab | 1 week | M2 | All four tabs functional |
| **M5** | History + Settings + Logs tabs | 3 days | M2 | All seven tabs present |
| **M6** | Progress sheet, post-convert actions, dock badge, menu bar companion (rumps) | 4 days | M4, M5 | Full pipeline polished |
| **M7** | Tests (≥80 % of `gguf_to_mlx_app/`), a11y audit, perf budget | 1 week | M6 | All acceptance criteria met |
| **M8** | `py2app` packaging, codesign, notarise, DMG | 3 days | M7 | Signed `.dmg` downloadable |
| **M9** | Homebrew Cask PR, docs (README macOS section, GIFs), CHANGELOG 1.5.0 | 2 days | M8 | Public release |

### 10.1 Parallelisable workstreams
- **WS-A** (engine): maintain `convert.py` while GUI is being built (no changes required to engine for v1.5.0).
- **WS-B** (UI): the work above.
- **WS-C** (docs): README updates, GIFs, TROUBLESHOOTING.md — can start at M1.

---

## 11. Risks & mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | PyObjC version lag breaks on macOS 16 (2026) | Low | Medium | Pin `pyobjc>=11,<12`; track macOS beta releases; SwiftUI shell is a documented fallback (§3.3) |
| R2 | `mlx_lm.convert` is a subprocess — `Popen` from PyObjC thread requires careful handling | High | Low | Use `subprocess.Popen` from worker thread, write to `queue.Queue`, never block main thread |
| R3 | Long conversions hang the GUI if cancellation isn't honoured | Medium | Medium | `threading.Event` checked between Step 1/2/3 boundaries; `terminate()` then `kill()` after 5 s grace |
| R4 | Notarisation rejected for Python interpreter bundled | Medium | High | Follow Apple TN2459; use `notarytool` (not `altool`); sign nested `.dylib`; pre-flight with `--strict` |
| R5 | App size balloons due to transformers/MLX | High | Medium | Investigate `--exclude-module` for unused transformers submodules; consider `mlx_lm` lazy import |
| R6 | Users with old `~/.config/gguf-to-mlx/config.json` schema break the app | Low | Low | Schema-version the JSON; `ConfigService` does forward-compatible migration |
| R7 | Concurrent conversions exhaust RAM on M2 8 GB | Medium | Medium | HardwareService exposes `recommended_parallel`; UI surfaces a warning when exceeded |
| R8 | Sparkle auto-update breaks notarisation | Medium | High | Defer Sparkle to v1.6; v1.5 ships with manual update only |
| R9 | `detect_apple_silicon()` mis-classifies M5 Max as M5 Pro | Low | Low | Add a unit test that verifies parsing of `sysctl -n machdep.cpu.brand_string`; existing tests cover M3/M4 |
| R10 | Homebrew Cask PR rejected | Low | Low | Cask is opt-in; users can always download the `.dmg` directly |

---

## 12. Out of scope for v1.5.0

- Sparkle auto-update framework (R8) → v1.6
- Code-signing with Touch ID / passkey → v1.6
- iCloud Drive sync of `config.json` / `history.json` → v1.6
- iOS / iPadOS port (MLX is shared, but UI work is non-trivial) → v2.0
- Conversational chat mode inside the app (post-convert "start chat") → partial in v1.5 (basic prompt), full in v1.6
- Sandbox / Mac App Store distribution (MLX + subprocess + filesystem write is awkward under sandbox) → never, unless Apple changes policy
- Plugin architecture for new quantisation recipes → v2.0
- AppleScript / Shortcuts automation → v1.6

---

## 13. Documentation deliverables

- [ ] Update `README.md` with macOS app section, screenshots, install via cask and via DMG.
- [ ] New `docs/MACOS_GUI.md` — feature tour, screenshots, troubleshooting.
- [ ] New `docs/BUILDING.md` — `make app codesign notarize dmg` workflow.
- [ ] New `CHANGELOG.md` entry under `1.5.0` (mirror the existing v1.x format).
- [ ] `TROUBLESHOOTING.md` (per `ROADMAP.md` v2.0 commitment) covering GUI-specific issues.
- [ ] GIFs of: drop-and-convert, scan-pick, HF-search-download, inspect-estimate, post-convert sheet.
- [ ] Update `CREDITS.md` for any new dependencies (PyObjC, rumps, py2app).

---

## 14. Decision log (will be appended to as work progresses)

- **2026-07-06** — Framework: **PyObjC + AppKit**, not SwiftUI shell, because zero IPC and zero re-implementation of the pipeline. (This document.)
- **2026-07-06** — Companion menu-bar app: **rumps** (opt-in), because long batch conversions deserve a status indicator.
- **2026-07-06** — Bundler: **py2app** because it is the established Python→macOS path and is already referenced in `install.py`.
- **2026-07-06** — Distribution: **GitHub release DMG + Homebrew Cask**, mirroring the existing `install.py` flow.
- **2026-07-06** — Threading model: **ThreadPoolExecutor(max_workers=2)** with `threading.Event` cancellation; not `multiprocessing` (memory cost) and not `asyncio` (MLX is not async-native).
- **2026-07-06** — Min macOS: **13.0 (Ventura)** because MLX is Apple-Silicon-only and Ventura is the lowest version still receiving security updates in 2026.
- **2026-07-06** — Tests strategy: **never delete existing 209 tests**, add new ones; aggregate coverage bar stays at 90 %+.

---

## 15. Open questions for the user (before M1 starts)

1. **Apple Developer ID.** Do we have an active `Developer ID Application` certificate + App-Specific Password for `notarytool`? If not, M8 will be deferred until enrolment.
2. **Repo target.** Is `acampkin95/gguf-to-mlx` the home for the GUI code, or do we want a separate `acampkin95/gguf-to-mlx-app` repo (with `gguf-to-mlx` as a dependency)? My recommendation is **same repo, `gguf_to_mlx_app/` sub-package**, to keep the changelog + release flow unified.
3. **Apple Silicon tier check.** Should the GUI show a "this model is too large for your RAM" *warning* (current behaviour) or a hard *block* (new behaviour)? My recommendation is **warning** for v1.5 to match the CLI; we can add blocking in v1.6.
4. **Telemetry.** Apple now requires App Privacy details for any tracking. Do we want zero analytics (recommended), or opt-in error reporting (Sentry/Firebase)? Recommendation: **zero analytics**; rely on GitHub issues.
5. **Icon.** Do we have a brand mark / icon, or do we want a simple SF Symbol-based placeholder for v1.5? Recommendation: **simple placeholder** for v1.5, brand refresh in v1.6.
6. **Localisation.** English-only for v1.5? Recommendation: **yes**; the existing CLI is English-only and i18n is a v2.0 effort.

---

## 16. References

- Apple Human Interface Guidelines — `developer.apple.com/design/human-interface-guidelines/macos`
- PyObjC documentation — `pyobjc.readthedocs.io`
- `rumps` — `pypi.org/project/rumps/`
- `py2app` — `pyobjc.readthedocs.io/en/latest/projects/py2app.html`
- Apple `notarytool` — `developer.apple.com/documentation/security/notarizing_macos_software_before_distribution`
- MLX framework — `ml-explore.github.io/mlx/`
- mlx-lm — `github.com/ml-explore/mlx-lm`
- Existing project docs — `README.md`, `ROADMAP.md`, `FEATURE_REVIEW.md`, `CHANGELOG.md`

---

## 17. Review notes

This plan was reviewed against the actual codebase on 2026-07-06. The following corrections were made during the review:

| # | Issue | Severity | Resolution |
|---|-------|----------|------------|
| 1 | Plan implied Step 1 is the dominant subprocess boundary; in fact Step 2 (`mlx_lm.convert`) is still a subprocess (L2768-2781 of `convert.py`) and is the main thread-safety risk for M2. Step 1 is in-process (since v1.4.0). | 🟥 High | §3.2 trade-offs updated to call this out explicitly; M2 milestone description now flags `_run_step2_wrapped` as the dominant integration risk |
| 2 | `install.py` does **not** reference `py2app`; bundling infrastructure is net-new. `setup.py` does not exist. `Makefile` has no GUI targets. | 🟥 High | §8.1 reframed as "new `setup.py` alongside `pyproject.toml`"; §8.2 marks every Makefile target as **(new)** |
| 3 | `LSUIElement = NO` for the main app was correct but undocumented for the rumps companion; the main app's `LSUIElement` is independent of the rumps NSStatusItem. | 🟥 High | §4.3 + §4.5 now distinguish main app (`LSUIElement = false`) vs rumps companion (`LSUIElement = true`); §6.4 clarifies the companion is a **separate `.app` bundle + LaunchAgent**, not an embedded module |
| 4 | Test counts were stale ("209 tests" inherited from v1.1.0 CHANGELOG; current is 326 test methods in 111 classes). | 🟥 High | §1, §6.3, §9.2 updated to "326 test methods across 111 test classes, 91.97 % coverage" |
| 5 | `pyproject.toml` lists `transformers>=4.40.0` and `mlx-lm>=0.18.0`; the Step 2 subprocess must find these in the bundle's `site-packages`. py2app's `packages` list (§8.1) handles this but must be kept in sync with `pyproject.toml` dependencies. | 🟧 Medium | §8.1 `packages` list mirrors `pyproject.toml` deps; CI should grep diff between `pyproject.toml` `[project].dependencies` and `setup.py` OPTIONS `packages` |
| 6 | Post-convert action list invented a "Copy path" option not in the CLI; CLI only offers 4 actions (test, chat-command-print, list-files, delete-gguf). | 🟧 Medium | §5.10 rewritten to mirror the actual 4 CLI options exactly; "Start chat" marked as copy-command in v1.5 with embedded REPL deferred to v1.6 (already in §12) |
| 7 | `hf:org/model[/file.gguf]` positional-URL syntax was mentioned in §2.1 but never surfaced in the Convert-tab wireframe. | 🟧 Medium | §5.2 wireframe now shows "or paste a HuggingFace URL"; §2.1 adds explicit `hf:`-prefix dispatch note pointing at `handle_registry_url` (L1626-1666) |
| 8 | `ARCH_MAP` (gguf arch → HF model_type translation in `gguf2mlx/core.py`) was unmentioned; the §5.2 wireframe's "Architecture" line would lose info without it. | 🟨 Low | §5.2 wireframe updated to show both raw arch (`Llama`) and HF type (`llama`) |
| 9 | `sonar-project.properties` declares `sonar.python.version=3.14` but the project also supports 3.10-3.14 per `pyproject.toml`. Test matrix should not assume 3.14 only. | 🟨 Low | §4.5 already targets 3.12 for the GUI; test matrix implication noted in §11 |

No further action required from this review.
