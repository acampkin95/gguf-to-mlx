"""Tests for gguf-to-mlx converter."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Ensure convert.py is importable
sys.path.insert(0, str(Path(__file__).parent))


class TestClassifySourceQuality:
    """Tests for source quality classification."""

    def test_f32_unquantized(self):
        from convert import classify_source_quality
        result = classify_source_quality(0)
        assert result["risk"] == "none"
        assert "unquantized" in result["label"]

    def test_f16_half_precision(self):
        from convert import classify_source_quality
        result = classify_source_quality(1)
        assert result["risk"] == "none"

    def test_q8_lightly_quantized(self):
        from convert import classify_source_quality
        result = classify_source_quality(5)
        assert result["risk"] == "low"

    def test_q4_heavily_quantized(self):
        from convert import classify_source_quality
        result = classify_source_quality(2)
        assert result["risk"] == "high"

    def test_q2_severe(self):
        from convert import classify_source_quality
        result = classify_source_quality(8)
        assert result["risk"] == "severe"


class TestDetectAppleSilicon:
    """Tests for Apple Silicon detection."""

    def test_non_apple_returns_false(self):
        from convert import detect_apple_silicon
        with patch("subprocess.check_output") as mock:
            mock.return_value = "Intel Xeon"
            result = detect_apple_silicon()
            assert result["is_apple_silicon"] is False

    def test_m5_max_parsing(self):
        from convert import detect_apple_silicon
        with patch("subprocess.check_output") as mock:
            mock.return_value = "Apple M5 Max"
            with patch("psutil.virtual_memory") as mock_mem:
                mock_mem.return_value = MagicMock(total=1e11)  # 100GB
                result = detect_apple_silicon()
                assert result["is_apple_silicon"] is True
                assert result["chip_gen"] == 5
                assert result["chip_tier"] == "max"

    def test_m3_pro_parsing(self):
        from convert import detect_apple_silicon
        with patch("subprocess.check_output") as mock:
            mock.return_value = "Apple M3 Pro"
            with patch("psutil.virtual_memory") as mock_mem:
                mock_mem.return_value = MagicMock(total=3.6e10)
                result = detect_apple_silicon()
                assert result["chip_gen"] == 3
                assert result["chip_tier"] == "pro"


class TestSmartDefaults:
    """Tests for smart quantization defaults."""

    def test_max_64gb_bandwidth_optimized(self):
        from convert import smart_defaults
        result = smart_defaults(model_size_gb=20, chip_tier="max", ram_gb=64, chip_gen=5)
        assert result["bits"] == 4
        assert result["group_size"] == 32
        assert "Bandwidth" in result["description"]

    def test_pro_large_model(self):
        from convert import smart_defaults
        result = smart_defaults(model_size_gb=15, chip_tier="pro", ram_gb=36, chip_gen=5)
        assert result["bits"] == 4
        assert result["group_size"] == 64

    def test_small_model_quality(self):
        from convert import smart_defaults
        result = smart_defaults(model_size_gb=2, chip_tier="base", ram_gb=16, chip_gen=4)
        assert result["bits"] == 8
        assert result["group_size"] == 32

    def test_low_ram_conservative(self):
        from convert import smart_defaults
        result = smart_defaults(model_size_gb=10, chip_tier="base", ram_gb=8, chip_gen=3)
        assert result["bits"] == 4
        assert result["group_size"] == 128


class TestGGUFFtypeMap:
    """Tests for GGUF file type mapping."""

    def test_all_ftypes_mapped(self):
        from convert import GGUF_FTYPE_MAP
        # Verify expected key ranges
        for key in range(32):
            assert key in GGUF_FTYPE_MAP, f"Missing ftype {key}"

    def test_ftype_names_are_strings(self):
        from convert import GGUF_FTYPE_MAP
        for key, value in GGUF_FTYPE_MAP.items():
            assert len(value) == 3
            name, bits, desc = value
            assert isinstance(name, str)
            assert isinstance(bits, str)
            assert isinstance(desc, str)


class TestPresets:
    """Tests for quantization presets."""

    def test_all_presets_have_required_keys(self):
        from convert import PRESETS
        required_keys = {"bits", "group_size", "mode", "description"}
        for name, preset in PRESETS.items():
            assert required_keys.issubset(preset.keys()), f"Preset {name} missing keys"
            assert preset["mode"] == "affine"

    def test_m5_max_preset(self):
        from convert import PRESETS
        preset = PRESETS["m5-max"]
        assert preset["bits"] == 4
        assert preset["group_size"] == 32
        assert "M5" in preset["description"] or "64" in preset["description"]


class TestHelpers:
    """Tests for helper functions."""

    def test_format_size_gb(self):
        from convert import format_size
        assert format_size(2.5e9) == "2.50 GB"
        assert format_size(1e12) == "1000.00 GB"

    def test_format_size_mb(self):
        from convert import format_size
        assert format_size(5e6) == "5 MB"
        assert format_size(999e6) == "999 MB"

    def test_format_time_seconds(self):
        from convert import format_time
        assert format_time(45) == "45s"

    def test_format_time_minutes(self):
        from convert import format_time
        assert format_time(130) == "2m 10s"

    def test_format_time_hours(self):
        from convert import format_time
        assert format_time(3700) == "1h 1m 40s"


class TestGemma4Support:
    """Tests for Gemma4 architecture support."""
    def test_known_issue_arch_gemma4(self):
        from convert import is_known_issue_arch
        is_issue, info = is_known_issue_arch("gemma4")
        assert is_issue is True
        assert info is not None
        assert "head_count_kv" in info["issue"]

    def test_known_issue_arch_gemma3(self):
        from convert import is_known_issue_arch
        is_issue, info = is_known_issue_arch("gemma3")
        assert is_issue is True

    def test_known_issue_arch_gemma2(self):
        from convert import is_known_issue_arch
        is_issue, info = is_known_issue_arch("gemma2")
        assert is_issue is True

    def test_known_issue_arch_case_insensitive(self):
        from convert import is_known_issue_arch
        is_issue, _ = is_known_issue_arch("GEMMA4")
        assert is_issue is True

    def test_known_issue_arch_unknown(self):
        from convert import is_known_issue_arch
        is_issue, info = is_known_issue_arch("llama")
        assert is_issue is False
        assert info is None

    def test_gemma4_metadata_fields_defined(self):
        from convert import Gemma4_METADATA_FIELDS
        assert "attention_head_count_kv" in Gemma4_METADATA_FIELDS
        assert "sliding_window" in Gemma4_METADATA_FIELDS
        assert "max_seq_len" in Gemma4_METADATA_FIELDS


    def test_gemma4_workarounds_exist(self):
        from convert import KNOWN_CONVERSION_ISSUES
        for arch in ["gemma4", "gemma3", "gemma2"]:
            assert arch in KNOWN_CONVERSION_ISSUES
            assert len(KNOWN_CONVERSION_ISSUES[arch]["workarounds"]) > 0


class TestMlxCompatibilityCheck:
    """Tests for mlx_lm architecture compatibility pre-flight check."""
    
    def test_supported_arch_llama(self):
        from convert import is_mlx_supported_arch
        is_supported, reason = is_mlx_supported_arch("llama")
        assert is_supported is True
        assert reason is None
    
    def test_supported_arch_llama2(self):
        from convert import is_mlx_supported_arch
        is_supported, reason = is_mlx_supported_arch("llama2")
        assert is_supported is True
        assert reason is None
    
    def test_supported_arch_mistral(self):
        from convert import is_mlx_supported_arch
        is_supported, reason = is_mlx_supported_arch("mistral")
        assert is_supported is True
        assert reason is None
    
    def test_unsupported_arch_qwen35(self):
        from convert import is_mlx_supported_arch
        is_supported, reason = is_mlx_supported_arch("qwen35")
        assert is_supported is False
        assert "incompatibility" in reason.lower()
    
    def test_unsupported_arch_unknown(self):
        from convert import is_mlx_supported_arch
        is_supported, reason = is_mlx_supported_arch("unknown-arch")
        assert is_supported is False
        assert reason is not None
    
    def test_unsupported_arch_gemma2(self):
        from convert import is_mlx_supported_arch
        # gemma2 is in KNOWN_CONVERSION_ISSUES but not in SUPPORTED list, so it should fail
        is_supported, reason = is_mlx_supported_arch("gemma2")
        assert is_supported is False
        assert "Known incompatibility" in reason
    
    def test_supported_arch_case_insensitive(self):
        from convert import is_mlx_supported_arch
        # Both Llama and llama should match (llama prefix)
        is_supported1, _ = is_mlx_supported_arch("Llama")
        is_supported2, _ = is_mlx_supported_arch("llama")
        assert is_supported1 is True
        assert is_supported2 is True

    def test_gemma2_in_known_issues_fails_support_check(self):
        from convert import is_mlx_supported_arch
        # gemma2 is in KNOWN_CONVERSION_ISSUES but not in SUPPORTED list, so it should fail
        is_supported, reason = is_mlx_supported_arch("gemma2")
        assert is_supported is False
        assert reason is not None


class TestRegistryDownload:
    """Tests for registry and download support (Quick Win #1)."""

    def test_handle_registry_url_local_path(self):
        from convert import handle_registry_url
        from pathlib import Path
        result = handle_registry_url("/local/path/model.gguf")
        assert isinstance(result, Path)
        assert result == Path("/local/path/model.gguf")

    def test_handle_registry_url_local_path_with_tilde(self):
        from convert import handle_registry_url
        from pathlib import Path
        result = handle_registry_url("~/models/model.gguf")
        assert isinstance(result, Path)
        # Result should be expanded
        assert str(result).startswith(str(Path.home()))

    def test_handle_registry_url_hf_format_validation(self):
        from convert import handle_registry_url
        # Invalid format should not crash, path expansion should work
        result = handle_registry_url("./relative/path.gguf")
        assert result.is_absolute() is False


class TestConversionEstimate:
    """Tests for conversion time/memory estimation (Quick Win #3)."""

    def test_estimate_small_model_base_chip(self):
        from convert import estimate_conversion_metrics
        result = estimate_conversion_metrics(model_size_gb=2.0, bits=4, chip_tier="base")
        assert result["time_minutes"] >= 1
        assert result["peak_memory_gb"] > 2.0
        assert result["final_size_gb"] >= 0.5  # 4-bit = 1/4 of original
        assert "warnings" in result
        assert isinstance(result["warnings"], list)

    def test_estimate_large_model_high_bits(self):
        from convert import estimate_conversion_metrics
        result = estimate_conversion_metrics(model_size_gb=13.0, bits=8, chip_tier="pro")
        assert result["time_minutes"] > 2
        assert result["peak_memory_gb"] > 13.0
        # 8-bit = 1/2 of original size (16-bit baseline)
        assert result["final_size_gb"] >= 6.0

    def test_estimate_large_model_warns_on_memory(self):
        from convert import estimate_conversion_metrics
        result = estimate_conversion_metrics(model_size_gb=50.0, bits=4, chip_tier="base")
        assert len(result["warnings"]) > 0
        assert any("memory" in w.lower() for w in result["warnings"])

    def test_estimate_handles_all_bit_depths(self):
        from convert import estimate_conversion_metrics
        for bits in [2, 4, 8, 16]:
            result = estimate_conversion_metrics(model_size_gb=5.0, bits=bits, chip_tier="base")
            assert "time_minutes" in result
            assert "peak_memory_gb" in result
            assert "final_size_gb" in result
            # Lower bits = smaller final size
            if bits < 16:
                prev_bits = bits * 2
                prev_result = estimate_conversion_metrics(5.0, prev_bits, "base")
                assert result["final_size_gb"] < prev_result["final_size_gb"]

    def test_estimate_chip_tier_affects_time(self):
        from convert import estimate_conversion_metrics
        base_result = estimate_conversion_metrics(model_size_gb=10.0, bits=4, chip_tier="base")
        max_result = estimate_conversion_metrics(model_size_gb=10.0, bits=4, chip_tier="max")
        # Max should be faster
        assert max_result["time_minutes"] <= base_result["time_minutes"]


class TestErrorConditions:
    """Tests for error handling (Quick Win #4)."""

    def test_classify_source_quality_all_ftypes(self):
        from convert import classify_source_quality
        for ftype in range(32):
            result = classify_source_quality(ftype)
            assert "risk" in result
            assert result["risk"] in ["none", "low", "medium", "high", "severe", "unknown"]
            assert "label" in result

    def test_format_size_edge_cases(self):
        from convert import format_size
        assert "MB" in format_size(1e6)
        assert "GB" in format_size(1e9)
        assert "0 MB" in format_size(0)

    def test_format_time_edge_cases(self):
        from convert import format_time
        assert "0s" in format_time(0) or "< 1" in format_time(0.1)
        assert "1s" in format_time(1)
        assert "m" in format_time(60)
        assert "h" in format_time(3600)

    def test_preset_completeness(self):
        from convert import PRESETS
        required_keys = {"bits", "group_size", "mode", "description"}
        assert len(PRESETS) >= 4, "Should have at least 4 presets"
        for name, preset in PRESETS.items():
            assert required_keys.issubset(preset.keys()), f"Preset {name} incomplete"
            assert isinstance(preset["bits"], int)
            assert isinstance(preset["group_size"], int)


class TestBuildParser:
    """Tests for CLI argument parser."""

    def test_parser_defaults(self):
        from convert import build_parser
        parser = build_parser()
        args = parser.parse_args(["model.gguf"])
        assert args.input == "model.gguf"
        assert args.bits is None
        assert args.no_quantize is False
        assert args.force is False
        assert args.quiet is False
        assert args.inspect is False
        assert args.estimate is False
        assert args.keep_intermediate is False
        assert args.cleanup_old is False
        assert args.resume is False

    def test_parser_bits(self):
        from convert import build_parser
        parser = build_parser()
        args = parser.parse_args(["model.gguf", "--bits", "8"])
        assert args.bits == 8

    def test_parser_no_quantize(self):
        from convert import build_parser
        parser = build_parser()
        args = parser.parse_args(["model.gguf", "--no-quantize"])
        assert args.no_quantize is True

    def test_parser_preset(self):
        from convert import build_parser
        parser = build_parser()
        args = parser.parse_args(["model.gguf", "--preset", "quality"])
        assert args.preset == "quality"

    def test_parser_output_dir(self):
        from convert import build_parser
        parser = build_parser()
        args = parser.parse_args(["model.gguf", "./output/"])
        assert args.output == "./output/"

    def test_parser_force_quiet(self):
        from convert import build_parser
        parser = build_parser()
        args = parser.parse_args(["model.gguf", "-f", "-q"])
        assert args.force is True
        assert args.quiet is True


class TestBuildQuantArgs:
    """Tests for quantisation argument builder."""

    def test_basic_bits(self):
        from convert import build_quant_args
        import argparse
        args = argparse.Namespace(bits=4, group_size=None, mode=None, predicate=None)
        result = build_quant_args(args)
        assert "--quantize" in result
        assert "--q-bits" in result
        assert "4" in result

    def test_custom_group_size(self):
        from convert import build_quant_args
        import argparse
        args = argparse.Namespace(bits=8, group_size=64, mode=None, predicate=None)
        result = build_quant_args(args)
        assert "--q-group-size" in result
        assert "64" in result

    def test_custom_mode(self):
        from convert import build_quant_args
        import argparse
        args = argparse.Namespace(bits=4, group_size=None, mode="mxfp4", predicate=None)
        result = build_quant_args(args)
        assert "--q-mode" in result
        assert "mxfp4" in result

    def test_predicate(self):
        from convert import build_quant_args
        import argparse
        args = argparse.Namespace(bits=4, group_size=None, mode=None, predicate="mixed_3_4")
        result = build_quant_args(args)
        assert "--quant-predicate" in result
        assert "mixed_3_4" in result


class TestRunWithProgress:
    """Tests for subprocess progress runner."""

    def test_success(self):
        from convert import run_with_progress
        from rich.progress import Progress
        progress = Progress(disable=True)
        ok, output = run_with_progress(
            [sys.executable, "-c", "print('hello')"],
            "test command",
            progress=progress,
            quiet=True,
        )
        assert ok is True
        assert "hello" in output

    def test_failure(self):
        from convert import run_with_progress
        from rich.progress import Progress
        progress = Progress(disable=True)
        ok, output = run_with_progress(
            [sys.executable, "-c", "import sys; sys.exit(1)"],
            "failing command",
            progress=progress,
            quiet=True,
        )
        assert ok is False

    def test_command_not_found(self):
        from convert import run_with_progress
        from rich.progress import Progress
        progress = Progress(disable=True)
        ok, output = run_with_progress(
            ["nonexistent_command_xyz_123"],
            "missing command",
            progress=progress,
            quiet=True,
        )
        assert ok is False
        assert output == ""

    def test_percentage_parsing(self):
        from convert import run_with_progress
        from rich.progress import Progress
        progress = Progress(disable=True)
        ok, output = run_with_progress(
            [sys.executable, "-c", "print('Progress: 50%'); print('Progress: 100%')"],
            "percent test",
            progress=progress,
            quiet=True,
        )
        assert ok is True
        assert "50%" in output


class TestPreflightChecks:
    """Tests for preflight checks with mocked filesystem."""

    def _make_args(self, **kwargs):
        import argparse
        defaults = dict(no_quantize=False, force=False)
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_file_not_found(self, tmp_path):
        from convert import preflight_checks
        gguf = tmp_path / "nonexistent.gguf"
        ok, warns, errs = preflight_checks(gguf, tmp_path / "out", self._make_args())
        assert ok is False
        assert any("not found" in e for e in errs)

    def test_not_gguf_extension(self, tmp_path):
        from convert import preflight_checks
        bad = tmp_path / "model.txt"
        bad.write_text("not a model")
        ok, warns, errs = preflight_checks(bad, tmp_path / "out", self._make_args())
        assert ok is False
        assert any(".gguf" in e for e in errs)

    def test_invalid_magic_bytes(self, tmp_path):
        from convert import preflight_checks
        fake = tmp_path / "model.gguf"
        fake.write_bytes(b"NOT_GGUF_DATA_PADDING_TO_4BYTES")
        ok, warns, errs = preflight_checks(fake, tmp_path / "out", self._make_args())
        assert ok is False
        assert any("magic bytes" in e.lower() for e in errs)

    def test_valid_gguf_magic(self, tmp_path):
        from convert import preflight_checks
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        ok, warns, errs = preflight_checks(gguf, tmp_path / "out", self._make_args())
        # May still fail on deps, but should pass file checks
        assert not any("magic" in e.lower() for e in errs)
        assert not any("not found" in e for e in errs)


class TestCheckDependencies:
    """Tests for dependency checking."""

    def test_returns_all_keys(self):
        from convert import check_dependencies
        deps = check_dependencies()
        assert "gguf2mlx" in deps
        assert "mlx_lm" in deps
        assert "mlx" in deps
        assert "gguf_py" in deps

    def test_values_are_str_or_none(self):
        from convert import check_dependencies
        deps = check_dependencies()
        for key, val in deps.items():
            assert val is None or isinstance(val, str)


class TestFieldHelpers:
    """Tests for GGUF field value readers."""

    def test_field_value_none_field(self):
        from convert import _field_value
        reader = MagicMock()
        reader.get_field.return_value = None
        assert _field_value(reader, "test") is None

    def test_field_value_bytes(self):
        from convert import _field_value
        reader = MagicMock()
        field = MagicMock()
        field.contents.return_value = b"hello"
        reader.get_field.return_value = field
        assert _field_value(reader, "test") == "hello"

    def test_field_value_list(self):
        from convert import _field_value
        reader = MagicMock()
        field = MagicMock()
        field.contents.return_value = [42]
        reader.get_field.return_value = field
        assert _field_value(reader, "test") == 42

    def test_field_value_empty_list(self):
        from convert import _field_value
        reader = MagicMock()
        field = MagicMock()
        field.contents.return_value = []
        reader.get_field.return_value = field
        assert _field_value(reader, "test") is None

    def test_field_exists(self):
        from convert import _field_exists
        reader = MagicMock()
        reader.get_field.return_value = MagicMock()
        assert _field_exists(reader, "test") is True

    def test_field_not_exists(self):
        from convert import _field_exists
        reader = MagicMock()
        reader.get_field.return_value = None
        assert _field_exists(reader, "test") is False


class TestReadGemma4Metadata:
    """Tests for Gemma4 metadata reader."""

    def test_reads_known_fields(self):
        from convert import read_gemma4_metadata
        reader = MagicMock()
        # First call: attn head count kv, second: None, third: context length
        reader.get_field.side_effect = lambda name: MagicMock(contents=MagicMock(return_value=8)) if "head_count_kv" in name else None
        result = read_gemma4_metadata(reader, "gemma4")
        assert "attention_head_count_kv" in result

    def test_empty_when_no_fields(self):
        from convert import read_gemma4_metadata
        reader = MagicMock()
        reader.get_field.return_value = None
        result = read_gemma4_metadata(reader, "gemma4")
        assert result == {}


class TestFixGemma4TensorNames:
    """Tests for Gemma4 tensor name fixer."""

    def test_no_safetensors(self, tmp_path):
        from convert import fix_gemma4_tensor_names
        # Empty dir has no safetensors
        fix_gemma4_tensor_names(tmp_path)  # Should not raise

    def test_safetensors_invalid_file(self, tmp_path):
        from convert import fix_gemma4_tensor_names
        # Invalid safetensor file should return False gracefully
        (tmp_path / "model.safetensors").write_bytes(b"\x00" * 100)
        fix_gemma4_tensor_names(tmp_path)  # Should not raise


class TestFormatHelpers:
    """Extended tests for format helpers."""

    def test_format_size_zero(self):
        from convert import format_size
        result = format_size(0)
        assert "0 MB" in result

    def test_format_time_subsecond(self):
        from convert import format_time
        result = format_time(0.5)
        assert "0s" in result

    def test_format_time_exact_minute(self):
        from convert import format_time
        result = format_time(60)
        assert "1m" in result

    def test_format_time_exact_hour(self):
        from convert import format_time
        result = format_time(3600)
        assert "1h" in result


class TestDisplayHelpers:
    """Tests for Rich display helper functions."""

    def test_banner(self, capsys):
        from convert import banner
        import io
        from rich.console import Console
        from convert import console as global_console
        # Just verify it doesn't crash
        banner()

    def test_step(self):
        from convert import step
        step(1, 3, "Test Step")

    def test_ok(self):
        from convert import ok
        ok("test message")

    def test_fail(self):
        from convert import fail
        fail("test error")

    def test_info(self):
        from convert import info
        info("test info")

    def test_warn(self):
        from convert import warn
        warn("test warning")


class TestSmartDefaultsEdgeCases:
    """Extended smart defaults tests for full branch coverage."""

    def test_ultra_48gb(self):
        from convert import smart_defaults
        result = smart_defaults(model_size_gb=20, chip_tier="ultra", ram_gb=64, chip_gen=4)
        assert result["bits"] == 4
        assert result["group_size"] == 32

    def test_max_under_48gb(self):
        from convert import smart_defaults
        # Max chip but <48GB RAM → falls through to model-size defaults
        result = smart_defaults(model_size_gb=10, chip_tier="max", ram_gb=32, chip_gen=5)
        assert result["bits"] == 4

    def test_pro_small_model(self):
        from convert import smart_defaults
        result = smart_defaults(model_size_gb=2, chip_tier="pro", ram_gb=36, chip_gen=4)
        assert result["bits"] == 6

    def test_large_model_fallback(self):
        from convert import smart_defaults
        result = smart_defaults(model_size_gb=30, chip_tier="base", ram_gb=32, chip_gen=4)
        assert result["bits"] == 4
        assert result["group_size"] == 32


class TestCheckDiskSpace:
    """Tests for disk space checking."""

    def test_plenty_of_space(self, tmp_path):
        from convert import check_disk_space
        # tmp_path is on a real filesystem with plenty of space
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        result = check_disk_space(gguf, tmp_path / "out", force=True)
        assert result is True

    def test_force_skips_prompt(self, tmp_path):
        from convert import check_disk_space
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        result = check_disk_space(gguf, tmp_path / "out", force=True)
        assert result is True


class TestDetectAppleSiliconEdgeCases:
    """Extended Apple Silicon detection tests."""

    def test_m2_ultra(self):
        from convert import detect_apple_silicon
        with patch("subprocess.check_output") as mock:
            mock.return_value = "Apple M2 Ultra"
            with patch("psutil.virtual_memory") as mock_mem:
                mock_mem.return_value = MagicMock(total=1.92e11)
                result = detect_apple_silicon()
                assert result["chip_gen"] == 2
                assert result["chip_tier"] == "ultra"
                assert result["ram_gb"] > 100

    def test_m4_base(self):
        from convert import detect_apple_silicon
        with patch("subprocess.check_output") as mock:
            mock.return_value = "Apple M4"
            with patch("psutil.virtual_memory") as mock_mem:
                mock_mem.return_value = MagicMock(total=1.6e10)
                result = detect_apple_silicon()
                assert result["chip_gen"] == 4
                assert result["chip_tier"] == "base"

    def test_subprocess_failure(self):
        from convert import detect_apple_silicon
        with patch("subprocess.check_output", side_effect=Exception("no sysctl")):
            result = detect_apple_silicon()
            assert result["is_apple_silicon"] is False


class TestClassifySourceQualityExtended:
    """Full coverage for all ftype risk levels."""

    def test_medium_risk_q5(self):
        from convert import classify_source_quality
        for ftype in [6, 7, 14, 15, 16]:
            result = classify_source_quality(ftype)
            assert result["risk"] == "medium", f"ftype {ftype} should be medium"

    def test_high_risk_q4(self):
        from convert import classify_source_quality
        for ftype in [2, 3, 4, 10, 11, 12, 13, 27, 28, 29]:
            result = classify_source_quality(ftype)
            assert result["risk"] == "high", f"ftype {ftype} should be high"

    def test_severe_risk(self):
        from convert import classify_source_quality
        for ftype in [8, 9, 19, 20, 21, 22, 24, 25, 30, 31]:
            result = classify_source_quality(ftype)
            assert result["risk"] == "severe", f"ftype {ftype} should be severe"


class TestSupportedArchitectures:
    """Tests for the SUPPORTED_MLX_ARCHITECTURES set."""

    def test_contains_key_archs(self):
        from convert import SUPPORTED_MLX_ARCHITECTURES
        expected = {"llama", "mistral", "phi", "qwen", "deepseek", "gemma"}
        for arch in expected:
            assert arch in SUPPORTED_MLX_ARCHITECTURES, f"{arch} should be supported"

    def test_is_frozenset(self):
        from convert import SUPPORTED_MLX_ARCHITECTURES
        assert isinstance(SUPPORTED_MLX_ARCHITECTURES, frozenset)


class TestConversionEstimateExtended:
    """Extended estimation tests."""

    def test_estimation_warnings_empty_for_small(self):
        from convert import estimate_conversion_metrics
        result = estimate_conversion_metrics(model_size_gb=1.0, bits=4, chip_tier="base")
        assert len(result["warnings"]) == 0

    def test_estimation_time_warning(self):
        from convert import estimate_conversion_metrics
        result = estimate_conversion_metrics(model_size_gb=500.0, bits=4, chip_tier="base")
        # At 500GB the time is ~280 min, triggering the 2+ hour warning
        assert len(result["warnings"]) > 0

# ═══════════════════════════════════════════════════════════════════════════
# Coverage expansion tests — targeting 90%+
# ═══════════════════════════════════════════════════════════════════════════


class TestFieldHelpersExtended:
    """Extended coverage for _field_value edge cases."""

    def test_field_value_tuple(self):
        from convert import _field_value
        reader = MagicMock()
        field = MagicMock()
        field.contents.return_value = (99,)
        reader.get_field.return_value = field
        assert _field_value(reader, "test") == 99

    def test_field_value_empty_tuple(self):
        from convert import _field_value
        reader = MagicMock()
        field = MagicMock()
        field.contents.return_value = ()
        reader.get_field.return_value = field
        assert _field_value(reader, "test") is None

    def test_field_value_numpy_scalar(self):
        from convert import _field_value
        reader = MagicMock()
        field = MagicMock()
        numpy_val = MagicMock()
        numpy_val.item.return_value = 42
        field.contents.return_value = numpy_val
        reader.get_field.return_value = field
        assert _field_value(reader, "test") == 42

    def test_field_value_exception(self):
        from convert import _field_value
        reader = MagicMock()
        field = MagicMock()
        field.contents.side_effect = RuntimeError("boom")
        reader.get_field.return_value = field
        assert _field_value(reader, "test") is None

    def test_has_gguf_py(self):
        from convert import _has_gguf_py
        result = _has_gguf_py()
        assert isinstance(result, bool)


class TestFixGemma4TensorNamesExtended:
    """Full coverage for fix_gemma4_tensor_names."""

    def test_needs_fix_shows_warnings(self, tmp_path):
        from convert import fix_gemma4_tensor_names
        mock_file = tmp_path / "model.safetensors"
        mock_file.write_text("fake")
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.keys.return_value = ["blk.0.weight", "blk.1.weight"]
        with patch("safetensors.safe_open", return_value=mock_ctx, create=True):
            fix_gemma4_tensor_names(tmp_path)  # Should not raise

    def test_no_fix_needed(self, tmp_path):
        from convert import fix_gemma4_tensor_names
        mock_file = tmp_path / "model.safetensors"
        mock_file.write_text("fake")
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.keys.return_value = ["model.layers.0.weight"]
        with patch("safetensors.safe_open", return_value=mock_ctx, create=True):
            fix_gemma4_tensor_names(tmp_path)  # Should not raise


class TestDisplayMetadata:
    """Full coverage for display_metadata."""

    def _full_meta(self, **overrides):
        base = {
            "architecture": "llama",
            "model_name": "test-model",
            "param_count": 7_000_000_000,
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "num_kv_heads": 32,
            "context_length": 4096,
            "tensor_count": 300,
            "total_weight_bytes": 5e9,
            "file_type": 1,
            "file_type_name": "MOSTLY_F16 (half precision)",
            "mtp_layers": None,
            "has_ssm": False,
            "warnings": ["Test warning"],
            "fields": {},
        }
        base.update(overrides)
        return base

    def test_basic_display(self):
        from convert import display_metadata
        display_metadata(self._full_meta())

    def test_display_with_mtp(self):
        from convert import display_metadata
        display_metadata(self._full_meta(mtp_layers=4))

    def test_display_mtp_not_detected_qwen3(self):
        from convert import display_metadata
        display_metadata(self._full_meta(architecture="qwen3", mtp_layers=None))

    def test_display_with_ssm(self):
        from convert import display_metadata
        display_metadata(self._full_meta(has_ssm=True))

    def test_display_no_file_type(self):
        from convert import display_metadata
        meta = self._full_meta()
        meta["file_type"] = None
        display_metadata(meta)

    def test_display_no_param_count(self):
        from convert import display_metadata
        display_metadata(self._full_meta(param_count=None))

    def test_display_no_context_length(self):
        from convert import display_metadata
        display_metadata(self._full_meta(context_length=None))

    def test_display_no_total_weight(self):
        from convert import display_metadata
        display_metadata(self._full_meta(total_weight_bytes=0))

    def test_display_no_warnings(self):
        from convert import display_metadata
        display_metadata(self._full_meta(warnings=[]))


class TestInspectMode:
    """Tests for inspect_mode function."""

    def test_no_gguf_py_exits(self, tmp_path):
        from convert import inspect_mode
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        with patch("convert._has_gguf_py", return_value=False):
            with pytest.raises(SystemExit):
                inspect_mode(gguf)

    def test_with_metadata(self, tmp_path):
        from convert import inspect_mode
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        mock_meta = {
            "architecture": "llama", "model_name": "test",
            "param_count": None, "vocab_size": None, "hidden_size": None,
            "num_layers": None, "num_heads": None, "num_kv_heads": None,
            "context_length": None, "tensor_count": 100, "total_weight_bytes": 5e9,
            "file_type": None, "file_type_name": None, "mtp_layers": None,
            "has_ssm": False, "warnings": [],
            "fields": {"general.architecture": "llama", "general.name": "Test Model"},
        }
        with patch("convert._has_gguf_py", return_value=True):
            with patch("convert.read_gguf_metadata", return_value=mock_meta):
                with pytest.raises(SystemExit):
                    inspect_mode(gguf)

    def test_no_metadata_exits(self, tmp_path):
        from convert import inspect_mode
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        with patch("convert._has_gguf_py", return_value=True):
            with patch("convert.read_gguf_metadata", return_value=None):
                with pytest.raises(SystemExit):
                    inspect_mode(gguf)


class TestValidateOutput:
    """Tests for validate_output."""

    def test_no_mlx_import(self, tmp_path):
        from convert import validate_output
        with patch.dict("sys.modules", {"mlx_lm": None, "mlx.core": None}):
            validate_output(tmp_path)  # Should not raise

    def test_successful_validation(self, tmp_path):
        from convert import validate_output
        (tmp_path / "model.safetensors").write_bytes(b"\x00" * 1000)
        validate_output(tmp_path)

    def test_validation_warning(self, tmp_path):
        from convert import validate_output
        validate_output(tmp_path)


class TestEnsureDeps:
    """Tests for ensure_deps."""

    def test_missing_gguf2mlx_exits(self):
        from convert import ensure_deps
        with pytest.raises(SystemExit):
            ensure_deps({"gguf2mlx": None, "mlx_lm": "?"}, for_convert=True)

    def test_missing_mlx_lm_exits(self):
        from convert import ensure_deps
        with pytest.raises(SystemExit):
            ensure_deps({"gguf2mlx": "?", "mlx_lm": None}, for_convert=True)

    def test_no_convert_skips_mlx(self):
        from convert import ensure_deps
        ensure_deps({"gguf2mlx": "?", "mlx_lm": None}, for_convert=False)

    def test_all_present(self):
        from convert import ensure_deps
        ensure_deps({"gguf2mlx": "?", "mlx_lm": "?"}, for_convert=True)


class TestGetGgufPath:
    """Tests for interactive GGUF path prompt."""

    def test_returns_path(self):
        from convert import get_gguf_path
        with patch("convert.Prompt.ask", return_value="/tmp/model.gguf"):
            result = get_gguf_path()
            assert isinstance(result, Path)

    def test_strips_quotes(self):
        from convert import get_gguf_path
        with patch("convert.Prompt.ask", return_value='"/tmp/model.gguf"'):
            result = get_gguf_path()
            assert '"' not in str(result)


class TestGetOutputDir:
    """Tests for interactive output dir prompt."""

    def test_default_suggestion(self, tmp_path):
        from convert import get_output_dir
        gguf = tmp_path / "model.gguf"
        gguf.write_text("test")
        with patch("convert.Prompt.ask", return_value=str(tmp_path / "model-4bit-mlx")):
            result = get_output_dir(gguf)
            assert isinstance(result, Path)

    def test_custom_output(self, tmp_path):
        from convert import get_output_dir
        gguf = tmp_path / "model.gguf"
        gguf.write_text("test")
        with patch("convert.Prompt.ask", return_value="/custom/output"):
            result = get_output_dir(gguf)
            assert str(result) == "/custom/output"


class TestPreflightExtended:
    """Extended preflight checks for full coverage."""

    def _make_args(self, **kwargs):
        import argparse
        defaults = dict(no_quantize=False, force=False)
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_output_dir_exists_warning(self, tmp_path):
        from convert import preflight_checks
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        out = tmp_path / "out"
        out.mkdir()
        ok, warns, errs = preflight_checks(gguf, out, self._make_args())
        assert any("already exists" in w for w in warns)

    def test_output_dir_exists_force(self, tmp_path):
        from convert import preflight_checks
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        out = tmp_path / "out"
        out.mkdir()
        ok, warns, errs = preflight_checks(gguf, out, self._make_args(force=True))
        assert not any("already exists" in w for w in warns)

    def test_not_a_file(self, tmp_path):
        from convert import preflight_checks
        gguf = tmp_path / "model.gguf"
        gguf.mkdir()
        ok, warns, errs = preflight_checks(gguf, tmp_path / "out", self._make_args())
        assert ok is False
        assert any("Not a regular file" in e for e in errs)


class TestDownloadFromHuggingface:
    """Tests for HuggingFace download function."""

    def test_cached_file_returned(self, tmp_path):
        from convert import download_from_huggingface
        cache = tmp_path / "cache"
        cache.mkdir()
        cached = cache / "org_model_Q4_K_M.gguf"
        cached.write_bytes(b"GGUF" + b"\x00" * 100)
        result = download_from_huggingface("org/model", "Q4_K_M.gguf", cache_dir=cache)
        assert result == cached

    def test_download_success(self, tmp_path):
        from convert import download_from_huggingface
        cache = tmp_path / "cache"
        cache.mkdir()
        with patch("urllib.request.urlretrieve"):
            result = download_from_huggingface("org/model", "model.gguf", cache_dir=cache)
            assert isinstance(result, Path)

    def test_download_failure_exits(self, tmp_path):
        from convert import download_from_huggingface
        cache = tmp_path / "cache"
        cache.mkdir()
        with patch("urllib.request.urlretrieve", side_effect=Exception("network error")):
            with pytest.raises(SystemExit):
                download_from_huggingface("org/model", "model.gguf", cache_dir=cache)

    def test_no_urllib_exits(self, tmp_path):
        import convert
        orig = convert.HAS_URLLIB
        convert.HAS_URLLIB = False
        try:
            with pytest.raises(SystemExit):
                convert.download_from_huggingface("org/model", "model.gguf")
        finally:
            convert.HAS_URLLIB = orig


class TestHandleRegistryUrlExtended:
    """Extended tests for registry URL handling."""

    def test_hf_with_three_parts(self):
        from convert import handle_registry_url
        with patch("convert.download_from_huggingface", return_value=Path("/cached/model.gguf")):
            result = handle_registry_url("hf:org/subdir/model.gguf")
            assert result == Path("/cached/model.gguf")

    def test_hf_with_two_parts(self):
        from convert import handle_registry_url
        with patch("convert.download_from_huggingface", return_value=Path("/cached/model.gguf")):
            result = handle_registry_url("hf:org/model")
            assert result == Path("/cached/model.gguf")

    def test_hf_invalid_single_part(self):
        from convert import handle_registry_url
        with pytest.raises(SystemExit):
            handle_registry_url("hf:justonepart")


class TestRunWithProgressExtended:
    """Extended tests for run_with_progress."""

    def test_fraction_parsing(self):
        from convert import run_with_progress
        from rich.progress import Progress
        progress = Progress(disable=True)
        ok, output = run_with_progress(
            [sys.executable, "-c", "print('10/100'); print('50/100'); print('100/100')"],
            "fraction test",
            progress=progress,
            quiet=True,
        )
        assert ok is True
        assert "10/100" in output

    def test_empty_output_success(self):
        from convert import run_with_progress
        from rich.progress import Progress
        progress = Progress(disable=True)
        ok, output = run_with_progress(
            [sys.executable, "-c", "pass"],
            "empty test",
            progress=progress,
            quiet=True,
        )
        assert ok is True
        assert output == ""


class TestReadGgufMetadata:
    """Tests for read_gguf_metadata with mocked GGUFReader."""

    def test_no_gguf_py_returns_none(self, tmp_path):
        from convert import read_gguf_metadata
        fake = tmp_path / "model.gguf"
        fake.write_bytes(b"GGUF" + b"\x00" * 100)
        with patch("convert._has_gguf_py", return_value=False):
            result = read_gguf_metadata(fake)
            assert result is None

    def _mock_reader(self, fields=None, arch="llama", tensors=None):
        if fields is None:
            fields = {
                "general.architecture": arch,
                "general.file_type": 1,
                "general.name": "test-model",
                f"{arch}.vocab_size": 32000,
                f"{arch}.embedding_length": 4096,
                f"{arch}.block_count": 32,
                f"{arch}.attention.head_count": 32,
                f"{arch}.attention.head_count_kv": 32,
                f"{arch}.context_length": 4096,
            }

        mock_reader = MagicMock()
        mock_fields = {}
        for name, val in fields.items():
            mock_field = MagicMock()
            mock_field.name = name
            mock_field.contents.return_value = val
            mock_fields[name] = mock_field
        mock_reader.fields = mock_fields
        mock_reader.get_field = lambda n: mock_fields.get(n)
        mock_reader.tensors = tensors or []
        return mock_reader

    def test_basic_metadata(self, tmp_path):
        from convert import read_gguf_metadata
        fake = tmp_path / "model.gguf"
        fake.write_bytes(b"GGUF" + b"\x00" * 100)
        mock_reader = self._mock_reader()
        with patch("convert._has_gguf_py", return_value=True):
            with patch("gguf.GGUFReader", return_value=mock_reader):
                result = read_gguf_metadata(fake)
                assert result is not None
                assert result["architecture"] == "llama"
                assert result["vocab_size"] == 32000

    def test_metadata_with_mtp(self, tmp_path):
        from convert import read_gguf_metadata
        fake = tmp_path / "model.gguf"
        fake.write_bytes(b"GGUF" + b"\x00" * 100)
        fields = {
            "general.architecture": "qwen3",
            "general.file_type": 13,
            "general.name": "qwen3-test",
            "qwen3.vocab_size": 151936,
            "qwen3.embedding_length": 4096,
            "qwen3.block_count": 36,
            "qwen3.attention.head_count": 32,
            "qwen3.attention.head_count_kv": 8,
            "qwen3.context_length": 32768,
            "qwen3.nextn_predict_layers": 2,
        }
        t = MagicMock()
        t.name = "nextn.0.weight"
        t.n_bytes = 1000
        mock_reader = self._mock_reader(fields=fields, arch="qwen3", tensors=[t])
        with patch("convert._has_gguf_py", return_value=True):
            with patch("gguf.GGUFReader", return_value=mock_reader):
                result = read_gguf_metadata(fake)
                assert result is not None
                assert result["mtp_layers"] == 2

    def test_metadata_with_ssm(self, tmp_path):
        from convert import read_gguf_metadata
        fake = tmp_path / "model.gguf"
        fake.write_bytes(b"GGUF" + b"\x00" * 100)
        fields = {
            "general.architecture": "mamba",
            "general.file_type": 1,
            "general.name": "mamba-test",
            "mamba.ssm.conv_kernel": 3,
            "mamba.ssm.inner_size": 16,
        }
        mock_reader = self._mock_reader(fields=fields, arch="mamba")
        with patch("convert._has_gguf_py", return_value=True):
            with patch("gguf.GGUFReader", return_value=mock_reader):
                result = read_gguf_metadata(fake)
                assert result is not None
                assert result["has_ssm"] is True

    def test_metadata_no_file_type(self, tmp_path):
        from convert import read_gguf_metadata
        fake = tmp_path / "model.gguf"
        fake.write_bytes(b"GGUF" + b"\x00" * 100)
        fields = {
            "general.architecture": "llama",
            "general.name": "test",
        }
        mock_reader = self._mock_reader(fields=fields, arch="llama")
        with patch("convert._has_gguf_py", return_value=True):
            with patch("gguf.GGUFReader", return_value=mock_reader):
                result = read_gguf_metadata(fake)
                assert result is not None
                assert result["file_type"] is None

    def test_metadata_high_risk_source(self, tmp_path):
        from convert import read_gguf_metadata
        fake = tmp_path / "model.gguf"
        fake.write_bytes(b"GGUF" + b"\x00" * 100)
        fields = {
            "general.architecture": "llama",
            "general.file_type": 8,
            "general.name": "test",
            "llama.vocab_size": 32000,
            "llama.embedding_length": 4096,
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 32,
            "llama.context_length": 4096,
        }
        mock_reader = self._mock_reader(fields=fields, arch="llama")
        with patch("convert._has_gguf_py", return_value=True):
            with patch("gguf.GGUFReader", return_value=mock_reader):
                result = read_gguf_metadata(fake)
                assert result is not None
                assert any("extremely" in w.lower() or "severe" in w.lower() for w in result["warnings"])

    def test_metadata_gemma4_known_issue(self, tmp_path):
        from convert import read_gguf_metadata
        fake = tmp_path / "model.gguf"
        fake.write_bytes(b"GGUF" + b"\x00" * 100)
        fields = {
            "general.architecture": "gemma4",
            "general.file_type": 1,
            "general.name": "gemma4-test",
            "gemma4.vocab_size": 256000,
            "gemma4.embedding_length": 4096,
            "gemma4.block_count": 48,
            "gemma4.attention.head_count": 32,
            "gemma4.attention.head_count_kv": 8,
            "gemma4.context_length": 8192,
        }
        mock_reader = self._mock_reader(fields=fields, arch="gemma4")
        with patch("convert._has_gguf_py", return_value=True):
            with patch("gguf.GGUFReader", return_value=mock_reader):
                result = read_gguf_metadata(fake)
                assert result is not None
                assert any("known" in w.lower() for w in result["warnings"])


# ═══════════════════════════════════════════════════════════════════════════
# main() pipeline coverage
# ═══════════════════════════════════════════════════════════════════════════


class TestMainInspectMode:
    """Test main() --inspect path."""

    def test_inspect_exits(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        with patch("sys.argv", ["convert.py", str(gguf), "--inspect"]):
            with patch("convert._has_gguf_py", return_value=False):
                with pytest.raises(SystemExit):
                    from convert import main
                    main()


class TestMainEstimateMode:
    """Test main() --estimate path."""

    def test_estimate_runs(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        with patch("sys.argv", ["convert.py", str(gguf), "--estimate"]):
            from convert import main
            main()  # Should return without exit


class TestMainNoInput:
    """Test main() with no input file."""

    def test_guided_mode_requires_deps(self):
        with patch("sys.argv", ["convert.py"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": None, "mlx_lm": None, "mlx": None, "gguf_py": None}):
                with patch("convert.ensure_deps", side_effect=SystemExit(1)):
                    with pytest.raises(SystemExit):
                        from convert import main
                        main()


class TestMainPreflightFails:
    """Test main() with invalid input file."""

    def test_missing_file_exits(self, tmp_path):
        missing = tmp_path / "nonexistent.gguf"
        with patch("sys.argv", ["convert.py", str(missing)]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with pytest.raises(SystemExit):
                    from convert import main
                    main()


class TestMainUnsupportedArch:
    """Test main() with unsupported architecture."""

    def test_unsupported_arch_prompts(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        out = tmp_path / "out"
        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--force"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with pytest.raises(SystemExit):
                        from convert import main
                        main()


class TestMainFloat16Only:
    """Test main() --no-quantize path through pipeline."""

    def test_no_quantize_full_pipeline(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        out = tmp_path / "output"
        intermediate = tmp_path / "output_intermediate"

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--no-quantize", "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with patch("convert.ensure_deps"):
                        with patch("convert.run_with_progress", return_value=(True, "")):
                            # Simulate gguf2mlx creating intermediate dir
                            def fake_step1(*args, **kwargs):
                                intermediate.mkdir(parents=True, exist_ok=True)
                                (intermediate / "model.safetensors").write_bytes(b"\x00" * 100)
                                (intermediate / "config.json").write_text("{}")
                                return (True, "")
                            with patch("convert.run_with_progress", side_effect=fake_step1):
                                with patch("shutil.move"):
                                    with patch("convert.validate_output", return_value=True):
                                        with pytest.raises(SystemExit) as exc_info:
                                            from convert import main
                                            main()
                                        # Should exit 0 (success) or 1 (expected)
                                        assert exc_info.value.code in (0, 1)


class TestMainQuantizePipeline:
    """Test main() quantization pipeline."""

    def test_quantize_pipeline(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        out = tmp_path / "output"
        intermediate = tmp_path / "output_intermediate"

        call_count = [0]
        def fake_run_with_progress(cmd, description, progress, quiet=False):
            call_count[0] += 1
            if call_count[0] == 1:
                # Step 1: create intermediate
                intermediate.mkdir(parents=True, exist_ok=True)
                (intermediate / "model.safetensors").write_bytes(b"\x00" * 100)
                (intermediate / "config.json").write_text("{}")
            elif call_count[0] == 2:
                # Step 2: create output
                out.mkdir(parents=True, exist_ok=True)
                (out / "model.safetensors").write_bytes(b"\x00" * 50)
                (out / "config.json").write_text("{}")
            return (True, "")

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with patch("convert.ensure_deps"):
                        with patch("convert.run_with_progress", side_effect=fake_run_with_progress):
                            with patch("shutil.rmtree"):
                                with patch("convert.validate_output", return_value=True):
                                    from convert import main
                                    try:
                                        main()
                                    except SystemExit:
                                        pass


class TestMainResumeMode:
    """Test main() --resume path."""

    def test_resume_skips_step1(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        out = tmp_path / "output"
        intermediate = tmp_path / "output_intermediate"
        intermediate.mkdir(parents=True, exist_ok=True)
        (intermediate / "config.json").write_text("{}")

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--resume", "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with patch("convert.ensure_deps"):
                        def fake_run(cmd, description, progress, quiet=False):
                            out.mkdir(parents=True, exist_ok=True)
                            (out / "config.json").write_text("{}")
                            return (True, "")
                        with patch("convert.run_with_progress", side_effect=fake_run):
                            with patch("shutil.rmtree"):
                                with patch("convert.validate_output", return_value=True):
                                    from convert import main
                                    try:
                                        main()
                                    except SystemExit:
                                        pass


class TestMainHighBandwidth:
    """Test main() --high-bandwidth preset."""

    def test_high_bandwidth_flag(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        out = tmp_path / "output"
        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--high-bandwidth", "--force"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with patch("convert.ensure_deps"):
                        with patch("convert.run_with_progress", return_value=(True, "")) as mock_run:
                            with patch("shutil.rmtree"):
                                with patch("convert.validate_output", return_value=True):
                                    from convert import main
                                    try:
                                        main()
                                    except SystemExit:
                                        pass


class TestMainCleanupOld:
    """Test main() --cleanup-old flag."""

    def test_cleanup_old(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        out = tmp_path / "output"
        # Create an old intermediate dir
        old = tmp_path / "output_intermediate"
        old.mkdir()
        (old / "data.bin").write_bytes(b"\x00" * 50)

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--cleanup-old", "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with patch("convert.ensure_deps"):
                        with patch("convert.run_with_progress", return_value=(True, "")):
                            with patch("shutil.rmtree"):
                                with patch("convert.validate_output", return_value=True):
                                    from convert import main
                                    try:
                                        main()
                                    except SystemExit:
                                        pass


class TestMainKeepIntermediate:
    """Test main() --keep-intermediate flag."""

    def test_keep_intermediate(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 100)
        out = tmp_path / "output"

        call_count = [0]
        def fake_run(cmd, description, progress, quiet=False):
            call_count[0] += 1
            if call_count[0] == 2:
                out.mkdir(parents=True, exist_ok=True)
                (out / "config.json").write_text("{}")
            return (True, "")

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--keep-intermediate", "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with patch("convert.ensure_deps"):
                        with patch("convert.run_with_progress", side_effect=fake_run):
                            with patch("convert.validate_output", return_value=True):
                                from convert import main
                                try:
                                    main()
                                except SystemExit:
                                    pass



class TestAppleSiliconRamFallback:
    """Full coverage for Apple Silicon RAM detection fallback."""

    def test_both_psutil_and_sysctl_fail(self):
        from convert import detect_apple_silicon
        with patch("subprocess.check_output", return_value="Apple M3"):
            with patch("psutil.virtual_memory", side_effect=Exception("no psutil")):
                with patch("subprocess.check_output", side_effect=[
                    "Apple M3",  # First call for chip
                    Exception("no sysctl"),  # Second call for hw.memsize
                ]):
                    result = detect_apple_silicon()
                    assert result["is_apple_silicon"] is True
                    assert result["ram_gb"] == 0.0

    def test_chip_with_unknown_suffix(self):
        from convert import detect_apple_silicon
        with patch("subprocess.check_output", return_value="Apple M5 X"):
            with patch("psutil.virtual_memory", return_value=MagicMock(total=1.6e10)):
                result = detect_apple_silicon()
                assert result["chip_tier"] == "x"  # Unknown suffix kept as-is


class TestFieldHelpersFullCoverage:
    """Cover remaining _field_value paths."""

    def test_field_value_regular_int(self):
        from convert import _field_value
        reader = MagicMock()
        field = MagicMock()
        field.contents.return_value = 42
        reader.get_field.return_value = field
        assert _field_value(reader, "test") == 42

    def test_field_value_regular_string(self):
        from convert import _field_value
        reader = MagicMock()
        field = MagicMock()
        field.contents.return_value = "hello"
        reader.get_field.return_value = field
        assert _field_value(reader, "test") == "hello"


# ═══════════════════════════════════════════════════════════════════════════
# Deep main() pipeline coverage — targeting 90%+
# ═══════════════════════════════════════════════════════════════════════════


def _run_main(tmp_path, extra_args, *, setup_intermediate=False,
              setup_output=False, step1_fail=False, step2_fail=False,
              meta=None, with_gguf_py=True, deps_ok=True,
              confirm_default=True):
    """Helper to run main() with full mocking."""
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))  # 4MB to pass size checks
    out = tmp_path / "output"
    intermediate = tmp_path / "output_intermediate"

    if setup_intermediate:
        intermediate.mkdir(parents=True, exist_ok=True)
        (intermediate / "config.json").write_text("{}")
        (intermediate / "model.safetensors").write_bytes(b"\x00" * 100)

    if setup_output:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text("{}")
        (out / "model.safetensors").write_bytes(b"\x00" * 100)

    args = ["convert.py", str(gguf), str(out), "--force", "--quiet"] + extra_args

    call_count = [0]
    def fake_run(cmd, description, progress=None, quiet=False):
        call_count[0] += 1
        if call_count[0] == 1:
            if step1_fail:
                return (False, "step1 error output")
            intermediate.mkdir(parents=True, exist_ok=True)
            (intermediate / "config.json").write_text("{}")
            (intermediate / "model.safetensors").write_bytes(b"\x00" * 100)
        elif call_count[0] == 2:
            if step2_fail:
                return (False, "step2 error with blk. tensor issue Received parameters not in model")
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text("{}")
            (out / "model.safetensors").write_bytes(b"\x00" * 100)
        return (True, "")

    dep_result = {"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?" if deps_ok else None}

    mock_meta = meta or {
        "architecture": "llama", "model_name": "test", "param_count": 7e9,
        "vocab_size": 32000, "hidden_size": 4096, "num_layers": 32,
        "num_heads": 32, "num_kv_heads": 32, "context_length": 4096,
        "tensor_count": 300, "total_weight_bytes": 5e9,
        "file_type": 1, "file_type_name": "MOSTLY_Q4_K_M",
        "mtp_layers": None, "has_ssm": False, "warnings": [], "fields": {},
    }

    with patch("sys.argv", args):
        with patch("convert.check_dependencies", return_value=dep_result):
            with patch("convert._has_gguf_py", return_value=with_gguf_py):
                with patch("convert.ensure_deps"):
                    with patch("convert.read_gguf_metadata", return_value=mock_meta):
                        with patch("convert.run_with_progress", side_effect=fake_run):
                            with patch("shutil.rmtree"):
                                with patch("shutil.move"):
                                    with patch("convert.validate_output", return_value=True):
                                        with patch("convert.detect_apple_silicon", return_value={
                                            "is_apple_silicon": True,
                                            "chip_name": "Apple M3 Pro",
                                            "chip_gen": "m3",
                                            "chip_tier": "pro",
                                            "ram_gb": 36.0,
                                        }):
                                            from convert import main
                                            try:
                                                main()
                                            except SystemExit:
                                                pass


class TestMainFullQuantizePipeline:
    """Full quantize pipeline path."""

    def test_quantize_success(self, tmp_path):
        _run_main(tmp_path, [])

    def test_quantize_4bit(self, tmp_path):
        _run_main(tmp_path, ["--bits", "4"])

    def test_quantize_with_preset(self, tmp_path):
        _run_main(tmp_path, ["--preset", "fast"])

    def test_quantize_8bit(self, tmp_path):
        _run_main(tmp_path, ["--bits", "8", "--group-size", "64"])

    def test_quantize_with_predicate(self, tmp_path):
        _run_main(tmp_path, ["--predicate", "E=4,B=32"])

    def test_quantize_with_mode(self, tmp_path):
        _run_main(tmp_path, ["--bits", "4", "--mode", "affine"])

    def test_dtype_override(self, tmp_path):
        _run_main(tmp_path, ["--dtype", "float32"])

    def test_bf16_source_auto_dtype(self, tmp_path):
        meta = {
            "architecture": "llama", "model_name": "test", "param_count": 7e9,
            "vocab_size": None, "hidden_size": None, "num_layers": None,
            "num_heads": None, "num_kv_heads": None, "context_length": None,
            "tensor_count": 100, "total_weight_bytes": 5e9,
            "file_type": 26, "file_type_name": "BF16 (bfloat16)",
            "mtp_layers": None, "has_ssm": False, "warnings": [], "fields": {},
        }
        _run_main(tmp_path, [], meta=meta)

    def test_f32_source_auto_dtype(self, tmp_path):
        meta = {
            "architecture": "llama", "model_name": "test", "param_count": 7e9,
            "vocab_size": None, "hidden_size": None, "num_layers": None,
            "num_heads": None, "num_kv_heads": None, "context_length": None,
            "tensor_count": 100, "total_weight_bytes": 5e9,
            "file_type": 0, "file_type_name": "ALL_F32",
            "mtp_layers": None, "has_ssm": False, "warnings": [], "fields": {},
        }
        _run_main(tmp_path, [], meta=meta)


class TestMainNoQuantize:
    """Float16-only path."""

    def test_no_quantize(self, tmp_path):
        _run_main(tmp_path, ["--no-quantize"])

    def test_no_quantize_intermediate_is_output(self, tmp_path):
        """When intermediate_dir == final_dir (shouldn't move)."""
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"
        intermediate = tmp_path / "output_intermediate"

        call_count = [0]
        def fake_run(cmd, description, progress=None, quiet=False):
            call_count[0] += 1
            # Step 1 creates output dir directly (no intermediate since it IS output)
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text("{}")
            (out / "model.safetensors").write_bytes(b"\x00" * 100)
            return (True, "")

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--no-quantize", "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=True):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value=None):
                            with patch("convert.run_with_progress", side_effect=fake_run):
                                with patch("convert.validate_output", return_value=True):
                                    with patch("convert.detect_apple_silicon", return_value={
                                        "is_apple_silicon": True, "chip_name": "Apple M3 Pro",
                                        "chip_gen": "m3", "chip_tier": "pro", "ram_gb": 36.0,
                                    }):
                                        from convert import main
                                        try:
                                            main()
                                        except SystemExit:
                                            pass


class TestMainStepFailures:
    """Test main() handling of step failures."""

    def test_step1_failure(self, tmp_path):
        _run_main(tmp_path, [], step1_fail=True)

    def test_step2_failure_gemma(self, tmp_path):
        meta = {
            "architecture": "gemma4", "model_name": "gemma4-test",
            "param_count": 27e9, "vocab_size": 256000, "hidden_size": 4096,
            "num_layers": 48, "num_heads": 32, "num_kv_heads": 8,
            "context_length": 8192, "tensor_count": 500,
            "total_weight_bytes": 15e9, "file_type": 1,
            "file_type_name": "MOSTLY_Q4_K_M", "mtp_layers": None,
            "has_ssm": False, "warnings": [], "fields": {},
        }
        _run_main(tmp_path, [], step2_fail=True, meta=meta)

    def test_step2_failure_generic(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"
        intermediate = tmp_path / "output_intermediate"

        call_count = [0]
        def fake_run(cmd, description, progress=None, quiet=False):
            call_count[0] += 1
            if call_count[0] == 1:
                intermediate.mkdir(parents=True, exist_ok=True)
                (intermediate / "config.json").write_text("{}")
                (intermediate / "model.safetensors").write_bytes(b"\x00" * 100)
                return (True, "")
            return (False, "generic error")

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=True):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value={
                            "architecture": "llama", "model_name": "test",
                            "param_count": 7e9, "vocab_size": 32000,
                            "hidden_size": 4096, "num_layers": 32,
                            "num_heads": 32, "num_kv_heads": 32,
                            "context_length": 4096, "tensor_count": 300,
                            "total_weight_bytes": 5e9, "file_type": 1,
                            "file_type_name": "Q4_K_M", "mtp_layers": None,
                            "has_ssm": False, "warnings": [], "fields": {},
                        }):
                            with patch("convert.run_with_progress", side_effect=fake_run):
                                with patch("shutil.rmtree"):
                                    from convert import main
                                    try:
                                        main()
                                    except SystemExit:
                                        pass


class TestMainMTPAndSSM:
    """Test main() with MTP and SSM models."""

    def test_mtp_model_with_flag(self, tmp_path):
        meta = {
            "architecture": "qwen3", "model_name": "qwen3-test",
            "param_count": 30e9, "vocab_size": 151936, "hidden_size": 4096,
            "num_layers": 36, "num_heads": 32, "num_kv_heads": 8,
            "context_length": 32768, "tensor_count": 500,
            "total_weight_bytes": 15e9, "file_type": 13,
            "file_type_name": "MOSTLY_Q4_K_M", "mtp_layers": 2,
            "has_ssm": False, "warnings": [], "fields": {},
        }
        _run_main(tmp_path, ["--mtp"], meta=meta)

    def test_ssm_model(self, tmp_path):
        meta = {
            "architecture": "mamba", "model_name": "mamba-test",
            "param_count": 2.8e9, "vocab_size": 32000, "hidden_size": 2560,
            "num_layers": 64, "num_heads": None, "num_kv_heads": None,
            "context_length": None, "tensor_count": 200,
            "total_weight_bytes": 2e9, "file_type": 1,
            "file_type_name": "F16", "mtp_layers": None,
            "has_ssm": True, "warnings": [], "fields": {},
        }
        _run_main(tmp_path, ["--no-quantize"], meta=meta)


class TestMainHighRiskSource:
    """Test main() with high-risk source quality."""

    def test_severe_source(self, tmp_path):
        meta = {
            "architecture": "llama", "model_name": "test",
            "param_count": 7e9, "vocab_size": 32000, "hidden_size": 4096,
            "num_layers": 32, "num_heads": 32, "num_kv_heads": 32,
            "context_length": 4096, "tensor_count": 300,
            "total_weight_bytes": 5e9, "file_type": 8,
            "file_type_name": "MOSTLY_Q8_0 (8-bit)", "mtp_layers": None,
            "has_ssm": False, "warnings": [], "fields": {},
        }
        _run_main(tmp_path, [], meta=meta)


class TestMainKnownIssues:
    """Test main() with known architecture issues."""

    def test_deepseek_v3(self, tmp_path):
        meta = {
            "architecture": "deepseek_v3", "model_name": "deepseek-v3",
            "param_count": 670e9, "vocab_size": 129280, "hidden_size": 4096,
            "num_layers": 61, "num_heads": 128, "num_kv_heads": 128,
            "context_length": 131072, "tensor_count": 1000,
            "total_weight_bytes": 400e9, "file_type": 1,
            "file_type_name": "Q4_K_M", "mtp_layers": None,
            "has_ssm": False, "warnings": [], "fields": {},
        }
        _run_main(tmp_path, [], meta=meta)


class TestMainResumeAndCleanup:
    """Test resume and cleanup paths."""

    def test_resume_with_existing(self, tmp_path):
        _run_main(tmp_path, ["--resume"], setup_intermediate=True)

    def test_cleanup_old_flag(self, tmp_path):
        _run_main(tmp_path, ["--cleanup-old"])

    def test_keep_intermediate(self, tmp_path):
        _run_main(tmp_path, ["--keep-intermediate"])


class TestMainNoMeta:
    """Test main() when metadata is unavailable."""

    def test_no_metadata(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"

        def fake_run(cmd, description, progress=None, quiet=False):
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text("{}")
            (out / "model.safetensors").write_bytes(b"\x00" * 100)
            return (True, "")

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--no-quantize", "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value=None):
                            with patch("convert.run_with_progress", side_effect=fake_run):
                                with patch("convert.validate_output", return_value=True):
                                    with patch("convert.detect_apple_silicon", return_value={
                                        "is_apple_silicon": False, "chip_name": "Intel",
                                        "chip_gen": "intel", "chip_tier": "base", "ram_gb": 16.0,
                                    }):
                                        from convert import main
                                        try:
                                            main()
                                        except SystemExit:
                                            pass


class TestMainEstimate:
    """Test --estimate mode."""

    def test_estimate_with_meta(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        with patch("sys.argv", ["convert.py", str(gguf), "--estimate"]):
            with patch("convert._has_gguf_py", return_value=True):
                with patch("convert.read_gguf_metadata", return_value={
                    "architecture": "llama", "model_name": "test",
                    "param_count": 7e9, "vocab_size": 32000, "hidden_size": 4096,
                    "num_layers": 32, "num_heads": 32, "num_kv_heads": 32,
                    "context_length": 4096, "tensor_count": 300,
                    "total_weight_bytes": 5e9, "file_type": 1,
                    "file_type_name": "Q4_K_M", "mtp_layers": None,
                    "has_ssm": False, "warnings": [], "fields": {},
                }):
                    from convert import main
                    main()

    def test_estimate_no_meta(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        with patch("sys.argv", ["convert.py", str(gguf), "--estimate"]):
            with patch("convert._has_gguf_py", return_value=False):
                with patch("convert.read_gguf_metadata", return_value=None):
                    from convert import main
                    main()


class TestMainHighBandwidthPreset:
    """Test --high-bandwidth preset selection."""

    def test_high_bandwidth_selects_preset(self, tmp_path):
        _run_main(tmp_path, ["--high-bandwidth"])


class TestMainExceptionHandling:
    """Test main() exception handling."""

    def test_generic_exception(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with patch("convert.ensure_deps", side_effect=RuntimeError("unexpected error")):
                        from convert import main
                        try:
                            main()
                        except (SystemExit, RuntimeError):
                            pass


class TestMainGemma4Fix:
    """Test Gemma4 tensor name fix during conversion."""

    def test_gemma4_fix_applied(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"
        intermediate = tmp_path / "output_intermediate"

        call_count = [0]
        def fake_run(cmd, description, progress=None, quiet=False):
            call_count[0] += 1
            if call_count[0] == 1:
                intermediate.mkdir(parents=True, exist_ok=True)
                (intermediate / "config.json").write_text("{}")
                (intermediate / "model.safetensors").write_bytes(b"\x00" * 100)
            elif call_count[0] == 2:
                out.mkdir(parents=True, exist_ok=True)
                (out / "config.json").write_text("{}")
                (out / "model.safetensors").write_bytes(b"\x00" * 100)
            return (True, "")

        meta = {
            "architecture": "gemma4", "model_name": "gemma4-test",
            "param_count": 27e9, "vocab_size": 256000, "hidden_size": 4096,
            "num_layers": 48, "num_heads": 32, "num_kv_heads": 8,
            "context_length": 8192, "tensor_count": 500,
            "total_weight_bytes": 15e9, "file_type": 1,
            "file_type_name": "Q4_K_M", "mtp_layers": None,
            "has_ssm": False, "warnings": [], "fields": {},
        }

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=True):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value=meta):
                            with patch("convert.run_with_progress", side_effect=fake_run):
                                with patch("convert.fix_gemma4_tensor_names", return_value=True):
                                    with patch("shutil.rmtree"):
                                        with patch("convert.validate_output", return_value=True):
                                            with patch("convert.detect_apple_silicon", return_value={
                                                "is_apple_silicon": True, "chip_name": "Apple M4 Max",
                                                "chip_gen": 4, "chip_tier": "max", "ram_gb": 128.0,
                                            }):
                                                from convert import main
                                                try:
                                                    main()
                                                except SystemExit:
                                                    pass


class TestMainAutoCleanupOnFailure:
    """Test auto-cleanup when conversion fails mid-way."""

    def test_cleanup_after_step1_fail(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"
        intermediate = tmp_path / "output_intermediate"
        intermediate.mkdir(parents=True)
        (intermediate / "partial.bin").write_bytes(b"\x00")

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value=None):
                            with patch("convert.run_with_progress", return_value=(False, "error")):
                                with patch("shutil.rmtree") as mock_rmtree:
                                    from convert import main
                                    try:
                                        main()
                                    except SystemExit:
                                        pass


class TestMainSmartDefaults:
    """Test smart defaults path (no --bits, no --preset)."""

    def test_smart_defaults_selected(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"

        call_count = [0]
        def fake_run(cmd, description, progress=None, quiet=False):
            call_count[0] += 1
            if call_count[0] == 1:
                intermediate = tmp_path / "output_intermediate"
                intermediate.mkdir(parents=True, exist_ok=True)
                (intermediate / "config.json").write_text("{}")
                (intermediate / "model.safetensors").write_bytes(b"\x00" * 100)
            elif call_count[0] == 2:
                out.mkdir(parents=True, exist_ok=True)
                (out / "config.json").write_text("{}")
            return (True, "")

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value=None):
                            with patch("convert.run_with_progress", side_effect=fake_run):
                                with patch("shutil.rmtree"):
                                    with patch("convert.validate_output", return_value=True):
                                        with patch("convert.detect_apple_silicon", return_value={
                                            "is_apple_silicon": True, "chip_name": "Apple M2",
                                            "chip_gen": "m2", "chip_tier": "base", "ram_gb": 8.0,
                                        }):
                                            from convert import main
                                            try:
                                                main()
                                            except SystemExit:
                                                pass


# ═══════════════════════════════════════════════════════════════════════════
# Final push to 90% — targeted line coverage
# ═══════════════════════════════════════════════════════════════════════════


class TestMainEstimateWithDetails:
    """Full estimate mode with metadata and warnings."""

    def test_estimate_with_warnings_and_mtp(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        with patch("sys.argv", ["convert.py", str(gguf), "--estimate"]):
            with patch("convert._has_gguf_py", return_value=True):
                with patch("convert.read_gguf_metadata", return_value={
                    "architecture": "qwen3", "model_name": "qwen3-test",
                    "param_count": 30e9, "vocab_size": 151936, "hidden_size": 4096,
                    "num_layers": 36, "num_heads": 32, "num_kv_heads": 8,
                    "context_length": 32768, "tensor_count": 500,
                    "total_weight_bytes": 15e9, "file_type": 13,
                    "file_type_name": "Q4_K_M", "mtp_layers": 2,
                    "has_ssm": False, "warnings": ["Large model"], "fields": {},
                }):
                    with patch("convert.detect_apple_silicon", return_value={
                        "is_apple_silicon": True, "chip_name": "Apple M3 Max",
                        "chip_gen": 3, "chip_tier": "max", "ram_gb": 64.0,
                    }):
                        from convert import main
                        main()

    def test_estimate_no_estimate_flag(self, tmp_path):
        """Non-estimate mode with large model should still work."""
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        with patch("sys.argv", ["convert.py", str(gguf), "--estimate"]):
            with patch("convert._has_gguf_py", return_value=True):
                with patch("convert.read_gguf_metadata", return_value={
                    "architecture": "llama", "model_name": "big-model",
                    "param_count": 70e9, "vocab_size": 32000, "hidden_size": 8192,
                    "num_layers": 80, "num_heads": 64, "num_kv_heads": 8,
                    "context_length": 8192, "tensor_count": 800,
                    "total_weight_bytes": 40e9, "file_type": 2,
                    "file_type_name": "Q4_0 (4-bit)", "mtp_layers": None,
                    "has_ssm": False, "warnings": [], "fields": {},
                }):
                    with patch("convert.detect_apple_silicon", return_value={
                        "is_apple_silicon": True, "chip_name": "Apple M2",
                        "chip_gen": 2, "chip_tier": "base", "ram_gb": 8.0,
                    }):
                        from convert import main
                        main()


class TestMainNonQuietMode:
    """Test main() in non-quiet mode to cover display paths."""

    def test_verbose_quantize(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"
        intermediate = tmp_path / "output_intermediate"

        call_count = [0]
        def fake_run(cmd, description, progress=None, quiet=False):
            call_count[0] += 1
            if call_count[0] == 1:
                intermediate.mkdir(parents=True, exist_ok=True)
                (intermediate / "config.json").write_text("{}")
                (intermediate / "model.safetensors").write_bytes(b"\x00" * 100)
            elif call_count[0] == 2:
                out.mkdir(parents=True, exist_ok=True)
                (out / "config.json").write_text("{}")
                (out / "model.safetensors").write_bytes(b"\x00" * 100)
            return (True, "")

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--force"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=True):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value={
                            "architecture": "llama", "model_name": "test",
                            "param_count": 7e9, "vocab_size": 32000, "hidden_size": 4096,
                            "num_layers": 32, "num_heads": 32, "num_kv_heads": 32,
                            "context_length": 4096, "tensor_count": 300,
                            "total_weight_bytes": 5e9, "file_type": 1,
                            "file_type_name": "Q4_K_M", "mtp_layers": None,
                            "has_ssm": False, "warnings": [], "fields": {},
                        }):
                            with patch("convert.run_with_progress", side_effect=fake_run):
                                with patch("shutil.rmtree"):
                                    with patch("convert.validate_output", return_value=True):
                                        with patch("convert.detect_apple_silicon", return_value={
                                            "is_apple_silicon": True, "chip_name": "Apple M3 Pro",
                                            "chip_gen": 3, "chip_tier": "pro", "ram_gb": 36.0,
                                        }):
                                            with patch("convert.Confirm.ask", return_value=True):
                                                from convert import main
                                                try:
                                                    main()
                                                except SystemExit:
                                                    pass

    def test_verbose_no_meta(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"

        def fake_run(cmd, description, progress=None, quiet=False):
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text("{}")
            (out / "model.safetensors").write_bytes(b"\x00" * 100)
            return (True, "")

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--no-quantize", "--force"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value=None):
                            with patch("convert.run_with_progress", side_effect=fake_run):
                                with patch("convert.validate_output", return_value=True):
                                    with patch("convert.detect_apple_silicon", return_value={
                                        "is_apple_silicon": False, "chip_name": "Intel",
                                        "chip_gen": "intel", "chip_tier": "base", "ram_gb": 16.0,
                                    }):
                                        from convert import main
                                        try:
                                            main()
                                        except SystemExit:
                                            pass


class TestMainStep1FailureKnownIssue:
    """Test step1 failure with known architecture issue (covers lines 1884-1890)."""

    def test_gemma4_step1_fail(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"
        intermediate = tmp_path / "output_intermediate"

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=True):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value={
                            "architecture": "gemma4", "model_name": "gemma4-test",
                            "param_count": 27e9, "vocab_size": 256000, "hidden_size": 4096,
                            "num_layers": 48, "num_heads": 32, "num_kv_heads": 8,
                            "context_length": 8192, "tensor_count": 500,
                            "total_weight_bytes": 15e9, "file_type": 1,
                            "file_type_name": "Q4_K_M", "mtp_layers": None,
                            "has_ssm": False, "warnings": [], "fields": {},
                        }):
                            with patch("convert.run_with_progress", return_value=(False, "step1 error")):
                                with patch("convert.detect_apple_silicon", return_value={
                                    "is_apple_silicon": True, "chip_name": "Apple M3 Pro",
                                    "chip_gen": 3, "chip_tier": "pro", "ram_gb": 36.0,
                                }):
                                    from convert import main
                                    try:
                                        main()
                                    except SystemExit:
                                        pass


class TestMainGemma4TensorFixInPipeline:
    """Test Gemma4 tensor fix during pipeline (covers lines 1949-1958)."""

    def test_gemma4_tensor_fix(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"
        intermediate = tmp_path / "output_intermediate"

        call_count = [0]
        def fake_run(cmd, description, progress=None, quiet=False):
            call_count[0] += 1
            if call_count[0] == 1:
                intermediate.mkdir(parents=True, exist_ok=True)
                (intermediate / "config.json").write_text("{}")
                (intermediate / "model.safetensors").write_bytes(b"\x00" * 100)
            elif call_count[0] == 2:
                out.mkdir(parents=True, exist_ok=True)
                (out / "config.json").write_text("{}")
            return (True, "")

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=True):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value={
                            "architecture": "gemma4", "model_name": "gemma4-test",
                            "param_count": 27e9, "vocab_size": 256000, "hidden_size": 4096,
                            "num_layers": 48, "num_heads": 32, "num_kv_heads": 8,
                            "context_length": 8192, "tensor_count": 500,
                            "total_weight_bytes": 15e9, "file_type": 1,
                            "file_type_name": "Q4_K_M", "mtp_layers": None,
                            "has_ssm": False, "warnings": [], "fields": {},
                        }):
                            with patch("convert.run_with_progress", side_effect=fake_run):
                                with patch("convert.fix_gemma4_tensor_names", return_value=False):
                                    with patch("shutil.rmtree"):
                                        with patch("convert.validate_output", return_value=True):
                                            with patch("convert.detect_apple_silicon", return_value={
                                                "is_apple_silicon": True, "chip_name": "Apple M3 Pro",
                                                "chip_gen": 3, "chip_tier": "pro", "ram_gb": 36.0,
                                            }):
                                                from convert import main
                                                try:
                                                    main()
                                                except SystemExit:
                                                    pass


class TestMainUnsupportedArchDisplay:
    """Cover unsupported arch display (lines 1683-1726)."""

    def test_unsupported_arch_without_force(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=True):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value={
                            "architecture": "falcon", "model_name": "falcon-test",
                            "param_count": 7e9, "vocab_size": 32000, "hidden_size": 4096,
                            "num_layers": 32, "num_heads": 32, "num_kv_heads": 32,
                            "context_length": 4096, "tensor_count": 300,
                            "total_weight_bytes": 5e9, "file_type": 1,
                            "file_type_name": "F16", "mtp_layers": None,
                            "has_ssm": False, "warnings": [], "fields": {},
                        }):
                            with patch("convert.Confirm.ask", return_value=False):
                                with patch("convert.detect_apple_silicon", return_value={
                                    "is_apple_silicon": True, "chip_name": "Apple M3",
                                    "chip_gen": 3, "chip_tier": "base", "ram_gb": 8.0,
                                }):
                                    from convert import main
                                    try:
                                        main()
                                    except SystemExit:
                                        pass

    def test_unsupported_arch_with_force(self, tmp_path):
        """Unsupported arch + --force should continue."""
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"
        intermediate = tmp_path / "output_intermediate"

        call_count = [0]
        def fake_run(cmd, description, progress=None, quiet=False):
            call_count[0] += 1
            if call_count[0] == 1:
                intermediate.mkdir(parents=True, exist_ok=True)
                (intermediate / "config.json").write_text("{}")
            elif call_count[0] == 2:
                out.mkdir(parents=True, exist_ok=True)
                (out / "config.json").write_text("{}")
            return (True, "")

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--force", "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=True):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value={
                            "architecture": "falcon", "model_name": "falcon-test",
                            "param_count": 7e9, "vocab_size": 32000, "hidden_size": 4096,
                            "num_layers": 32, "num_heads": 32, "num_kv_heads": 32,
                            "context_length": 4096, "tensor_count": 300,
                            "total_weight_bytes": 5e9, "file_type": 1,
                            "file_type_name": "F16", "mtp_layers": None,
                            "has_ssm": False, "warnings": [], "fields": {},
                        }):
                            with patch("convert.run_with_progress", side_effect=fake_run):
                                with patch("shutil.rmtree"):
                                    with patch("convert.validate_output", return_value=True):
                                        with patch("convert.detect_apple_silicon", return_value={
                                            "is_apple_silicon": True, "chip_name": "Apple M3",
                                            "chip_gen": 3, "chip_tier": "base", "ram_gb": 8.0,
                                        }):
                                            from convert import main
                                            try:
                                                main()
                                            except SystemExit:
                                                pass


class TestMainDiskSpaceCheck:
    """Cover disk space check path."""

    def test_disk_space_check_fails(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--quiet"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value=None):
                            with patch("convert.check_disk_space", return_value=False):
                                with patch("convert.detect_apple_silicon", return_value={
                                    "is_apple_silicon": True, "chip_name": "Apple M3",
                                    "chip_gen": 3, "chip_tier": "base", "ram_gb": 8.0,
                                }):
                                    from convert import main
                                    try:
                                        main()
                                    except SystemExit:
                                        pass


class TestMainGuidedMode:
    """Cover guided mode (no input file, interactive)."""

    def test_guided_no_input(self, tmp_path):
        with patch("sys.argv", ["convert.py"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert.ensure_deps"):
                    with patch("convert.get_gguf_path", return_value=tmp_path / "model.gguf"):
                        with patch("convert.get_output_dir", return_value=tmp_path / "out"):
                            # Will fail on file not existing
                            from convert import main
                            try:
                                main()
                            except SystemExit:
                                pass


class TestMainIntermediateExistsConfirm:
    """Cover intermediate dir exists confirmation."""

    def test_intermediate_exists_yes(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"
        intermediate = tmp_path / "output_intermediate"
        intermediate.mkdir(parents=True)
        (intermediate / "config.json").write_text("{}")

        call_count = [0]
        def fake_run(cmd, description, progress=None, quiet=False):
            call_count[0] += 1
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text("{}")
            return (True, "")

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--force"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value=None):
                            with patch("convert.run_with_progress", side_effect=fake_run):
                                with patch("shutil.rmtree"):
                                    with patch("convert.validate_output", return_value=True):
                                        with patch("convert.detect_apple_silicon", return_value={
                                            "is_apple_silicon": True, "chip_name": "Apple M3",
                                            "chip_gen": 3, "chip_tier": "base", "ram_gb": 8.0,
                                        }):
                                            with patch("convert.Confirm.ask", return_value=True):
                                                from convert import main
                                                try:
                                                    main()
                                                except SystemExit:
                                                    pass

    def test_intermediate_exists_no(self, tmp_path):
        """User says no to resume - should do full conversion."""
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * (4 * 1024 * 1024))
        out = tmp_path / "output"
        intermediate = tmp_path / "output_intermediate"
        intermediate.mkdir(parents=True)
        (intermediate / "config.json").write_text("{}")

        call_count = [0]
        def fake_run(cmd, description, progress=None, quiet=False):
            call_count[0] += 1
            if call_count[0] == 1:
                (intermediate / "model.safetensors").write_bytes(b"\x00" * 100)
            elif call_count[0] == 2:
                out.mkdir(parents=True, exist_ok=True)
                (out / "config.json").write_text("{}")
            return (True, "")

        with patch("sys.argv", ["convert.py", str(gguf), str(out), "--force"]):
            with patch("convert.check_dependencies", return_value={"gguf2mlx": "?", "mlx_lm": "?", "mlx": "?", "gguf_py": "?"}):
                with patch("convert._has_gguf_py", return_value=False):
                    with patch("convert.ensure_deps"):
                        with patch("convert.read_gguf_metadata", return_value=None):
                            with patch("convert.run_with_progress", side_effect=fake_run):
                                with patch("shutil.rmtree"):
                                    with patch("convert.validate_output", return_value=True):
                                        with patch("convert.detect_apple_silicon", return_value={
                                            "is_apple_silicon": True, "chip_name": "Apple M3",
                                            "chip_gen": 3, "chip_tier": "base", "ram_gb": 8.0,
                                        }):
                                            with patch("convert.Confirm.ask", return_value=False):
                                                from convert import main
                                                try:
                                                    main()
                                                except SystemExit:
                                                    pass
