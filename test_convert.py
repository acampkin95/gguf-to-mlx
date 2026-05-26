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