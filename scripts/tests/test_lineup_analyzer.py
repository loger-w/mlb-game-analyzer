"""Tests for lineup_analyzer helpers (Plan B 2026-04-22 §4.6 extension)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_last7_babip_averaged_from_core_lineup():
    """last7_babip = 核心打線近 7 天 BABIP 平均（忽略 None / 非數值）。"""
    from lineup_analyzer import compute_last7_babip

    core_lineup = [
        {"last_7": {"babip": "0.320"}},
        {"last_7": {"babip": "0.280"}},
        {"last_7": {"babip": "0.340"}},
        {"last_7": {"babip": None}},
        {"last_7": {}},
    ]
    result = compute_last7_babip(core_lineup)
    assert result is not None
    # (0.320 + 0.280 + 0.340) / 3 ≈ 0.313
    assert abs(result - 0.313) < 0.002


def test_last7_babip_empty_lineup_returns_none():
    from lineup_analyzer import compute_last7_babip
    assert compute_last7_babip([]) is None


def test_last7_babip_all_missing_returns_none():
    from lineup_analyzer import compute_last7_babip
    assert compute_last7_babip([{"last_7": {}}, {}, {"last_7": {"babip": None}}]) is None


def test_last7_babip_numeric_values_accepted():
    """float 值（非 string）也能處理。"""
    from lineup_analyzer import compute_last7_babip
    core_lineup = [
        {"last_7": {"babip": 0.300}},
        {"last_7": {"babip": 0.320}},
    ]
    result = compute_last7_babip(core_lineup)
    assert result is not None
    assert abs(result - 0.310) < 0.001


def test_last7_babip_invalid_value_ignored():
    """非數值 babip（如 "N/A"）應被忽略。"""
    from lineup_analyzer import compute_last7_babip
    core_lineup = [
        {"last_7": {"babip": "0.280"}},
        {"last_7": {"babip": "N/A"}},
        {"last_7": {"babip": "0.320"}},
    ]
    result = compute_last7_babip(core_lineup)
    assert result is not None
    assert abs(result - 0.300) < 0.001
