"""Tests for analyze_moneyline/over_under/run_line Kelly extensions."""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from odds_analyzer import analyze_moneyline


def test_analyze_ml_has_kelly_fractional():
    """ML 分析回傳含 kelly_fractional 區塊，對方向與金額有正確數值。"""
    # Home at -150 (implied 60%), model says 65% → edge on home
    result = analyze_moneyline(home_ml=-150, away_ml=+140, model_win_pct=0.65)
    assert "kelly_fractional" in result
    kf = result["kelly_fractional"]
    assert kf["direction"] == "HOME"  # 與 result["direction"] 同
    # raw Kelly at -150 with p=0.65:
    # b = 100/150 ≈ 0.6667; raw = (0.65*1.6667 - 1)/0.6667 = 0.0833/0.6667 ≈ 12.5%
    assert kf["raw_kelly_pct"] > 10
    assert kf["raw_kelly_pct"] < 15
    assert kf["fractional_pct"] == round(kf["raw_kelly_pct"] / 4, 4)
    assert kf["capped_pct"] <= 3.0
    assert kf["units"] >= 0


def test_analyze_ml_no_edge_zero_kelly():
    """若 model 跟 implied 一致 → Kelly 0。"""
    # Home -110 implied ~52.4%; model says exactly 52.4% → zero edge
    result = analyze_moneyline(home_ml=-110, away_ml=-110, model_win_pct=0.524)
    kf = result["kelly_fractional"]
    # direction 由 EV 比較決定，但 Kelly 應接近 0
    assert kf["raw_kelly_pct"] <= 0.1
    assert kf["units"] == 0.0


def test_analyze_ml_custom_kelly_params():
    """kelly_params override 預設 divisor/cap。"""
    result = analyze_moneyline(
        home_ml=-150, away_ml=+140, model_win_pct=0.65,
        kelly_params={"divisor": 2, "cap_pct": 5.0, "unit_size_pct": 1.0},
    )
    kf = result["kelly_fractional"]
    # half-Kelly: fractional = raw / 2
    assert kf["fractional_pct"] == round(kf["raw_kelly_pct"] / 2, 4)
