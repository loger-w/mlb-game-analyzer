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


from odds_analyzer import analyze_over_under


def test_analyze_ou_kelly_both_sides():
    """line=8.5, predicted=10.0 → Over 有 edge；Under 無 edge。"""
    result = analyze_over_under(
        line=8.5, predicted_total=10.0,
        over_odds_ml=-110, under_odds_ml=-110,
    )
    assert result["direction"] == "OVER"
    assert "kelly_fractional" in result
    kf = result["kelly_fractional"]
    assert "over" in kf and "under" in kf
    # Over 應該有正 Kelly
    assert kf["over"]["raw_kelly_pct"] > 0
    # Under 應該 0
    assert kf["under"]["raw_kelly_pct"] == 0


def test_analyze_ou_no_odds_kelly_null():
    """未傳 odds → kelly_fractional 為 null。"""
    result = analyze_over_under(line=8.5, predicted_total=10.0)
    assert result["kelly_fractional"] is None


def test_analyze_ou_partial_odds():
    """只有 Over odds → Under 側 null，Over 側有值。"""
    result = analyze_over_under(
        line=8.5, predicted_total=10.0, over_odds_ml=-110,
    )
    kf = result["kelly_fractional"]
    assert kf is not None
    assert kf["over"] is not None
    assert kf["under"] is None
    assert kf["over"]["raw_kelly_pct"] > 0


from odds_analyzer import analyze_run_line


def test_analyze_rl_kelly_favorite_cover():
    """margin=+2.5, model_home_win=0.65, home ML -150 熱門（market 與 model 同向）。"""
    result = analyze_run_line(
        predicted_margin=2.5,
        model_home_win_pct=0.65,
        home_ml=-150,
        away_ml=+140,
        home_rl_odds_ml=-110,  # home -1.5 at -110
        away_rl_odds_ml=-110,  # away +1.5 at -110
        home_point=-1.5,       # Pinnacle: home 熱門 → home point = -1.5
    )
    assert "kelly_fractional" in result
    kf = result["kelly_fractional"]
    assert "favorite_cover" in kf
    assert "underdog_cover" in kf
    # Market favorite = home (ml=-150 更負), fav_ml=-150 → bucket 0.615
    # P(cover_fav) = 0.65 × 0.615 ≈ 0.3998；implied at -110 ≈ 0.5238 → Kelly ~0
    # P(cover_dog) ≈ 0.6002, implied 0.5238 → edge ~7.6%
    assert kf["underdog_cover"]["raw_kelly_pct"] > 0
    assert kf["favorite_cover"]["raw_kelly_pct"] >= 0  # 可能 0
    assert kf["favorite_cover"]["side"] == "HOME_-1.5"


def test_analyze_rl_market_favorite_when_model_disagrees():
    """C2 bug 測試：model 認為 home 贏 +0.5 分，但 market 熱門是 away。
    查 bucket 必須用 away_ml（市場熱門），不是 home（model 預測贏）。
    """
    # market: away 熱門 (ml=-150)，home 冷門 (ml=+140)
    # model: predicted_margin=+0.5（home 小勝，但 model P(home)=0.55 也不強）
    result = analyze_run_line(
        predicted_margin=0.5,
        model_home_win_pct=0.55,
        home_ml=+140,           # home 冷門
        away_ml=-150,           # away 熱門（market favorite）
        home_rl_odds_ml=+200,   # home +1.5 at +200（dog RL odds）
        away_rl_odds_ml=-260,   # away -1.5 at -260（fav RL odds）
        home_point=+1.5,        # Pinnacle: home 拿 +1.5 → home 是 dog
    )
    kf = result["kelly_fractional"]
    # fav_is_home 必須是 False（market favorite = away，不是 home）
    # fav_ml = away_ml = -150 → bucket = 0.615
    # p_cover_fav = (1 - 0.55) × 0.615 = 0.45 × 0.615 ≈ 0.2768（away win × margin 條件）
    # p_cover_dog = 1 - 0.2768 ≈ 0.7232（home 拿 +1.5 cover 機率）
    # Side 標註來自 home_point=+1.5 → fav_side = "AWAY_-1.5"
    assert kf["favorite_cover"]["side"] == "AWAY_-1.5"
    # favorite (away -1.5 @ -260) implied ≈ 72.2%, model p_cover ≈ 27.7% → raw Kelly = 0（負 edge）
    assert kf["favorite_cover"]["raw_kelly_pct"] == 0
    # underdog (home +1.5 @ +200) implied ≈ 33.3%, model p_cover ≈ 72.3% → 強正 edge
    assert kf["underdog_cover"]["raw_kelly_pct"] > 0
    # favorite Kelly 用的 odds 應該是 away_rl_odds_ml (-260)，不是 home_rl_odds_ml (+200)
    # 若 C2/C3 bug 未修，code 會誤用 home_rl_odds_ml=+200 當 favorite odds，
    # 導致 favorite_cover.raw_kelly_pct 反而顯示大正值 —— 此斷言會 fail
    assert kf["favorite_cover"]["decimal_odds"] == pytest.approx(1.385, abs=0.01)  # -260 → 1.385


def test_analyze_rl_side_label_falls_back_when_home_point_missing():
    """home_point 未傳 → 用 market ML 推 side（fallback path）。"""
    result = analyze_run_line(
        predicted_margin=2.5,
        model_home_win_pct=0.65,
        home_ml=-150, away_ml=+140,
        home_rl_odds_ml=-110, away_rl_odds_ml=-110,
        # home_point 省略
    )
    kf = result["kelly_fractional"]
    assert kf["favorite_cover"]["side"] == "HOME_-1.5"  # fav_is_home=True → home 是 -1.5


def test_analyze_rl_no_odds_kelly_null():
    """沒傳 RL odds → kelly_fractional 為 null。"""
    result = analyze_run_line(predicted_margin=2.5, model_home_win_pct=0.65)
    assert result["kelly_fractional"] is None


def test_analyze_rl_missing_ml_kelly_null():
    """有 RL odds 但沒 ML（無法判 market favorite / 查 bucket）→ null。"""
    result = analyze_run_line(
        predicted_margin=2.5, model_home_win_pct=0.65,
        home_rl_odds_ml=-110, away_rl_odds_ml=-110,
        # home_ml / away_ml 未傳
    )
    assert result["kelly_fractional"] is None
