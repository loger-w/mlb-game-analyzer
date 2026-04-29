"""Tests for phase3_skeleton_renderer."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _minimal_bundle():
    return {
        "game_data": {"date": "2026-04-28", "home": {"team": "CLE"}, "away": {"team": "TB"}},
        "home_pitcher": {"name": "Bibee", "tier_emoji": "🟢", "info": {"pitch_hand": "R", "age": 27}},
        "away_pitcher": {"name": "Martínez", "tier_emoji": "🟠", "info": {"pitch_hand": "R", "age": 35}},
        "home_lineup": {"tier_emoji": "🟡", "heat_emoji": "⚖️"},
        "away_lineup": {"tier_emoji": "🟢", "heat_emoji": "⚖️"},
        "merged": {"home_bullpen_era": 4.57, "away_bullpen_era": 5.18,
                   "home_bullpen_il_count": 2, "away_bullpen_il_count": 8,
                   "park_factor": 101},
    }


def _minimal_formula_pred():
    return {"home_expected_runs": 4.5, "away_expected_runs": 4.2}


def test_skeleton_contains_7_required_h2():
    """spec §5.3：7 個 H2 永遠存在"""
    from phase3_skeleton_renderer import render_skeleton
    output = render_skeleton(_minimal_bundle(), _minimal_formula_pred())
    for h2 in ["## 投手對決", "## 打線評級", "## 牛棚", "## 風險提示",
               "## 條件修正", "## 修正後預期得分", "## 整體判斷"]:
        assert h2 in output, f"缺 {h2}"


def test_skeleton_no_yoy_or_babip_h2():
    """spec §5.3：刪除 ## YoY 對比結論 / ## BABIP 回歸判定"""
    from phase3_skeleton_renderer import render_skeleton
    output = render_skeleton(_minimal_bundle(), _minimal_formula_pred())
    assert "## YoY 對比結論" not in output
    assert "## BABIP 回歸判定" not in output


def test_skeleton_tier_override_slot_present():
    """spec §5.2：Tier 覆寫 slot 在投手 + 打線段都要有"""
    from phase3_skeleton_renderer import render_skeleton
    output = render_skeleton(_minimal_bundle(), _minimal_formula_pred())
    # 投手對決有 2 個（home + away），打線有 2 個 → 至少 4 處
    assert output.count("**Tier 覆寫**") >= 4


def test_skeleton_risk_section_lists_triggers_when_present():
    """Flag 13 / Flag 3 觸發 → 預填條目至 ## 風險提示"""
    from phase3_skeleton_renderer import render_skeleton
    bundle = _minimal_bundle()
    bundle["away_pitcher"]["season"] = {"era": 2.10, "ip": 31.3}
    bundle["away_pitcher"]["expected"] = {"xera": 4.64}  # gap = 2.54 → Flag 13
    bundle["away_lineup"]["last7_babip"] = 0.241  # Flag 3
    output = render_skeleton(bundle, _minimal_formula_pred())
    assert "Flag 13" in output
    assert "Flag 3" in output
    assert "era_xera_delta" in output or "ERA-xERA" in output


def test_skeleton_risk_section_says_no_flag_when_clean():
    """無 Flag 觸發 → ## 風險提示 內文「無風險提示」"""
    from phase3_skeleton_renderer import render_skeleton
    output = render_skeleton(_minimal_bundle(), _minimal_formula_pred())
    after_risk = output.split("## 風險提示", 1)[1].split("##", 1)[0]
    assert "無風險提示" in after_risk


def test_skeleton_park_factor_correction_prefilled():
    """## 條件修正 段預填 Park Factor 修正值"""
    from phase3_skeleton_renderer import render_skeleton
    bundle = _minimal_bundle()
    bundle["merged"]["park_factor"] = 110
    output = render_skeleton(bundle, _minimal_formula_pred())
    # PF 110 → +0.5 run 修正
    assert "Park Factor: 110" in output or "PF=110" in output
    assert "+0.50" in output or "+0.5" in output


def test_skeleton_expected_runs_table_uses_formula_pred():
    """## 修正後預期得分 base 列從 formula_pred 取得"""
    from phase3_skeleton_renderer import render_skeleton
    output = render_skeleton(_minimal_bundle(), _minimal_formula_pred())
    assert "4.5" in output  # home_expected_runs
    assert "4.2" in output  # away_expected_runs


def test_skeleton_line_count_within_50():
    """spec §9 驗收：phase3_skeleton.md ≤ 50 行（無 Flag 觸發時）"""
    from phase3_skeleton_renderer import render_skeleton
    output = render_skeleton(_minimal_bundle(), _minimal_formula_pred())
    assert len(output.split("\n")) <= 50
