"""Tests for summary_renderer.

Schema reality check：fixture 必須鏡射 pitcher_stats / lineup_analyzer / roster_checker
真實輸出 schema，不再用虛構 mock（避免 schema drift 讓測試綠但產出壞）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _minimal_bundle():
    """真實 schema fixture：pitcher_stats / lineup_analyzer / roster_checker top-level keys。"""
    return {
        "game_data": {"date": "2026-04-28", "home": {"team": "CLE"}, "away": {"team": "TB"}},
        # pitcher_stats schema：top-level pitch_hand / age / age_assessment / tier
        "home_pitcher": {
            "name": "Bibee", "tier": "🟢 Back-end",
            "pitch_hand": "R", "age": 27, "age_assessment": "⚡ 巔峰期",
        },
        "away_pitcher": {
            "name": "Martinez", "tier": "🟠 Strong Ace",
            "pitch_hand": "R", "age": 35, "age_assessment": "📉📉 顯著退化",
        },
        # lineup_analyzer schema：tier / recent_heat（已含 emoji）
        "home_lineup": {"tier": "🟡 Average", "recent_heat": "⚖️ Normal"},
        "away_lineup": {"tier": "🟢 Weak", "recent_heat": "⚖️ Normal"},
        # roster_checker schema：injured_list with position
        "home_roster": {"injured_list": [
            {"name": "X", "position": "Pitcher", "status": "Injured 15-Day"},
            {"name": "Y", "position": "Pitcher", "status": "Injured 60-Day"},
        ]},
        "away_roster": {"injured_list": [
            {"name": f"P{i}", "position": "Pitcher", "status": "Injured 60-Day"}
            for i in range(8)
        ]},
        "merged": {
            "home_bullpen_era": 4.57, "away_bullpen_era": 5.18,
            "park_factor": 101,
        },
    }


def _minimal_formula_pred():
    return {"home_score": 4.5, "away_score": 4.2}


def test_skeleton_contains_7_required_h2():
    """7 個 H2 永遠存在（即使 Flag 未觸發）。"""
    from summary_renderer import render_summary
    output = render_summary(_minimal_bundle(), _minimal_formula_pred())
    for h2 in ["## 投手對決", "## 打線評級", "## 牛棚", "## 風險提示",
               "## 條件修正", "## 修正後預期得分", "## 整體判斷"]:
        assert h2 in output, f"缺 {h2}"


def test_skeleton_no_yoy_or_babip_h2():
    """不應有 ## YoY 對比結論 / ## BABIP 回歸判定 H2。"""
    from summary_renderer import render_summary
    output = render_summary(_minimal_bundle(), _minimal_formula_pred())
    assert "## YoY 對比結論" not in output
    assert "## BABIP 回歸判定" not in output


def test_skeleton_tier_override_slot_present():
    """Tier 覆寫 slot 在投手 + 打線段都要有（至少 4 處）。"""
    from summary_renderer import render_summary
    output = render_summary(_minimal_bundle(), _minimal_formula_pred())
    assert output.count("**Tier 覆寫**") >= 4


def test_skeleton_risk_section_lists_triggers_when_present():
    """Flag 8 / Flag 3 觸發 → 預填條目至 ## 風險提示"""
    from summary_renderer import render_summary
    bundle = _minimal_bundle()
    bundle["away_pitcher"]["season"] = {"era": 2.10, "ip": 31.3}
    bundle["away_pitcher"]["expected"] = {"xera": 4.64}  # gap = -2.54 → Flag 8 cond 1
    bundle["away_lineup"]["last7_babip"] = 0.241  # Flag 3
    output = render_summary(bundle, _minimal_formula_pred())
    assert "Flag 8" in output
    assert "Flag 3" in output
    assert "era_xera_delta" in output or "ERA-xERA" in output


def test_skeleton_risk_section_says_no_flag_when_clean():
    """無 Flag 觸發 → ## 風險提示 內文「無風險提示」"""
    from summary_renderer import render_summary
    output = render_summary(_minimal_bundle(), _minimal_formula_pred())
    after_risk = output.split("## 風險提示", 1)[1].split("##", 1)[0]
    assert "無風險提示" in after_risk


def test_skeleton_park_factor_correction_prefilled():
    """## 條件修正 段預填 Park Factor 修正值"""
    from summary_renderer import render_summary
    bundle = _minimal_bundle()
    bundle["merged"]["park_factor"] = 110
    output = render_summary(bundle, _minimal_formula_pred())
    assert "Park Factor: 110" in output or "PF=110" in output
    assert "+0.50" in output or "+0.5" in output


def test_skeleton_expected_runs_table_uses_formula_pred():
    """## 修正後預期得分 base 列從 formula_pred 取得"""
    from summary_renderer import render_summary
    output = render_summary(_minimal_bundle(), _minimal_formula_pred())
    assert "4.5" in output  # home_expected_runs
    assert "4.2" in output  # away_expected_runs


def test_skeleton_line_count_within_70():
    """summary.md template line count ≤ 70（含 +信號 caveat note）"""
    from summary_renderer import render_summary
    output = render_summary(_minimal_bundle(), _minimal_formula_pred())
    assert len(output.split("\n")) <= 70


def test_summary_uses_real_predict_with_formula_keys(tmp_path):
    """Regression：render_summary 對 scoring_formula.predict_with_formula 真實 return shape 仍能填入 base"""
    from summary_renderer import render_summary
    from scoring_formula import predict_with_formula
    merged = {
        "home_recent_rs": 4.5, "home_recent_ra": 4.5,
        "away_recent_rs": 4.2, "away_recent_ra": 4.2,
        "home_recent_30_rs": 4.5, "home_recent_30_ra": 4.5,
        "away_recent_30_rs": 4.2, "away_recent_30_ra": 4.2,
        "park_factor": 100,
    }
    formula_pred = predict_with_formula(merged)
    output = render_summary(_minimal_bundle(), formula_pred)
    expected_section = output.split("## 修正後預期得分", 1)[1].split("##", 1)[0]
    assert "?" not in expected_section.replace("<!-- AI 補 -->", ""), \
        f"base column should be filled from formula_pred, got:\n{expected_section}"


# ---------------------------------------------------------------------------
# Bug A — schema fixes：投手 / 打線 / 牛棚 IL 全部讀真實 schema 才不破洞
# ---------------------------------------------------------------------------

def test_pitcher_section_renders_real_schema():
    """投手 section 必須讀 top-level pitch_hand / age（不是 info.* 子 dict）。"""
    from summary_renderer import render_summary
    output = render_summary(_minimal_bundle(), _minimal_formula_pred())
    pitcher_section = output.split("## 投手對決")[1].split("\n## ")[0]
    # 不該出現 placeholder ?HP（schema 讀錯的徵兆）
    assert "?HP" not in pitcher_section, \
        f"pitcher_section 出現 ?HP，schema drift：\n{pitcher_section}"
    # 應出現實際 hand / age
    assert "Bibee" in pitcher_section
    assert "RHP" in pitcher_section
    assert "27" in pitcher_section
    assert "⚡ 巔峰期" in pitcher_section


def test_lineup_section_uses_tier_and_recent_heat():
    """打線 section 必須讀 top-level tier / recent_heat（不是 tier_emoji / heat_emoji）。"""
    from summary_renderer import render_summary
    output = render_summary(_minimal_bundle(), _minimal_formula_pred())
    lineup_section = output.split("## 打線評級")[1].split("\n## ")[0]
    assert "🟡 Average" in lineup_section
    assert "🟢 Weak" in lineup_section
    assert "⚖️ Normal" in lineup_section
    # 不該出現 ? / ? 這種 schema 讀錯的徵兆
    assert "### HOME — ? / ?" not in lineup_section
    assert "### AWAY — ? / ?" not in lineup_section


def test_bullpen_il_count_from_roster():
    """牛棚段「投手 IL」數量來自 roster.injured_list（不是 merged.home_bullpen_il_count）。"""
    from summary_renderer import render_summary
    output = render_summary(_minimal_bundle(), _minimal_formula_pred())
    bullpen_section = output.split("## 牛棚")[1].split("\n## ")[0]
    # home roster 有 2 個 IL pitcher / away 8 個
    # 不該是 ? 占位
    assert "/ ? /" not in bullpen_section
    # 預期格式 "ERA / 2 / <!-- AI -->" 或類似
    assert "/ 2 /" in bullpen_section or "| 2 |" in bullpen_section
    assert "/ 8 /" in bullpen_section or "| 8 |" in bullpen_section


# ---------------------------------------------------------------------------
# Bug D — Flag 8 條件 1 / 條件 2 標籤分支（不再把 small-sample 標成 era_xera_delta）
# ---------------------------------------------------------------------------

def test_risk_section_labels_era_xera_gap_correctly():
    """Flag 8 條件 1（ERA-xERA gap）→ 標籤含 era_xera_delta、保留正負號"""
    from summary_renderer import render_summary
    bundle = _minimal_bundle()
    bundle["away_pitcher"]["season"] = {"era": 2.10, "ip": 50.0}
    bundle["away_pitcher"]["expected"] = {"xera": 4.64}  # gap = -2.54
    output = render_summary(bundle, _minimal_formula_pred())
    risk_section = output.split("## 風險提示")[1].split("\n## ")[0]
    assert "era_xera_delta=" in risk_section
    assert "-2.54" in risk_section


def test_risk_section_labels_small_sample_regression_correctly():
    """Flag 8 條件 2（IP<30 + prior_year regression）→ 標籤是 prior_year_regression 不是 era_xera_delta"""
    from summary_renderer import render_summary
    bundle = _minimal_bundle()
    # IP < 30 + prior_era - era >= 1.0 → Flag 8 cond 2
    bundle["away_pitcher"]["season"] = {"era": 2.50, "ip": 25.0}
    bundle["away_pitcher"]["expected"] = {"xera": 2.80}  # gap = -0.30，cond 1 不觸發
    bundle["away_pitcher"]["prior_year"] = {"era": 4.20}  # prior_era - era = 1.70
    output = render_summary(bundle, _minimal_formula_pred())
    risk_section = output.split("## 風險提示")[1].split("\n## ")[0]
    assert "prior_year_regression=" in risk_section
    assert "+1.70" in risk_section
    # 不該出現 era_xera_delta（cond 1 沒觸發）
    assert "era_xera_delta" not in risk_section


# ---------------------------------------------------------------------------
# Bug G — 「+ 信號」欄上方須有 BABIP / ERA-xERA 不入欄的 caveat
# ---------------------------------------------------------------------------

def test_expected_runs_has_signal_column_caveat():
    """## 修正後預期得分 段必須在表格前提示 BABIP / era_xera 不入「+ 信號」欄"""
    from summary_renderer import render_summary
    output = render_summary(_minimal_bundle(), _minimal_formula_pred())
    expected_section = output.split("## 修正後預期得分")[1].split("\n## ")[0]
    assert "BABIP" in expected_section
    assert "ERA-xERA" in expected_section or "era_xera" in expected_section
    assert "不入此欄" in expected_section or "不入欄" in expected_section


def test_summary_lineup_section_marks_source():
    """home=official → summary `## 打線評級` HOME 段含「打線來源：🟢 official」。"""
    from summary_renderer import render_summary

    bundle = {
        "home_lineup": {"tier": "🟡 Average", "recent_heat": "⚖️ Normal",
                        "lineup_source": "official"},
        "away_lineup": {"tier": "🟡 Average", "recent_heat": "⚖️ Normal",
                        "lineup_source": "projected"},
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "age": 28},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "age": 30},
        "merged": {"park_factor": 100,
                   "home_bullpen_era": 4.0, "away_bullpen_era": 4.0},
        "home_roster": None, "away_roster": None,
    }
    formula_pred = {"home_score": 4.5, "away_score": 4.0}

    md = render_summary(bundle, formula_pred)
    assert "🟢 official" in md
    assert "🟡 projected" in md


def test_summary_conditional_weather_present():
    """merged.weather 三欄齊 → ## 條件修正 出現「天氣：Sunny, 78°F, ...」+ AI 影響判讀 placeholder。"""
    from summary_renderer import render_summary

    bundle = {
        "home_lineup": {"tier": "?", "recent_heat": "?", "lineup_source": "projected"},
        "away_lineup": {"tier": "?", "recent_heat": "?", "lineup_source": "projected"},
        "home_pitcher": {"name": "?", "pitch_hand": "R"}, "away_pitcher": {"name": "?", "pitch_hand": "R"},
        "merged": {
            "park_factor": 100,
            "weather": {"condition": "Sunny", "temp_f": 78,
                        "wind_text": "10 mph, Out To CF", "indoor": False},
        },
        "home_roster": None, "away_roster": None,
    }
    md = render_summary(bundle, {"home_score": 0, "away_score": 0})
    assert "天氣：Sunny, 78°F, wind 10 mph, Out To CF" in md
    assert "AI 補：對得分 / HR 影響判讀" in md or "影響判讀" in md


def test_summary_conditional_weather_indoor():
    """indoor=True → 顯示「天氣：室內（Roof Closed，不適用）」、不出現 AI placeholder。"""
    from summary_renderer import render_summary

    bundle = {
        "home_lineup": {"tier": "?", "recent_heat": "?", "lineup_source": "projected"},
        "away_lineup": {"tier": "?", "recent_heat": "?", "lineup_source": "projected"},
        "home_pitcher": {"name": "?"}, "away_pitcher": {"name": "?"},
        "merged": {
            "park_factor": 100,
            "weather": {"condition": "Roof Closed", "temp_f": 72,
                        "wind_text": None, "indoor": True},
        },
        "home_roster": None, "away_roster": None,
    }
    md = render_summary(bundle, {"home_score": 0, "away_score": 0})
    assert "天氣：室內" in md
    assert "Roof Closed" in md
    assert "對得分 / HR 影響判讀" not in md


def test_summary_conditional_weather_absent():
    """weather=None → 顯示「天氣：未公布（跳過天氣分析）」、不出現 AI placeholder。"""
    from summary_renderer import render_summary

    bundle = {
        "home_lineup": {"tier": "?", "recent_heat": "?", "lineup_source": "projected"},
        "away_lineup": {"tier": "?", "recent_heat": "?", "lineup_source": "projected"},
        "home_pitcher": {"name": "?"}, "away_pitcher": {"name": "?"},
        "merged": {"park_factor": 100, "weather": None},
        "home_roster": None, "away_roster": None,
    }
    md = render_summary(bundle, {"home_score": 0, "away_score": 0})
    assert "天氣：未公布" in md
    assert "對得分 / HR 影響判讀" not in md
