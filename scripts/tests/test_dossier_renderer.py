"""Tests for dossier_renderer."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Task 8a tests (unchanged)
# ---------------------------------------------------------------------------

def test_select_top5_pa_filter():
    """PA ≥ 30、IL'd 排除、最多 5 人，按 PA 降序"""
    from dossier_renderer import select_top5_vs_pitcher
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "season_ops": 0.800, "vs_rhp_ops": 0.750,
             "last7_ops": 0.700, "last7_babip": 0.300, "ev95_pct": 50.0, "barrel_pct": 10.0},
            {"name": "B", "pa": 80,  "season_ops": 0.750, "vs_rhp_ops": 0.700,
             "last7_ops": 0.650, "last7_babip": 0.280, "ev95_pct": 45.0, "barrel_pct": 8.0},
            {"name": "C", "pa": 25,  "season_ops": 0.900, "vs_rhp_ops": 0.850,
             "last7_ops": 0.800, "last7_babip": 0.330, "ev95_pct": 55.0, "barrel_pct": 12.0},  # PA < 30
            {"name": "D", "pa": 60,  "season_ops": 0.700, "vs_rhp_ops": 0.650,
             "last7_ops": 0.600, "last7_babip": 0.260, "ev95_pct": 40.0, "barrel_pct": 7.0},
            {"name": "E", "pa": 45,  "season_ops": 0.680, "vs_rhp_ops": 0.620,
             "last7_ops": 0.580, "last7_babip": 0.240, "ev95_pct": 38.0, "barrel_pct": 6.0},
            {"name": "F", "pa": 35,  "season_ops": 0.660, "vs_rhp_ops": 0.600,
             "last7_ops": 0.560, "last7_babip": 0.220, "ev95_pct": 36.0, "barrel_pct": 5.0},
            {"name": "G", "pa": 32,  "season_ops": 0.640, "vs_rhp_ops": 0.580,
             "last7_ops": 0.540, "last7_babip": 0.200, "ev95_pct": 34.0, "barrel_pct": 4.0},
        ]
    }
    il_names = set()
    result = select_top5_vs_pitcher(lineup, il_names)
    names = [p["name"] for p in result]
    assert names == ["A", "B", "D", "E", "F"]  # G 被擠出（取 top 5）；C 被 PA 過濾


def test_select_top5_excludes_il():
    """IL 名單上的球員直接濾掉"""
    from dossier_renderer import select_top5_vs_pitcher
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "season_ops": 0.8},
            {"name": "B", "pa": 80,  "season_ops": 0.75},
            {"name": "C", "pa": 60,  "season_ops": 0.7},
        ]
    }
    il_names = {"B"}
    result = select_top5_vs_pitcher(lineup, il_names)
    names = [p["name"] for p in result]
    assert names == ["A", "C"]


def test_select_top5_fewer_than_5_returns_what_exists():
    """候選池 < 5 → 返回所有合格者"""
    from dossier_renderer import select_top5_vs_pitcher
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "season_ops": 0.8},
            {"name": "B", "pa": 50,  "season_ops": 0.75},
        ]
    }
    result = select_top5_vs_pitcher(lineup, set())
    assert len(result) == 2


def test_select_top5_last7_top1_outside_pa_top5():
    """last7 OPS top1 不在 PA top5 內 → annotate"""
    from dossier_renderer import find_last7_top1_outside_pa_top5
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "last7_ops": 0.700},
            {"name": "B", "pa": 90,  "last7_ops": 0.650},
            {"name": "C", "pa": 80,  "last7_ops": 0.600},
            {"name": "D", "pa": 70,  "last7_ops": 0.550},
            {"name": "E", "pa": 60,  "last7_ops": 0.500},
            {"name": "Schneemann", "pa": 35, "last7_ops": 1.164},  # 不在 PA top5（被 E 擠掉）但 last7 OPS top1
        ]
    }
    pa_top5 = {"A", "B", "C", "D", "E"}
    annotation = find_last7_top1_outside_pa_top5(lineup, pa_top5, set())
    assert annotation is not None
    assert annotation["name"] == "Schneemann"
    assert annotation["last7_ops"] == 1.164


def test_select_top5_last7_top1_already_in_pa_top5_returns_none():
    """last7 OPS top1 已在 PA top5 內 → 不需 annotate"""
    from dossier_renderer import find_last7_top1_outside_pa_top5
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "last7_ops": 1.000},  # PA top1 + last7 top1
            {"name": "B", "pa": 90,  "last7_ops": 0.500},
        ]
    }
    pa_top5 = {"A", "B"}
    assert find_last7_top1_outside_pa_top5(lineup, pa_top5, set()) is None


# ---------------------------------------------------------------------------
# Minimal bundle fixture helper
# ---------------------------------------------------------------------------

def _minimal_bundle(**overrides):
    """Return a minimal valid bundle for testing section renderers."""
    bundle = {
        "game_data": {
            "game": {
                "date": "2026-04-28T22:10:00Z",
                "status": "Preview",
                "venue": "Test Park",
                "home": {"team": "Home Team", "probable_pitcher": "H Pitcher"},
                "away": {"team": "Away Team", "probable_pitcher": "A Pitcher"},
            },
            "home_recent": {"record": "5-5", "wins": 5, "losses": 5,
                            "rs_per_game": 4.0, "ra_per_game": 4.0, "run_diff": 0, "streak": 2, "games": []},
            "away_recent": {"record": "6-4", "wins": 6, "losses": 4,
                            "rs_per_game": 4.5, "ra_per_game": 3.5, "run_diff": 10, "streak": -1, "games": []},
            "home_recent_30": {"record": "15-15", "wins": 15, "losses": 15,
                               "rs_per_game": 3.8, "ra_per_game": 4.0, "run_diff": -4, "streak": 2},
            "away_recent_30": {"record": "17-13", "wins": 17, "losses": 13,
                               "rs_per_game": 4.8, "ra_per_game": 4.0, "run_diff": 16, "streak": -1},
            "home_season": {"record": "15-16", "wins": 15, "losses": 16,
                            "rs_per_game": 3.87, "ra_per_game": 4.16, "run_diff": -9, "streak": 2},
            "away_season": {"record": "17-11", "wins": 17, "losses": 11,
                            "rs_per_game": 4.82, "ra_per_game": 4.82, "run_diff": 0, "streak": -1},
            "home_season_games_count": 31,
            "away_season_games_count": 28,
            "series_prev": None,
        },
        "home_roster": {"injured_list": [], "summary": {}},
        "away_roster": {"injured_list": [], "summary": {}},
        "home_pitcher": {
            "name": "H Pitcher", "mlbam_id": 111111, "age": 27, "pitch_hand": "R",
            "age_assessment": "⚡ 巔峰期", "tier": "🟢 Back-end Starter",
            "season": {"era": 4.45, "whip": 1.45, "k_pct": 19.8, "bb_pct": 8.4,
                       "k_bb_pct": 11.4, "fip": 4.62, "xfip": 3.85, "ip": 30.0, "gs": 6},
            "expected": {"xera": 4.64},
            "statcast": {"avg_velo": 87.3, "max_velo": 96.1, "hard_hit_pct": 34.5,
                         "barrel_pct": 8.5, "ev95percent": 53.2, "whiff_pct": 12.1,
                         "pitch_types": {"FF": 27.8, "FC": 27.5, "CH": 17.9}},
            "game_log": [
                {"date": "2026-03-26", "ip": 5.0, "er": 3},
                {"date": "2026-03-31", "ip": 4.0, "er": 1},
                {"date": "2026-04-06", "ip": 4.7, "er": 1},
            ],
            "platoon_splits": {
                "vs_left": {"avg": ".238", "obp": ".314", "slg": ".492", "bf": 70},
                "vs_right": {"avg": ".316", "obp": ".361", "slg": ".386", "bf": 61},
            },
            "prior_year": {"era": 4.24},
        },
        "away_pitcher": {
            "name": "A Pitcher", "mlbam_id": 222222, "age": 35, "pitch_hand": "R",
            "age_assessment": "📉📉 明顯退化", "tier": "🟠 Strong Ace",
            "season": {"era": 2.1, "whip": 1.1, "k_pct": 16.4, "bb_pct": 6.6,
                       "k_bb_pct": 9.8, "fip": 3.87, "xfip": 4.34, "ip": 30.0, "gs": 5},
            "expected": {"xera": 4.64},
            "statcast": {"avg_velo": 86.8, "max_velo": 94.4, "hard_hit_pct": 25.8,
                         "barrel_pct": 8.5, "ev95percent": 33.0, "whiff_pct": 7.1,
                         "pitch_types": {"SI": 31.3, "CH": 27.1, "FC": 18.8}},
            "game_log": [
                {"date": "2026-03-30", "ip": 6.0, "er": 2},
                {"date": "2026-04-05", "ip": 6.0, "er": 1},
                {"date": "2026-04-11", "ip": 4.7, "er": 1},
            ],
            "platoon_splits": {
                "vs_left": {"avg": ".239", "obp": ".311", "slg": ".388", "bf": 74},
                "vs_right": {"avg": ".196", "obp": ".208", "slg": ".283", "bf": 48},
            },
            "prior_year": {"era": 4.45},
        },
        "home_lineup": {
            "team": "HOME", "team_id": 100,
            "tier": "🟡 Average", "recent_heat": "⚖️ Normal",
            "avg_ops": 0.707, "avg_xwoba": 0.331, "avg_k_pct": 20.3, "avg_bb_pct": 10.7,
            "last7_babip": 0.284,
            "chain": {"obp_top3": 0.329, "slg_mid": 0.452},
            "lineup": [
                {
                    "name": "Player H1", "pa": 130, "ops": 0.804, "ev95pct": 45.7, "barrel_pct": 10.9,
                    "platoon": {"vs_rhp": {"ops": ".703"}, "vs_lhp": {"ops": "1.016"}},
                    "last_7": {"ops": ".663", "babip": ".273"},
                },
                {
                    "name": "Player H2", "pa": 121, "ops": 0.574, "ev95pct": 8.6, "barrel_pct": 1.1,
                    "platoon": {"vs_rhp": {"ops": ".652"}, "vs_lhp": {"ops": ".419"}},
                    "last_7": {"ops": ".380", "babip": ".150"},
                },
            ],
        },
        "away_lineup": {
            "team": "AWAY", "team_id": 200,
            "tier": "🟢 Weak", "recent_heat": "⚖️ Normal",
            "avg_ops": 0.711, "avg_xwoba": 0.304, "avg_k_pct": 17.8, "avg_bb_pct": 7.8,
            "last7_babip": 0.241,
            "chain": {"obp_top3": 0.368, "slg_mid": 0.299},
            "lineup": [
                {
                    "name": "Player A1", "pa": 126, "ops": 0.833, "ev95pct": 45.2, "barrel_pct": 10.8,
                    "platoon": {"vs_rhp": {"ops": ".830"}, "vs_lhp": {"ops": ".824"}},
                    "last_7": {"ops": ".917", "babip": ".227"},
                },
                {
                    "name": "Player A2", "pa": 125, "ops": 0.91, "ev95pct": 43.5, "barrel_pct": 6.5,
                    "platoon": {"vs_rhp": {"ops": ".921"}, "vs_lhp": {"ops": ".873"}},
                    "last_7": {"ops": ".831", "babip": ".300"},
                },
            ],
        },
        "merged": {
            "home_bullpen_era": 4.57,
            "away_bullpen_era": 5.18,
            "park_factor": 101.0,
            "_meta": {
                "home_team": "Home Team",
                "away_team": "Away Team",
                "home_sp": "H Pitcher",
                "away_sp": "A Pitcher",
                "home_sp_starts": 6,
                "away_sp_starts": 5,
                "venue": "Progressive Field",
                "game_date": "2026-04-28T22:10:00Z",
                "official_date": "2026-04-28",
            },
        },
    }
    bundle.update(overrides)
    return bundle


# ---------------------------------------------------------------------------
# Per-section unit tests
# ---------------------------------------------------------------------------

def test_render_header_basic():
    from dossier_renderer import _render_header
    bundle = _minimal_bundle()
    lines = _render_header(bundle)
    header = lines[0]
    assert header.startswith("# Game Dossier — ")
    assert "Away Team" in header
    assert "Home Team" in header
    # Header uses ET date (= official_date); UTC 22:10Z = ET 18:10 (officialDate 04-28)
    assert "2026-04-28" in header


def test_render_header_missing_fields():
    """Should not crash on empty bundle."""
    from dossier_renderer import _render_header
    lines = _render_header({})
    assert len(lines) >= 1
    assert lines[0].startswith("# Game Dossier")


def test_render_game_info_contains_key_fields():
    from dossier_renderer import _render_game_info
    bundle = _minimal_bundle()
    lines = _render_game_info(bundle)
    text = "\n".join(lines)
    assert "## 比賽資訊" in text
    assert "Progressive Field" in text
    assert "H Pitcher" in text
    assert "A Pitcher" in text
    assert "GS 6" in text
    assert "GS 5" in text
    # ET only; UTC 22:10Z = ET 18:10 (officialDate 04-28)
    assert "- 日期 (ET): 2026-04-28" in text
    assert "2026-04-28 18:10 ET" in text


def test_render_record_summary_structure():
    from dossier_renderer import _render_record_summary
    bundle = _minimal_bundle()
    lines = _render_record_summary(bundle)
    text = "\n".join(lines)
    assert "## 戰績速查" in text
    assert "近 10" in text
    assert "近 30" in text
    assert "本季" in text
    assert "趨勢" in text


def test_render_record_summary_streak_format():
    """Streak should be formatted as +N / -N."""
    from dossier_renderer import _render_record_summary
    bundle = _minimal_bundle()
    lines = _render_record_summary(bundle)
    text = "\n".join(lines)
    assert "+2" in text or "−2" in text or "+5" in text or "−3" in text


def test_render_series_context_with_prev_game():
    from dossier_renderer import _render_series_context
    bundle = _minimal_bundle()
    bundle["game_data"]["series_prev"] = {
        "date": "2026-04-27",
        "home": "Home Team",
        "away": "Away Team",
        "home_score": 2,
        "away_score": 3,
        "winner": "Away Team",
    }
    lines = _render_series_context(bundle)
    text = "\n".join(lines)
    assert "## 系列脈絡" in text
    assert "G1" in text
    assert "Streak" in text


def test_render_series_context_no_prev():
    """No series_prev → should still render without crashing."""
    from dossier_renderer import _render_series_context
    bundle = _minimal_bundle()
    lines = _render_series_context(bundle)
    text = "\n".join(lines)
    assert "## 系列脈絡" in text


def test_render_pitcher_matchup_structure():
    from dossier_renderer import _render_pitcher_matchup
    bundle = _minimal_bundle()
    lines = _render_pitcher_matchup(bundle)
    text = "\n".join(lines)
    assert "## 投手對決" in text
    assert "Tier (script)" in text
    assert "ERA / xERA" in text
    assert "FIP / xFIP" in text
    assert "K-BB% / WHIP" in text
    assert "velo" in text
    assert "whiff%" in text
    assert "vs LHB" in text
    assert "vs RHB" in text
    assert "近 3 場 ER/IP" in text
    assert "風險提示" in text


def test_render_pitcher_matchup_flag8_triggers():
    """When ERA-xERA gap ≥ 1.5 for away pitcher, ⚠️ appears."""
    from dossier_renderer import _render_pitcher_matchup
    bundle = _minimal_bundle()
    # away pitcher already has era=2.1, xera=4.64 → gap=2.54 → Flag 8
    lines = _render_pitcher_matchup(bundle)
    text = "\n".join(lines)
    assert "⚠️" in text
    assert "Flag 8" in text


def test_render_pitcher_matchup_no_flag_when_gap_small():
    """When ERA-xERA gap < 1.5, no ⚠️ flag row."""
    from dossier_renderer import _render_pitcher_matchup
    bundle = _minimal_bundle()
    # Both pitchers: era close to xera
    bundle["home_pitcher"]["season"]["era"] = 4.0
    bundle["home_pitcher"]["expected"]["xera"] = 4.1
    bundle["away_pitcher"]["season"]["era"] = 3.8
    bundle["away_pitcher"]["expected"]["xera"] = 3.9
    lines = _render_pitcher_matchup(bundle)
    text = "\n".join(lines)
    assert "⚠️" not in text


def test_render_lineup_overview_structure():
    from dossier_renderer import _render_lineup_overview
    bundle = _minimal_bundle()
    lines = _render_lineup_overview(bundle)
    text = "\n".join(lines)
    assert "## 打線" in text
    assert "Tier (script)" in text
    assert "Heat (script)" in text
    assert "xwOBA / OPS" in text
    assert "K% / BB%" in text
    assert "chain OBP top3" in text
    assert "last7 BABIP" in text
    assert "對方先發手感" in text or "Top 5" in text or "Top 5 vs" in text


def test_render_lineup_overview_flag3_triggers():
    """Away lineup last7 BABIP=0.241 → Flag 3."""
    from dossier_renderer import _render_lineup_overview
    bundle = _minimal_bundle()
    lines = _render_lineup_overview(bundle)
    text = "\n".join(lines)
    assert "⚠️" in text
    assert "Flag 3" in text


def test_render_lineup_overview_top5_rows():
    """Top 5 table should have player rows."""
    from dossier_renderer import _render_lineup_overview
    bundle = _minimal_bundle()
    lines = _render_lineup_overview(bundle)
    text = "\n".join(lines)
    assert "Player H1" in text
    assert "Player A1" in text


def test_render_lineup_overview_includes_wrc_plus_row():
    """Backlog #2: avg wRC+ row renders in lineup table with HOME/AWAY values."""
    from dossier_renderer import _render_lineup_overview
    bundle = _minimal_bundle()
    bundle["home_lineup"]["avg_wrc_plus"] = 120.0
    bundle["away_lineup"]["avg_wrc_plus"] = 95.0
    lines = _render_lineup_overview(bundle)
    text = "\n".join(lines)
    assert "avg wRC+" in text
    # wRC+ rendered as integer (decimals=0): 120.0 → "120", 95.0 → "95"
    assert "| 120 | 95 |" in text


def test_render_lineup_overview_wrc_plus_dash_when_none():
    """Backlog #2: avg_wrc_plus = None (early season / fetch failure) → row shows '—'."""
    from dossier_renderer import _render_lineup_overview
    bundle = _minimal_bundle()
    bundle["home_lineup"]["avg_wrc_plus"] = None
    bundle["away_lineup"]["avg_wrc_plus"] = None
    lines = _render_lineup_overview(bundle)
    wrc_row = [l for l in lines if "avg wRC+" in l][0]
    # Both home and away cells are "—"
    assert "| — | — |" in wrc_row


def test_render_bullpen_park_structure():
    from dossier_renderer import _render_bullpen_park
    bundle = _minimal_bundle()
    lines = _render_bullpen_park(bundle)
    text = "\n".join(lines)
    assert "## 牛棚 / Park" in text
    assert "Bullpen ERA" in text
    # Cleanup #5: label uses core bullpen IL semantics (matches merged.{side}_core_bullpen_il_count)
    assert "Core 牛棚 IL" in text
    assert "Park Factor" in text
    assert "4.57" in text
    assert "5.18" in text


def test_render_bullpen_park_il_count():
    """Core bullpen IL count should appear with names list."""
    from dossier_renderer import _render_bullpen_park
    bundle = _minimal_bundle()
    bundle["merged"]["home_core_bullpen_il_count"] = 2
    bundle["merged"]["away_core_bullpen_il_count"] = 1
    bundle["home_roster"] = {
        "injured_list": [
            {"name": "P1", "status": "Injured 15-Day", "position": "Pitcher", "core_role": "Closer"},
            {"name": "P2", "status": "Injured 60-Day", "position": "Pitcher", "core_role": "Setup"},
        ]
    }
    bundle["away_roster"] = {
        "injured_list": [
            {"name": "P3", "status": "Injured 15-Day", "position": "Pitcher", "core_role": "High-leverage RP"},
        ]
    }
    lines = _render_bullpen_park(bundle)
    text = "\n".join(lines)
    # Counts (HOME=2, AWAY=1) come from merged
    assert "| 2 | 1 |" in text
    # Names render in IL 名單 row
    assert "P1" in text
    assert "P2" in text
    assert "P3" in text


def test_render_bullpen_park_count_from_merged_not_substring():
    """Cleanup #5: count must come from merged.{side}_core_bullpen_il_count, never re-derived
    from roster.position substring. Roster IL without core_role must NOT inflate the count."""
    from dossier_renderer import _render_bullpen_park
    bundle = _minimal_bundle()
    # Merged is the canonical source: HOME=3, AWAY=0
    bundle["merged"]["home_core_bullpen_il_count"] = 3
    bundle["merged"]["away_core_bullpen_il_count"] = 0
    # Roster contradicts: HOME has 1 pitcher entry, AWAY has 1 pitcher entry (no core_role).
    # Old substring filter on position="Pitcher" would yield HOME=1, AWAY=1.
    # New code must yield HOME=3, AWAY=0 — strictly from merged.
    bundle["home_roster"] = {
        "injured_list": [
            {"name": "OnlyOne", "position": "Pitcher", "status": "Injured 15-Day"},
        ]
    }
    bundle["away_roster"] = {
        "injured_list": [
            {"name": "SPName", "position": "Pitcher", "status": "Injured 60-Day"},
        ]
    }
    lines = _render_bullpen_park(bundle)
    text = "\n".join(lines)
    # Count row: "| Core 牛棚 IL... | 3 | 0 |"
    assert "| 3 | 0 |" in text
    # Old substring would have shown "| 1 | 1 |" — must NOT appear
    assert "| 1 | 1 |" not in text


def test_render_bullpen_park_names_filter_by_core_role():
    """Cleanup #5: IL names list shows only core_role ∈ CORE_BULLPEN_ROLES
    (Closer / Setup / High-leverage RP / Co-Closer). SP IL or Long Relief must be hidden."""
    from dossier_renderer import _render_bullpen_park
    bundle = _minimal_bundle()
    bundle["merged"]["home_core_bullpen_il_count"] = 1
    bundle["merged"]["away_core_bullpen_il_count"] = 0
    bundle["home_roster"] = {
        "injured_list": [
            {"name": "AceSP", "status": "Injured 60-Day", "position": "Pitcher",
             "core_role": "Starter"},
            {"name": "CloserGuy", "status": "Injured 15-Day", "position": "Pitcher",
             "core_role": "Closer"},
            {"name": "LongRelief", "status": "Injured 15-Day", "position": "Pitcher",
             "core_role": "Long Relief"},
        ]
    }
    lines = _render_bullpen_park(bundle)
    text = "\n".join(lines)
    assert "CloserGuy" in text       # core (Closer)
    assert "AceSP" not in text       # Starter is not core
    assert "LongRelief" not in text  # Long Relief is not core


def test_render_risk_summary_with_flags():
    """Risk summary should list all triggered flags."""
    from dossier_renderer import _render_risk_summary
    bundle = _minimal_bundle()
    # away_pitcher has Flag 8 (era=2.1, xera=4.64)
    # away_lineup has Flag 3 (last7_babip=0.241)
    lines = _render_risk_summary(bundle)
    text = "\n".join(lines)
    assert "## ⚠️ 風險提示摘要" in text
    assert "Flag 8" in text
    assert "Flag 3" in text


def test_render_risk_summary_no_flags():
    """No flags → '無風險提示'."""
    from dossier_renderer import _render_risk_summary
    bundle = _minimal_bundle()
    # Set ERA close to xERA for no Flag 8
    bundle["away_pitcher"]["season"]["era"] = 4.5
    bundle["away_pitcher"]["expected"]["xera"] = 4.6
    # Set BABIP to normal for no Flag 3
    bundle["away_lineup"]["last7_babip"] = 0.300
    bundle["home_lineup"]["last7_babip"] = 0.300
    lines = _render_risk_summary(bundle)
    text = "\n".join(lines)
    assert "無風險提示" in text


# ---------------------------------------------------------------------------
# PR-3 commit 13: ## 🎯 訊號摘要 section
# ---------------------------------------------------------------------------

def _bundle_with_fired_signals():
    """Bundle wired so signals_lib.compute_all_signals fires multiple signals."""
    bundle = _minimal_bundle()
    # tier_mismatch fires on HOME
    bundle["home_pitcher"]["tier_gap"] = {
        "expected_score": 80.0, "era_only_score": 60, "gap": 20.0,
    }
    # reverse_platoon fires on HOME (RHP, RHB OPS > LHB OPS)
    bundle["home_pitcher"]["pitch_hand"] = "R"
    bundle["home_pitcher"]["platoon_splits"] = {
        "vs_left": {"ops": ".545", "bf": 60},
        "vs_right": {"ops": ".932", "bf": 80},
    }
    # core_il_count fires on HOME with severity high
    bundle["merged"]["home_core_bullpen_il_count"] = 2
    bundle["merged"]["away_core_bullpen_il_count"] = 0
    # strong_park fires (PF 112)
    bundle["merged"]["park_factor"] = 112
    return bundle


def test_render_signal_summary_section_appears_before_pitcher_matchup():
    """Signal summary must appear above ## 投手對決 in the rendered dossier."""
    from dossier_renderer import render_dossier
    bundle = _bundle_with_fired_signals()
    output = render_dossier(bundle)
    sig_idx = output.index("## 🎯 訊號摘要")
    pitcher_idx = output.index("## 投手對決")
    assert sig_idx < pitcher_idx


def test_render_signal_summary_lists_fired_signals():
    """Each fired signal renders one line with side + label."""
    from dossier_renderer import _render_signal_summary
    bundle = _bundle_with_fired_signals()
    lines = _render_signal_summary(bundle)
    text = "\n".join(lines)
    assert "## 🎯 訊號摘要" in text
    # tier_mismatch HOME label fragment
    assert "ERA 低估" in text
    # reverse_platoon HOME label fragment
    assert "reverse" in text.lower() or "反向" in text
    # core_il_count HOME ×2 — verify "core IL ×2" or similar in label
    assert "×2" in text or "core IL" in text
    # strong_park GAME — PF 112 label
    assert "112" in text or "打者友善" in text


def test_render_signal_summary_no_fires_shows_default_message():
    """When zero signals fire, section shows '無顯著訊號'."""
    from dossier_renderer import _render_signal_summary
    bundle = _minimal_bundle()
    # Wipe out anything that might fire
    bundle["home_pitcher"]["tier_gap"] = None
    bundle["away_pitcher"]["tier_gap"] = None
    bundle["home_pitcher"]["platoon_splits"] = {}
    bundle["away_pitcher"]["platoon_splits"] = {}
    bundle["home_pitcher"]["statcast"] = {"pitch_types": {"FF": 35.0, "SL": 35.0, "CH": 30.0}}
    bundle["away_pitcher"]["statcast"] = {"pitch_types": {"FF": 35.0, "SL": 35.0, "CH": 30.0}}
    bundle["home_lineup"]["recent_heat"] = "⚖️ Normal"
    bundle["away_lineup"]["recent_heat"] = "⚖️ Normal"
    bundle["home_lineup"]["last7_babip"] = 0.300
    bundle["away_lineup"]["last7_babip"] = 0.300
    bundle["home_lineup"]["lineup"] = []
    bundle["away_lineup"]["lineup"] = []
    bundle["merged"]["park_factor"] = 100
    bundle["merged"]["home_core_bullpen_il_count"] = 0
    bundle["merged"]["away_core_bullpen_il_count"] = 0
    lines = _render_signal_summary(bundle)
    text = "\n".join(lines)
    assert "## 🎯 訊號摘要" in text
    assert "無顯著訊號" in text


def test_render_signal_summary_severity_emoji_prefix():
    """Each fired signal line is prefixed with severity emoji (🔴/🟠/ℹ️)."""
    from dossier_renderer import _render_signal_summary
    bundle = _bundle_with_fired_signals()
    lines = _render_signal_summary(bundle)
    text = "\n".join(lines)
    # core_il_count ×2 is high severity → 🔴
    # strong_park PF 112 is medium → 🟠
    # tier_mismatch gap 20 is high → 🔴
    assert "🔴" in text
    assert "🟠" in text


def test_render_signal_summary_marks_short_half_life_with_hourglass():
    """Short half_life signals (heat / core_il_count / etc.) get ⏳ badge so
    analyst knows the reading is short-window (對手會調整). Structural and
    medium signals don't get the badge. See reference/matchup-factors.md §半衰期."""
    from dossier_renderer import _render_signal_summary
    bundle = _bundle_with_fired_signals()
    text = "\n".join(_render_signal_summary(bundle))
    # core_il_count is short → ⏳ on that line
    # Find line containing "core IL" or "×2" and assert ⏳ adjacent
    core_lines = [ln for ln in text.split("\n") if "core IL" in ln or "×2" in ln]
    assert core_lines, "core_il_count line missing"
    assert any("⏳" in ln for ln in core_lines), (
        f"core_il_count is short half_life — expected ⏳; got: {core_lines}"
    )


def test_render_signal_summary_does_not_mark_structural_with_hourglass():
    """strong_park is structural (multi-year) — must NOT carry ⏳."""
    from dossier_renderer import _render_signal_summary
    bundle = _bundle_with_fired_signals()
    text = "\n".join(_render_signal_summary(bundle))
    park_lines = [ln for ln in text.split("\n") if "park" in ln.lower() or "112" in ln]
    assert park_lines, "strong_park line missing"
    assert not any("⏳" in ln for ln in park_lines), (
        f"strong_park is structural — must not carry ⏳; got: {park_lines}"
    )


def test_render_dossier_signal_summary_in_required_sections():
    """## 🎯 訊號摘要 should now be among rendered H2 markers."""
    from dossier_renderer import render_dossier
    bundle = _bundle_with_fired_signals()
    output = render_dossier(bundle)
    assert "## 🎯 訊號摘要" in output


# ---------------------------------------------------------------------------
# PR-3 commit 14: pitcher matchup table — new 4 rows + <details> collapse
# ---------------------------------------------------------------------------

def _bundle_with_pr2_pitcher_fields():
    """Bundle with tier_v2 / tier_gap / arsenal_top fields populated for both
    sides + lineup tier_vs_hand."""
    bundle = _minimal_bundle()
    bundle["home_pitcher"]["tier_v2"] = "🟠 Strong Ace"
    bundle["home_pitcher"]["tier_gap"] = {
        "expected_score": 75.0, "era_only_score": 70, "gap": 5.0,
    }
    bundle["home_pitcher"]["arsenal"] = [
        {"pitch_type": "SL", "usage": 32.0, "rv_per_100": -1.8},
        {"pitch_type": "FF", "usage": 25.0, "rv_per_100": 0.4},
        {"pitch_type": "SI", "usage": 22.0, "rv_per_100": -0.6},
    ]
    bundle["away_pitcher"]["tier_v2"] = "🟡 Solid Starter"
    bundle["away_pitcher"]["tier_gap"] = {
        "expected_score": 55.0, "era_only_score": 70, "gap": -15.0,
    }
    bundle["away_pitcher"]["arsenal"] = [
        {"pitch_type": "SI", "usage": 45.0, "rv_per_100": -0.2},
        {"pitch_type": "CU", "usage": 30.0, "rv_per_100": -1.5},
        {"pitch_type": "CH", "usage": 22.0, "rv_per_100": 0.3},
    ]
    # Cleanup #8: dossier reads pre-filtered arsenal_top from merged.{side}_pitcher
    # (parallels production where merge_game_data.extract_pitcher_nested writes it).
    bundle["merged"]["home_pitcher"] = {
        "arsenal_top": list(bundle["home_pitcher"]["arsenal"]),
    }
    bundle["merged"]["away_pitcher"] = {
        "arsenal_top": list(bundle["away_pitcher"]["arsenal"]),
    }
    bundle["home_lineup"]["tier_vs_lhp"] = "🟠 Strong"
    bundle["home_lineup"]["tier_vs_rhp"] = "🟡 Average"
    bundle["away_lineup"]["tier_vs_lhp"] = "🟡 Average"
    bundle["away_lineup"]["tier_vs_rhp"] = "🟠 Strong"
    return bundle


def test_pitcher_matchup_renders_tier_v2_row():
    from dossier_renderer import _render_pitcher_matchup
    bundle = _bundle_with_pr2_pitcher_fields()
    text = "\n".join(_render_pitcher_matchup(bundle))
    assert "Tier (xFIP-blend)" in text
    assert "🟠 Strong Ace" in text
    assert "🟡 Solid Starter" in text


def test_pitcher_matchup_renders_tier_gap_row():
    from dossier_renderer import _render_pitcher_matchup
    bundle = _bundle_with_pr2_pitcher_fields()
    text = "\n".join(_render_pitcher_matchup(bundle))
    assert "Tier gap" in text
    # gap +5 / gap -15
    assert "+5" in text or "+5.0" in text
    assert "-15" in text or "-15.0" in text


def test_pitcher_matchup_renders_lineup_tier_vs_hand_row():
    """Pitcher's HOME column shows AWAY lineup's tier vs HOME pitcher's hand
    (the threat to the pitcher), and vice versa."""
    from dossier_renderer import _render_pitcher_matchup
    bundle = _bundle_with_pr2_pitcher_fields()
    # HOME pitcher hand R → AWAY lineup tier_vs_rhp = 🟠 Strong
    # AWAY pitcher hand R → HOME lineup tier_vs_rhp = 🟡 Average
    text = "\n".join(_render_pitcher_matchup(bundle))
    assert "對手打線 tier" in text or "對手手別" in text


def test_pitcher_matchup_renders_arsenal_top3_row():
    from dossier_renderer import _render_pitcher_matchup
    bundle = _bundle_with_pr2_pitcher_fields()
    text = "\n".join(_render_pitcher_matchup(bundle))
    assert "主球種 RV" in text or "RV/100" in text
    # HOME's SL -1.8 should appear
    assert "SL" in text and "-1.8" in text


def test_pitcher_matchup_renders_stuff_plus_row():
    """Stuff+ / Pitching+ row appears in the top 投手對決 table after refactor.
    velo stays in the <details> block (informational only)."""
    from dossier_renderer import _render_pitcher_matchup
    bundle = _bundle_with_pr2_pitcher_fields()
    bundle["home_pitcher"]["stuff"] = {
        "stuff_plus": 122.5, "location_plus": 105.3, "pitching_plus": 115.8,
    }
    bundle["away_pitcher"]["stuff"] = {
        "stuff_plus": 95.0, "location_plus": 100.0, "pitching_plus": 96.5,
    }
    text = "\n".join(_render_pitcher_matchup(bundle))
    assert "Stuff+" in text or "Pitching+" in text
    # HOME 122.5 should appear (formatted as 123 with .0f or 122.5 with .1f)
    assert "122" in text or "123" in text
    assert "115" in text or "116" in text  # pitching+ home


def test_pitcher_matchup_handles_missing_stuff_gracefully():
    """If pitcher.stuff missing (legacy data), Stuff+ row still renders with —
    placeholder rather than crashing."""
    from dossier_renderer import _render_pitcher_matchup
    bundle = _bundle_with_pr2_pitcher_fields()
    # No stuff key on either pitcher
    text = "\n".join(_render_pitcher_matchup(bundle))
    assert "Stuff+" in text or "Pitching+" in text
    # Should render placeholder, not crash
    assert "—" in text


def test_pitcher_matchup_legacy_13_rows_inside_details():
    """The existing 13-row deep-dive table moves under <details>."""
    from dossier_renderer import _render_pitcher_matchup
    bundle = _bundle_with_pr2_pitcher_fields()
    text = "\n".join(_render_pitcher_matchup(bundle))
    assert "<details>" in text
    assert "</details>" in text
    # Existing rows still appear (substring) — moved, not removed
    assert "ERA / xERA" in text
    assert "K-BB% / WHIP" in text
    assert "vs LHB (slash)" in text


def test_pitcher_matchup_section_header_unchanged():
    """## 投手對決 header preserved."""
    from dossier_renderer import _render_pitcher_matchup
    bundle = _bundle_with_pr2_pitcher_fields()
    text = "\n".join(_render_pitcher_matchup(bundle))
    assert "## 投手對決" in text


def test_pitcher_matchup_handles_missing_pr2_fields_gracefully():
    """If tier_v2 / tier_gap / arsenal absent (legacy data), new rows show '—'
    but section still renders without crashing."""
    from dossier_renderer import _render_pitcher_matchup
    bundle = _minimal_bundle()  # no tier_v2 / tier_gap / arsenal
    text = "\n".join(_render_pitcher_matchup(bundle))
    assert "## 投手對決" in text
    assert "Tier (xFIP-blend)" in text  # row still present
    assert "—" in text  # placeholder for missing data


def test_render_file_index():
    from dossier_renderer import _render_file_index
    lines = _render_file_index({}, game_dir="analysis-data/2026-04-28/TB@CLE")
    text = "\n".join(lines)
    assert "## File 索引" in text
    assert "merged.json" in text
    assert "summary.md" in text
    assert "analysis-data/2026-04-28/TB@CLE/" in text


def test_render_file_index_empty_game_dir():
    from dossier_renderer import _render_file_index
    lines = _render_file_index({}, game_dir="")
    text = "\n".join(lines)
    assert "merged.json" in text


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_render_dossier_full_output_within_250_lines():
    """Spec §4.2: dossier ≤ 250 lines."""
    from dossier_renderer import render_dossier
    bundle = _minimal_bundle()
    output = render_dossier(bundle, game_dir="analysis-data/2026-04-28/TB@CLE")
    lines = output.split("\n")
    assert len(lines) <= 250, f"Dossier exceeded 250 lines: {len(lines)} lines"


def test_render_dossier_required_sections_present():
    """All required H2 markers must be in output."""
    from dossier_renderer import render_dossier
    bundle = _minimal_bundle()
    output = render_dossier(bundle)
    required = [
        "## 比賽資訊",
        "## 戰績速查",
        "## 系列脈絡",
        "## 投手對決",
        "## 打線",
        "## 牛棚 / Park",
        "## ⚠️ 風險提示摘要",
        "## File 索引",
    ]
    for section in required:
        assert section in output, f"Missing section: {section}"


def test_render_dossier_no_yoy_section_after_redesign():
    """Spec §4.2: 'YoY 對比' must NOT appear after redesign."""
    from dossier_renderer import render_dossier
    bundle = _minimal_bundle()
    output = render_dossier(bundle)
    assert "YoY 對比" not in output


def test_render_dossier_returns_string():
    """render_dossier must return a string."""
    from dossier_renderer import render_dossier
    bundle = _minimal_bundle()
    result = render_dossier(bundle)
    assert isinstance(result, str)
    assert len(result) > 100


def test_render_dossier_missing_bundle_keys_no_crash():
    """Gracefully handle completely empty bundle."""
    from dossier_renderer import render_dossier
    result = render_dossier({})
    assert isinstance(result, str)
    assert "## File 索引" in result


def test_render_dossier_game_dir_in_file_index():
    """game_dir should appear in File 索引."""
    from dossier_renderer import render_dossier
    bundle = _minimal_bundle()
    result = render_dossier(bundle, game_dir="analysis-data/2026-04-28/TB@CLE")
    assert "analysis-data/2026-04-28/TB@CLE/" in result


def test_render_series_context_handles_none_winner_and_empty_names():
    """C1 regression: live API may return winner=None or empty team names — must not crash"""
    from dossier_renderer import _render_series_context
    bundle = {
        "merged": {
            "_meta": {
                "home_team": "",    # empty home name
                "away_team": "",    # empty away name
            }
        },
        "game_data": {
            "series_prev": {
                "winner": None,     # None winner — not missing
                "date": "2026-04-27",
                "home_score": 2,
                "away_score": 1,
            },
            "home_recent": {
                "streak": 1,
                "games": [
                    {"opponent": "", "result": "W", "is_winner": True, "date": "2026-04-27"},
                ],
            },
            "away_recent": {
                "streak": -1,
                "games": [
                    {"opponent": "", "result": "L", "is_winner": False, "date": "2026-04-27"},
                ],
            },
        },
    }
    # Must not raise
    lines = _render_series_context(bundle)
    assert isinstance(lines, list)


# ---------------------------------------------------------------------------
# P9 tests: lineup source label + 9 棒 vs 對方先發
# ---------------------------------------------------------------------------

def _make_lineup(source="projected", batters=None):
    if batters is None:
        batters = [
            {"mlbam_id": 100 + i, "name": f"P{i}", "position": "DH",
             "pa": 200 - i * 10, "avg": 0.250, "obp": 0.330, "slg": 0.420,
             "ops": 0.750, "iso": 0.170, "babip": 0.300, "k_pct": 22.0, "bb_pct": 9.0,
             "xwoba": 0.330, "xba": 0.250, "xslg": 0.420,
             "ev95pct": 50.0, "barrel_pct": 8.0,
             "platoon": None, "last_7": None, "bvp": None,
             "batting_order": (i + 1) if source == "official" else None}
            for i in range(9)
        ]
    return {
        "team": "NYY", "team_id": 147, "tier": "🟡 Average",
        "avg_ops": 0.750, "avg_xwoba": 0.330, "avg_babip": 0.300,
        "avg_k_pct": 22.0, "avg_bb_pct": 9.0, "over_under_lean": 0,
        "recent_heat": "⚖️ Normal", "last7_babip": 0.300, "chain": {},
        "lineup_source": source, "lineup_source_detail": None, "lineup": batters,
    }


def test_dossier_lineup_section_official():
    """home/away 都 official → 標題出現「打線來源：🟢 official」、9 棒 vs 對方先發 table。"""
    from dossier_renderer import render_dossier
    bundle = {
        "game_data": {"game": {
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "HP",
                     "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "AP",
                     "probable_pitcher_id": 2},
            "venue": "Yankee Stadium",
            "officialDate": "2026-04-30",
            "date": "2026-04-30T23:00:00Z",
        }},
        "home_lineup": _make_lineup("official"),
        "away_lineup": _make_lineup("official"),
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "season": {}},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "season": {}},
        "merged": {"park_factor": 100, "home_bullpen_era": 4.0, "away_bullpen_era": 4.0},
    }
    md = render_dossier(bundle, game_dir="/tmp", summary_filename="summary.md")
    assert "🟢 official" in md
    # 9 棒 vs 對方先發應出現
    assert "9 棒 vs" in md or "1-9 棒 vs" in md or "All 9 vs" in md  # 依實作命名


def test_dossier_lineup_section_projected():
    """home/away 都 projected → 標題「🟡 projected」、Top 5 sub-block 維持。"""
    from dossier_renderer import render_dossier
    bundle = {
        "game_data": {"game": {
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "HP",
                     "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "AP",
                     "probable_pitcher_id": 2},
            "venue": "Yankee Stadium",
            "officialDate": "2026-04-30",
            "date": "2026-04-30T23:00:00Z",
        }},
        "home_lineup": _make_lineup("projected"),
        "away_lineup": _make_lineup("projected"),
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "season": {}},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "season": {}},
        "merged": {"park_factor": 100, "home_bullpen_era": 4.0, "away_bullpen_era": 4.0},
    }
    md = render_dossier(bundle, game_dir="/tmp", summary_filename="summary.md")
    assert "🟡 projected" in md
    # 既有 Top 5 sub-block 標題仍在
    assert "Top 5" in md or "PA top" in md or "對方先發" in md  # 依現行命名


def test_dossier_lineup_section_no_source_field():
    """缺 lineup_source（舊 merged.json） → 預設 projected，向下相容。"""
    from dossier_renderer import render_dossier
    home_l = _make_lineup("projected")
    home_l.pop("lineup_source")
    home_l.pop("lineup_source_detail")
    bundle = {
        "game_data": {"game": {
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "HP",
                     "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "AP",
                     "probable_pitcher_id": 2},
            "venue": "Yankee Stadium",
            "officialDate": "2026-04-30",
            "date": "2026-04-30T23:00:00Z",
        }},
        "home_lineup": home_l,
        "away_lineup": _make_lineup("projected"),
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "season": {}},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "season": {}},
        "merged": {"park_factor": 100, "home_bullpen_era": 4.0, "away_bullpen_era": 4.0},
    }
    # 不該 raise KeyError
    md = render_dossier(bundle, game_dir="/tmp", summary_filename="summary.md")
    assert "🟡 projected" in md


def test_dossier_lineup_section_mixed():
    """home=official + away=projected → 兩種 source label + 兩種 sub-block table 都出現。"""
    from dossier_renderer import render_dossier
    bundle = {
        "game_data": {"game": {
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "HP",
                     "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "AP",
                     "probable_pitcher_id": 2},
            "venue": "Yankee Stadium",
            "officialDate": "2026-04-30",
            "date": "2026-04-30T23:00:00Z",
        }},
        "home_lineup": _make_lineup("official"),
        "away_lineup": _make_lineup("projected"),
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "season": {}},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "season": {}},
        "merged": {"park_factor": 100, "home_bullpen_era": 4.0, "away_bullpen_era": 4.0},
    }
    md = render_dossier(bundle, game_dir="/tmp", summary_filename="summary.md")
    # 兩種 source label 都出現
    assert "🟢 official" in md
    assert "🟡 projected" in md
    # 兩種 sub-block 都渲染
    assert "1–9 棒" in md  # full-9 for HOME
    assert "Top 5" in md or "PA top" in md or "對方先發手感" in md  # top-5 for AWAY


# ---------------------------------------------------------------------------
# P10 tests: weather row (3 states)
# ---------------------------------------------------------------------------

def test_dossier_weather_row_present():
    """merged.weather 三欄齊 → dossier 出現 weather row。"""
    from dossier_renderer import render_dossier
    bundle = {
        "game_data": {"game": {
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "HP",
                     "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "AP",
                     "probable_pitcher_id": 2},
            "venue": "Yankee Stadium",
            "officialDate": "2026-04-30",
            "date": "2026-04-30T23:00:00Z",
        }},
        "home_lineup": _make_lineup("projected"),
        "away_lineup": _make_lineup("projected"),
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "season": {}},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "season": {}},
        "merged": {
            "park_factor": 100, "home_bullpen_era": 4.0, "away_bullpen_era": 4.0,
            "weather": {"condition": "Sunny", "temp_f": 78,
                        "wind_text": "10 mph, Out To CF", "indoor": False},
        },
    }
    md = render_dossier(bundle, game_dir="/tmp", summary_filename="summary.md")
    assert "Sunny" in md and "78°F" in md and "Out To CF" in md


def test_dossier_weather_row_indoor():
    """indoor=True → 顯示「室內（Roof Closed，不適用天氣分析）」。"""
    from dossier_renderer import render_dossier
    bundle = {
        "game_data": {"game": {
            "home": {"team": "TOR", "team_id": 141, "probable_pitcher": "HP",
                     "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "AP",
                     "probable_pitcher_id": 2},
            "venue": "Rogers Centre",
            "officialDate": "2026-04-30",
            "date": "2026-04-30T23:00:00Z",
        }},
        "home_lineup": _make_lineup("projected"),
        "away_lineup": _make_lineup("projected"),
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "season": {}},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "season": {}},
        "merged": {
            "park_factor": 100, "home_bullpen_era": 4.0, "away_bullpen_era": 4.0,
            "weather": {"condition": "Roof Closed", "temp_f": 72,
                        "wind_text": None, "indoor": True},
        },
    }
    md = render_dossier(bundle, game_dir="/tmp", summary_filename="summary.md")
    assert "室內" in md
    assert "不適用天氣分析" in md


def test_dossier_weather_row_absent():
    """merged.weather=None → 整行省略（dossier 不應有 'weather:' / '室內' / '未公布' 字樣）。"""
    from dossier_renderer import render_dossier
    bundle = {
        "game_data": {"game": {
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "HP",
                     "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "AP",
                     "probable_pitcher_id": 2},
            "venue": "Yankee Stadium",
            "officialDate": "2026-04-30",
            "date": "2026-04-30T23:00:00Z",
        }},
        "home_lineup": _make_lineup("projected"),
        "away_lineup": _make_lineup("projected"),
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "season": {}},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "season": {}},
        "merged": {
            "park_factor": 100, "home_bullpen_era": 4.0, "away_bullpen_era": 4.0,
            "weather": None,
        },
    }
    md = render_dossier(bundle, game_dir="/tmp", summary_filename="summary.md")
    # weather 整行不應出現
    assert "**weather**:" not in md
    assert "室內" not in md
    # 不應出現 weather 脈絡的「未公布」佔位符
    assert "**weather**: 未公布" not in md
