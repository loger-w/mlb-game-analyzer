"""Smoke tests for format_md() in pitcher_stats / lineup_analyzer / roster_checker / merge_game_data.

These verify the MD output:
- Has `# Title` header
- Includes/excludes `## 🚨 Triggers` based on data
- Includes `## Source` footer
- Returns valid string with reasonable length
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# pitcher_stats.format_md
# ---------------------------------------------------------------------------

def _pitcher_lugo_data():
    """Lugo 真實 trigger 案例：ERA 1.15 / xERA 3.96 / gap -2.81。"""
    return {
        "name": "Seth Lugo",
        "mlbam_id": 607625,
        "age": 36,
        "birth_date": "1989-11-17",
        "pitch_hand": "R",
        "age_assessment": "📉📉 明顯退化",
        "tier": "🟠 Strong Ace",
        "season": {
            "era": 1.15, "whip": 0.93, "k_pct": 24.1, "bb_pct": 8.6, "k_bb_pct": 15.5,
            "fip": 2.27, "xfip": 3.51, "hr_per_9": 0.0, "gb_pct": 49.2,
            "ip": 31.33, "games": 5, "gs": 5,
        },
        "expected": {"xera": 3.96, "xwoba": 0.316, "xba": 0.249},
        "statcast": {
            "avg_velo": 86.3, "max_velo": 95.3, "hard_hit_pct": 25.0,
            "pitch_types": {"SI": 21.5, "FF": 18.2, "CU": 15.0, "FC": 14.3, "SL": 11.3},
            "barrel_pct": 9.0, "ev95percent": 39.7, "whiff_pct": 8.9, "csw_pct": 27.5,
        },
        "game_log": [
            {"date": "2026-03-29", "opponent": "Atlanta Braves", "ip": 6.33, "era": 0.0,
             "k": 3, "bb": 0, "h": 5, "er": 0, "pitches": 77, "strikes": 55},
        ],
        "platoon_splits": {
            "vs_left": {"avg": ".212", "obp": ".307", "slg": ".288", "k": 20, "bb": 9, "bf": 75, "k_pct": 26.7, "bb_pct": 12.0},
            "vs_right": {"avg": ".128", "obp": ".146", "slg": ".179", "k": 8, "bb": 1, "bf": 41, "k_pct": 19.5, "bb_pct": 2.4},
        },
        "prior_year": {
            "era": 4.15, "fip": 4.93, "k_pct": 20.5, "bb_pct": 9.0,
            "ip": 145.33, "hr_per_9": 1.67, "gb_pct": 43.5,
        },
    }


def _pitcher_detmers_data():
    """Detmers：ERA 4.08 / xERA 2.75 / gap +1.33（< 1.5，不觸發 Flag 13）。"""
    return {
        "name": "Reid Detmers",
        "mlbam_id": 672282,
        "age": 26,
        "pitch_hand": "L",
        "tier": "🟡 Solid Starter",
        "age_assessment": "⚡ 巔峰期",
        "season": {
            "era": 4.08, "whip": 1.08, "k_pct": 25.8, "bb_pct": 6.7,
            "fip": 2.68, "xfip": 3.13, "ip": 28.67, "games": 5, "gs": 5,
        },
        "expected": {"xera": 2.75},
        "statcast": {
            "avg_velo": 88.0, "whiff_pct": 12.6, "hard_hit_pct": 19.9,
            "pitch_types": {"FF": 41.4, "SL": 34.6},
        },
        "game_log": [],
        "platoon_splits": {},
        "prior_year": {"era": 3.96},
    }


def test_pitcher_md_has_required_sections():
    from pitcher_stats import format_md
    md = format_md(_pitcher_lugo_data())
    assert md.startswith("# Pitcher — Seth Lugo")
    assert "## Season Stats" in md
    assert "## Source" in md
    # 應包含基本識別資訊
    assert "MLBAM 607625" in md
    assert "🟠 Strong Ace" in md


def test_pitcher_md_lugo_includes_flag13_trigger():
    """Lugo MD 必須含 🚨 Triggers + Flag 13。"""
    from pitcher_stats import format_md
    md = format_md(_pitcher_lugo_data())
    assert "## 🚨 Triggers" in md
    assert "Flag 13" in md
    assert "ERA-xERA gap" in md


def test_pitcher_md_detmers_no_triggers_section():
    """Detmers gap=1.33 不觸發，MD 完全沒有 Triggers section。"""
    from pitcher_stats import format_md
    md = format_md(_pitcher_detmers_data())
    assert "## 🚨 Triggers" not in md


def test_pitcher_md_includes_pitch_mix_when_present():
    from pitcher_stats import format_md
    md = format_md(_pitcher_lugo_data())
    assert "## Pitch Mix" in md
    assert "SI" in md
    assert "21.5" in md  # SI 21.5%


def test_pitcher_md_includes_platoon_when_present():
    from pitcher_stats import format_md
    md = format_md(_pitcher_lugo_data())
    assert "## Platoon Splits" in md
    assert "vs LHB" in md
    assert "vs RHB" in md


def test_pitcher_md_includes_prior_year():
    from pitcher_stats import format_md
    md = format_md(_pitcher_lugo_data())
    assert "## Prior Year" in md
    assert "4.15" in md  # prior ERA


def test_pitcher_md_handles_error_in_season():
    """season 有 error 不該 crash，MD 仍可生成。"""
    from pitcher_stats import format_md
    data = {
        "name": "Test",
        "mlbam_id": 0,
        "season": {"error": "no data"},
        "expected": {},
        "statcast": {},
    }
    md = format_md(data)
    assert "# Pitcher — Test" in md
    assert "Season stats error" in md


# ---------------------------------------------------------------------------
# lineup_analyzer.format_md
# ---------------------------------------------------------------------------

def _lineup_kc_data():
    """KC vs Detmers 真實案例。"""
    return {
        "team": "KC",
        "team_id": 118,
        "tier": "🟡 Average",
        "avg_ops": 0.686,
        "avg_xwoba": 0.319,
        "avg_babip": 0.297,
        "avg_k_pct": 23.4,
        "avg_bb_pct": 10.3,
        "over_under_lean": 0,
        "recent_heat": "⚖️ Normal",
        "last7_babip": 0.326,  # < 0.370，不觸發 Flag 3
        "chain": {"obp_top3": 0.321, "slg_mid": 0.435},
        "lineup": [
            {
                "name": "Bobby Witt Jr.", "position": "SS", "pa": 100,
                "avg": 0.250, "obp": 0.300, "slg": 0.442, "ops": 0.742,
                "xwoba": 0.373, "babip": 0.310, "k_pct": 22.0, "bb_pct": 7.0,
                "ev95pct": 45.0, "barrel_pct": 12.5,
                "platoon": {"vs_lhp": {"pa": 25, "ops": "0.800"}, "vs_rhp": {"pa": 75, "ops": "0.720"}},
                "last_7": {"pa": 30, "avg": "0.280", "obp": "0.350", "slg": "0.480", "ops": "0.830", "babip": "0.330"},
                "bvp": {"pa": 3, "avg": ".333", "obp": ".333", "slg": "1.333", "h": 1, "hr": 1, "so": 0, "bb": 0, "sample_sufficient": False},
            },
        ],
    }


def _lineup_extreme_babip_data():
    """虛構 last7 BABIP=.380 觸發 Flag 3。"""
    data = _lineup_kc_data()
    data["last7_babip"] = 0.380
    return data


def test_lineup_md_has_required_sections():
    from lineup_analyzer import format_md
    md = format_md(_lineup_kc_data())
    assert md.startswith("# Lineup Analysis")
    assert "## Team-Level Aggregates" in md
    assert "## Core Lineup" in md
    assert "## Source" in md


def test_lineup_md_kc_no_trigger():
    """KC last7 BABIP .326 不觸發 Flag 3 → 無 Triggers section。"""
    from lineup_analyzer import format_md
    md = format_md(_lineup_kc_data())
    assert "## 🚨 Triggers" not in md


def test_lineup_md_extreme_babip_includes_trigger():
    from lineup_analyzer import format_md
    md = format_md(_lineup_extreme_babip_data())
    assert "## 🚨 Triggers" in md
    assert "Flag 3" in md


def test_lineup_md_includes_lineup_table():
    from lineup_analyzer import format_md
    md = format_md(_lineup_kc_data())
    assert "Bobby Witt Jr." in md


def test_lineup_md_includes_platoon_when_present():
    from lineup_analyzer import format_md
    md = format_md(_lineup_kc_data())
    assert "## Platoon Splits" in md


def test_lineup_md_includes_bvp_with_sample_marker():
    from lineup_analyzer import format_md
    md = format_md(_lineup_kc_data())
    assert "## BvP" in md
    # PA<15 應顯示為 ❌（PA<15）
    assert "PA<15" in md


def test_lineup_md_handles_error_data():
    from lineup_analyzer import format_md
    md = format_md({"error": "no roster"})
    assert "ERROR" in md


# ---------------------------------------------------------------------------
# roster_checker.format_md
# ---------------------------------------------------------------------------

def _roster_kc_data():
    return {
        "team": "Kansas City Royals",
        "team_id": 118,
        "active_roster": {
            "pitchers": ["Seth Lugo", "Cole Ragans"],
            "position_players": ["Bobby Witt Jr.", "Vinnie Pasquantino"],
        },
        "injured_list": [
            {"name": "Carlos Estévez", "status": "Injured 15-Day", "position": "Pitcher"},
            {"name": "Jonathan India", "status": "Injured 10-Day", "position": "Second Base"},
        ],
        "not_active_40man": ["Eric Cerantola", "John Rave"],
        "summary": {
            "total_active": 4,
            "total_active_pitchers": 2,
            "total_active_position": 2,
            "total_il": 2,
            "total_40man_not_active": 2,
        },
    }


def test_roster_md_has_required_sections():
    from roster_checker import format_md
    md = format_md(_roster_kc_data())
    assert "# Roster Check" in md
    assert "## Active Pitchers" in md
    assert "## Active Position Players" in md
    assert "## Injured List" in md
    assert "## Source" in md


def test_roster_md_separates_pitcher_il_from_position_il():
    from roster_checker import format_md
    md = format_md(_roster_kc_data())
    assert "### Pitchers on IL" in md
    assert "### Position Players on IL" in md


def test_roster_md_no_trigger_when_starter_in_active():
    from roster_checker import format_md
    md = format_md(_roster_kc_data(), expected_starter="Seth Lugo")
    assert "## 🚨 Triggers" not in md


def test_roster_md_trigger_when_starter_not_active():
    from roster_checker import format_md
    md = format_md(_roster_kc_data(), expected_starter="Phantom Pitcher")
    assert "## 🚨 Triggers" in md
    assert "STARTER_NOT_ACTIVE" in md


def test_roster_md_handles_no_il():
    from roster_checker import format_md
    data = _roster_kc_data()
    data["injured_list"] = []
    md = format_md(data)
    assert "## Injured List" in md


# ---------------------------------------------------------------------------
# merge_game_data.format_md
# ---------------------------------------------------------------------------

def _merged_data():
    return {
        "_meta": {
            "home_team": "Kansas City Royals",
            "away_team": "Los Angeles Angels",
            "home_sp": "Seth Lugo",
            "away_sp": "Reid Detmers",
            "home_sp_starts": 5,
            "away_sp_starts": 5,
            "venue": "Kauffman Stadium",
            "game_pk": 824122,
            "game_date": "2026-04-26T23:20:00Z",
        },
        "home_starter_fip": 2.27, "away_starter_fip": 2.68,
        "home_starter_k_bb": 15.5, "away_starter_k_bb": 19.1,
        "home_starter_whip": 0.93, "away_starter_whip": 1.08,
        "home_batting_xwoba": 0.319, "away_batting_xwoba": 0.322,
        "home_batting_ops": 0.686, "away_batting_ops": 0.720,
        "home_batting_k_pct": 23.4, "away_batting_k_pct": 25.7,
        "home_pitcher": {"era": 1.15, "xera": 3.96, "ip": 31.33, "era_xera_delta": 2.81, "prior_year": {"era": 4.15}},
        "away_pitcher": {"era": 4.08, "xera": 2.75, "ip": 28.67, "era_xera_delta": 1.33, "prior_year": {"era": 3.96}},
        "home_lineup": {"recent_babip": 0.326},
        "away_lineup": {"recent_babip": 0.291},
        "home_bullpen_era": 4.50, "away_bullpen_era": 4.20,
        "park_factor": 95.0,
        "home_recent_rs": 5.1, "home_recent_ra": 6.0,
        "away_recent_rs": 4.0, "away_recent_ra": 4.5,
        "home_recent_30_rs": 3.79, "home_recent_30_ra": 4.54,
        "away_recent_30_rs": 4.64, "away_recent_30_ra": 4.79,
        "home_season_rs": 3.79, "home_season_ra": 4.54,
        "away_season_rs": 4.64, "away_season_ra": 4.79,
        "home_season_games": 28, "away_season_games": 28,
    }


def test_merged_md_has_required_sections():
    from merge_game_data import format_md
    md = format_md(_merged_data())
    assert md.startswith("# Merged Phase 2 — Los Angeles Angels @ Kansas City Royals")
    assert "## Starting Pitchers" in md
    assert "## Lineups" in md
    assert "## Bullpen / Park" in md
    assert "## Multi-Window RS / RA" in md
    assert "## Source" in md


def test_merged_md_includes_flag13_summary_for_lugo():
    """home_pitcher era_xera_delta=2.81 ≥ 1.5 → Phase 2 整合視角顯示 Flag 13。"""
    from merge_game_data import format_md
    md = format_md(_merged_data())
    assert "## 🚨 Triggers" in md
    assert "Kansas City Royals 投手 Flag 13" in md


def test_merged_md_no_flag3_in_normal_babip():
    """home_lineup recent_babip 0.326 / away 0.291 均在區間內 → 無 Flag 3。"""
    from merge_game_data import format_md
    md = format_md(_merged_data())
    assert "Flag 3" not in md


def test_merged_md_flag3_when_babip_extreme():
    from merge_game_data import format_md
    data = _merged_data()
    data["home_lineup"]["recent_babip"] = 0.380
    md = format_md(data)
    assert "Flag 3" in md


def test_merged_md_handles_missing_meta():
    from merge_game_data import format_md
    data = {"_meta": {}}
    md = format_md(data)
    assert "# Merged Phase 2" in md
    assert "## Source" in md
