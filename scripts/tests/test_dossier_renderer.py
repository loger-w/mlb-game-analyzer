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


def test_render_pitcher_matchup_flag13_triggers():
    """When ERA-xERA gap ≥ 1.5 for away pitcher, ⚠️ appears."""
    from dossier_renderer import _render_pitcher_matchup
    bundle = _minimal_bundle()
    # away pitcher already has era=2.1, xera=4.64 → gap=2.54 → Flag 13
    lines = _render_pitcher_matchup(bundle)
    text = "\n".join(lines)
    assert "⚠️" in text
    assert "Flag 13" in text


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
    assert "Top 5 vs" in text


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


def test_render_bullpen_park_structure():
    from dossier_renderer import _render_bullpen_park
    bundle = _minimal_bundle()
    lines = _render_bullpen_park(bundle)
    text = "\n".join(lines)
    assert "## 牛棚 / Park" in text
    assert "Bullpen ERA" in text
    assert "投手 IL 數" in text
    assert "Park Factor" in text
    assert "4.57" in text
    assert "5.18" in text


def test_render_bullpen_park_il_count():
    """IL pitcher count should appear correctly."""
    from dossier_renderer import _render_bullpen_park
    bundle = _minimal_bundle()
    bundle["home_roster"] = {
        "injured_list": [
            {"name": "P1", "status": "Injured 15-Day", "position": "Pitcher"},
            {"name": "P2", "status": "Injured 60-Day", "position": "Pitcher"},
        ]
    }
    bundle["away_roster"] = {
        "injured_list": [
            {"name": "P3", "status": "Injured 15-Day", "position": "Pitcher"},
        ]
    }
    lines = _render_bullpen_park(bundle)
    text = "\n".join(lines)
    # HOME has 2 pitchers on IL, AWAY has 1
    assert "| 2 |" in text or "2 (" in text
    assert "| 1 |" in text or "1 (" in text


def test_render_risk_summary_with_flags():
    """Risk summary should list all triggered flags."""
    from dossier_renderer import _render_risk_summary
    bundle = _minimal_bundle()
    # away_pitcher has Flag 13 (era=2.1, xera=4.64)
    # away_lineup has Flag 3 (last7_babip=0.241)
    lines = _render_risk_summary(bundle)
    text = "\n".join(lines)
    assert "## ⚠️ 風險提示摘要" in text
    assert "Flag 13" in text
    assert "Flag 3" in text


def test_render_risk_summary_no_flags():
    """No flags → '無風險提示'."""
    from dossier_renderer import _render_risk_summary
    bundle = _minimal_bundle()
    # Set ERA close to xERA for no Flag 13
    bundle["away_pitcher"]["season"]["era"] = 4.5
    bundle["away_pitcher"]["expected"]["xera"] = 4.6
    # Set BABIP to normal for no Flag 3
    bundle["away_lineup"]["last7_babip"] = 0.300
    bundle["home_lineup"]["last7_babip"] = 0.300
    lines = _render_risk_summary(bundle)
    text = "\n".join(lines)
    assert "無風險提示" in text


def test_render_file_index():
    from dossier_renderer import _render_file_index
    lines = _render_file_index({}, game_dir="analysis-data/2026-04-28/TB@CLE")
    text = "\n".join(lines)
    assert "## File 索引" in text
    assert "merged.json" in text
    assert "phase3_skeleton.md" in text
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


# ---------------------------------------------------------------------------
# Integration test with actual TB@CLE JSON fixture
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent.parent / "analysis-data" / "2026-04-28" / "TB@CLE"


def _load_fixture(name: str) -> dict:
    p = _DATA_DIR / name
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def test_render_dossier_with_actual_tb_cle_bundle():
    """Load real JSON files from analysis-data/2026-04-28/TB@CLE/ and verify output."""
    from dossier_renderer import render_dossier
    bundle = {
        "game_data": _load_fixture("game_data.json"),
        "home_roster": _load_fixture("home_roster.json"),
        "away_roster": _load_fixture("away_roster.json"),
        "home_pitcher": _load_fixture("home_pitcher.json"),
        "away_pitcher": _load_fixture("away_pitcher.json"),
        "home_lineup": _load_fixture("home_lineup.json"),
        "away_lineup": _load_fixture("away_lineup.json"),
        "merged": _load_fixture("merged.json"),
    }

    game_dir = str(_DATA_DIR)
    output = render_dossier(bundle, game_dir=game_dir)

    # Must contain key content from each section
    assert "## 比賽資訊" in output
    assert "Tanner Bibee" in output
    assert "Nick Martínez" in output or "Nick Martinez" in output
    assert "Progressive Field" in output

    assert "## 戰績速查" in output
    assert "4-6" in output   # home_recent record

    assert "## 系列脈絡" in output

    assert "## 投手對決" in output
    assert "4.45" in output   # home ERA
    assert "2.10" in output   # away ERA
    assert "Flag 13" in output  # away pitcher flag

    assert "## 打線" in output
    assert "Flag 3" in output   # away lineup BABIP flag
    assert "José Ramírez" in output

    assert "## 牛棚 / Park" in output
    assert "4.57" in output   # home bullpen ERA
    assert "5.18" in output   # away bullpen ERA

    assert "## ⚠️ 風險提示摘要" in output

    assert "## File 索引" in output
    assert "merged.json" in output

    # Line count constraint
    lines = output.split("\n")
    assert len(lines) <= 250, f"TB@CLE dossier exceeded 250 lines: {len(lines)}"

    # No YoY section
    assert "YoY 對比" not in output


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
