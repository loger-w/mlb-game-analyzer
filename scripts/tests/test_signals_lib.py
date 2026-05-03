"""Tests for signals_lib — derived signals (PR-3 commits 10-12).

Each signal is a pure function returning the standard signal dict:
    {
        "name": str,
        "fired": bool,
        "value": float | None,
        "severity": "low" | "medium" | "high",
        "label": str,
        "details": dict,
        "confidence": "data" | "heuristic" | "small_sample",
    }

Signals batch 1 (commit 10):
    signal_tier_mismatch(tier_gap)
    signal_heat_vs_babip(heat, last7_babip)
    signal_platoon_advantage(core_lineup, pitcher_hand)
    signal_strong_park(park_factor)
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _signal_contract(s: dict):
    """Assert every signal returns the canonical schema."""
    for k in ("name", "fired", "value", "severity", "label", "details", "confidence", "half_life"):
        assert k in s, f"signal missing key: {k}"
    assert s["severity"] in {"low", "medium", "high"}, f"bad severity: {s['severity']}"
    assert s["confidence"] in {"data", "heuristic", "small_sample"}
    assert s["half_life"] in {"structural", "medium", "short"}, f"bad half_life: {s['half_life']}"


# ---------------------------------------------------------------------------
# signal_tier_mismatch — pitcher tier_v2 vs ERA-only gap
# ---------------------------------------------------------------------------

def test_tier_mismatch_fires_when_gap_above_15():
    """gap = +20 (v2 says better than ERA suggests) → fires high severity."""
    from signals_lib import signal_tier_mismatch
    s = signal_tier_mismatch({"expected_score": 90.0, "era_only_score": 70, "gap": 20.0})
    _signal_contract(s)
    assert s["fired"] is True
    assert s["value"] == 20.0
    assert s["severity"] == "high"
    assert "ERA 低估" in s["label"]


def test_tier_mismatch_fires_when_gap_below_minus15():
    """gap = -25 (ERA flatters) → fires high severity."""
    from signals_lib import signal_tier_mismatch
    s = signal_tier_mismatch({"expected_score": 50.0, "era_only_score": 75, "gap": -25.0})
    assert s["fired"] is True
    assert "ERA 高估" in s["label"]
    assert s["severity"] == "high"


def test_tier_mismatch_medium_severity_at_15_to_30():
    from signals_lib import signal_tier_mismatch
    s = signal_tier_mismatch({"expected_score": 80.0, "era_only_score": 70, "gap": 16.0})
    assert s["fired"] is True
    assert s["severity"] == "medium"


def test_tier_mismatch_does_not_fire_when_gap_under_threshold():
    from signals_lib import signal_tier_mismatch
    s = signal_tier_mismatch({"expected_score": 75.0, "era_only_score": 70, "gap": 5.0})
    assert s["fired"] is False


def test_tier_mismatch_does_not_fire_when_gap_none():
    from signals_lib import signal_tier_mismatch
    s = signal_tier_mismatch({"expected_score": None, "era_only_score": 75, "gap": None})
    assert s["fired"] is False
    assert s["confidence"] == "small_sample"


# ---------------------------------------------------------------------------
# signal_heat_vs_babip — lineup heat × last7 BABIP cross-check
# ---------------------------------------------------------------------------

def test_heat_vs_babip_fires_lucky_hot():
    """🔥 Hot + last7 BABIP ≥ .350 → lucky-hot."""
    from signals_lib import signal_heat_vs_babip
    s = signal_heat_vs_babip(heat="🔥 Hot", last7_babip=0.380)
    _signal_contract(s)
    assert s["fired"] is True
    assert "lucky-hot" in s["label"].lower() or "運氣" in s["label"]
    assert s["value"] == 0.380


def test_heat_vs_babip_fires_unlucky_cold():
    """🥶 Cold + last7 BABIP ≤ .270 → unlucky-cold (regression up)."""
    from signals_lib import signal_heat_vs_babip
    s = signal_heat_vs_babip(heat="🥶 Cold", last7_babip=0.250)
    assert s["fired"] is True
    assert "unlucky" in s["label"].lower() or "反彈" in s["label"]


def test_heat_vs_babip_does_not_fire_normal_heat():
    """⚖️ Normal heat → no signal regardless of BABIP."""
    from signals_lib import signal_heat_vs_babip
    s = signal_heat_vs_babip(heat="⚖️ Normal", last7_babip=0.380)
    assert s["fired"] is False


def test_heat_vs_babip_does_not_fire_when_babip_in_normal_range():
    """🔥 Hot + BABIP 0.310 (normal) → real heat, not lucky → no signal."""
    from signals_lib import signal_heat_vs_babip
    s = signal_heat_vs_babip(heat="🔥 Hot", last7_babip=0.310)
    assert s["fired"] is False


def test_heat_vs_babip_handles_missing_data():
    from signals_lib import signal_heat_vs_babip
    assert signal_heat_vs_babip(heat=None, last7_babip=None)["fired"] is False
    assert signal_heat_vs_babip(heat="🔥 Hot", last7_babip=None)["fired"] is False


# ---------------------------------------------------------------------------
# signal_platoon_advantage — top-5 batters with vs-hand OPS uplift
# ---------------------------------------------------------------------------

def _batter(season_ops, vs_lhp_ops=None, vs_rhp_ops=None):
    platoon = {}
    if vs_lhp_ops is not None:
        platoon["vs_lhp"] = {"ops": str(vs_lhp_ops)}
    if vs_rhp_ops is not None:
        platoon["vs_rhp"] = {"ops": str(vs_rhp_ops)}
    return {"ops": season_ops, "platoon": platoon}


def test_platoon_advantage_fires_when_4_of_5_uplifted():
    """4 batters in top 5 have vs-LHP OPS ≥ season OPS + 0.050 → fire."""
    from signals_lib import signal_platoon_advantage
    lineup = [
        _batter(season_ops=0.700, vs_lhp_ops=0.800),  # +0.100 uplift
        _batter(season_ops=0.750, vs_lhp_ops=0.850),  # +0.100
        _batter(season_ops=0.700, vs_lhp_ops=0.760),  # +0.060
        _batter(season_ops=0.700, vs_lhp_ops=0.760),  # +0.060
        _batter(season_ops=0.700, vs_lhp_ops=0.700),  # 0
    ]
    s = signal_platoon_advantage(lineup, pitcher_hand="L")
    _signal_contract(s)
    assert s["fired"] is True
    assert s["value"] == 4
    assert "platoon" in s["label"].lower() or "手別" in s["label"]


def test_platoon_advantage_does_not_fire_when_uplifts_only_2():
    from signals_lib import signal_platoon_advantage
    lineup = [
        _batter(season_ops=0.700, vs_lhp_ops=0.800),  # +0.100
        _batter(season_ops=0.700, vs_lhp_ops=0.760),  # +0.060
        _batter(season_ops=0.700, vs_lhp_ops=0.700),  # 0
        _batter(season_ops=0.700, vs_lhp_ops=0.700),  # 0
        _batter(season_ops=0.700, vs_lhp_ops=0.700),  # 0
    ]
    s = signal_platoon_advantage(lineup, pitcher_hand="L")
    assert s["fired"] is False


def test_platoon_advantage_handles_missing_platoon_data():
    """Batters without platoon entry → not counted as uplift (neither for nor against)."""
    from signals_lib import signal_platoon_advantage
    lineup = [{"ops": 0.700, "platoon": {}} for _ in range(5)]
    s = signal_platoon_advantage(lineup, pitcher_hand="L")
    assert s["fired"] is False


def test_platoon_advantage_empty_lineup_does_not_fire():
    from signals_lib import signal_platoon_advantage
    s = signal_platoon_advantage([], pitcher_hand="L")
    assert s["fired"] is False


# ---------------------------------------------------------------------------
# signal_strong_park — extreme park factor surfaces as signal
# ---------------------------------------------------------------------------

def test_strong_park_fires_high_pf():
    """Park factor 110 → hitter-friendly signal."""
    from signals_lib import signal_strong_park
    s = signal_strong_park(park_factor=112)
    _signal_contract(s)
    assert s["fired"] is True
    assert s["value"] == 112
    assert "打者友善" in s["label"] or "hitter" in s["label"].lower()


def test_strong_park_fires_low_pf():
    """Park factor 88 → pitcher-friendly signal."""
    from signals_lib import signal_strong_park
    s = signal_strong_park(park_factor=88)
    assert s["fired"] is True
    assert "投手友善" in s["label"] or "pitcher" in s["label"].lower()


def test_strong_park_does_not_fire_neutral():
    """PF 95-105 → neutral, no signal."""
    from signals_lib import signal_strong_park
    assert signal_strong_park(park_factor=98)["fired"] is False
    assert signal_strong_park(park_factor=102)["fired"] is False


def test_strong_park_handles_none():
    from signals_lib import signal_strong_park
    s = signal_strong_park(park_factor=None)
    assert s["fired"] is False


# ---------------------------------------------------------------------------
# signal_reverse_platoon — pitcher vs-LHB/vs-RHB OPS unexpectedly inverted
# ---------------------------------------------------------------------------

def test_reverse_platoon_rhp_fires_when_rhb_ops_higher():
    """RHP normal: LHB hits better. Reverse: vs RHB OPS > vs LHB OPS by ≥0.080.
    Bradish-style sweeper-heavy reverse."""
    from signals_lib import signal_reverse_platoon
    splits = {
        "vs_left": {"ops": ".545", "bf": 60},
        "vs_right": {"ops": ".932", "bf": 80},
    }
    s = signal_reverse_platoon(splits, pitcher_hand="R")
    _signal_contract(s)
    assert s["fired"] is True
    assert s["value"] > 0.30  # delta ≈ 0.387
    assert "reverse" in s["label"].lower() or "反向" in s["label"]


def test_reverse_platoon_lhp_fires_when_lhb_ops_higher():
    """LHP normal: RHB hits better. Reverse: vs LHB OPS > vs RHB OPS by ≥0.080."""
    from signals_lib import signal_reverse_platoon
    splits = {
        "vs_left": {"ops": ".880", "bf": 50},
        "vs_right": {"ops": ".620", "bf": 100},
    }
    s = signal_reverse_platoon(splits, pitcher_hand="L")
    assert s["fired"] is True


def test_reverse_platoon_normal_split_does_not_fire():
    """RHP with LHB > RHB → expected platoon, no fire."""
    from signals_lib import signal_reverse_platoon
    splits = {
        "vs_left": {"ops": ".750", "bf": 60},
        "vs_right": {"ops": ".600", "bf": 80},
    }
    s = signal_reverse_platoon(splits, pitcher_hand="R")
    assert s["fired"] is False


def test_reverse_platoon_below_threshold_does_not_fire():
    """RHB OPS only 0.050 above LHB → noise, no fire."""
    from signals_lib import signal_reverse_platoon
    splits = {
        "vs_left": {"ops": ".700", "bf": 60},
        "vs_right": {"ops": ".750", "bf": 80},
    }
    s = signal_reverse_platoon(splits, pitcher_hand="R")
    assert s["fired"] is False


def test_reverse_platoon_small_sample_marks_heuristic():
    """Both BF ≥ 30 but one side < 50 → confidence heuristic (still fires)."""
    from signals_lib import signal_reverse_platoon
    splits = {
        "vs_left": {"ops": ".545", "bf": 32},
        "vs_right": {"ops": ".900", "bf": 40},
    }
    s = signal_reverse_platoon(splits, pitcher_hand="R")
    assert s["fired"] is True
    assert s["confidence"] == "heuristic"


def test_reverse_platoon_too_small_sample_does_not_fire():
    """Either side BF < 30 → can't trust split → no fire."""
    from signals_lib import signal_reverse_platoon
    splits = {
        "vs_left": {"ops": ".545", "bf": 20},
        "vs_right": {"ops": ".900", "bf": 25},
    }
    s = signal_reverse_platoon(splits, pitcher_hand="R")
    assert s["fired"] is False


def test_reverse_platoon_falls_back_to_obp_plus_slg_when_ops_missing():
    """MLB API statSplits returns avg/obp/slg but NOT ops. Signal must fall
    back to obp+slg or it never fires for live data.

    Example: vs LHB .253/.353/.460 (ops≈.813); vs RHB .396/.473/.583
    (ops≈1.056). Δ ≈ +0.243 → reverse platoon should fire on RHP."""
    from signals_lib import signal_reverse_platoon
    splits = {
        "vs_left": {"obp": ".353", "slg": ".460", "bf": 102},   # NO 'ops' key
        "vs_right": {"obp": ".473", "slg": ".583", "bf": 55},   # NO 'ops' key
    }
    s = signal_reverse_platoon(splits, pitcher_hand="R")
    assert s["fired"] is True, "must fall back to obp+slg when ops missing"
    assert s["value"] == pytest.approx(0.243, abs=0.005)
    assert s["severity"] == "high"  # Δ ≥ 0.200


# ---------------------------------------------------------------------------
# signal_chain_break — largest adjacent OPS drop in batting order
# ---------------------------------------------------------------------------

def test_chain_break_fires_when_adjacent_drop_exceeds_0_150():
    """1-3 strong, 4 weak → chain breaks at #3-4."""
    from signals_lib import signal_chain_break
    lineup = [
        {"name": f"P{i}", "ops": ops, "batting_order": i + 1}
        for i, ops in enumerate([0.900, 0.850, 0.880, 0.500, 0.700, 0.600, 0.620, 0.580, 0.550])
    ]
    s = signal_chain_break(lineup)
    _signal_contract(s)
    assert s["fired"] is True
    assert "3" in s["label"] or "4" in s["label"]
    # value should be the drop magnitude
    assert s["value"] > 0.30


def test_chain_break_does_not_fire_when_lineup_smooth():
    """All OPS within 0.100 of neighbors → no chain break."""
    from signals_lib import signal_chain_break
    lineup = [
        {"name": f"P{i}", "ops": ops, "batting_order": i + 1}
        for i, ops in enumerate([0.800, 0.780, 0.760, 0.740, 0.720, 0.700, 0.680, 0.660, 0.640])
    ]
    s = signal_chain_break(lineup)
    assert s["fired"] is False


def test_chain_break_handles_short_lineup():
    """Lineup < 5 → not enough data → no fire."""
    from signals_lib import signal_chain_break
    lineup = [{"name": "P1", "ops": 0.900}, {"name": "P2", "ops": 0.500}]
    s = signal_chain_break(lineup)
    assert s["fired"] is False


def test_chain_break_uses_supplied_order_not_resort():
    """signal respects caller's order; doesn't re-sort by OPS."""
    from signals_lib import signal_chain_break
    # Hand-built order: 0.900 → 0.500 drop at position 0-1
    lineup = [
        {"name": "P1", "ops": 0.900},
        {"name": "P2", "ops": 0.500},
        {"name": "P3", "ops": 0.480},
        {"name": "P4", "ops": 0.460},
        {"name": "P5", "ops": 0.440},
        {"name": "P6", "ops": 0.420},
    ]
    s = signal_chain_break(lineup)
    assert s["fired"] is True
    # The drop is at position 1-2 (0.900 → 0.500 = 0.400)
    assert "1" in s["label"] or "2" in s["label"]


# ---------------------------------------------------------------------------
# signal_pitch_mix_concentration — max usage % (NOT HHI per plan)
# ---------------------------------------------------------------------------

def test_pitch_mix_concentration_fires_single_pitch_dependent():
    """Max usage 60% → single-pitch dependent."""
    from signals_lib import signal_pitch_mix_concentration
    pitch_types = {"FF": 60.0, "SL": 25.0, "CH": 15.0}
    s = signal_pitch_mix_concentration(pitch_types)
    _signal_contract(s)
    assert s["fired"] is True
    assert s["value"] == 60.0
    assert "single-pitch" in s["label"].lower() or "依賴" in s["label"]


def test_pitch_mix_concentration_fires_balanced():
    """Max usage 22% → balanced (4+ pitches each meaningful)."""
    from signals_lib import signal_pitch_mix_concentration
    pitch_types = {"FF": 22.0, "SL": 21.0, "CH": 20.0, "SI": 19.0, "CU": 18.0}
    s = signal_pitch_mix_concentration(pitch_types)
    assert s["fired"] is True
    assert "balanced" in s["label"].lower() or "均衡" in s["label"]


def test_pitch_mix_concentration_does_not_fire_in_normal_range():
    """Max 35% → typical 3-pitch mix, no fire."""
    from signals_lib import signal_pitch_mix_concentration
    pitch_types = {"FF": 35.0, "SL": 30.0, "CH": 35.0}
    s = signal_pitch_mix_concentration(pitch_types)
    assert s["fired"] is False


def test_pitch_mix_concentration_empty_does_not_fire():
    from signals_lib import signal_pitch_mix_concentration
    s = signal_pitch_mix_concentration({})
    assert s["fired"] is False


# ---------------------------------------------------------------------------
# signal_core_il_count — wraps merged.{side}_core_bullpen_il_count
# ---------------------------------------------------------------------------

def test_core_il_count_fires_at_1():
    """1 core IL → 🟠 medium severity."""
    from signals_lib import signal_core_il_count
    s = signal_core_il_count(count=1, side="HOME")
    _signal_contract(s)
    assert s["fired"] is True
    assert s["value"] == 1
    assert s["severity"] == "medium"


def test_core_il_count_label_does_not_prefix_side():
    """Dossier renderer prepends side from signal["side"] (already on the
    dict). Label must NOT also start with side or output reads
    'AWAY AWAY 牛棚 core IL ×1' double-prefix."""
    from signals_lib import signal_core_il_count
    for count in (1, 2, 3):
        s = signal_core_il_count(count=count, side="AWAY")
        assert not s["label"].startswith("HOME") and not s["label"].startswith("AWAY"), (
            f"label must not start with side; got: {s['label']!r}"
        )


def test_core_il_count_fires_high_at_2():
    """2 core IL → 🔴 high severity (matchup-factors §牛棚傷兵累計效應)."""
    from signals_lib import signal_core_il_count
    s = signal_core_il_count(count=2, side="HOME")
    assert s["severity"] == "high"


def test_core_il_count_fires_extreme_at_3_plus():
    """3+ core IL → high severity, label notes extreme."""
    from signals_lib import signal_core_il_count
    s = signal_core_il_count(count=4, side="AWAY")
    assert s["severity"] == "high"
    assert "極高" in s["label"] or "extreme" in s["label"].lower() or "🔴🔴" in s["label"]


def test_core_il_count_does_not_fire_at_0():
    from signals_lib import signal_core_il_count
    s = signal_core_il_count(count=0, side="HOME")
    assert s["fired"] is False


def test_core_il_count_handles_none():
    from signals_lib import signal_core_il_count
    s = signal_core_il_count(count=None, side="HOME")
    assert s["fired"] is False
    assert s["confidence"] == "small_sample"


# ---------------------------------------------------------------------------
# compute_all_signals — aggregator over bundle (PR-3 commit 12)
# ---------------------------------------------------------------------------

def _minimal_bundle():
    """Compact bundle that exercises every signal path."""
    return {
        "home_pitcher": {
            "pitch_hand": "R",
            "tier_gap": {"expected_score": 80.0, "era_only_score": 60, "gap": 20.0},
            "platoon_splits": {
                "vs_left": {"ops": ".545", "bf": 60},
                "vs_right": {"ops": ".932", "bf": 80},
            },
            "statcast": {"pitch_types": {"SL": 50.0, "FF": 30.0, "CH": 20.0}},
        },
        "away_pitcher": {
            "pitch_hand": "L",
            "tier_gap": {"expected_score": 55.0, "era_only_score": 75, "gap": -20.0},
            "platoon_splits": {
                "vs_left": {"ops": ".700", "bf": 50},
                "vs_right": {"ops": ".750", "bf": 100},
            },
            "statcast": {"pitch_types": {"FF": 40.0, "SL": 30.0, "CH": 30.0}},
        },
        "home_lineup": {
            "recent_heat": "🔥 Hot",
            "last7_babip": 0.380,
            "lineup": [
                {"name": "P1", "ops": 0.900, "platoon": {"vs_lhp": {"ops": ".950"}}},
                {"name": "P2", "ops": 0.850, "platoon": {"vs_lhp": {"ops": ".920"}}},
                {"name": "P3", "ops": 0.800, "platoon": {"vs_lhp": {"ops": ".870"}}},
                {"name": "P4", "ops": 0.500, "platoon": {"vs_lhp": {"ops": ".600"}}},
                {"name": "P5", "ops": 0.700, "platoon": {"vs_lhp": {"ops": ".780"}}},
                {"name": "P6", "ops": 0.650},
                {"name": "P7", "ops": 0.620},
                {"name": "P8", "ops": 0.580},
                {"name": "P9", "ops": 0.560},
            ],
        },
        "away_lineup": {
            "recent_heat": "⚖️ Normal",
            "last7_babip": 0.300,
            "lineup": [{"name": f"AP{i}", "ops": 0.700} for i in range(9)],
        },
        "merged": {
            "park_factor": 112,
            "home_core_bullpen_il_count": 2,
            "away_core_bullpen_il_count": 0,
        },
    }


def test_compute_all_signals_returns_dict_with_signals_list_and_count():
    from signals_lib import compute_all_signals
    result = compute_all_signals(_minimal_bundle())
    assert "signals" in result
    assert "fired_count" in result
    assert isinstance(result["signals"], list)
    assert isinstance(result["fired_count"], int)


def test_compute_all_signals_attaches_side_to_each_signal():
    """Every signal dict has 'side' key set to HOME / AWAY / GAME."""
    from signals_lib import compute_all_signals
    result = compute_all_signals(_minimal_bundle())
    for s in result["signals"]:
        assert "side" in s
        assert s["side"] in ("HOME", "AWAY", "GAME")


def test_compute_all_signals_fires_expected_signals_for_minimal_bundle():
    """The minimal bundle is constructed to fire 6 signals."""
    from signals_lib import compute_all_signals
    result = compute_all_signals(_minimal_bundle())
    fired = [s for s in result["signals"] if s["fired"]]
    fired_names = {(s["name"], s["side"]) for s in fired}

    # Expected fires:
    assert ("tier_mismatch", "HOME") in fired_names           # gap +20
    assert ("tier_mismatch", "AWAY") in fired_names           # gap -20
    assert ("reverse_platoon", "HOME") in fired_names         # RHP RHB > LHB
    assert ("pitch_mix_concentration", "HOME") in fired_names # SL 50%
    assert ("heat_vs_babip", "HOME") in fired_names           # lucky-hot
    assert ("strong_park", "GAME") in fired_names             # PF 112
    assert ("core_il_count", "HOME") in fired_names           # 2 core IL


def test_compute_all_signals_handles_empty_bundle_gracefully():
    """Empty bundle → all signals return fired=False, no crash."""
    from signals_lib import compute_all_signals
    result = compute_all_signals({})
    assert result["fired_count"] == 0
    assert all(s["fired"] is False for s in result["signals"])


def test_compute_all_signals_fired_count_matches_list():
    """fired_count == len([s for s in signals if s.fired])."""
    from signals_lib import compute_all_signals
    result = compute_all_signals(_minimal_bundle())
    assert result["fired_count"] == sum(1 for s in result["signals"] if s["fired"])


# ---------------------------------------------------------------------------
# Signal staleness (half_life classification) — added by Item 4 refactor
# ---------------------------------------------------------------------------
#
# Each signal declares its data half_life class so analyst can discount
# short-window readings (對手會調整). See reference/matchup-factors.md §半衰期.
#   structural — multi-year / season-to-date aggregate (e.g. park, tier_mismatch)
#   medium     — season split, mid-season adjustable (platoon, chain, mix)
#   short      — last7 / daily window (heat, IL count, reverse_platoon)

def test_signal_strong_park_half_life_structural():
    """park factor 是多年物理特徵 → structural"""
    from signals_lib import signal_strong_park
    assert signal_strong_park(115.0)["half_life"] == "structural"
    # Even non-fired calls carry the classification for schema consistency
    assert signal_strong_park(100.0)["half_life"] == "structural"


def test_signal_tier_mismatch_half_life_structural():
    """tier_mismatch 是 season-to-date 累計，反身慢 → structural"""
    from signals_lib import signal_tier_mismatch
    assert signal_tier_mismatch({"expected_score": 90, "era_only_score": 70, "gap": 20})["half_life"] == "structural"


def test_signal_heat_vs_babip_half_life_short():
    """heat 是 last7 window，對手會立即調整 → short"""
    from signals_lib import signal_heat_vs_babip
    assert signal_heat_vs_babip("🔥 Hot", 0.380)["half_life"] == "short"


def test_signal_core_il_count_half_life_short():
    """IL 名單每天異動 → short"""
    from signals_lib import signal_core_il_count
    assert signal_core_il_count(2, "HOME")["half_life"] == "short"


def test_signal_chain_break_half_life_medium():
    """打線 OPS 結構是 season aggregate，但會被傷兵 / 換人改變 → medium"""
    from signals_lib import signal_chain_break
    lineup = [{"name": f"P{i}", "ops": 0.500 if i >= 5 else 0.800} for i in range(9)]
    assert signal_chain_break(lineup)["half_life"] == "medium"


def test_signal_platoon_advantage_half_life_medium():
    """打線 platoon 是 season split → medium"""
    from signals_lib import signal_platoon_advantage
    lineup = [{"ops": 0.700, "platoon": {"vs_rhp": {"ops": 0.760}}} for _ in range(5)]
    assert signal_platoon_advantage(lineup, "R")["half_life"] == "medium"


def test_signal_reverse_platoon_half_life_medium():
    """vs-LHB / vs-RHB 數據是 season split，對手知道後可換打者 → medium"""
    from signals_lib import signal_reverse_platoon
    splits = {
        "vs_left": {"ops": ".700", "bf": 60},
        "vs_right": {"ops": ".900", "bf": 60},
    }
    assert signal_reverse_platoon(splits, "R")["half_life"] == "medium"


def test_signal_pitch_mix_concentration_half_life_medium():
    """投手球種 mix 是 multi-month aggregate，但季中可調 → medium"""
    from signals_lib import signal_pitch_mix_concentration
    assert signal_pitch_mix_concentration({"FF": 50.0, "SL": 30.0, "CH": 20.0})["half_life"] == "medium"


# ---------------------------------------------------------------------------
# Cleanup #7 — signals_for_bundle: bundle-level cache for compute_all_signals
# ---------------------------------------------------------------------------


def test_signals_for_bundle_returns_cached_when_present():
    """When bundle["signals"] is already populated, helper returns the cached dict
    verbatim and does NOT recompute. Verified via a sentinel that compute_all_signals
    would never produce."""
    from signals_lib import signals_for_bundle

    sentinel = {"sentinel_cached": True, "signals": [], "fired_count": 0}
    bundle = _minimal_bundle()
    bundle["signals"] = sentinel

    result = signals_for_bundle(bundle)
    assert result is sentinel  # exact same object — cache, not recompute


def test_signals_for_bundle_computes_and_stores_when_missing():
    """When bundle has no "signals" key, helper computes via compute_all_signals,
    writes the result back into bundle["signals"], and a subsequent call hits cache."""
    from signals_lib import signals_for_bundle

    bundle = _minimal_bundle()
    assert "signals" not in bundle

    first = signals_for_bundle(bundle)
    # Compute happened and result was stored
    assert "signals" in bundle
    assert bundle["signals"] is first
    # Shape matches compute_all_signals output
    assert "signals" in first
    assert "fired_count" in first
    assert isinstance(first["signals"], list)

    # Second call returns the same object (cache hit, no recompute)
    second = signals_for_bundle(bundle)
    assert second is first


def test_signals_for_bundle_output_matches_compute_all_signals():
    """Regression guard: signals_for_bundle (cold cache) returns the exact same
    dict shape as compute_all_signals — no transformations or omissions."""
    from signals_lib import signals_for_bundle, compute_all_signals

    bundle1 = _minimal_bundle()
    bundle2 = _minimal_bundle()  # separate dict; ensures no cache cross-talk

    direct = compute_all_signals(bundle1)
    via_helper = signals_for_bundle(bundle2)

    assert sorted(direct.keys()) == sorted(via_helper.keys())
    assert direct["fired_count"] == via_helper["fired_count"]
    # Same set of fired signal names
    direct_names = {s["name"] for s in direct["signals"] if s.get("fired")}
    helper_names = {s["name"] for s in via_helper["signals"] if s.get("fired")}
    assert direct_names == helper_names
