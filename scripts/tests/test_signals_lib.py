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
    for k in ("name", "fired", "value", "severity", "label", "details", "confidence"):
        assert k in s, f"signal missing key: {k}"
    assert s["severity"] in {"low", "medium", "high"}, f"bad severity: {s['severity']}"
    assert s["confidence"] in {"data", "heuristic", "small_sample"}


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
