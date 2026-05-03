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
