"""Tests for lineup_analyzer.compute_tier_vs_hand — vs-pitcher-hand re-aggregation.

Each batter's vs-this-hand OPS is preferred over season OPS for the matchup
tier. When platoon data is missing for a batter, fall back to season OPS.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _batter(name: str, season_ops: float, lhp_ops=None, rhp_ops=None) -> dict:
    """Build a minimal batter dict with platoon splits."""
    platoon = {}
    if lhp_ops is not None:
        platoon["vs_lhp"] = {"ops": str(lhp_ops), "pa": 50}
    if rhp_ops is not None:
        platoon["vs_rhp"] = {"ops": str(rhp_ops), "pa": 100}
    return {"name": name, "ops": season_ops, "platoon": platoon}


def test_tier_vs_hand_uses_platoon_ops_when_available():
    """All 9 batters have vs_lhp data → use platoon OPS, not season OPS."""
    from lineup_analyzer import compute_tier_vs_hand
    lineup = [_batter(f"P{i}", season_ops=0.700, lhp_ops=0.900) for i in range(9)]
    result = compute_tier_vs_hand(lineup, pitcher_hand="L")
    assert result["avg_ops"] == 0.900
    assert result["platoon_count"] == 9
    assert result["fallback_count"] == 0


def test_tier_vs_hand_falls_back_to_season_ops_when_platoon_missing():
    """Batter without platoon entry → use season OPS."""
    from lineup_analyzer import compute_tier_vs_hand
    lineup = [
        _batter("with-platoon", season_ops=0.700, lhp_ops=0.900),
        _batter("no-platoon-1", season_ops=0.800),  # no platoon split data
        _batter("no-platoon-2", season_ops=0.750),
    ]
    result = compute_tier_vs_hand(lineup, pitcher_hand="L")
    assert result["platoon_count"] == 1
    assert result["fallback_count"] == 2
    # avg = (0.900 + 0.800 + 0.750) / 3 = 0.8167
    assert abs(result["avg_ops"] - 0.817) < 0.002


def test_tier_vs_hand_handles_string_ops_from_api():
    """MLB API returns OPS as string (e.g. '.850'). Helper converts to float."""
    from lineup_analyzer import compute_tier_vs_hand
    lineup = [
        {"ops": 0.700, "platoon": {"vs_rhp": {"ops": ".850", "pa": 80}}},
        {"ops": 0.700, "platoon": {"vs_rhp": {"ops": ".750", "pa": 80}}},
    ]
    result = compute_tier_vs_hand(lineup, pitcher_hand="R")
    # avg = (0.850 + 0.750) / 2 = 0.800
    assert abs(result["avg_ops"] - 0.800) < 0.002


def test_tier_vs_hand_buckets_by_tier_map_ops():
    """Tier label must come from TIER_MAP_OPS bucket boundaries."""
    from lineup_analyzer import compute_tier_vs_hand
    # avg 0.870 → 🔴 Elite (≥ 0.830)
    elite = [_batter(f"P{i}", 0.700, lhp_ops=0.870) for i in range(5)]
    assert compute_tier_vs_hand(elite, "L")["tier"] == "🔴 Elite"
    # avg 0.770 → 🟠 Strong (≥ 0.760)
    strong = [_batter(f"P{i}", 0.700, lhp_ops=0.770) for i in range(5)]
    assert compute_tier_vs_hand(strong, "L")["tier"] == "🟠 Strong"
    # avg 0.720 → 🟡 Average (≥ 0.700)
    avg = [_batter(f"P{i}", 0.700, lhp_ops=0.720) for i in range(5)]
    assert compute_tier_vs_hand(avg, "L")["tier"] == "🟡 Average"
    # avg 0.650 → 🟢 Weak
    weak = [_batter(f"P{i}", 0.700, lhp_ops=0.650) for i in range(5)]
    assert compute_tier_vs_hand(weak, "L")["tier"] == "🟢 Weak"


def test_tier_vs_hand_separate_lhp_rhp_aggregates():
    """Same lineup queried twice (L vs R) → different aggregates from per-side data."""
    from lineup_analyzer import compute_tier_vs_hand
    lineup = [
        _batter("RHB-1", 0.700, lhp_ops=0.900, rhp_ops=0.700),
        _batter("RHB-2", 0.700, lhp_ops=0.900, rhp_ops=0.700),
    ]
    lhp = compute_tier_vs_hand(lineup, "L")
    rhp = compute_tier_vs_hand(lineup, "R")
    assert lhp["avg_ops"] == 0.900
    assert rhp["avg_ops"] == 0.700
    # Different tier buckets too
    assert lhp["tier"] != rhp["tier"]


def test_tier_vs_hand_empty_lineup_returns_weak_default():
    from lineup_analyzer import compute_tier_vs_hand
    result = compute_tier_vs_hand([], "L")
    assert result["tier"] == "🟢 Weak"
    assert result["avg_ops"] is None
    assert result["platoon_count"] == 0
    assert result["fallback_count"] == 0


def test_tier_vs_hand_invalid_ops_string_skipped():
    """OPS '.---' or other non-numeric strings → fall back to season OPS."""
    from lineup_analyzer import compute_tier_vs_hand
    lineup = [
        {"ops": 0.750, "platoon": {"vs_lhp": {"ops": ".---", "pa": 0}}},
        {"ops": 0.800, "platoon": {"vs_lhp": {"ops": "0.900", "pa": 50}}},
    ]
    result = compute_tier_vs_hand(lineup, "L")
    # First batter: invalid platoon → fallback to season 0.750
    # Second batter: valid platoon 0.900
    # avg = (0.750 + 0.900) / 2 = 0.825
    assert abs(result["avg_ops"] - 0.825) < 0.002
    assert result["platoon_count"] == 1
    assert result["fallback_count"] == 1
