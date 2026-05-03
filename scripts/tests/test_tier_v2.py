"""Tests for lib_tier_v2 — blended pitcher tier formula.

The module exposes:
- compute_pct(value, percentile_dict, direction) -> 0..1 (1 = best)
- compute_age_factor(age) -> 0.7..1.0
- compute_tier_v2(season, statcast, age, baseline=None) -> dict
- score_to_tier(score) -> str (matches v1 emoji+label conventions)

All functions are pure (no I/O) so tests inject baseline data inline.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _baseline():
    """Test fixture mirroring the structure in data/league_pitcher_baseline.json."""
    return {
        "year": 2025,
        "qualifier_min_ip": 50,
        "metrics": {
            "xfip": {
                "direction": "lower_is_better",
                "p10": 3.45, "p25": 3.78, "p50": 4.12, "p75": 4.55, "p90": 4.95,
            },
            "k_bb_pct": {
                "direction": "higher_is_better",
                "p10": 18.5, "p25": 14.2, "p50": 10.0, "p75": 6.5, "p90": 3.0,
            },
            "avg_velo": {
                "direction": "higher_is_better",
                "p10": 96.2, "p25": 94.5, "p50": 93.1, "p75": 91.7, "p90": 90.3,
            },
        },
    }


# ---------------------------------------------------------------------------
# compute_pct — boundary / interpolation behavior
# ---------------------------------------------------------------------------

def test_compute_pct_lower_is_better_at_anchor_boundaries():
    from lib_tier_v2 import compute_pct
    block = _baseline()["metrics"]["xfip"]
    assert compute_pct(block["p10"], block, "lower_is_better") == pytest.approx(0.90)
    assert compute_pct(block["p25"], block, "lower_is_better") == pytest.approx(0.75)
    assert compute_pct(block["p50"], block, "lower_is_better") == pytest.approx(0.50)
    assert compute_pct(block["p75"], block, "lower_is_better") == pytest.approx(0.25)
    assert compute_pct(block["p90"], block, "lower_is_better") == pytest.approx(0.10)


def test_compute_pct_higher_is_better_at_anchor_boundaries():
    from lib_tier_v2 import compute_pct
    block = _baseline()["metrics"]["k_bb_pct"]
    assert compute_pct(block["p10"], block, "higher_is_better") == pytest.approx(0.90)
    assert compute_pct(block["p25"], block, "higher_is_better") == pytest.approx(0.75)
    assert compute_pct(block["p50"], block, "higher_is_better") == pytest.approx(0.50)
    assert compute_pct(block["p75"], block, "higher_is_better") == pytest.approx(0.25)
    assert compute_pct(block["p90"], block, "higher_is_better") == pytest.approx(0.10)


def test_compute_pct_value_better_than_p10_returns_top_5():
    from lib_tier_v2 import compute_pct
    xfip = _baseline()["metrics"]["xfip"]
    # value below p10 (better than top decile) → 0.95
    assert compute_pct(2.50, xfip, "lower_is_better") == pytest.approx(0.95)
    kbb = _baseline()["metrics"]["k_bb_pct"]
    # value above p10 (better than top decile)
    assert compute_pct(25.0, kbb, "higher_is_better") == pytest.approx(0.95)


def test_compute_pct_value_worse_than_p90_returns_bottom_5():
    from lib_tier_v2 import compute_pct
    xfip = _baseline()["metrics"]["xfip"]
    assert compute_pct(6.50, xfip, "lower_is_better") == pytest.approx(0.05)
    kbb = _baseline()["metrics"]["k_bb_pct"]
    assert compute_pct(0.0, kbb, "higher_is_better") == pytest.approx(0.05)


def test_compute_pct_interpolates_between_anchors():
    """xFIP halfway between p25 (3.78) and p50 (4.12) → halfway between 0.75 and 0.50 = 0.625."""
    from lib_tier_v2 import compute_pct
    xfip = _baseline()["metrics"]["xfip"]
    midpoint = (3.78 + 4.12) / 2  # 3.95
    assert compute_pct(midpoint, xfip, "lower_is_better") == pytest.approx(0.625, rel=1e-3)


def test_compute_pct_none_returns_none():
    from lib_tier_v2 import compute_pct
    xfip = _baseline()["metrics"]["xfip"]
    assert compute_pct(None, xfip, "lower_is_better") is None


# ---------------------------------------------------------------------------
# compute_age_factor — age curve
# ---------------------------------------------------------------------------

def test_compute_age_factor_young_no_penalty():
    from lib_tier_v2 import compute_age_factor
    assert compute_age_factor(22) == 1.0
    assert compute_age_factor(27) == 1.0


def test_compute_age_factor_mild_decline_28_to_33():
    from lib_tier_v2 import compute_age_factor
    # 28 → 1.0 - 0.04*1 = 0.96
    assert compute_age_factor(28) == pytest.approx(0.96)
    # 30 → 0.88
    assert compute_age_factor(30) == pytest.approx(0.88)
    # 33 → 0.76
    assert compute_age_factor(33) == pytest.approx(0.76)


def test_compute_age_factor_steep_decline_clamps_at_0_70():
    from lib_tier_v2 import compute_age_factor
    # 35 → max(0.7, 0.76 - 0.10) = 0.7 (clamped)
    assert compute_age_factor(35) == 0.7
    # 40 → 0.7 (clamped)
    assert compute_age_factor(40) == 0.7


def test_compute_age_factor_none_returns_neutral():
    from lib_tier_v2 import compute_age_factor
    assert compute_age_factor(None) == 1.0


# ---------------------------------------------------------------------------
# score_to_tier — bucket boundaries (must match v1 emoji+label strings exactly)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected", [
    (95, "🔴 Elite Ace"),
    (85, "🔴 Elite Ace"),  # boundary
    (84.99, "🟠 Strong Ace"),
    (75, "🟠 Strong Ace"),
    (70, "🟠 Strong Ace"),  # boundary
    (69.99, "🟡 Solid Starter"),
    (55, "🟡 Solid Starter"),
    (50, "🟡 Solid Starter"),  # boundary
    (49.99, "🟢 Back-end Starter"),
    (35, "🟢 Back-end Starter"),
    (30, "🟢 Back-end Starter"),  # boundary
    (29.99, "⚪ Below Average"),
    (10, "⚪ Below Average"),
])
def test_score_to_tier_buckets(score, expected):
    from lib_tier_v2 import score_to_tier
    assert score_to_tier(score) == expected


def test_score_to_tier_none_returns_unknown():
    from lib_tier_v2 import score_to_tier
    assert score_to_tier(None) == "Unknown"


# ---------------------------------------------------------------------------
# compute_tier_v2 — full integration
# ---------------------------------------------------------------------------

def test_compute_tier_v2_elite_pitcher_returns_high_score():
    """Skenes-tier: xfip=2.50, k_bb%=25, velo=98, age=22 → all top-5%, score ~95."""
    from lib_tier_v2 import compute_tier_v2
    season = {"era": 1.80, "xfip": 2.50, "k_bb_pct": 25.0, "ip": 100.0}
    statcast = {"avg_velo": 98.0}
    result = compute_tier_v2(season, statcast, age=22, baseline=_baseline())
    assert result["confidence"] == "data"
    assert result["score"] >= 90
    assert result["tier_v2"] == "🔴 Elite Ace"


def test_compute_tier_v2_average_pitcher_returns_solid():
    """Median across the board (xfip 4.12, k_bb 10.0, velo 93.1) + age 28 (0.96).
    Score = 20 + 17.5 + 7.5 + 9.6 = 54.6 → 🟡 Solid Starter."""
    from lib_tier_v2 import compute_tier_v2
    season = {"xfip": 4.12, "k_bb_pct": 10.0, "ip": 100.0}
    statcast = {"avg_velo": 93.1}
    result = compute_tier_v2(season, statcast, age=28, baseline=_baseline())
    assert result["confidence"] == "data"
    assert result["tier_v2"] == "🟡 Solid Starter"
    # score should be in 50..69 bucket
    assert 50 <= result["score"] <= 69


def test_compute_tier_v2_small_sample_returns_no_score():
    """IP < 30 → confidence 'small_sample', score None, tier_v2 None
    (caller falls back to v1 ERA-only tier)."""
    from lib_tier_v2 import compute_tier_v2
    season = {"xfip": 3.0, "k_bb_pct": 20.0, "ip": 25.0}
    statcast = {"avg_velo": 96.0}
    result = compute_tier_v2(season, statcast, age=24, baseline=_baseline())
    assert result["confidence"] == "small_sample"
    assert result["score"] is None
    assert result["tier_v2"] is None


def test_compute_tier_v2_missing_velo_reweights_to_85():
    """No avg_velo → drop velo component, normalize to 100. xfip+k_bb median + age 25
    score = (20 + 17.5 + 10) / 85 * 100 = 55.88 → 🟡 Solid Starter."""
    from lib_tier_v2 import compute_tier_v2
    season = {"xfip": 4.12, "k_bb_pct": 10.0, "ip": 100.0}
    statcast = {}  # no avg_velo
    result = compute_tier_v2(season, statcast, age=25, baseline=_baseline())
    assert result["confidence"] == "missing_velo"
    assert result["tier_v2"] == "🟡 Solid Starter"
    assert result["components"]["velo_pct"] is None


def test_compute_tier_v2_missing_baseline_falls_back():
    """baseline empty → confidence 'missing_baseline', score None, tier_v2 None."""
    from lib_tier_v2 import compute_tier_v2
    season = {"xfip": 3.0, "k_bb_pct": 20.0, "ip": 100.0}
    statcast = {"avg_velo": 96.0}
    result = compute_tier_v2(season, statcast, age=24, baseline={})
    assert result["confidence"] == "missing_baseline"
    assert result["score"] is None
    assert result["tier_v2"] is None


def test_compute_tier_v2_missing_core_metric_returns_insufficient():
    """xfip or k_bb_pct missing entirely → can't compute. (Both are weighted 35-40%.)"""
    from lib_tier_v2 import compute_tier_v2
    season = {"xfip": None, "k_bb_pct": 20.0, "ip": 100.0}
    statcast = {"avg_velo": 96.0}
    result = compute_tier_v2(season, statcast, age=24, baseline=_baseline())
    assert result["confidence"] == "insufficient_data"
    assert result["tier_v2"] is None


def test_compute_tier_v2_old_pitcher_age_drag():
    """Same metrics, age 36 → age_factor 0.7 → score lower than age 25 case."""
    from lib_tier_v2 import compute_tier_v2
    season = {"xfip": 3.78, "k_bb_pct": 14.2, "ip": 100.0}
    statcast = {"avg_velo": 94.5}
    young = compute_tier_v2(season, statcast, age=25, baseline=_baseline())
    old = compute_tier_v2(season, statcast, age=36, baseline=_baseline())
    assert young["score"] > old["score"]
    # Diff should be (1.0 - 0.7) * 10 = 3.0 points
    assert young["score"] - old["score"] == pytest.approx(3.0, rel=0.05)


def test_compute_tier_v2_components_in_output():
    """Output must expose components dict for downstream tier_gap / dossier rendering."""
    from lib_tier_v2 import compute_tier_v2
    season = {"xfip": 4.12, "k_bb_pct": 10.0, "ip": 100.0}
    statcast = {"avg_velo": 93.1}
    result = compute_tier_v2(season, statcast, age=28, baseline=_baseline())
    c = result["components"]
    assert "xfip_pct" in c
    assert "k_bb_pct" in c
    assert "velo_pct" in c
    assert "age_factor" in c
    assert c["xfip_pct"] == pytest.approx(0.50)
    assert c["k_bb_pct"] == pytest.approx(0.50)
    assert c["velo_pct"] == pytest.approx(0.50)
    assert c["age_factor"] == pytest.approx(0.96)
