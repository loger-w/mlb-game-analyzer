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
    """Test fixture mirroring the structure in data/league_pitcher_baseline.json.

    Includes stuff_plus / pitching_plus so post-Stuff+-refactor tests can resolve
    pct values; existing avg_velo block kept for legacy info-only display.
    """
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
            "stuff_plus": {
                "direction": "higher_is_better",
                "p10": 115.0, "p25": 107.5, "p50": 100.0, "p75": 92.5, "p90": 85.0,
            },
            "pitching_plus": {
                "direction": "higher_is_better",
                "p10": 112.0, "p25": 106.0, "p50": 100.0, "p75": 94.0, "p90": 88.0,
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
    """Skenes-tier: xfip=2.50, k_bb%=25, stuff+ 130, age=22 → all top-5%, score ~95."""
    from lib_tier_v2 import compute_tier_v2
    season = {"era": 1.80, "xfip": 2.50, "k_bb_pct": 25.0, "ip": 100.0}
    statcast = {"avg_velo": 98.0}
    stuff = {"stuff_plus": 130.0, "pitching_plus": 125.0}
    result = compute_tier_v2(season, statcast, age=22, stuff=stuff, baseline=_baseline())
    assert result["confidence"] == "data"
    assert result["score"] >= 90
    assert result["tier_v2"] == "🔴 Elite Ace"


def test_compute_tier_v2_average_pitcher_returns_solid():
    """Median across the board (xfip 4.12 / k_bb 10.0 / stuff+ 100) + age 28 (0.96).
    Score = 0.5×30 + 0.5×25 + 0.5×30 + 0.96×15 = 15 + 12.5 + 15 + 14.4 = 56.9 → 🟡 Solid."""
    from lib_tier_v2 import compute_tier_v2
    season = {"xfip": 4.12, "k_bb_pct": 10.0, "ip": 100.0}
    statcast = {"avg_velo": 93.1}
    stuff = {"stuff_plus": 100.0, "pitching_plus": 100.0}
    result = compute_tier_v2(season, statcast, age=28, stuff=stuff, baseline=_baseline())
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


def test_compute_tier_v2_missing_stuff_reweights_to_70():
    """No stuff data → drop Stuff+ 30-pt component, renormalize remaining 70 → 100.
    xfip + k_bb median + age 25 (1.0):
        raw = 0.5×30 + 0.5×25 + 1.0×15 = 15 + 12.5 + 15 = 42.5
        renormalized = 42.5 / 70 * 100 ≈ 60.7 → 🟡 Solid Starter."""
    from lib_tier_v2 import compute_tier_v2
    season = {"xfip": 4.12, "k_bb_pct": 10.0, "ip": 100.0}
    statcast = {"avg_velo": 93.1}  # velo no longer in formula but still in JSON
    result = compute_tier_v2(season, statcast, age=25, stuff=None, baseline=_baseline())
    assert result["confidence"] == "missing_stuff"
    assert result["tier_v2"] == "🟡 Solid Starter"
    assert result["components"]["stuff_pct"] is None


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
    """Same metrics, age 36 → age_factor 0.7 → score lower than age 25 case.

    With Stuff+ refactor age weight is 15 (up from 10), so diff = (1.0 - 0.7) * 15 = 4.5."""
    from lib_tier_v2 import compute_tier_v2
    season = {"xfip": 3.78, "k_bb_pct": 14.2, "ip": 100.0}
    statcast = {"avg_velo": 94.5}
    stuff = {"stuff_plus": 107.5, "pitching_plus": 106.0}
    young = compute_tier_v2(season, statcast, age=25, stuff=stuff, baseline=_baseline())
    old = compute_tier_v2(season, statcast, age=36, stuff=stuff, baseline=_baseline())
    assert young["score"] > old["score"]
    # Diff should be (1.0 - 0.7) * 15 = 4.5 points
    assert young["score"] - old["score"] == pytest.approx(4.5, rel=0.05)


def test_compute_tier_v2_components_in_output():
    """Output must expose components dict for downstream tier_gap / dossier rendering.

    Stuff+ refactor: components hold xfip / k_bb / stuff / age. velo_pct is gone
    (velo no longer enters the score formula; raw avg_velo still in pitcher.statcast)."""
    from lib_tier_v2 import compute_tier_v2
    season = {"xfip": 4.12, "k_bb_pct": 10.0, "ip": 100.0}
    statcast = {"avg_velo": 93.1}
    stuff = {"stuff_plus": 100.0, "pitching_plus": 100.0}
    result = compute_tier_v2(season, statcast, age=28, stuff=stuff, baseline=_baseline())
    c = result["components"]
    assert "xfip_pct" in c
    assert "k_bb_pct" in c
    assert "stuff_pct" in c
    assert "age_factor" in c
    assert "velo_pct" not in c, "velo_pct removed from components after Stuff+ refactor"
    assert c["xfip_pct"] == pytest.approx(0.50)
    assert c["k_bb_pct"] == pytest.approx(0.50)
    assert c["stuff_pct"] == pytest.approx(0.50)
    assert c["age_factor"] == pytest.approx(0.96)


# ---------------------------------------------------------------------------
# Stuff+ specific behavior (added by Stuff+ refactor)
# ---------------------------------------------------------------------------

def test_compute_tier_v2_velo_no_longer_affects_score():
    """velo varying with all else identical → score unchanged. velo is purely
    informational after refactor; Stuff+ subsumes the physical-stuff signal."""
    from lib_tier_v2 import compute_tier_v2
    season = {"xfip": 4.12, "k_bb_pct": 10.0, "ip": 100.0}
    stuff = {"stuff_plus": 100.0, "pitching_plus": 100.0}
    slow = compute_tier_v2(season, {"avg_velo": 88.0}, age=28,
                           stuff=stuff, baseline=_baseline())
    fast = compute_tier_v2(season, {"avg_velo": 99.0}, age=28,
                           stuff=stuff, baseline=_baseline())
    assert slow["score"] == fast["score"]


def test_compute_tier_v2_stuff_weight_is_30():
    """Stuff+ at p10 (best) vs p90 (worst) with everything else median should
    swing score by ~0.85 × 30 = 25.5 points (the Stuff+ weight)."""
    from lib_tier_v2 import compute_tier_v2
    season = {"xfip": 4.12, "k_bb_pct": 10.0, "ip": 100.0}
    statcast = {"avg_velo": 93.1}
    elite_stuff = {"stuff_plus": 115.0, "pitching_plus": 100.0}  # p10
    poor_stuff = {"stuff_plus": 85.0, "pitching_plus": 100.0}    # p90
    elite = compute_tier_v2(season, statcast, age=28, stuff=elite_stuff,
                            baseline=_baseline())
    poor = compute_tier_v2(season, statcast, age=28, stuff=poor_stuff,
                           baseline=_baseline())
    diff = elite["score"] - poor["score"]
    # (0.90 - 0.10) × 30 = 24
    assert diff == pytest.approx(24.0, rel=0.05)


def test_compute_tier_v2_stuff_dict_with_none_value_is_missing_stuff():
    """stuff dict present but stuff_plus=None (e.g. fetch returned partial data)
    → treated as missing_stuff, not insufficient_data."""
    from lib_tier_v2 import compute_tier_v2
    season = {"xfip": 4.12, "k_bb_pct": 10.0, "ip": 100.0}
    statcast = {"avg_velo": 93.1}
    result = compute_tier_v2(season, statcast, age=25,
                             stuff={"stuff_plus": None, "pitching_plus": None},
                             baseline=_baseline())
    assert result["confidence"] == "missing_stuff"
    assert result["components"]["stuff_pct"] is None


# ---------------------------------------------------------------------------
# compute_era_only_score — linear interpolation (Bug 4 fix)
# ---------------------------------------------------------------------------

def test_compute_era_only_score_at_anchors():
    """Each anchor ERA returns its anchored score exactly (no interpolation)."""
    from lib_tier_v2 import compute_era_only_score
    assert compute_era_only_score(2.00) == pytest.approx(95.0)
    assert compute_era_only_score(2.50) == pytest.approx(90.0)
    assert compute_era_only_score(3.20) == pytest.approx(75.0)
    assert compute_era_only_score(4.20) == pytest.approx(55.0)
    assert compute_era_only_score(5.00) == pytest.approx(35.0)
    assert compute_era_only_score(6.00) == pytest.approx(15.0)


def test_compute_era_only_score_interpolates_between_anchors():
    """ERA 4.5 falls between 4.20→55 and 5.00→35: linear → 55 - 0.375×20 = 47.5."""
    from lib_tier_v2 import compute_era_only_score
    assert compute_era_only_score(4.50) == pytest.approx(47.5)


def test_compute_era_only_score_boundary_smooth():
    """Bug 4 was: 4.99 → 35 (Back-end mid) but 5.00 → 15 (Below mid) — 0.01
    ERA jumped 20 score points. Linear interpolation must keep diff < 1.0."""
    from lib_tier_v2 import compute_era_only_score
    diff = compute_era_only_score(4.99) - compute_era_only_score(5.00)
    assert abs(diff) < 1.0


def test_compute_era_only_score_clamps_at_extremes():
    """ERA below 2.00 clamps at 95 (don't extrapolate Elite to ridiculous values).
    ERA above 6.00 clamps at 15 (don't go negative for terrible ERAs)."""
    from lib_tier_v2 import compute_era_only_score
    assert compute_era_only_score(1.50) == pytest.approx(95.0)
    assert compute_era_only_score(0.00) == pytest.approx(95.0)
    assert compute_era_only_score(7.00) == pytest.approx(15.0)
    assert compute_era_only_score(15.0) == pytest.approx(15.0)


def test_compute_era_only_score_none_returns_none():
    """ERA None (e.g. small_sample, no IP) → None."""
    from lib_tier_v2 import compute_era_only_score
    assert compute_era_only_score(None) is None


def test_compute_era_only_score_invalid_string_returns_none():
    """ERA non-numeric → None (defensive against API drift)."""
    from lib_tier_v2 import compute_era_only_score
    assert compute_era_only_score("abc") is None


# ---------------------------------------------------------------------------
# compute_tier_gap (Bug 4) — uses linear-interpolated ERA score, not table lookup
# ---------------------------------------------------------------------------

def test_compute_tier_gap_positive_when_v2_better_than_era():
    """tier_v2 score 90 vs ERA 4.50 → era_only_score 47.5 → gap +42.5.
    Means: ERA understates real level (e.g. high BABIP / low LOB% inflating ERA)."""
    from lib_tier_v2 import compute_tier_gap
    tier_v2_result = {"score": 90.0, "tier_v2": "🔴 Elite Ace"}
    result = compute_tier_gap(tier_v2_result, era=4.50)
    assert result["expected_score"] == 90.0
    assert result["era_only_score"] == pytest.approx(47.5)
    assert result["gap"] == pytest.approx(42.5)


def test_compute_tier_gap_negative_when_era_flatters():
    """ERA 2.40 → era_only_score 91; v2 score 55 → gap −36.
    Means: ERA flatters (low BABIP / high LOB inflating). t=(2.4-2.0)/(2.5-2.0)=0.8;
    score = 95 + 0.8*(90-95) = 91."""
    from lib_tier_v2 import compute_tier_gap
    tier_v2_result = {"score": 55.0, "tier_v2": "🟡 Solid Starter"}
    result = compute_tier_gap(tier_v2_result, era=2.40)
    assert result["era_only_score"] == pytest.approx(91.0)
    assert result["gap"] == pytest.approx(-36.0)


def test_compute_tier_gap_aligned_returns_small_gap():
    """ERA 3.20 (anchor → 75) + v2 score 75 → gap 0."""
    from lib_tier_v2 import compute_tier_gap
    tier_v2_result = {"score": 75.0, "tier_v2": "🟠 Strong Ace"}
    result = compute_tier_gap(tier_v2_result, era=3.20)
    assert result["gap"] == pytest.approx(0.0)


def test_compute_tier_gap_none_when_v2_score_unavailable():
    """tier_v2 score is None (small sample) → gap None, era_only_score still
    populated (caller can show 'no v2 score' but still know the ERA-derived score)."""
    from lib_tier_v2 import compute_tier_gap
    tier_v2_result = {"score": None, "tier_v2": None}
    result = compute_tier_gap(tier_v2_result, era=3.20)
    assert result["expected_score"] is None
    assert result["era_only_score"] == pytest.approx(75.0)
    assert result["gap"] is None


def test_compute_tier_gap_when_era_is_none():
    """era is None (e.g. pitcher hasn't pitched yet) → era_only_score None, gap None."""
    from lib_tier_v2 import compute_tier_gap
    tier_v2_result = {"score": 75.0, "tier_v2": "🟠 Strong Ace"}
    result = compute_tier_gap(tier_v2_result, era=None)
    assert result["expected_score"] == 75.0
    assert result["era_only_score"] is None
    assert result["gap"] is None
