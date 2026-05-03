"""lib_tier_v2 — blended pitcher tier formula based on xFIP / K-BB% / Stuff+ / age.

Pure functions, no I/O at call time (baseline is loaded once on first use; tests
inject baseline=... to bypass disk).

Score formula (Stuff+ refactor; velo no longer enters formula but stays in
pitcher.statcast for analyst reference):
    score = 30 × pct(xfip, lower_is_better)
          + 25 × pct(k_bb_pct, higher_is_better)
          + 30 × pct(stuff_plus, higher_is_better)
          + 15 × age_factor

Tier buckets (must match v1 emoji+label strings exactly so existing substring
assertions stay green):
    ≥85 → 🔴 Elite Ace
    70-84 → 🟠 Strong Ace
    50-69 → 🟡 Solid Starter
    30-49 → 🟢 Back-end Starter
    <30 → ⚪ Below Average
"""

from __future__ import annotations

import json
from pathlib import Path

_BASELINE_CACHE: dict | None = None


def _load_baseline() -> dict:
    """Lazy-load `data/league_pitcher_baseline.json`. Returns {} if missing."""
    global _BASELINE_CACHE
    if _BASELINE_CACHE is not None:
        return _BASELINE_CACHE
    path = Path(__file__).parent / "data" / "league_pitcher_baseline.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            _BASELINE_CACHE = json.load(f)
    except FileNotFoundError:
        _BASELINE_CACHE = {}
    return _BASELINE_CACHE


# Anchor ranks: p10 → 0.90, p25 → 0.75, p50 → 0.50, p75 → 0.25, p90 → 0.10.
# Convention: p10 means "top decile" (best 10%). Lower-is-better metrics have
# p10 < p25 < ... < p90; higher-is-better is reversed.
_ANCHOR_RANKS = (
    ("p10", 0.90),
    ("p25", 0.75),
    ("p50", 0.50),
    ("p75", 0.25),
    ("p90", 0.10),
)


def compute_pct(value, percentile_dict: dict, direction: str) -> float | None:
    """Map metric value to percentile rank in 0..1 (1 = best).

    Linear interpolation between the 5 anchor points. Values outside the p10/p90
    range clamp to 0.95 (top 5%) / 0.05 (bottom 5%) — we don't extrapolate.

    direction: "lower_is_better" or "higher_is_better".
    """
    if value is None:
        return None

    anchors = [(percentile_dict[k], rank) for k, rank in _ANCHOR_RANKS]

    if direction == "lower_is_better":
        # anchors[0] = (smallest value, 0.90 = best)
        # anchors[-1] = (largest value, 0.10 = worst)
        if value < anchors[0][0]:
            return 0.95
        if value > anchors[-1][0]:
            return 0.05
        for i in range(len(anchors) - 1):
            v_low, r_high = anchors[i]      # better value, higher rank
            v_high, r_low = anchors[i + 1]  # worse value, lower rank
            if v_low <= value <= v_high:
                if v_high == v_low:
                    return r_high
                fraction = (v_high - value) / (v_high - v_low)  # 1 at v_low, 0 at v_high
                return r_low + fraction * (r_high - r_low)
        return 0.5  # defensive fallback

    if direction == "higher_is_better":
        # anchors[0] = (largest value, 0.90 = best)
        # anchors[-1] = (smallest value, 0.10 = worst)
        if value > anchors[0][0]:
            return 0.95
        if value < anchors[-1][0]:
            return 0.05
        for i in range(len(anchors) - 1):
            v_high, r_high = anchors[i]     # better value, higher rank
            v_low, r_low = anchors[i + 1]   # worse value, lower rank
            if v_low <= value <= v_high:
                if v_high == v_low:
                    return r_high
                fraction = (value - v_low) / (v_high - v_low)  # 1 at v_high, 0 at v_low
                return r_low + fraction * (r_high - r_low)
        return 0.5

    raise ValueError(f"Unknown direction: {direction!r}")


def compute_age_factor(age: int | None) -> float:
    """Age curve mapping pitcher age → 0..1 multiplier on the 10-pt age component.

    ≤ 27 → 1.0 (no penalty)
    28..33 → linear decline 1.0 → 0.76 (−0.04 per year)
    ≥ 34 → steeper decline 0.76 → 0.71 (−0.05 per year), clamped at 0.7

    None → 1.0 (neutral; rookies sometimes lack birthdate in API).
    """
    if age is None:
        return 1.0
    if age <= 27:
        return 1.0
    if age <= 33:
        return round(1.0 - 0.04 * (age - 27), 4)
    return round(max(0.7, 0.76 - 0.05 * (age - 33)), 4)


def score_to_tier(score: float | None) -> str:
    """Bucket numeric score into tier label. Strings match v1 conventions verbatim."""
    if score is None:
        return "Unknown"
    if score >= 85:
        return "🔴 Elite Ace"
    if score >= 70:
        return "🟠 Strong Ace"
    if score >= 50:
        return "🟡 Solid Starter"
    if score >= 30:
        return "🟢 Back-end Starter"
    return "⚪ Below Average"


# Component weights (sum = 100). When stuff_plus is missing we drop its 30 and
# renormalize on the remaining 70.
_WEIGHT_XFIP = 30
_WEIGHT_KBB = 25
_WEIGHT_STUFF = 30
_WEIGHT_AGE = 15


# Mapping v1 ERA-only tier → numeric anchor for tier_gap comparison.
# Anchors are bucket midpoints: Elite 90, Strong 75, Solid 55, Back-end 35,
# Below 15. Diff vs tier_v2 score surfaces "ERA flatters / understates real level".
ERA_ONLY_SCORE_MAP = {
    "🔴 Elite Ace": 90,
    "🟠 Strong Ace": 75,
    "🟡 Solid Starter": 55,
    "🟢 Back-end Starter": 35,
    "⚪ Below Average": 15,
}


def compute_tier_gap(tier_v2_result: dict, era_only_tier: str) -> dict:
    """Compare tier_v2 numeric score to v1 ERA-only tier anchor.

    Output:
        expected_score (float | None) — tier_v2 score
        era_only_score (int | None)   — anchor for v1 tier (None if unknown)
        gap (float | None)            — expected_score − era_only_score

    Sign convention:
        gap > 0 → ERA understates real level (e.g. luck inflating ERA upward)
        gap < 0 → ERA flatters real level (e.g. low BABIP, high LOB%)
        |gap| ≥ 15 is the threshold the dossier surfaces (handled in PR-3
        signals_lib.tier_mismatch); we DO NOT auto-trigger Flag 9 here.
    """
    expected_score = (tier_v2_result or {}).get("score")
    era_only_score = ERA_ONLY_SCORE_MAP.get(era_only_tier)
    if expected_score is None or era_only_score is None:
        return {
            "expected_score": expected_score,
            "era_only_score": era_only_score,
            "gap": None,
        }
    return {
        "expected_score": expected_score,
        "era_only_score": era_only_score,
        "gap": round(expected_score - era_only_score, 1),
    }


def compute_tier_v2(
    season: dict | None,
    statcast: dict | None,
    age: int | None = None,
    stuff: dict | None = None,
    baseline: dict | None = None,
) -> dict:
    """Blend xFIP / K-BB% / Stuff+ / age into a 0..100 tier score.

    `stuff` is the FanGraphs Stuff+/Pitching+ dict from `pitcher_stats.fetch_stuff_pitching_plus`.
    Pass None when fetch failed; component drops out and remaining 70 weight
    renormalizes to 100 (confidence "missing_stuff"). velo is no longer in the
    formula but `statcast.avg_velo` still flows through to JSON / dossier display.

    Returns dict with keys:
        score (float | None) — 0..100, None if can't compute
        tier_v2 (str | None) — bucketed label, None if score is None
        components (dict)    — {xfip_pct, k_bb_pct, stuff_pct, age_factor}
        confidence (str)     — one of:
            "data"               — full data, all 4 components present
            "missing_stuff"      — stuff_plus unavailable, reweighted to 70 → 100
            "small_sample"       — IP < 30, no v2 score (caller falls back to v1)
            "missing_baseline"   — baseline.json missing or empty
            "insufficient_data"  — xfip or k_bb_pct missing (cannot compute)
    """
    if baseline is None:
        baseline = _load_baseline()

    metrics = (baseline or {}).get("metrics") or {}
    if not metrics:
        return {
            "score": None, "tier_v2": None,
            "components": {}, "confidence": "missing_baseline",
        }

    season = season or {}
    statcast = statcast or {}
    stuff = stuff or {}

    ip = season.get("ip")
    if ip is not None and ip < 30:
        return {
            "score": None, "tier_v2": None,
            "components": {}, "confidence": "small_sample",
        }

    xfip = season.get("xfip")
    k_bb_pct = season.get("k_bb_pct")
    stuff_plus = stuff.get("stuff_plus")

    xfip_pct = compute_pct(xfip, metrics["xfip"], "lower_is_better") if xfip is not None else None
    kbb_pct = compute_pct(k_bb_pct, metrics["k_bb_pct"], "higher_is_better") if k_bb_pct is not None else None
    stuff_metrics = metrics.get("stuff_plus")
    stuff_pct = (
        compute_pct(stuff_plus, stuff_metrics, "higher_is_better")
        if (stuff_plus is not None and stuff_metrics)
        else None
    )
    age_factor = compute_age_factor(age)

    components = {
        "xfip_pct": round(xfip_pct, 3) if xfip_pct is not None else None,
        "k_bb_pct": round(kbb_pct, 3) if kbb_pct is not None else None,
        "stuff_pct": round(stuff_pct, 3) if stuff_pct is not None else None,
        "age_factor": age_factor,
    }

    if xfip_pct is None or kbb_pct is None:
        return {
            "score": None, "tier_v2": None,
            "components": components, "confidence": "insufficient_data",
        }

    if stuff_pct is None:
        # Drop Stuff+ weight, renormalize 30+25+15 = 70 → 100
        raw = xfip_pct * _WEIGHT_XFIP + kbb_pct * _WEIGHT_KBB + age_factor * _WEIGHT_AGE
        score = raw / (_WEIGHT_XFIP + _WEIGHT_KBB + _WEIGHT_AGE) * 100
        confidence = "missing_stuff"
    else:
        score = (
            xfip_pct * _WEIGHT_XFIP
            + kbb_pct * _WEIGHT_KBB
            + stuff_pct * _WEIGHT_STUFF
            + age_factor * _WEIGHT_AGE
        )
        confidence = "data"

    score = round(score, 1)
    return {
        "score": score,
        "tier_v2": score_to_tier(score),
        "components": components,
        "confidence": confidence,
    }
