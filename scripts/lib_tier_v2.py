"""lib_tier_v2 — blended pitcher tier formula based on xFIP / K-BB% / velo / age.

Pure functions, no I/O at call time (baseline is loaded once on first use; tests
inject baseline=... to bypass disk).

Score formula:
    score = 40 × pct(xfip, lower_is_better)
          + 35 × pct(k_bb_pct, higher_is_better)
          + 15 × pct(avg_velo, higher_is_better)
          + 10 × age_factor

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


# Component weights (sum = 100). When velo is missing we drop its 15 and
# renormalize on the remaining 85.
_WEIGHT_XFIP = 40
_WEIGHT_KBB = 35
_WEIGHT_VELO = 15
_WEIGHT_AGE = 10


def compute_tier_v2(
    season: dict | None,
    statcast: dict | None,
    age: int | None = None,
    baseline: dict | None = None,
) -> dict:
    """Blend xFIP / K-BB% / velo / age into a 0..100 tier score.

    Returns dict with keys:
        score (float | None) — 0..100, None if can't compute
        tier_v2 (str | None) — bucketed label, None if score is None
        components (dict)    — {xfip_pct, k_bb_pct, velo_pct, age_factor}
        confidence (str)     — one of:
            "data"               — full data, all 4 components present
            "missing_velo"       — velo unavailable, reweighted to 85 → 100
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

    ip = season.get("ip")
    if ip is not None and ip < 30:
        return {
            "score": None, "tier_v2": None,
            "components": {}, "confidence": "small_sample",
        }

    xfip = season.get("xfip")
    k_bb_pct = season.get("k_bb_pct")
    velo = statcast.get("avg_velo")

    xfip_pct = compute_pct(xfip, metrics["xfip"], "lower_is_better") if xfip is not None else None
    kbb_pct = compute_pct(k_bb_pct, metrics["k_bb_pct"], "higher_is_better") if k_bb_pct is not None else None
    velo_pct = compute_pct(velo, metrics["avg_velo"], "higher_is_better") if velo is not None else None
    age_factor = compute_age_factor(age)

    components = {
        "xfip_pct": round(xfip_pct, 3) if xfip_pct is not None else None,
        "k_bb_pct": round(kbb_pct, 3) if kbb_pct is not None else None,
        "velo_pct": round(velo_pct, 3) if velo_pct is not None else None,
        "age_factor": age_factor,
    }

    if xfip_pct is None or kbb_pct is None:
        return {
            "score": None, "tier_v2": None,
            "components": components, "confidence": "insufficient_data",
        }

    if velo_pct is None:
        # Drop velo weight, renormalize 40+35+10 = 85 → 100
        raw = xfip_pct * _WEIGHT_XFIP + kbb_pct * _WEIGHT_KBB + age_factor * _WEIGHT_AGE
        score = raw / (_WEIGHT_XFIP + _WEIGHT_KBB + _WEIGHT_AGE) * 100
        confidence = "missing_velo"
    else:
        score = (
            xfip_pct * _WEIGHT_XFIP
            + kbb_pct * _WEIGHT_KBB
            + velo_pct * _WEIGHT_VELO
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
