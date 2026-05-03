"""signals_lib — derived signals computed from bundle data.

Pure-function leaf module: imports stdlib only. Signals are read by:
  - dossier_renderer (## 訊號摘要 section, PR-3 commit 13)
  - summary_renderer (## 風險提示 § 額外信號, PR-3 commit 16)

Signal contract:
    {
        "name": str,                                 # canonical key
        "fired": bool,                               # whether to display
        "value": float | int | str | None,           # quantification
        "severity": "low" | "medium" | "high",       # display urgency
        "label": str,                                # one-line summary for dossier
        "details": dict,                             # extra context for AI
        "confidence": "data" | "heuristic" | "small_sample",
    }

Signals do NOT enter the scoring formula (see flags-checklist.md §3 / §8 —
"不主動 ±run value"). They are surfaced for AI judgment in summary.md.
"""

from __future__ import annotations


def _make(
    name: str,
    fired: bool,
    *,
    value=None,
    severity: str = "low",
    label: str = "",
    details: dict | None = None,
    confidence: str = "data",
) -> dict:
    return {
        "name": name,
        "fired": fired,
        "value": value,
        "severity": severity,
        "label": label,
        "details": details or {},
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# 1. tier_mismatch — pitcher tier_v2 score vs ERA-only tier anchor
# ---------------------------------------------------------------------------

def signal_tier_mismatch(tier_gap: dict | None) -> dict:
    """Surface |tier_gap.gap| ≥ 15 as a tier mismatch signal.

    gap > 0 → ERA 低估真實水平; gap < 0 → ERA 高估真實水平.
    Severity: |gap| ≥ 20 high, 15 ≤ |gap| < 20 medium, otherwise no fire.

    Does NOT auto-trigger Flag 9; AI in summary.md judges luck vs structural.
    """
    name = "tier_mismatch"
    if not tier_gap:
        return _make(name, False, confidence="small_sample")
    gap = tier_gap.get("gap")
    if gap is None:
        return _make(name, False, confidence="small_sample")
    abs_gap = abs(gap)
    if abs_gap < 15:
        return _make(name, False, value=gap, label=f"tier_gap {gap:+.1f} 在容許範圍")
    severity = "high" if abs_gap >= 20 else "medium"
    if gap > 0:
        label = f"ERA 低估真實水平 +{gap:.1f}（v2 score {tier_gap.get('expected_score')} vs ERA-only {tier_gap.get('era_only_score')}）"
    else:
        label = f"ERA 高估真實水平 {gap:.1f}（v2 score {tier_gap.get('expected_score')} vs ERA-only {tier_gap.get('era_only_score')}）"
    return _make(
        name, True, value=gap, severity=severity, label=label,
        details={
            "expected_score": tier_gap.get("expected_score"),
            "era_only_score": tier_gap.get("era_only_score"),
        },
    )


# ---------------------------------------------------------------------------
# 2. heat_vs_babip — lineup recent_heat × last7 BABIP cross-check
# ---------------------------------------------------------------------------

def signal_heat_vs_babip(heat: str | None, last7_babip: float | None) -> dict:
    """Cross-check lineup heat vs last7 BABIP for luck/regression signal.

    🔥 Hot + BABIP ≥ .350 → "lucky-hot, 注意回歸"
    🥶 Cold + BABIP ≤ .270 → "unlucky-cold, 可能反彈"
    其他組合 → no fire (real heat / cold not BABIP-driven).
    """
    name = "heat_vs_babip"
    if heat is None or last7_babip is None:
        return _make(name, False, confidence="small_sample")
    try:
        babip = float(last7_babip)
    except (TypeError, ValueError):
        return _make(name, False, confidence="small_sample")

    is_hot = heat and ("Hot" in heat or "🔥" in heat)
    is_cold = heat and ("Cold" in heat or "🥶" in heat)

    if is_hot and babip >= 0.350:
        return _make(
            name, True, value=babip, severity="medium",
            label=f"lucky-hot：last7 BABIP {babip:.3f} 偏高，熱度可能含運氣",
            details={"heat": heat, "last7_babip": babip},
            confidence="heuristic",
        )
    if is_cold and babip <= 0.270:
        return _make(
            name, True, value=babip, severity="medium",
            label=f"unlucky-cold：last7 BABIP {babip:.3f} 偏低，冷期可能反彈",
            details={"heat": heat, "last7_babip": babip},
            confidence="heuristic",
        )
    return _make(name, False, value=babip)


# ---------------------------------------------------------------------------
# 3. platoon_advantage — top-5 batters with vs-hand OPS uplift ≥ 0.050
# ---------------------------------------------------------------------------

_PLATOON_UPLIFT_THRESHOLD = 0.050  # OPS uplift considered meaningful
_PLATOON_FIRE_COUNT = 4  # 4 of top 5 → strong signal


def signal_platoon_advantage(core_lineup: list, pitcher_hand: str) -> dict:
    """Count top-5 batters whose vs-this-hand OPS exceeds season OPS by ≥ 0.050.

    Fires when ≥ 4 of 5. Useful for matchup tier inference (e.g. NYY 4-5 RHB
    facing Bradish RHP with reverse-platoon).
    """
    name = "platoon_advantage"
    if not core_lineup:
        return _make(name, False, confidence="small_sample")
    key = "vs_lhp" if pitcher_hand == "L" else "vs_rhp"
    top5 = core_lineup[:5]
    uplifted = 0
    for b in top5:
        season = b.get("ops")
        if season is None:
            continue
        try:
            season_f = float(season)
        except (TypeError, ValueError):
            continue
        platoon = (b.get("platoon") or {}).get(key) or {}
        vs_ops = platoon.get("ops")
        if vs_ops is None:
            continue
        try:
            vs_f = float(vs_ops)
        except (TypeError, ValueError):
            continue
        if vs_f - season_f >= _PLATOON_UPLIFT_THRESHOLD:
            uplifted += 1
    if uplifted >= _PLATOON_FIRE_COUNT:
        return _make(
            name, True, value=uplifted, severity="medium",
            label=f"platoon advantage：top 5 中 {uplifted} 人對 {pitcher_hand}HP OPS 較 season +0.050 以上",
            details={"uplifted_count": uplifted, "top5_size": len(top5), "pitcher_hand": pitcher_hand},
            confidence="data",
        )
    return _make(name, False, value=uplifted)


# ---------------------------------------------------------------------------
# 4. strong_park — extreme park factor (≥ 110 or ≤ 90) signal
# ---------------------------------------------------------------------------

_PF_HIGH = 110
_PF_LOW = 90


def signal_strong_park(park_factor: float | None) -> dict:
    """Surface extreme park factors. Neutral parks (90 < PF < 110) don't fire."""
    name = "strong_park"
    if park_factor is None:
        return _make(name, False, confidence="small_sample")
    try:
        pf = float(park_factor)
    except (TypeError, ValueError):
        return _make(name, False, confidence="small_sample")
    if pf >= _PF_HIGH:
        severity = "high" if pf >= 115 else "medium"
        return _make(
            name, True, value=pf, severity=severity,
            label=f"打者友善球場 PF {pf:.0f}（≥{_PF_HIGH}）",
            details={"park_factor": pf},
        )
    if pf <= _PF_LOW:
        severity = "high" if pf <= 85 else "medium"
        return _make(
            name, True, value=pf, severity=severity,
            label=f"投手友善球場 PF {pf:.0f}（≤{_PF_LOW}）",
            details={"park_factor": pf},
        )
    return _make(name, False, value=pf)
