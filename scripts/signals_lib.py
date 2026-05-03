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
        "half_life": "structural" | "medium" | "short",
    }

`half_life` classifies how quickly the signal goes stale (reflexivity 物理):
    structural — multi-year / 物理特徵：park, tier_mismatch
    medium     — season split，季中可調：platoon, mix, chain
    short      — last7 / 每天異動：heat, IL count
Renderers prepend ⏳ badge to short half_life signals so analyst discounts
short-window readings.

Signals do NOT enter the scoring formula (see flags-checklist.md §3 / §8 —
"不主動 ±run value"). They are surfaced for AI judgment in summary.md.
"""

from __future__ import annotations


def _to_float(v) -> float | None:
    """Coerce v to float; return None on None / non-numeric strings."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# Static half_life classification per signal name.
# Single source of truth; _make() looks up by name unless explicitly overridden.
# Reflexivity 物理：對手會根據 signal 調整，short half_life 容易被治療，
# structural / medium 較穩定。
_HALF_LIFE_BY_NAME = {
    "tier_mismatch": "structural",          # season-to-date 累計
    "heat_vs_babip": "short",               # last7 window
    "platoon_advantage": "medium",          # season split
    "strong_park": "structural",            # multi-year 物理特徵
    "reverse_platoon": "medium",            # season split, 對手換打者就變
    "chain_break": "medium",                # season OPS 結構
    "pitch_mix_concentration": "medium",    # multi-month aggregate
    "core_il_count": "short",               # IL 名單每天異動
}


def _make(
    name: str,
    fired: bool,
    *,
    value=None,
    severity: str = "low",
    label: str = "",
    details: dict | None = None,
    confidence: str = "data",
    half_life: str | None = None,
) -> dict:
    """Build canonical signal dict. half_life looked up from _HALF_LIFE_BY_NAME
    by default; pass explicit override only for tests / synthetic signals."""
    if half_life is None:
        half_life = _HALF_LIFE_BY_NAME.get(name, "medium")
    return {
        "name": name,
        "fired": fired,
        "value": value,
        "severity": severity,
        "label": label,
        "details": details or {},
        "confidence": confidence,
        "half_life": half_life,
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
    babip = _to_float(last7_babip)
    if babip is None:
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
        season_f = _to_float(b.get("ops"))
        if season_f is None:
            continue
        platoon = (b.get("platoon") or {}).get(key) or {}
        vs_f = _to_float(platoon.get("ops"))
        if vs_f is None:
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
    pf = _to_float(park_factor)
    if pf is None:
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


# ---------------------------------------------------------------------------
# 5. reverse_platoon — pitcher's vs-LHB / vs-RHB OPS unexpectedly inverted
# ---------------------------------------------------------------------------

_REVERSE_PLATOON_DELTA = 0.080
_REVERSE_PLATOON_MIN_BF = 30
_REVERSE_PLATOON_DATA_BF = 50  # 50+ both sides → confidence "data"


def signal_reverse_platoon(splits: dict | None, pitcher_hand: str) -> dict:
    """Detect reverse platoon (e.g. sweeper-heavy RHP performing worse vs RHB).

    Normal platoon expectations:
        RHP → vs LHB OPS > vs RHB OPS (LHB has platoon advantage)
        LHP → vs RHB OPS > vs LHB OPS

    Reverse fires when the inequality flips by ≥ 0.080 AND both sides BF ≥ 30.
    """
    name = "reverse_platoon"
    if not splits or pitcher_hand not in ("L", "R"):
        return _make(name, False, confidence="small_sample")

    left = splits.get("vs_left") or {}
    right = splits.get("vs_right") or {}

    def _ops_from_side(side: dict) -> float | None:
        """Prefer side.ops; fallback to obp+slg if MLB API stat doesn't ship ops
        (true for /people/{pid}/stats?stats=statSplits — only avg/obp/slg/k/bb)."""
        ops = _to_float(side.get("ops"))
        if ops is not None:
            return ops
        obp = _to_float(side.get("obp"))
        slg = _to_float(side.get("slg"))
        if obp is None or slg is None:
            return None
        return obp + slg

    lhb_ops = _ops_from_side(left)
    rhb_ops = _ops_from_side(right)
    lhb_bf = left.get("bf") or 0
    rhb_bf = right.get("bf") or 0

    if lhb_ops is None or rhb_ops is None:
        return _make(name, False, confidence="small_sample")
    if lhb_bf < _REVERSE_PLATOON_MIN_BF or rhb_bf < _REVERSE_PLATOON_MIN_BF:
        return _make(name, False, confidence="small_sample",
                     details={"lhb_bf": lhb_bf, "rhb_bf": rhb_bf})

    # For RHP: reverse = vs RHB > vs LHB
    # For LHP: reverse = vs LHB > vs RHB
    if pitcher_hand == "R":
        delta = rhb_ops - lhb_ops  # positive = reverse
        better_side, better_ops, worse_side, worse_ops = "RHB", rhb_ops, "LHB", lhb_ops
    else:
        delta = lhb_ops - rhb_ops
        better_side, better_ops, worse_side, worse_ops = "LHB", lhb_ops, "RHB", rhb_ops

    if delta < _REVERSE_PLATOON_DELTA:
        return _make(name, False, value=delta)

    confidence = "data" if (lhb_bf >= _REVERSE_PLATOON_DATA_BF and rhb_bf >= _REVERSE_PLATOON_DATA_BF) else "heuristic"
    severity = "high" if delta >= 0.200 else "medium"
    label = (
        f"reverse platoon Δ +{delta:.3f}（vs {better_side} OPS {better_ops:.3f} > "
        f"vs {worse_side} OPS {worse_ops:.3f}）— {pitcher_hand}HP 對非預期手別反而吃虧"
    )
    return _make(
        name, True, value=round(delta, 3), severity=severity, label=label,
        details={
            "pitcher_hand": pitcher_hand,
            "vs_lhb_ops": lhb_ops, "vs_lhb_bf": lhb_bf,
            "vs_rhb_ops": rhb_ops, "vs_rhb_bf": rhb_bf,
        },
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# 6. chain_break — largest adjacent OPS drop in batting order
# ---------------------------------------------------------------------------

_CHAIN_BREAK_THRESHOLD = 0.150


def signal_chain_break(core_lineup: list | None) -> dict:
    """Find largest adjacent OPS drop in batting order. Fires when ≥ 0.150.

    Caller's order is respected (does NOT re-sort). For official lineups
    the order is 1-9 batting order; for projected lineups it's PA-descending.
    """
    name = "chain_break"
    if not core_lineup or len(core_lineup) < 5:
        return _make(name, False, confidence="small_sample")

    # Pull OPS from each batter in caller-supplied order
    ops_pairs = []  # (position, ops, name)
    for i, b in enumerate(core_lineup):
        ops_f = _to_float(b.get("ops"))
        if ops_f is None:
            continue
        ops_pairs.append((i + 1, ops_f, b.get("name", "?")))

    if len(ops_pairs) < 5:
        return _make(name, False, confidence="small_sample")

    # Find max adjacent drop (positive = previous slot was better)
    max_drop = 0.0
    drop_pos = None
    for i in range(len(ops_pairs) - 1):
        drop = ops_pairs[i][1] - ops_pairs[i + 1][1]
        if drop > max_drop:
            max_drop = drop
            drop_pos = (ops_pairs[i][0], ops_pairs[i + 1][0])

    if max_drop < _CHAIN_BREAK_THRESHOLD or drop_pos is None:
        return _make(name, False, value=max_drop)

    severity = "high" if max_drop >= 0.300 else "medium"
    label = (
        f"chain breaks at #{drop_pos[0]}-{drop_pos[1]}：OPS 落差 {max_drop:.3f}"
    )
    return _make(
        name, True, value=round(max_drop, 3), severity=severity, label=label,
        details={"drop_position": drop_pos, "ops_drop": round(max_drop, 3)},
    )


# ---------------------------------------------------------------------------
# 7. pitch_mix_concentration — max usage % (NOT HHI; bimodal-safe)
# ---------------------------------------------------------------------------

_MIX_SINGLE_PITCH = 45.0   # ≥ 45% of one pitch → "single-pitch dependent"
_MIX_BALANCED_MAX = 25.0   # max < 25% → "balanced (4+ pitches)"


def signal_pitch_mix_concentration(pitch_types: dict | None) -> dict:
    """Surface single-pitch dependence (max ≥ 45%) or balanced 4+ pitches (max < 25%).

    Avoid HHI — bimodal across pitcher types makes raw HHI confusing. Max
    pitch usage is interpretable and aligns with how analysts talk.
    """
    name = "pitch_mix_concentration"
    if not pitch_types:
        return _make(name, False, confidence="small_sample")
    max_usage = max(pitch_types.values())
    if max_usage >= _MIX_SINGLE_PITCH:
        return _make(
            name, True, value=max_usage, severity="medium",
            label=f"single-pitch dependent：主球種使用率 {max_usage:.1f}%（≥{_MIX_SINGLE_PITCH}%）",
            details={"max_usage": max_usage, "pitch_types": pitch_types},
        )
    if max_usage < _MIX_BALANCED_MAX:
        return _make(
            name, True, value=max_usage, severity="low",
            label=f"balanced 4+ pitches：最高球種僅 {max_usage:.1f}%（<{_MIX_BALANCED_MAX}%）",
            details={"max_usage": max_usage, "pitch_types": pitch_types},
        )
    return _make(name, False, value=max_usage)


# ---------------------------------------------------------------------------
# 8. core_il_count — wraps merged.{side}_core_bullpen_il_count to 1/2/3+ ladder
# ---------------------------------------------------------------------------

def signal_core_il_count(count: int | None, side: str) -> dict:
    """Map count to matchup-factors.md §牛棚傷兵累計效應 ladder.

        0 → no fire
        1 → 🟠 medium
        2 → 🔴 high
        3+ → 🔴🔴 high (extreme)
    """
    name = "core_il_count"
    if count is None:
        return _make(name, False, confidence="small_sample")
    if count <= 0:
        return _make(name, False, value=count)
    # Dossier renderer prepends side from signal["side"]; do not duplicate it
    # in the label or output reads "AWAY AWAY 牛棚 core IL ×1".
    if count == 1:
        return _make(
            name, True, value=count, severity="medium",
            label="牛棚 core IL ×1：🟠 中高（後段防守變薄）",
            details={"side": side, "count": count},
        )
    if count == 2:
        return _make(
            name, True, value=count, severity="high",
            label="牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）",
            details={"side": side, "count": count},
        )
    return _make(
        name, True, value=count, severity="high",
        label=f"牛棚 core IL ×{count}：🔴🔴 極高（牛棚崩盤級）",
        details={"side": side, "count": count},
    )


# ---------------------------------------------------------------------------
# compute_all_signals — top-level aggregator
# ---------------------------------------------------------------------------

def _tag(signal_dict: dict, side: str) -> dict:
    """Attach a side annotation to a signal dict (HOME / AWAY / GAME)."""
    signal_dict["side"] = side
    return signal_dict


def compute_all_signals(bundle: dict | None) -> dict:
    """Run every signal over the bundle and return aggregated results.

    Args:
        bundle: Output of prepare_game._load_bundle (keys: home_pitcher,
            away_pitcher, home_lineup, away_lineup, merged, ...). Missing
            or partial bundle returns the schema with all signals fired=False.

    Returns:
        {
            "signals": list[dict],   # every computed signal (with side annotation)
            "fired_count": int,      # count where fired=True
        }
    """
    bundle = bundle or {}
    home_p = bundle.get("home_pitcher") or {}
    away_p = bundle.get("away_pitcher") or {}
    home_l = bundle.get("home_lineup") or {}
    away_l = bundle.get("away_lineup") or {}
    merged = bundle.get("merged") or {}

    signals: list[dict] = []

    # Per-pitcher signals (tier_mismatch, reverse_platoon, pitch_mix_concentration)
    for side, p in (("HOME", home_p), ("AWAY", away_p)):
        signals.append(_tag(signal_tier_mismatch(p.get("tier_gap")), side))
        signals.append(_tag(
            signal_reverse_platoon(p.get("platoon_splits"), p.get("pitch_hand")),
            side,
        ))
        statcast = p.get("statcast") or {}
        signals.append(_tag(
            signal_pitch_mix_concentration(statcast.get("pitch_types")),
            side,
        ))

    # Per-lineup signals (heat_vs_babip, platoon_advantage, chain_break).
    # platoon_advantage uses opposing pitcher's hand so the lineup is matched
    # against today's opponent rather than aggregate platoon splits.
    for side, l, opp_p in (
        ("HOME", home_l, away_p),
        ("AWAY", away_l, home_p),
    ):
        signals.append(_tag(
            signal_heat_vs_babip(l.get("recent_heat"), l.get("last7_babip")),
            side,
        ))
        signals.append(_tag(
            signal_platoon_advantage(l.get("lineup") or [], opp_p.get("pitch_hand", "R")),
            side,
        ))
        signals.append(_tag(signal_chain_break(l.get("lineup") or []), side))

    # Per-side bullpen IL count
    for side in ("HOME", "AWAY"):
        count = merged.get(f"{side.lower()}_core_bullpen_il_count")
        signals.append(_tag(signal_core_il_count(count, side), side))

    # Game-level (single instance)
    signals.append(_tag(signal_strong_park(merged.get("park_factor")), "GAME"))

    fired_count = sum(1 for s in signals if s["fired"])
    return {"signals": signals, "fired_count": fired_count}


def signals_for_bundle(bundle: dict) -> dict:
    """Return cached `bundle["signals"]` if present, else compute_all_signals and cache.

    Cleanup #7: dossier (`_render_signal_summary`) and summary (`_render_extra_signals`)
    both fan out the same signal computation over the same bundle. Computing once
    and stashing under `bundle["signals"]` cuts the duplicate work.

    The cache lives on the caller's bundle dict — prepare_game already shares one
    bundle across step_f / step_g, so the second renderer hits the cache. Callers
    that hand-build bundles (tests, stand-alone CLIs) can skip pre-population and
    let this helper do it lazily.
    """
    cached = bundle.get("signals")
    if cached is not None:
        return cached
    result = compute_all_signals(bundle)
    bundle["signals"] = result
    return result
