"""渲染 per-ET-date Markdown 報告。"""
from __future__ import annotations

from typing import Iterable

from movement import GameMovementReport, FieldMovement, _abbr


# ── 入口 ──────────────────────────────────────────────────────────────────────

def render(
    et_date: str,
    snapshot_count: int,
    snapshot_times_et: list[str],
    reports: list[GameMovementReport],
    rendered_at_et: str,
) -> str:
    """產生完整 md 字串。"""
    out: list[str] = []
    out.append(f"# Smart Money Tracker — {et_date} (ET)\n")
    out.append(f"_最新更新：{rendered_at_et}_  ")
    out.append("_資料來源：Pinnacle (The Odds API)_  ")
    snap_label = " / ".join(snapshot_times_et) if snapshot_times_et else "無"
    out.append(f"_涵蓋 snapshot：{snapshot_count} 份({snap_label} ET)_\n")

    if not reports:
        out.append("> 此 ET 日期目前無 Pinnacle 快照可分析。\n")
        return "\n".join(out)

    by_tier: dict[str, list[GameMovementReport]] = {"major": [], "significant": [], "watch": [], "quiet": []}
    for r in reports:
        by_tier.setdefault(r.tier, []).append(r)

    if by_tier["major"]:
        out.append("## 🔥 Major (≥5pp)\n")
        for r in by_tier["major"]:
            out.append(_render_detailed(r))
            out.append("")

    if by_tier["significant"]:
        out.append("## 🟡 Significant (3-5pp)\n")
        for r in by_tier["significant"]:
            out.append(_render_detailed(r))
            out.append("")

    if by_tier["watch"]:
        out.append("## 🔵 Watch (1-3pp)\n")
        for r in by_tier["watch"]:
            out.append(_render_compact(r))
        out.append("")

    if by_tier["quiet"]:
        out.append("## ⚪ Quiet\n")
        for r in by_tier["quiet"]:
            out.append(_render_quiet_line(r))
        out.append("")

    out.append(_render_anchor_notes(reports))
    return "\n".join(out)


# ── 主要場（major / significant）──────────────────────────────────────────────

def _render_detailed(r: GameMovementReport) -> str:
    lines: list[str] = []
    title = f"### {r.away} @ {r.home} — 開球 {r.commence_et}({_format_hours_to_game(r.hours_to_game)})"
    if r.is_thin_market:
        title += " **[薄盤]**"
    if r.tier_downgraded:
        title += "(已降一檔)"
    lines.append(title)

    headline_field = _pick_headline_field(r.fields)
    if headline_field is not None:
        prefix = _headline_prefix(headline_field.field)
        lines.append(f"- **{prefix}**({r.anchor_age_hours}h 窗口)：{headline_field.direction_label}")

    # Total 變化
    total_point = next((f for f in r.fields if f.field == "total_point"), None)
    if total_point and total_point.delta_pp != 0:
        cross_marker = " ⚠️" if any("Total" in c and "跨越" in c for c in r.key_number_crosses) else ""
        lines.append(f"- **Total**：{total_point.anchor_value} → {total_point.latest_value}{cross_marker}")

    # 時間軸表格
    if r.timeline:
        lines.append("- **時間軸**：")
        lines.append("")
        lines.extend(_render_time_series_table(r))
        lines.append("")

    # Flags
    flags = _collect_flags(r)
    if flags:
        lines.append("- **Flags**：" + "、".join(flags))

    return "\n".join(lines)


def _render_compact(r: GameMovementReport) -> str:
    headline = _pick_headline_field(r.fields)
    label = headline.direction_label if headline else "—"
    suffix = " **[薄盤]**" if r.is_thin_market else ""
    return f"- {r.away} @ {r.home} — 開球 {r.commence_et}({_format_hours_to_game(r.hours_to_game)}){suffix}：{label}"


def _render_quiet_line(r: GameMovementReport) -> str:
    suffix = " **[薄盤]**" if r.is_thin_market else ""
    return f"- {r.away} @ {r.home} — 開球 {r.commence_et}({_format_hours_to_game(r.hours_to_game)}){suffix}"


# ── 時間軸表格 ────────────────────────────────────────────────────────────────

def _render_time_series_table(r: GameMovementReport) -> list[str]:
    home_abbr = _abbr(r.home)
    away_abbr = _abbr(r.away)
    header = f"| ET 時間 | {home_abbr} ML | {away_abbr} ML | Total | RL {home_abbr} | RL {away_abbr} |"
    sep = "|---|---|---|---|---|---|"
    rows = [header, sep]

    for rec in r.timeline:
        ml_h = rec.pinnacle["ml"].get(r.home, {})
        ml_a = rec.pinnacle["ml"].get(r.away, {})
        ou_o = rec.pinnacle["ou"].get("Over", {})
        rl_h = rec.pinnacle["rl"].get(r.home, {})
        rl_a = rec.pinnacle["rl"].get(r.away, {})

        ml_h_cell = _fmt_odds_with_imp(ml_h)
        ml_a_cell = _fmt_odds_with_imp(ml_a)
        total_cell = f"{ou_o.get('point', '?')}"
        # 在跨 key 那一行標 ⚠️
        for k in (7, 9, 11):
            if ou_o.get("point") == k:
                total_cell = f"{ou_o.get('point')} ⚠️"
                break
        rl_h_cell = f"{rl_h.get('odds', '?'):.2f}" if isinstance(rl_h.get("odds"), (int, float)) else "?"
        rl_a_cell = f"{rl_a.get('odds', '?'):.2f}" if isinstance(rl_a.get("odds"), (int, float)) else "?"

        rows.append(
            f"| {rec.snapshot_time_et_label} | {ml_h_cell} | {ml_a_cell} | {total_cell} | {rl_h_cell} | {rl_a_cell} |"
        )

    return rows


def _fmt_odds_with_imp(d: dict) -> str:
    odds = d.get("odds")
    imp = d.get("implied_pct")
    if not isinstance(odds, (int, float)):
        return "?"
    if isinstance(imp, (int, float)):
        return f"{odds:.2f} ({imp:.1f}%)"
    return f"{odds:.2f}"


# ── Flags / headline pick ─────────────────────────────────────────────────────

def _format_hours_to_game(h: float) -> str:
    """h >= 0 → 'Xh 後'；h < 0 → '已開球'。"""
    if h >= 0:
        return f"{h}h 後"
    return "已開球"


def _headline_prefix(field_name: str) -> str:
    if field_name.startswith("ml_"):
        return "ML 累積位移"
    if field_name.startswith("rl_"):
        return "RL 累積位移"
    if field_name.startswith("total_juice_"):
        return "Total juice 位移"
    return "位移"


def _pick_headline_field(fields: list[FieldMovement]) -> FieldMovement | None:
    """挑 |delta_pp| 最大的 pp-field（避開 total_point — 那是 runs，不可比）。"""
    candidates = [f for f in fields if f.field != "total_point"]
    if not candidates:
        return None
    return max(candidates, key=lambda f: abs(f.delta_pp))


def _collect_flags(r: GameMovementReport) -> list[str]:
    flags: list[str] = []
    # Headline tier label
    headline = _pick_headline_field(r.fields)
    if headline and abs(headline.delta_pp) >= 1.0:
        flags.append(f"{headline.direction_label}({r.tier})")
    # Key number / RL flip
    flags.extend(r.key_number_crosses)
    if r.tier_downgraded:
        flags.append("薄盤降級")
    return flags


def _render_anchor_notes(reports: list[GameMovementReport]) -> str:
    lines = ["## ℹ️ Anchor Notes\n"]
    # 統計 anchor age 分布
    ages = sorted({r.anchor_age_hours for r in reports})
    if ages:
        lines.append(f"- Anchor 窗口跨度：{min(ages)}h ~ {max(ages)}h(同 ET 日 0 天回溯)")
    thin_games = [r for r in reports if r.is_thin_market]
    if thin_games:
        names = "、".join(f"{r.away} @ {r.home}" for r in thin_games)
        lines.append(f"- 薄盤場次(< 4h)：{names}")
    just_appeared = [r for r in reports if r.snapshot_count <= 1]
    if just_appeared:
        names = "、".join(f"{r.away} @ {r.home}" for r in just_appeared)
        lines.append(f"- 僅 1 份 snapshot 可用(無累積位移)：{names}")
    return "\n".join(lines)
