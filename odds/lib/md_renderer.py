"""渲染 per-ET-date Markdown 報告。

v2: 所有 tier 一律使用 _render_detailed (含 7 欄 timeline 表格 + headline + flags)。
末段永遠 append AI 解讀說明 footnote (即使 reports 為空)。
"""
from __future__ import annotations

from movement import GameMovementReport, FieldMovement, _abbr
from odds_math import no_vig_two_way


# ── 入口 ──────────────────────────────────────────────────────────────────────

def render(
    et_date: str,
    snapshot_count: int,
    snapshot_times_et: list[str],
    reports: list[GameMovementReport],
    rendered_at: str,
) -> str:
    """產生完整 md 字串。

    snapshot_times_et 元素格式為 "MM-DD HH:MM"（含日期前綴），cover line 內若連續同日
    會壓縮成 "MM-DD HH:MM / HH:MM / ..." 以節省版面。
    """
    out: list[str] = []
    out.append(f"# Smart Money Tracker — {et_date} (ET)\n")
    out.append(f"_最新更新：{rendered_at}_  ")
    out.append("_資料來源：Pinnacle (The Odds API)_  ")
    snap_label = _compress_snapshot_times(snapshot_times_et)
    out.append(f"_涵蓋 snapshot：{snapshot_count} 份({snap_label} ET)_\n")

    if not reports:
        out.append("> 此 ET 日期目前無 Pinnacle 快照可分析。\n")
        out.append(_render_ai_footnote())
        return "\n".join(out)

    by_tier: dict[str, list[GameMovementReport]] = {
        "major": [], "significant": [], "watch": [], "quiet": []
    }
    for r in reports:
        by_tier.setdefault(r.tier, []).append(r)

    tier_headers = {
        "major":       "## 🔥 Major (≥5pp)\n",
        "significant": "## 🟡 Significant (3-5pp)\n",
        "watch":       "## 🔵 Watch (1-3pp)\n",
        "quiet":       "## ⚪ Quiet\n",
    }
    for tier_key in ("major", "significant", "watch", "quiet"):
        if by_tier[tier_key]:
            out.append(tier_headers[tier_key])
            for r in by_tier[tier_key]:
                out.append(_render_detailed(r))
                out.append("")

    out.append(_render_anchor_notes(reports))
    out.append(_render_ai_footnote())
    return "\n".join(out)


def _compress_snapshot_times(times: list[str]) -> str:
    """["04-29 09:24", "04-29 12:00", "04-29 15:00"] → "04-29 09:24 / 12:00 / 15:00"。

    僅在所有元素同日（前 5 字元相同）時壓縮；不同日則保留完整格式。
    """
    if not times:
        return "無"
    if len(times) == 1:
        return times[0]
    first_date = times[0][:5]
    if all(t[:5] == first_date for t in times):
        tail = " / ".join(t[6:] for t in times[1:])   # 跳過 "MM-DD " (6 chars)
        return f"{times[0]} / {tail}"
    return " / ".join(times)


# ── 場次詳細 (所有 tier 一律使用此渲染) ───────────────────────────────────────

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

    # Total point summary (僅 point 有變化時)
    total_point = next((f for f in r.fields if f.field == "total_point"), None)
    if total_point and total_point.delta_pp != 0:
        cross_marker = " ⚠️" if any("Total" in c and "跨越" in c for c in r.key_number_crosses) else ""
        lines.append(f"- **Total**：{total_point.anchor_value} → {total_point.latest_value}{cross_marker}")

    # 7 欄時間軸表格
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


# ── 7 欄時間軸表格 ────────────────────────────────────────────────────────────

def _render_time_series_table(r: GameMovementReport) -> list[str]:
    home_abbr = _abbr(r.home)
    away_abbr = _abbr(r.away)
    header = (
        f"| ET 時間 | {home_abbr} ML | {away_abbr} ML | Over | Under | "
        f"RL {home_abbr} | RL {away_abbr} |"
    )
    sep = "|---|---|---|---|---|---|---|"
    rows = [header, sep]

    for rec in r.timeline:
        ml_h = rec.pinnacle.get("ml", {}).get(r.home, {}) or {}
        ml_a = rec.pinnacle.get("ml", {}).get(r.away, {}) or {}
        ou_o = rec.pinnacle.get("ou", {}).get("Over", {}) or {}
        ou_u = rec.pinnacle.get("ou", {}).get("Under", {}) or {}
        rl_h = rec.pinnacle.get("rl", {}).get(r.home, {}) or {}
        rl_a = rec.pinnacle.get("rl", {}).get(r.away, {}) or {}

        ml_h_cell = _fmt_odds_with_imp(ml_h, ml_a)
        ml_a_cell = _fmt_odds_with_imp(ml_a, ml_h)
        over_cell = _fmt_ou_cell(ou_o, ou_u)
        under_cell = _fmt_ou_cell(ou_u, ou_o)
        rl_h_cell = _fmt_odds_with_imp(rl_h, rl_a)
        rl_a_cell = _fmt_odds_with_imp(rl_a, rl_h)

        rows.append(
            f"| {rec.snapshot_time_et_label} | {ml_h_cell} | {ml_a_cell} | "
            f"{over_cell} | {under_cell} | {rl_h_cell} | {rl_a_cell} |"
        )

    return rows


def _fmt_odds_with_imp(d: dict, paired: dict | None = None) -> str:
    """ML / RL cell：'<odds> (<raw%> / <no-vig%>)'；缺資料回 '?'。"""
    odds = d.get("odds")
    imp = d.get("implied_pct")
    if not isinstance(odds, (int, float)):
        return "?"
    if isinstance(imp, (int, float)):
        nv = _resolve_no_vig(d, paired)
        if nv is not None:
            return f"{odds:.2f} ({imp:.1f}% / {nv:.1f}%)"
        return f"{odds:.2f} ({imp:.1f}%)"
    return f"{odds:.2f}"


def _fmt_ou_cell(d: dict, paired: dict | None = None) -> str:
    """Over / Under cell：'<odds> (<raw%> / <no-vig%>) @ <point>'；point in {7,9,11} 加 ⚠️。"""
    odds = d.get("odds")
    imp = d.get("implied_pct")
    point = d.get("point")
    if not isinstance(odds, (int, float)) or point is None:
        return "?"
    cell = f"{odds:.2f}"
    if isinstance(imp, (int, float)):
        nv = _resolve_no_vig(d, paired)
        if nv is not None:
            cell += f" ({imp:.1f}% / {nv:.1f}%)"
        else:
            cell += f" ({imp:.1f}%)"
    cell += f" @ {point}"
    try:
        if float(point) in (7.0, 9.0, 11.0):
            cell += " ⚠️"
    except (TypeError, ValueError):
        pass
    return cell


def _resolve_no_vig(d: dict, paired: dict | None) -> float | None:
    """讀本側 no_vig_pct（新 snapshot）；無則用本側+對邊 raw 即時算（舊 snapshot fallback）。"""
    nv = d.get("no_vig_pct")
    if isinstance(nv, (int, float)):
        return float(nv)
    if not paired:
        return None
    raw_self = d.get("implied_pct")
    raw_other = paired.get("implied_pct")
    fair, _ = no_vig_two_way(raw_self, raw_other)
    return fair


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
    """挑 |delta_pp| 最大的 pp-field (避開 total_point — 那是 runs，不可比)。"""
    candidates = [f for f in fields if f.field != "total_point"]
    if not candidates:
        return None
    return max(candidates, key=lambda f: abs(f.delta_pp))


def _collect_flags(r: GameMovementReport) -> list[str]:
    flags: list[str] = []
    headline = _pick_headline_field(r.fields)
    if headline and abs(headline.delta_pp) >= 1.0:
        flags.append(f"{headline.direction_label}({r.tier})")
    flags.extend(r.key_number_crosses)
    if r.tier_downgraded:
        flags.append("薄盤降級")
    return flags


def _is_cross_day_anchor(report: GameMovementReport) -> bool:
    """anchor (timeline[0]) 的 snapshot 日期是否與 game_date_et 不同。"""
    if not report.timeline:
        return False
    anchor = report.timeline[0]
    return anchor.snapshot_time_et.strftime("%Y-%m-%d") != anchor.game_date_et


def _render_anchor_notes(reports: list[GameMovementReport]) -> str:
    lines = ["## ℹ️ Anchor Notes\n"]
    ages = sorted({r.anchor_age_hours for r in reports})
    if ages:
        lines.append(f"- Anchor 窗口跨度：{min(ages)}h ~ {max(ages)}h")
    cross_day = [r for r in reports if _is_cross_day_anchor(r)]
    if cross_day:
        shown = cross_day[:5]
        names = "、".join(f"{r.away} @ {r.home}" for r in shown)
        more = f"，… 等共 {len(cross_day)} 場" if len(cross_day) > len(shown) else ""
        lines.append(f"- 含跨 ET 日 anchor：{len(cross_day)} 場（{names}{more}）")
    thin_games = [r for r in reports if r.is_thin_market]
    if thin_games:
        names = "、".join(f"{r.away} @ {r.home}" for r in thin_games)
        lines.append(f"- 薄盤場次(< 4h)：{names}")
    just_appeared = [r for r in reports if r.snapshot_count <= 1]
    if just_appeared:
        names = "、".join(f"{r.away} @ {r.home}" for r in just_appeared)
        lines.append(f"- 僅 1 份 snapshot 可用(無累積位移)：{names}")
    return "\n".join(lines)


# ── AI footnote (v2 新增) ─────────────────────────────────────────────────────

def _render_ai_footnote() -> str:
    """末段附給 AI 消費者的解讀規則。即使 reports 為空也輸出。"""
    return (
        "\n---\n\n"
        "## 解讀說明(給 AI)\n\n"
        "- **百分比 (raw / no-vig)**：cell 顯示兩個值，例 `1.91 (52.4% / 50.9%)`。"
        "前者是 Pinnacle raw implied(含 vig，雙邊合計常 102-104%)；"
        "後者是 vig 剝除後的 fair %(`p_self / (p_self + p_other) * 100`，雙邊合計 = 100)。\n"
        "- **Tier 基於 no-vig delta**：取 ML/RL/Total juice 6 個 no-vig pp-fields 的 max |delta|。"
        "raw 顯示供對照，不參與 tier 計算 — 因為 raw delta 無法區分『莊家整體調 juice』(雙邊都 +1pp，無方向訊號)"
        "與『單邊被打』(home +2pp、away 不動，sharp tell)，no-vig 化後兩者立判。\n"
        "- **Tier 門檻**：\n"
        "  - 🔥 Major ≥ 5pp\n"
        "  - 🟡 Significant ≥ 3pp\n"
        "  - 🔵 Watch ≥ 1pp\n"
        "  - ⚪ Quiet < 1pp\n"
        "- **Total point 推 tier**：|Δpoint| ≥ 1.0 runs → 至少 Significant；≥ 0.5 → 至少 Watch\n"
        "- **Key numbers (Total)**：7 / 9 / 11 — 跨越時 cell 標 ⚠️\n"
        "- **RL price flip**：home 或 away decimal odds 跨越 2.0(熱門 ↔ 冷門翻轉)，列為 flag\n"
        "- **薄盤**：latest snapshot 距開球 < 4h → tier 自動降一檔並標 [薄盤]；訊號可能被晚場閉盤動作污染，可信度較低\n"
        "- **direction_label `→ TEAM +Xpp`**：no-vig latest - no-vig anchor 的 pp 差異，方向偏向 TEAM\n"
        "- **資料來源**：Pinnacle (The Odds API)，單一 book；不偵測 RLM / steam move(需多 book + 公眾下注 % 資料)\n"
        "- **Anchor**：timeline 中最早含本場的 snapshot；可跨 ET 日（看 timeline 首列 MM-DD 前綴判斷實際觀察期）\n"
        "- **時區**：所有顯示時間皆為 ET (UTC-4，MLB 球季 EDT)；snapshot 時間 cell 含 MM-DD 前綴以區分跨日\n"
    )
