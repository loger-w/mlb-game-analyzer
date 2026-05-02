"""Dossier renderer：將各 JSON 整合為 ~250 行 markdown，作 AI 主入口。

設計原則：
- 純函式：輸入 dict bundle，輸出 markdown str
- 無 side effect、無 I/O：所有 render 函式（含 render_dossier）只回傳字串；檔案寫入由 prepare_game.py 的 step_f 負責
- game_dir 參數僅用於將檔案路徑作為 cross-reference 嵌入 markdown 內容
- 子函式逐節獨立可測

Bundle keys:
  game_data, home_roster, away_roster, home_pitcher, away_pitcher,
  home_lineup, away_lineup, merged
"""

from __future__ import annotations

import json
from pathlib import Path

from fetch_game_data import format_game_times

from park_factors_lib import get_park_factor

PA_FLOOR = 30  # Top 5 候選池下限（PA ≥ 30）


# ---------------------------------------------------------------------------
# Helpers exported from Task 8a
# ---------------------------------------------------------------------------

def _il_names_from_roster(roster: dict | None) -> set[str]:
    """從 roster_checker.py 輸出取出 IL'd 球員名字。"""
    if not roster:
        return set()
    return {p.get("name") for p in roster.get("injured_list", []) if p.get("name")}


def il_pitcher_count(roster: dict | None) -> int:
    """從 roster_checker.py 輸出計算 IL 上的投手人數（含先發 / 長傷 60-day）。

    注意：API 不直接給 reliever role，所以這個 count 是「投手 IL 全體」，
    不是「核心牛棚 IL」。matchup-factors.md §牛棚傷兵累計效應 的「核心 IL」
    估計交由 summary.md 的 AI placeholder 補完。
    """
    if not roster:
        return 0
    return sum(
        1 for p in roster.get("injured_list", []) or []
        if (p.get("position") or "").lower() in ("pitcher", "p")
    )


def select_top5_vs_pitcher(lineup: dict | None, il_names: set[str]) -> list[dict]:
    """從 lineup（lineup_analyzer.py 輸出）選 Top 5 vs 對方先發。

    規則：
    - active && PA ≥ 30 && !IL'd
    - 按 PA 降序
    - 最多 5 人，候選池 < 5 就少
    """
    candidates = [
        p for p in (lineup or {}).get("lineup", []) or []
        if (p.get("pa") or 0) >= PA_FLOOR and p.get("name") not in il_names
    ]
    candidates.sort(key=lambda p: p.get("pa") or 0, reverse=True)
    return candidates[:5]


def find_last7_top1_outside_pa_top5(
    lineup: dict | None,
    pa_top5_names: set[str],
    il_names: set[str],
) -> dict | None:
    """找出 last7 OPS top1 球員，若不在 PA top 5 內則回傳；否則 None。

    候選池套用同樣的 IL 過濾與 PA ≥ 30。
    """
    candidates = [
        p for p in (lineup or {}).get("lineup", []) or []
        if (p.get("pa") or 0) >= PA_FLOOR and p.get("name") not in il_names
    ]
    candidates_with_last7 = [p for p in candidates if p.get("last7_ops") is not None]
    if not candidates_with_last7:
        return None
    top1 = max(candidates_with_last7, key=lambda p: p["last7_ops"])
    if top1.get("name") in pa_top5_names:
        return None
    return top1


# ---------------------------------------------------------------------------
# Formatting helpers (private)
# ---------------------------------------------------------------------------

def _v(value, decimals: int = 2, default: str = "—") -> str:
    """Format numeric value; None → default placeholder."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if decimals == 0:
            return f"{value:.0f}"
        return f"{value:.{decimals}f}"
    return str(value)


def _ops(value, default: str = "—") -> str:
    """Format OPS-style (.NNN) from float like 0.804 → '.804'."""
    if value is None:
        return default
    try:
        f = float(value)
        return f"{f:.3f}".lstrip("0") or ".000"
    except (TypeError, ValueError):
        return str(value)


def _streak_str(streak) -> str:
    """Convert streak int → '+5' / '−3' / '0'."""
    if streak is None:
        return "—"
    try:
        s = int(streak)
    except (TypeError, ValueError):
        return str(streak)
    if s > 0:
        return f"+{s}"
    if s < 0:
        return f"−{abs(s)}"
    return "0"


def _trend_arrow(recent_val, window30_val) -> str:
    """Compute trend arrow: |Δ| ≥ 0.5 → ↑/↓, else →."""
    try:
        delta = float(recent_val) - float(window30_val)
    except (TypeError, ValueError):
        return "→"
    if delta >= 0.5:
        return "↑"
    if delta <= -0.5:
        return "↓"
    return "→"


def _pitch_types_top3(statcast: dict | None) -> str:
    """Return top-3 pitch types as 'FF 27.8 / FC 27.5 / CH 17.9'."""
    if not statcast:
        return "—"
    pt = statcast.get("pitch_types") or {}
    if not pt:
        return "—"
    sorted_types = sorted(pt.items(), key=lambda x: x[1], reverse=True)[:3]
    return " / ".join(f"{k} {v:.1f}" for k, v in sorted_types)


def _recent_er_ip(game_log: list | None, n: int = 3) -> str:
    """Summarise last-N game log as 'ER/IP' totals."""
    if not game_log:
        return "—"
    recent = game_log[-n:]
    total_er = sum(g.get("er") or 0 for g in recent)
    total_ip = sum(g.get("ip") or 0.0 for g in recent)
    return f"{total_er}/{total_ip:.1f}"


def _platoon_slash(splits: dict | None, hand: str) -> str:
    """Return 'avg/obp/slg (BF BF)' for vs_left or vs_right."""
    if not splits:
        return "—"
    key = "vs_left" if hand == "L" else "vs_right"
    side = splits.get(key) or {}
    avg = side.get("avg", "—")
    obp = side.get("obp", "—")
    slg = side.get("slg", "—")
    bf = side.get("bf")
    bf_str = f" ({bf} BF)" if bf is not None else ""
    return f"{avg}/{obp}/{slg}{bf_str}"


def _lineup_vs_hand_ops(player: dict, pitcher_hand: str) -> str:
    """Return vs_rhp or vs_lhp OPS string from lineup player platoon data."""
    platoon = player.get("platoon") or {}
    key = "vs_rhp" if pitcher_hand == "R" else "vs_lhp"
    side = platoon.get(key) or {}
    ops_val = side.get("ops")
    if ops_val is None:
        return "—"
    # ops_val might already be a string like '.703' or a float
    try:
        return f"{float(ops_val):.3f}".lstrip("0") or ".000"
    except (TypeError, ValueError):
        return str(ops_val)


def _last7_ops_from_player(player: dict) -> float | None:
    """Extract last7 OPS float from player dict (last_7.ops or last7_ops)."""
    # lineup_analyzer stores under last_7 key
    last7 = player.get("last_7") or {}
    if isinstance(last7, dict):
        ops = last7.get("ops")
        if ops is not None:
            try:
                return float(ops)
            except (TypeError, ValueError):
                pass
    # fallback: direct last7_ops key (used in tests)
    v = player.get("last7_ops")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return None


def _last7_babip_from_player(player: dict) -> str:
    """Extract last7 BABIP from player dict."""
    last7 = player.get("last_7") or {}
    if isinstance(last7, dict):
        babip = last7.get("babip")
        if babip is not None:
            return str(babip)
    return player.get("last7_babip", "—") or "—"


def _last7_ops_str(player: dict) -> str:
    """Return last7 OPS as formatted string."""
    v = _last7_ops_from_player(player)
    if v is None:
        return "—"
    return f"{v:.3f}".lstrip("0") or ".000"


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_header(bundle: dict) -> list[str]:
    """# Game Dossier — {AWAY} @ {HOME} ({ET-YYYY-MM-DD})"""
    merged = bundle.get("merged") or {}
    meta = merged.get("_meta") or {}
    game_data = bundle.get("game_data") or {}
    game = game_data.get("game") or {}

    away = meta.get("away_team") or game.get("away", {}).get("team", "AWAY")
    home = meta.get("home_team") or game.get("home", {}).get("team", "HOME")
    date_raw = meta.get("game_date") or game.get("date", "")
    official = meta.get("official_date") or game.get("officialDate")
    times = format_game_times(date_raw, official)
    date_str = times.get("et_date", "—")

    return [f"# Game Dossier — {away} @ {home} ({date_str})", ""]


def _render_game_info(bundle: dict) -> list[str]:
    """## 比賽資訊 section."""
    merged = bundle.get("merged") or {}
    meta = merged.get("_meta") or {}
    game_data = bundle.get("game_data") or {}
    game = game_data.get("game") or {}

    date_raw = meta.get("game_date") or game.get("date", "")
    official = meta.get("official_date") or game.get("officialDate")
    times = format_game_times(date_raw, official)
    et_date = times.get("et_date", "—")
    et_time = times.get("et_time", "—")
    venue = meta.get("venue") or game.get("venue", "—")
    status = game.get("status", "—")

    home_sp = meta.get("home_sp") or game.get("home", {}).get("probable_pitcher", "—")
    away_sp = meta.get("away_sp") or game.get("away", {}).get("probable_pitcher", "—")
    home_gs = meta.get("home_sp_starts", "?")
    away_gs = meta.get("away_sp_starts", "?")
    away_team = meta.get("away_team") or game.get("away", {}).get("team", "AWAY")
    home_team = meta.get("home_team") or game.get("home", {}).get("team", "HOME")

    # Get pitcher IDs from pitcher JSON
    home_pitcher = bundle.get("home_pitcher") or {}
    away_pitcher = bundle.get("away_pitcher") or {}
    home_id = home_pitcher.get("mlbam_id", "?")
    away_id = away_pitcher.get("mlbam_id", "?")

    lines = [
        "## 比賽資訊",
        f"- 日期 (ET): {et_date}",
        f"- 開賽: {et_date} {et_time} ET",
        f"- 球場: {venue}",
        f"- 狀態: {status}",
        f"- 先發: {away_sp} (#{away_id}, {away_team}, GS {away_gs}) vs {home_sp} (#{home_id}, {home_team}, GS {home_gs})",
        "",
    ]
    return lines


def _render_record_summary(bundle: dict) -> list[str]:
    """## 戰績速查 table with trend row."""
    game_data = bundle.get("game_data") or {}
    merged = bundle.get("merged") or {}
    meta = merged.get("_meta") or {}

    home_name = meta.get("home_team", "HOME")
    away_name = meta.get("away_team", "AWAY")

    home10 = game_data.get("home_recent") or {}
    away10 = game_data.get("away_recent") or {}
    home30 = game_data.get("home_recent_30") or {}
    away30 = game_data.get("away_recent_30") or {}
    home_s = game_data.get("home_season") or {}
    away_s = game_data.get("away_season") or {}

    def _diff_str(d) -> str:
        if isinstance(d, int) and d > 0:
            return f"+{d}"
        return str(d) if d is not None else "?"

    def _row(label, home_d, away_d, mode="streak") -> str:
        h_rec = home_d.get("record", "?-?")
        h_rs = _v(home_d.get("rs_per_game"), 2)
        h_ra = _v(home_d.get("ra_per_game"), 2)
        h_diff = _diff_str(home_d.get("run_diff"))
        a_rec = away_d.get("record", "?-?")
        a_rs = _v(away_d.get("rs_per_game"), 2)
        a_ra = _v(away_d.get("ra_per_game"), 2)
        a_diff = _diff_str(away_d.get("run_diff"))
        if mode == "streak":
            h_streak = _streak_str(home_d.get("streak"))
            a_streak = _streak_str(away_d.get("streak"))
            h_cell = f"{h_rec} RS {h_rs} / RA {h_ra} / diff {h_diff} / streak {h_streak}"
            a_cell = f"{a_rec} RS {a_rs} / RA {a_ra} / diff {a_diff} / streak {a_streak}"
        elif mode == "season":
            h_games = game_data.get("home_season_games_count", "?")
            a_games = game_data.get("away_season_games_count", "?")
            h_cell = f"{h_rec} ({h_games} 場)"
            a_cell = f"{a_rec} ({a_games} 場)"
        else:
            # 近 30: RS/RA/diff only (no per-game streak field)
            h_cell = f"{h_rec} RS {h_rs} / RA {h_ra} / diff {h_diff}"
            a_cell = f"{a_rec} RS {a_rs} / RA {a_ra} / diff {a_diff}"
        return f"| {label} | {h_cell} | {a_cell} |"

    # Trend row
    h_atk = _trend_arrow(home10.get("rs_per_game"), home30.get("rs_per_game"))
    h_def = _trend_arrow(home30.get("ra_per_game"), home10.get("ra_per_game"))  # RA up = defense worse
    a_atk = _trend_arrow(away10.get("rs_per_game"), away30.get("rs_per_game"))
    a_def = _trend_arrow(away30.get("ra_per_game"), away10.get("ra_per_game"))

    h_rs10 = _v(home10.get("rs_per_game"), 2)
    h_rs30 = _v(home30.get("rs_per_game"), 2)
    h_ra10 = _v(home10.get("ra_per_game"), 2)
    h_ra30 = _v(home30.get("ra_per_game"), 2)
    a_rs10 = _v(away10.get("rs_per_game"), 2)
    a_rs30 = _v(away30.get("rs_per_game"), 2)
    a_ra10 = _v(away10.get("ra_per_game"), 2)
    a_ra30 = _v(away30.get("ra_per_game"), 2)

    h_trend = f"攻{h_atk} (RS {h_rs10} vs {h_rs30}) / 守{h_def} (RA {h_ra10} vs {h_ra30})"
    a_trend = f"攻{a_atk} (RS {a_rs10} vs {a_rs30}) / 守{a_def} (RA {a_ra10} vs {a_ra30})"

    lines = [
        "## 戰績速查",
        f"| 區間 | {home_name} (RS/RA/diff/streak) | {away_name} (RS/RA/diff/streak) |",
        "|------|------|------|",
        _row("近 10", home10, away10, mode="streak"),
        _row("近 30", home30, away30, mode="window30"),
        _row("本季", home_s, away_s, mode="season"),
        f"| 趨勢 | {h_trend} | {a_trend} |",
        "",
    ]
    return lines


def _render_series_context(bundle: dict) -> list[str]:
    """## 系列脈絡 from game_data."""
    game_data = bundle.get("game_data") or {}
    merged = bundle.get("merged") or {}
    meta = merged.get("_meta") or {}

    home_name = meta.get("home_team", "HOME")
    away_name = meta.get("away_team", "AWAY")

    # Series prev game
    series_prev = game_data.get("series_prev")
    # Streak context from home/away recent
    home10 = game_data.get("home_recent") or {}
    away10 = game_data.get("away_recent") or {}
    h_streak = home10.get("streak", 0) or 0
    a_streak = away10.get("streak", 0) or 0
    home_games = home10.get("games") or []
    away_games = away10.get("games") or []

    lines = ["## 系列脈絡", ""]

    # Series context
    if series_prev:
        prev_date = series_prev.get("date", "?")
        h_score = series_prev.get("home_score", "?")
        a_score = series_prev.get("away_score", "?")
        winner = series_prev.get("winner") or "?"
        away_short = away_name.split()[-1] if away_name else "?"
        home_short = home_name.split()[-1] if home_name else "?"
        prev_away = series_prev.get("away", away_name)
        prev_home = series_prev.get("home", home_name)
        lines.append(f"**當前系列賽 ({away_name} @ {home_name})**")
        lines.append(f"- G1 ({prev_date[-5:]}): {home_short} {h_score}–{a_score} {away_short} → {winner.split()[-1]} 勝")
        lines.append("- G2: 本場")
        winner_side = "AWAY" if winner == prev_away else "HOME"
        series_score = f"{home_name} 0-1 {away_name}" if winner_side == "AWAY" else f"{home_name} 1-0 {away_name}"
        lines.append(f"- 系列累計: **{series_score}**")
    else:
        lines.append(f"**當前系列賽 ({away_name} @ {home_name})**")
        lines.append("- 系列賽資料不可用")
    lines.append("")

    # Streak context
    def _streak_context(streak_val: int, games: list, team: str) -> str:
        streak_str = _streak_str(streak_val)
        if not games or streak_val == 0:
            return f"- {team} streak {streak_str}"
        if streak_val < 0:
            # show last N losses
            loss_games = [g for g in games if not g.get("is_winner")][:abs(streak_val)]
            opps = ", ".join(f"{(g.get('opponent') or '?').split()[-1]} ({g.get('date', '?')[-5:]})" for g in loss_games)
            return f"- {team} {streak_str}: 連敗對手 → {opps}"
        else:
            win_games = [g for g in games if g.get("is_winner")][:streak_val]
            opps = ", ".join(f"{(g.get('opponent') or '?').split()[-1]} ({g.get('date', '?')[-5:]})" for g in win_games)
            return f"- {team} {streak_str}: 連勝對手 → {opps}"

    lines.append("**Streak 脈絡**")
    lines.append(_streak_context(h_streak, home_games, home_name))
    lines.append(_streak_context(a_streak, away_games, away_name))
    lines.append("")

    return lines


def _render_pitcher_matchup(bundle: dict) -> list[str]:
    """## 投手對決 table."""
    # Import here to avoid circular import in tests
    try:
        from pitcher_stats import detect_triggers as detect_pitcher_triggers
    except ImportError:
        def detect_pitcher_triggers(data):
            return []

    home_p = bundle.get("home_pitcher") or {}
    away_p = bundle.get("away_pitcher") or {}
    merged = bundle.get("merged") or {}
    meta = merged.get("_meta") or {}

    home_name = home_p.get("name") or meta.get("home_sp", "HOME SP")
    away_name = away_p.get("name") or meta.get("away_sp", "AWAY SP")

    h_tier = home_p.get("tier", "—")
    a_tier = away_p.get("tier", "—")

    h_hand = home_p.get("pitch_hand", "?")
    h_age = home_p.get("age", "?")
    h_age_tag = home_p.get("age_assessment", "")
    a_hand = away_p.get("pitch_hand", "?")
    a_age = away_p.get("age", "?")
    a_age_tag = away_p.get("age_assessment", "")

    h_hand_label = f"{h_hand}HP" if h_hand in ("R", "L") else h_hand
    a_hand_label = f"{a_hand}HP" if a_hand in ("R", "L") else a_hand

    h_season = home_p.get("season") or {}
    a_season = away_p.get("season") or {}
    h_exp = home_p.get("expected") or {}
    a_exp = away_p.get("expected") or {}
    h_stat = home_p.get("statcast") or {}
    a_stat = away_p.get("statcast") or {}
    h_splits = home_p.get("platoon_splits") or {}
    a_splits = away_p.get("platoon_splits") or {}
    h_log = home_p.get("game_log") or []
    a_log = away_p.get("game_log") or []

    # Trigger detection
    h_triggers = detect_pitcher_triggers(home_p)
    a_triggers = detect_pitcher_triggers(away_p)

    def _risk_cell(triggers: list) -> str:
        if not triggers:
            return "—"
        parts = []
        for t in triggers:
            flag = t.get("flag", "?")
            val = t.get("value")
            name = t.get("name", "")
            if name == "ERA-xERA gap" and val is not None:
                parts.append(f"era_xera_delta={val:+.2f}（Flag {flag}）")
            elif name == "Small-sample regression risk" and val is not None:
                parts.append(f"prior_year_regression={val:+.2f}（Flag {flag}, IP<30）")
            elif val is not None:
                parts.append(f"{name}={val}（Flag {flag}）")
            else:
                parts.append(f"{name}（Flag {flag}）")
        return " / ".join(parts)

    h_risk = _risk_cell(h_triggers)
    a_risk = _risk_cell(a_triggers)
    # Show ⚠️ in the row label if either side has a trigger
    risk_row_label = "⚠️ 風險提示" if (h_triggers or a_triggers) else "風險提示"

    lines = [
        "## 投手對決",
        f"| | HOME ({home_name}) | AWAY ({away_name}) |",
        "|---|------|------|",
        f"| Tier (script) | {h_tier} | {a_tier} |",
        f"| Hand / Age | {h_hand_label} / {h_age} {h_age_tag} | {a_hand_label} / {a_age} {a_age_tag} |",
        f"| ERA / xERA | {_v(h_season.get('era'))} / {_v(h_exp.get('xera'))} | {_v(a_season.get('era'))} / {_v(a_exp.get('xera'))} |",
        f"| FIP / xFIP | {_v(h_season.get('fip'))} / {_v(h_season.get('xfip'))} | {_v(a_season.get('fip'))} / {_v(a_season.get('xfip'))} |",
        f"| K-BB% / WHIP | {_v(h_season.get('k_bb_pct'), 1)} / {_v(h_season.get('whip'))} | {_v(a_season.get('k_bb_pct'), 1)} / {_v(a_season.get('whip'))} |",
        f"| velo (avg/max) | {_v(h_stat.get('avg_velo'), 1)} / {_v(h_stat.get('max_velo'), 1)} | {_v(a_stat.get('avg_velo'), 1)} / {_v(a_stat.get('max_velo'), 1)} |",
        f"| whiff% / hard_hit% / barrel% | {_v(h_stat.get('whiff_pct'), 1)} / {_v(h_stat.get('hard_hit_pct'), 1)} / {_v(h_stat.get('barrel_pct'), 1)} | {_v(a_stat.get('whiff_pct'), 1)} / {_v(a_stat.get('hard_hit_pct'), 1)} / {_v(a_stat.get('barrel_pct'), 1)} |",
        f"| 主球種 (top3) | {_pitch_types_top3(h_stat)} | {_pitch_types_top3(a_stat)} |",
        f"| vs LHB (slash) | {_platoon_slash(h_splits, 'L')} | {_platoon_slash(a_splits, 'L')} |",
        f"| vs RHB (slash) | {_platoon_slash(h_splits, 'R')} | {_platoon_slash(a_splits, 'R')} |",
        f"| 近 3 場 ER/IP | {_recent_er_ip(h_log)} | {_recent_er_ip(a_log)} |",
        f"| {risk_row_label} | {h_risk} | {a_risk} |",
        "",
    ]
    return lines


def _source_label(source: str) -> str:
    """Return emoji-prefixed label for lineup_source field."""
    if source == "official":
        return "🟢 official"
    return "🟡 projected（PA 排序近似 — 打線尚未公布）"


def _render_top5_block(
    label: str,
    lineup: dict,
    il_names: set,
    opposing_sp_name: str,
    opposing_hand: str,
) -> list[str]:
    """渲染單側 Top 5 vs 對方先發 sub-block（含 last7 OPS top1 註腳）"""
    pa_top5 = select_top5_vs_pitcher(lineup, il_names)
    pa_top5_names = {p.get("name") for p in pa_top5}
    lines = [f"**{label} vs {opposing_sp_name}（{opposing_hand}HP）**:"]
    lines.append(f"| # | Name | season OPS | vs {opposing_hand}HP OPS | last7 OPS | last7 BABIP | EV95% | Barrel% |")
    lines.append("|---|------|------|------|------|------|------|------|")
    for i, player in enumerate(pa_top5, start=1):
        name = player.get("name", "?")
        s_ops = _ops(player.get("ops"))
        vs_ops = _lineup_vs_hand_ops(player, opposing_hand)
        l7_ops = _last7_ops_str(player)
        l7_babip = _last7_babip_from_player(player)
        ev95 = _v(player.get("ev95pct"), 1)
        barrel = _v(player.get("barrel_pct"), 1)
        lines.append(f"| {i} | {name} | {s_ops} | {vs_ops} | {l7_ops} | {l7_babip} | {ev95} | {barrel} |")
    extra = find_last7_top1_outside_pa_top5(lineup, pa_top5_names, il_names)
    if extra:
        extra_name = extra.get("name", "?")
        extra_ops = _last7_ops_str(extra)
        lines.append(f"> last7 OPS top1（不在 PA top5 內）：{extra_name} (last7 OPS {extra_ops})")
    return lines


def _render_full9_vs_pitcher(
    label: str,
    lineup: dict,
    opposing_sp_name: str,
    opposing_hand: str,
) -> list[str]:
    """渲染 official 打序：9 棒 vs 對方先發 sub-block（按 batting_order 排序）"""
    all_batters = list((lineup or {}).get("lineup", []) or [])
    # Sort by batting_order; players without batting_order go to end
    all_batters.sort(key=lambda p: p.get("batting_order") or 99)
    lines = [f"### {label} 1–9 棒 vs {opposing_sp_name} ({opposing_hand}HP)"]
    lines.append(f"| 棒 | Name | Pos | season OPS | vs {opposing_hand}HP OPS | last7 OPS | EV95% | Barrel% |")
    lines.append("|---|------|------|------|------|------|------|------|")
    for player in all_batters:
        order = player.get("batting_order") or "—"
        name = player.get("name", "?")
        pos = player.get("position", "—") or "—"
        s_ops = _ops(player.get("ops"))
        vs_ops = _lineup_vs_hand_ops(player, opposing_hand)
        l7_ops = _last7_ops_str(player)
        ev95 = _v(player.get("ev95pct"), 1)
        barrel = _v(player.get("barrel_pct"), 1)
        lines.append(f"| {order} | {name} | {pos} | {s_ops} | {vs_ops} | {l7_ops} | {ev95} | {barrel} |")
    return lines


def _render_lineup_overview(bundle: dict) -> list[str]:
    """## 打線 table + Top 5 sub-block."""
    try:
        from lineup_analyzer import detect_triggers as detect_lineup_triggers
    except ImportError:
        def detect_lineup_triggers(data):
            return []

    home_lu = bundle.get("home_lineup") or {}
    away_lu = bundle.get("away_lineup") or {}
    home_roster = bundle.get("home_roster")
    away_roster = bundle.get("away_roster")
    home_p = bundle.get("home_pitcher") or {}
    away_p = bundle.get("away_pitcher") or {}

    h_tier = home_lu.get("tier", "—")
    a_tier = away_lu.get("tier", "—")
    h_heat = home_lu.get("recent_heat", "—")
    a_heat = away_lu.get("recent_heat", "—")

    h_xwoba = _v(home_lu.get("avg_xwoba"), 3)
    h_ops = _ops(home_lu.get("avg_ops"))
    a_xwoba = _v(away_lu.get("avg_xwoba"), 3)
    a_ops = _ops(away_lu.get("avg_ops"))

    h_k = _v(home_lu.get("avg_k_pct"), 1)
    h_bb = _v(home_lu.get("avg_bb_pct"), 1)
    a_k = _v(away_lu.get("avg_k_pct"), 1)
    a_bb = _v(away_lu.get("avg_bb_pct"), 1)

    h_chain = home_lu.get("chain") or {}
    a_chain = away_lu.get("chain") or {}
    h_obp3 = _ops(h_chain.get("obp_top3"))
    h_slg_mid = _ops(h_chain.get("slg_mid"))
    a_obp3 = _ops(a_chain.get("obp_top3"))
    a_slg_mid = _ops(a_chain.get("slg_mid"))

    h_babip7 = _v(home_lu.get("last7_babip"), 3)
    a_babip7 = _v(away_lu.get("last7_babip"), 3)

    # Trigger detection
    h_lu_triggers = detect_lineup_triggers({"last7_babip": home_lu.get("last7_babip")})
    a_lu_triggers = detect_lineup_triggers({"last7_babip": away_lu.get("last7_babip")})

    def _lu_risk_cell(triggers: list) -> str:
        if not triggers:
            return "—"
        parts = []
        for t in triggers:
            flag = t.get("flag", "?")
            val = t.get("value")
            if val is not None:
                parts.append(f"last7 BABIP={val:.3f}（Flag {flag}）")
            else:
                parts.append(f"Flag {flag}")
        return " / ".join(parts)

    h_lu_risk = _lu_risk_cell(h_lu_triggers)
    a_lu_risk = _lu_risk_cell(a_lu_triggers)
    lu_risk_row_label = "⚠️ 風險提示" if (h_lu_triggers or a_lu_triggers) else "風險提示"

    # Read lineup_source for each side (default "projected" for backward compat)
    home_source = home_lu.get("lineup_source", "projected") or "projected"
    away_source = away_lu.get("lineup_source", "projected") or "projected"
    h_source_label = _source_label(home_source)
    a_source_label = _source_label(away_source)

    lines = [
        "## 打線",
        f"- HOME 打線來源：{h_source_label}",
        f"- AWAY 打線來源：{a_source_label}",
        "",
        "| | HOME | AWAY |",
        "|---|------|------|",
        f"| Tier (script) | {h_tier} | {a_tier} |",
        f"| Heat (script) | {h_heat} | {a_heat} |",
        f"| xwOBA / OPS | {h_xwoba} / {h_ops} | {a_xwoba} / {a_ops} |",
        f"| K% / BB% | {h_k} / {h_bb} | {a_k} / {a_bb} |",
        f"| chain OBP top3 / SLG mid | {h_obp3} / {h_slg_mid} | {a_obp3} / {a_slg_mid} |",
        f"| last7 BABIP | {h_babip7} | {a_babip7} |",
        f"| {lu_risk_row_label} | {h_lu_risk} | {a_lu_risk} |",
        "",
    ]

    # HOME vs AWAY pitcher
    away_hand = away_p.get("pitch_hand", "?")
    away_sp_name = away_p.get("name", "AWAY SP")
    home_il = _il_names_from_roster(home_roster)

    # AWAY vs HOME pitcher
    home_hand = home_p.get("pitch_hand", "?")
    home_sp_name = home_p.get("name", "HOME SP")
    away_il = _il_names_from_roster(away_roster)

    # If either side is projected, emit the shared Top 5 section header once
    either_projected = (home_source != "official") or (away_source != "official")
    if either_projected:
        lines.append("### Top 5 vs 對方先發手感（PA 排序，PA ≥ 30，IL'd 排除）")
        lines.append("")

    if home_source == "official":
        lines += _render_full9_vs_pitcher("HOME", home_lu, away_sp_name, away_hand)
    else:
        lines += _render_top5_block("HOME", home_lu, home_il, away_sp_name, away_hand)
    lines.append("")

    if away_source == "official":
        lines += _render_full9_vs_pitcher("AWAY", away_lu, home_sp_name, home_hand)
    else:
        lines += _render_top5_block("AWAY", away_lu, away_il, home_sp_name, home_hand)
    lines.append("")

    return lines


def _render_bullpen_park(bundle: dict) -> list[str]:
    """## 牛棚 / Park table."""
    merged = bundle.get("merged") or {}
    meta = merged.get("_meta") or {}
    home_roster = bundle.get("home_roster") or {}
    away_roster = bundle.get("away_roster") or {}

    h_bullpen_era = _v(merged.get("home_bullpen_era"))
    a_bullpen_era = _v(merged.get("away_bullpen_era"))

    # IL counts from roster
    h_il_list = home_roster.get("injured_list") or []
    a_il_list = away_roster.get("injured_list") or []
    h_il_pitchers = [p for p in h_il_list if p.get("position", "").lower() in ("pitcher", "p")]
    a_il_pitchers = [p for p in a_il_list if p.get("position", "").lower() in ("pitcher", "p")]

    h_il_count = len(h_il_pitchers)
    a_il_count = len(a_il_pitchers)

    # Core IL — list pitcher names with position tag "HL setup" style (role unknown → use status)
    def _core_il_str(il_pitchers: list) -> str:
        if not il_pitchers:
            return "—"
        count = len(il_pitchers)
        # show first 2 names with their status abbreviation
        parts = []
        for p in il_pitchers[:2]:
            name = p.get("name", "?")
            status = p.get("status", "")
            # abbreviate status: 'Injured 15-Day' → 'IL15'
            s_short = status.replace("Injured ", "IL").replace("-Day", "d")
            parts.append(f"{name} ({s_short})")
        suffix = ", ..." if count > 2 else ""
        return f"{count} ({', '.join(parts)}{suffix})"

    h_core_il = _core_il_str(h_il_pitchers)
    a_core_il = _core_il_str(a_il_pitchers)

    # Park factor
    venue = meta.get("venue", "")
    park_factor = merged.get("park_factor")
    pf_str = _v(park_factor, 0) if park_factor is not None else "—"

    # Park note: 透過 park_factors_lib 解 alias（Tropicana → Steinbrenner 等）
    pf_data = get_park_factor(venue)
    hr_pf = pf_data.get("hr_pf")
    if hr_pf is not None and venue:
        delta_pct = hr_pf - 100
        if delta_pct > 0:
            park_note = f"{venue} HR +{delta_pct}%"
        elif delta_pct < 0:
            park_note = f"{venue} HR {delta_pct}%"
        else:
            park_note = f"{venue} HR 平均"
    else:
        park_note = venue if venue else "—"

    lines = [
        "## 牛棚 / Park",
        "| | HOME | AWAY |",
        "|---|------|------|",
        f"| Bullpen ERA | {h_bullpen_era} | {a_bullpen_era} |",
        f"| 投手 IL 數（含先發/長傷） | {h_il_count} | {a_il_count} |",
        f"| IL 名單（前 2） | {h_core_il} | {a_core_il} |",
        f"| Park Factor (runs) | {pf_str} | — |",
        f"| Park 備註 | {park_note} | — |",
        "",
    ]
    return lines


def _render_risk_summary(bundle: dict) -> list[str]:
    """## ⚠️ 風險提示摘要 section."""
    try:
        from pitcher_stats import detect_triggers as detect_pitcher_triggers
    except ImportError:
        def detect_pitcher_triggers(data):
            return []

    try:
        from lineup_analyzer import detect_triggers as detect_lineup_triggers
    except ImportError:
        def detect_lineup_triggers(data):
            return []

    merged = bundle.get("merged") or {}
    meta = merged.get("_meta") or {}
    home_name = meta.get("home_team", "HOME")
    away_name = meta.get("away_team", "AWAY")

    home_p = bundle.get("home_pitcher") or {}
    away_p = bundle.get("away_pitcher") or {}
    home_lu = bundle.get("home_lineup") or {}
    away_lu = bundle.get("away_lineup") or {}

    h_p_triggers = detect_pitcher_triggers(home_p)
    a_p_triggers = detect_pitcher_triggers(away_p)
    h_lu_triggers = detect_lineup_triggers({"last7_babip": home_lu.get("last7_babip")})
    a_lu_triggers = detect_lineup_triggers({"last7_babip": away_lu.get("last7_babip")})

    items: list[str] = []

    for t in h_p_triggers:
        flag = t.get("flag", "?")
        val = t.get("value")
        val_str = f"={val:.2f}" if val is not None else ""
        items.append(
            f"- {home_name} 投手：era_xera_delta{val_str}（Flag {flag}）"
            f"— 運氣或結構性？AI 判斷，**不自動補跑 YoY、不自動下修預測**"
        )

    for t in a_p_triggers:
        flag = t.get("flag", "?")
        val = t.get("value")
        val_str = f"={val:.2f}" if val is not None else ""
        items.append(
            f"- {away_name} 投手：era_xera_delta{val_str}（Flag {flag}）"
            f"— 運氣或結構性？AI 判斷，**不自動補跑 YoY、不自動下修預測**"
        )

    for t in h_lu_triggers:
        flag = t.get("flag", "?")
        val = t.get("value")
        val_str = f"={val:.3f}" if val is not None else ""
        items.append(
            f"- {home_name} 打線：last7 BABIP{val_str}（Flag {flag}）"
            f"— 可能回歸或可能持續？AI 判斷，**不自動 ±run value**"
        )

    for t in a_lu_triggers:
        flag = t.get("flag", "?")
        val = t.get("value")
        val_str = f"={val:.3f}" if val is not None else ""
        items.append(
            f"- {away_name} 打線：last7 BABIP{val_str}（Flag {flag}）"
            f"— 可能回歸或可能持續？AI 判斷，**不自動 ±run value**"
        )

    lines = ["## ⚠️ 風險提示摘要（AI 在 summary 風險提示段處理）"]
    if items:
        lines.extend(items)
    else:
        lines.append("無風險提示")
    lines.append("")

    return lines


def _render_file_index(
    bundle: dict, game_dir: str, summary_filename: str = "summary.md"
) -> list[str]:
    """## File 索引 section.

    summary_filename: DH 場次帶 suffix（如 "summary-G1.md"），由 render_dossier 傳入。
    """
    # Use POSIX forward slashes regardless of OS path separators
    posix_dir = Path(game_dir).as_posix() if game_dir else ""
    if posix_dir and not posix_dir.endswith("/"):
        posix_dir = posix_dir + "/"

    lines = [
        "## File 索引",
        f"- merged.json (機讀資料): `{posix_dir}merged.json`",
        f"- 分析寫作: `{posix_dir}{summary_filename}`",
        f"- 個別 detail summary（drill-down / debug）: 同目錄下 `<basename>_summary.md`",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_dossier(
    bundle: dict, *, game_dir: str = "", summary_filename: str = "summary.md"
) -> str:
    """主入口：渲染整份 dossier.md，回傳 markdown 字串（不寫檔；caller 寫檔）。

    game_dir: 用於 markdown 內 File 索引段的路徑文字（不開檔）。
    summary_filename: DH 場次帶 suffix（如 "summary-G1.md"），不指定時用 default。
    """
    sections: list[str] = []
    sections += _render_header(bundle)
    sections += _render_game_info(bundle)
    sections += _render_record_summary(bundle)
    sections += _render_series_context(bundle)
    sections += _render_pitcher_matchup(bundle)
    sections += _render_lineup_overview(bundle)
    sections += _render_bullpen_park(bundle)
    sections += _render_risk_summary(bundle)
    sections += _render_file_index(bundle, game_dir, summary_filename=summary_filename)

    return "\n".join(sections)
