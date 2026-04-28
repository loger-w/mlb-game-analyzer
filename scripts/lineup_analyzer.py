#!/usr/bin/env python3
"""MLB Lineup Analyzer — Phase 2 打線分析（MLB Stats API + Statcast）"""

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from _team_resolver import resolve_team_id


@contextlib.contextmanager
def _redirect_pybaseball_stdout():
    """將 pybaseball 內部的 print() 訊息導向 stderr，保持 stdout 純淨"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield
    output = buf.getvalue()
    if output:
        sys.stderr.write(output)

def _import_pybaseball():
    """Lazy import：pybaseball 在實際抓 Statcast 時才載入。
    讓 format_md / detect_triggers 等純函式在沒有 pybaseball 的環境下也能測試。"""
    try:
        from pybaseball import (
            statcast_batter_expected_stats,
            statcast_batter_exitvelo_barrels,
        )
        return statcast_batter_expected_stats, statcast_batter_exitvelo_barrels
    except ImportError as e:
        raise RuntimeError(
            "pybaseball not installed or cannot import. Run: pip install pybaseball"
        ) from e


MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# xwOBA-based tier thresholds (replacing wRC+)
TIER_MAP = [
    ("🔴 Elite", lambda xwoba: xwoba >= 0.370),
    ("🟠 Strong", lambda xwoba: xwoba >= 0.340),
    ("🟡 Average", lambda xwoba: xwoba >= 0.310),
    ("🟢 Weak", lambda _: True),
]

# Fallback tier using OPS when xwOBA unavailable
TIER_MAP_OPS = [
    ("🔴 Elite", lambda ops: ops >= 0.830),
    ("🟠 Strong", lambda ops: ops >= 0.760),
    ("🟡 Average", lambda ops: ops >= 0.700),
    ("🟢 Weak", lambda _: True),
]


def fetch_team_roster(team_id: int, year: int) -> list[dict]:
    """從 MLB Stats API 取得球隊 active roster"""
    resp = requests.get(
        f"{MLB_API_BASE}/teams/{team_id}/roster",
        params={"rosterType": "active", "season": year},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    players = []
    for p in data.get("roster", []):
        person = p["person"]
        pos = p.get("position", {}).get("abbreviation", "")
        # 排除投手
        if pos == "P":
            continue
        players.append({
            "id": person["id"],
            "name": person["fullName"],
            "position": pos,
        })
    return players


def fetch_player_batting(mlbam_id: int, year: int) -> dict | None:
    """從 MLB Stats API 取得單一球員本季打擊數據"""
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{mlbam_id}/stats",
            params={"stats": "season", "group": "hitting", "season": year},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        stats_list = data.get("stats", [])
        if not stats_list or not stats_list[0].get("splits"):
            return None

        s = stats_list[0]["splits"][0]["stat"]
        pa = int(s.get("plateAppearances", 0))
        if pa == 0:
            return None

        avg = float(s.get("avg", 0))
        obp = float(s.get("obp", 0))
        slg = float(s.get("slg", 0))
        ops = float(s.get("ops", 0))
        babip = float(s.get("babip", 0))
        so = int(s.get("strikeOuts", 0))
        bb = int(s.get("baseOnBalls", 0))

        k_pct = round(so / pa * 100, 1) if pa > 0 else 0.0
        bb_pct = round(bb / pa * 100, 1) if pa > 0 else 0.0
        iso = round(slg - avg, 3)

        return {
            "mlbam_id": mlbam_id,
            "pa": pa,
            "avg": round(avg, 3),
            "obp": round(obp, 3),
            "slg": round(slg, 3),
            "ops": round(ops, 3),
            "iso": iso,
            "babip": round(babip, 3),
            "k_pct": k_pct,
            "bb_pct": bb_pct,
        }
    except Exception:
        return None


def fetch_statcast_batting_leaderboard(year: int) -> tuple[dict, dict]:
    """取得 Statcast 打者 expected stats 和 exit velo/barrels leaderboard
    回傳兩個 dict，key = player_id (str)
    """
    expected_map = {}
    barrels_map = {}
    statcast_batter_expected_stats, statcast_batter_exitvelo_barrels = _import_pybaseball()

    try:
        with _redirect_pybaseball_stdout():
            df_exp = statcast_batter_expected_stats(year, minPA=1)
        if not df_exp.empty:
            for _, row in df_exp.iterrows():
                pid = str(row.get("player_id", ""))
                expected_map[pid] = {
                    "xwoba": round(float(row.get("est_woba", 0)), 3) if row.get("est_woba") is not None else None,
                    "xba": round(float(row.get("est_ba", 0)), 3) if row.get("est_ba") is not None else None,
                    "xslg": round(float(row.get("est_slg", 0)), 3) if row.get("est_slg") is not None else None,
                }
    except Exception:
        pass

    try:
        with _redirect_pybaseball_stdout():
            df_bar = statcast_batter_exitvelo_barrels(year, minBBE=1)
        if not df_bar.empty:
            for _, row in df_bar.iterrows():
                pid = str(row.get("player_id", ""))
                barrels_map[pid] = {
                    "ev95pct": round(float(row.get("ev95percent", 0)), 1) if row.get("ev95percent") is not None else None,
                    "barrel_pct": round(float(row.get("brl_percent", 0)), 1) if row.get("brl_percent") is not None else None,
                }
    except Exception:
        pass

    return expected_map, barrels_map


def fetch_il_names(team_id: int, year: int) -> set[str]:
    """從 40Man roster 取得 IL 球員姓名集合。

    失敗時直接 raise — IL 過濾是 lineup 正確性的必要條件，
    不可 silent fallback（會把「IL fetch 壞了」偽裝成「沒人 IL」）。
    """
    resp = requests.get(
        f"{MLB_API_BASE}/teams/{team_id}/roster",
        params={"rosterType": "40Man", "season": year},
        timeout=10,
    )
    resp.raise_for_status()
    return {
        entry.get("person", {}).get("fullName", "")
        for entry in resp.json().get("roster", [])
        if entry.get("status", {}).get("code", "A") in ("D7", "D10", "D15", "D60")
    }


def fetch_player_last7(mlbam_id: int) -> dict | None:
    """D3: 取得打者近 7 場打擊數據"""
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{mlbam_id}/stats",
            params={"stats": "lastXGames", "group": "hitting", "limit": 7},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        for sg in data.get("stats", []):
            for split in sg.get("splits", [])[:1]:
                s = split.get("stat", {})
                pa = int(s.get("plateAppearances", 0))
                if pa == 0:
                    return None
                return {
                    "avg": s.get("avg"),
                    "obp": s.get("obp"),
                    "slg": s.get("slg"),
                    "ops": s.get("ops"),
                    "babip": s.get("babip"),
                    "pa": pa,
                }
        return None
    except Exception:
        return None


def fetch_player_platoon(mlbam_id: int, year: int) -> dict | None:
    """D2: 取得打者 Platoon Splits（vs LHP / vs RHP）"""
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{mlbam_id}/stats",
            params={
                "stats": "statSplits",
                "group": "hitting",
                "season": year,
                "sitCodes": "vl,vr",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        result = {}
        for sg in data.get("stats", []):
            for split in sg.get("splits", []):
                desc = split.get("split", {}).get("description", "")
                s = split.get("stat", {})
                pa = int(s.get("plateAppearances", 0))
                key = "vs_lhp" if "Left" in desc else "vs_rhp"
                result[key] = {
                    "avg": s.get("avg"),
                    "obp": s.get("obp"),
                    "slg": s.get("slg"),
                    "ops": s.get("ops"),
                    "pa": pa,
                }
        return result if result else None
    except Exception:
        return None


def fetch_bvp(batter_id: int, pitcher_id: int) -> dict | None:
    """D4: 取得打者 vs 特定投手的生涯對戰紀錄

    API 回傳兩組 splits：
    - 第一組：彙總（無 season 欄位），取 PA 最大的一筆 = 生涯合計
    - 第二組：逐場明細（有 season 欄位），忽略
    """
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{batter_id}/stats",
            params={
                "stats": "vsPlayer",
                "group": "hitting",
                "opposingPlayerId": pitcher_id,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        # 只取第一組 splits（彙總資料，無 season 欄位）
        stats_groups = data.get("stats", [])
        if not stats_groups:
            return None

        first_group = stats_groups[0].get("splits", [])
        if not first_group:
            return None

        # 取 PA 最大的一筆 = 生涯對戰合計
        best = max(first_group, key=lambda s: int(s.get("stat", {}).get("plateAppearances", 0)))
        s = best.get("stat", {})
        pa = int(s.get("plateAppearances", 0))
        if pa == 0:
            return None

        return {
            "pa": pa,
            "avg": s.get("avg"),
            "obp": s.get("obp"),
            "slg": s.get("slg"),
            "h": int(s.get("hits", 0)),
            "hr": int(s.get("homeRuns", 0)),
            "so": int(s.get("strikeOuts", 0)),
            "bb": int(s.get("baseOnBalls", 0)),
            "sample_sufficient": pa >= 15,
        }
    except Exception:
        return None


def analyze_team(team: str, year: int, opposing_pitcher_id: int | None = None) -> dict:
    """完整的球隊打線分析"""
    team_id = resolve_team_id(team)

    # 1. 取 active roster + IL 名單排除
    roster = fetch_team_roster(team_id, year)
    if not roster:
        return {"error": f"No active roster found for {team}"}

    il_names = fetch_il_names(team_id, year)

    # 2. 批次取 MLB API 打擊數據（排除 IL 球員）
    batters = []
    for player in roster:
        if player["name"] in il_names:
            continue
        stats = fetch_player_batting(player["id"], year)
        if stats:
            stats["name"] = player["name"]
            stats["position"] = player["position"]
            batters.append(stats)

    if not batters:
        return {"error": f"No batting stats found for {team} in {year}"}

    # D1: 按 PA 排序取前 9 人（完整打線）
    batters.sort(key=lambda b: b["pa"], reverse=True)
    core_lineup = batters[:9]

    # 3. 取 Statcast leaderboard（一次拉全聯盟，記憶體內 merge）
    expected_map, barrels_map = fetch_statcast_batting_leaderboard(year)

    # 4. Merge Statcast + D2 Platoon + D3 Last7 + D4 BvP
    for batter in core_lineup:
        pid = str(batter["mlbam_id"])
        exp = expected_map.get(pid, {})
        bar = barrels_map.get(pid, {})
        # D5: 修正命名（ev95pct 取代 hard_hit_pct）
        batter["xwoba"] = exp.get("xwoba")
        batter["xba"] = exp.get("xba")
        batter["xslg"] = exp.get("xslg")
        batter["ev95pct"] = bar.get("ev95pct")
        batter["barrel_pct"] = bar.get("barrel_pct")

        # D2: Platoon Splits
        batter["platoon"] = fetch_player_platoon(batter["mlbam_id"], year)

        # D3: 近 7 場熱度
        batter["last_7"] = fetch_player_last7(batter["mlbam_id"])

        # D4: BvP（如果有提供對方投手 ID）
        if opposing_pitcher_id:
            batter["bvp"] = fetch_bvp(batter["mlbam_id"], opposing_pitcher_id)

    # 5. 整體指標
    avg_ops = sum(b["ops"] for b in core_lineup) / len(core_lineup)
    avg_babip = sum(b["babip"] for b in core_lineup) / len(core_lineup)
    avg_k_pct = sum(b["k_pct"] for b in core_lineup) / len(core_lineup)
    avg_bb_pct = sum(b["bb_pct"] for b in core_lineup) / len(core_lineup)

    xwoba_values = [b["xwoba"] for b in core_lineup if b.get("xwoba") is not None]
    avg_xwoba = sum(xwoba_values) / len(xwoba_values) if xwoba_values else None

    # 6. 打線評級（優先用 xwOBA，fallback OPS）
    tier = "🟢 Weak"
    if avg_xwoba is not None:
        for tier_name, check_fn in TIER_MAP:
            if check_fn(avg_xwoba):
                tier = tier_name
                break
    else:
        for tier_name, check_fn in TIER_MAP_OPS:
            if check_fn(avg_ops):
                tier = tier_name
                break

    # 7. 大小分傾向
    over_under_lean = 0
    if avg_babip <= 0.270:
        over_under_lean += 1
    if avg_babip >= 0.320:
        over_under_lean -= 1
    if avg_k_pct >= 25:
        over_under_lean -= 1
    if avg_xwoba is not None and avg_xwoba >= 0.350:
        over_under_lean += 1

    # 8. 串聯分析
    chain = {}
    if len(core_lineup) >= 3:
        chain["obp_top3"] = round(sum(b["obp"] for b in core_lineup[:3]) / 3, 3)
    if len(core_lineup) >= 5:
        chain["slg_mid"] = round(sum(b["slg"] for b in core_lineup[3:5]) / 2, 3)

    # D3: 整體近 7 場熱度
    last7_ops_values = []
    for b in core_lineup:
        if b.get("last_7") and b["last_7"].get("ops"):
            try:
                last7_ops_values.append(float(b["last_7"]["ops"]))
            except (ValueError, TypeError):
                pass
    recent_heat = None
    if last7_ops_values:
        avg_last7_ops = sum(last7_ops_values) / len(last7_ops_values)
        if avg_last7_ops >= 0.830:
            recent_heat = "🔥 Hot"
        elif avg_last7_ops <= 0.600:
            recent_heat = "🥶 Cold"
        else:
            recent_heat = "⚖️ Normal"

    return {
        "team": team,
        "team_id": team_id,
        "tier": tier,
        "avg_ops": round(avg_ops, 3),
        "avg_xwoba": round(avg_xwoba, 3) if avg_xwoba else None,
        "avg_babip": round(avg_babip, 3),
        "avg_k_pct": round(avg_k_pct, 1),
        "avg_bb_pct": round(avg_bb_pct, 1),
        "over_under_lean": over_under_lean,
        "recent_heat": recent_heat,
        "last7_babip": compute_last7_babip(core_lineup),  # Plan B §4.6（B10 觸發用）
        "chain": chain,
        "lineup": core_lineup,
    }


def compute_last7_babip(core_lineup: list[dict]) -> float | None:
    """近 7 天 BABIP 平均（Plan B 2026-04-22 §4.6）。

    從 core_lineup 每個打者的 `last_7.babip` 取值；非數值或缺失則忽略。
    空結果回 None（「沒數據」vs「平均 0」的語義差別）。
    """
    values = []
    for b in core_lineup:
        val = (b.get("last_7") or {}).get("babip")
        if val is None:
            continue
        try:
            values.append(float(val))
        except (ValueError, TypeError):
            continue
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def detect_triggers(data: dict) -> list[dict]:
    """偵測打線層級 Flag。回傳觸發列表，未觸發則回空 list。

    Flag 3：last7_babip ≤ 0.260（極端低）或 ≥ 0.370（極端高）
    """
    triggers = []
    last7 = data.get("last7_babip")
    if last7 is None:
        return triggers
    try:
        v = float(last7)
    except (ValueError, TypeError):
        return triggers
    if v <= 0.260:
        triggers.append({
            "flag": 3,
            "name": "BABIP 回歸（極端低）",
            "value": v,
            "threshold": "≤ 0.260",
            "interpretation": "近 7 天 BABIP 偏低，預示反彈（運氣差於常態）。",
            "action": "Phase 3.4 Hot/Cold 判定前先做 BABIP 回歸檢查（見 matchup-factors.md §BABIP 回歸檢查）",
        })
    elif v >= 0.370:
        triggers.append({
            "flag": 3,
            "name": "BABIP 回歸（極端高）",
            "value": v,
            "threshold": "≥ 0.370",
            "interpretation": "近 7 天 BABIP 偏高，預示回歸 ~.300（熱度含運氣成份）。",
            "action": "Phase 3.4 Hot/Cold 判定前先做 BABIP 回歸檢查（見 matchup-factors.md §BABIP 回歸檢查）",
        })
    return triggers


def _md_fmt(v, decimals: int = 3) -> str:
    """格式化數值；None → '—'。"""
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        if decimals == 0:
            return f"{v:.0f}"
        return f"{v:.{decimals}f}"
    return str(v)


def format_md(data: dict, command: str | None = None) -> str:
    """渲染打線分析 MD 摘要（厚 MD：team-level + 9 人 table + Triggers）。

    純函數：只讀 data dict，不呼叫外部 API。
    """
    if "error" in data:
        return f"# Lineup Analysis — ERROR\n\n{data['error']}\n"

    team = data.get("team", "?")
    team_id = data.get("team_id", "?")
    tier = data.get("tier", "—")
    avg_ops = data.get("avg_ops")
    avg_xwoba = data.get("avg_xwoba")
    avg_babip = data.get("avg_babip")
    avg_k = data.get("avg_k_pct")
    avg_bb = data.get("avg_bb_pct")
    heat = data.get("recent_heat", "—")
    last7_babip = data.get("last7_babip")
    over_under = data.get("over_under_lean", 0)
    chain = data.get("chain", {}) or {}
    lineup = data.get("lineup", []) or []

    lines = [
        f"# Lineup Analysis — {team} (team_id {team_id})",
        f"**Tier**: {tier}",
        f"**Recent heat**: {heat}",
        f"**Lineup size**: {len(lineup)}",
        "",
        "---",
        "",
    ]

    triggers = detect_triggers(data)
    if triggers:
        lines += ["## 🚨 Triggers", ""]
        for t in triggers:
            lines += [
                f"### Flag {t['flag']} 觸發 — {t['name']}",
                f"- 數值：**{t['value']:.3f}**（觸發閾值 {t['threshold']}）",
                f"- 解讀：{t['interpretation']}",
                f"- 處理：{t['action']}",
                "",
            ]
        lines += ["---", ""]

    # Team-level stats
    lines += [
        "## Team-Level Aggregates",
        "",
        "| 指標 | 數值 |",
        "|------|------|",
        f"| avg OPS | {_md_fmt(avg_ops)} |",
        f"| avg xwOBA | {_md_fmt(avg_xwoba)} |",
        f"| avg BABIP | {_md_fmt(avg_babip)} |",
        f"| avg K% | {_md_fmt(avg_k, 1)} |",
        f"| avg BB% | {_md_fmt(avg_bb, 1)} |",
        f"| last7 BABIP | {_md_fmt(last7_babip)} |",
        f"| over_under_lean | {over_under:+d} |",
        f"| chain OBP top3 | {_md_fmt(chain.get('obp_top3'))} |",
        f"| chain SLG mid | {_md_fmt(chain.get('slg_mid'))} |",
        "",
    ]

    # Lineup table
    if lineup:
        lines += [
            "## Core Lineup (sorted by PA)",
            "",
            "| # | Name | Pos | PA | AVG | OBP | SLG | OPS | xwOBA | xBA | BABIP | K% | BB% |",
            "|---|------|-----|----|-----|-----|-----|-----|-------|-----|-------|----|----|",
        ]
        for i, b in enumerate(lineup, start=1):
            lines.append(
                f"| {i} | {b.get('name', '?')} | {b.get('position', '—')} | "
                f"{b.get('pa', '—')} | {_md_fmt(b.get('avg'))} | {_md_fmt(b.get('obp'))} | "
                f"{_md_fmt(b.get('slg'))} | {_md_fmt(b.get('ops'))} | "
                f"{_md_fmt(b.get('xwoba'))} | {_md_fmt(b.get('xba'))} | "
                f"{_md_fmt(b.get('babip'))} | {_md_fmt(b.get('k_pct'), 1)} | {_md_fmt(b.get('bb_pct'), 1)} |"
            )
        lines.append("")

        # Statcast 物理數據
        if any(b.get("ev95pct") is not None or b.get("barrel_pct") is not None for b in lineup):
            lines += [
                "## Statcast (Hard Contact)",
                "",
                "| # | Name | EV95% | Barrel% |",
                "|---|------|-------|---------|",
            ]
            for i, b in enumerate(lineup, start=1):
                lines.append(
                    f"| {i} | {b.get('name', '?')} | {_md_fmt(b.get('ev95pct'), 1)} | {_md_fmt(b.get('barrel_pct'), 1)} |"
                )
            lines.append("")

        # Last 7 days
        if any(b.get("last_7") for b in lineup):
            lines += [
                "## Last 7 Days",
                "",
                "| # | Name | PA | AVG | OBP | SLG | OPS | BABIP |",
                "|---|------|----|-----|-----|-----|-----|-------|",
            ]
            for i, b in enumerate(lineup, start=1):
                l7 = b.get("last_7") or {}
                lines.append(
                    f"| {i} | {b.get('name', '?')} | {l7.get('pa', '—')} | "
                    f"{l7.get('avg', '—')} | {l7.get('obp', '—')} | {l7.get('slg', '—')} | "
                    f"{l7.get('ops', '—')} | {l7.get('babip', '—')} |"
                )
            lines.append("")

        # Platoon
        if any(b.get("platoon") for b in lineup):
            lines += [
                "## Platoon Splits",
                "",
                "| # | Name | vs LHP (PA / OPS) | vs RHP (PA / OPS) |",
                "|---|------|-------------------|-------------------|",
            ]
            for i, b in enumerate(lineup, start=1):
                p = b.get("platoon") or {}
                lhp = p.get("vs_lhp") or {}
                rhp = p.get("vs_rhp") or {}
                lhp_str = f"{lhp.get('pa', '—')} / {lhp.get('ops', '—')}" if lhp else "—"
                rhp_str = f"{rhp.get('pa', '—')} / {rhp.get('ops', '—')}" if rhp else "—"
                lines.append(f"| {i} | {b.get('name', '?')} | {lhp_str} | {rhp_str} |")
            lines.append("")

        # BvP（如有）
        if any(b.get("bvp") for b in lineup):
            lines += [
                "## BvP (vs opposing starter)",
                "",
                "| # | Name | PA | AVG | OBP | SLG | H | HR | SO | BB | sample≥15? |",
                "|---|------|----|-----|-----|-----|----|----|----|----|------------|",
            ]
            for i, b in enumerate(lineup, start=1):
                v = b.get("bvp")
                if not v:
                    continue
                lines.append(
                    f"| {i} | {b.get('name', '?')} | {v.get('pa', '—')} | {v.get('avg', '—')} | "
                    f"{v.get('obp', '—')} | {v.get('slg', '—')} | {v.get('h', '—')} | "
                    f"{v.get('hr', '—')} | {v.get('so', '—')} | {v.get('bb', '—')} | "
                    f"{'✅' if v.get('sample_sufficient') else '❌（PA<15）'} |"
                )
            lines.append("")

    # Source
    lines += [
        "---",
        "",
        "## Source",
        f"- Generated by: `{command or 'lineup_analyzer.py'}`",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        "- JSON sibling: see same directory `<basename>.json`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Analyze team lineup")
    parser.add_argument("--team", required=True, help="Team abbreviation (e.g., KC, LAA), full name, or Chinese name")
    parser.add_argument("--year", type=int, default=datetime.now().year)
    parser.add_argument("--opposing-pitcher-id", type=int, default=None,
                        help="Opposing pitcher MLBAM ID for BvP lookup")
    parser.add_argument("--output", "-o", help="Output file path (default: print to stdout)")
    parser.add_argument("--no-md", action="store_true",
                        help="Skip MD summary output (only write JSON)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "lineup_analyzer test mode"}, indent=2))
        return

    result = analyze_team(args.team, args.year, args.opposing_pitcher_id)
    json_output = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Saved to {args.output}", file=sys.stderr)

        if not args.no_md:
            json_path = Path(args.output)
            md_path = json_path.with_name(json_path.stem + "_summary.md")
            command = f"lineup_analyzer.py --team {args.team} --year {args.year}" + (
                f" --opposing-pitcher-id {args.opposing_pitcher_id}" if args.opposing_pitcher_id else ""
            )
            try:
                md_path.write_text(format_md(result, command=command), encoding="utf-8")
                print(f"Saved summary to {md_path}", file=sys.stderr)
            except Exception as e:
                print(f"Skipped summary md: {e}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
