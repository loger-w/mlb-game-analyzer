#!/usr/bin/env python3
"""MLB Roster Checker — 查詢球隊 active/40Man 名單，確認球員在隊狀態"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

from _team_resolver import resolve_team_id, team_abbr
from pitcher_stats import parse_ip

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

PITCHER_POSITIONS = {"Pitcher", "Starting Pitcher", "Relief Pitcher", "Closer"}


def fetch_roster(team_id: int, season: int, roster_type: str = "active") -> dict:
    """呼叫 MLB Stats API roster endpoint"""
    params = {"rosterType": roster_type, "season": season}
    resp = requests.get(f"{MLB_API_BASE}/teams/{team_id}/roster", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def parse_roster(roster_data: dict) -> dict:
    """解析 roster API 回傳，分類為投手、野手、傷兵。

    Output schema:
        pitchers: list[str]              — 投手名 (sorted by name; backward-compat)
        pitcher_ids: list[int]           — 與 pitchers 平行排列的 player_id
        position_players: list[str]
        injured_list: list[dict]         — {name, status, position, player_id}
    """
    pitcher_pairs: list[tuple[str, int | None]] = []
    position_players = []
    injured_list = []

    for entry in roster_data.get("roster", []):
        person = entry.get("person", {})
        name = person.get("fullName", "Unknown")
        pid = person.get("id")
        position = entry.get("position", {}).get("name", "Unknown")
        status_code = entry.get("status", {}).get("code", "A")
        status_desc = entry.get("status", {}).get("description", "Active")

        # 傷兵判定（含 player_id 給下游 role tagging join 用）
        if status_code in ("D7", "D10", "D15", "D60"):
            injured_list.append({
                "name": name,
                "status": status_desc,
                "position": position,
                "player_id": pid,
            })
        elif position in PITCHER_POSITIONS or "Pitcher" in position:
            pitcher_pairs.append((name, pid))
        else:
            position_players.append(name)

    pitcher_pairs.sort(key=lambda t: t[0])
    pitcher_names = [p[0] for p in pitcher_pairs]
    pitcher_ids = [p[1] for p in pitcher_pairs]

    return {
        "pitchers": pitcher_names,
        "pitcher_ids": pitcher_ids,
        "position_players": sorted(position_players),
        "injured_list": injured_list,
        "total_active": len(pitcher_names) + len(position_players),
        "total_pitchers": len(pitcher_names),
        "total_position": len(position_players),
        "total_il": len(injured_list),
    }


def fetch_combined_roster(team_id: int, season: int) -> dict:
    """同時撈 active + 40Man，交叉比對產出三層分類"""
    active_data = fetch_roster(team_id, season, "active")
    fortyman_data = fetch_roster(team_id, season, "40Man")

    active_parsed = parse_roster(active_data)
    fortyman_parsed = parse_roster(fortyman_data)

    # active 名單上的人（今天可上場）
    active_names = set(active_parsed["pitchers"] + active_parsed["position_players"])

    # 40Man 上所有人（含 IL）
    fortyman_all_names = set(fortyman_parsed["pitchers"] + fortyman_parsed["position_players"])
    il_names = {p["name"] for p in fortyman_parsed["injured_list"]}

    # 在 40Man 但不在 active 也不在 IL = 下放/其他
    not_active_40man = sorted(
        (fortyman_all_names | il_names) - active_names - il_names
    )

    return {
        "active_roster": {
            "pitchers": active_parsed["pitchers"],
            "pitcher_ids": active_parsed["pitcher_ids"],
            "position_players": active_parsed["position_players"],
        },
        "injured_list": fortyman_parsed["injured_list"],
        "not_active_40man": not_active_40man,
        "summary": {
            "total_active": active_parsed["total_active"],
            "total_active_pitchers": active_parsed["total_pitchers"],
            "total_active_position": active_parsed["total_position"],
            "total_il": fortyman_parsed["total_il"],
            "total_40man_not_active": len(not_active_40man),
        },
    }


def fetch_pitcher_season_stats_bulk(player_ids: list[int], season: int) -> dict:
    """取多位投手的本季 saves/holds/G/GS/IP。回傳 {player_id: stats_dict}。

    Per-pitcher MLB API call（hydrate fragility too high），用
    ThreadPoolExecutor(max_workers=8) 並行（Cleanup #6）。失敗 / 空 splits 的 pid
    直接從結果省略（caller 用 dict.get(pid) → None fallback "Unknown" role）。

    Prior-year fallback (Bug 3, Backlog #3): 若當季 stats 缺失 (no splits) 或 G < 5
    （長期 IL 或 April small-sample），回頭抓 season-1 stats；prior G ≥ 5 時改用
    prior 並標 from_prior_year=True，讓 tag_role 把 confidence 後綴為
    "data, prior_year" 並 disable small_sample（prior 是全季資料）。
    """
    def _fetch_one_season(pid: int, yr: int) -> dict | None:
        try:
            resp = requests.get(
                f"{MLB_API_BASE}/people/{pid}/stats",
                params={"stats": "season", "group": "pitching", "season": yr},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            stats_list = data.get("stats", [])
            if not stats_list or not stats_list[0].get("splits"):
                return None
            s = stats_list[0]["splits"][0]["stat"]
            return {
                "saves": int(s.get("saves", 0)),
                "holds": int(s.get("holds", 0)),
                "g": int(s.get("gamesPlayed", 0)),
                "gs": int(s.get("gamesStarted", 0)),
                "ip": parse_ip(s.get("inningsPitched", "0")),
            }
        except Exception as e:
            print(f"⚠️ Failed pitcher stats fetch pid={pid} season={yr}: {e}", file=sys.stderr)
            return None

    def _fetch_one(pid: int) -> tuple[int, dict | None]:
        current = _fetch_one_season(pid, season)
        if current is not None and current["g"] >= 5:
            return (pid, current)
        # Fallback: long-IL all-year (current splits empty) OR sparse current (G < 5)
        prior = _fetch_one_season(pid, season - 1)
        if prior is not None and prior["g"] >= 5:
            prior["from_prior_year"] = True
            return (pid, prior)
        return (pid, current)

    valid_pids = [pid for pid in player_ids if pid is not None]
    if not valid_pids:
        return {}

    result: dict = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for pid, stats in executor.map(_fetch_one, valid_pids):
            if stats is not None:
                result[pid] = stats
    return result


def enrich_roster_with_roles(
    roster: dict,
    stats_by_pid: dict,
    team_total_games: int | None = None,
) -> dict:
    """Decorate parsed roster output with per-pitcher core_role data.

    Pure function — does NOT call MLB API. Caller passes pre-fetched
    `stats_by_pid` (typically from `fetch_pitcher_season_stats_bulk`).

    Adds:
        - `active_roster.pitcher_roles`: list[dict] with {name, player_id,
          core_role, core_role_confidence, core_role_small_sample,
          core_role_evidence}
        - For each pitcher in `injured_list` (position contains "Pitcher"):
          merge in core_role fields. Non-pitcher IL entries untouched.

    Existing fields (`active_roster.pitchers` list[str], untouched IL fields)
    are preserved verbatim for backward compat.

    After per-pitcher tagging, `detect_committee_closer` runs over the active
    roster's roles to relabel two-headed save situations as Co-Closer.
    """
    from lib_role_tagging import tag_role, detect_committee_closer

    active = roster.get("active_roster", {}) or {}
    pitcher_names = active.get("pitchers", []) or []
    pitcher_ids = active.get("pitcher_ids", []) or []

    pitcher_roles = []
    for name, pid in zip(pitcher_names, pitcher_ids):
        if pid is not None and pid in stats_by_pid:
            tagged = tag_role(stats_by_pid[pid], team_total_games=team_total_games)
        else:
            tagged = tag_role({}, team_total_games=team_total_games)  # → Unknown
        pitcher_roles.append({"name": name, "player_id": pid, **tagged})

    detect_committee_closer(pitcher_roles)

    # Decorate IL pitcher entries with role data (pre-injury state)
    decorated_il = []
    for il_entry in roster.get("injured_list", []) or []:
        position = il_entry.get("position") or ""
        if "Pitcher" not in position:
            decorated_il.append(il_entry)
            continue
        pid = il_entry.get("player_id")
        if pid is not None and pid in stats_by_pid:
            tagged = tag_role(stats_by_pid[pid], team_total_games=team_total_games)
        else:
            tagged = tag_role({}, team_total_games=team_total_games)
        decorated_il.append({**il_entry, **tagged})

    return {
        **roster,
        "active_roster": {
            **active,
            "pitcher_roles": pitcher_roles,
        },
        "injured_list": decorated_il,
    }


def detect_triggers(data: dict, expected_starter: str | None = None) -> list[dict]:
    """偵測 roster 層級 Flag。回傳觸發列表。

    Trigger（可選）：當 expected_starter 提供且不在 active_roster.pitchers → 觸發
    """
    triggers = []
    if expected_starter:
        active = data.get("active_roster") or {}
        pitchers = active.get("pitchers") or data.get("pitchers") or []
        if expected_starter not in pitchers:
            il_names = [p.get("name") for p in (data.get("injured_list") or [])]
            on_il = expected_starter in il_names
            triggers.append({
                "flag": "STARTER_NOT_ACTIVE",
                "name": "預期先發投手不在 active roster",
                "value": expected_starter,
                "on_il": on_il,
                "interpretation": (
                    f"投手 {expected_starter} {'在 IL 名單中' if on_il else '不在 active 也不在 IL'}，"
                    "先發投手不在 active roster。"
                ),
                "action": "向使用者回報並暫停 Skill。",
            })
    return triggers


def format_md(data: dict, command: str | None = None, expected_starter: str | None = None) -> str:
    """渲染 roster MD 摘要（厚 MD：active 投手 / 野手 / IL highlight / 40man 非 active）。"""
    team = data.get("team", "?")
    team_id = data.get("team_id", "?")
    summary = data.get("summary", {}) or {}
    active = data.get("active_roster", {}) or {}
    pitchers = active.get("pitchers") or data.get("pitchers") or []
    position_players = active.get("position_players") or data.get("position_players") or []
    il = data.get("injured_list", []) or []
    not_active = data.get("not_active_40man", []) or []

    abbr = team_abbr(team_id if isinstance(team_id, int) else None, team or "")

    lines = [
        f"# Roster Check — {team} ({abbr}, team_id {team_id})",
        f"**Active**: {summary.get('total_active', len(pitchers) + len(position_players))} ({summary.get('total_active_pitchers', len(pitchers))} P / {summary.get('total_active_position', len(position_players))} 野)",
        f"**IL**: {summary.get('total_il', len(il))}",
        f"**40-man not active**: {summary.get('total_40man_not_active', len(not_active))}",
        "",
        "---",
        "",
    ]

    triggers = detect_triggers(data, expected_starter=expected_starter)
    if triggers:
        lines += ["## 🚨 Triggers", ""]
        for t in triggers:
            lines += [
                f"### {t['flag']} — {t['name']}",
                f"- 數值：**{t['value']}**",
                f"- IL?: {t.get('on_il', False)}",
                f"- 解讀：{t['interpretation']}",
                f"- 處理：{t['action']}",
                "",
            ]
        lines += ["---", ""]

    # Active pitchers
    lines += [
        "## Active Pitchers",
        "",
    ]
    if pitchers:
        for p in pitchers:
            lines.append(f"- {p}")
    else:
        lines.append("- —")
    lines.append("")

    # Active position players
    lines += [
        "## Active Position Players",
        "",
    ]
    if position_players:
        for p in position_players:
            lines.append(f"- {p}")
    else:
        lines.append("- —")
    lines.append("")

    # Injured list（含關鍵角色 highlight）
    lines += [
        "## Injured List",
        "",
    ]
    if il:
        # Highlight: 投手 IL（影響本場分析）
        pitcher_il = [p for p in il if "Pitcher" in (p.get("position") or "")]
        position_il = [p for p in il if "Pitcher" not in (p.get("position") or "")]

        if pitcher_il:
            lines += ["### Pitchers on IL", ""]
            for p in pitcher_il:
                lines.append(f"- **{p.get('name', '?')}** ({p.get('position', '—')}) — {p.get('status', '—')}")
            lines.append("")

        if position_il:
            lines += ["### Position Players on IL", ""]
            for p in position_il:
                lines.append(f"- **{p.get('name', '?')}** ({p.get('position', '—')}) — {p.get('status', '—')}")
            lines.append("")
    else:
        lines += ["- —", ""]

    # 40-man not active
    lines += [
        "## 40-Man Not Active",
        "",
    ]
    if not_active:
        for p in not_active:
            lines.append(f"- {p}")
    else:
        lines.append("- —")
    lines.append("")

    # Player check（如有）
    pc = data.get("player_check")
    if pc:
        lines += [
            "## Player Check",
            "",
            f"- Query: `{pc.get('query')}`",
            f"- On active roster: {pc.get('on_active_roster')}",
            f"- On IL: {pc.get('on_injured_list')}",
        ]
        if "on_40man_not_active" in pc:
            lines.append(f"- On 40-man not active: {pc.get('on_40man_not_active')}")
        lines.append(f"- Found: {pc.get('found')}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Source",
        f"- Generated by: `{command or 'roster_checker.py'}`",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        "- JSON sibling: see same directory `<basename>.json`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Check MLB team roster")
    parser.add_argument("--team", help="Team abbreviation (e.g., KC, LAA), full name, or Chinese name. Numeric IDs are no longer accepted.")
    parser.add_argument("--season", type=int, help="Season year")
    parser.add_argument("--type", default=None, choices=["active", "40Man", "fullRoster"],
                        help="Roster type (omit for combined active+40Man)")
    parser.add_argument("--check-player", help="Check if a specific player is on the roster")
    parser.add_argument("--expected-starter", help="先發投手姓名（提供時會檢查是否在 active roster；不在則觸發 STARTER_NOT_ACTIVE flag）")
    parser.add_argument("--output", "-o", help="Output file path (default: print to stdout)")
    parser.add_argument("--no-md", action="store_true", help="Skip MD summary output (only write JSON)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "roster_checker test mode"}, indent=2))
        return

    if not args.team or not args.season:
        parser.error("--team and --season are required unless --test is specified")

    # CLI 標準化：team 必須為縮寫/全名/中文，純數字 ID 一律拒絕
    team_id = resolve_team_id(args.team)

    # 取得隊名
    team_name = "Unknown"
    try:
        team_resp = requests.get(f"{MLB_API_BASE}/teams/{team_id}", timeout=10)
        team_resp.raise_for_status()
        team_info = team_resp.json()
        teams = team_info.get("teams", [])
        if teams:
            team_name = teams[0].get("name", "Unknown")
    except Exception:
        pass

    # 合併模式（預設）vs 單一模式（向後相容）
    if args.type is None:
        combined = fetch_combined_roster(team_id, args.season)
        # PR-2 commit 8: per-pitcher core_role tagging.
        # Active pitchers + IL pitchers get role decoration.
        active_pids = combined["active_roster"].get("pitcher_ids") or []
        il_pids = [
            p.get("player_id") for p in combined["injured_list"] or []
            if "Pitcher" in (p.get("position") or "") and p.get("player_id")
        ]
        all_pids = [pid for pid in (active_pids + il_pids) if pid is not None]
        stats_by_pid = fetch_pitcher_season_stats_bulk(all_pids, args.season)
        combined = enrich_roster_with_roles(combined, stats_by_pid)
        result = {
            "team": team_name,
            "team_id": team_id,
            **combined,
        }
    else:
        roster_data = fetch_roster(team_id, args.season, args.type)
        parsed = parse_roster(roster_data)
        result = {
            "team": team_name,
            "team_id": team_id,
            "roster_type": args.type,
            **parsed,
        }

    # 特定球員查詢
    if args.check_player:
        player_lower = args.check_player.lower()
        if args.type is None:
            active_names = (
                result["active_roster"]["pitchers"]
                + result["active_roster"]["position_players"]
            )
            il_names = [p["name"] for p in result["injured_list"]]
            not_active = result["not_active_40man"]
            found_active = any(player_lower in n.lower() for n in active_names)
            found_il = any(player_lower in n.lower() for n in il_names)
            found_not_active = any(player_lower in n.lower() for n in not_active)
            result["player_check"] = {
                "query": args.check_player,
                "on_active_roster": found_active,
                "on_injured_list": found_il,
                "on_40man_not_active": found_not_active,
                "found": found_active or found_il or found_not_active,
            }
        else:
            all_names = result.get("pitchers", []) + result.get("position_players", [])
            il_names = [p["name"] for p in result.get("injured_list", [])]
            found_active = any(player_lower in n.lower() for n in all_names)
            found_il = any(player_lower in n.lower() for n in il_names)
            result["player_check"] = {
                "query": args.check_player,
                "on_active_roster": found_active,
                "on_injured_list": found_il,
                "found": found_active or found_il,
            }

    json_output = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Saved to {args.output}", file=sys.stderr)

        if not args.no_md:
            json_path = Path(args.output)
            md_path = json_path.with_name(json_path.stem + "_summary.md")
            command = (
                f"roster_checker.py --team {args.team} --season {args.season}"
                + (f" --type {args.type}" if args.type else "")
                + (f" --check-player \"{args.check_player}\"" if args.check_player else "")
                + (f" --expected-starter \"{args.expected_starter}\"" if args.expected_starter else "")
            )
            try:
                md_path.write_text(
                    format_md(result, command=command, expected_starter=args.expected_starter),
                    encoding="utf-8",
                )
                print(f"Saved summary to {md_path}", file=sys.stderr)
            except Exception as e:
                print(f"Skipped summary md: {e}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
