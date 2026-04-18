#!/usr/bin/env python3
"""MLB Roster Checker — 查詢球隊 active/40Man 名單，確認球員在隊狀態"""

import argparse
import json
import sys

import requests

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
    """解析 roster API 回傳，分類為投手、野手、傷兵"""
    pitchers = []
    position_players = []
    injured_list = []

    for entry in roster_data.get("roster", []):
        person = entry.get("person", {})
        name = person.get("fullName", "Unknown")
        position = entry.get("position", {}).get("name", "Unknown")
        status_code = entry.get("status", {}).get("code", "A")
        status_desc = entry.get("status", {}).get("description", "Active")

        player_info = {
            "name": name,
            "position": position,
            "status": status_desc,
            "player_id": person.get("id"),
        }

        # 傷兵判定
        if status_code in ("D7", "D10", "D15", "D60"):
            injured_list.append({
                "name": name,
                "status": status_desc,
                "position": position,
            })
        elif position in PITCHER_POSITIONS or "Pitcher" in position:
            pitchers.append(name)
        else:
            position_players.append(name)

    return {
        "pitchers": sorted(pitchers),
        "position_players": sorted(position_players),
        "injured_list": injured_list,
        "total_active": len(pitchers) + len(position_players),
        "total_pitchers": len(pitchers),
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


def main():
    parser = argparse.ArgumentParser(description="Check MLB team roster")
    parser.add_argument("--team", help="Team ID (numeric)")
    parser.add_argument("--season", type=int, help="Season year")
    parser.add_argument("--type", default=None, choices=["active", "40Man", "fullRoster"],
                        help="Roster type (omit for combined active+40Man)")
    parser.add_argument("--check-player", help="Check if a specific player is on the roster")
    parser.add_argument("--output", "-o", help="Output file path (default: print to stdout)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "roster_checker test mode"}, indent=2))
        return

    if not args.team or not args.season:
        parser.error("--team and --season are required unless --test is specified")

    team_id = int(args.team)

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
    else:
        print(json_output)


if __name__ == "__main__":
    main()
