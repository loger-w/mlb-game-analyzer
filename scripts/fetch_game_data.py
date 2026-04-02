#!/usr/bin/env python3
"""MLB Game Data Fetcher — Phase 1 API 資料一次撈齊"""

import argparse
import json
import sys
from datetime import datetime, timedelta

import requests

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

TEAM_MAP = {
    # English abbreviations
    "NYY": 147, "NYM": 121, "BOS": 111, "LAD": 119, "LAA": 108,
    "HOU": 117, "ATL": 144, "PHI": 143, "SD": 135, "SF": 137,
    "CHC": 112, "CWS": 145, "CIN": 113, "STL": 138, "MIL": 158,
    "PIT": 134, "ARI": 109, "COL": 115, "BAL": 110, "TB": 139,
    "TOR": 141, "MIN": 142, "KC": 118, "DET": 116, "CLE": 114,
    "SEA": 136, "OAK": 133, "TEX": 140, "MIA": 146, "WSH": 120,
    # Chinese names
    "洋基": 147, "大都會": 121, "紅襪": 111, "道奇": 119, "天使": 108,
    "太空人": 117, "勇士": 144, "費城人": 143, "教士": 135, "巨人": 137,
    "小熊": 112, "白襪": 145, "紅人": 113, "紅雀": 138, "釀酒人": 158,
    "海盜": 134, "響尾蛇": 109, "落磯": 115, "金鶯": 110, "光芒": 139,
    "藍鳥": 141, "雙城": 142, "皇家": 118, "老虎": 116, "守護者": 114,
    "水手": 136, "運動家": 133, "遊騎兵": 140, "馬林魚": 146, "國民": 120,
}

# Full English names for fuzzy matching
FULL_NAMES = {
    "new york yankees": 147, "new york mets": 121, "boston red sox": 111,
    "los angeles dodgers": 119, "los angeles angels": 108, "houston astros": 117,
    "atlanta braves": 144, "philadelphia phillies": 143, "san diego padres": 135,
    "san francisco giants": 137, "chicago cubs": 112, "chicago white sox": 145,
    "cincinnati reds": 113, "st. louis cardinals": 138, "milwaukee brewers": 158,
    "pittsburgh pirates": 134, "arizona diamondbacks": 109, "colorado rockies": 115,
    "baltimore orioles": 110, "tampa bay rays": 139, "toronto blue jays": 141,
    "minnesota twins": 142, "kansas city royals": 118, "detroit tigers": 116,
    "cleveland guardians": 114, "seattle mariners": 136, "athletics": 133,
    "texas rangers": 140, "miami marlins": 146, "washington nationals": 120,
}


def resolve_team_id(team_input: str) -> int:
    """將隊名（中文/英文/縮寫）轉為 team ID"""
    # Direct match (abbreviation or Chinese)
    upper = team_input.upper()
    if upper in TEAM_MAP:
        return TEAM_MAP[upper]
    if team_input in TEAM_MAP:
        return TEAM_MAP[team_input]
    # Full English name
    lower = team_input.lower()
    if lower in FULL_NAMES:
        return FULL_NAMES[lower]
    # Fuzzy match
    for name, tid in FULL_NAMES.items():
        if lower in name:
            return tid
    raise ValueError(f"Unknown team: {team_input}")


def fetch_schedule(date: str, team_id: int = None, hydrate: str = "probablePitcher(note)"):
    """呼叫 MLB Stats API schedule endpoint"""
    params = {"sportId": 1, "date": date, "hydrate": hydrate}
    if team_id:
        params["teamId"] = team_id
    resp = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def find_game(schedule_data: dict, team_id: int) -> dict | None:
    """從 schedule 中找到指定球隊的比賽"""
    for date_entry in schedule_data.get("dates", []):
        for game in date_entry.get("games", []):
            home_id = game["teams"]["home"]["team"]["id"]
            away_id = game["teams"]["away"]["team"]["id"]
            if team_id in (home_id, away_id):
                return game
    return None


def extract_game_info(game: dict) -> dict:
    """從 game object 提取比賽資訊"""
    home = game["teams"]["home"]
    away = game["teams"]["away"]
    return {
        "gamePk": game["gamePk"],
        "date": game["gameDate"],
        "status": game["status"]["abstractGameState"],
        "venue": game["venue"]["name"],
        "home": {
            "team": home["team"]["name"],
            "team_id": home["team"]["id"],
            "probable_pitcher": home.get("probablePitcher", {}).get("fullName", "TBD"),
        },
        "away": {
            "team": away["team"]["name"],
            "team_id": away["team"]["id"],
            "probable_pitcher": away.get("probablePitcher", {}).get("fullName", "TBD"),
        },
    }


def fetch_recent_games(team_id: int, before_date: str, num_days: int = 20) -> list[dict]:
    """取得指定球隊在 before_date 前 num_days 天內已完成的比賽"""
    end_dt = datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=1)
    start_dt = datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=num_days)

    params = {
        "sportId": 1,
        "teamId": team_id,
        "startDate": start_dt.strftime("%Y-%m-%d"),
        "endDate": end_dt.strftime("%Y-%m-%d"),
        "hydrate": "linescore",
    }
    resp = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game["status"]["abstractGameState"] != "Final":
                continue
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            is_home = home["team"]["id"] == team_id
            team_side = home if is_home else away
            opp_side = away if is_home else home
            games.append({
                "date": game["gameDate"][:10],
                "is_home": is_home,
                "opponent": opp_side["team"]["name"],
                "team_score": team_side.get("score") or 0,
                "opp_score": opp_side.get("score") or 0,
                "is_winner": team_side.get("isWinner", False),
            })

    games.sort(key=lambda g: g["date"], reverse=True)
    return games[:10]


def compute_recent_stats(games: list[dict]) -> dict:
    """計算近期戰績統計"""
    if not games:
        return {"record": "0-0", "wins": 0, "losses": 0, "rs_per_game": 0, "ra_per_game": 0, "run_diff": 0, "streak": 0, "games": []}

    wins = sum(1 for g in games if g["is_winner"])
    losses = len(games) - wins
    total_rs = sum(g["team_score"] for g in games)
    total_ra = sum(g["opp_score"] for g in games)

    # 連勝/連敗
    streak = 0
    streak_win = games[0]["is_winner"]
    for g in games:
        if g["is_winner"] == streak_win:
            streak += 1
        else:
            break
    if not streak_win:
        streak = -streak

    return {
        "record": f"{wins}-{losses}",
        "wins": wins,
        "losses": losses,
        "rs_per_game": round(total_rs / len(games), 2),
        "ra_per_game": round(total_ra / len(games), 2),
        "run_diff": total_rs - total_ra,
        "streak": streak,
        "games": games,
    }


def fetch_series_prev(team_id: int, opponent_id: int, game_date: str) -> dict | None:
    """檢查同系列賽前場比分"""
    prev_date = (datetime.strptime(game_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    params = {
        "sportId": 1,
        "teamId": team_id,
        "date": prev_date,
        "hydrate": "linescore",
    }
    resp = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game["status"]["abstractGameState"] != "Final":
                continue
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            home_id = home["team"]["id"]
            away_id = away["team"]["id"]
            # 確認是同一組對手
            if {home_id, away_id} != {team_id, opponent_id}:
                continue
            return {
                "date": prev_date,
                "home": home["team"]["name"],
                "away": away["team"]["name"],
                "home_score": home.get("score") or 0,
                "away_score": away.get("score") or 0,
                "winner": home["team"]["name"] if home.get("isWinner") else away["team"]["name"],
            }
    return None


def main():
    parser = argparse.ArgumentParser(description="Fetch MLB game data for analysis")
    parser.add_argument("--date", help="Game date (YYYY-MM-DD)")
    parser.add_argument("--team", help="Team name/abbreviation")
    parser.add_argument("--test", action="store_true", help="Run with test data")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({
            "test": "OK",
            "message": "fetch_game_data test mode",
            "team_count": len(set(TEAM_MAP.values())),
            "sample": {"NYY": resolve_team_id("NYY"), "洋基": resolve_team_id("洋基")},
        }, indent=2, ensure_ascii=False))
        return

    if not args.date or not args.team:
        parser.error("--date and --team are required unless --test is specified")

    team_id = resolve_team_id(args.team)
    game_date = args.date

    # 1. 取當日賽程
    schedule = fetch_schedule(game_date)
    game = find_game(schedule, team_id)

    if not game:
        print(json.dumps({"error": f"No game found for team {args.team} on {game_date}"}, indent=2, ensure_ascii=False))
        sys.exit(1)

    game_info = extract_game_info(game)

    # 2. 取雙方近 10 場戰績
    home_id = game_info["home"]["team_id"]
    away_id = game_info["away"]["team_id"]

    home_recent = compute_recent_stats(fetch_recent_games(home_id, game_date))
    away_recent = compute_recent_stats(fetch_recent_games(away_id, game_date))

    # 3. 檢查系列賽前場
    series_prev = fetch_series_prev(home_id, away_id, game_date)

    result = {
        "game": game_info,
        "home_recent": home_recent,
        "away_recent": away_recent,
        "series_prev": series_prev,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
