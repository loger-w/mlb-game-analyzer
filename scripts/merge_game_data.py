#!/usr/bin/env python3
"""MLB Merge Game Data — 合併 Phase 1/2 腳本輸出為 predict.py 所需的 merged.json"""

import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def load_json(path: str) -> dict:
    """讀取 JSON 檔案"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_pitcher_features(pitcher_data: dict, prefix: str) -> dict:
    """從 pitcher_stats.py 輸出提取 predict.py 所需特徵

    對應 FEATURE_COLS: {prefix}_starter_fip, {prefix}_starter_k_bb, {prefix}_starter_whip
    """
    season = pitcher_data.get("season", {})
    if "error" in season:
        season = {}

    fip = season.get("fip")
    k_bb = season.get("k_bb_pct")
    whip = season.get("whip")

    return {
        f"{prefix}_starter_fip": fip if fip is not None else 4.50,
        f"{prefix}_starter_k_bb": k_bb if k_bb is not None else 5.0,
        f"{prefix}_starter_whip": whip if whip is not None else 1.35,
    }


def extract_lineup_features(lineup_data: dict, prefix: str) -> dict:
    """從 lineup_analyzer.py 輸出提取 predict.py 所需特徵

    對應 FEATURE_COLS: {prefix}_batting_xwoba, {prefix}_batting_ops, {prefix}_batting_k_pct
    """
    xwoba = lineup_data.get("avg_xwoba")
    ops = lineup_data.get("avg_ops")
    k_pct = lineup_data.get("avg_k_pct")

    return {
        f"{prefix}_batting_xwoba": xwoba if xwoba is not None else 0.315,
        f"{prefix}_batting_ops": ops if ops is not None else 0.710,
        f"{prefix}_batting_k_pct": k_pct if k_pct is not None else 22.0,
    }


def extract_game_features(game_data: dict) -> dict:
    """從 fetch_game_data.py 輸出提取近期得失分

    對應 FEATURE_COLS: home_recent_rs, home_recent_ra, away_recent_rs, away_recent_ra
    """
    home = game_data.get("home_recent", {})
    away = game_data.get("away_recent", {})

    h_rs = home.get("rs_per_game")
    h_ra = home.get("ra_per_game")
    a_rs = away.get("rs_per_game")
    a_ra = away.get("ra_per_game")

    return {
        "home_recent_rs": h_rs if h_rs is not None else 4.5,
        "home_recent_ra": h_ra if h_ra is not None else 4.5,
        "away_recent_rs": a_rs if a_rs is not None else 4.5,
        "away_recent_ra": a_ra if a_ra is not None else 4.5,
    }


def extract_meta(game_data: dict, home_pitcher: dict, away_pitcher: dict) -> dict:
    """從各腳本輸出提取 metadata（隊名、投手名、先發場次、場館等）"""
    game = game_data.get("game", {})
    home_season = home_pitcher.get("season", {})
    away_season = away_pitcher.get("season", {})

    return {
        "_meta": {
            "home_team": game.get("home", {}).get("team"),
            "away_team": game.get("away", {}).get("team"),
            "home_sp": home_pitcher.get("name"),
            "away_sp": away_pitcher.get("name"),
            "home_sp_starts": home_season.get("gs"),
            "away_sp_starts": away_season.get("gs"),
            "venue": game.get("venue"),
            "game_pk": game.get("gamePk"),
            "game_date": game.get("date"),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Merge all script outputs for predict.py")
    parser.add_argument("--game", help="fetch_game_data.py output JSON path")
    parser.add_argument("--home-pitcher", help="pitcher_stats.py output for home starter")
    parser.add_argument("--away-pitcher", help="pitcher_stats.py output for away starter")
    parser.add_argument("--home-lineup", help="lineup_analyzer.py output for home team")
    parser.add_argument("--away-lineup", help="lineup_analyzer.py output for away team")
    parser.add_argument("--home-bullpen-era", type=float, default=4.0,
                        help="Home bullpen ERA (from WebSearch, not available via API)")
    parser.add_argument("--away-bullpen-era", type=float, default=4.0,
                        help="Away bullpen ERA (from WebSearch)")
    parser.add_argument("--park-factor", type=float, default=100.0,
                        help="Park factor (from WebSearch)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "merge_game_data test mode"}, indent=2))
        return

    required = ["game", "home_pitcher", "away_pitcher", "home_lineup", "away_lineup"]
    missing = [f"--{r.replace('_', '-')}" for r in required if getattr(args, r) is None]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    game_data = load_json(args.game)
    home_pitcher_data = load_json(args.home_pitcher)
    away_pitcher_data = load_json(args.away_pitcher)

    merged = {}
    merged.update(extract_game_features(game_data))
    merged.update(extract_pitcher_features(home_pitcher_data, "home"))
    merged.update(extract_pitcher_features(away_pitcher_data, "away"))
    merged.update(extract_lineup_features(load_json(args.home_lineup), "home"))
    merged.update(extract_lineup_features(load_json(args.away_lineup), "away"))
    merged["home_bullpen_era"] = args.home_bullpen_era
    merged["away_bullpen_era"] = args.away_bullpen_era
    merged["park_factor"] = args.park_factor
    merged.update(extract_meta(game_data, home_pitcher_data, away_pitcher_data))

    print(json.dumps(merged, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
