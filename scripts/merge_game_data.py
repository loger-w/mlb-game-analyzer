#!/usr/bin/env python3
"""MLB Merge Game Data — 合併 Phase 1/2 腳本輸出為 predict.py 所需的 merged.json"""

import argparse
import json
import sys

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# E2: 30 座球場 Park Factor 對照表（5 年回歸值，2024-2025 基準）
# 來源：FanGraphs Park Factors / ESPN Park Factors
# 100 = 聯盟平均，>100 = 打者友善，<100 = 投手友善
PARK_FACTORS = {
    "Coors Field": 115,
    "Fenway Park": 105,
    "Citizens Bank Park": 104,
    "Great American Ball Park": 104,
    "Yankee Stadium": 104,
    "Globe Life Field": 103,
    "Guaranteed Rate Field": 102,
    "Wrigley Field": 102,
    "Nationals Park": 101,
    "Rogers Centre": 101,
    "Oriole Park at Camden Yards": 101,
    "Busch Stadium": 100,
    "Target Field": 100,
    "Angel Stadium": 100,
    "American Family Field": 100,
    "Minute Maid Park": 100,
    "PNC Park": 99,
    "Comerica Park": 99,
    "Chase Field": 99,
    "Kauffman Stadium": 99,
    "loanDepot park": 98,
    "Dodger Stadium": 98,
    "Progressive Field": 98,
    "Truist Park": 98,
    "Citi Field": 97,
    "T-Mobile Park": 97,
    "Tropicana Field": 96,
    "Oracle Park": 96,
    "Oakland Coliseum": 96,
    "Petco Park": 95,
}


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


def extract_pitcher_nested(
    pitcher_data: dict | None,
    prior_pitcher_data: dict | None,
    prefix: str,
) -> dict:
    """Plan B §4.6：產 nested `{prefix}_pitcher` dict，包含 era/xera/ip + prior_year.era。

    給 predict.py 的 pitcher_triggers_yoy 讀（B7 YoY 觸發判斷）。
    與現有 `extract_pitcher_features` 共存（不動 flat keys，確保 review_stats 等 backward-compat）。
    """
    season = pitcher_data.get("season", {}) if pitcher_data else {}
    if "error" in season:
        season = {}
    prior_season = (prior_pitcher_data or {}).get("season", {}) if prior_pitcher_data else {}
    if "error" in prior_season:
        prior_season = {}

    era = season.get("era")
    xera = season.get("xera")
    ip = season.get("ip")
    delta = None
    if era is not None and xera is not None:
        delta = round(abs(era - xera), 3)

    return {
        f"{prefix}_pitcher": {
            "era": era,
            "xera": xera,
            "ip": ip,
            "era_xera_delta": delta,
            "prior_year": {"era": prior_season.get("era")},
        }
    }


def extract_lineup_nested(lineup_data: dict | None, prefix: str) -> dict:
    """Plan B §4.6：產 nested `{prefix}_lineup` dict，包含 recent_babip。

    給 predict.py 的 lineup_triggers_babip 讀（B10 BABIP 回歸觸發判斷）。
    """
    recent = lineup_data.get("last7_babip") if lineup_data else None
    return {
        f"{prefix}_lineup": {
            "recent_babip": recent,
        }
    }


def extract_game_features(game_data: dict) -> dict:
    """從 fetch_game_data.py 輸出提取多窗口得失分

    對應 FEATURE_COLS: home_recent_rs, home_recent_ra, away_recent_rs, away_recent_ra
    額外 pass-through: recent_30, season 窗口（供 predict.py 交叉驗證和趨勢標籤用）
    """
    home = game_data.get("home_recent", {})
    away = game_data.get("away_recent", {})
    home_30 = game_data.get("home_recent_30", {})
    away_30 = game_data.get("away_recent_30", {})
    home_season = game_data.get("home_season", {})
    away_season = game_data.get("away_season", {})

    h_rs = home.get("rs_per_game")
    h_ra = home.get("ra_per_game")
    a_rs = away.get("rs_per_game")
    a_ra = away.get("ra_per_game")

    result = {
        # FEATURE_COLS（近 10 場，XGBoost 用）
        "home_recent_rs": h_rs if h_rs is not None else 4.5,
        "home_recent_ra": h_ra if h_ra is not None else 4.5,
        "away_recent_rs": a_rs if a_rs is not None else 4.5,
        "away_recent_ra": a_ra if a_ra is not None else 4.5,
        # 近 30 場（交叉驗證用）
        "home_recent_30_rs": home_30.get("rs_per_game") if home_30.get("rs_per_game") is not None else 4.5,
        "home_recent_30_ra": home_30.get("ra_per_game") if home_30.get("ra_per_game") is not None else 4.5,
        "away_recent_30_rs": away_30.get("rs_per_game") if away_30.get("rs_per_game") is not None else 4.5,
        "away_recent_30_ra": away_30.get("ra_per_game") if away_30.get("ra_per_game") is not None else 4.5,
        # 本季全部（趨勢標籤用）
        "home_season_rs": home_season.get("rs_per_game") if home_season.get("rs_per_game") is not None else 4.5,
        "home_season_ra": home_season.get("ra_per_game") if home_season.get("ra_per_game") is not None else 4.5,
        "away_season_rs": away_season.get("rs_per_game") if away_season.get("rs_per_game") is not None else 4.5,
        "away_season_ra": away_season.get("ra_per_game") if away_season.get("ra_per_game") is not None else 4.5,
        # 賽季場次數
        "home_season_games": game_data.get("home_season_games_count", 0),
        "away_season_games": game_data.get("away_season_games_count", 0),
    }
    return result


def fetch_bullpen_era(team_id: int, year: int) -> float:
    """E1: 從 MLB Stats API 自動取得球隊牛棚 ERA"""
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/teams/{team_id}/stats",
            params={
                "stats": "statSplits",
                "group": "pitching",
                "season": year,
                "sitCodes": "rp",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        for sg in data.get("stats", []):
            for split in sg.get("splits", []):
                era = split.get("stat", {}).get("era")
                if era is not None:
                    return float(era)
    except Exception:
        pass
    return 4.00  # fallback


def resolve_park_factor(venue_name: str | None) -> float:
    """E2: 從球場名稱查 Park Factor 對照表"""
    if venue_name and venue_name in PARK_FACTORS:
        return float(PARK_FACTORS[venue_name])
    return 100.0  # fallback


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
    parser.add_argument("--home-bullpen-era", type=float, default=None,
                        help="Override home bullpen ERA (auto-fetched if omitted)")
    parser.add_argument("--away-bullpen-era", type=float, default=None,
                        help="Override away bullpen ERA (auto-fetched if omitted)")
    parser.add_argument("--park-factor", type=float, default=None,
                        help="Override park factor (auto-resolved from venue if omitted)")
    parser.add_argument("--output", "-o", help="Output file path (default: print to stdout)")
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

    # 從 game_data 取得 team_id 和 venue
    game_info = game_data.get("game", {})
    home_team_id = game_info.get("home", {}).get("team_id")
    away_team_id = game_info.get("away", {}).get("team_id")
    venue_name = game_info.get("venue")
    game_year = int(game_info.get("date", "2026")[:4]) if game_info.get("date") else 2026

    # E1: 自動抓牛棚 ERA（可手動 override）
    if args.home_bullpen_era is not None:
        home_bp_era = args.home_bullpen_era
    elif home_team_id:
        home_bp_era = fetch_bullpen_era(home_team_id, game_year)
        print(f"Auto-fetched home bullpen ERA: {home_bp_era}", file=sys.stderr)
    else:
        home_bp_era = 4.00

    if args.away_bullpen_era is not None:
        away_bp_era = args.away_bullpen_era
    elif away_team_id:
        away_bp_era = fetch_bullpen_era(away_team_id, game_year)
        print(f"Auto-fetched away bullpen ERA: {away_bp_era}", file=sys.stderr)
    else:
        away_bp_era = 4.00

    # E2: 自動 Park Factor（可手動 override）
    if args.park_factor is not None:
        park_factor = args.park_factor
    else:
        park_factor = resolve_park_factor(venue_name)
        print(f"Auto-resolved park factor for {venue_name}: {park_factor}", file=sys.stderr)

    home_lineup_data = load_json(args.home_lineup)
    away_lineup_data = load_json(args.away_lineup)

    merged = {}
    merged.update(extract_game_features(game_data))
    merged.update(extract_pitcher_features(home_pitcher_data, "home"))
    merged.update(extract_pitcher_features(away_pitcher_data, "away"))
    merged.update(extract_lineup_features(home_lineup_data, "home"))
    merged.update(extract_lineup_features(away_lineup_data, "away"))
    # Plan B §4.6: nested dict 供 B7 YoY / B10 BABIP 觸發判斷；flat keys 保留 backward-compat
    merged.update(extract_pitcher_nested(home_pitcher_data, None, "home"))
    merged.update(extract_pitcher_nested(away_pitcher_data, None, "away"))
    merged.update(extract_lineup_nested(home_lineup_data, "home"))
    merged.update(extract_lineup_nested(away_lineup_data, "away"))
    merged["home_bullpen_era"] = home_bp_era
    merged["away_bullpen_era"] = away_bp_era
    merged["park_factor"] = park_factor
    merged.update(extract_meta(game_data, home_pitcher_data, away_pitcher_data))

    json_output = json.dumps(merged, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
