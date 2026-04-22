#!/usr/bin/env python3
"""MLB Results Fetcher — 抓 MLB Stats API Final 比分、回填 actual_*、計算 result codes、雙寫 per-game + per-date jsonl

用法：
  python fetch_results.py --date 2026-04-21

取代原 upload_results.py（後端上傳已廢；actual_* 手動填寫流程改為自動）。
"""

import argparse
import json
import os
import sys

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_ROOT = os.path.dirname(SCRIPT_DIR)
ANALYSIS_DATA_DIR = os.path.join(SKILL_ROOT, "analysis-data")

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def fetch_final_scores(date: str) -> list[dict]:
    """抓 MLB API 當日 Final 比分。回傳 list of {home_team, away_team, home_score, away_score}"""
    params = {"sportId": 1, "date": date, "hydrate": "linescore"}
    resp = requests.get(MLB_SCHEDULE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            if g.get("gameType") != "R":
                continue
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            results.append({
                "home_team": home.get("team", {}).get("name", ""),
                "away_team": away.get("team", {}).get("name", ""),
                "home_score": home.get("score", 0),
                "away_score": away.get("score", 0),
            })
    return results


def find_game_folder(date: str, home_team: str, away_team: str, analysis_data_dir=None) -> str | None:
    """在 analysis-data/{date}/ 下尋找 per-game 資料夾（匹配隊名關鍵字）。"""
    base = str(analysis_data_dir) if analysis_data_dir else ANALYSIS_DATA_DIR
    date_dir = os.path.join(base, date)
    if not os.path.isdir(date_dir):
        return None
    home_kw = home_team.split()[-1].lower()
    away_kw = away_team.split()[-1].lower()
    for name in os.listdir(date_dir):
        folder = os.path.join(date_dir, name)
        pred_path = os.path.join(folder, "prediction.json")
        if not os.path.exists(pred_path):
            continue
        try:
            with open(pred_path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            if home_kw in (rec.get("home_team") or "").lower() and \
               away_kw in (rec.get("away_team") or "").lower():
                return folder
        except (json.JSONDecodeError, OSError):
            continue
    return None


def apply_scores_to_predictions(date: str, scores: list[dict], analysis_data_dir=None) -> int:
    """依隊名 match，把 Final 比分寫入 per-game prediction.json 的 actual_*。回傳寫入筆數"""
    count = 0
    for s in scores:
        folder = find_game_folder(date, s["home_team"], s["away_team"], analysis_data_dir)
        if not folder:
            print(f"WARNING: 找不到 {s['away_team']} @ {s['home_team']} 的資料夾", file=sys.stderr)
            continue
        pred_path = os.path.join(folder, "prediction.json")
        with open(pred_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        rec["actual_home_score"] = s["home_score"]
        rec["actual_away_score"] = s["away_score"]
        rec["actual_winner"] = "HOME" if s["home_score"] > s["away_score"] else "AWAY"
        rec["actual_total"] = s["home_score"] + s["away_score"]
        rec["verified"] = True
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Fetch MLB results + compute result codes + save")
    parser.add_argument("--date", required=True, help="日期 YYYY-MM-DD")
    args = parser.parse_args()
    # Task 1.3 填入剩餘邏輯
    scores = fetch_final_scores(args.date)
    print(f"抓到 {len(scores)} 場 Final 比分")


if __name__ == "__main__":
    main()
