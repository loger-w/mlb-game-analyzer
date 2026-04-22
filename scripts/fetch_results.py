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

from review_stats import judge_ml, judge_ou, judge_rl, daily_jsonl_path


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


def load_records(date: str) -> list[dict]:
    """讀取指定日期的 analysis-data/{date}/predictions.jsonl"""
    path = daily_jsonl_path(date)
    if not os.path.exists(path):
        print(f"ERROR: 找不到 {path}（先跑 summarize_predictions.py --date {date}）", file=sys.stderr)
        sys.exit(1)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def result_code(judge_result, rec_field: str) -> str | None:
    """將 judge_*() 回傳值轉為結果代碼。
    - judge 回傳 None 且 rec 為 PASS → "PASS"
    - judge 回傳 None 且 rec 有值 → None（比賽未結束）
    - 其他 → 原值（WIN / LOSS / PUSH）
    """
    if judge_result is not None:
        return judge_result
    if not rec_field or str(rec_field).upper() == "PASS":
        return "PASS"
    return None


def compute_results(r: dict) -> tuple[str | None, str | None, str | None]:
    """回傳 (ml_result, ou_result, run_line_result)"""
    ml = result_code(judge_ml(r), r.get("ml_rec", "PASS"))
    ou = result_code(judge_ou(r), r.get("ou_rec", "PASS"))
    rl = result_code(judge_rl(r), r.get("run_line_rec", "PASS"))
    return ml, ou, rl


def save_daily_jsonl(date: str, records: list[dict]):
    """重寫 analysis-data/{date}/predictions.jsonl"""
    path = daily_jsonl_path(date)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_per_game_predictions(date: str, records: list[dict]) -> int:
    """同步更新每場 prediction.json（真相來源）"""
    count = 0
    for r in records:
        folder = find_game_folder(date, r.get("home_team", ""), r.get("away_team", ""))
        if not folder:
            print(f"WARNING: 找不到 {r.get('away_team')} @ {r.get('home_team')} 的資料夾，略過 per-game 寫入", file=sys.stderr)
            continue
        with open(os.path.join(folder, "prediction.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        count += 1
    return count


def update_records(records: list[dict]) -> list[dict]:
    """為已驗證紀錄補上 ml_result / ou_result / run_line_result（不含 CLV）"""
    updated = []
    for r in records:
        if r.get("verified"):
            ml, ou, rl = compute_results(r)
            r = {**r, "ml_result": ml, "ou_result": ou, "run_line_result": rl}
        updated.append(r)
    return updated


def main():
    parser = argparse.ArgumentParser(description="Fetch MLB results + compute result codes + save")
    parser.add_argument("--date", required=True, help="日期 YYYY-MM-DD")
    args = parser.parse_args()

    # Step 1: 抓 MLB API Final 比分
    scores = fetch_final_scores(args.date)
    if not scores:
        print(f"WARNING: {args.date} 無 Final 比分（比賽未結束？）", file=sys.stderr)
        return

    # Step 2: 寫 actual_* + verified=true 到 per-game prediction.json
    n_applied = apply_scores_to_predictions(args.date, scores)
    print(f"✅ 寫入 actual_* 到 {n_applied} 場 prediction.json")

    # Step 3: 讀 jsonl（需要先跑 summarize_predictions）
    try:
        records = load_records(args.date)
    except SystemExit:
        print(f"INFO: predictions.jsonl 不存在，請先跑 summarize_predictions.py --date {args.date}", file=sys.stderr)
        return

    # Step 4: 計算 result codes + 雙寫
    updated = update_records(records)
    save_daily_jsonl(args.date, updated)
    n_per_game = save_per_game_predictions(args.date, updated)
    n_verified = sum(1 for r in updated if r.get("verified"))
    print(f"✅ predictions.jsonl 已更新（{n_verified} 筆 result）")
    print(f"✅ per-game prediction.json 同步 {n_per_game} 筆")


if __name__ == "__main__":
    main()
