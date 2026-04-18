#!/usr/bin/env python3
"""MLB Prediction Uploader — 將最新預測推送至後端 API"""

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
ENV_PATH = os.path.join(SKILL_ROOT, "API_KEY.env")

from review_stats import team_to_abbr, all_daily_jsonl_paths


def _abbr_to_side(abbr: str, home_team: str, away_team: str) -> str | None:
    """Convert team abbreviation to 'home' or 'away'. Returns None if PASS."""
    upper = (abbr or "").upper()
    if upper == "PASS" or not upper:
        return None
    home_abbr = team_to_abbr(home_team).upper()
    away_abbr = team_to_abbr(away_team).upper()
    if upper == home_abbr:
        return "home"
    if upper == away_abbr:
        return "away"
    return abbr.lower()


def _parse_env_file(path: str) -> dict[str, str]:
    """簡易 .env 解析（不依賴 python-dotenv）"""
    result = {}
    if not os.path.exists(path):
        return result
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip()
    return result


def load_env():
    env = _parse_env_file(ENV_PATH)
    base_url = env.get("API_BASE_URL") or os.environ.get("API_BASE_URL")
    secret = env.get("CRON_SECRET") or os.environ.get("CRON_SECRET")
    if not base_url or not secret:
        print(f"ERROR: API_BASE_URL 或 CRON_SECRET 未設定（{ENV_PATH}）", file=sys.stderr)
        sys.exit(1)
    return base_url, secret


def load_records() -> list[dict]:
    """讀取所有 analysis-data/*/predictions.jsonl（按日期排序）"""
    paths = all_daily_jsonl_paths()
    if not paths:
        print(f"ERROR: 找不到任何 predictions.jsonl（{ANALYSIS_DATA_DIR}）", file=sys.stderr)
        sys.exit(1)
    records = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def find_record(records: list[dict], last_n: int = 1, date: str = None, home: str = None) -> list[dict]:
    """依條件篩選要上傳的紀錄"""
    if date and home:
        result = [
            r for r in records
            if r.get("date") == date and team_to_abbr(r.get("home_team", "")) == home.upper()
        ]
        if not result:
            print(f"ERROR: 找不到 {date} 主隊 {home} 的紀錄", file=sys.stderr)
            sys.exit(1)
        return result
    return records[-last_n:]


def build_market(line, direction, stars, default_stars: int = 3) -> tuple:
    """回傳 (line, direction, stars)，若 direction=PASS 或 null 則 stars=1（API 最低值）"""
    if not direction or str(direction).upper() == "PASS":
        return None, None, 1
    final_stars = stars if (stars is not None and stars >= 1) else default_stars
    return line, str(direction).lower(), final_stars


def _load_analysis(r: dict, analysis_dir: str | None) -> dict | None:
    """Load analysis.json for a prediction record if available."""
    # Explicit directory
    if analysis_dir:
        path = os.path.join(analysis_dir, "analysis.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    # Same directory as this script
    default_path = os.path.join(SCRIPT_DIR, "..", "analysis.json")
    if os.path.exists(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def record_to_payload(r: dict, analysis: dict | None = None) -> dict:
    """將 predictions.jsonl 的一筆紀錄轉為 API predictions payload"""
    game_time = r.get("game_time")
    if not game_time:
        print(f"WARNING: game_time 欄位不存在，改用 date={r.get('date')}", file=sys.stderr)
        game_time = r.get("date")

    # --- Moneyline: use analyst direction (ml_rec), 0 = PASS ---
    ml_rec_raw = (r.get("ml_rec") or "").upper()
    ml_stars = r.get("ml_stars")
    is_ml_pass = ml_rec_raw == "PASS" or (ml_stars is not None and ml_stars <= 0)

    if is_ml_pass:
        # PASS: keep model direction (DB requires NOT NULL), stars = 1 (API minimum)
        predicted_winner = (r.get("predicted_winner") or "").lower()
        ml_stars = 1
    else:
        # Active pick: use analyst's recommendation as direction
        analyst_side = _abbr_to_side(ml_rec_raw, r.get("home_team", ""), r.get("away_team", ""))
        predicted_winner = analyst_side if analyst_side else (r.get("predicted_winner") or "").lower()

    # --- Over/Under ---
    ou_line, ou_rec, ou_stars = build_market(
        r.get("ou_line"), r.get("ou_rec"), r.get("ou_stars")
    )

    # --- Run Line: convert team abbreviation → home/away ---
    rl_direction = _abbr_to_side(
        r.get("run_line_rec", "PASS"),
        r.get("home_team", ""),
        r.get("away_team", ""),
    )
    rl_line_raw = r.get("run_line")
    if rl_line_raw is not None:
        try:
            rl_line_raw = float(rl_line_raw)
        except (ValueError, TypeError):
            rl_line_raw = -1.5
    elif rl_direction is not None:
        rl_line_raw = -1.5  # Standard MLB run line

    rl_line, rl_rec, rl_stars = build_market(
        rl_line_raw, rl_direction, r.get("run_line_stars")
    )

    payload = {
        "home_team": team_to_abbr(r.get("home_team", "")),
        "away_team": team_to_abbr(r.get("away_team", "")),
        "game_time": game_time,
        "predicted_winner": predicted_winner,
        "predicted_home_pct": r.get("predicted_home_pct"),
        "ml_stars": ml_stars,
        "ou_line": ou_line,
        "ou_rec": ou_rec,
        "ou_stars": ou_stars,
        "run_line": rl_line,
        "run_line_rec": rl_rec,
        "run_line_stars": rl_stars,
    }
    if analysis:
        payload["analysis"] = analysis
    return payload


def upload(base_url: str, secret: str, records: list[dict], analysis_dir: str | None = None):
    """POST predictions 到後端"""
    if not records:
        print("沒有要上傳的紀錄", file=sys.stderr)
        return

    # Load analysis if available
    analysis = _load_analysis({}, analysis_dir)

    # 按日期分組（通常同一天一起送）
    by_date: dict[str, list] = {}
    for r in records:
        date = r.get("date", "unknown")
        by_date.setdefault(date, []).append(r)

    for date, day_records in by_date.items():
        # Only attach analysis to the most recent record (typically one game at a time)
        payloads = []
        for i, r in enumerate(day_records):
            a = analysis if (i == len(day_records) - 1 and analysis) else None
            payloads.append(record_to_payload(r, a))
        payload = {
            "date": date,
            "predictions": payloads,
        }
        url = f"{base_url.rstrip('/')}/api/ingest/predictions"
        headers = {
            "Authorization": f"Bearer {secret}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code >= 200 and resp.status_code < 300:
            print(f"✅ {date}: {len(day_records)} 筆預測上傳成功（{resp.status_code}）")
        else:
            print(f"ERROR: {date} 上傳失敗 {resp.status_code} — {resp.text}", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload latest predictions to backend API")
    parser.add_argument("--last", type=int, default=1, metavar="N",
                        help="上傳最後 N 筆（預設 1）")
    parser.add_argument("--date", help="依日期篩選（YYYY-MM-DD），需搭配 --home")
    parser.add_argument("--home", help="主隊縮寫，搭配 --date 精確找到特定比賽")
    parser.add_argument("--analysis-dir", default=None,
                        help="analysis.json 所在目錄（預設自動偵測 skill 根目錄）")
    parser.add_argument("--test", action="store_true", help="印出 payload 但不實際送出")
    args = parser.parse_args()

    base_url, secret = load_env()
    all_records = load_records()
    target = find_record(all_records, last_n=args.last, date=args.date, home=args.home)

    if args.test:
        analysis = _load_analysis({}, args.analysis_dir)
        by_date: dict[str, list] = {}
        for r in target:
            by_date.setdefault(r.get("date", "unknown"), []).append(r)
        for date, day_records in by_date.items():
            payloads = []
            for i, r in enumerate(day_records):
                a = analysis if (i == len(day_records) - 1 and analysis) else None
                payloads.append(record_to_payload(r, a))
            payload = {
                "date": date,
                "predictions": payloads,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        if analysis:
            print(f"\n📊 analysis.json attached ({len(json.dumps(analysis))} bytes)", file=sys.stderr)
        return

    upload(base_url, secret, target, args.analysis_dir)


if __name__ == "__main__":
    main()
