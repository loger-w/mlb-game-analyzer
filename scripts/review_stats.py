#!/usr/bin/env python3
"""MLB Post-Game Review Stats — 從 analysis-data/{date}/predictions.jsonl 計算統計數據"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "analysis-data")


def daily_jsonl_path(date: str) -> str:
    return os.path.join(ANALYSIS_DATA_DIR, date, "predictions.jsonl")


def all_daily_jsonl_paths() -> list[str]:
    pattern = os.path.join(ANALYSIS_DATA_DIR, "*", "predictions.jsonl")
    return sorted(glob.glob(pattern))

ABBR_TO_KEYWORD = {
    "NYY": "Yankees", "NYM": "Mets", "BOS": "Red Sox", "LAD": "Dodgers",
    "LAA": "Angels", "HOU": "Astros", "ATL": "Braves", "PHI": "Phillies",
    "SD": "Padres", "SF": "Giants", "CHC": "Cubs", "CWS": "White Sox",
    "CIN": "Reds", "STL": "Cardinals", "MIL": "Brewers", "PIT": "Pirates",
    "ARI": "Diamondbacks", "COL": "Rockies", "BAL": "Orioles", "TB": "Rays",
    "TOR": "Blue Jays", "MIN": "Twins", "KC": "Royals", "DET": "Tigers",
    "CLE": "Guardians", "SEA": "Mariners", "OAK": "Athletics", "TEX": "Rangers",
    "MIA": "Marlins", "WSH": "Nationals",
}

KEYWORD_TO_ABBR = {v: k for k, v in ABBR_TO_KEYWORD.items()}


def is_home_team(abbr: str, home_team: str) -> bool:
    """判斷縮寫是否對應主隊"""
    keyword = ABBR_TO_KEYWORD.get(abbr, "")
    return keyword != "" and keyword in home_team


def team_to_abbr(team_name: str) -> str:
    """從完整隊名取得縮寫"""
    for keyword, abbr in KEYWORD_TO_ABBR.items():
        if keyword in team_name:
            return abbr
    return team_name


def load_predictions(path: str = None):
    """讀取單一 jsonl（若指定）或全部 per-date jsonl"""
    paths = [path] if path else all_daily_jsonl_paths()
    records = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def filter_verified(records: list) -> list:
    """篩選已驗證紀錄"""
    return [r for r in records if r.get("verified")]


def filter_by_date(records: list, date_str: str) -> list:
    """篩選指定日期的紀錄"""
    return [r for r in records if r.get("date") == date_str]


def judge_ml(record: dict):
    """判定 ML 推薦結果。回傳 'WIN' / 'LOSS' / None（PASS）"""
    rec = record.get("ml_rec")
    if not rec or rec == "PASS":
        return None
    actual = record.get("actual_winner")
    if not actual:
        return None
    rec_is_home = is_home_team(rec, record["home_team"])
    if (rec_is_home and actual == "HOME") or (not rec_is_home and actual == "AWAY"):
        return "WIN"
    return "LOSS"


def judge_ou(record: dict):
    """判定 O/U 推薦結果。回傳 'WIN' / 'LOSS' / 'PUSH' / None（PASS）"""
    rec = record.get("ou_rec")
    if not rec or rec == "PASS":
        return None
    actual_total = record.get("actual_total")
    line = record.get("ou_line")
    if actual_total is None or line is None:
        return None
    if actual_total == line:
        return "PUSH"
    if (rec == "OVER" and actual_total > line) or (rec == "UNDER" and actual_total < line):
        return "WIN"
    return "LOSS"


def judge_rl(record: dict):
    """判定 Run Line -1.5 推薦結果。回傳 'WIN' / 'LOSS' / None（PASS）"""
    rec = record.get("run_line_rec")
    if not rec or rec == "PASS":
        return None
    home_score = record.get("actual_home_score")
    away_score = record.get("actual_away_score")
    if home_score is None or away_score is None:
        return None
    margin = home_score - away_score
    rec_is_home = is_home_team(rec, record["home_team"])
    if (rec_is_home and margin >= 2) or (not rec_is_home and margin <= -2):
        return "WIN"
    return "LOSS"


def compute_record(results: list) -> dict:
    """從結果列表計算 W-L-P 戰績"""
    wins = sum(1 for r in results if r == "WIN")
    losses = sum(1 for r in results if r == "LOSS")
    pushes = sum(1 for r in results if r == "PUSH")
    passes = sum(1 for r in results if r is None)
    total = wins + losses
    pct = (wins / total * 100) if total > 0 else 0.0
    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "passes": passes,
        "total": total,
        "pct": round(pct, 1),
    }


def compute_stats(records: list) -> dict:
    """計算完整統計"""
    ml_results = [judge_ml(r) for r in records]
    ou_results = [judge_ou(r) for r in records]
    rl_results = [judge_rl(r) for r in records]

    # 總計
    overall = {
        "ml": compute_record(ml_results),
        "ou": compute_record(ou_results),
        "rl": compute_record(rl_results),
    }

    # ML 星級拆分
    by_stars = defaultdict(list)
    for r in records:
        if r.get("ml_rec") and r["ml_rec"] != "PASS":
            stars = r.get("ml_stars", 0)
            by_stars[stars].append(judge_ml(r))

    stars_breakdown = {}
    for stars in sorted(by_stars.keys()):
        stars_breakdown[f"{stars}*"] = compute_record(by_stars[stars])

    # Tag 統計（ML 勝負按 tag 分組）
    tag_stats = defaultdict(list)
    for r in records:
        ml_result = judge_ml(r)
        if ml_result is not None:
            for tag in r.get("tags", []):
                tag_stats[tag].append(ml_result)

    tag_breakdown = {}
    for tag, results in sorted(tag_stats.items()):
        tag_breakdown[tag] = compute_record(results)

    return {
        "overall": overall,
        "stars_breakdown": stars_breakdown,
        "tag_breakdown": tag_breakdown,
    }


def generate_daily_detail(records: list) -> list:
    """產生單日每場比賽的明細"""
    details = []
    for r in records:
        away = r.get("away_team", "")
        home = r.get("home_team", "")
        away_abbr = team_to_abbr(away)
        home_abbr = team_to_abbr(home)

        # ML
        ml_rec = r.get("ml_rec", "PASS")
        ml_stars = r.get("ml_stars", 0)
        ml_display = f"{ml_rec} ({ml_stars}*)" if ml_rec != "PASS" else "PASS"
        ml_result = judge_ml(r)
        ml_result = ml_result if ml_result is not None else "\u2014"

        # RL
        rl_rec = r.get("run_line_rec", "PASS")
        rl_display = f"{rl_rec} -1.5" if rl_rec != "PASS" else "PASS"
        rl_result = judge_rl(r)
        rl_result = rl_result if rl_result is not None else "\u2014"

        # O/U
        ou_rec = r.get("ou_rec", "PASS")
        ou_line = r.get("ou_line", "")
        ou_display = f"{ou_rec} ({ou_line})" if ou_rec != "PASS" else f"PASS ({ou_line})"
        ou_result = judge_ou(r)
        ou_result = ou_result if ou_result is not None else "\u2014"

        # 比分
        pred_away = r.get("predicted_away_score", 0)
        pred_home = r.get("predicted_home_score", 0)
        actual_away = r.get("actual_away_score", 0)
        actual_home = r.get("actual_home_score", 0)

        details.append({
            "game": f"{away_abbr} @ {home_abbr}",
            "ml_rec": ml_display,
            "ml_result": ml_result,
            "rl_rec": rl_display,
            "rl_result": rl_result,
            "ou_rec": ou_display,
            "ou_result": ou_result,
            "predicted_score": f"{pred_away}-{pred_home}",
            "actual_score": f"{actual_away}-{actual_home}",
        })

    return details


def main():
    parser = argparse.ArgumentParser(description="MLB Post-Game Review Stats")
    parser.add_argument("--date", required=True, help="日期 YYYY-MM-DD")
    parser.add_argument("--mode", choices=["daily", "cumulative", "both"],
                        default="both", help="輸出模式")
    args = parser.parse_args()

    all_records = load_predictions()
    verified = filter_verified(all_records)

    output = {}

    if args.mode in ("daily", "both"):
        daily_records = filter_by_date(verified, args.date)
        if not daily_records:
            # 回退：直接讀該日 jsonl（可能尚未 verified 但仍想檢視）
            daily_path = daily_jsonl_path(args.date)
            if os.path.exists(daily_path):
                daily_records = filter_verified(load_predictions(daily_path))
        daily_stats = compute_stats(daily_records)
        daily_detail = generate_daily_detail(daily_records)
        output["daily"] = {
            "date": args.date,
            "games": len(daily_records),
            "detail": daily_detail,
            "stats": daily_stats,
        }

    if args.mode in ("cumulative", "both"):
        cumulative_stats = compute_stats(verified)
        output["cumulative"] = {
            "total_verified": len(verified),
            "date_range": {
                "from": verified[0]["date"] if verified else None,
                "to": verified[-1]["date"] if verified else None,
            },
            "stats": cumulative_stats,
        }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
