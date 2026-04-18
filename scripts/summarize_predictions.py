#!/usr/bin/env python3
"""MLB Prediction Summarizer — 掃 analysis-data/**/prediction.json 重建每日 predictions.jsonl

用法：
  python summarize_predictions.py --date 2026-04-16    # 重建單日
  python summarize_predictions.py --all                # 重建所有日期

輸出：analysis-data/{date}/predictions.jsonl（全量重建，按 game_time 排序）
"""

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
SKILL_ROOT = os.path.dirname(SCRIPT_DIR)
ANALYSIS_DATA_DIR = os.path.join(SKILL_ROOT, "analysis-data")


def scan_predictions() -> list[tuple[str, dict]]:
    """掃描所有 per-game prediction.json，回傳 [(path, record), ...]"""
    pattern = os.path.join(ANALYSIS_DATA_DIR, "*", "*", "prediction.json")
    records = []
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                records.append((path, json.load(f)))
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: 略過 {path}：{e}", file=sys.stderr)
    return records


def group_by_date(records: list[tuple[str, dict]]) -> dict[str, list[dict]]:
    """依 record['date'] 分組（非依資料夾名）"""
    groups = defaultdict(list)
    for _, rec in records:
        date = rec.get("date")
        if date:
            groups[date].append(rec)
    return groups


def sort_by_time(records: list[dict]) -> list[dict]:
    """依 game_time 排序，game_time 缺失者排最後"""
    return sorted(records, key=lambda r: (r.get("game_time") or "z", r.get("game_pk") or 0))


def write_jsonl(date: str, records: list[dict]) -> str:
    """寫入 analysis-data/{date}/predictions.jsonl，回傳路徑"""
    out_dir = os.path.join(ANALYSIS_DATA_DIR, date)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in sort_by_time(records):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="重建每日 predictions.jsonl summary")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--date", help="單日 YYYY-MM-DD")
    grp.add_argument("--all", action="store_true", help="重建所有日期")
    args = parser.parse_args()

    all_records = scan_predictions()
    if not all_records:
        print(f"ERROR: 找不到任何 prediction.json（{ANALYSIS_DATA_DIR}）", file=sys.stderr)
        sys.exit(1)

    groups = group_by_date(all_records)

    if args.date:
        if args.date not in groups:
            print(f"ERROR: {args.date} 沒有任何 prediction.json", file=sys.stderr)
            sys.exit(1)
        path = write_jsonl(args.date, groups[args.date])
        print(f"[OK] {args.date}: {len(groups[args.date])} 筆 -> {path}")
    else:
        for date in sorted(groups):
            path = write_jsonl(date, groups[date])
            print(f"[OK] {date}: {len(groups[date])} 筆 -> {path}")


if __name__ == "__main__":
    main()
