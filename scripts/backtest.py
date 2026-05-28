#!/usr/bin/env python3
"""MLB Skill Backtest — entry point(v2:讀 features.json,聚焦 RL / O/U / edge)。

用法：
  python scripts/backtest.py run --month 2026-05
  python scripts/backtest.py run --month 2026-05 --days 2026-05-02,2026-05-03
  python scripts/backtest.py run --month 2026-05 --out /tmp/out
"""

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.load import build_dataframe_for_month
from lib.metrics import compute_rl_metrics, compute_ou_metrics, compute_edge_calibration
from lib.render import render_report, render_details_csv


def cmd_run(args):
    days_filter = set(args.days.split(",")) if args.days else None
    out_dir = Path(args.out) if args.out else SKILL_ROOT / "analysis-data" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for month={args.month}, days={days_filter or 'all'}...")
    df = build_dataframe_for_month(month=args.month, days_filter=days_filter)
    print(f"Loaded {len(df)} rows.")

    rl = compute_rl_metrics(df)
    ou = compute_ou_metrics(df)
    edge = compute_edge_calibration(df)

    report_path = out_dir / f"{args.month}-report.md"
    csv_path = out_dir / f"{args.month}-details.csv"
    render_report(df=df, rl=rl, ou=ou, edge=edge, month=args.month, out_path=report_path)
    render_details_csv(df, out_path=csv_path)

    print(f"Report: {report_path}")
    print(f"CSV:    {csv_path}")
    if len(df):
        valid = int(((~df["odds_missing"]) & (~df["result_missing"])).sum())
        print(f"Valid (odds+result): {valid} / {len(df)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--month", required=True, help="YYYY-MM")
    p_run.add_argument("--days", help="comma-separated YYYY-MM-DD, optional")
    p_run.add_argument("--out", help="output directory (default: analysis-data/backtest/)")
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
