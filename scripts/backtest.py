#!/usr/bin/env python3
"""MLB Skill Backtest — entry point.

用法：
  python scripts/backtest.py run --month 2026-05
  python scripts/backtest.py run --month 2026-05 --days 2026-05-02,2026-05-03
  python scripts/backtest.py run --month 2026-05 --out /tmp/out
"""

import argparse
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.load import build_dataframe_for_month
from lib.metrics import (
    compute_direction_metrics, compute_total_metrics,
    compute_calibration, compute_slice_metrics,
)
from lib.diagnostic import select_failure_cases, extract_main_signal
from lib.render import render_report, render_details_csv


def _attach_main_signal(failure_cases, df):
    """For each failure row, read summary.md and extract main signal."""
    if len(failure_cases) == 0:
        failure_cases["main_signal"] = []
        return failure_cases
    signals = []
    for _, r in failure_cases.iterrows():
        summary_path = SKILL_ROOT / "analysis-data" / r["date"] / r["matchup"] / "summary.md"
        if summary_path.exists():
            signals.append(extract_main_signal(summary_path.read_text(encoding="utf-8")))
        else:
            signals.append("")
    failure_cases["main_signal"] = signals
    return failure_cases


def cmd_run(args):
    days_filter = set(args.days.split(",")) if args.days else None
    out_dir = Path(args.out) if args.out else SKILL_ROOT / "analysis-data" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for month={args.month}, days={days_filter or 'all'}...")
    df = build_dataframe_for_month(month=args.month, days_filter=days_filter)
    print(f"Loaded {len(df)} rows.")

    print("Computing metrics...")
    dm = compute_direction_metrics(df)
    tm = compute_total_metrics(df)
    cal = compute_calibration(df)
    slices = compute_slice_metrics(df)

    print("Selecting failure cases...")
    failures = select_failure_cases(df, top_total_miss=10)
    failures = _attach_main_signal(failures, df)

    print("Rendering...")
    report_path = out_dir / f"{args.month}-report.md"
    csv_path = out_dir / f"{args.month}-details.csv"
    render_report(df=df, direction_metrics=dm, total_metrics=tm,
                  calibration=cal, slices=slices, failure_cases=failures,
                  month=args.month, out_path=report_path)
    render_details_csv(df, out_path=csv_path)

    print(f"Report: {report_path}")
    print(f"CSV:    {csv_path}")
    print(f"Valid:  {((~df['closing_missing']) & (~df['result_missing'])).sum()} / {len(df)}")


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
