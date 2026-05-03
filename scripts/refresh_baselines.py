#!/usr/bin/env python3
"""scripts/refresh_baselines.py — yearly tool to regenerate league_pitcher_baseline.json.

Usage:
    python scripts/refresh_baselines.py --year 2025 \
        --output scripts/data/league_pitcher_baseline.json

Run manually each February before the season opens. The output JSON is the
single source of truth for tier_v2 percentile lookups (consumed by
lib_tier_v2.py from PR-2 onward). Do NOT auto-cron — silent baseline drift
mid-season would invalidate historical tier comparisons.

Convention: percentile keys p10..p90 always reference top-decile-first
(p10 = best, p90 = worst). For lower_is_better metrics (xFIP), this means p10
has the smallest value; for higher_is_better (K-BB%, FBv) p10 has the largest.
"""

import argparse
import json
import sys
from datetime import datetime, timezone


METRIC_SPEC = (
    # (json_key, fangraphs_column, direction)
    ("xfip", "xFIP", "lower_is_better"),
    ("k_bb_pct", "K-BB%", "higher_is_better"),
    ("avg_velo", "FBv", "higher_is_better"),
    # Stuff+/Pitching+ are FanGraphs composite metrics (velo+spin+movement / +location).
    # 100 = league average. Drive tier_v2 score (Stuff+ refactor 2026-05-03).
    ("stuff_plus", "Stuff+", "higher_is_better"),
    ("pitching_plus", "Pitching+", "higher_is_better"),
)


def _percentile_block(series, direction: str) -> dict:
    """Compute p10/p25/p50/p75/p90 with top-decile-first convention.

    For lower_is_better, p10 is the 10th percentile from bottom (smallest).
    For higher_is_better, p10 is the 90th percentile from bottom (largest).
    """
    import numpy as np

    if direction == "lower_is_better":
        rank_pcts = (10, 25, 50, 75, 90)
    else:
        rank_pcts = (90, 75, 50, 25, 10)

    keys = ("p10", "p25", "p50", "p75", "p90")
    out = {"direction": direction}
    for key, pct in zip(keys, rank_pcts):
        out[key] = round(float(np.percentile(series, pct)), 2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh league_pitcher_baseline.json")
    parser.add_argument("--year", type=int, required=True, help="Season to compute baseline from")
    parser.add_argument("--qual", type=int, default=50, help="Min IP to qualify (default 50)")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path")
    args = parser.parse_args()

    try:
        from pybaseball import pitching_stats
    except ImportError:
        print("⛔ pybaseball not installed. Run: pip install pybaseball", file=sys.stderr)
        return 1

    print(f"Fetching pitching_stats(year={args.year}, qual={args.qual}) ...", file=sys.stderr)
    df = pitching_stats(args.year, qual=args.qual)
    if df.empty:
        print(f"⛔ No qualifying pitchers found for year={args.year}", file=sys.stderr)
        return 2

    metrics_out = {}
    missing_cols = []
    for json_key, fg_col, direction in METRIC_SPEC:
        if fg_col not in df.columns:
            missing_cols.append(fg_col)
            continue
        series = df[fg_col].dropna()
        if series.empty:
            missing_cols.append(fg_col)
            continue
        metrics_out[json_key] = _percentile_block(series, direction)

    if missing_cols:
        print(f"⚠️ FanGraphs columns missing: {missing_cols}", file=sys.stderr)

    output = {
        "year": args.year,
        "qualifier_min_ip": args.qual,
        "metrics": metrics_out,
        "metadata": {
            "source": f"pybaseball.pitching_stats({args.year}, qual={args.qual}) — FanGraphs leaderboard",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "refresh_command": f"python scripts/refresh_baselines.py --year {args.year} --output {args.output}",
            "notes": (
                "Convention: p10 = top decile (best). For lower_is_better metrics (xfip), "
                "p10 has the smallest value. For higher_is_better metrics (k_bb_pct, avg_velo), "
                "p10 has the largest value. Refresh annually each February before season opens."
            ),
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {args.output} (n={len(df)} pitchers)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
