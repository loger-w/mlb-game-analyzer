#!/usr/bin/env python3
"""OU Total Error 分佈診斷 — 拆分 PASS vs 推薦兩群，識別 total 公式結構問題

用法：
  python diagnose_ou_total_error.py --from 2026-04-18 --to 2026-04-22
  python diagnose_ou_total_error.py --from 2026-04-18 --to 2026-04-22 --format json

輸出：
  - 兩群 abs_error 分佈（n / mean / median / std / 誤差 ≥3 比例 / 方向偏差）
  - signal_adj_delta 分佈（adjusted_total - pre_signal_total）與 abs_error 的相關
  - 逐日 PASS 群 breakdown（方便看趨勢）
"""

import argparse
import json
import os
import statistics
import sys
from datetime import date, datetime, timedelta

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_ROOT = os.path.dirname(SCRIPT_DIR)
ANALYSIS_DATA_DIR = os.path.join(SKILL_ROOT, "analysis-data")


def iter_dates(start: str, end: str):
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    cur = d0
    while cur <= d1:
        yield cur.isoformat()
        cur += timedelta(days=1)


def load_records(date_str: str) -> list[dict]:
    path = os.path.join(ANALYSIS_DATA_DIR, date_str, "predictions.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stats_dict(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    n = len(values)
    return {
        "n": n,
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "std": round(statistics.stdev(values), 2) if n > 1 else 0.0,
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "pct_ge_3": round(100 * sum(1 for v in values if v >= 3) / n, 1),
        "pct_ge_5": round(100 * sum(1 for v in values if v >= 5) / n, 1),
    }


def analyze(records: list[dict]) -> dict:
    verified = [r for r in records if r.get("verified")
                and r.get("actual_total") is not None
                and r.get("predicted_total") is not None]

    rows = []
    for r in verified:
        pred = r["predicted_total"]
        actual = r["actual_total"]
        err = actual - pred  # 正 = OVER（低估）, 負 = UNDER（高估）
        abs_err = abs(err)
        ou_rec = (r.get("ou_rec") or "PASS").upper()
        is_pass = ou_rec == "PASS"
        signal_delta = None
        if r.get("pre_signal_total") is not None and r.get("adjusted_total") is not None:
            signal_delta = r["adjusted_total"] - r["pre_signal_total"]
        rows.append({
            "date": r.get("date"),
            "game": r.get("game"),
            "ou_rec": ou_rec,
            "is_pass": is_pass,
            "predicted_total": pred,
            "actual_total": actual,
            "err_signed": round(err, 2),
            "abs_err": round(abs_err, 2),
            "pre_signal_total": r.get("pre_signal_total"),
            "adjusted_total": r.get("adjusted_total"),
            "signal_delta": round(signal_delta, 2) if signal_delta is not None else None,
            "tags": r.get("tags", []),
        })

    pass_group = [x for x in rows if x["is_pass"]]
    rec_group = [x for x in rows if not x["is_pass"]]

    def group_summary(group: list[dict]) -> dict:
        abs_errs = [x["abs_err"] for x in group]
        signed_errs = [x["err_signed"] for x in group]
        over_count = sum(1 for e in signed_errs if e > 0)  # 實際 > 預測 = 低估 = OVER 方向
        under_count = sum(1 for e in signed_errs if e < 0)
        zero_count = sum(1 for e in signed_errs if e == 0)
        return {
            "abs_err_stats": stats_dict(abs_errs),
            "direction": {
                "actual_over": over_count,  # 低估次數
                "actual_under": under_count,  # 高估次數
                "exact": zero_count,
                "bias_pct": round(100 * (over_count - under_count) / len(group), 1) if group else 0,
            },
            "worst_5": sorted(
                [{k: v for k, v in x.items() if k in ("date", "game", "predicted_total",
                                                      "actual_total", "err_signed", "tags")}
                 for x in group],
                key=lambda x: abs(x["err_signed"]), reverse=True
            )[:5],
        }

    pass_summary = group_summary(pass_group)
    rec_summary = group_summary(rec_group)

    # 逐日 PASS 群誤差 ≥3 計數（看趨勢）
    by_date = {}
    for x in pass_group:
        d = x["date"]
        by_date.setdefault(d, {"n": 0, "ge3": 0, "ge5": 0, "over": 0, "under": 0})
        by_date[d]["n"] += 1
        if x["abs_err"] >= 3:
            by_date[d]["ge3"] += 1
        if x["abs_err"] >= 5:
            by_date[d]["ge5"] += 1
        if x["err_signed"] > 0:
            by_date[d]["over"] += 1
        elif x["err_signed"] < 0:
            by_date[d]["under"] += 1

    # signal_adj delta 分佈（全群）
    sig_deltas = [x["signal_delta"] for x in rows if x["signal_delta"] is not None]
    sig_delta_stats = stats_dict([abs(d) for d in sig_deltas]) if sig_deltas else {"n": 0}

    # signal_delta 與 abs_err 的相關性（簡單版：分桶）
    buckets = {"|Δ| < 0.5": [], "0.5 ≤ |Δ| < 1.0": [], "|Δ| ≥ 1.0": []}
    for x in rows:
        if x["signal_delta"] is None:
            continue
        ad = abs(x["signal_delta"])
        if ad < 0.5:
            buckets["|Δ| < 0.5"].append(x["abs_err"])
        elif ad < 1.0:
            buckets["0.5 ≤ |Δ| < 1.0"].append(x["abs_err"])
        else:
            buckets["|Δ| ≥ 1.0"].append(x["abs_err"])
    bucket_summary = {k: stats_dict(v) for k, v in buckets.items()}

    return {
        "meta": {
            "total_verified": len(verified),
            "pass_n": len(pass_group),
            "rec_n": len(rec_group),
        },
        "pass_group": pass_summary,
        "rec_group": rec_summary,
        "pass_by_date": by_date,
        "signal_adj_delta_abs_stats": sig_delta_stats,
        "abs_err_by_signal_delta_bucket": bucket_summary,
    }


def print_text(result: dict):
    meta = result["meta"]
    print(f"\n=== OU Total Error 診斷 ({meta['total_verified']} 場 verified) ===\n")
    print(f"PASS 群: {meta['pass_n']} 場 | 推薦群: {meta['rec_n']} 場\n")

    for label, key in [("PASS 群", "pass_group"), ("推薦群", "rec_group")]:
        g = result[key]
        print(f"--- {label} ---")
        s = g["abs_err_stats"]
        if s["n"] == 0:
            print("  (無樣本)\n")
            continue
        print(f"  abs_err: mean={s['mean']} median={s['median']} std={s['std']} "
              f"range=[{s['min']}, {s['max']}]")
        print(f"  誤差 ≥3 比例: {s['pct_ge_3']}% | ≥5 比例: {s['pct_ge_5']}%")
        d = g["direction"]
        print(f"  方向: 實際OVER(低估)={d['actual_over']} 實際UNDER(高估)={d['actual_under']} "
              f"bias={d['bias_pct']}%")
        print(f"  最差 5 場:")
        for w in g["worst_5"]:
            print(f"    {w['date']} {w['game']}: 預 {w['predicted_total']} 實 {w['actual_total']} "
                  f"(差 {w['err_signed']:+.2f})")
        print()

    print("--- PASS 群逐日 breakdown ---")
    for d, v in sorted(result["pass_by_date"].items()):
        print(f"  {d}: n={v['n']} ≥3={v['ge3']} ≥5={v['ge5']} "
              f"over={v['over']} under={v['under']}")
    print()

    print("--- signal_adj_delta |Δ| 分佈 ---")
    s = result["signal_adj_delta_abs_stats"]
    if s["n"] > 0:
        print(f"  mean={s['mean']} median={s['median']} range=[{s['min']}, {s['max']}]")
    print()

    print("--- abs_err by |signal_delta| 分桶 ---")
    for bucket, s in result["abs_err_by_signal_delta_bucket"].items():
        if s["n"] == 0:
            print(f"  {bucket}: (無樣本)")
        else:
            print(f"  {bucket}: n={s['n']} mean_abs_err={s['mean']} ≥3={s['pct_ge_3']}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="OU total error 分佈診斷")
    parser.add_argument("--from", dest="date_from", required=True, help="起始日 YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="結束日 YYYY-MM-DD")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    all_records = []
    for d in iter_dates(args.date_from, args.date_to):
        all_records.extend(load_records(d))

    if not all_records:
        print(f"ERROR: 找不到 {args.date_from} ~ {args.date_to} 的資料", file=sys.stderr)
        sys.exit(1)

    result = analyze(all_records)

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_text(result)


if __name__ == "__main__":
    main()
