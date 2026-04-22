#!/usr/bin/env python3
"""RL 門檻放寬 — 方案 A/B/C 回測比較

方案：
  A  強 tag only         : conf=LOW AND |diff|>=1.3 AND 至少一個強 tag
  B  純 diff             : conf=LOW AND |diff|>=1.3
  C  混合（推薦）        : conf=LOW AND |diff|>=1.3 AND (|diff|>=2.5 OR 強 tag)

星級：|diff|<=2.0 → 1★；>2.0 → 2★
強 tag 集合：{home/away}-pitching-slump + {home/away}-bullpen-slump（後者以 merged.json bullpen_era>=5.0 模擬，因 Step 1 尚未實作）

用法：
  python scripts/_backtest_rl_relaxation.py                # 預設方案 C
  python scripts/_backtest_rl_relaxation.py --option A|B|C
  python scripts/_backtest_rl_relaxation.py --compare      # 三方案並列
"""
import argparse
import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

SKILL_ROOT = Path(__file__).parent.parent
ANALYSIS = SKILL_ROOT / "analysis-data"

STRONG_TAGS = {
    "home-pitching-slump", "away-pitching-slump",
    "home-bullpen-slump", "away-bullpen-slump",
}
DIFF_MIN = 1.5
DIFF_BIG = 2.2
DIFF_STAR = 2.0
BULLPEN_SLUMP_ERA = 5.0


def load_records(dates):
    recs = []
    for date in dates:
        date_dir = ANALYSIS / date
        if not date_dir.is_dir():
            continue
        for folder in sorted(os.listdir(date_dir)):
            pred_path = date_dir / folder / "prediction.json"
            merged_path = date_dir / folder / "merged.json"
            if not pred_path.is_file():
                continue
            with pred_path.open(encoding="utf-8") as f:
                p = json.load(f)
            if not p.get("verified"):
                continue
            m = {}
            if merged_path.is_file():
                with merged_path.open(encoding="utf-8") as f:
                    m = json.load(f)
            recs.append((folder, p, m))
    return recs


def simulate_bullpen_slump_tags(merged):
    """模擬 Step 1 實作後會產生的 bullpen-slump tag"""
    tags = set()
    for side in ("home", "away"):
        era = merged.get(f"{side}_bullpen_era")
        if era is not None and era >= BULLPEN_SLUMP_ERA:
            tags.add(f"{side}-bullpen-slump")
    return tags


def judge_rl(fav, actual_home, actual_away):
    margin = actual_home - actual_away
    if fav == "HOME":
        return "WIN" if margin >= 2 else "LOSS"
    return "WIN" if margin <= -2 else "LOSS"


def evaluate(option, pred, merged):
    """回傳 (triggered, stars, fav, reason)"""
    conf = pred.get("confidence")
    ph = pred["predicted_home_score"]
    pa = pred["predicted_away_score"]
    diff = ph - pa
    adiff = abs(diff)
    if conf != "LOW" or adiff < DIFF_MIN:
        return False, None, None, "skip"

    existing = set(pred.get("tags") or [])
    all_strong = (STRONG_TAGS & existing) | simulate_bullpen_slump_tags(merged)

    if option == "A":
        if not all_strong:
            return False, None, None, "no-strong-tag"
        reason = f"strong-tag ({sorted(all_strong)})"
    elif option == "B":
        reason = "pure-diff"
    elif option == "C":
        if adiff >= DIFF_BIG:
            reason = f"big-diff (>={DIFF_BIG})"
        elif all_strong:
            reason = f"mid-diff+strong-tag ({sorted(all_strong)})"
        else:
            return False, None, None, "mid-diff-no-strong-tag"
    else:
        raise ValueError(f"unknown option {option}")

    stars = 2 if adiff > DIFF_STAR else 1
    fav = "HOME" if diff > 0 else "AWAY"
    return True, stars, fav, reason


def run(option, recs, verbose=True):
    triggered = []
    for folder, p, m in recs:
        ok, stars, fav, reason = evaluate(option, p, m)
        if ok:
            res = judge_rl(fav, p["actual_home_score"], p["actual_away_score"])
            triggered.append({
                "date": p["date"],
                "game": folder,
                "diff": round(p["predicted_home_score"] - p["predicted_away_score"], 1),
                "stars": stars,
                "fav": fav,
                "reason": reason,
                "actual": f"{p['actual_away_score']}-{p['actual_home_score']}",
                "result": res,
            })
    w = sum(1 for t in triggered if t["result"] == "WIN")
    l = len(triggered) - w
    total = len(recs)
    return {
        "option": option,
        "total_games": total,
        "triggered": len(triggered),
        "trigger_rate": len(triggered) / total * 100 if total else 0,
        "wins": w,
        "losses": l,
        "win_pct": w / len(triggered) * 100 if triggered else 0,
        "rows": triggered,
    }


def print_summary(result):
    print(f"\n=== 方案 {result['option']} ===")
    print(f"總驗證場次：{result['total_games']}")
    print(f"觸發場次：{result['triggered']} 場（觸發率 {result['trigger_rate']:.1f}%）")
    print(f"戰績：{result['wins']}W-{result['losses']}L  ({result['win_pct']:.1f}%)")
    if result["rows"]:
        print(f"\n{'Date':<12}{'Game':<10}{'Diff':<8}{'★':<4}{'Fav':<6}{'Reason':<55}{'Actual':<9}{'Result'}")
        print("-" * 110)
        for r in result["rows"]:
            print(f"{r['date']:<12}{r['game']:<10}{r['diff']:<+8.1f}{r['stars']}★  "
                  f"{r['fav']:<6}{r['reason'][:54]:<55}{r['actual']:<9}{r['result']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["A", "B", "C"], default="C")
    parser.add_argument("--compare", action="store_true", help="並列比較三方案")
    parser.add_argument("--dates", nargs="+", default=["2026-04-18", "2026-04-19"])
    args = parser.parse_args()

    recs = load_records(args.dates)
    if not recs:
        print(f"ERROR: 找不到任何已驗證紀錄於 {args.dates}", file=sys.stderr)
        sys.exit(1)

    if args.compare:
        results = [run(opt, recs) for opt in ("A", "B", "C")]
        print(f"\n{'方案':<6}{'觸發':<8}{'觸發率':<10}{'戰績':<10}{'勝率'}")
        print("-" * 50)
        for r in results:
            wl = f"{r['wins']}W-{r['losses']}L"
            print(f"{r['option']:<6}{r['triggered']:<8}{r['trigger_rate']:<10.1f}{wl:<10}{r['win_pct']:.1f}%")
        for r in results:
            print_summary(r)
    else:
        print_summary(run(args.option, recs))


if __name__ == "__main__":
    main()
