#!/usr/bin/env python3
"""MLB 預測指標診斷腳本

三段分析：
(A) PASS 召回率：實際推薦 vs 強制全下
(B) 指標校準：ML 分檔命中、O/U MAE、RL margin MAE
(C) Signal 類別 ablation：觸發 vs 未觸發 adjusted_total MAE
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import date as date_cls

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ANALYSIS_LOGS_DIR = os.path.join(PROJECT_DIR, "analysis-logs")

sys.path.insert(0, SCRIPT_DIR)
from review_stats import (  # noqa: E402
    load_predictions,
    filter_verified,
    judge_ml,
    judge_ou,
    judge_rl,
    is_home_team,
    team_to_abbr,
)

DEFAULT_SINCE = "2026-04-18"
SCHEMA_BOUNDARY = "2026-04-18"


# --- Signal 分類 ---
SIGNAL_CATEGORIES = [
    ("bullpen/injury", [re.compile(r".*_il$"), re.compile(r".*bullpen.*"), re.compile(r".*_bp_.*"), re.compile(r".*_bp$")]),
    ("park", [re.compile(r"^park_.*")]),
    ("pitcher quality", [re.compile(r".*xera.*"), re.compile(r".*_era.*"), re.compile(r".*_fip.*"), re.compile(r".*_k_.*"), re.compile(r".*_k$")]),
    ("recent form", [re.compile(r".*_hot$"), re.compile(r".*_cold$"), re.compile(r".*_slump$"), re.compile(r".*_offense$")]),
    ("weather", [re.compile(r"^cold_weather$"), re.compile(r"^warm_weather$"), re.compile(r"^wind.*"), re.compile(r".*_wind.*")]),
]


def classify_signal(key: str) -> str:
    for cat, patterns in SIGNAL_CATEGORIES:
        for p in patterns:
            if p.match(key):
                return cat
    return "other"


# --- 強制下注邏輯 ---

def forced_ml(record: dict) -> str:
    """回傳 WIN / LOSS / None（無法判定）"""
    pred = record.get("predicted_winner")
    actual = record.get("actual_winner")
    if pred is None or actual is None:
        return None
    return "WIN" if pred == actual else "LOSS"


def forced_ou(record: dict) -> str:
    """回傳 WIN / LOSS / PUSH / None"""
    pred_total = record.get("predicted_total")
    line = record.get("ou_line")
    actual_total = record.get("actual_total")
    if pred_total is None or line is None or actual_total is None:
        return None
    if actual_total == line:
        return "PUSH"
    forced_pick = "OVER" if pred_total > line else "UNDER"
    if (forced_pick == "OVER" and actual_total > line) or (forced_pick == "UNDER" and actual_total < line):
        return "WIN"
    return "LOSS"


def forced_rl(record: dict) -> str:
    """依預測 margin 強制取 -1.5 / +1.5。回傳 WIN / LOSS / PUSH / None"""
    pred_home = record.get("predicted_home_score")
    pred_away = record.get("predicted_away_score")
    actual_home = record.get("actual_home_score")
    actual_away = record.get("actual_away_score")
    if pred_home is None or pred_away is None or actual_home is None or actual_away is None:
        return None
    pred_margin = pred_home - pred_away
    actual_margin = actual_home - actual_away
    # 預測贏家
    if pred_margin > 0:
        pick_side = "HOME"
    elif pred_margin < 0:
        pick_side = "AWAY"
    else:
        pick_side = "HOME"  # 平手預測視為主隊

    if abs(pred_margin) >= 1.5:
        # 取 -1.5
        if pick_side == "HOME":
            diff = actual_margin - 1.5
        else:
            diff = -actual_margin - 1.5
        if diff > 0:
            return "WIN"
        if diff < 0:
            return "LOSS"
        return "PUSH"
    else:
        # 取 +1.5
        if pick_side == "HOME":
            diff = actual_margin + 1.5
        else:
            diff = -actual_margin + 1.5
        if diff > 0:
            return "WIN"
        if diff < 0:
            return "LOSS"
        return "PUSH"


def record_wl(results: list) -> tuple[int, int, int]:
    w = sum(1 for r in results if r == "WIN")
    l = sum(1 for r in results if r == "LOSS")
    p = sum(1 for r in results if r == "PUSH")
    return w, l, p


# --- (A) PASS 召回率 ---

def analyze_pass_recall(records: list) -> dict:
    """對每盤口計算實際推薦 vs 強制全下"""
    out = {}
    for market, actual_fn, forced_fn, rec_key in [
        ("ML", judge_ml, forced_ml, "ml_rec"),
        ("O/U", judge_ou, forced_ou, "ou_rec"),
        ("RL", judge_rl, forced_rl, "run_line_rec"),
    ]:
        actual_results = []
        forced_results = []
        pass_but_correct = []  # (record, tag_reason)
        for r in records:
            a = actual_fn(r)
            if a is not None:
                actual_results.append(a)
            f = forced_fn(r)
            if f is not None:
                forced_results.append(f)
            # PASS 但方向正確
            rec = r.get(rec_key)
            is_pass = (rec is None or rec == "PASS")
            if is_pass and f == "WIN":
                pass_but_correct.append(r)

        aw, al, ap = record_wl(actual_results)
        fw, fl, fp = record_wl(forced_results)
        out[market] = {
            "actual": (aw, al, ap),
            "forced": (fw, fl, fp),
            "delta_w": fw - aw,
            "delta_l": fl - al,
            "pass_but_correct": pass_but_correct,
        }
    return out


def format_pass_reason(record: dict) -> str:
    parts = []
    conf = record.get("confidence")
    if conf:
        parts.append(conf)
    xval = record.get("cross_validation")
    if xval and xval != "OK":
        parts.append(xval)
    tags = record.get("tags", [])
    interesting_tags = [t for t in tags if t in ("divergent", "insufficient-sample", "low-confidence")]
    parts.extend(interesting_tags)
    return ", ".join(parts) if parts else "-"


# --- (B) 指標校準 ---

def analyze_calibration(records: list) -> dict:
    out = {}

    # ML: predicted_home_pct 分檔
    bins = [
        ("< 50%", lambda p: p < 50),
        ("50–55%", lambda p: 50 <= p < 55),
        ("55–60%", lambda p: 55 <= p < 60),
        ("60–65%", lambda p: 60 <= p < 65),
        ("≥ 65%", lambda p: p >= 65),
    ]
    ml_bins = {label: {"n": 0, "hits": 0} for label, _ in bins}
    for r in records:
        pct = r.get("predicted_home_pct")
        actual = r.get("actual_winner")
        if pct is None or actual is None:
            continue
        predicted_side = "HOME" if pct >= 50 else "AWAY"
        hit = 1 if predicted_side == actual else 0
        for label, cond in bins:
            if cond(pct):
                ml_bins[label]["n"] += 1
                ml_bins[label]["hits"] += hit
                break
    out["ml_bins"] = ml_bins

    # O/U MAE
    def mae(pairs):
        vals = [abs(a - b) for a, b in pairs if a is not None and b is not None]
        return sum(vals) / len(vals) if vals else None

    formula_pairs = []
    adjusted_pairs = []
    predicted_pairs = []
    pre_signal_pairs = []
    ou_dir_correct = 0
    ou_dir_total = 0
    for r in records:
        actual_total = r.get("actual_total")
        if actual_total is None:
            continue
        fh = r.get("formula_home_score")
        fa = r.get("formula_away_score")
        if fh is not None and fa is not None:
            formula_pairs.append((fh + fa, actual_total))
        adj = r.get("adjusted_total")
        if adj is not None:
            adjusted_pairs.append((adj, actual_total))
        pred = r.get("predicted_total")
        if pred is not None:
            predicted_pairs.append((pred, actual_total))
        pre = r.get("pre_signal_total")
        if pre is not None:
            pre_signal_pairs.append((pre, actual_total))

        line = r.get("ou_line")
        if pred is not None and line is not None and actual_total != line:
            pred_dir = "OVER" if pred > line else "UNDER"
            actual_dir = "OVER" if actual_total > line else "UNDER"
            ou_dir_total += 1
            if pred_dir == actual_dir:
                ou_dir_correct += 1

    out["ou"] = {
        "formula_mae": mae(formula_pairs),
        "formula_n": len(formula_pairs),
        "adjusted_mae": mae(adjusted_pairs),
        "adjusted_n": len(adjusted_pairs),
        "predicted_mae": mae(predicted_pairs),
        "predicted_n": len(predicted_pairs),
        "pre_signal_mae": mae(pre_signal_pairs),
        "pre_signal_n": len(pre_signal_pairs),
        "direction_accuracy": (ou_dir_correct / ou_dir_total) if ou_dir_total else None,
        "direction_n": ou_dir_total,
    }

    # RL margin
    margin_pairs = []
    rl_winner_correct = 0
    rl_winner_total = 0
    home_side_pairs = []
    away_side_pairs = []
    for r in records:
        ph = r.get("predicted_home_score")
        pa = r.get("predicted_away_score")
        ah = r.get("actual_home_score")
        aa = r.get("actual_away_score")
        if None in (ph, pa, ah, aa):
            continue
        margin_pairs.append((ph - pa, ah - aa))
        home_side_pairs.append((ph, ah))
        away_side_pairs.append((pa, aa))
        pred_winner = "HOME" if ph > pa else ("AWAY" if pa > ph else None)
        actual_winner = r.get("actual_winner")
        if pred_winner and actual_winner:
            rl_winner_total += 1
            if pred_winner == actual_winner:
                rl_winner_correct += 1
    out["rl"] = {
        "margin_mae": mae(margin_pairs),
        "margin_n": len(margin_pairs),
        "winner_accuracy": (rl_winner_correct / rl_winner_total) if rl_winner_total else None,
        "winner_n": rl_winner_total,
        "home_side_mae": mae(home_side_pairs),
        "away_side_mae": mae(away_side_pairs),
    }

    # XGB raw vs final pct（若可用）
    xgb_vs_final = []
    for r in records:
        xgb = r.get("xgb_raw_home_pct")
        final = r.get("predicted_home_pct")
        actual = r.get("actual_winner")
        if xgb is None or final is None or actual is None:
            continue
        xgb_side = "HOME" if xgb >= 50 else "AWAY"
        final_side = "HOME" if final >= 50 else "AWAY"
        xgb_vs_final.append({
            "xgb_pct": xgb,
            "final_pct": final,
            "xgb_correct": xgb_side == actual,
            "final_correct": final_side == actual,
            "agree": xgb_side == final_side,
        })
    out["xgb_vs_final"] = xgb_vs_final

    return out


# --- (C) Signal ablation ---

def analyze_signal_ablation(records: list) -> dict:
    """只對 post-schema（有 signal_adjustments 非空）紀錄分析"""
    eligible = [r for r in records if r.get("signal_adjustments")]

    per_category = defaultdict(lambda: {
        "triggers": 0,
        "sum_contrib": 0.0,
        "game_ids": set(),
    })

    for i, r in enumerate(eligible):
        sigs = r.get("signal_adjustments", {})
        for key, val in sigs.items():
            cat = classify_signal(key)
            per_category[cat]["triggers"] += 1
            per_category[cat]["sum_contrib"] += val
            per_category[cat]["game_ids"].add(i)

    def mae(pairs):
        vals = [abs(a - b) for a, b in pairs if a is not None and b is not None]
        return sum(vals) / len(vals) if vals else None

    results = {}
    for cat, info in per_category.items():
        triggered_ids = info["game_ids"]
        triggered_pairs = []
        untriggered_pairs = []
        for i, r in enumerate(eligible):
            adj = r.get("adjusted_total")
            actual = r.get("actual_total")
            if adj is None or actual is None:
                continue
            if i in triggered_ids:
                triggered_pairs.append((adj, actual))
            else:
                untriggered_pairs.append((adj, actual))
        results[cat] = {
            "triggers": info["triggers"],
            "games_touched": len(triggered_ids),
            "avg_contrib": (info["sum_contrib"] / info["triggers"]) if info["triggers"] else 0.0,
            "triggered_mae": mae(triggered_pairs),
            "triggered_n": len(triggered_pairs),
            "untriggered_mae": mae(untriggered_pairs),
            "untriggered_n": len(untriggered_pairs),
        }

    return {"eligible_n": len(eligible), "by_category": results}


# --- 報告輸出 ---

def fmt_num(x, places=2):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{places}f}"
    return str(x)


def fmt_pct(x):
    if x is None:
        return "—"
    return f"{x * 100:.1f}%"


def build_report(records: list, since: str, include_legacy: bool) -> str:
    from io import StringIO
    buf = StringIO()

    today = date_cls.today().isoformat()
    buf.write(f"# 指標診斷報告 - {today}\n\n")

    # 樣本範圍
    buf.write("## 樣本範圍\n\n")
    dates = sorted({r.get("date") for r in records if r.get("date")})
    buf.write(f"- 起日：{dates[0] if dates else '—'}\n")
    buf.write(f"- 止日：{dates[-1] if dates else '—'}\n")
    buf.write(f"- 總場次：{len(records)}\n")
    buf.write(f"- 過濾條件：`date >= {since}`{'（含 legacy schema）' if include_legacy else ''}\n")
    ml_rec_n = sum(1 for r in records if r.get("ml_rec") and r["ml_rec"] != "PASS")
    ou_rec_n = sum(1 for r in records if r.get("ou_rec") and r["ou_rec"] != "PASS")
    rl_rec_n = sum(1 for r in records if r.get("run_line_rec") and r["run_line_rec"] != "PASS")
    buf.write(f"- ML 推薦場次：{ml_rec_n}\n")
    buf.write(f"- O/U 推薦場次：{ou_rec_n}\n")
    buf.write(f"- RL 推薦場次：{rl_rec_n}\n\n")

    # (A)
    buf.write("## (A) PASS 召回率\n\n")
    pass_recall = analyze_pass_recall(records)
    buf.write("| 盤口 | 實際推薦 W-L-P | 強制全下 W-L-P | 差距（W/L） |\n")
    buf.write("|------|---------------|----------------|-------------|\n")
    for market in ("ML", "O/U", "RL"):
        info = pass_recall[market]
        aw, al, ap = info["actual"]
        fw, fl, fp = info["forced"]
        buf.write(f"| {market} | {aw}-{al}-{ap} | {fw}-{fl}-{fp} | +{info['delta_w']} W / +{info['delta_l']} L |\n")
    buf.write("\n")

    for market in ("ML", "O/U", "RL"):
        pbc = pass_recall[market]["pass_but_correct"]
        if not pbc:
            continue
        buf.write(f"### {market} PASS 但方向正確（{len(pbc)} 場）\n\n")
        buf.write("| 比賽 | 預測 | 實際 | PASS 原因 |\n")
        buf.write("|------|------|------|----------|\n")
        for r in pbc:
            away_abbr = team_to_abbr(r.get("away_team", ""))
            home_abbr = team_to_abbr(r.get("home_team", ""))
            game = f"{away_abbr} @ {home_abbr}"
            if market == "ML":
                pred = r.get("predicted_winner", "—")
                actual = r.get("actual_winner", "—")
            elif market == "O/U":
                pt = r.get("predicted_total")
                line = r.get("ou_line")
                at = r.get("actual_total")
                pred = f"{pt} vs {line}"
                actual = f"{at}"
            else:
                ph = r.get("predicted_home_score")
                pa = r.get("predicted_away_score")
                ah = r.get("actual_home_score")
                aa = r.get("actual_away_score")
                pred = f"{ph}-{pa}"
                actual = f"{ah}-{aa}"
            buf.write(f"| {game} | {pred} | {actual} | {format_pass_reason(r)} |\n")
        buf.write("\n")

    # (B)
    buf.write("## (B) 指標校準\n\n")
    cal = analyze_calibration(records)

    buf.write("### ML：`predicted_home_pct` 分檔命中率\n\n")
    buf.write("| 分檔 | N | 命中 | 命中率 |\n|------|---|------|--------|\n")
    for label, info in cal["ml_bins"].items():
        n = info["n"]
        hits = info["hits"]
        pct = f"{hits/n*100:.1f}%" if n >= 3 else "insufficient"
        buf.write(f"| {label} | {n} | {hits} | {pct} |\n")
    buf.write("\n")

    buf.write("### O/U：總分預測 MAE\n\n")
    ou = cal["ou"]
    buf.write("| 指標 | N | MAE |\n|------|---|-----|\n")
    buf.write(f"| formula_total (home+away pre-signal) | {ou['formula_n']} | {fmt_num(ou['formula_mae'])} |\n")
    if ou["pre_signal_n"] > 0:
        buf.write(f"| pre_signal_total (直接欄位) | {ou['pre_signal_n']} | {fmt_num(ou['pre_signal_mae'])} |\n")
    buf.write(f"| adjusted_total | {ou['adjusted_n']} | {fmt_num(ou['adjusted_mae'])} |\n")
    buf.write(f"| predicted_total | {ou['predicted_n']} | {fmt_num(ou['predicted_mae'])} |\n\n")
    buf.write(f"方向準確率（vs ou_line）：{fmt_pct(ou['direction_accuracy'])}（N={ou['direction_n']}）\n\n")

    buf.write("### RL：margin 與 per-side\n\n")
    rl = cal["rl"]
    buf.write(f"- margin MAE：{fmt_num(rl['margin_mae'])}（N={rl['margin_n']}）\n")
    buf.write(f"- winner 一致率：{fmt_pct(rl['winner_accuracy'])}（N={rl['winner_n']}）\n")
    buf.write(f"- home 得分 MAE：{fmt_num(rl['home_side_mae'])}\n")
    buf.write(f"- away 得分 MAE：{fmt_num(rl['away_side_mae'])}\n\n")

    xgb_list = cal.get("xgb_vs_final", [])
    if xgb_list:
        buf.write("### XGB raw vs final pct\n\n")
        xgb_hits = sum(1 for x in xgb_list if x["xgb_correct"])
        final_hits = sum(1 for x in xgb_list if x["final_correct"])
        agree = sum(1 for x in xgb_list if x["agree"])
        n = len(xgb_list)
        buf.write(f"- N={n}, xgb 命中 {xgb_hits} ({xgb_hits/n*100:.1f}%), final 命中 {final_hits} ({final_hits/n*100:.1f}%), 兩者同方向 {agree}/{n}\n\n")
    else:
        buf.write("### XGB raw vs final pct\n\n（`xgb_raw_home_pct` 欄位不在樣本中，跳過）\n\n")

    # (C)
    buf.write("## (C) Signal 類別 ablation\n\n")
    abl = analyze_signal_ablation(records)
    buf.write(f"納入分析場次（signal_adjustments 非空）：{abl['eligible_n']}\n\n")
    if not abl["by_category"]:
        buf.write("（無可分析的 signal）\n\n")
    else:
        buf.write("| 類別 | 觸發次數 | 涉及場次 | 平均貢獻 | 觸發 MAE (N) | 未觸發 MAE (N) | Δ |\n")
        buf.write("|------|---------|---------|---------|-------------|----------------|---|\n")
        sorted_cats = sorted(abl["by_category"].items(), key=lambda kv: -kv[1]["triggers"])
        for cat, info in sorted_cats:
            t_mae = info["triggered_mae"]
            u_mae = info["untriggered_mae"]
            delta = (t_mae - u_mae) if (t_mae is not None and u_mae is not None) else None
            buf.write(
                f"| {cat} | {info['triggers']} | {info['games_touched']} | "
                f"{fmt_num(info['avg_contrib'], 2)} | "
                f"{fmt_num(t_mae)} ({info['triggered_n']}) | "
                f"{fmt_num(u_mae)} ({info['untriggered_n']}) | "
                f"{fmt_num(delta)} |\n"
            )
        buf.write("\n註：Δ > 0 代表觸發此類 signal 的場次 adjusted_total MAE 反而更差（可能負貢獻）。\n\n")

    # 結論摘要
    buf.write("## 結論摘要\n\n")
    findings = []
    for market in ("ML", "O/U", "RL"):
        info = pass_recall[market]
        aw, al, _ = info["actual"]
        fw, fl, _ = info["forced"]
        actual_n = aw + al
        forced_n = fw + fl
        # 強制全下比實際多贏多少場（占總場次比）
        extra_w = fw - aw
        if forced_n >= 5 and extra_w / max(forced_n, 1) >= 0.3:
            findings.append(f"- **{market} PASS 門檻可能過嚴**：強制全下多贏 {extra_w} 場（{extra_w/forced_n*100:.0f}% of forced）。")

    ou = cal["ou"]
    if ou["formula_mae"] is not None and ou["adjusted_mae"] is not None:
        if ou["formula_mae"] < ou["adjusted_mae"]:
            findings.append(
                f"- **Signal 層負貢獻**：formula MAE {ou['formula_mae']:.2f} < adjusted MAE {ou['adjusted_mae']:.2f}。"
            )

    abl_results = abl["by_category"]
    for cat, info in abl_results.items():
        t_mae = info["triggered_mae"]
        u_mae = info["untriggered_mae"]
        if t_mae is not None and u_mae is not None and info["triggered_n"] >= 3:
            if t_mae - u_mae >= 0.5:
                findings.append(
                    f"- **`{cat}` 類 signal 觸發後 MAE 顯著惡化**："
                    f"{t_mae:.2f} vs {u_mae:.2f}（Δ +{t_mae - u_mae:.2f}）。"
                )

    if findings:
        buf.write("\n".join(findings) + "\n")
    else:
        buf.write("（未觸發任何自動結論閾值；樣本可能仍太小，請累積更多資料。）\n")

    return buf.getvalue()


# --- main ---

def main():
    parser = argparse.ArgumentParser(description="MLB 預測指標診斷")
    parser.add_argument("--since", default=DEFAULT_SINCE, help=f"起日 YYYY-MM-DD（預設 {DEFAULT_SINCE}）")
    parser.add_argument("--include-legacy", action="store_true",
                        help="納入所有已驗證場次（pre-schema 紀錄會在 signal ablation 自動跳過）")
    parser.add_argument("--output", default=None, help="輸出路徑（預設 analysis-logs/diagnostic-{today}.md）")
    args = parser.parse_args()

    all_records = load_predictions()
    verified = filter_verified(all_records)

    if args.include_legacy:
        records = verified
    else:
        records = [r for r in verified if (r.get("date") or "") >= args.since]

    if not records:
        print("⚠️ 沒有符合條件的紀錄", file=sys.stderr)
        sys.exit(1)

    report = build_report(records, args.since, args.include_legacy)

    today = date_cls.today().isoformat()
    if args.output:
        out_path = args.output
    else:
        os.makedirs(ANALYSIS_LOGS_DIR, exist_ok=True)
        out_path = os.path.join(ANALYSIS_LOGS_DIR, f"diagnostic-{today}.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ 報告已輸出：{out_path}", file=sys.stderr)
    print(f"   樣本 N={len(records)}", file=sys.stderr)


if __name__ == "__main__":
    main()
