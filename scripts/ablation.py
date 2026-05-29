#!/usr/bin/env python3
"""RA-defense 特徵 ablation(model-read-only)。把球隊 RA 摻進防禦項(w_ra),
用三~四月以得分殘差 fit w_ra,五月 OOS 以 vs-market log-loss 裁決。不改線上模型。

用法:
  python scripts/ablation.py
  python scripts/ablation.py --train 2026-03,2026-04 --test 2026-05
"""
import argparse
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config
import run_model
import fit_config

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

W_RA_GRID = [round(0.05 * i, 2) for i in range(17)]   # 0.00 .. 0.80


def _ra_blend(recent: float, season: float) -> float:
    return config.RECENT_W * recent + (1 - config.RECENT_W) * season


def recompute_mu_ra(row: dict, league_rg: float, w_ra: float) -> tuple[float, float]:
    """μ with RA blended into each side's defense term. w_ra=0 ≡ fit_config.recompute_mu."""
    home_rs = config.RECENT_W * row["home_rs_recent"] + (1 - config.RECENT_W) * row["home_rs_season"]
    away_rs = config.RECENT_W * row["away_rs_recent"] + (1 - config.RECENT_W) * row["away_rs_season"]
    home_fip = row["home_starter_fip"] if row["home_starter_fip"] is not None else league_rg
    away_fip = row["away_starter_fip"] if row["away_starter_fip"] is not None else league_rg
    home_pitch = run_model.pitch_today(home_fip, row["home_bullpen_era"])
    away_pitch = run_model.pitch_today(away_fip, row["away_bullpen_era"])
    home_def = (1 - w_ra) * home_pitch + w_ra * _ra_blend(row["home_ra_recent"], row["home_ra_season"])
    away_def = (1 - w_ra) * away_pitch + w_ra * _ra_blend(row["away_ra_recent"], row["away_ra_season"])
    mu_home, mu_away = run_model.expected_runs(home_rs, away_rs, home_def, away_def,
                                               row["park_factor"], league_rg=league_rg)
    return mu_home + mu_away, mu_home - mu_away


def fit_params(rows: list, w_ra: float) -> dict:
    """在固定 w_ra 下,mean-match 出 league_rg、殘差 MLE 出 sigma_team。"""
    mu_fn = lambda r, L: recompute_mu_ra(r, L, w_ra)
    L = fit_config.fit_league_rg(rows, mu_fn=mu_fn)
    s = fit_config.fit_sigma_team(rows, L, mu_fn=mu_fn)
    return {"w_ra": w_ra, "league_rg": L, "sigma_team": s}


def select_w_ra(rows: list, grid: list) -> tuple:
    """回 (w_ra*, [(w, sigma_train)...])。w_ra* = argmin σ_train(訓練得分殘差最小)。
    平手時偏好較小的 w(0 優先,符合『沒幫助就不加』)。"""
    table = []
    for w in grid:
        table.append((w, fit_params(rows, w)["sigma_team"]))
    w_star = min(table, key=lambda t: (t[1], t[0]))[0]
    return w_star, table


def _clamp(p: float, eps: float = 1e-9) -> float:
    return min(max(p, eps), 1 - eps)


def _ll(p: float, y: float) -> float:
    p = _clamp(p)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def eval_logloss(rows: list, params: dict) -> dict:
    """每注 log-loss 陣列(model 與 market)。只取有盤口者;O-U 排除 push。"""
    sigma = params["sigma_team"] * math.sqrt(2)
    L, w = params["league_rg"], params["w_ra"]
    out = {"rl": [], "ou": [], "market_rl": [], "market_ou": []}
    for r in rows:
        if not r["has_odds"] or r["rl_home_point"] is None:
            continue
        mt, mm = recompute_mu_ra(r, L, w)
        p_cov = run_model.cover_prob_home(mm, r["rl_home_point"], sigma=sigma)
        y = 1.0 if r["actual_margin"] > -r["rl_home_point"] else 0.0
        out["rl"].append(_ll(p_cov, y))
        out["market_rl"].append(_ll(r["rl_home_no_vig"], y))
        if r["total_line"] is not None and r["actual_total"] != r["total_line"]:
            p_ov = run_model.over_prob(mt, r["total_line"], sigma=sigma)
            yo = 1.0 if r["actual_total"] > r["total_line"] else 0.0
            out["ou"].append(_ll(p_ov, yo))
            out["market_ou"].append(_ll(r["over_no_vig"], yo))
    return out


def _pooled(ev: dict) -> list:
    return ev["rl"] + ev["ou"]


def _pooled_market(ev: dict) -> list:
    return ev["market_rl"] + ev["market_ou"]


def _mean(xs: list):
    return sum(xs) / len(xs) if xs else None


def ablate_ra(train_rows: list, test_rows: list, grid: list) -> dict:
    """baseline(w=0) vs candidate(w*) 的 OOS 比較。accept = OOS pooled log-loss 改善 > 1 SE。"""
    w_star, train_table = select_w_ra(train_rows, grid)
    p_base = fit_params(train_rows, 0.0)
    p_cand = fit_params(train_rows, w_star)

    ev_base = eval_logloss(test_rows, p_base)
    ev_cand = eval_logloss(test_rows, p_cand)

    base_pool = _pooled(ev_base)
    cand_pool = _pooled(ev_cand)
    mkt_pool = _pooled_market(ev_base)   # market 與 model 無關,base/cand 相同

    diffs = [b - c for b, c in zip(base_pool, cand_pool)]   # >0 表示 candidate 較好
    n = len(diffs)
    improve = _mean(diffs) or 0.0
    if n > 1:
        var = sum((d - improve) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = float("inf")

    accept = (w_star > 0.0) and (improve > se)   # 改善為正且超過 1 SE

    def _summ(p, ev):
        return {"w_ra": p["w_ra"], "league_rg": p["league_rg"], "sigma_team": p["sigma_team"],
                "rl_ll": _mean(ev["rl"]), "ou_ll": _mean(ev["ou"]), "pooled_ll": _mean(_pooled(ev))}

    return {
        "w_ra_star": w_star,
        "train_table": train_table,
        "baseline": _summ(p_base, ev_base),
        "candidate": _summ(p_cand, ev_cand),
        "pooled_improve": improve,
        "pooled_se": se,
        "accept": accept,
        "market_pooled_ll": _mean(mkt_pool),
        "gap_baseline": (_mean(base_pool) - _mean(mkt_pool)) if mkt_pool else None,
        "gap_candidate": (_mean(cand_pool) - _mean(mkt_pool)) if mkt_pool else None,
    }


def _f(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def render_report(result: dict, train_n: int, test_n: int) -> str:
    b, c = result["baseline"], result["candidate"]
    verdict = "ACCEPT" if result["accept"] else "REJECT"
    lines = [
        "# RA-defense ablation — 2026 (train Mar–Apr → test May)",
        "",
        f"訓練={train_n} 場  測試(有盤口)={test_n} 注場",
        f"選出 w_ra* = {result['w_ra_star']}",
        "",
        "| 模型 | w_ra | league_rg | sigma_team | RL ll | OU ll | pooled ll |",
        "|------|------|-----------|------------|-------|-------|-----------|",
        f"| baseline | {b['w_ra']} | {b['league_rg']} | {b['sigma_team']} | {_f(b['rl_ll'])} | {_f(b['ou_ll'])} | {_f(b['pooled_ll'])} |",
        f"| candidate | {c['w_ra']} | {c['league_rg']} | {c['sigma_team']} | {_f(c['rl_ll'])} | {_f(c['ou_ll'])} | {_f(c['pooled_ll'])} |",
        "",
        f"OOS pooled 改善(baseline − candidate)= {_f(result['pooled_improve'])} ± {_f(result['pooled_se'])} (1 SE)",
        f"**判決:{verdict}**(接受條件:改善 > 1 SE)",
        "",
        f"離市場差距(pooled ll − market {_f(result['market_pooled_ll'])}):"
        f" baseline {_f(result['gap_baseline'])} → candidate {_f(result['gap_candidate'])}",
        "",
        "> 北極星=差距≤0(打敗市場)。此判決僅決定 RA 是否值得進模型;"
        "baking 進 config/run_model 是另一個決定。",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="RA-defense 特徵 ablation(read-only)")
    p.add_argument("--train", default="2026-03,2026-04", help="逗號分隔 YYYY-MM")
    p.add_argument("--test", default="2026-05", help="逗號分隔 YYYY-MM(取有盤口者)")
    args = p.parse_args(argv)

    train_rows = [r for r in fit_config.load_fit_rows(set(args.train.split(",")))]
    test_rows = [r for r in fit_config.load_fit_rows(set(args.test.split(","))) if r["has_odds"]]
    if not train_rows or not test_rows:
        print("資料不足:確認三~五月已 backfill + fetch_results。", file=sys.stderr)
        return 1

    result = ablate_ra(train_rows, test_rows, W_RA_GRID)
    report = render_report(result, train_n=len(train_rows), test_n=len(test_rows))
    print(report)
    out_path = SKILL_ROOT / "analysis-data" / "backtest" / "ablation-ra-2026.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[record] {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
