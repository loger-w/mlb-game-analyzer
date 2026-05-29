#!/usr/bin/env python3
"""Fit 2 個係數(LEAGUE_RG, SIGMA_TEAM)使模型機率校準。從凍結的 features.json 重算,零 refetch。

用法:
  python scripts/fit_config.py                 # 提案 + before/after,不改檔
  python scripts/fit_config.py --write          # 同時把兩個值寫回 config.py
  python scripts/fit_config.py --months 2026-04,2026-05
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
ANALYSIS_DATA_DIR = SKILL_ROOT / "analysis-data"
CONFIG_PATH = SCRIPT_DIR / "config.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config
import run_model

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def recompute_mu(row: dict, league_rg: float) -> tuple[float, float]:
    """從凍結 inputs 以 candidate league_rg 重算 (mu_total, mu_margin)。RECENT_W 等用 config 先驗。"""
    home_rs = config.RECENT_W * row["home_rs_recent"] + (1 - config.RECENT_W) * row["home_rs_season"]
    away_rs = config.RECENT_W * row["away_rs_recent"] + (1 - config.RECENT_W) * row["away_rs_season"]
    home_fip = row["home_starter_fip"] if row["home_starter_fip"] is not None else league_rg
    away_fip = row["away_starter_fip"] if row["away_starter_fip"] is not None else league_rg
    home_pitch = run_model.pitch_today(home_fip, row["home_bullpen_era"])
    away_pitch = run_model.pitch_today(away_fip, row["away_bullpen_era"])
    mu_home, mu_away = run_model.expected_runs(home_rs, away_rs, home_pitch, away_pitch,
                                               row["park_factor"], league_rg=league_rg)
    return mu_home + mu_away, mu_home - mu_away


def load_fit_rows(months: set, data_dir: Path = ANALYSIS_DATA_DIR) -> list:
    """讀 features.json(schema 2)+ result.json → fit row 清單。無 result 或非 v2 略過。"""
    rows = []
    for date_dir in sorted(Path(data_dir).iterdir()):
        if not date_dir.is_dir() or date_dir.name.endswith(".local-backup"):
            continue
        if date_dir.name[:7] not in months:
            continue
        for m in sorted(date_dir.iterdir()):
            if not m.is_dir():
                continue
            fp = m / "features.json"
            rp = m / "result.json"
            if not fp.exists() or not rp.exists():
                continue
            try:
                feats = json.loads(fp.read_text(encoding="utf-8"))
                result = json.loads(rp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if feats.get("schema_version") != 2:
                continue
            inp = feats.get("inputs") or {}
            if not inp:
                continue
            odds = feats.get("odds") or {}
            rl = (odds.get("rl") or {}) if odds else {}
            total = (odds.get("total") or {}) if odds else {}
            rows.append({
                "date": date_dir.name, "matchup": m.name,
                "home_rs_recent": inp["home_rs_recent"], "home_rs_season": inp["home_rs_season"],
                "away_rs_recent": inp["away_rs_recent"], "away_rs_season": inp["away_rs_season"],
                "home_starter_fip": (inp.get("home_starter") or {}).get("fip"),
                "away_starter_fip": (inp.get("away_starter") or {}).get("fip"),
                "home_bullpen_era": inp["home_bullpen_era"], "away_bullpen_era": inp["away_bullpen_era"],
                "park_factor": inp["park_factor"],
                "actual_total": result["home_score"] + result["away_score"],
                "actual_margin": result["home_score"] - result["away_score"],
                "has_odds": bool(odds),
                "rl_home_point": rl.get("home_point"), "rl_home_no_vig": rl.get("home_no_vig"),
                "total_line": total.get("line"), "over_no_vig": total.get("over_no_vig"),
            })
    return rows


def fit_league_rg(rows: list, lo: float = 2.0, hi: float = 8.0, iters: int = 50) -> float:
    """二分搜尋 LEAGUE_RG 使 mean(預測 mu_total) = mean(實際 total)。mu_total 隨 L 單調遞減。"""
    target = sum(r["actual_total"] for r in rows) / len(rows)

    def mean_mu_total(L: float) -> float:
        return sum(recompute_mu(r, L)[0] for r in rows) / len(rows)

    for _ in range(iters):
        mid = (lo + hi) / 2
        if mean_mu_total(mid) > target:   # 預測偏高 → 提高 L 以壓低 mu
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3)


def fit_sigma_team(rows: list, league_rg: float) -> float:
    """SIGMA_TEAM = sqrt( (mean(r_total^2)+mean(r_margin^2)) / 4 )，r = 實際 - 預測(高斯 MLE)。"""
    sse = 0.0
    for r in rows:
        mt, mm = recompute_mu(r, league_rg)
        sse += (r["actual_total"] - mt) ** 2 + (r["actual_margin"] - mm) ** 2
    n = len(rows)
    return round(math.sqrt(sse / (4 * n)), 3)


def _clamp(p: float, eps: float = 1e-9) -> float:
    return min(max(p, eps), 1 - eps)


def eval_calibration(rows: list, league_rg: float, sigma_team: float) -> dict:
    """在有盤口的列上算 RL/O-U log-loss、正 edge 命中率。SIGMA = sigma_team*sqrt(2)。"""
    sigma = sigma_team * math.sqrt(2)
    rl_ll, ou_ll = [], []
    rl_edge_y, ou_edge_y = [], []
    for r in rows:
        if not r["has_odds"] or r["rl_home_point"] is None:
            continue
        mt, mm = recompute_mu(r, league_rg)
        # RL
        p_cover = _clamp(run_model.cover_prob_home(mm, r["rl_home_point"], sigma=sigma))
        y = 1.0 if r["actual_margin"] > -r["rl_home_point"] else 0.0
        rl_ll.append(-(y * math.log(p_cover) + (1 - y) * math.log(1 - p_cover)))
        if (p_cover - r["rl_home_no_vig"]) * 100 > 0:
            rl_edge_y.append(y)
        # O/U (exclude push)
        if r["total_line"] is not None and r["actual_total"] != r["total_line"]:
            p_over = _clamp(run_model.over_prob(mt, r["total_line"], sigma=sigma))
            yo = 1.0 if r["actual_total"] > r["total_line"] else 0.0
            ou_ll.append(-(yo * math.log(p_over) + (1 - yo) * math.log(1 - p_over)))
            if (p_over - r["over_no_vig"]) * 100 > 0:
                ou_edge_y.append(yo)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "n_rl": len(rl_ll), "rl_log_loss": _mean(rl_ll),
        "n_ou": len(ou_ll), "ou_log_loss": _mean(ou_ll),
        "rl_pos_edge_n": len(rl_edge_y), "rl_pos_edge_hit": _mean(rl_edge_y),
        "ou_pos_edge_n": len(ou_edge_y), "ou_pos_edge_hit": _mean(ou_edge_y),
    }


def rewrite_config_text(text: str, league_rg: float, sigma_team: float) -> str:
    """只改 LEAGUE_RG / SIGMA_TEAM 的數值,保留行內註解與其餘內容。"""
    text = re.sub(r"^(LEAGUE_RG = )[\d.]+", rf"\g<1>{league_rg}", text, flags=re.M)
    text = re.sub(r"^(SIGMA_TEAM = )[\d.]+", rf"\g<1>{sigma_team}", text, flags=re.M)
    return text


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "—"


def main(argv=None):
    p = argparse.ArgumentParser(description="Fit LEAGUE_RG + SIGMA_TEAM(機率校準)")
    p.add_argument("--months", default="2026-03,2026-04,2026-05",
                   help="逗號分隔 YYYY-MM(μ/σ fit 用全部;校準驗證自動取有盤口者)")
    p.add_argument("--write", action="store_true", help="把擬合值寫回 config.py")
    args = p.parse_args(argv)

    months = set(args.months.split(","))
    rows = load_fit_rows(months)
    if not rows:
        print("沒有可用資料(features.json + result.json)。先跑 backfill + fetch_results。", file=sys.stderr)
        return 1

    train = [r for r in rows]                                   # μ/σ:全部有 result 的列
    may_odds = [r for r in rows if r["date"][:7] == "2026-05" and r["has_odds"]]
    apr = [r for r in rows if r["date"][:7] in ("2026-03", "2026-04")]

    L_fit = fit_league_rg(train)
    S_fit = fit_sigma_team(train, L_fit)

    print(f"樣本:fit(有 result)={len(train)}  五月有盤口(驗證)={len(may_odds)}  三四月={len(apr)}")
    print(f"\n提案係數:")
    print(f"  LEAGUE_RG : {config.LEAGUE_RG} → {L_fit}")
    print(f"  SIGMA_TEAM: {config.SIGMA_TEAM} → {S_fit}")

    before = eval_calibration(may_odds, config.LEAGUE_RG, config.SIGMA_TEAM)
    after = eval_calibration(may_odds, L_fit, S_fit)
    print(f"\n五月校準 before → after(log-loss 越低越好):")
    print(f"  RL log-loss : {_fmt(before['rl_log_loss'])} → {_fmt(after['rl_log_loss'])}  (n={after['n_rl']})")
    print(f"  OU log-loss : {_fmt(before['ou_log_loss'])} → {_fmt(after['ou_log_loss'])}  (n={after['n_ou']})")
    print(f"  RL +edge 命中: {_fmt(before['rl_pos_edge_hit'])} → {_fmt(after['rl_pos_edge_hit'])}  (n={after['rl_pos_edge_n']})")
    print(f"  OU +edge 命中: {_fmt(before['ou_pos_edge_hit'])} → {_fmt(after['ou_pos_edge_hit'])}  (n={after['ou_pos_edge_n']})")

    if apr:
        L_apr = fit_league_rg(apr)
        S_apr = fit_sigma_team(apr, L_apr)
        may_apr = eval_calibration(may_odds, L_apr, S_apr)
        print(f"\n穩定性(三四月 fit → 五月驗證):L={L_apr} σ={S_apr}  "
              f"RL ll={_fmt(may_apr['rl_log_loss'])} OU ll={_fmt(may_apr['ou_log_loss'])}")

    if args.write:
        new_text = rewrite_config_text(CONFIG_PATH.read_text(encoding="utf-8"), L_fit, S_fit)
        CONFIG_PATH.write_text(new_text, encoding="utf-8")
        print(f"\n[WRITE] 已更新 {CONFIG_PATH}(LEAGUE_RG={L_fit}, SIGMA_TEAM={S_fit})")
    else:
        print(f"\n(未寫檔。確認後加 --write 套用,或手動改 config.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
