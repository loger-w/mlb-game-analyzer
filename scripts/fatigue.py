#!/usr/bin/env python3
"""牛棚短休疲勞:近 2 天後援 IP。Path A=μ 懲罰(經一般化 ablation 台子),
Path B=重用尾巴過濾器(正 edge 注的命中率+CLV,tail vs non-tail)。對線上模型唯讀。

用法:
  python scripts/fatigue.py
  python scripts/fatigue.py --train 2026-03,2026-04 --test 2026-05
"""
import argparse
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bullpen
import fit_config
from _team_resolver import resolve_team_id
from lib.closing_line import _parse_iso_utc, find_entry_close_snapshots
from lib import clv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

FAT_W_GRID = [round(0.02 * i, 2) for i in range(26)]   # 0.00 .. 0.50(寬到能框住內部最佳點)
TAIL_IP = 12.0
SNAPSHOTS_DIR = SKILL_ROOT / "odds" / "odds_snapshots"


def team_ids_from_matchup(matchup):
    """'AWAY@HOME'(可含 -G1/-G2)→ (away_id, home_id)。失敗 → (None, None)。"""
    try:
        away_abbr, _, rest = matchup.partition("@")
        home_abbr = rest.split("-")[0]
        return resolve_team_id(away_abbr.strip()), resolve_team_id(home_abbr.strip())
    except Exception:
        return None, None


def add_fatigue_to_rows(rows, year, k=2, cache_dir=bullpen.DEFAULT_CACHE_DIR, index=None):
    """每列加 home_fat_ip / away_fat_ip(近 k 天後援 IP)。index 預設『直接讀快取』,
    絕不因 row 日期超出快取覆蓋而觸發重建(此分析唯讀、不該再抓網路)。"""
    if index is None:
        p = bullpen._cache_path(year, cache_dir)
        if p.exists():
            index = json.loads(p.read_text(encoding="utf-8")).get("index", {})
        else:
            index = bullpen.load_or_build_index(year, needed_through=f"{year}-05-25", cache_dir=cache_dir)
    out = []
    for r in rows:
        away_id, home_id = team_ids_from_matchup(r.get("matchup", ""))
        d = r.get("date")
        hf = bullpen.relief_ip_last_k(home_id, year, d, k, index=index) if (home_id and d) else 0.0
        af = bullpen.relief_ip_last_k(away_id, year, d, k, index=index) if (away_id and d) else 0.0
        r2 = dict(r)
        r2["home_fat_ip"] = hf
        r2["away_fat_ip"] = af
        out.append(r2)
    return out


def recompute_mu_fatigue(row, league_rg, w_fat):
    """μ with 疲勞懲罰:該隊 bullpen_era += w_fat × fat_ip。w_fat=0 ≡ fit_config.recompute_mu。"""
    r2 = dict(row)
    r2["home_bullpen_era"] = row["home_bullpen_era"] + w_fat * row.get("home_fat_ip", 0.0)
    r2["away_bullpen_era"] = row["away_bullpen_era"] + w_fat * row.get("away_fat_ip", 0.0)
    return fit_config.recompute_mu(r2, league_rg)


def _edge_clv(row, snapshots_dir, market, side):
    """正 edge 方向的 CLV(close−entry pp)。無 headroom / 無盤 → None。"""
    entry, close = find_entry_close_snapshots(snapshots_dir, row.get("date"),
                                              row.get("home_team"), row.get("away_team"))
    if entry is None or close is None:
        return None
    e_ts = _parse_iso_utc(entry.get("snapshot_time_utc", ""))
    c_ts = _parse_iso_utc(close.get("snapshot_time_utc", ""))
    if e_ts is None or c_ts is None or not (e_ts < c_ts):
        return None
    return clv.clv_pp(entry, close, market, side)


def _bet_summ(bets):
    n = len(bets)
    if n == 0:
        return {"n": 0, "hit_rate": None, "clv_mean": None, "n_clv": 0}
    hit = sum(1 for h, _ in bets if h) / n
    clvs = [c for _, c in bets if c is not None]
    return {"n": n, "hit_rate": round(hit, 3),
            "clv_mean": (round(sum(clvs) / len(clvs), 3) if clvs else None), "n_clv": len(clvs)}


def fatigue_filter_report(rows, snapshots_dir):
    """正 edge 注(home_rl_pp>0→主過盤;over_pp>0→Over)按尾巴(max fat_ip≥TAIL_IP)分組,
    各報 n / 命中率 / CLV。對齊 compute_edge_calibration 的正 edge 定義。"""
    tail, non = [], []
    for r in rows:
        bets = []
        e = r.get("home_rl_pp")
        if isinstance(e, (int, float)) and math.isfinite(e) and e > 0:
            mg, pt = r.get("actual_margin"), r.get("rl_home_point")
            if mg is not None and pt is not None:
                bets.append((mg > -pt, _edge_clv(r, snapshots_dir, "rl", "home")))
        o = r.get("over_pp")
        if isinstance(o, (int, float)) and math.isfinite(o) and o > 0:
            tt, ln = r.get("actual_total"), r.get("total_line")
            if tt is not None and ln is not None and tt != ln:
                bets.append((tt > ln, _edge_clv(r, snapshots_dir, "ou", "over")))
        is_tail = max(r.get("home_fat_ip", 0.0) or 0.0, r.get("away_fat_ip", 0.0) or 0.0) >= TAIL_IP
        (tail if is_tail else non).extend(bets)
    return {"tail": _bet_summ(tail), "non_tail": _bet_summ(non)}


def _f(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def _pct(x):
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "—"


def _fmt_subset(name, s):
    return (f"- {name}:n={s['n']}｜命中率 {_pct(s['hit_rate'])}"
            f"｜CLV mean {_f(s['clv_mean'], 3)}pp(n_clv={s['n_clv']})")


def render_report(path_a, path_b, train_n, test_n, valid_n):
    b, c = path_a["baseline"], path_a["candidate"]
    verdict = "ACCEPT" if path_a["accept"] else "REJECT"
    lines = [
        "# 牛棚短休疲勞 ablation — 2026",
        "",
        "## Path A:μ 懲罰(w_fat × 近2天後援IP 加進牛棚ERA)",
        f"_train Mar–Apr={train_n}｜test May(有盤口)={test_n}_",
        f"- w_fat* = {path_a['w_star']}",
        f"- pooled log-loss:baseline {_f(b['pooled_ll'])} → candidate {_f(c['pooled_ll'])}"
        f"(改善 {_f(path_a['pooled_improve'])} ± {_f(path_a['pooled_se'])} SE)",
        f"- **判決:{verdict}**(接受條件:OOS 改善 > 1 SE)",
        "",
        f"## Path B:尾巴過濾器(任一隊近2天後援IP ≥ {TAIL_IP})— 正 edge 注,valid={valid_n}",
        _fmt_subset("尾巴(tail)", path_b["tail"]),
        _fmt_subset("非尾巴", path_b["non_tail"]),
        "",
        "> 尾巴 n 小屬正常;命中率與 CLV 要同向且離噪音才算數,否則 inconclusive。",
        "> 對線上模型唯讀;此判決僅決定是否值得進一步,baking 是另一個決定。",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    import ablation
    from lib.load import build_dataframe_for_month
    p = argparse.ArgumentParser(description="牛棚疲勞 ablation(read-only)")
    p.add_argument("--train", default="2026-03,2026-04")
    p.add_argument("--test", default="2026-05")
    args = p.parse_args(argv)
    year = int(args.test[:4])

    train = add_fatigue_to_rows(fit_config.load_fit_rows(set(args.train.split(","))), year)
    test = add_fatigue_to_rows(fit_config.load_fit_rows(set(args.test.split(","))), year)
    test_odds = [r for r in test if r["has_odds"]]
    if not train or not test_odds:
        print("資料不足:確認三~五月已 backfill + fetch_results。", file=sys.stderr)
        return 1
    path_a = ablation.ablate(train, test_odds, FAT_W_GRID, recompute=recompute_mu_fatigue)

    dfrows = add_fatigue_to_rows(build_dataframe_for_month(args.test).to_dict("records"), year)
    valid = [r for r in dfrows if not r["odds_missing"] and not r["result_missing"]]
    path_b = fatigue_filter_report(valid, SNAPSHOTS_DIR)

    report = render_report(path_a, path_b, len(train), len(test_odds), len(valid))
    print(report)
    out_path = SKILL_ROOT / "analysis-data" / "backtest" / "ablation-fatigue-2026.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[record] {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
