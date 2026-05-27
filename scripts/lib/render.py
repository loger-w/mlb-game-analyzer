"""Render backtest output: Markdown report + details CSV."""

from datetime import date as _date
from pathlib import Path
from typing import Optional

import pandas as pd


CSV_COLUMNS = [
    "date", "matchup", "game_pk",
    "skill_direction", "skill_total", "skill_confidence", "skill_confidence_pct", "skill_prob_mapped",
    "market_home_winprob_no_vig", "market_total_line", "market_favorite", "market_favorite_winprob",
    "actual_winner", "actual_total", "actual_home_score", "actual_away_score",
    "direction_hit", "ou_hit", "total_abs_error", "total_signed_error",
    "brier_score", "log_loss",
    "park_factor", "has_reverse_platoon", "has_chain_break_300", "has_bullpen_il_2plus",
    "closing_snapshot_ts", "closing_missing", "result_missing", "parse_failed",
    "dossier_path",
]


def _enrich_with_per_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add direction_hit / ou_hit / total_abs_error / total_signed_error / brier / log_loss
    per-row (NaN where not applicable)."""
    import numpy as np
    out = df.copy()
    eps = 1e-12

    # Coerce numeric columns — when all parse_failed/result_missing, dtype is object
    # and downstream arithmetic / np.sign fails.
    for col in ("skill_total", "actual_total", "market_total_line", "skill_prob_mapped"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["direction_hit"] = (out["skill_direction"] == out["actual_winner"]).where(
        out["skill_direction"].isin(["HOME", "AWAY"]) & out["actual_winner"].notna()
    )
    out["total_abs_error"] = (out["skill_total"] - out["actual_total"]).abs()
    out["total_signed_error"] = out["skill_total"] - out["actual_total"]

    line = out["market_total_line"]
    no_push_mask = (out["actual_total"] != line) & (out["skill_total"] != line) & line.notna()
    ss = np.sign(out["skill_total"] - line)
    sa = np.sign(out["actual_total"] - line)
    out["ou_hit"] = (ss == sa).where(no_push_mask)

    # Brier / log-loss per row (NaN if skill_prob_mapped missing)
    p = out["skill_prob_mapped"]
    y = out["direction_hit"].astype(float)
    out["brier_score"] = ((p - y) ** 2).where(p.notna() & y.notna())
    log_term = -(y * np.log(p.clip(eps, 1 - eps)) + (1 - y) * np.log((1 - p).clip(eps, 1 - eps)))
    out["log_loss"] = log_term.where(p.notna() & y.notna())

    return out


def render_details_csv(df: pd.DataFrame, out_path: Path):
    enriched = _enrich_with_per_row_metrics(df)
    # Ensure all CSV_COLUMNS exist, fill missing with NaN
    for c in CSV_COLUMNS:
        if c not in enriched.columns:
            enriched[c] = None
    enriched = enriched[CSV_COLUMNS]
    # Convert bool/NaN combo cleanly
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False, encoding="utf-8")


def _fmt_pct(x: Optional[float], digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:.{digits}f}%"


def _fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def render_report(
    df: pd.DataFrame,
    direction_metrics: dict,
    total_metrics: dict,
    calibration: dict,
    slices: pd.DataFrame,
    failure_cases: pd.DataFrame,
    month: str,
    out_path: Path,
):
    n_input = len(df)
    n_parse_ok = int((~df["parse_failed"]).sum())
    n_closing_ok = int((~df["closing_missing"]).sum())
    n_result_ok = int((~df["result_missing"]).sum())
    n_valid = int(((~df["parse_failed"]) & (~df["closing_missing"]) & (~df["result_missing"])).sum())

    today = _date.today().isoformat()
    lines = []
    lines.append(f"# MLB Skill 回測 — {month[:4]} 年 {int(month[5:7])} 月")
    lines.append("")
    lines.append(f"_樣本：{n_valid} 場有效（{month} 全月） ｜ baseline: Pinnacle no-vig 收盤線 ｜ 生成於 {today}_")
    lines.append("")

    # 資料健康度
    lines.append("## 資料健康度")
    lines.append(f"- 輸入 summary.md：{n_input}")
    lines.append(f"- 通過解析：{n_parse_ok}（剔出 parse_failed {n_input - n_parse_ok}）")
    lines.append(f"- 通過 closing snapshot 匹配：{n_closing_ok}（剔出 closing_missing {n_input - n_closing_ok}）")
    lines.append(f"- 通過 result 取得：{n_result_ok}（剔出 result_missing {n_input - n_result_ok}）")
    lines.append(f"- **有效樣本：{n_valid} 場**")
    lines.append("")

    # TL;DR
    lines.append("## TL;DR")
    edge_str = _fmt_pct(direction_metrics.get("edge_pp"), 1) if direction_metrics.get("edge_pp") is not None else "—"
    lines.append(f"- 方向命中率：skill {_fmt_pct(direction_metrics.get('skill_hit_rate'))} vs market {_fmt_pct(direction_metrics.get('market_hit_rate'))}，edge {edge_str}")
    lines.append(f"- 總分 MAE：{_fmt_num(total_metrics.get('total_mae'))}，bias {_fmt_num(total_metrics.get('total_bias'))}")
    lines.append(f"- 反市場時：skill 命中 {_fmt_pct(direction_metrics.get('skill_against_hit_rate'))}（n={direction_metrics.get('skill_against_n')}）")
    lines.append(f"- Brier: skill {_fmt_num(calibration.get('brier_score'))} vs market {_fmt_num(calibration.get('brier_baseline_market'))}")
    lines.append("")

    # 方向類
    lines.append("## 1. 方向類指標")
    lines.append("")
    lines.append("| 指標 | 值 | n | 95% CI |")
    lines.append("|---|---|---|---|")
    sci = direction_metrics.get("skill_ci")
    mci = direction_metrics.get("market_ci")
    lines.append(f"| skill 命中率 | {_fmt_pct(direction_metrics.get('skill_hit_rate'))} | {direction_metrics.get('skill_n')} | "
                 f"{_fmt_pct(sci[0]) if sci else '—'} – {_fmt_pct(sci[1]) if sci else '—'} |")
    lines.append(f"| market favorite 命中率 | {_fmt_pct(direction_metrics.get('market_hit_rate'))} | {direction_metrics.get('market_n')} | "
                 f"{_fmt_pct(mci[0]) if mci else '—'} – {_fmt_pct(mci[1]) if mci else '—'} |")
    lines.append(f"| **skill edge (pp)** | {edge_str} | — | — |")
    lines.append(f"| skill 同意市場時命中率 | {_fmt_pct(direction_metrics.get('skill_aligned_hit_rate'))} | {direction_metrics.get('skill_aligned_n')} | — |")
    lines.append(f"| skill 反市場時命中率 | {_fmt_pct(direction_metrics.get('skill_against_hit_rate'))} | {direction_metrics.get('skill_against_n')} | — |")
    lines.append("")

    # 總分類
    lines.append("## 2. 總分類指標")
    lines.append("")
    lines.append("| 指標 | 值 | n |")
    lines.append("|---|---|---|")
    lines.append(f"| MAE | {_fmt_num(total_metrics.get('total_mae'))} | {total_metrics.get('n')} |")
    lines.append(f"| bias (skill − actual) | {_fmt_num(total_metrics.get('total_bias'))} | {total_metrics.get('n')} |")
    lines.append(f"| Over/Under 命中率 | {_fmt_pct(total_metrics.get('ou_hit_rate'))} | {total_metrics.get('ou_n')} |")
    lines.append("")

    # Calibration
    lines.append("## 3. 信心 Calibration")
    lines.append("")
    lines.append("| 信心 | n | mapped p | 實際命中率 | 落差 | CI |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in calibration["reliability_table"].iterrows():
        ci_str = f"{_fmt_pct(r['ci_low'])} – {_fmt_pct(r['ci_high'])}" if r.get("ci_low") is not None else "—"
        lines.append(f"| {r['confidence']} | {r['n']} | {_fmt_num(r['mapped_p'])} | "
                     f"{_fmt_pct(r['actual_hit_rate'])} | {_fmt_pct(r['delta'])} | {ci_str} |")
    lines.append("")
    lines.append(f"- Brier (skill / market): {_fmt_num(calibration.get('brier_score'))} / "
                 f"{_fmt_num(calibration.get('brier_baseline_market'))}")
    lines.append(f"- log-loss (skill / market): {_fmt_num(calibration.get('log_loss'))} / "
                 f"{_fmt_num(calibration.get('log_loss_baseline_market'))}")
    lines.append("")
    lines.append("<!-- 結論待人工填 -->")
    lines.append("")

    # 切片
    lines.append("## 4. 分組切片")
    lines.append("")
    lines.append("| 切片 | n | dir_n | dir_hit | ou_hit | mae | bias |")
    lines.append("|---|---|---|---|---|---|---|")
    for idx, r in slices.iterrows():
        lines.append(f"| {idx} | {int(r['n'])} | {int(r['dir_n'])} | {_fmt_pct(r['dir_hit'])} | {_fmt_pct(r['ou_hit'])} | "
                     f"{_fmt_num(r['mae'])} | {_fmt_num(r['bias'])} |")
    lines.append("")

    # 失敗案例
    lines.append("## 5. 失敗案例")
    lines.append("")
    if len(failure_cases) == 0:
        lines.append("_無_")
    else:
        lines.append("| date | matchup | skill 方向 (信心) | 實際勝方 | skill total | 實際 total | 主訊號 | dossier |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in failure_cases.iterrows():
            dual = " ⚠️" if r.get("dual_failure") else ""
            dossier = r.get("dossier_path", "")
            sig = r.get("main_signal", "")
            # Confidence display: prefer bucket, fallback to pct %
            conf_display = r.get("skill_confidence") or (
                f"{r['skill_confidence_pct']*100:.0f}%" if pd.notna(r.get("skill_confidence_pct")) else "—"
            )
            lines.append(
                f"| {r['date']} | {r['matchup']}{dual} | {r['skill_direction']} ({conf_display}) | "
                f"{r['actual_winner']} | {_fmt_num(r['skill_total'])} | {_fmt_num(r['actual_total'], 0)} | "
                f"{sig} | [link]({dossier}) |"
            )
    lines.append("")

    # 結論 stub
    lines.append("## 6. 結論與下一步")
    lines.append("")
    lines.append("<!-- 結論待人工填 -->")
    lines.append("")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
