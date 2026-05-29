"""回測報告渲染(v2):RL / O/U / edge 三段 + per-game CSV。"""
from pathlib import Path

import pandas as pd


def _pct(x) -> str:
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "—"


def _f(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def render_clv_section(clv: dict) -> str:
    rl, ou = clv["rl"], clv["ou"]
    hours = "".join(f"  - {h}:00 ET｜n={v['n']}｜mean {_f(v['mean'])}pp\n"
                    for h, v in clv["entry_hour_table"].items())
    return (
        "## CLV(領先指標:線往模型 pick 那側移動了多少 pp)\n"
        f"_headroom 子集(entry 嚴格早於 close):{clv['n_headroom']} / {clv['n_total']}"
        f"｜entry→close 中位 {_f(clv['median_minutes_gap'], 0)} 分_\n\n"
        f"- RL:n={rl['n']}｜mean {_f(rl['mean'])}pp｜median {_f(rl['median'])}pp"
        f"｜往我方比例 {_pct(rl['share_pos'])}｜corr(|edge|,CLV) {_f(rl['corr_edge'])}"
        f"｜點數穩定 {_pct(rl['pct_point_stable'])}\n"
        f"- O/U:n={ou['n']}｜mean {_f(ou['mean'])}pp｜median {_f(ou['median'])}pp"
        f"｜往我方比例 {_pct(ou['share_pos'])}｜corr(|edge|,CLV) {_f(ou['corr_edge'])}"
        f"｜點數穩定 {_pct(ou['pct_point_stable'])}\n"
        f"- 依 entry 時段:\n{hours}\n"
        "> mean CLV 顯著 >0 才代表模型 edge 訊號有領先力;≈0 = 與無 alpha 結論一致。\n"
        "> ⚠️ 存檔『close』為軟代理(中位約開賽前 71 分)、4/28 前無盤中 odds、"
        "點數移動時 CLV 為近似(看點數穩定比例)。\n"
    )


def render_report(*, df: pd.DataFrame, rl: dict, ou: dict, edge: dict,
                  month: str, out_path: Path, clv: dict | None = None) -> None:
    valid = 0
    if len(df):
        valid = int(((~df["odds_missing"]) & (~df["result_missing"])).sum())
    lines = [
        f"# MLB 回測(v2)— {month}",
        "",
        f"_有效樣本(odds + result 皆有):{valid} / {len(df)}_",
        "",
        "## RL 過盤(model p>0.5 預測主過盤是否命中)",
        f"- n = {rl['n']}｜命中率 = {_pct(rl['rl_hit_rate'])}",
        "",
        "## O/U(model p>0.5 預測 Over 是否命中,排除 push)",
        f"- n = {ou['n']}｜命中率 = {_pct(ou['ou_hit_rate'])}",
        "",
        "## edge 校準(正 edge 那側實際命中率)",
        f"- RL 正 edge:n = {edge['rl_pos_edge_n']}｜命中 = {_pct(edge['rl_pos_edge_hit'])}",
        f"- O/U 正 edge:n = {edge['ou_pos_edge_n']}｜命中 = {_pct(edge['ou_pos_edge_hit'])}",
        "",
    ]
    if clv is not None:
        lines += [render_clv_section(clv), ""]
    lines += [
        "> σ_team / 權重未經回測重新擬合前,edge 命中僅供觀察,不可當下注依據。",
        "",
        "<!-- 結論待人工填 -->",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_details_csv(df: pd.DataFrame, out_path: Path) -> None:
    if len(df) == 0:
        out_path.write_text("", encoding="utf-8")
        return
    df.to_csv(out_path, index=False, encoding="utf-8")
