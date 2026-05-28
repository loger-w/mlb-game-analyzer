"""回測報告渲染(v2):RL / O/U / edge 三段 + per-game CSV。"""
from pathlib import Path

import pandas as pd


def _pct(x) -> str:
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "—"


def render_report(*, df: pd.DataFrame, rl: dict, ou: dict, edge: dict,
                  month: str, out_path: Path) -> None:
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
