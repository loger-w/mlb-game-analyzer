"""phase3_skeleton.md renderer：產生 7 個 H2 + 預填數值表 + AI 填空 placeholder。

設計（spec §5）：
- 7 個 H2 永遠存在（即使 Flag 未觸發）
- ## 風險提示 段：prepare_game.py 偵測到的 Flag 13/3 預填條目；無則「無風險提示」
- 所有 render 函式只回傳字串；檔案寫入由 prepare_game.py 的 step_g 負責
- ## 條件修正 段：Park PF 修正預填
- ## 修正後預期得分 段：base 列從 formula_pred 預填
- 其餘為 AI 填空 (`<!-- AI 補：... -->`)
"""


def _age_emoji(age: int | None) -> str:
    """spec matchup-factors §球員年齡退化 投手版"""
    if age is None:
        return ""
    if age <= 24: return "📈"
    if age <= 29: return "⚡"
    if age <= 33: return "📉"
    if age <= 36: return "📉📉"
    return "📉📉📉"


def _render_pitcher_matchup_section(bundle: dict) -> list[str]:
    home_p = bundle.get("home_pitcher", {})
    away_p = bundle.get("away_pitcher", {})
    home_info = home_p.get("info", {})
    away_info = away_p.get("info", {})
    home_age = home_info.get("age")
    away_age = away_info.get("age")
    return [
        "## 投手對決",
        "",
        f"### {home_p.get('name', '?')} (HOME, {home_info.get('pitch_hand', '?')}HP, {home_age or '?'} {_age_emoji(home_age)})",
        "- **Tier 覆寫**：<!-- AI 補：覆寫 + 理由 / 或「沿用腳本」 -->",
        "- 真實水平判斷：<!-- AI 補：基於 ERA/xERA/FIP/Statcast/年齡綜合 -->",
        "- 對手打線威脅：<!-- AI 補 -->",
        "",
        f"### {away_p.get('name', '?')} (AWAY, {away_info.get('pitch_hand', '?')}HP, {away_age or '?'} {_age_emoji(away_age)})",
        "- **Tier 覆寫**：<!-- AI 補 -->",
        "- 真實水平判斷：<!-- AI 補 -->",
        "- 對手打線威脅：<!-- AI 補 -->",
        "",
    ]


def _render_lineup_section(bundle: dict) -> list[str]:
    home_l = bundle.get("home_lineup", {})
    away_l = bundle.get("away_lineup", {})
    return [
        "## 打線評級",
        "",
        f"### HOME — {home_l.get('tier_emoji', '?')} / {home_l.get('heat_emoji', '?')}",
        "- **Tier 覆寫**：<!-- AI 補 -->",
        "",
        f"### AWAY — {away_l.get('tier_emoji', '?')} / {away_l.get('heat_emoji', '?')}",
        "- **Tier 覆寫**：<!-- AI 補 -->",
        "",
    ]


def _render_bullpen_section(bundle: dict) -> list[str]:
    m = bundle.get("merged", {})
    return [
        "## 牛棚",
        "",
        "| | HOME | AWAY |",
        "|---|---|---|",
        f"| ERA / IL 數 / 核心 IL 估計 | {m.get('home_bullpen_era', '?')} / {m.get('home_bullpen_il_count', '?')} / <!-- AI --> | "
        f"{m.get('away_bullpen_era', '?')} / {m.get('away_bullpen_il_count', '?')} / <!-- AI --> |",
        "",
        "### 牛棚雙向修正值",
        "- HOME 牛棚：對手 +<!-- AI --> run | HOME ML <!-- AI -->%",
        "- AWAY 牛棚：對手 +<!-- AI --> run | AWAY ML <!-- AI -->%",
        "",
    ]


def _detect_risk_notes(bundle: dict) -> list[str]:
    """偵測 Flag 13 / Flag 3，回傳「條目 markdown 行」list（不含 H2 開頭）。"""
    try:
        from pitcher_stats import detect_triggers as detect_pitcher_triggers
    except ImportError:
        detect_pitcher_triggers = lambda x: []
    try:
        from lineup_analyzer import detect_triggers as detect_lineup_triggers
    except ImportError:
        detect_lineup_triggers = lambda x: []
    notes = []
    for side in ("home", "away"):
        triggers = detect_pitcher_triggers(bundle.get(f"{side}_pitcher", {}))
        for t in triggers:
            if t.get("flag") == 13:
                gap = t.get("value", "?")
                notes.append(f"- ⚠️ {side.upper()} 投手 Flag 13 (era_xera_delta={gap}):")
                notes.append("  - <!-- AI 補：是運氣還結構性？是否影響本場判斷？不自動下修預測 -->")
    for side in ("home", "away"):
        triggers = detect_lineup_triggers(bundle.get(f"{side}_lineup", {}))
        for t in triggers:
            if t.get("flag") == 3:
                babip = bundle.get(f"{side}_lineup", {}).get("last7_babip", "?")
                notes.append(f"- ⚠️ {side.upper()} 打線 Flag 3 (last7 BABIP={babip}):")
                notes.append("  - <!-- AI 補：可能回歸或可能持續？是否影響本場判斷？不自動 ±run value -->")
    return notes


def _render_risk_section(bundle: dict) -> list[str]:
    notes = _detect_risk_notes(bundle)
    if not notes:
        return ["## 風險提示", "", "無風險提示", ""]
    return ["## 風險提示", ""] + notes + [""]


def _render_conditional_section(bundle: dict) -> list[str]:
    pf = bundle.get("merged", {}).get("park_factor", 100)
    pf_correction = (pf - 100) * 0.05
    return [
        "## 條件修正",
        "",
        f"- Park Factor: {pf} → {pf_correction:+.2f} run",
        "- 先發 tier / doubleheader / 天氣：<!-- AI 補 -->",
        "",
    ]


def _render_expected_runs_section(bundle: dict, formula_pred: dict) -> list[str]:
    home_base = formula_pred.get("home_expected_runs", "?")
    away_base = formula_pred.get("away_expected_runs", "?")
    total_base = (
        (home_base + away_base)
        if isinstance(home_base, (int, float)) and isinstance(away_base, (int, float))
        else "?"
    )
    return [
        "## 修正後預期得分",
        "",
        "| | base (formula) | + 信號 | adjusted |",
        "|---|---|---|---|",
        f"| HOME | {home_base} | <!-- AI 補 --> | <!-- AI 補 --> |",
        f"| AWAY | {away_base} | <!-- AI 補 --> | <!-- AI 補 --> |",
        f"| Total | {total_base} | <!-- AI 補 --> | <!-- AI 補 --> |",
        "",
    ]


def _render_overall_section() -> list[str]:
    return [
        "## 整體判斷",
        "",
        "- **方向（基本面）**：<!-- AI 補 -->",
        "- **總分（基本面）**：<!-- AI 補 -->",
        "- **信心**：<!-- AI 補 LOW/MEDIUM/HIGH -->",
        "- **風險**：<!-- AI 補 1-4 點 -->",
        "",
        "⛔ MUST NOT contain：星級、明確盤口推薦",
    ]


def render_skeleton(bundle: dict, formula_pred: dict) -> str:
    """主入口：渲染整份 phase3_skeleton.md，回傳 markdown 字串（不寫檔；caller 寫檔）。"""
    lines: list[str] = []
    lines += _render_pitcher_matchup_section(bundle)
    lines += _render_lineup_section(bundle)
    lines += _render_bullpen_section(bundle)
    lines += _render_risk_section(bundle)
    lines += _render_conditional_section(bundle)
    lines += _render_expected_runs_section(bundle, formula_pred)
    lines += _render_overall_section()
    return "\n".join(lines)
