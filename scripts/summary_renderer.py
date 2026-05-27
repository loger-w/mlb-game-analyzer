"""summary.md template renderer：產生 7 個 H2 + 預填數值表 + AI 填空 placeholder。

注意：與 dossier 同目錄的 `<basename>_summary.md`（drill-down 用）是不同的檔，
那些由各 source 腳本的 `format_md()` 產出。本模組只負責 pipeline 最終輸出 `summary.md`。

設計：
- 7 個 H2 永遠存在（即使 Flag 未觸發）
- ## 風險提示 段：prepare_game.py 偵測到的 Flag 8/3 預填條目；無則「無風險提示」
- 所有 render 函式只回傳字串；檔案寫入由 prepare_game.py 的 step_g 負責
- ## 條件修正 段：Park PF 修正預填
- ## 修正後預期得分 段：base 列從 formula_pred 預填
- 其餘為 AI 填空 (`<!-- AI 補：... -->`)
- AI 填完所有 placeholder 即為最終輸出（不需另存檔）

Schema 對齊：直接讀 pitcher_stats / lineup_analyzer / roster_checker / merge_game_data
真實 top-level 欄位，避免 schema drift。
"""


def _components_summary(components: dict | None) -> str:
    """Format tier_components as 'xFIP p75, K-BB% p60, velo p40'."""
    if not components:
        return "—"
    parts = []
    for label, key in (("xFIP", "xfip_pct"), ("K-BB%", "k_bb_pct"), ("velo", "velo_pct")):
        v = components.get(key)
        if v is None:
            continue
        parts.append(f"{label} p{int(v * 100)}")
    return ", ".join(parts) if parts else "—"


def _gap_str(tier_gap: dict | None) -> str:
    if not tier_gap or tier_gap.get("gap") is None:
        return "—"
    return f"{tier_gap['gap']:+.1f}"


def _pitcher_block(side: str, p: dict) -> list[str]:
    hand = p.get("pitch_hand", "?")
    age = p.get("age", "?")
    age_tag = p.get("age_assessment", "")
    tier_v2 = p.get("tier_v2") or "—"
    components = p.get("tier_components") or {}
    comp_str = _components_summary(components)
    gap = _gap_str(p.get("tier_gap"))

    return [
        f"### {p.get('name', '?')} ({side}, {hand}HP, {age} {age_tag})".rstrip(),
        f"- **Tier 驗證**：腳本 tier_v2 = {tier_v2}（{comp_str}），gap vs ERA-only = {gap}",
        "  - <!-- AI 補：是否同意 score-derived tier？若 |gap| ≥ 15 → 簡述運氣 vs 結構性，不自動下修預測 -->",
        "- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）",
        "  - <!-- AI 補：若 fired，本場對手核心打者手別組成是否放大此風險？ -->",
        "- **對手打線威脅**：<!-- AI 補：基於 dossier 投手對決表 + 上述兩信號 -->",
        "",
    ]


def _render_pitcher_matchup_section(bundle: dict) -> list[str]:
    """## 投手對決 — placeholder 從「合成」改「驗證 signal X」（PR-3 commit 15）。

    AI 不再從零合成 tier 判斷，改為驗證腳本給出的 tier_v2 + tier_gap +
    reverse_platoon 信號。降低 prompt 漂移風險，縮短 AI 推理鏈。
    """
    home_p = bundle.get("home_pitcher") or {}
    away_p = bundle.get("away_pitcher") or {}
    return [
        "## 投手對決",
        "",
        *_pitcher_block("HOME", home_p),
        *_pitcher_block("AWAY", away_p),
    ]


_HAND_TO_TIER_KEYS = {
    "L": ("tier_vs_lhp", "vs LHP"),
    "R": ("tier_vs_rhp", "vs RHP"),
}


def _lineup_block(side: str, l: dict, opposing_pitch_hand: str) -> list[str]:
    source = l.get("lineup_source", "projected")
    source_label = "🟢 official" if source == "official" else "🟡 projected（PA 排序近似 — 打線尚未公布）"
    season_tier = l.get("tier", "?")
    heat = l.get("recent_heat", "?")
    tier_key, matchup_label = _HAND_TO_TIER_KEYS.get(opposing_pitch_hand, (None, "vs ?HP"))
    matchup_tier = (l.get(tier_key) or "—") if tier_key else "—"
    return [
        f"### {side} — season tier {season_tier} / heat {heat}",
        f"- 打線來源：{source_label}",
        f"- **Matchup tier ({matchup_label})**：{matchup_tier}",
        "  - <!-- AI 補：matchup tier 與 season tier 落差 → 本場對打線評估方向（同意/上修/下修） -->",
        "- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`",
        "  - <!-- AI 補：若 fired，影響本場攻擊 chain 哪一段 → 簡述 -->",
        "",
    ]


def _render_lineup_section(bundle: dict) -> list[str]:
    """## 打線評級 — placeholder 從「合成」改「驗證 matchup tier + signals」。"""
    home_l = bundle.get("home_lineup") or {}
    away_l = bundle.get("away_lineup") or {}
    home_p = bundle.get("home_pitcher") or {}
    away_p = bundle.get("away_pitcher") or {}
    return [
        "## 打線評級",
        "",
        *_lineup_block("HOME", home_l, opposing_pitch_hand=away_p.get("pitch_hand", "?")),
        *_lineup_block("AWAY", away_l, opposing_pitch_hand=home_p.get("pitch_hand", "?")),
    ]


def _render_bullpen_section(bundle: dict) -> list[str]:
    from dossier_renderer import il_pitcher_count
    m = bundle.get("merged") or {}
    home_il = il_pitcher_count(bundle.get("home_roster"))
    away_il = il_pitcher_count(bundle.get("away_roster"))
    return [
        "## 牛棚",
        "",
        "| | HOME | AWAY |",
        "|---|---|---|",
        f"| ERA / 投手 IL / 核心 IL 估計 | {m.get('home_bullpen_era', '?')} / {home_il} / <!-- AI --> | "
        f"{m.get('away_bullpen_era', '?')} / {away_il} / <!-- AI --> |",
        "",
        "> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，"
        "對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。",
        "",
        "### 牛棚影響判讀",
        "- HOME 牛棚：<!-- AI 補：可用性 / 近 3 天消耗 / 對對手末段威脅 -->",
        "- AWAY 牛棚：<!-- AI 補：同上 -->",
        "",
    ]


# AI placeholders for Flag 8 / Flag 3 risk notes — single source of truth so the
# "不自動..." discipline text isn't repeated inline. Mirrors dossier_renderer
# `_FLAG8_TAIL` / `_FLAG3_TAIL` (different format: dossier is one-line summary,
# summary is two-line note + AI placeholder).
_FLAG8_AI_PLACEHOLDER = "  - <!-- AI 補：是運氣還結構性？是否影響本場判斷？不自動下修預測 -->"
_FLAG3_AI_PLACEHOLDER = "  - <!-- AI 補：可能回歸或可能持續？是否影響本場判斷？不自動 ±run value -->"


def _detect_risk_notes(bundle: dict) -> list[str]:
    """偵測 Flag 8 / Flag 3，回傳「條目 markdown 行」list（不含 H2 開頭）。"""
    from pitcher_stats import detect_triggers as detect_pitcher_triggers
    from lineup_analyzer import detect_triggers as detect_lineup_triggers
    notes = []
    for side in ("home", "away"):
        triggers = detect_pitcher_triggers(bundle.get(f"{side}_pitcher", {}))
        for t in triggers:
            if t.get("flag") != 8:
                continue
            val = t.get("value")
            name = t.get("name", "")
            if name == "ERA-xERA gap" and val is not None:
                detail = f"era_xera_delta={val:+.2f}"
            elif name == "Small-sample regression risk" and val is not None:
                detail = f"prior_year_regression={val:+.2f}, IP<30"
            else:
                detail = f"{name}={val}" if val is not None else name
            notes.append(f"- ⚠️ {side.upper()} 投手 Flag 8 ({detail}):")
            notes.append(_FLAG8_AI_PLACEHOLDER)
    for side in ("home", "away"):
        triggers = detect_lineup_triggers(bundle.get(f"{side}_lineup", {}))
        for t in triggers:
            if t.get("flag") == 3:
                babip = bundle.get(f"{side}_lineup", {}).get("last7_babip", "?")
                notes.append(f"- ⚠️ {side.upper()} 打線 Flag 3 (last7 BABIP={babip}):")
                notes.append(_FLAG3_AI_PLACEHOLDER)
    return notes


_SEVERITY_EMOJI_RISK = {"high": "🔴", "medium": "🟠", "low": "ℹ️"}

# Signals that are functionally redundant with Flag 8 / Flag 3 — exclude from
# 額外信號 to avoid double-listing (AI already handles them in the main notes).
_RISK_SECTION_EXCLUDED_SIGNALS = frozenset({"tier_mismatch", "heat_vs_babip"})


def _render_extra_signals(bundle: dict) -> list[str]:
    """### 額外信號 — non-flag fired signals (PR-3 commit 16).

    Returns empty list when no qualifying signals fire. Caller decides whether
    to inline these inside ## 風險提示.
    """
    from signals_lib import signals_for_bundle
    result = signals_for_bundle(bundle)
    fired = [
        s for s in result.get("signals", [])
        if s.get("fired") and s.get("name") not in _RISK_SECTION_EXCLUDED_SIGNALS
    ]
    if not fired:
        return []
    lines = ["", "### 額外信號"]
    for s in fired:
        emoji = _SEVERITY_EMOJI_RISK.get(s.get("severity", "low"), "ℹ️")
        side = s.get("side", "")
        side_prefix = f"{side} " if side and side != "GAME" else ""
        stale_badge = " ⏳" if s.get("half_life") == "short" else ""
        lines.append(f"- {emoji}{stale_badge} {side_prefix}{s.get('label', '')}")
    lines.append("  - <!-- AI 補：本場是否受此信號影響？是否與 Flag 3/8 雙重壓力 → 1-2 句敘事 -->")
    return lines


def _render_risk_section(bundle: dict) -> list[str]:
    notes = _detect_risk_notes(bundle)
    extra_signals = _render_extra_signals(bundle)
    if not notes and not extra_signals:
        return ["## 風險提示", "", "無風險提示", ""]
    if not notes:
        return ["## 風險提示", "", "Flag 3/8 無觸發；額外信號如下："] + extra_signals + [""]
    return ["## 風險提示", ""] + notes + extra_signals + [""]


def _render_weather_state_line(weather: dict | None) -> list[str]:
    """Return summary 的天氣狀態列（含可能的 AI placeholder 子行）。

    三狀態：
    - 有資料：「天氣：{condition}, {temp}°F, wind {wind}」+ AI 影響判讀子行
    - 室內：「天氣：室內（{condition}，不適用）」（無 AI 子行）
    - 缺資料：「天氣：未公布（跳過天氣分析）」（無 AI 子行）
    """
    if not weather:
        return ["- 天氣：未公布（跳過天氣分析）"]
    if weather.get("indoor"):
        cond = weather.get("condition", "Indoor")
        return [f"- 天氣：室內（{cond}，不適用）"]
    parts = []
    if weather.get("condition"):
        parts.append(weather["condition"])
    if weather.get("temp_f") is not None:
        parts.append(f"{weather['temp_f']}°F")
    if weather.get("wind_text"):
        parts.append(f"wind {weather['wind_text']}")
    if not parts:
        return ["- 天氣：未公布（跳過天氣分析）"]
    return [
        f"- 天氣：{', '.join(parts)}",
        "  - 影響判讀：<!-- AI 補：對得分 / HR 影響判讀 -->",
    ]


def _render_conditional_section(bundle: dict) -> list[str]:
    merged = bundle.get("merged") or {}
    pf = merged.get("park_factor", 100)
    pf_correction = (pf - 100) * 0.05
    lines = [
        "## 條件修正",
        "",
        f"- Park Factor: {pf} → {pf_correction:+.2f} run",
    ]
    lines += _render_weather_state_line(merged.get("weather"))
    lines += [
        "- 先發 tier / doubleheader：<!-- AI 補 -->",
        "",
    ]
    return lines


def _render_expected_runs_section(bundle: dict, formula_pred: dict) -> list[str]:
    home_base = formula_pred.get("home_score", "?")
    away_base = formula_pred.get("away_score", "?")
    total_base = (
        (home_base + away_base)
        if isinstance(home_base, (int, float)) and isinstance(away_base, (int, float))
        else "?"
    )
    return [
        "## 修正後預期得分",
        "",
        "> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。",
        "> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。",
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
        "- **方向（基本面）**：<!-- AI 補 HOME / AWAY / 持平 -->",
        "- **總分（基本面）**：<!-- AI 補 數值（formula base ± 信號修正後） -->",
        "- **方向信心**：<!-- AI 補 50-75% 機率。**門檻（2026-05 回測校準，MEDIUM 桶 over-confident −10pp 故收緊）**：≤ 55% 寫「持平」；55-65% 須同時滿足 (adjusted total HOME-AWAY gap ≥ 0.8 run) AND (無同強度反向信號) 否則改寫「持平」；> 75% 需在風險段說明依據 -->",
        "- **風險**：<!-- AI 補 1-4 點 -->",
        "",
        "⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組",
    ]


def render_summary(bundle: dict, formula_pred: dict) -> str:
    """主入口：渲染 summary.md template，回傳 markdown 字串（不寫檔；caller 寫檔）。"""
    lines: list[str] = []
    lines += _render_pitcher_matchup_section(bundle)
    lines += _render_lineup_section(bundle)
    lines += _render_bullpen_section(bundle)
    lines += _render_risk_section(bundle)
    lines += _render_conditional_section(bundle)
    lines += _render_expected_runs_section(bundle, formula_pred)
    lines += _render_overall_section()
    return "\n".join(lines)
