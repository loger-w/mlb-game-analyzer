#!/usr/bin/env python3
"""MLB Game Predictor — Log5 + 期望得分公式 + 信號計分表"""

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Fix Windows encoding for emoji output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# F2: 完整 30 隊隊名 → 縮寫映射（用於方向矛盾檢查）
TEAM_ABBREV = {
    "New York Yankees": "NYY", "New York Mets": "NYM", "Boston Red Sox": "BOS",
    "Los Angeles Dodgers": "LAD", "Los Angeles Angels": "LAA", "Houston Astros": "HOU",
    "Atlanta Braves": "ATL", "Philadelphia Phillies": "PHI", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "St. Louis Cardinals": "STL", "Milwaukee Brewers": "MIL",
    "Pittsburgh Pirates": "PIT", "Arizona Diamondbacks": "ARI", "Colorado Rockies": "COL",
    "Baltimore Orioles": "BAL", "Tampa Bay Rays": "TB", "Toronto Blue Jays": "TOR",
    "Minnesota Twins": "MIN", "Kansas City Royals": "KC", "Detroit Tigers": "DET",
    "Cleveland Guardians": "CLE", "Seattle Mariners": "SEA", "Athletics": "OAK",
    "Texas Rangers": "TEX", "Miami Marlins": "MIA", "Washington Nationals": "WSH",
}

# W4: --game-data 路徑規範（Plan B 2026-04-22 §4.2）
# 合法格式: analysis-data/YYYY-MM-DD/AWAY@HOME[-G1|-G2]/merged.json
GAME_DATA_PATTERN = re.compile(
    r"analysis-data[/\\]\d{4}-\d{2}-\d{2}[/\\][A-Z]{2,3}@[A-Z]{2,3}(-G[12])?[/\\]merged\.json$"
)

# W3: signal_adjustments 合法 key 前綴 / 模式（Plan B 2026-04-22 §4.2）
# 設計：prefix + pattern，非靜態完整 key（pitcher 名每週變動）
# 2026-04-18~21 predictions.jsonl 觀察到 94 unique keys；以下涵蓋其中大多數。
SIGNAL_KEYS_PREFIXES = frozenset({
    "bullpen_", "weather_", "cold_", "babip_", "park_",
    "coors_", "lineup_", "both_", "env_", "home_", "away_",
    "wind_", "sp_", "early_", "pitcher_",
})

# pitcher / player 個人化 signal 的後綴模式
SIGNAL_PITCHER_SUFFIX_RE = re.compile(
    r"_(velo|xera|era|bb|hr|hrluck|luck|new|role|small|ev|partial|gap|"
    r"regression|decline|collapse|wildness|drop|arsenal|il|out|variance|"
    r"platoon|disadvantage|reversion|reversal|warmth|vs|improvement|"
    r"half|solid|strong|weak|depth|edge|buffer|k_collapse|tempered|"
    r"cluster|cold|hot|in|pair|comerica|oracle|april|rhh|lhh|luck_regression|"
    r"age_risk|walk_collapse|hr_risk)(_|$)"
)


# === RL relaxation thresholds (see docs/specs/2026-04-20-rl-threshold-relaxation.md) ===
RL_DIFF_MIN = 1.5       # 觸發 RL-1b 最低分差
RL_DIFF_BIG = 2.2       # 免強 tag 門檻
RL_DIFF_STAR = 2.0      # 1★/2★ 邊界
RL_STRONG_TAGS = frozenset({
    "home-pitching-slump", "away-pitching-slump",
    "home-bullpen-slump", "away-bullpen-slump",
})
BULLPEN_SLUMP_ERA = 5.0  # 對齊既有 signal_table bullpen 規則
BULLPEN_STRONG_ERA = 3.0

def log5(home_pct: float, away_pct: float) -> float:
    """Log5 勝率公式"""
    p = (home_pct * (1 - away_pct)) / (home_pct * (1 - away_pct) + away_pct * (1 - home_pct))
    return p


def apply_close_game_cap(
    adj_home: float, adj_away: float, current_cap: int
) -> tuple[int, str | None]:
    """Y-new-2（Plan B §4.4）：調整後比分差 < 0.5 → 上限 1 星（SD ≈ 4.5，噪音範圍）。

    cumulative #3 連 4 天觸發，規則下沉到 code 層。
    """
    diff = abs(adj_home - adj_away)
    if diff < 0.5:
        reason = f"近身戰 |adj 比分差|={diff:.2f} < 0.5 上限 1（Y-new-2）"
        return min(current_cap, 1), reason
    return current_cap, None


def apply_divergent_user_tag_cap(
    user_tags: list[str], current_cap: int
) -> tuple[int, str | None]:
    """Y-new-3（Plan B §4.4）：user-supplied 'divergent' tag → 上限 2 星。

    'divergent' 是 Phase 3 Claude 手動加的 user tag（非 compute_trend_tags 自動產生），
    代表基本面判讀與模型輸出方向不一致。cumulative #4 顯示推薦場 0W-4L。
    """
    if "divergent" in user_tags:
        return min(current_cap, 2), "'divergent' tag 上限 2（Y-new-3）"
    return current_cap, None


def should_add_home_2star_tag(
    predicted_winner: str, final_ml_stars: int | None, final_ml_rec: str | None
) -> bool:
    """Y-new-1（Plan B §4.4）：主場 2 星推薦 audit tag（cumulative #1 連 4 天觸發）。

    規則尚在觀察期（條件難定義，先 audit-only），tag `home-2star-risk` 供
    post-game review 分桶；不 cap / 不 force PASS。
    """
    return (
        predicted_winner == "HOME"
        and final_ml_stars == 2
        and final_ml_rec is not None
        and final_ml_rec != "PASS"
    )


def validate_ml_rec(ml_rec: str | None, team_abbrevs: set[str]) -> None:
    """W2（Plan B 2026-04-22 §4.2）：`--ml-rec` 必須是 team abbr / PASS / None。

    舊 bug：傳 "HOME"/"AWAY" 字面值會寫進 predictions.jsonl，導致
    `review_stats.is_home_team` 查表失敗 → 反向判 WIN/LOSS（cumulative #9）。
    """
    valid = team_abbrevs | {"PASS"} | {None}
    if ml_rec not in valid:
        sorted_abbrs = sorted(team_abbrevs)
        sys.exit(
            f"⛔ --ml-rec 必須是 team abbr（如 NYY）或 PASS，收到 {ml_rec!r}\n"
            f"  合法值: {sorted_abbrs + ['PASS']}"
        )


def validate_game_data_path(path: str) -> None:
    """W4（Plan B §4.2）：`--game-data` 路徑必須是 analysis-data/{date}/{AWAY}@{HOME}[-G1|G2]/merged.json。

    支援 Windows 反斜線 + absolute / relative path。腦補路徑（如 /tmp/foo.json）直接 reject。
    """
    normalized = path.replace("\\", "/")
    if not GAME_DATA_PATTERN.search(normalized):
        sys.exit(
            f"⛔ --game-data 路徑不符規範: {path}\n"
            f"  合法格式: analysis-data/YYYY-MM-DD/AWAY@HOME[-G1|-G2]/merged.json"
        )


# W3: team abbr 小寫前綴（在函數定義之後初始化，確保 TEAM_ABBREV 已載入）
_SIGNAL_TEAM_PREFIXES = frozenset(
    abbr.lower() + "_" for abbr in TEAM_ABBREV.values()
)


def _is_known_signal_key(key: str) -> bool:
    """W3: 判斷 signal key 是否在 allowlist 範圍（prefix / team / pitcher pattern）。"""
    for prefix in SIGNAL_KEYS_PREFIXES:
        if key.startswith(prefix):
            return True
    for prefix in _SIGNAL_TEAM_PREFIXES:
        if key.startswith(prefix):
            return True
    if SIGNAL_PITCHER_SUFFIX_RE.search(key):
        return True
    return False


def warn_unknown_signal_keys(signals: dict | None) -> None:
    """W3（Plan B §4.2）：未知 signal_adjustments key → stderr warning，不 exit。

    原則：允許新 signal 擴充（pitcher 名每週變動），但 typo / 幻想 key 留記錄。
    """
    if not signals:
        return
    unknown = [k for k in signals if not _is_known_signal_key(k)]
    if unknown:
        print(
            f"⚠️ unknown signal key(s): {sorted(unknown)}\n"
            f"  若為新 signal 請更新 SIGNAL_KEYS_PREFIXES；若為 typo 請修正",
            file=sys.stderr,
        )


def pythagorean_runs(rs: float, ra: float, g: float = 10) -> float:
    """Pythagenport 動態指數公式（Smyth & Patriot, 2003）

    exponent = 1.50 × log10[(RS + RA) / G] + 0.45
    Pythagenport RMSE = 3.991 勝（優於固定指數 1.83 的 4.126）
    """
    if rs + ra == 0:
        return 0.5
    exponent = 1.50 * math.log10((rs + ra) / g) + 0.45
    return (rs ** exponent) / (rs ** exponent + ra ** exponent)


def compute_trend_tags(data: dict) -> list[str]:
    """比較近 10 場 vs 本季平均，產出趨勢標籤"""
    tags = []
    for side in ("home", "away"):
        rs_10 = data.get(f"{side}_recent_rs", 4.5)
        ra_10 = data.get(f"{side}_recent_ra", 4.5)
        rs_season = data.get(f"{side}_season_rs", rs_10)
        ra_season = data.get(f"{side}_season_ra", ra_10)

        if rs_season > 0 and rs_10 > rs_season * 1.2:
            tags.append(f"{side}-hot-offense")
        elif rs_season > 0 and rs_10 < rs_season * 0.8:
            tags.append(f"{side}-cold-offense")

        if ra_season > 0 and ra_10 > ra_season * 1.2:
            tags.append(f"{side}-pitching-slump")
        elif ra_season > 0 and ra_10 < ra_season * 0.8:
            tags.append(f"{side}-pitching-hot")

        bp_era = data.get(f"{side}_bullpen_era", 4.0)
        if bp_era >= BULLPEN_SLUMP_ERA:
            tags.append(f"{side}-bullpen-slump")
        elif bp_era <= BULLPEN_STRONG_ERA:
            tags.append(f"{side}-bullpen-strong")

    return tags


def _inactive_rl_override() -> dict:
    """Return rl_override dict with active=False (all other fields null).

    See docs/specs/2026-04-20-rl-threshold-relaxation.md (Step 4):
    rl_override 欄位永遠存在，inactive 時其他欄位為 null，
    讓下游 parser 能用固定路徑讀 active flag 而不擔心 KeyError。
    """
    return {
        "active": False,
        "path": None,
        "diff": None,
        "stars": None,
        "tags": None,
        "warnings": None,
        "thresholds": None,
    }


def apply_rl_guardrail(
    *,
    adj_home: float,
    adj_away: float,
    trend_tags: list[str],
    predicted_winner: str,
    home_team: str,
    away_team: str,
) -> tuple[str, int | None, dict]:
    """Apply RL guardrails and produce rl_override audit dict.

    Plan B 2026-04-22（W1）：廢除 `--run-line-rec` / `--run-line-stars` CLI args；
    RL 全走 auto override。沒有 user input 路徑。

    Rules:
      RL-1b (auto-推薦): |diff| >= RL_DIFF_MIN → auto-upgrade to team-abbr + 1/2 stars.
                        A  |diff| >= RL_DIFF_BIG                  → big-diff (no tag needed)
                        B  RL_DIFF_MIN <= |diff| < RL_DIFF_BIG + strong tag
                                                                  → mid-diff+strong-tag
                        Stars: |diff| <= RL_DIFF_STAR → 1; else 2.

    Defensive (Q4): if predicted_winner != diff_side, record
    'pw_diff_direction_mismatch' warning but do not block the override.

    Args:
      adj_home / adj_away: adjusted scores (already applied signal/adjusted args)
      trend_tags: full trend_tags list from compute_trend_tags(data)
      predicted_winner: 'HOME' | 'AWAY' (result['final']['recommended_winner'])
      home_team / away_team: full team names (used by TEAM_ABBREV lookup)

    Returns:
      (final_rl_rec, final_rl_stars, rl_override_dict)
      Note: final_rl_stars is None when no override fires AND final_rl_rec stays "PASS".
    """
    final_rl_rec = "PASS"
    final_rl_stars = None
    rl_override = _inactive_rl_override()

    diff = abs(adj_home - adj_away)
    diff_side = "HOME" if adj_home > adj_away else "AWAY"
    strong_rl = RL_STRONG_TAGS & set(trend_tags)

    # RL-1b gate: 只讀 diff + tags
    override_path = None
    if diff >= RL_DIFF_MIN:
        if diff >= RL_DIFF_BIG:
            override_path = "big-diff"
        elif strong_rl:
            override_path = "mid-diff+strong-tag"

    if override_path is not None:
        warnings: list[str] = []
        if predicted_winner != diff_side:
            warnings.append("pw_diff_direction_mismatch")
            print(
                f"⚠️ RL-1b: predicted_winner={predicted_winner} 與 diff_side={diff_side} 不一致 "
                f"(pw 應來自同一 adj_home/adj_away，可能是上游 bug)",
                file=sys.stderr,
            )

        fav_team = home_team if diff_side == "HOME" else away_team
        fav_abbr = TEAM_ABBREV.get(fav_team, "")

        if fav_abbr:  # 防禦：abbr 查不到時不 override
            stars = 2 if diff > RL_DIFF_STAR else 1
            final_rl_rec = fav_abbr
            final_rl_stars = stars
            rl_override = {
                "active": True,
                "path": override_path,
                "diff": round(diff, 2),
                "stars": stars,
                "tags": sorted(strong_rl),
                "warnings": warnings,
                "thresholds": {
                    "diff_min": RL_DIFF_MIN,
                    "diff_big": RL_DIFF_BIG,
                    "diff_star": RL_DIFF_STAR,
                },
            }
            print(
                f"ℹ️ RL-1b 自動推薦（{override_path}）：|diff|={diff:.2f} "
                f"tags={sorted(strong_rl) or '(pure-diff)'} → {fav_abbr} {stars}★",
                file=sys.stderr,
            )

    return final_rl_rec, final_rl_stars, rl_override


def compute_signal_table(data: dict) -> dict:
    """F1: 信號計分表 — 使用 Run Value 修正（對齊 reference/prediction.md）

    每個信號轉為 ±run 修正值，最終加總到預測比分上。
    正值 = 得分上升（Over 方向），負值 = 得分下降（Under 方向）。
    """
    signals = []

    # --- 總分上修信號 ---

    # 雙方打線近期 Hot（場均 ≥ 5）
    home_rs = data.get("home_recent_rs", 4.5)
    away_rs = data.get("away_recent_rs", 4.5)
    if home_rs >= 5 and away_rs >= 5:
        signals.append({"signal": "雙方打線近期 Hot（場均 ≥ 5 分）", "run_value": +0.5})
    if home_rs <= 2 and away_rs <= 2:
        signals.append({"signal": "雙方打線近期 Cold（場均 ≤ 2 分）", "run_value": -0.5})

    # 牛棚 ERA ≥ 5.0（對手得分上升）
    home_bp = data.get("home_bullpen_era", 4.0)
    away_bp = data.get("away_bullpen_era", 4.0)
    if home_bp >= 5.0:
        signals.append({"signal": f"主隊牛棚 ERA {home_bp:.2f} ≥ 5.0", "run_value": +0.5})
    if away_bp >= 5.0:
        signals.append({"signal": f"客隊牛棚 ERA {away_bp:.2f} ≥ 5.0", "run_value": +0.5})

    # Park Factor 修正：(PF - 100) × 0.05
    pf = data.get("park_factor", 100)
    pf_adj = round((pf - 100) * 0.05, 2)
    if abs(pf_adj) >= 0.1:
        signals.append({"signal": f"Park Factor {pf}（修正 {pf_adj:+.2f}）", "run_value": pf_adj})

    # 雙方先發投手等級
    home_fip = data.get("home_starter_fip", 4.0)
    away_fip = data.get("away_starter_fip", 4.0)
    if home_fip <= 3.0 and away_fip <= 3.0:
        signals.append({"signal": "雙方先發 FIP ≤ 3.0（Ace 級）", "run_value": -1.0})
    elif home_fip <= 3.2 and away_fip <= 3.2:
        signals.append({"signal": "雙方先發 FIP ≤ 3.2（Strong Ace+）", "run_value": -0.5})

    # 打線 xwOBA
    home_xwoba = data.get("home_batting_xwoba", 0.320)
    away_xwoba = data.get("away_batting_xwoba", 0.320)
    if home_xwoba >= 0.350 and away_xwoba >= 0.350:
        signals.append({"signal": "雙方打線 xwOBA ≥ .350", "run_value": +0.5})

    # 打線 K%
    home_k = data.get("home_batting_k_pct", 20)
    away_k = data.get("away_batting_k_pct", 20)
    if home_k >= 25 and away_k >= 25:
        signals.append({"signal": "雙方打線 K% ≥ 25%", "run_value": -0.3})

    total_run_adj = round(sum(s["run_value"] for s in signals), 2)

    return {
        "signals": signals,
        "total_run_adjustment": total_run_adj,
    }


def _format_pct_with_flip(
    formula_pct: float,
    predicted_winner: str,
    adj_home: float,
    adj_away: float,
    has_adjusted: bool,
) -> str:
    """渲染勝率行；adjusted 比分翻轉方向時加 ⚠️ 註明。

    formula_pct 是 home_win_pct（永遠以主隊視角）；side label 固定為 HOME。
    翻轉條件：has_adjusted=True 且 (formula_pct > 50) != (predicted_winner == "HOME")
    """
    if not has_adjusted:
        return f"Formula log5: **{formula_pct:.1f}% (HOME)**"
    formula_winner = "HOME" if formula_pct > 50 else "AWAY"
    if formula_winner == predicted_winner:
        return f"Formula log5: **{formula_pct:.1f}% (HOME)**"
    cmp = "<" if adj_home < adj_away else ">"
    # Format adjusted scores: use 2 decimal places, but strip trailing zeros after the decimal point
    # while preserving at least one decimal place (e.g., 5.0, 4.85)
    adj_home_str = f"{adj_home:.2f}".rstrip("0").rstrip(".") if "." in f"{adj_home:.2f}" else f"{adj_home:.1f}"
    adj_away_str = f"{adj_away:.2f}".rstrip("0").rstrip(".") if "." in f"{adj_away:.2f}" else f"{adj_away:.1f}"
    # Ensure at least one decimal place
    if "." not in adj_home_str:
        adj_home_str += ".0"
    if "." not in adj_away_str:
        adj_away_str += ".0"
    return (
        f"⚠️ Formula {formula_pct:.1f}% (HOME) → adjusted 比分 "
        f"{adj_home_str} {cmp} {adj_away_str} 判 {predicted_winner} 勝"
        "（pct 未隨翻轉重算）"
    )


def format_signal_table_md(auto_signals: list[dict], user_signals: dict) -> str:
    """組 auto + user 兩個 mini-table；各自空時顯示「（無）」一行。"""
    lines = ["### Auto signals"]
    if auto_signals:
        lines.append("| 信號 | ±run |")
        lines.append("|------|------|")
        total = 0.0
        for s in auto_signals:
            rv = s["run_value"]
            lines.append(f"| {s['signal']} | {rv:+.2f} |")
            total += rv
        lines.append(f"| **總和** | **{total:+.2f}** |")
    else:
        lines.append("（無）")

    lines.append("")
    lines.append("### User-supplied signals")
    if user_signals:
        lines.append("| Key | ±run |")
        lines.append("|-----|------|")
        total = 0.0
        for k, v in user_signals.items():
            lines.append(f"| `{k}` | {v:+.2f} |")
            total += v
        lines.append(f"| **總和** | **{total:+.2f}** |")
    else:
        lines.append("（無）")
    return "\n".join(lines)


def _stars_str(stars: int | None) -> str:
    """渲染星級；None / 0 → '—'，否則 ⭐ * stars。"""
    if not stars:
        return "—"
    return "⭐" * stars


def format_recommendation_rows(
    record: dict, cap_reasons: list[str]
) -> tuple[str, str]:
    """產生 (tldr_table_md, full_rows_md)。共用 reason 字串確保 TL;DR 與
    推薦結果 section 一致（同來源避免漂移）。

    一句話理由規則見 spec section 2「推薦行 一句話理由 + tag 折進規則」。
    """
    home_team = record.get("home_team", "")
    away_team = record.get("away_team", "")
    home_abbr = TEAM_ABBREV.get(home_team, home_team[:3].upper())
    away_abbr = TEAM_ABBREV.get(away_team, away_team[:3].upper())
    pct = record.get("predicted_home_pct", 0.0)
    pw = record.get("predicted_winner", "HOME")
    side = "HOME" if pct > 50 else "AWAY"
    tags = record.get("tags") or []

    # ===== ML row =====
    ml_rec = record.get("ml_rec") or "PASS"
    ml_stars = record.get("ml_stars")
    original_ml_stars = record.get("original_ml_stars")

    # Folded tags for ML row
    ml_folded = [t for t in tags if t in ("divergent", "direction-override", "home-2star-risk")]

    if ml_rec == "PASS":
        ml_dir = "PASS"
        ml_stars_str = "—"
        ml_reason = f"Log5 {pct:.1f}% ({side})"
    else:
        ml_dir = ml_rec
        ml_stars_str = _stars_str(ml_stars)
        reason_parts = [f"Log5 {pct:.1f}% ({side})"]
        if ml_folded:
            reason_parts.append("audit " + ", ".join(f"`{t}`" for t in ml_folded))
        ml_reason = "，".join(reason_parts)

    # ===== O/U row =====
    ou_rec = record.get("ou_rec") or "PASS"
    ou_stars = record.get("ou_stars")
    ou_line = record.get("ou_line")
    adj_total = record.get("adjusted_total")

    if ou_line is not None and adj_total is not None:
        gap = abs(adj_total - ou_line)
        gap_str = f"adj_total {adj_total:.1f} vs line {ou_line}，差距 {gap:.1f} run"
    else:
        gap = None
        gap_str = "—"

    if ou_rec == "PASS":
        ou_dir = "PASS"
        ou_stars_str = "—"
        if gap is not None and gap < 1.5:
            ou_reason = f"差距 {gap:.1f} < 1.5 run"
        else:
            ou_reason = gap_str
    else:
        ou_dir = ou_rec
        ou_stars_str = _stars_str(ou_stars)
        ou_reason = gap_str

    # ===== Run Line row =====
    rl_rec = record.get("run_line_rec") or "PASS"
    rl_stars = record.get("run_line_stars")
    rl_override = record.get("rl_override") or {}

    if rl_override.get("active"):
        rl_dir = rl_rec
        rl_stars_str = _stars_str(rl_stars)
        path = rl_override.get("path", "?")
        diff = rl_override.get("diff", 0.0)
        ov_tags = rl_override.get("tags") or []
        if ov_tags:
            tag_str = ", ".join(f"`{t}`" for t in ov_tags)
            rl_reason = f"override `{path}`，|diff|={diff:.1f}，tags={tag_str}"
        else:
            rl_reason = f"override `{path}`，|diff|={diff:.1f}"
    else:
        rl_dir = "PASS"
        rl_stars_str = "—"
        adj_home = record.get("predicted_home_score") or 0
        adj_away = record.get("predicted_away_score") or 0
        diff = abs(adj_home - adj_away)
        rl_reason = f"|diff|={diff:.1f} < 1.5（RL_DIFF_MIN）"

    # ===== TL;DR table =====
    tldr_lines = [
        "| 市場 | 方向 | 推薦指數 | 一句話理由 |",
        "|------|------|----------|-----------|",
        f"| ML | {ml_dir} | {ml_stars_str} | {ml_reason} |",
        f"| O/U | {ou_dir} | {ou_stars_str} | {ou_reason} |",
        f"| Run Line | {rl_dir} | {rl_stars_str} | {rl_reason} |",
    ]
    tldr = "\n".join(tldr_lines)

    # ===== Full rows =====
    def _full_row(market: str, direction: str, stars_str: str, reason: str) -> str:
        if stars_str == "—":
            head = f"**{direction}**"
        else:
            head = f"**{direction} {stars_str}**"
        return f"- **{market}**: {head} — {reason}"

    full_lines = [_full_row("ML", ml_dir, ml_stars_str, ml_reason)]
    # ML cap reason appended
    if (original_ml_stars is not None and ml_stars is not None
            and original_ml_stars > ml_stars and cap_reasons):
        full_lines[0] += f"（原 {_stars_str(original_ml_stars)} 降為 {_stars_str(ml_stars)}：{'; '.join(cap_reasons)}）"

    full_lines.append(_full_row("O/U", ou_dir, ou_stars_str, ou_reason))
    full_lines.append(_full_row("Run Line", rl_dir, rl_stars_str, rl_reason))

    full = "\n".join(full_lines)
    return tldr, full


def format_discipline_check(record: dict) -> str:
    """渲染 D1-D5 紀律檢查 4 行（D4 已棄用）。"""
    home_team = record.get("home_team", "")
    away_team = record.get("away_team", "")
    home_abbr = TEAM_ABBREV.get(home_team, home_team[:3].upper())
    away_abbr = TEAM_ABBREV.get(away_team, away_team[:3].upper())
    pw = record.get("predicted_winner", "")
    ml_rec = record.get("ml_rec") or "PASS"
    ou_line = record.get("ou_line")
    ou_rec = record.get("ou_rec") or "PASS"
    adj_total = record.get("adjusted_total")
    rl_rec = record.get("run_line_rec") or "PASS"
    tags = record.get("tags") or []

    lines = []

    # D1: predicted_winner 方向是否與 ml_rec 一致
    if "direction-override" in tags:
        lines.append(
            f"- ⚠️ D1 模型方向：direction-override（ml_rec={ml_rec}, predicted_winner={pw}）"
        )
    elif ml_rec == "PASS":
        lines.append("- ✅ D1 模型方向：ml_rec=PASS")
    else:
        winner_abbr = home_abbr if pw == "HOME" else away_abbr
        if ml_rec == winner_abbr:
            lines.append(
                f"- ✅ D1 模型方向：predicted_winner={pw}({winner_abbr}) 與 ml_rec={ml_rec} 一致"
            )
        else:
            lines.append(
                f"- ⚠️ D1 模型方向：predicted_winner={pw}({winner_abbr}) 與 ml_rec={ml_rec} 不一致"
            )

    # D2: 信號量化（永遠 ✅，predict.py 只接受 run_value 形式）
    lines.append("- ✅ D2 信號量化：所有信號已轉為 run value")

    # D3: 同場無對立推薦
    if rl_rec == "PASS" or ml_rec == "PASS":
        lines.append("- ✅ D3 同場無對立推薦")
    else:
        opposite = (
            (ml_rec == home_abbr and rl_rec == away_abbr)
            or (ml_rec == away_abbr and rl_rec == home_abbr)
        )
        if opposite:
            lines.append(
                f"- ⚠️ D3 同場推對立：ml_rec={ml_rec} + run_line_rec={rl_rec}"
            )
        else:
            lines.append("- ✅ D3 同場無對立推薦")

    # D5: 比分盤口一致
    if ou_rec == "PASS" or ou_line is None or adj_total is None:
        lines.append("- ✅ D5 比分盤口一致：ou_rec=PASS 或無 line")
    else:
        if ou_rec == "OVER" and adj_total > ou_line:
            lines.append(
                f"- ✅ D5 比分盤口一致：adj_total {adj_total} > ou_line {ou_line} vs ou_rec=OVER"
            )
        elif ou_rec == "UNDER" and adj_total < ou_line:
            lines.append(
                f"- ✅ D5 比分盤口一致：adj_total {adj_total} < ou_line {ou_line} vs ou_rec=UNDER"
            )
        else:
            lines.append(
                f"- ⚠️ D5 比分盤口矛盾：adj_total {adj_total} vs ou_line {ou_line}, ou_rec={ou_rec}"
            )

    return "\n".join(lines)


def format_rl_override_block(rl_override: dict) -> str | None:
    """RL override active=True 時渲染細節 section；inactive 回 None。"""
    if not rl_override or not rl_override.get("active"):
        return None
    path = rl_override.get("path")
    diff = rl_override.get("diff")
    stars = rl_override.get("stars")
    tags = rl_override.get("tags") or []
    warnings = rl_override.get("warnings") or []
    thr = rl_override.get("thresholds") or {}

    lines = ["## Run Line override 細節"]
    lines.append(f"- 路徑: `{path}`")
    if diff is not None:
        lines.append(f"- |diff|: {diff:.2f}")
    if stars is not None:
        lines.append(f"- stars: {stars}")
    if tags:
        lines.append(f"- 觸發 tags: {', '.join(f'`{t}`' for t in tags)}")
    if warnings:
        lines.append(f"- ⚠️ warnings: {', '.join(warnings)}")
    if thr:
        lines.append(
            f"- thresholds: diff_min={thr.get('diff_min')}, "
            f"diff_big={thr.get('diff_big')}, diff_star={thr.get('diff_star')}"
        )
    return "\n".join(lines)


def format_env_block(record: dict) -> str | None:
    """渲染環境補充 section；所有欄位皆 null → None（整段省略）。"""
    fields = [
        ("氣溫 (°F)", record.get("temperature_f")),
        ("風速 (mph)", record.get("wind_mph")),
        ("風向", record.get("wind_direction")),
        ("主審", record.get("umpire_name")),
        ("主審 Over%", record.get("umpire_ou_rate")),
    ]
    non_null = [(k, v) for k, v in fields if v is not None]
    if not non_null:
        return None
    lines = ["## 環境補充"]
    for k, v in non_null:
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def format_trend_tags_block(tags: list[str], recommendation_tags: set[str]) -> str | None:
    """扣除已折進推薦行的 tags；剩下空 → None。"""
    remaining = [t for t in tags if t not in recommendation_tags]
    if not remaining:
        return None
    lines = ["## 趨勢標記"]
    lines.append("- " + "、".join(f"`{t}`" for t in remaining))
    return "\n".join(lines)


def format_prediction_summary_md(
    record: dict, signal_table: dict, cap_reasons: list[str]
) -> str:
    """組合 prediction_summary.md 完整內容。
    Hard sections（必出現）：TL;DR / 比分預測 / 勝率預測 / 信號修正表 / 推薦結果 / 紀律檢查
    Soft sections（缺資料省略）：Run Line override 細節 / 環境補充 / 趨勢標記
    Fail-fast：缺 home_team / away_team / predicted_winner → raise ValueError
    """
    if "home_team" not in record or "away_team" not in record:
        raise ValueError("record missing home_team / away_team")
    if "predicted_winner" not in record:
        raise ValueError("record missing predicted_winner")

    home_team = record["home_team"]
    away_team = record["away_team"]
    home_abbr = TEAM_ABBREV.get(home_team, home_team[:3].upper())
    away_abbr = TEAM_ABBREV.get(away_team, away_team[:3].upper())
    date = record.get("date", "—")

    pw = record["predicted_winner"]
    pct = record.get("predicted_home_pct", 0.0)
    home_score = record.get("predicted_home_score", 0.0)
    away_score = record.get("predicted_away_score", 0.0)
    formula_home = record.get("formula_home_score", 0.0)
    formula_away = record.get("formula_away_score", 0.0)
    formula_total = round(formula_home + formula_away, 1)
    adj_total = record.get("adjusted_total", 0.0)
    ou_line = record.get("ou_line")

    # has_adjusted: predicted_score 與 formula_score 不同 = 用戶有傳 --adjusted-*
    has_adjusted = (
        abs(formula_home - home_score) > 0.01 or abs(formula_away - away_score) > 0.01
    )

    tldr_table, full_rows = format_recommendation_rows(record, cap_reasons)

    # 開打時間 meta（spec 2026-04-29）：TW 為主、ET 副欄
    game_time_iso = record.get("game_time")
    if game_time_iso:
        try:
            utc_dt = datetime.fromisoformat(game_time_iso.replace("Z", "+00:00"))
            tw_label = utc_dt.astimezone(_TW_TZ).strftime("%Y-%m-%d %H:%M TW")
            et_label = utc_dt.astimezone(_ET_TZ).strftime("%m-%d %H:%M")
            time_label = f"{tw_label}（ET {et_label}）"
        except ValueError:
            time_label = "未知"
    else:
        time_label = "未知"

    lines = [
        f"# Prediction Summary — {away_abbr} @ {home_abbr} ({date})",
        "",
        f"**開打時間**: {time_label}",
        "",
        "## TL;DR",
        f"- 預測比分: **{home_abbr} {home_score:.1f} − {away_score:.1f} {away_abbr}**"
        f"（{pw} 勝，勝率 {pct:.1f}%）",
        "- 比賽走勢: <!-- narrative: AI 根據先發 tier、牛棚壓力、打線強度選 1-2 句描述比賽走向（不含星級 / 盤口） -->",
        "",
        "📊 推薦速查:",
        "",
        tldr_table,
        "",
        "---",
        "",
        "## 比分預測",
        f"- Formula 比分: {home_abbr} {formula_home:.1f} / {away_abbr} {formula_away:.1f}"
        f"（總分 {formula_total:.1f}）",
    ]
    if has_adjusted:
        lines.append(
            f"- Adjusted 比分: {home_abbr} {home_score:.1f} / {away_abbr} {away_score:.1f}"
            f"（總分 {adj_total:.1f}）"
        )
    if ou_line is not None:
        gap = abs(adj_total - ou_line)
        lines.append(
            f"- O/U gap: |adj_total {adj_total:.1f} − line {ou_line}| = {gap:.1f}"
        )

    lines.extend([
        "",
        "## 勝率預測",
        f"- {_format_pct_with_flip(pct, pw, home_score, away_score, has_adjusted)}",
        "",
        "## 信號修正表",
        "",
        format_signal_table_md(
            signal_table.get("signals") or [],
            record.get("signal_adjustments") or {},
        ),
        "",
        "## 推薦結果",
        full_rows,
        "",
        "## 紀律檢查 (D1-D5)",
        format_discipline_check(record),
    ])

    # Soft sections
    rl_block = format_rl_override_block(record.get("rl_override") or {})
    if rl_block:
        lines.extend(["", rl_block])

    env_block = format_env_block(record)
    if env_block:
        lines.extend(["", env_block])

    # Build folded tags set: 已被推薦行折入的不再進趨勢標記
    tags = record.get("tags") or []
    folded: set[str] = set()
    for t in ("divergent", "direction-override", "home-2star-risk"):
        if t in tags:
            folded.add(t)
    rl_override = record.get("rl_override") or {}
    if rl_override.get("active") and rl_override.get("tags"):
        folded.update(rl_override["tags"])

    trend_block = format_trend_tags_block(tags, folded)
    if trend_block:
        lines.extend(["", trend_block])

    return "\n".join(lines).rstrip() + "\n"


def predict_with_formula(data: dict) -> dict:
    """F3: 用 Log5 + 期望得分公式預測（納入對方投手壓制力）

    E[R] = 聯盟平均得分 × (打線 xwOBA / 聯盟 xwOBA) × (對方投手 ERA / 聯盟 ERA) × (PF / 100)
    聯盟平均（2024-2025 基準）：R/G ≈ 4.5, xwOBA ≈ 0.315, ERA ≈ 4.20
    """
    LEAGUE_RPG = 4.5
    LEAGUE_XWOBA = 0.315
    LEAGUE_ERA = 4.20

    home_rs = data.get("home_recent_rs", 4.5)
    home_ra = data.get("home_recent_ra", 4.5)
    away_rs = data.get("away_recent_rs", 4.5)
    away_ra = data.get("away_recent_ra", 4.5)
    pf = data.get("park_factor", 100)

    # Log5 勝率（用 Pythagorean）
    home_pct = pythagorean_runs(home_rs, home_ra)
    away_pct = pythagorean_runs(away_rs, away_ra)
    log5_pct = log5(home_pct, away_pct)
    log5_pct = min(log5_pct + 0.03, 0.95)  # 主場優勢 +3%

    # 期望得分（納入打線 xwOBA + 對方投手 FIP）
    home_xwoba = data.get("home_batting_xwoba", LEAGUE_XWOBA)
    away_xwoba = data.get("away_batting_xwoba", LEAGUE_XWOBA)
    # 用 FIP 代替 ERA（FIP 更能反映投手真實能力）
    away_pitcher_fip = data.get("away_starter_fip", LEAGUE_ERA)
    home_pitcher_fip = data.get("home_starter_fip", LEAGUE_ERA)

    pf_mult = pf / 100
    home_score = round(
        LEAGUE_RPG * (home_xwoba / LEAGUE_XWOBA) * (away_pitcher_fip / LEAGUE_ERA) * pf_mult, 1
    )
    away_score = round(
        LEAGUE_RPG * (away_xwoba / LEAGUE_XWOBA) * (home_pitcher_fip / LEAGUE_ERA) * pf_mult, 1
    )

    return {
        "log5_pct": round(log5_pct * 100, 1),
        "pythag_home_pct": round(home_pct * 100, 1),
        "pythag_away_pct": round(away_pct * 100, 1),
        "home_score": home_score,
        "away_score": away_score,
        "total": round(home_score + away_score, 1),
    }


# ET timezone（MLB 球季 EDT = UTC-4）
_ET_TZ = timezone(timedelta(hours=-4))
_TW_TZ = timezone(timedelta(hours=+8))
_ANALYSIS_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")


def _extract_game_date_tw(args, meta: dict) -> str | None:
    """Derive game_date_tw from --game-data analysis-data path segment, or UTC→ET+1 fallback.

    User 規則（spec 2026-04-29）：folder = US gameday slate 用 TW notation = ET_date + 1。
    early ET day game（ET 4/29 11:00 = TW 4/29 23:00）也歸 ET_date+1（4/30），
    所以 fallback 用 `astimezone(ET).date() + 1` 而非直接 `astimezone(TW)`。
    """
    game_date_tw = None
    if getattr(args, "game_data", None):
        for part in os.path.normpath(args.game_data).split(os.sep):
            if _ANALYSIS_DATE_RE.match(part):
                game_date_tw = part
                break
    if not game_date_tw:
        game_date_iso = meta.get("game_date") or ""
        if game_date_iso:
            try:
                utc_dt = datetime.fromisoformat(game_date_iso.replace("Z", "+00:00"))
                et_date = utc_dt.astimezone(_ET_TZ).date()
                game_date_tw = (et_date + timedelta(days=1)).strftime("%Y-%m-%d")
            except ValueError:
                pass
    return game_date_tw


def main():
    parser = argparse.ArgumentParser(description="Predict MLB game outcome")
    parser.add_argument("--game-data", help="Path to JSON with merged game data")
    parser.add_argument("--save", action="store_true",
                        help="Save prediction.json into the game's analysis-data folder (parent dir of --game-data)")
    parser.add_argument("-o", "--output",
                        help="Explicit output path for prediction.json (overrides --save auto-path)")
    parser.add_argument("--test", action="store_true")
    # Post-analysis manual fields
    parser.add_argument("--adjusted-home", type=float, help="Adjusted home runs scored")
    parser.add_argument("--adjusted-away", type=float, help="Adjusted away runs scored")
    parser.add_argument("--ou-line", type=float, help="O/U line (effective, e.g. 9.75 for quarter-ball)")
    parser.add_argument("--ou-rec", choices=["OVER", "UNDER", "PASS"], help="O/U recommendation")
    parser.add_argument("--ml-rec", help="ML recommendation (team abbr or PASS)")
    parser.add_argument("--ml-stars", type=int, choices=[0, 1, 2, 3, 4, 5], help="ML star rating")
    parser.add_argument("--run-line", help="Run line value, e.g. '-1.5' or '1+50'")
    parser.add_argument("--ou-stars", type=int, choices=[0, 1, 2, 3, 4, 5], help="O/U star rating")
    # --run-line-rec / --run-line-stars: 廢除於 Plan B 2026-04-22（W1）
    # RL 全走 apply_rl_guardrail 自動 override（RL-1b gate）
    parser.add_argument("--signal-adjustments", type=json.loads,
                        help='Signal adjustments JSON, e.g. \'{"puk_il":0.3}\'')
    parser.add_argument("--tags", help="Comma-separated tags, e.g. divergent,early-season")
    parser.add_argument("--temperature", type=float, help="Temperature (F)")
    parser.add_argument("--wind-mph", type=float, help="Wind speed (mph)")
    parser.add_argument("--wind-direction", help="Wind direction")
    parser.add_argument("--umpire", help="Home plate umpire name")
    parser.add_argument("--umpire-ou-rate", type=float, help="Umpire career Over pct")
    # Plan B 2026-04-22 edge-case 跳過旗（測試 / 非正式流程用）
    parser.add_argument("--skip-phase3-check", action="store_true",
                        help="Bypass phase3_summary.md section check (Plan B)")
    args = parser.parse_args()

    # P5：--ou-rec OVER/UNDER 必須同時提供 --ou-stars（避免 silent fallback 為 PASS）
    if args.ou_rec in ("OVER", "UNDER") and args.ou_stars is None:
        print(
            "⛔ --ou-rec=OVER/UNDER 必須同時提供 --ou-stars (0-5)\n"
            "  例：--ou-rec OVER --ou-stars 3",
            file=sys.stderr,
        )
        sys.exit(6)
    # PASS 時 ou_stars 預設 0（保留既有行為）
    if args.ou_rec == "PASS" and args.ou_stars is None:
        args.ou_stars = 0

    if args.test:
        print(json.dumps({"test": "OK", "message": "predict test mode"}))
        return

    # W2: ml_rec schema validation（Plan B 2026-04-22 §4.2）
    validate_ml_rec(args.ml_rec, set(TEAM_ABBREV.values()))

    # W3: signal_adjustments allowlist warning（Plan B 2026-04-22 §4.2）
    warn_unknown_signal_keys(args.signal_adjustments)

    if not args.game_data:
        parser.error("--game-data is required unless --test is specified")

    # W4: game_data path regex（Plan B 2026-04-22 §4.2）
    validate_game_data_path(args.game_data)

    with open(args.game_data, "r") as f:
        data = json.load(f)

    # 公式預測
    formula_pred = predict_with_formula(data)

    # 信號計分表
    signal_table = compute_signal_table(data)

    # 用 30 場 Formula 做交叉驗證（比 10 場更穩定）
    home_season_games = data.get("home_season_games", 0)
    away_season_games = data.get("away_season_games", 0)
    min_season_games = min(home_season_games, away_season_games)

    formula_30_pred = None
    if min_season_games >= 30:
        # 用 30 場數據計算 Formula
        data_30 = {
            "home_recent_rs": data.get("home_recent_30_rs", data.get("home_recent_rs", 4.5)),
            "home_recent_ra": data.get("home_recent_30_ra", data.get("home_recent_ra", 4.5)),
            "away_recent_rs": data.get("away_recent_30_rs", data.get("away_recent_rs", 4.5)),
            "away_recent_ra": data.get("away_recent_30_ra", data.get("away_recent_ra", 4.5)),
            "park_factor": data.get("park_factor", 100),
        }
        formula_30_pred = predict_with_formula(data_30)

    # 最終推薦
    final_pct = formula_pred["log5_pct"]
    final_home = formula_pred["home_score"]
    final_away = formula_pred["away_score"]

    # 計算 adjusted 比分（用於決定最終方向）
    adj_home = args.adjusted_home if args.adjusted_home is not None else final_home
    adj_away = args.adjusted_away if args.adjusted_away is not None else final_away
    adj_total = round(adj_home + adj_away, 1)

    # 決定最終方向：adjusted 比分優先於 formula 勝率
    has_adjusted = args.adjusted_home is not None or args.adjusted_away is not None
    if has_adjusted and (adj_home > adj_away) != (final_pct > 50):
        adjusted_winner = "HOME" if adj_home > adj_away else "AWAY"
        display_home_pct = round(formula_pred["log5_pct"], 1)
    else:
        adjusted_winner = "HOME" if final_pct > 50 else "AWAY"
        display_home_pct = round(final_pct, 1)

    result = {
        "formula_prediction": formula_pred,
        "signal_table": signal_table,
        "final": {
            "recommended_winner": adjusted_winner,
            "home_win_pct": display_home_pct,
            "confidence": "MEDIUM",
            "predicted_home_score": adj_home,
            "predicted_away_score": adj_away,
            "predicted_total": adj_total,
            "signal_run_adjustment": signal_table["total_run_adjustment"],
            "over_under_lean": "OVER" if signal_table["total_run_adjustment"] > 0 else ("UNDER" if signal_table["total_run_adjustment"] < 0 else "NEUTRAL"),
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 自動存檔到 predictions.jsonl
    if args.save:
        from datetime import datetime as _dt
        meta = data.get("_meta", {})
        home_team = meta.get("home_team") or "HOME"
        away_team = meta.get("away_team") or "AWAY"

        record_date = _extract_game_date_tw(args, meta) or _dt.now(_TW_TZ).strftime("%Y-%m-%d")

        # phase3_summary.md 必須存在（structural 完整性由 phase3_skeleton.md 預填保證）
        if not args.skip_phase3_check:
            game_dir_p3 = Path(args.game_data).parent
            phase3_path = game_dir_p3 / "phase3_summary.md"
            if not phase3_path.exists():
                sys.exit(
                    f"⛔ {phase3_path} 不存在 — Phase 3 結論未存檔\n"
                    f"  請先在 phase3_skeleton.md 補結論並另存為 phase3_summary.md；"
                    f"或加 --skip-phase3-check 跳過（測試用）。"
                )

        # === 護欄機制：星級自動上限 ===
        original_ml_stars = args.ml_stars
        ml_stars_cap = 5  # 預設無上限
        force_ml_pass = False  # 強制 ml_rec = PASS 旗標
        cap_reasons = []

        # 規則 4：formula 勝率 50-55% → 上限 2
        rec_side_pct = final_pct if result["final"]["recommended_winner"] == "HOME" else (100 - final_pct)
        if 50 <= rec_side_pct < 55:
            ml_stars_cap = min(ml_stars_cap, 2)
            cap_reasons.append(f"formula 勝率 {rec_side_pct:.1f}%（50-55%）上限 2")

        # F2: 規則 5：方向矛盾（ml_rec 與 predicted_winner 不一致）→ 上限 2
        direction_override = False
        if args.ml_rec and args.ml_rec != "PASS":
            predicted_winner = result["final"]["recommended_winner"]
            home_abbr = TEAM_ABBREV.get(home_team, "")
            away_abbr = TEAM_ABBREV.get(away_team, "")
            rec_is_home = args.ml_rec == home_abbr
            rec_is_away = args.ml_rec == away_abbr
            if (predicted_winner == "HOME" and rec_is_away) or (predicted_winner == "AWAY" and rec_is_home):
                direction_override = True
                ml_stars_cap = min(ml_stars_cap, 2)
                cap_reasons.append(f"ml_rec={args.ml_rec} 與 formula predicted_winner={predicted_winner}({home_abbr if predicted_winner == 'HOME' else away_abbr}) 方向矛盾，上限 2")

        # Y-new-2（Plan B §4.4）：近身戰（|adj 比分差| < 0.5）上限 1 星（cumulative #3）
        ml_stars_cap, y_new_2_reason = apply_close_game_cap(adj_home, adj_away, ml_stars_cap)
        if y_new_2_reason:
            cap_reasons.append(y_new_2_reason)

        # Y-new-3（Plan B §4.4）：user-supplied 'divergent' tag 上限 2 星（cumulative #4）
        user_tags_raw = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
        ml_stars_cap, y_new_3_reason = apply_divergent_user_tag_cap(user_tags_raw, ml_stars_cap)
        if y_new_3_reason:
            cap_reasons.append(y_new_3_reason)

        # 套用星級上限
        final_ml_stars = args.ml_stars
        if final_ml_stars is not None and final_ml_stars > ml_stars_cap:
            print(f"⚠️ ml_stars 從 {final_ml_stars} 降為 {ml_stars_cap}（原因：{'; '.join(cap_reasons)}）", file=sys.stderr)
            final_ml_stars = ml_stars_cap

        # 套用 ml_rec 強制 PASS
        final_ml_rec = args.ml_rec
        if force_ml_pass and final_ml_rec and final_ml_rec != "PASS":
            print(f"⚠️ ml_rec 從 {final_ml_rec} 改為 PASS（原因：{'; '.join(cap_reasons)}）", file=sys.stderr)
            final_ml_rec = "PASS"

        # === 趨勢標籤 ===
        trend_tags = compute_trend_tags(data)

        # 合併使用者 tags + 趨勢 tags + direction-override
        user_tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
        if direction_override and "direction-override" not in user_tags:
            user_tags.append("direction-override")
        all_tags = list(dict.fromkeys(user_tags + trend_tags))  # 去重保序

        # Y-new-1 audit tag（Plan B §4.4）：主場 2 星（cumulative #1，觀察期先 tag 不 cap）
        if should_add_home_2star_tag(
            result["final"]["recommended_winner"], final_ml_stars, final_ml_rec
        ) and "home-2star-risk" not in all_tags:
            all_tags.append("home-2star-risk")

        # === 護欄機制：O/U 自動 PASS ===
        final_ou_rec = args.ou_rec if args.ou_rec is not None else result["final"]["over_under_lean"]
        if final_ou_rec == "NEUTRAL":
            final_ou_rec = "PASS"
        final_ou_stars = args.ou_stars

        # OU-1: 差距 < 1.5 run → PASS（SD ≈ 4.5，< 1.5 在噪音範圍）
        if final_ou_rec != "PASS" and args.ou_line is not None:
            ou_gap = abs(adj_total - args.ou_line)
            if ou_gap < 1.5:
                print(f"⚠️ O/U 從 {final_ou_rec} 改為 PASS（差距 {ou_gap:.1f} < 1.5 run）", file=sys.stderr)
                final_ou_rec = "PASS"
                final_ou_stars = 0

        # OU-2: 方向與調整後比分矛盾 → PASS
        if final_ou_rec == "OVER" and args.ou_line is not None and adj_total <= args.ou_line:
            print(f"⚠️ O/U 從 OVER 改為 PASS（調整後總分 {adj_total} ≤ 線 {args.ou_line}）", file=sys.stderr)
            final_ou_rec = "PASS"
            final_ou_stars = 0
        elif final_ou_rec == "UNDER" and args.ou_line is not None and adj_total >= args.ou_line:
            print(f"⚠️ O/U 從 UNDER 改為 PASS（調整後總分 {adj_total} ≥ 線 {args.ou_line}）", file=sys.stderr)
            final_ou_rec = "PASS"
            final_ou_stars = 0

        # Invariant: --ou-stars 必填（OVER/UNDER）由 main() 開頭的 P5 validation 保證；PASS 時預設 0
        assert final_ou_stars is not None, "final_ou_stars unexpectedly None — P5 guard bypassed"

        # === 護欄機制：讓分盤（RL-1b 自動推薦 + RL-2 stars required）===
        # 對稱 ML/OU（spec 2026-04-21）：不再讀 confidence，只看 diff + tags + user input
        # 邏輯封裝於 apply_rl_guardrail；rl_override 產出 audit dict 寫入 saved record
        final_rl_rec, final_rl_stars, rl_override = apply_rl_guardrail(
            adj_home=adj_home,
            adj_away=adj_away,
            trend_tags=trend_tags,
            predicted_winner=result["final"]["recommended_winner"],
            home_team=home_team,
            away_team=away_team,
        )

        record = {
            "date": record_date,
            "game_time": meta.get("game_date"),
            "game": f"{away_team} vs {home_team}",
            "home_team": home_team,
            "away_team": away_team,
            "home_sp": meta.get("home_sp"),
            "away_sp": meta.get("away_sp"),
            "home_sp_starts": meta.get("home_sp_starts"),
            "away_sp_starts": meta.get("away_sp_starts"),
            "venue": meta.get("venue"),
            "game_pk": meta.get("game_pk"),
            "predicted_winner": result["final"]["recommended_winner"],
            "predicted_home_pct": result["final"]["home_win_pct"],
            "predicted_home_score": adj_home,
            "predicted_away_score": adj_away,
            "predicted_total": adj_total,
            "formula_home_score": final_home,
            "formula_away_score": final_away,
            "pre_signal_total": formula_pred["total"] if formula_pred else None,
            "adjusted_total": adj_total,
            "signal_adjustments": args.signal_adjustments if args.signal_adjustments is not None else {},
            "ou_line": args.ou_line,
            "ou_rec": final_ou_rec,
            "ou_stars": final_ou_stars,
            "run_line_rec": final_rl_rec,
            "run_line": args.run_line,
            "run_line_stars": final_rl_stars,
            "rl_override": rl_override,
            "ml_rec": final_ml_rec,
            "ml_stars": final_ml_stars,
            "original_ml_stars": original_ml_stars,
            "confidence": result["final"]["confidence"],
            "tags": all_tags,
            "park_factor": data.get("park_factor"),
            "temperature_f": args.temperature,
            "wind_mph": args.wind_mph,
            "wind_direction": args.wind_direction,
            "umpire_name": args.umpire,
            "umpire_ou_rate": args.umpire_ou_rate,
            "actual_winner": None,
            "actual_home_score": None,
            "actual_away_score": None,
            "actual_total": None,
            "verified": False,
        }
        # 寫入 per-game prediction.json（放在 --game-data 所在資料夾）
        if args.output:
            prediction_path = args.output
        else:
            game_dir = os.path.dirname(os.path.abspath(args.game_data))
            prediction_path = os.path.join(game_dir, "prediction.json")

        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
        with open(prediction_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"Saved to {prediction_path}", file=sys.stderr)

        # 額外輸出 prediction_summary.md（同目錄）
        summary_path = Path(prediction_path).parent / "prediction_summary.md"
        try:
            summary_md = format_prediction_summary_md(record, signal_table, cap_reasons)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_md)
            print(f"Saved summary to {summary_path}", file=sys.stderr)
        except ValueError as e:
            print(f"Skipped summary (data incomplete): {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
