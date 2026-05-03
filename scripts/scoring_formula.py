#!/usr/bin/env python3
"""比分公式 — Path B baseline，簡單期望得分（formula-as-guardrail）。

被 summary_renderer 用來產出 `## 修正後預期得分` 段的 base 列。
複雜建模刻意外置給 AI 在 summary `+ 信號` 欄做 magnitude judgment
（見 reference/matchup-factors.md §量級錨點）。
"""


def predict_with_formula(data: dict) -> dict:
    """Path B baseline：簡單期望得分公式，當 AI judgment 的 sanity-check rail。

    E[R] = 聯盟平均得分 × (打線 xwOBA / 聯盟 xwOBA) × (對方投手 FIP / 聯盟 ERA) × (PF / 100)

    聯盟平均（2024-2025 基準）：R/G ≈ 4.5, xwOBA ≈ 0.315, ERA ≈ 4.20
    """
    LEAGUE_RPG = 4.5
    LEAGUE_XWOBA = 0.315
    LEAGUE_ERA = 4.20

    pf = data.get("park_factor", 100)
    home_xwoba = data.get("home_batting_xwoba", LEAGUE_XWOBA)
    away_xwoba = data.get("away_batting_xwoba", LEAGUE_XWOBA)
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
        "home_score": home_score,
        "away_score": away_score,
        "total": round(home_score + away_score, 1),
    }
