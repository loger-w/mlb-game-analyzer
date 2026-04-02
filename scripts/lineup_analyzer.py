#!/usr/bin/env python3
"""MLB Lineup Analyzer — Phase 2 打線分析 + Platoon + 大小分傾向"""

import argparse
import json
import sys
from datetime import datetime

try:
    from pybaseball import batting_stats
except ImportError:
    print(json.dumps({"error": "pybaseball not installed. Run: pip install pybaseball"}))
    sys.exit(1)


TIER_MAP = [
    ("🔴 Elite", lambda wrc: wrc >= 115),
    ("🟠 Strong", lambda wrc: wrc >= 105),
    ("🟡 Average", lambda wrc: wrc >= 95),
    ("🟢 Weak", lambda wrc: True),
]

# pybaseball Team column values → common abbreviations
TEAM_ALIASES = {
    "NYY": ["NYY", "Yankees"],
    "NYM": ["NYM", "Mets"],
    "BOS": ["BOS", "Red Sox"],
    "LAD": ["LAD", "Dodgers"],
    "LAA": ["LAA", "Angels"],
    "HOU": ["HOU", "Astros"],
    "ATL": ["ATL", "Braves"],
    "PHI": ["PHI", "Phillies"],
    "SDP": ["SDP", "SD", "Padres"],
    "SFG": ["SFG", "SF", "Giants"],
    "CHC": ["CHC", "Cubs"],
    "CHW": ["CHW", "CWS", "White Sox"],
    "CIN": ["CIN", "Reds"],
    "STL": ["STL", "Cardinals"],
    "MIL": ["MIL", "Brewers"],
    "PIT": ["PIT", "Pirates"],
    "ARI": ["ARI", "AZ", "Diamondbacks"],
    "COL": ["COL", "Rockies"],
    "BAL": ["BAL", "Orioles"],
    "TBR": ["TBR", "TB", "Rays"],
    "TOR": ["TOR", "Blue Jays"],
    "MIN": ["MIN", "Twins"],
    "KCR": ["KCR", "KC", "Royals"],
    "DET": ["DET", "Tigers"],
    "CLE": ["CLE", "Guardians"],
    "SEA": ["SEA", "Mariners"],
    "OAK": ["OAK", "Athletics"],
    "TEX": ["TEX", "Rangers"],
    "MIA": ["MIA", "Marlins"],
    "WSN": ["WSN", "WSH", "Nationals"],
}


def normalize_pct(val: float) -> float:
    """Normalize percentage: if < 1 assume decimal (0.25 -> 25.0)"""
    if val is None:
        return 0.0
    val = float(val)
    if 0 < val < 1:
        return round(val * 100, 1)
    return round(val, 1)


def find_team_in_df(df, team_input: str):
    """在 DataFrame 中找到指定球隊的球員"""
    team_upper = team_input.upper()

    # 直接匹配 Team column
    mask = df["Team"].str.upper() == team_upper
    if mask.sum() > 0:
        return df[mask]

    # 嘗試別名
    for pybb_abbrev, aliases in TEAM_ALIASES.items():
        if team_upper in [a.upper() for a in aliases]:
            mask = df["Team"].str.upper() == pybb_abbrev.upper()
            if mask.sum() > 0:
                return df[mask]

    # 模糊匹配
    mask = df["Team"].str.contains(team_input, case=False, na=False)
    if mask.sum() > 0:
        return df[mask]

    return None


def fetch_team_batting_stats(team: str, year: int) -> dict:
    """取得球隊打線整體數據"""
    try:
        df = batting_stats(year, year, qual=50)

        team_df = find_team_in_df(df, team)
        if team_df is None or team_df.empty:
            return {"error": f"No batting data found for {team} in {year}"}

        # 取前 6 名（模擬 1-6 棒核心）
        team_df = team_df.sort_values("PA", ascending=False).head(6)

        lineup = []
        for _, row in team_df.iterrows():
            wrc_plus = float(row.get("wRC+", 100))
            babip = float(row.get("BABIP", 0.300))
            k_pct = normalize_pct(row.get("K%", 0))
            bb_pct = normalize_pct(row.get("BB%", 0))
            hard_hit = normalize_pct(row.get("Hard%", 0))

            lineup.append({
                "name": row.get("Name", "Unknown"),
                "pa": int(row.get("PA", 0)),
                "wrc_plus": round(wrc_plus, 0),
                "ops": round(float(row.get("OPS", 0)), 3),
                "obp": round(float(row.get("OBP", 0)), 3),
                "slg": round(float(row.get("SLG", 0)), 3),
                "iso": round(float(row.get("ISO", 0)), 3),
                "babip": round(babip, 3),
                "k_pct": k_pct,
                "bb_pct": bb_pct,
                "hard_hit_pct": hard_hit,
            })

        if not lineup:
            return {"error": f"No qualified batters found for {team}"}

        # 整體指標
        avg_wrc = sum(p["wrc_plus"] for p in lineup) / len(lineup)
        avg_babip = sum(p["babip"] for p in lineup) / len(lineup)
        avg_k_pct = sum(p["k_pct"] for p in lineup) / len(lineup)

        # 打線評級
        tier = "🟢 Weak"
        for tier_name, check_fn in TIER_MAP:
            if check_fn(avg_wrc):
                tier = tier_name
                break

        # 大小分傾向計算
        over_under_lean = 0
        if avg_babip <= 0.270:
            over_under_lean += 1  # BABIP 偏低 → 回歸預期得分上升
        if avg_babip >= 0.320:
            over_under_lean -= 1  # BABIP 偏高 → 回歸預期得分下降
        if avg_k_pct >= 25:
            over_under_lean -= 1  # 高三振率壓制得分
        if avg_wrc >= 110:
            over_under_lean += 1  # 火力強勁

        # 串聯分析
        chain = {}
        if len(lineup) >= 3:
            chain["obp_top3"] = round(sum(p["obp"] for p in lineup[:3]) / 3, 3)
        if len(lineup) >= 5:
            chain["slg_mid"] = round(sum(p["slg"] for p in lineup[3:5]) / 2, 3)

        return {
            "team": team,
            "tier": tier,
            "avg_wrc_plus": round(avg_wrc, 0),
            "avg_babip": round(avg_babip, 3),
            "avg_k_pct": round(avg_k_pct, 1),
            "over_under_lean": over_under_lean,
            "chain": chain,
            "lineup": lineup,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Analyze team lineup")
    parser.add_argument("--team", required=True, help="Team name or abbreviation")
    parser.add_argument("--year", type=int, default=datetime.now().year)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "lineup_analyzer test mode"}, indent=2))
        return

    result = fetch_team_batting_stats(args.team, args.year)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
