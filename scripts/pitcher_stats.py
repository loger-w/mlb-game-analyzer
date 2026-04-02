#!/usr/bin/env python3
"""MLB Pitcher Stats — Phase 2 投手進階數據（pybaseball）"""

import argparse
import json
import sys
from datetime import datetime

import pandas as pd

try:
    from pybaseball import playerid_lookup, statcast_pitcher, pitching_stats
except ImportError:
    print(json.dumps({"error": "pybaseball not installed. Run: pip install pybaseball"}))
    sys.exit(1)


AGE_ASSESSMENT_PITCHER = {
    (0, 24): "📈 成長期",
    (25, 29): "⚡ 巔峰期",
    (30, 33): "📉 初期退化",
    (34, 36): "📉📉 明顯退化",
    (37, 99): "📉📉📉 快速退化",
}

TIER_THRESHOLDS = [
    ("🔴 Elite Ace", lambda s: s.get("era", 99) < 2.50 and s.get("k_bb_pct", 0) > 20),
    ("🟠 Strong Ace", lambda s: s.get("era", 99) < 3.20),
    ("🟡 Solid Starter", lambda s: s.get("era", 99) < 4.20),
    ("🟢 Back-end Starter", lambda s: s.get("era", 99) < 5.00),
    ("⚪ Below Average", lambda s: True),
]


def get_age_assessment(age: int) -> str:
    for (lo, hi), label in AGE_ASSESSMENT_PITCHER.items():
        if lo <= age <= hi:
            return label
    return "Unknown"


def get_tier(season_stats: dict) -> str:
    for tier_name, check_fn in TIER_THRESHOLDS:
        if check_fn(season_stats):
            return tier_name
    return "⚪ Below Average"


def normalize_pct(val: float) -> float:
    """Normalize percentage: if < 1 assume it's a decimal (0.25 -> 25.0)"""
    if val is None:
        return 0.0
    val = float(val)
    if 0 < val < 1:
        return round(val * 100, 1)
    return round(val, 1)


def lookup_pitcher_id(name: str) -> int | None:
    """用 pybaseball 查詢球員 MLBAM ID"""
    parts = name.strip().split()
    if len(parts) < 2:
        return None
    last = parts[-1]
    first = parts[0]
    try:
        result = playerid_lookup(last, first)
        if result.empty:
            return None
        return int(result.iloc[0]["key_mlbam"])
    except Exception:
        return None


def fetch_fangraphs_stats(year: int, name: str) -> dict:
    """從 pybaseball 取 FanGraphs 投手數據"""
    try:
        df = pitching_stats(year, year, qual=1)
        # 模糊匹配名字
        mask = df["Name"].str.contains(name.split()[-1], case=False, na=False)
        if mask.sum() == 0:
            return {"error": f"No pitching data found for {name} in {year}"}

        # 如果多筆，嘗試更精確匹配
        if mask.sum() > 1:
            first_name = name.split()[0]
            refined = df[mask & df["Name"].str.contains(first_name, case=False, na=False)]
            if not refined.empty:
                row = refined.iloc[0]
            else:
                row = df[mask].iloc[0]
        else:
            row = df[mask].iloc[0]

        k_pct = normalize_pct(row.get("K%", 0))
        bb_pct = normalize_pct(row.get("BB%", 0))

        return {
            "era": round(float(row.get("ERA", 0)), 2),
            "fip": round(float(row.get("FIP", 0)), 2),
            "xfip": round(float(row.get("xFIP", 0)), 2),
            "siera": round(float(row.get("SIERA", 0)), 2) if "SIERA" in row.index else None,
            "k_pct": k_pct,
            "bb_pct": bb_pct,
            "k_bb_pct": round(k_pct - bb_pct, 1),
            "whip": round(float(row.get("WHIP", 0)), 2),
            "hr_per_9": round(float(row.get("HR/9", 0)), 2),
            "gb_pct": normalize_pct(row.get("GB%", 0)),
            "ip": float(row.get("IP", 0)),
            "games": int(row.get("G", 0)),
            "gs": int(row.get("GS", 0)),
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_statcast_stats(pitcher_id: int, year: int) -> dict:
    """從 Statcast 取投手物理數據"""
    try:
        start = f"{year}-03-20"
        end = f"{year}-11-05"
        df = statcast_pitcher(start, end, pitcher_id)
        if df.empty:
            return {"error": "No Statcast data found"}

        # 均速 / 最高速
        avg_velo = None
        max_velo = None
        if "release_speed" in df.columns:
            velo_data = df["release_speed"].dropna()
            if not velo_data.empty:
                avg_velo = round(float(velo_data.mean()), 1)
                max_velo = round(float(velo_data.max()), 1)

        # 被擊球品質
        hard_hit_pct = None
        if "launch_speed" in df.columns:
            batted = df[df["launch_speed"].notna()]
            if not batted.empty:
                hard_hit_pct = round(float((batted["launch_speed"] >= 95).mean() * 100), 1)

        barrel_pct = None
        if "barrel" in df.columns and not df["barrel"].isna().all():
            barrel_pct = round(float(df["barrel"].mean() * 100), 1)

        # 球種組合
        pitch_types = {}
        if "pitch_type" in df.columns:
            counts = df["pitch_type"].dropna().value_counts(normalize=True)
            for pt, pct in counts.head(5).items():
                pitch_types[str(pt)] = round(float(pct * 100), 1)

        return {
            "avg_velo": avg_velo,
            "max_velo": max_velo,
            "hard_hit_pct": hard_hit_pct,
            "barrel_pct": barrel_pct,
            "pitch_types": pitch_types,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Fetch pitcher advanced stats")
    parser.add_argument("--name", required=True, help="Pitcher full name (e.g. 'Gerrit Cole')")
    parser.add_argument("--year", type=int, default=datetime.now().year, help="Season year")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "pitcher_stats test mode"}, indent=2))
        return

    # FanGraphs 數據
    season = fetch_fangraphs_stats(args.year, args.name)

    # Statcast 數據
    pitcher_id = lookup_pitcher_id(args.name)
    statcast = {}
    if pitcher_id:
        statcast = fetch_statcast_stats(pitcher_id, args.year)
    else:
        statcast = {"error": f"Could not find MLBAM ID for {args.name}"}

    # 投手等級
    tier = get_tier(season) if "error" not in season else "Unknown"

    # 年齡
    age = None
    age_assessment = None
    try:
        parts = args.name.strip().split()
        result = playerid_lookup(parts[-1], parts[0])
        if not result.empty:
            birth_year = int(result.iloc[0].get("mlb_played_first", args.year - 28))
            age = args.year - birth_year
            age_assessment = get_age_assessment(age)
    except Exception:
        pass

    output = {
        "name": args.name,
        "age": age,
        "age_assessment": age_assessment,
        "tier": tier,
        "season": season,
        "statcast": statcast,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
