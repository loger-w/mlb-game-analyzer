#!/usr/bin/env python3
"""MLB Pitcher Stats — Phase 2 投手進階數據（MLB Stats API + Statcast）"""

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime

import requests


@contextlib.contextmanager
def _redirect_pybaseball_stdout():
    """將 pybaseball 內部的 print() 訊息導向 stderr，保持 stdout 純淨"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield
    output = buf.getvalue()
    if output:
        sys.stderr.write(output)

try:
    from pybaseball import (
        playerid_lookup,
        statcast_pitcher,
        statcast_pitcher_expected_stats,
        statcast_pitcher_exitvelo_barrels,
    )
except ImportError:
    print(json.dumps({"error": "pybaseball not installed. Run: pip install pybaseball"}))
    sys.exit(1)


MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# FIP constant ≈ 3.10 (varies by year, this is a reasonable default)
FIP_CONSTANT = 3.10
# League average HR/FB rate for xFIP calculation
LEAGUE_HR_FB = 0.10

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


def lookup_pitcher_id(name: str) -> int | None:
    """用 pybaseball 查詢球員 MLBAM ID"""
    parts = name.strip().split()
    if len(parts) < 2:
        return None
    last = parts[-1]
    first = parts[0]
    try:
        with _redirect_pybaseball_stdout():
            result = playerid_lookup(last, first)
        if result.empty:
            return None
        return int(result.iloc[0]["key_mlbam"])
    except Exception:
        return None


def fetch_player_info(mlbam_id: int) -> dict:
    """從 MLB Stats API 取得球員基本資訊（年齡、投球手等）"""
    try:
        resp = requests.get(f"{MLB_API_BASE}/people/{mlbam_id}", timeout=10)
        resp.raise_for_status()
        person = resp.json()["people"][0]
        return {
            "age": person.get("currentAge"),
            "birth_date": person.get("birthDate"),
            "pitch_hand": person.get("pitchHand", {}).get("code"),
            "bat_side": person.get("batSide", {}).get("code"),
        }
    except Exception as e:
        return {"error": str(e)}


def parse_ip(ip_str: str) -> float:
    """將 MLB API 的 IP 字串轉為浮點數（如 '8.1' → 8.333）"""
    try:
        parts = str(ip_str).split(".")
        innings = int(parts[0])
        thirds = int(parts[1]) if len(parts) > 1 else 0
        return innings + thirds / 3.0
    except (ValueError, IndexError):
        return 0.0


def calc_fip(hr: int, bb: int, k: int, ip: float) -> float | None:
    """計算 FIP = (13×HR + 3×BB - 2×K) / IP + constant"""
    if ip <= 0:
        return None
    return round((13 * hr + 3 * bb - 2 * k) / ip + FIP_CONSTANT, 2)


def calc_xfip(air_outs: int, bb: int, k: int, ip: float) -> float | None:
    """計算 xFIP（用聯盟平均 HR/FB 替代實際 HR）"""
    if ip <= 0:
        return None
    expected_hr = air_outs * LEAGUE_HR_FB
    return round((13 * expected_hr + 3 * bb - 2 * k) / ip + FIP_CONSTANT, 2)


def fetch_mlb_api_stats(mlbam_id: int, year: int) -> dict:
    """從 MLB Stats API 取得本季投球數據並計算進階指標"""
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{mlbam_id}/stats",
            params={"stats": "season", "group": "pitching", "season": year},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        stats_list = data.get("stats", [])
        if not stats_list or not stats_list[0].get("splits"):
            return {"error": f"No {year} pitching stats found"}

        s = stats_list[0]["splits"][0]["stat"]

        era = float(s.get("era", 0))
        whip = float(s.get("whip", 0))
        ip = parse_ip(s.get("inningsPitched", "0"))
        k = int(s.get("strikeOuts", 0))
        bb = int(s.get("baseOnBalls", 0))
        hr = int(s.get("homeRuns", 0))
        bf = int(s.get("battersFaced", 0))
        go = int(s.get("groundOuts", 0))
        ao = int(s.get("airOuts", 0))
        g = int(s.get("gamesPlayed", 0))
        gs = int(s.get("gamesStarted", 0))

        # 自行計算進階指標
        k_pct = round(k / bf * 100, 1) if bf > 0 else 0.0
        bb_pct = round(bb / bf * 100, 1) if bf > 0 else 0.0
        k_bb_pct = round(k_pct - bb_pct, 1)
        gb_pct = round(go / (go + ao) * 100, 1) if (go + ao) > 0 else 0.0
        hr_per_9 = round(hr / ip * 9, 2) if ip > 0 else 0.0
        fip = calc_fip(hr, bb, k, ip)
        xfip = calc_xfip(ao, bb, k, ip)

        return {
            "era": era,
            "whip": whip,
            "k_pct": k_pct,
            "bb_pct": bb_pct,
            "k_bb_pct": k_bb_pct,
            "fip": fip,
            "xfip": xfip,
            "hr_per_9": hr_per_9,
            "gb_pct": gb_pct,
            "ip": ip,
            "games": g,
            "gs": gs,
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_statcast_expected(mlbam_id: int, year: int) -> dict:
    """從 Statcast expected stats 取得 xERA, xwOBA 等"""
    try:
        df = statcast_pitcher_expected_stats(year, minPA=1)
        if df.empty:
            return {"error": "No Statcast expected stats data"}

        row = df[df["player_id"].astype(str) == str(mlbam_id)]
        if row.empty:
            return {"error": f"Player {mlbam_id} not found in expected stats"}

        r = row.iloc[0]
        return {
            "xera": round(float(r.get("xera", 0)), 2) if r.get("xera") is not None else None,
            "xwoba": round(float(r.get("est_woba", 0)), 3) if r.get("est_woba") is not None else None,
            "xba": round(float(r.get("est_ba", 0)), 3) if r.get("est_ba") is not None else None,
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_statcast_stats(mlbam_id: int, year: int) -> dict:
    """從 Statcast 取投手物理數據（球速、球種等）"""
    try:
        start = f"{year}-03-20"
        end = f"{year}-11-05"
        df = statcast_pitcher(start, end, mlbam_id)
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
            "pitch_types": pitch_types,
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_statcast_barrels(mlbam_id: int, year: int) -> dict:
    """從 Statcast exit velo/barrels leaderboard 取 Barrel% 等"""
    try:
        df = statcast_pitcher_exitvelo_barrels(year, minBBE=1)
        if df.empty:
            return {"error": "No barrel data"}

        row = df[df["player_id"].astype(str) == str(mlbam_id)]
        if row.empty:
            return {"error": f"Player {mlbam_id} not found in barrel data"}

        r = row.iloc[0]
        return {
            "barrel_pct": round(float(r.get("brl_percent", 0)), 1) if r.get("brl_percent") is not None else None,
            "ev95percent": round(float(r.get("ev95percent", 0)), 1) if r.get("ev95percent") is not None else None,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Fetch pitcher advanced stats")
    parser.add_argument("--name", required=True, help="Pitcher full name (e.g. 'Gerrit Cole')")
    parser.add_argument("--year", type=int, default=datetime.now().year, help="Season year")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "pitcher_stats test mode"}, indent=2))
        return

    # 1. 查詢 MLBAM ID
    pitcher_id = lookup_pitcher_id(args.name)
    if not pitcher_id:
        print(json.dumps({"error": f"Could not find MLBAM ID for {args.name}"}, indent=2, ensure_ascii=False))
        sys.exit(1)

    # 2. 球員基本資訊（年齡、投球手）
    info = fetch_player_info(pitcher_id)
    age = info.get("age")
    age_assessment = get_age_assessment(age) if age else None

    # 3. MLB Stats API 本季數據
    season = fetch_mlb_api_stats(pitcher_id, args.year)

    # 4. 投手等級
    tier = get_tier(season) if "error" not in season else "Unknown"

    # 5. Statcast expected stats (xERA, xwOBA)
    expected = fetch_statcast_expected(pitcher_id, args.year)

    # 6. Statcast 物理數據（球速、球種）
    statcast = fetch_statcast_stats(pitcher_id, args.year)

    # 7. Statcast barrels
    barrels = fetch_statcast_barrels(pitcher_id, args.year)
    if "error" not in barrels and "error" not in statcast:
        statcast["barrel_pct"] = barrels.get("barrel_pct")
        statcast["ev95percent"] = barrels.get("ev95percent")

    output = {
        "name": args.name,
        "mlbam_id": pitcher_id,
        "age": age,
        "birth_date": info.get("birth_date"),
        "pitch_hand": info.get("pitch_hand"),
        "age_assessment": age_assessment,
        "tier": tier,
        "season": season,
        "expected": expected,
        "statcast": statcast,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
