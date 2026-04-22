#!/usr/bin/env python3
"""MLB Lineup Analyzer — Phase 2 打線分析（MLB Stats API + Statcast）"""

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
        statcast_batter_expected_stats,
        statcast_batter_exitvelo_barrels,
    )
except ImportError:
    print(json.dumps({"error": "pybaseball not installed. Run: pip install pybaseball"}))
    sys.exit(1)


MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

TEAM_MAP = {
    # English abbreviations
    "NYY": 147, "NYM": 121, "BOS": 111, "LAD": 119, "LAA": 108,
    "HOU": 117, "ATL": 144, "PHI": 143, "SD": 135, "SF": 137,
    "CHC": 112, "CWS": 145, "CIN": 113, "STL": 138, "MIL": 158,
    "PIT": 134, "ARI": 109, "COL": 115, "BAL": 110, "TB": 139,
    "TOR": 141, "MIN": 142, "KC": 118, "DET": 116, "CLE": 114,
    "SEA": 136, "OAK": 133, "TEX": 140, "MIA": 146, "WSH": 120,
    # Chinese names
    "洋基": 147, "大都會": 121, "紅襪": 111, "道奇": 119, "天使": 108,
    "太空人": 117, "勇士": 144, "費城人": 143, "教士": 135, "巨人": 137,
    "小熊": 112, "白襪": 145, "紅人": 113, "紅雀": 138, "釀酒人": 158,
    "海盜": 134, "響尾蛇": 109, "落磯": 115, "金鶯": 110, "光芒": 139,
    "藍鳥": 141, "雙城": 142, "皇家": 118, "老虎": 116, "守護者": 114,
    "水手": 136, "運動家": 133, "遊騎兵": 140, "馬林魚": 146, "國民": 120,
}

FULL_NAMES = {
    "new york yankees": 147, "new york mets": 121, "boston red sox": 111,
    "los angeles dodgers": 119, "los angeles angels": 108, "houston astros": 117,
    "atlanta braves": 144, "philadelphia phillies": 143, "san diego padres": 135,
    "san francisco giants": 137, "chicago cubs": 112, "chicago white sox": 145,
    "cincinnati reds": 113, "st. louis cardinals": 138, "milwaukee brewers": 158,
    "pittsburgh pirates": 134, "arizona diamondbacks": 109, "colorado rockies": 115,
    "baltimore orioles": 110, "tampa bay rays": 139, "toronto blue jays": 141,
    "minnesota twins": 142, "kansas city royals": 118, "detroit tigers": 116,
    "cleveland guardians": 114, "seattle mariners": 136, "athletics": 133,
    "texas rangers": 140, "miami marlins": 146, "washington nationals": 120,
}

# xwOBA-based tier thresholds (replacing wRC+)
TIER_MAP = [
    ("🔴 Elite", lambda xwoba: xwoba >= 0.370),
    ("🟠 Strong", lambda xwoba: xwoba >= 0.340),
    ("🟡 Average", lambda xwoba: xwoba >= 0.310),
    ("🟢 Weak", lambda _: True),
]

# Fallback tier using OPS when xwOBA unavailable
TIER_MAP_OPS = [
    ("🔴 Elite", lambda ops: ops >= 0.830),
    ("🟠 Strong", lambda ops: ops >= 0.760),
    ("🟡 Average", lambda ops: ops >= 0.700),
    ("🟢 Weak", lambda _: True),
]


def resolve_team_id(team_input: str) -> int:
    """將隊名（中文/英文/縮寫）轉為 team ID"""
    upper = team_input.upper()
    if upper in TEAM_MAP:
        return TEAM_MAP[upper]
    if team_input in TEAM_MAP:
        return TEAM_MAP[team_input]
    lower = team_input.lower()
    if lower in FULL_NAMES:
        return FULL_NAMES[lower]
    for name, tid in FULL_NAMES.items():
        if lower in name:
            return tid
    raise ValueError(f"Unknown team: {team_input}")


def fetch_team_roster(team_id: int, year: int) -> list[dict]:
    """從 MLB Stats API 取得球隊 active roster"""
    resp = requests.get(
        f"{MLB_API_BASE}/teams/{team_id}/roster",
        params={"rosterType": "active", "season": year},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    players = []
    for p in data.get("roster", []):
        person = p["person"]
        pos = p.get("position", {}).get("abbreviation", "")
        # 排除投手
        if pos == "P":
            continue
        players.append({
            "id": person["id"],
            "name": person["fullName"],
            "position": pos,
        })
    return players


def fetch_player_batting(mlbam_id: int, year: int) -> dict | None:
    """從 MLB Stats API 取得單一球員本季打擊數據"""
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{mlbam_id}/stats",
            params={"stats": "season", "group": "hitting", "season": year},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        stats_list = data.get("stats", [])
        if not stats_list or not stats_list[0].get("splits"):
            return None

        s = stats_list[0]["splits"][0]["stat"]
        pa = int(s.get("plateAppearances", 0))
        if pa == 0:
            return None

        avg = float(s.get("avg", 0))
        obp = float(s.get("obp", 0))
        slg = float(s.get("slg", 0))
        ops = float(s.get("ops", 0))
        babip = float(s.get("babip", 0))
        so = int(s.get("strikeOuts", 0))
        bb = int(s.get("baseOnBalls", 0))

        k_pct = round(so / pa * 100, 1) if pa > 0 else 0.0
        bb_pct = round(bb / pa * 100, 1) if pa > 0 else 0.0
        iso = round(slg - avg, 3)

        return {
            "mlbam_id": mlbam_id,
            "pa": pa,
            "avg": round(avg, 3),
            "obp": round(obp, 3),
            "slg": round(slg, 3),
            "ops": round(ops, 3),
            "iso": iso,
            "babip": round(babip, 3),
            "k_pct": k_pct,
            "bb_pct": bb_pct,
        }
    except Exception:
        return None


def fetch_statcast_batting_leaderboard(year: int) -> tuple[dict, dict]:
    """取得 Statcast 打者 expected stats 和 exit velo/barrels leaderboard
    回傳兩個 dict，key = player_id (str)
    """
    expected_map = {}
    barrels_map = {}

    try:
        with _redirect_pybaseball_stdout():
            df_exp = statcast_batter_expected_stats(year, minPA=1)
        if not df_exp.empty:
            for _, row in df_exp.iterrows():
                pid = str(row.get("player_id", ""))
                expected_map[pid] = {
                    "xwoba": round(float(row.get("est_woba", 0)), 3) if row.get("est_woba") is not None else None,
                    "xba": round(float(row.get("est_ba", 0)), 3) if row.get("est_ba") is not None else None,
                    "xslg": round(float(row.get("est_slg", 0)), 3) if row.get("est_slg") is not None else None,
                }
    except Exception:
        pass

    try:
        with _redirect_pybaseball_stdout():
            df_bar = statcast_batter_exitvelo_barrels(year, minBBE=1)
        if not df_bar.empty:
            for _, row in df_bar.iterrows():
                pid = str(row.get("player_id", ""))
                barrels_map[pid] = {
                    "ev95pct": round(float(row.get("ev95percent", 0)), 1) if row.get("ev95percent") is not None else None,
                    "barrel_pct": round(float(row.get("brl_percent", 0)), 1) if row.get("brl_percent") is not None else None,
                }
    except Exception:
        pass

    return expected_map, barrels_map


def fetch_il_names(team_id: int, year: int) -> set[str]:
    """從 40Man roster 取得 IL 球員姓名集合"""
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/teams/{team_id}/roster",
            params={"rosterType": "40Man", "season": year},
            timeout=10,
        )
        resp.raise_for_status()
        il_names = set()
        for entry in resp.json().get("roster", []):
            status_code = entry.get("status", {}).get("code", "A")
            if status_code in ("D7", "D10", "D15", "D60"):
                il_names.add(entry.get("person", {}).get("fullName", ""))
        return il_names
    except Exception:
        return set()


def fetch_player_last7(mlbam_id: int) -> dict | None:
    """D3: 取得打者近 7 場打擊數據"""
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{mlbam_id}/stats",
            params={"stats": "lastXGames", "group": "hitting", "limit": 7},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        for sg in data.get("stats", []):
            for split in sg.get("splits", [])[:1]:
                s = split.get("stat", {})
                pa = int(s.get("plateAppearances", 0))
                if pa == 0:
                    return None
                return {
                    "avg": s.get("avg"),
                    "obp": s.get("obp"),
                    "slg": s.get("slg"),
                    "ops": s.get("ops"),
                    "babip": s.get("babip"),
                    "pa": pa,
                }
        return None
    except Exception:
        return None


def fetch_player_platoon(mlbam_id: int, year: int) -> dict | None:
    """D2: 取得打者 Platoon Splits（vs LHP / vs RHP）"""
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{mlbam_id}/stats",
            params={
                "stats": "statSplits",
                "group": "hitting",
                "season": year,
                "sitCodes": "vl,vr",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        result = {}
        for sg in data.get("stats", []):
            for split in sg.get("splits", []):
                desc = split.get("split", {}).get("description", "")
                s = split.get("stat", {})
                pa = int(s.get("plateAppearances", 0))
                key = "vs_lhp" if "Left" in desc else "vs_rhp"
                result[key] = {
                    "avg": s.get("avg"),
                    "obp": s.get("obp"),
                    "slg": s.get("slg"),
                    "ops": s.get("ops"),
                    "pa": pa,
                }
        return result if result else None
    except Exception:
        return None


def fetch_bvp(batter_id: int, pitcher_id: int) -> dict | None:
    """D4: 取得打者 vs 特定投手的生涯對戰紀錄

    API 回傳兩組 splits：
    - 第一組：彙總（無 season 欄位），取 PA 最大的一筆 = 生涯合計
    - 第二組：逐場明細（有 season 欄位），忽略
    """
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{batter_id}/stats",
            params={
                "stats": "vsPlayer",
                "group": "hitting",
                "opposingPlayerId": pitcher_id,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        # 只取第一組 splits（彙總資料，無 season 欄位）
        stats_groups = data.get("stats", [])
        if not stats_groups:
            return None

        first_group = stats_groups[0].get("splits", [])
        if not first_group:
            return None

        # 取 PA 最大的一筆 = 生涯對戰合計
        best = max(first_group, key=lambda s: int(s.get("stat", {}).get("plateAppearances", 0)))
        s = best.get("stat", {})
        pa = int(s.get("plateAppearances", 0))
        if pa == 0:
            return None

        return {
            "pa": pa,
            "avg": s.get("avg"),
            "obp": s.get("obp"),
            "slg": s.get("slg"),
            "h": int(s.get("hits", 0)),
            "hr": int(s.get("homeRuns", 0)),
            "so": int(s.get("strikeOuts", 0)),
            "bb": int(s.get("baseOnBalls", 0)),
            "sample_sufficient": pa >= 15,
        }
    except Exception:
        return None


def analyze_team(team: str, year: int, opposing_pitcher_id: int | None = None) -> dict:
    """完整的球隊打線分析"""
    team_id = resolve_team_id(team)

    # 1. 取 active roster + IL 名單排除
    roster = fetch_team_roster(team_id, year)
    if not roster:
        return {"error": f"No active roster found for {team}"}

    il_names = fetch_il_names(team_id, year)

    # 2. 批次取 MLB API 打擊數據（排除 IL 球員）
    batters = []
    for player in roster:
        if player["name"] in il_names:
            continue
        stats = fetch_player_batting(player["id"], year)
        if stats:
            stats["name"] = player["name"]
            stats["position"] = player["position"]
            batters.append(stats)

    if not batters:
        return {"error": f"No batting stats found for {team} in {year}"}

    # D1: 按 PA 排序取前 9 人（完整打線）
    batters.sort(key=lambda b: b["pa"], reverse=True)
    core_lineup = batters[:9]

    # 3. 取 Statcast leaderboard（一次拉全聯盟，記憶體內 merge）
    expected_map, barrels_map = fetch_statcast_batting_leaderboard(year)

    # 4. Merge Statcast + D2 Platoon + D3 Last7 + D4 BvP
    for batter in core_lineup:
        pid = str(batter["mlbam_id"])
        exp = expected_map.get(pid, {})
        bar = barrels_map.get(pid, {})
        # D5: 修正命名（ev95pct 取代 hard_hit_pct）
        batter["xwoba"] = exp.get("xwoba")
        batter["xba"] = exp.get("xba")
        batter["xslg"] = exp.get("xslg")
        batter["ev95pct"] = bar.get("ev95pct")
        batter["barrel_pct"] = bar.get("barrel_pct")

        # D2: Platoon Splits
        batter["platoon"] = fetch_player_platoon(batter["mlbam_id"], year)

        # D3: 近 7 場熱度
        batter["last_7"] = fetch_player_last7(batter["mlbam_id"])

        # D4: BvP（如果有提供對方投手 ID）
        if opposing_pitcher_id:
            batter["bvp"] = fetch_bvp(batter["mlbam_id"], opposing_pitcher_id)

    # 5. 整體指標
    avg_ops = sum(b["ops"] for b in core_lineup) / len(core_lineup)
    avg_babip = sum(b["babip"] for b in core_lineup) / len(core_lineup)
    avg_k_pct = sum(b["k_pct"] for b in core_lineup) / len(core_lineup)
    avg_bb_pct = sum(b["bb_pct"] for b in core_lineup) / len(core_lineup)

    xwoba_values = [b["xwoba"] for b in core_lineup if b.get("xwoba") is not None]
    avg_xwoba = sum(xwoba_values) / len(xwoba_values) if xwoba_values else None

    # 6. 打線評級（優先用 xwOBA，fallback OPS）
    tier = "🟢 Weak"
    if avg_xwoba is not None:
        for tier_name, check_fn in TIER_MAP:
            if check_fn(avg_xwoba):
                tier = tier_name
                break
    else:
        for tier_name, check_fn in TIER_MAP_OPS:
            if check_fn(avg_ops):
                tier = tier_name
                break

    # 7. 大小分傾向
    over_under_lean = 0
    if avg_babip <= 0.270:
        over_under_lean += 1
    if avg_babip >= 0.320:
        over_under_lean -= 1
    if avg_k_pct >= 25:
        over_under_lean -= 1
    if avg_xwoba is not None and avg_xwoba >= 0.350:
        over_under_lean += 1

    # 8. 串聯分析
    chain = {}
    if len(core_lineup) >= 3:
        chain["obp_top3"] = round(sum(b["obp"] for b in core_lineup[:3]) / 3, 3)
    if len(core_lineup) >= 5:
        chain["slg_mid"] = round(sum(b["slg"] for b in core_lineup[3:5]) / 2, 3)

    # D3: 整體近 7 場熱度
    last7_ops_values = []
    for b in core_lineup:
        if b.get("last_7") and b["last_7"].get("ops"):
            try:
                last7_ops_values.append(float(b["last_7"]["ops"]))
            except (ValueError, TypeError):
                pass
    recent_heat = None
    if last7_ops_values:
        avg_last7_ops = sum(last7_ops_values) / len(last7_ops_values)
        if avg_last7_ops >= 0.830:
            recent_heat = "🔥 Hot"
        elif avg_last7_ops <= 0.600:
            recent_heat = "🥶 Cold"
        else:
            recent_heat = "⚖️ Normal"

    return {
        "team": team,
        "team_id": team_id,
        "tier": tier,
        "avg_ops": round(avg_ops, 3),
        "avg_xwoba": round(avg_xwoba, 3) if avg_xwoba else None,
        "avg_babip": round(avg_babip, 3),
        "avg_k_pct": round(avg_k_pct, 1),
        "avg_bb_pct": round(avg_bb_pct, 1),
        "over_under_lean": over_under_lean,
        "recent_heat": recent_heat,
        "last7_babip": compute_last7_babip(core_lineup),  # Plan B §4.6（B10 觸發用）
        "chain": chain,
        "lineup": core_lineup,
    }


def compute_last7_babip(core_lineup: list[dict]) -> float | None:
    """近 7 天 BABIP 平均（Plan B 2026-04-22 §4.6）。

    從 core_lineup 每個打者的 `last_7.babip` 取值；非數值或缺失則忽略。
    空結果回 None（「沒數據」vs「平均 0」的語義差別）。
    """
    values = []
    for b in core_lineup:
        val = (b.get("last_7") or {}).get("babip")
        if val is None:
            continue
        try:
            values.append(float(val))
        except (ValueError, TypeError):
            continue
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Analyze team lineup")
    parser.add_argument("--team", required=True, help="Team name or abbreviation")
    parser.add_argument("--year", type=int, default=datetime.now().year)
    parser.add_argument("--opposing-pitcher-id", type=int, default=None,
                        help="Opposing pitcher MLBAM ID for BvP lookup")
    parser.add_argument("--output", "-o", help="Output file path (default: print to stdout)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "lineup_analyzer test mode"}, indent=2))
        return

    result = analyze_team(args.team, args.year, args.opposing_pitcher_id)
    json_output = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
