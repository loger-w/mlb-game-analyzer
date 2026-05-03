#!/usr/bin/env python3
"""MLB Pitcher Stats — 投手進階數據（MLB Stats API + Statcast）"""

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
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

def _import_pybaseball():
    """Lazy import：pybaseball 在實際抓 Statcast 時才載入。
    讓 format_md / detect_triggers 等純函式在沒有 pybaseball 的環境下也能測試。"""
    try:
        from pybaseball import (
            playerid_lookup,
            statcast_pitcher,
            statcast_pitcher_expected_stats,
            statcast_pitcher_exitvelo_barrels,
        )
        return playerid_lookup, statcast_pitcher, statcast_pitcher_expected_stats, statcast_pitcher_exitvelo_barrels
    except ImportError as e:
        raise RuntimeError(
            "pybaseball not installed or cannot import. Run: pip install pybaseball"
        ) from e


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
    """用 pybaseball 查詢球員 MLBAM ID。

    Strategy:
      1. Strict match
      2. Empty / not-found → fuzzy fallback（解 P3：Nick Martinez vs Nick Martínez）
      3. fuzzy 結果按 mlb_played_last 排序取最新；過濾掉 < current_year - 1 的舊球員
    """
    parts = name.strip().split()
    if len(parts) < 2:
        return None
    last = parts[-1]
    first = parts[0]
    playerid_lookup, _, _, _ = _import_pybaseball()

    def _resolve(df):
        """從 DataFrame 取 mlb_played_last 最大者的 key_mlbam，套年份過濾。
        回傳 (int, row) 元組，或 None（過濾後無結果）。"""
        if df.empty:
            return None
        if "mlb_played_last" in df.columns and len(df) > 1:
            df = df.sort_values("mlb_played_last", ascending=False, na_position="last")
        row = df.iloc[0]
        last_year = row.get("mlb_played_last") if "mlb_played_last" in df.columns else None
        current_year = datetime.now().year
        # 拒絕 last_year < current_year - 1 的歷史球員（避免 fuzzy 命中退役同名球員）
        if last_year is not None and not pd.isna(last_year) and last_year < current_year - 1:
            return None
        return int(row["key_mlbam"]), row

    # Round 1: strict
    try:
        with _redirect_pybaseball_stdout():
            strict_result = playerid_lookup(last, first)
    except Exception:
        strict_result = None

    if strict_result is not None and not strict_result.empty:
        resolved = _resolve(strict_result)
        if resolved is not None:
            return resolved[0]  # only return ID; no warning needed for strict success

    # Round 2: fuzzy fallback
    try:
        with _redirect_pybaseball_stdout():
            fuzzy_result = playerid_lookup(last, first, fuzzy=True)
    except Exception:
        return None

    if fuzzy_result is None or fuzzy_result.empty:
        return None

    resolved = _resolve(fuzzy_result)
    if resolved is None:
        return None

    matched_id, matched_row = resolved
    matched_name = f"{matched_row.get('name_first', '?')} {matched_row.get('name_last', '?')}"
    print(f"⚠️ name \"{name}\" matched fuzzy → \"{matched_name}\" (mlbam={matched_id})",
          file=sys.stderr)
    return matched_id


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
    _, _, statcast_pitcher_expected_stats, _ = _import_pybaseball()
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
    _, statcast_pitcher, _, _ = _import_pybaseball()
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
    _, _, _, statcast_pitcher_exitvelo_barrels = _import_pybaseball()
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


def _import_arsenal_fn():
    """Lazy import for `statcast_pitcher_arsenal_stats`. Kept separate from
    `_import_pybaseball()` so existing 4-tuple monkeypatch sites stay unaffected."""
    try:
        from pybaseball import statcast_pitcher_arsenal_stats
        return statcast_pitcher_arsenal_stats
    except ImportError as e:
        raise RuntimeError(
            "pybaseball not installed or cannot import. Run: pip install pybaseball"
        ) from e


def _import_stuff_fns():
    """Lazy import for FanGraphs Stuff+ leaderboard pair: (playerid_reverse_lookup,
    pitching_stats). Returns 2-tuple. Reverse-lookup is the safer MLBAM→IDfg path
    that bypasses the J.T. / Castillo name-search bugs that name-based lookup hits."""
    try:
        from pybaseball import playerid_reverse_lookup, pitching_stats
        return playerid_reverse_lookup, pitching_stats
    except ImportError as e:
        raise RuntimeError(
            "pybaseball not installed or cannot import. Run: pip install pybaseball"
        ) from e


def _safe_round(value, decimals: int):
    """`round(float(v), n)` if v is not None; else None. Centralizes the
    `value is not None ? round(...) : None` boilerplate."""
    if value is None:
        return None
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None


def fetch_pitch_arsenal(mlbam_id: int, year: int) -> list[dict]:
    """從 Statcast pitcher arsenal leaderboard 取此投手所有球種的 RV/whiff/xwOBA。

    使用 pybaseball.statcast_pitcher_arsenal_stats(year, minPA=25)，與既有
    leaderboard 模式一致。回傳 list[dict]，每個 dict 對應一個球種，按 usage
    降序排序。錯誤情況回 [{"error": str}]，方便下游 `format_md` 直接 skip section。

    Schema per dict: pitch_type, pitch_name, usage, rv_per_100, xwoba_against,
    whiff_pct, put_away_pct, hard_hit_pct.
    """
    try:
        fn = _import_arsenal_fn()
    except RuntimeError as e:
        return [{"error": str(e)}]
    try:
        with _redirect_pybaseball_stdout():
            df = fn(year, minPA=25)
        if df.empty:
            return [{"error": "No arsenal data"}]
        rows = df[df["player_id"].astype(str) == str(mlbam_id)]
        if rows.empty:
            return [{"error": f"Player {mlbam_id} not found in arsenal data"}]
        result = []
        for _, r in rows.iterrows():
            result.append({
                "pitch_type": r.get("pitch_type"),
                "pitch_name": r.get("pitch_name"),
                "usage": _safe_round(r.get("pitch_usage"), 1),
                "rv_per_100": _safe_round(r.get("run_value_per_100"), 2),
                "xwoba_against": _safe_round(r.get("est_woba"), 3),
                "whiff_pct": _safe_round(r.get("whiff_percent"), 1),
                "put_away_pct": _safe_round(r.get("put_away"), 1),
                "hard_hit_pct": _safe_round(r.get("hard_hit_percent"), 1),
            })
        result.sort(key=lambda x: x["usage"] if x["usage"] is not None else -1, reverse=True)
        return result
    except Exception as e:
        return [{"error": str(e)}]


def fetch_stuff_pitching_plus(mlbam_id: int, year: int) -> dict:
    """Fetch FanGraphs Stuff+ / Location+ / Pitching+ for a given pitcher.

    Stuff+ is a composite of velo + spin + movement + release point, normalized
    to 100 = league average. It's FanGraphs IP, only exposed via
    `pybaseball.pitching_stats(year, qual=...)` which keys rows by `IDfg`.
    We resolve MLBAM → IDfg via `playerid_reverse_lookup` to avoid the
    name-based search bugs (J.T. dotted-initials, Luis Castillo id ambiguity).

    Use `qual=1` so small-sample early-season pitchers are still included
    (Stuff+ is the small-sample metric this fetch exists to support).

    Returns:
        Happy:    {"stuff_plus": float, "location_plus": float, "pitching_plus": float, "idfg": int}
        Failure:  {"error": str}  — caller treats missing keys as None and falls
                  through to compute_tier_v2's "missing_stuff" path.
    """
    try:
        reverse_lookup, pitching_stats = _import_stuff_fns()
    except RuntimeError as e:
        return {"error": str(e)}

    try:
        rev_df = reverse_lookup([mlbam_id], key_type="mlbam")
        if rev_df is None or len(rev_df) == 0:
            return {"error": f"MLBAM {mlbam_id} not found in FanGraphs reverse lookup"}
        idfg_raw = rev_df.iloc[0].get("key_fangraphs")
        if idfg_raw is None:
            return {"error": f"MLBAM {mlbam_id} has no key_fangraphs in lookup row"}
        idfg = int(idfg_raw)

        with _redirect_pybaseball_stdout():
            df = pitching_stats(year, qual=1)
        if df is None or df.empty:
            return {"error": f"FanGraphs pitching_stats({year}) returned empty"}
        rows = df[df["IDfg"] == idfg]
        if rows.empty:
            return {"error": f"IDfg {idfg} (MLBAM {mlbam_id}) not in FanGraphs pitching_stats — likely 0 IP this season"}
        r = rows.iloc[0]
        return {
            "stuff_plus": _safe_round(r.get("Stuff+"), 1),
            "location_plus": _safe_round(r.get("Location+"), 1),
            "pitching_plus": _safe_round(r.get("Pitching+"), 1),
            "idfg": idfg,
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_game_log(mlbam_id: int, year: int, limit: int = 3) -> list[dict]:
    """C1: 取得近 N 場 Game Log（含用球數）"""
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{mlbam_id}/stats",
            params={"stats": "gameLog", "group": "pitching", "season": year},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        games = []
        for sg in data.get("stats", []):
            for split in sg.get("splits", [])[:limit]:
                s = split.get("stat", {})
                games.append({
                    "date": split.get("date"),
                    "opponent": split.get("opponent", {}).get("name"),
                    "ip": parse_ip(s.get("inningsPitched", "0")),
                    "era": float(s.get("era", 0)) if s.get("era") else None,
                    "k": int(s.get("strikeOuts", 0)),
                    "bb": int(s.get("baseOnBalls", 0)),
                    "h": int(s.get("hits", 0)),
                    "er": int(s.get("earnedRuns", 0)),
                    "pitches": int(s.get("numberOfPitches", 0)),
                    "strikes": int(s.get("strikes", 0)),
                })
        return games
    except Exception as e:
        return [{"error": str(e)}]


def fetch_platoon_splits(mlbam_id: int, year: int) -> dict:
    """C2: 取得投手 Platoon Splits（vs LHB / vs RHB）"""
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/people/{mlbam_id}/stats",
            params={
                "stats": "statSplits",
                "group": "pitching",
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
                bf = int(s.get("battersFaced", 0))
                k = int(s.get("strikeOuts", 0))
                bb = int(s.get("baseOnBalls", 0))
                key = "vs_left" if "Left" in desc else "vs_right"
                result[key] = {
                    "avg": s.get("avg"),
                    "obp": s.get("obp"),
                    "slg": s.get("slg"),
                    "k": k,
                    "bb": bb,
                    "bf": bf,
                    "k_pct": round(k / bf * 100, 1) if bf > 0 else 0.0,
                    "bb_pct": round(bb / bf * 100, 1) if bf > 0 else 0.0,
                }
        return result if result else {"error": "No platoon split data"}
    except Exception as e:
        return {"error": str(e)}


def _pa_outcome_aggregates(pa_df: pd.DataFrame) -> dict:
    """從 PA-level DataFrame slice（一行一 PA，含 events 欄）算 OPS / K% / BB% / BF。

    OBP / SLG / AVG 由 events 計數 + sabermetric 公式合成（PA 級資料不直接給 OPS）。
    Plan B helper — input 是 statcast_pitcher 經 events.notna() filter 過的 slice。
    """
    bf = len(pa_df)
    if bf == 0:
        return {"ops": None, "k_pct": 0.0, "bb_pct": 0.0, "bf": 0}

    events = pa_df["events"]
    h_singles = int((events == "single").sum())
    h_doubles = int((events == "double").sum())
    h_triples = int((events == "triple").sum())
    h_hrs = int((events == "home_run").sum())
    h = h_singles + h_doubles + h_triples + h_hrs

    bb = int((events == "walk").sum())
    hbp = int((events == "hit_by_pitch").sum())
    k = int(events.isin(["strikeout", "strikeout_double_play"]).sum())
    sf = int(events.isin(["sac_fly", "sac_fly_double_play"]).sum())
    sh = int(events.isin(["sac_bunt", "sacrifice_bunt_double_play"]).sum())

    ab = bf - bb - hbp - sf - sh
    if ab <= 0:
        return {"ops": None,
                "k_pct": round(k / bf * 100, 1),
                "bb_pct": round(bb / bf * 100, 1),
                "bf": bf}

    # ab > 0 guaranteed below; obp_denom = ab + ... ≥ ab ≥ 1, so no division-by-zero defense needed
    obp_denom = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_denom
    tb = h_singles + 2 * h_doubles + 3 * h_triples + 4 * h_hrs
    slg = tb / ab
    ops = obp + slg

    return {
        "ops": round(ops, 3),
        "k_pct": round(k / bf * 100, 1),
        "bb_pct": round(bb / bf * 100, 1),
        "bf": bf,
    }


def _compute_tto_from_statcast(mlbam_id: int, year_start: int, year_end: int) -> dict:
    """從 pybaseball Statcast 逐球資料聚合成 TTO1 / TTO2 / TTO3 桶。

    對每個 PA（events 非 null 的 row），在 (game_pk, batter) 群組內依
    at_bat_number 升冪排序,cumcount + 1 即 PA ordinal（1st / 2nd / 3rd PA）。
    超過 3rd（4th+ PA）忽略，因為樣本太稀。
    """
    _, statcast_pitcher_fn, _, _ = _import_pybaseball()
    try:
        start = f"{year_start}-03-20"
        end = f"{year_end}-11-05"
        df = statcast_pitcher_fn(start, end, mlbam_id)
        if df is None or df.empty:
            return {"error": "No Statcast data"}

        pa_df = df[df["events"].notna()].copy()
        if pa_df.empty:
            return {"error": "No PA events in Statcast data"}

        pa_df = pa_df.sort_values(["game_pk", "at_bat_number"])
        pa_df["tto_ordinal"] = pa_df.groupby(["game_pk", "batter"]).cumcount() + 1

        result: dict = {}
        for ordinal in (1, 2, 3):
            bucket = pa_df[pa_df["tto_ordinal"] == ordinal]
            if len(bucket) == 0:
                continue
            result[f"tto{ordinal}"] = _pa_outcome_aggregates(bucket)
        return result if result else {"error": "No TTO buckets computed"}
    except Exception as e:
        return {"error": f"statcast TTO compute failed: {e}"}


_TTO_MIN_BF = 30  # tto3 bucket 最小 BF；不足走 career fallback


def _has_sufficient_tto3(data: dict) -> bool:
    """data 裡 tto3.bf 是否 ≥ _TTO_MIN_BF。error / 缺 tto3 → False。"""
    if "error" in data:
        return False
    tto3 = data.get("tto3") or {}
    return (tto3.get("bf") or 0) >= _TTO_MIN_BF


def fetch_tto_splits(mlbam_id: int, year: int) -> dict:
    """C2.5：取得投手 Times-Through-Order Splits（TTO1 / TTO2 / TTO3）。

    Plan B：用 pybaseball Statcast pitch-by-pitch 自行聚合。
    Season 優先；TTO3 BF < 30 → silent fallback 5-year career window。
    回傳：
      {
        "source": "season" | "career",
        "tto1": {...}, "tto2": {...}, "tto3": {...},
      }
      或 {"error": "..."} 兩條路徑都失敗時。

    Caller (signal_tto3_penalty) 看 tto3.bf 自行判斷 small_sample。
    """
    season_data = _compute_tto_from_statcast(mlbam_id, year, year)
    if _has_sufficient_tto3(season_data):
        season_data["source"] = "season"
        return season_data

    career_data = _compute_tto_from_statcast(mlbam_id, year - 4, year)
    if _has_sufficient_tto3(career_data):
        career_data["source"] = "career"
        return career_data

    if "error" not in season_data:
        season_data["source"] = "season"
        return season_data
    if "error" not in career_data:
        career_data["source"] = "career"
        return career_data
    return {"error": season_data.get("error", "TTO splits unavailable")}


def fetch_whiff_csw(mlbam_id: int, year: int) -> dict:
    """C3: 從 Statcast 原始資料計算 Whiff% 和 CSW%"""
    _, statcast_pitcher, _, _ = _import_pybaseball()
    try:
        start = f"{year}-03-20"
        end = f"{year}-11-05"
        df = statcast_pitcher(start, end, mlbam_id)
        if df.empty:
            return {"error": "No Statcast data"}

        total = len(df)
        if "description" not in df.columns or total == 0:
            return {"error": "No description column"}

        swinging_strikes = len(df[df["description"].str.contains("swinging_strike", na=False)])
        called_strikes = len(df[df["description"].str.contains("called_strike", na=False)])

        return {
            "whiff_pct": round(swinging_strikes / total * 100, 1),
            "csw_pct": round((swinging_strikes + called_strikes) / total * 100, 1),
            "total_pitches": total,
        }
    except Exception as e:
        return {"error": str(e)}


def detect_triggers(data: dict) -> list[dict]:
    """偵測投手層級 Flag。回傳觸發列表。

    Flag 8：
    - |ERA - xERA| ≥ 1.5  → ERA 與 xERA 落差過大
    - IP < 30 且 prior_year_ERA - current_ERA ≥ 1.0 → 開季小樣本超過去年水準
    """
    triggers = []
    season = data.get("season") or {}
    expected = data.get("expected") or {}
    prior = data.get("prior_year") or {}

    if isinstance(season, dict) and "error" in season:
        season = {}
    if isinstance(expected, dict) and "error" in expected:
        expected = {}
    if isinstance(prior, dict) and "error" in prior:
        prior = {}

    era = season.get("era")
    xera = expected.get("xera")
    ip = season.get("ip")
    prior_era = prior.get("era")

    # 條件 1: |ERA - xERA| ≥ 1.5
    if era is not None and xera is not None:
        gap = era - xera
        if abs(gap) >= 1.5:
            triggers.append({
                "flag": 8,
                "name": "ERA-xERA gap",
                "value": round(gap, 3),
                "threshold": "|gap| ≥ 1.5",
                "details": {
                    "era": era,
                    "xera": xera,
                    "ip": ip,
                    "prior_year_era": prior_era,
                },
                "interpretation": (
                    "ERA 顯著低於 xERA（運氣 / 樣本影響，預示回升）"
                    if gap < 0
                    else "ERA 顯著高於 xERA（壓制力被掩蓋，預示反彈）"
                ),
                "action": (
                    "腳本層自動標 ⚠️ 風險提示；AI 於 summary.md「## 風險提示」段判讀"
                    "（運氣 / 結構性退化 / 樣本噪音），不自動補跑 YoY、不自動下修預測。"
                    "詳見 reference/flags-checklist.md §8"
                ),
            })

    # 條件 2: IP < 30 且 prior_year_ERA - current_ERA ≥ 1.0
    if (
        ip is not None
        and ip < 30
        and era is not None
        and prior_era is not None
        and (prior_era - era) >= 1.0
    ):
        triggers.append({
            "flag": 8,
            "name": "Small-sample regression risk",
            "value": round(prior_era - era, 3),
            "threshold": "IP<30 且 prior_year_ERA − current_ERA ≥ 1.0",
            "details": {
                "ip": ip,
                "current_era": era,
                "prior_year_era": prior_era,
            },
            "interpretation": "本季 ERA 大幅優於去年但樣本不足 → 預示回歸",
            "action": (
                "腳本層自動標 ⚠️ 風險提示；AI 於 summary.md「## 風險提示」段判讀"
                "（小樣本 / 回歸風險），不自動補跑 YoY、不自動下修預測。"
                "詳見 reference/flags-checklist.md §8"
            ),
        })
    return triggers


def _md_fmt(v, decimals: int = 2) -> str:
    """格式化數值；None → '—'。"""
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        if decimals == 0:
            return f"{v:.0f}"
        return f"{v:.{decimals}f}"
    return str(v)


def format_md(data: dict, command: str | None = None) -> str:
    """渲染投手 MD 摘要（厚 MD：Season + Statcast + Platoon + Recent + Prior + Triggers）。

    純函數：只讀 data dict，不呼叫外部 API。
    """
    name = data.get("name", "?")
    mlbam_id = data.get("mlbam_id", "?")
    age = data.get("age", "?")
    birth = data.get("birth_date", "—")
    hand = data.get("pitch_hand", "?")
    age_assessment = data.get("age_assessment", "—")
    tier = data.get("tier", "—")

    season = data.get("season") or {}
    expected = data.get("expected") or {}
    statcast = data.get("statcast") or {}
    platoon = data.get("platoon_splits") or {}
    game_log = data.get("game_log") or []

    # 過濾 error 字典
    if isinstance(season, dict) and "error" in season:
        season_err = season.get("error")
        season = {}
    else:
        season_err = None
    if isinstance(expected, dict) and "error" in expected:
        expected = {}
    if isinstance(statcast, dict) and "error" in statcast:
        statcast = {}

    pitch_types = statcast.get("pitch_types") or {}

    lines = [
        f"# Pitcher — {name} (MLBAM {mlbam_id})",
        f"**Hand**: {hand}HP | **Age**: {age} ({birth})",
        f"**Tier**: {tier}",
        f"**Age assessment**: {age_assessment}",
        "",
        "---",
        "",
    ]

    triggers = detect_triggers(data)
    if triggers:
        lines += ["## 🚨 Triggers", ""]
        for t in triggers:
            lines += [
                f"### Flag {t['flag']} 觸發 — {t['name']}",
                f"- 數值：**{t['value']}**（觸發閾值 {t['threshold']}）",
            ]
            details = t.get("details") or {}
            if details:
                detail_str = ", ".join(f"{k}={_md_fmt(v)}" for k, v in details.items())
                lines.append(f"- 細節：{detail_str}")
            lines += [
                f"- 解讀：{t['interpretation']}",
                f"- 處理：{t['action']}",
                "",
            ]
        lines += ["---", ""]

    # Season stats
    lines += [
        "## Season Stats",
        "",
        "| 指標 | 數值 |",
        "|------|------|",
        f"| Games / GS | {_md_fmt(season.get('games'), 0)} / {_md_fmt(season.get('gs'), 0)} |",
        f"| IP | {_md_fmt(season.get('ip'))} |",
        f"| ERA | {_md_fmt(season.get('era'))} |",
        f"| WHIP | {_md_fmt(season.get('whip'))} |",
        f"| FIP | {_md_fmt(season.get('fip'))} |",
        f"| xFIP | {_md_fmt(season.get('xfip'))} |",
        f"| K% | {_md_fmt(season.get('k_pct'), 1)} |",
        f"| BB% | {_md_fmt(season.get('bb_pct'), 1)} |",
        f"| K-BB% | {_md_fmt(season.get('k_bb_pct'), 1)} |",
        f"| HR/9 | {_md_fmt(season.get('hr_per_9'))} |",
        f"| GB% | {_md_fmt(season.get('gb_pct'), 1)} |",
        "",
    ]
    if season_err:
        lines.append(f"> ⚠️ Season stats error: `{season_err}`")
        lines.append("")

    # Expected
    lines += [
        "## Expected (Statcast)",
        "",
        "| 指標 | 數值 |",
        "|------|------|",
        f"| xERA | {_md_fmt(expected.get('xera'))} |",
        f"| xwOBA | {_md_fmt(expected.get('xwoba'), 3)} |",
        f"| xBA | {_md_fmt(expected.get('xba'), 3)} |",
        "",
    ]

    # Statcast Physical
    lines += [
        "## Statcast Physical",
        "",
        "| 指標 | 數值 |",
        "|------|------|",
        f"| avg velocity (mph) | {_md_fmt(statcast.get('avg_velo'), 1)} |",
        f"| max velocity (mph) | {_md_fmt(statcast.get('max_velo'), 1)} |",
        f"| whiff% | {_md_fmt(statcast.get('whiff_pct'), 1)} |",
        f"| csw% | {_md_fmt(statcast.get('csw_pct'), 1)} |",
        f"| hard_hit% | {_md_fmt(statcast.get('hard_hit_pct'), 1)} |",
        f"| EV ≥95% | {_md_fmt(statcast.get('ev95percent'), 1)} |",
        f"| barrel% | {_md_fmt(statcast.get('barrel_pct'), 1)} |",
        "",
    ]

    # Pitch Mix
    if pitch_types:
        lines += ["## Pitch Mix (% usage)", "", "| 球種 | % |", "|------|---|"]
        for pt, pct in pitch_types.items():
            lines.append(f"| {pt} | {_md_fmt(pct, 1)} |")
        lines.append("")

    # Pitch Arsenal (RV/100, xwOBA, whiff%, etc per pitch)
    arsenal = data.get("arsenal") or []
    arsenal_valid = [a for a in arsenal if isinstance(a, dict) and "error" not in a]
    if arsenal_valid:
        lines += [
            "## Pitch Arsenal (RV/100)",
            "",
            "| 球種 | usage% | RV/100 | xwOBA | whiff% | put-away% | hard-hit% |",
            "|------|--------|--------|-------|--------|-----------|-----------|",
        ]
        for a in arsenal_valid:
            lines.append(
                f"| {a.get('pitch_type', '—')} | "
                f"{_md_fmt(a.get('usage'), 1)} | "
                f"{_md_fmt(a.get('rv_per_100'), 2)} | "
                f"{_md_fmt(a.get('xwoba_against'), 3)} | "
                f"{_md_fmt(a.get('whiff_pct'), 1)} | "
                f"{_md_fmt(a.get('put_away_pct'), 1)} | "
                f"{_md_fmt(a.get('hard_hit_pct'), 1)} |"
            )
        lines.append("")

    # Platoon
    if platoon and "error" not in platoon:
        lines += ["## Platoon Splits", ""]
        for key, label in [("vs_left", "vs LHB"), ("vs_right", "vs RHB")]:
            p = platoon.get(key)
            if not p:
                continue
            lines += [
                f"### {label}（{p.get('bf', '—')} BF）",
                "",
                "| AVG | OBP | SLG | K | BB | K% | BB% |",
                "|-----|-----|-----|---|----|----|-----|",
                (
                    f"| {p.get('avg', '—')} | {p.get('obp', '—')} | {p.get('slg', '—')} | "
                    f"{p.get('k', '—')} | {p.get('bb', '—')} | "
                    f"{_md_fmt(p.get('k_pct'), 1)} | {_md_fmt(p.get('bb_pct'), 1)} |"
                ),
                "",
            ]

    # Recent starts
    if game_log:
        valid = [g for g in game_log if isinstance(g, dict) and "error" not in g]
        if valid:
            lines += [
                f"## Recent {len(valid)} Starts",
                "",
                "| 日期 | 對手 | IP | ER | K | BB | H | Pitches | Strikes |",
                "|------|------|-----|----|----|----|----|---------|---------|",
            ]
            for g in valid:
                lines.append(
                    f"| {g.get('date', '—')} | {g.get('opponent', '—')} | "
                    f"{_md_fmt(g.get('ip'), 1)} | {_md_fmt(g.get('er'), 0)} | "
                    f"{_md_fmt(g.get('k'), 0)} | {_md_fmt(g.get('bb'), 0)} | "
                    f"{_md_fmt(g.get('h'), 0)} | {_md_fmt(g.get('pitches'), 0)} | {_md_fmt(g.get('strikes'), 0)} |"
                )
            lines.append("")

    # Source
    lines += [
        "---",
        "",
        "## Source",
        f"- Generated by: `{command or 'pitcher_stats.py'}`",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        "- JSON sibling: see same directory `<basename>.json`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Fetch pitcher advanced stats")
    parser.add_argument("--name", required=True, help="Pitcher full name (e.g. 'Gerrit Cole')")
    parser.add_argument("--year", type=int, default=datetime.now().year, help="Season year")
    parser.add_argument("--output", "-o", help="Output file path (default: print to stdout)")
    parser.add_argument("--no-md", action="store_true", help="Skip MD summary output (only write JSON)")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    parser.add_argument("--mlbam-id", type=int, default=None,
                        help="直接指定 MLBAM ID，跳過 name lookup")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "pitcher_stats test mode"}, indent=2))
        return

    # 1. 查詢 MLBAM ID（若 --mlbam-id 已提供則跳過 name lookup）
    if args.mlbam_id:
        pitcher_id = args.mlbam_id
    else:
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

    # 8. C1: 近 3 場 Game Log + 用球數
    game_log = fetch_game_log(pitcher_id, args.year, limit=3)

    # 9. C2: Platoon Splits（vs LHB / vs RHB）
    platoon_splits = fetch_platoon_splits(pitcher_id, args.year)

    # 9.5. C2.5: Times-Through-Order Splits（TTO1/2/3，Plan B Statcast 聚合）
    tto_splits = fetch_tto_splits(pitcher_id, args.year)

    # 10. C3: Whiff% / CSW%（從 Statcast 原始資料計算）
    whiff_csw = fetch_whiff_csw(pitcher_id, args.year)
    if "error" not in whiff_csw and "error" not in statcast:
        statcast["whiff_pct"] = whiff_csw.get("whiff_pct")
        statcast["csw_pct"] = whiff_csw.get("csw_pct")

    # 11. Pitch Arsenal (per-pitch RV/100, xwOBA, whiff%, etc)
    arsenal = fetch_pitch_arsenal(pitcher_id, args.year)

    # 12. Stuff+ / Pitching+ (FanGraphs leaderboard via MLBAM→IDfg reverse lookup)
    stuff = fetch_stuff_pitching_plus(pitcher_id, args.year)
    stuff_for_tier = stuff if "error" not in stuff else None

    # 13. Tier v2 — blended xFIP/K-BB%/Stuff+/age formula (existing v1 `tier` stays
    #     in place for backward-compat). See lib_tier_v2 for formula details.
    from lib_tier_v2 import compute_tier_v2, compute_tier_gap
    tier_v2_result = compute_tier_v2(season, statcast, age=age, stuff=stuff_for_tier)
    tier_gap = compute_tier_gap(tier_v2_result, era=season.get("era"))

    output = {
        "name": args.name,
        "mlbam_id": pitcher_id,
        "age": age,
        "birth_date": info.get("birth_date"),
        "pitch_hand": info.get("pitch_hand"),
        "age_assessment": age_assessment,
        "tier": tier,
        "tier_v2": tier_v2_result["tier_v2"],
        "tier_components": tier_v2_result["components"],
        "tier_confidence": tier_v2_result["confidence"],
        "tier_gap": tier_gap,
        "season": season,
        "expected": expected,
        "statcast": statcast,
        "game_log": game_log,
        "platoon_splits": platoon_splits,
        "tto_splits": tto_splits,
        "arsenal": arsenal,
        "stuff": stuff,
    }

    json_output = json.dumps(output, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Saved to {args.output}", file=sys.stderr)

        if not args.no_md:
            json_path = Path(args.output)
            md_path = json_path.with_name(json_path.stem + "_summary.md")
            command = f"pitcher_stats.py --name \"{args.name}\" --year {args.year}"
            try:
                md_path.write_text(format_md(output, command=command), encoding="utf-8")
                print(f"Saved summary to {md_path}", file=sys.stderr)
            except Exception as e:
                print(f"Skipped summary md: {e}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
