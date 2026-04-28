#!/usr/bin/env python3
"""MLB Pitcher Stats — Phase 2 投手進階數據（MLB Stats API + Statcast）"""

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    """用 pybaseball 查詢球員 MLBAM ID"""
    parts = name.strip().split()
    if len(parts) < 2:
        return None
    last = parts[-1]
    first = parts[0]
    playerid_lookup, _, _, _ = _import_pybaseball()
    try:
        with _redirect_pybaseball_stdout():
            result = playerid_lookup(last, first)
        if result.empty:
            return None
        if "mlb_played_last" in result.columns and len(result) > 1:
            result = result.sort_values("mlb_played_last", ascending=False, na_position="last")
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


def fetch_prior_year_stats(mlbam_id: int, year: int) -> dict:
    """C4: 取得去年數據作為開季小樣本回歸基準"""
    return fetch_mlb_api_stats(mlbam_id, year - 1)


def detect_triggers(data: dict) -> list[dict]:
    """偵測投手層級 Flag。回傳觸發列表。

    Flag 13：
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
                "flag": 13,
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
                    "必須補跑 `pitcher_stats.py --name \"...\" --year <YYYY-1>` 進行 YoY Statcast 對比；"
                    "TaskCreate B7 樣板（見 workflow.md §Phase 2 Step 2）"
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
            "flag": 13,
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
                "補跑 prior year + 對比 Statcast 5 項（avg_velo / pitch_types / whiff_pct / hard_hit_pct / xera）"
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
    prior = data.get("prior_year") or {}

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
    if isinstance(prior, dict) and "error" in prior:
        prior = {}

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

    # Prior year
    if prior:
        lines += [
            "## Prior Year",
            "",
            "| 指標 | 數值 |",
            "|------|------|",
            f"| Games / GS | {_md_fmt(prior.get('games'), 0)} / {_md_fmt(prior.get('gs'), 0)} |",
            f"| IP | {_md_fmt(prior.get('ip'))} |",
            f"| ERA | {_md_fmt(prior.get('era'))} |",
            f"| WHIP | {_md_fmt(prior.get('whip'))} |",
            f"| FIP | {_md_fmt(prior.get('fip'))} |",
            f"| xFIP | {_md_fmt(prior.get('xfip'))} |",
            f"| K% | {_md_fmt(prior.get('k_pct'), 1)} |",
            f"| BB% | {_md_fmt(prior.get('bb_pct'), 1)} |",
            f"| HR/9 | {_md_fmt(prior.get('hr_per_9'))} |",
            f"| GB% | {_md_fmt(prior.get('gb_pct'), 1)} |",
            "",
        ]

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

    # 8. C1: 近 3 場 Game Log + 用球數
    game_log = fetch_game_log(pitcher_id, args.year, limit=3)

    # 9. C2: Platoon Splits（vs LHB / vs RHB）
    platoon_splits = fetch_platoon_splits(pitcher_id, args.year)

    # 10. C3: Whiff% / CSW%（從 Statcast 原始資料計算）
    whiff_csw = fetch_whiff_csw(pitcher_id, args.year)
    if "error" not in whiff_csw and "error" not in statcast:
        statcast["whiff_pct"] = whiff_csw.get("whiff_pct")
        statcast["csw_pct"] = whiff_csw.get("csw_pct")

    # 11. C4: 去年數據
    prior_year = fetch_prior_year_stats(pitcher_id, args.year)

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
        "game_log": game_log,
        "platoon_splits": platoon_splits,
        "prior_year": prior_year,
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
