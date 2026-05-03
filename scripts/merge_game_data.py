#!/usr/bin/env python3
"""MLB Merge Game Data — 合併各腳本輸出為單一 merged.json"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from park_factors_lib import PARK_ALIASES, PARK_FACTORS, runs_pf
from lib_role_tagging import CORE_BULLPEN_ROLES

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def load_json(path: str) -> dict:
    """讀取 JSON 檔案"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_pitcher_features(pitcher_data: dict, prefix: str) -> dict:
    """從 pitcher_stats.py 輸出提取 predict.py 所需特徵

    輸出欄位: {prefix}_starter_fip, {prefix}_starter_k_bb, {prefix}_starter_whip
    """
    season = pitcher_data.get("season", {})
    if "error" in season:
        season = {}

    fip = season.get("fip")
    k_bb = season.get("k_bb_pct")
    whip = season.get("whip")

    return {
        f"{prefix}_starter_fip": fip if fip is not None else 4.50,
        f"{prefix}_starter_k_bb": k_bb if k_bb is not None else 5.0,
        f"{prefix}_starter_whip": whip if whip is not None else 1.35,
    }


def extract_lineup_features(lineup_data: dict, prefix: str) -> dict:
    """從 lineup_analyzer.py 輸出提取 predict.py 所需特徵

    輸出欄位: {prefix}_batting_xwoba, {prefix}_batting_ops, {prefix}_batting_k_pct
    """
    xwoba = lineup_data.get("avg_xwoba")
    ops = lineup_data.get("avg_ops")
    k_pct = lineup_data.get("avg_k_pct")

    return {
        f"{prefix}_batting_xwoba": xwoba if xwoba is not None else 0.315,
        f"{prefix}_batting_ops": ops if ops is not None else 0.710,
        f"{prefix}_batting_k_pct": k_pct if k_pct is not None else 22.0,
    }


def extract_pitcher_nested(
    pitcher_data: dict | None,
    prefix: str,
) -> dict:
    """產 nested `{prefix}_pitcher` dict，包含 era/xera/ip/era_xera_delta + arsenal_top。

    與現有 `extract_pitcher_features` 共存（兩者讀同一份 pitcher JSON）。
    """
    season = pitcher_data.get("season", {}) if pitcher_data else {}
    if isinstance(season, dict) and "error" in season:
        season = {}
    expected = pitcher_data.get("expected", {}) if pitcher_data else {}
    if isinstance(expected, dict) and "error" in expected:
        expected = {}

    era = season.get("era")
    # xera 來源優先序：season.xera → expected.xera
    xera = season.get("xera") if season.get("xera") is not None else expected.get("xera")
    ip = season.get("ip")
    delta = None
    # 統一用 signed (era - xera)：負 = ERA<<xERA（運氣 / 樣本噪音）
    # 正 = ERA>>xERA（壓制力被掩蓋）。下游 abs() 比 1.5 才是 Flag 8 cond 1。
    if era is not None and xera is not None:
        delta = round(era - xera, 3)

    # Arsenal top-3 (by usage; fetch_pitch_arsenal 已排序，這裡只過濾 error 並截前 3)
    arsenal_raw = pitcher_data.get("arsenal", []) if pitcher_data else []
    arsenal_valid = [a for a in arsenal_raw if isinstance(a, dict) and "error" not in a]
    arsenal_top = arsenal_valid[:3]

    return {
        f"{prefix}_pitcher": {
            "era": era,
            "xera": xera,
            "ip": ip,
            "era_xera_delta": delta,
            "arsenal_top": arsenal_top,
        }
    }


def extract_lineup_nested(lineup_data: dict | None, prefix: str) -> dict:
    """產 nested `{prefix}_lineup` dict，包含 recent_babip（Flag 3 偵測用）。"""
    recent = lineup_data.get("last7_babip") if lineup_data else None
    return {
        f"{prefix}_lineup": {
            "recent_babip": recent,
        }
    }


def extract_core_bullpen_il_count(roster_data: dict | None, prefix: str) -> dict:
    """數出 roster.injured_list 中 core bullpen 角色（CORE_BULLPEN_ROLES）的人數。

    對應 matchup-factors.md §牛棚傷兵累計效應 的 1/2/3+ 分級。AI 在 dossier /
    summary 解讀，腳本只負責計數。

    Roster 缺檔 / 沒 IL → 0（0 是合法計數值，None 會誤導下游）。
    IL pitcher 無 core_role 欄位 → 不計入。
    """
    if not roster_data:
        return {f"{prefix}_core_bullpen_il_count": 0}
    il = roster_data.get("injured_list") or []
    count = sum(
        1 for entry in il
        if entry.get("core_role") in CORE_BULLPEN_ROLES
    )
    return {f"{prefix}_core_bullpen_il_count": count}


def extract_game_features(game_data: dict) -> dict:
    """從 fetch_game_data.py 輸出提取多窗口得失分

    輸出欄位: home_recent_rs, home_recent_ra, away_recent_rs, away_recent_ra
    額外 pass-through: recent_30, season 窗口（供 predict.py 趨勢標籤用）
    """
    home = game_data.get("home_recent", {})
    away = game_data.get("away_recent", {})
    home_30 = game_data.get("home_recent_30", {})
    away_30 = game_data.get("away_recent_30", {})
    home_season = game_data.get("home_season", {})
    away_season = game_data.get("away_season", {})

    h_rs = home.get("rs_per_game")
    h_ra = home.get("ra_per_game")
    a_rs = away.get("rs_per_game")
    a_ra = away.get("ra_per_game")

    result = {
        "home_recent_rs": h_rs if h_rs is not None else 4.5,
        "home_recent_ra": h_ra if h_ra is not None else 4.5,
        "away_recent_rs": a_rs if a_rs is not None else 4.5,
        "away_recent_ra": a_ra if a_ra is not None else 4.5,
        # 近 30 場（交叉驗證用）
        "home_recent_30_rs": home_30.get("rs_per_game") if home_30.get("rs_per_game") is not None else 4.5,
        "home_recent_30_ra": home_30.get("ra_per_game") if home_30.get("ra_per_game") is not None else 4.5,
        "away_recent_30_rs": away_30.get("rs_per_game") if away_30.get("rs_per_game") is not None else 4.5,
        "away_recent_30_ra": away_30.get("ra_per_game") if away_30.get("ra_per_game") is not None else 4.5,
        # 本季全部（趨勢標籤用）
        "home_season_rs": home_season.get("rs_per_game") if home_season.get("rs_per_game") is not None else 4.5,
        "home_season_ra": home_season.get("ra_per_game") if home_season.get("ra_per_game") is not None else 4.5,
        "away_season_rs": away_season.get("rs_per_game") if away_season.get("rs_per_game") is not None else 4.5,
        "away_season_ra": away_season.get("ra_per_game") if away_season.get("ra_per_game") is not None else 4.5,
        # 賽季場次數
        "home_season_games": game_data.get("home_season_games_count", 0),
        "away_season_games": game_data.get("away_season_games_count", 0),
    }
    return result


def fetch_bullpen_era(team_id: int, year: int) -> float:
    """E1: 從 MLB Stats API 自動取得球隊牛棚 ERA"""
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/teams/{team_id}/stats",
            params={
                "stats": "statSplits",
                "group": "pitching",
                "season": year,
                "sitCodes": "rp",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        for sg in data.get("stats", []):
            for split in sg.get("splits", []):
                era = split.get("stat", {}).get("era")
                if era is not None:
                    return float(era)
    except Exception:
        pass
    return 4.00  # fallback


def _fetch_merge_runtime_inputs(
    *,
    home_team_id: int | None,
    away_team_id: int | None,
    game_year: int,
    game_pk: int | None,
    home_bp_override: float | None = None,
    away_bp_override: float | None = None,
) -> tuple[float, float, dict | None]:
    """Concurrently fetch (home bullpen ERA, away bullpen ERA, weather).

    Cleanup #12: merge_game_data.main 原本 sequential 跑 2× fetch_bullpen_era +
    1× fetch_weather（critical-path 上 3 個 round-trip）。本 helper 用
    ThreadPoolExecutor 同時 dispatch，wall-clock 收斂到最慢那一個。

    Override 旁路 fetch（直接用該值）；team_id=None 時 fallback 4.00；
    game_pk=None 時 weather=None。fetch_bullpen_era 自帶 try/except + 4.00
    fallback；fetch_weather 自帶 try/except + None fallback；helper 不再加殼。
    """
    def _home_bp() -> float:
        if home_bp_override is not None:
            return home_bp_override
        if home_team_id is None:
            return 4.00
        era = fetch_bullpen_era(home_team_id, game_year)
        print(f"Auto-fetched home bullpen ERA: {era}", file=sys.stderr)
        return era

    def _away_bp() -> float:
        if away_bp_override is not None:
            return away_bp_override
        if away_team_id is None:
            return 4.00
        era = fetch_bullpen_era(away_team_id, game_year)
        print(f"Auto-fetched away bullpen ERA: {era}", file=sys.stderr)
        return era

    def _weather() -> dict | None:
        return fetch_weather(game_pk) if game_pk else None

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_home = ex.submit(_home_bp)
        f_away = ex.submit(_away_bp)
        f_weather = ex.submit(_weather)
        return (f_home.result(), f_away.result(), f_weather.result())


def fetch_weather(game_pk: int) -> dict | None:
    """從 feed/live 取 gameData.weather。

    回傳：
      - dict：{condition, temp_f, wind_text, indoor}
      - None：API 失敗 / weather 欄位不存在或全空
    """
    try:
        resp = requests.get(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
            timeout=10,
        )
        resp.raise_for_status()
        w = resp.json().get("gameData", {}).get("weather", {}) or {}
        condition = (w.get("condition") or "").strip()
        temp = (w.get("temp") or "").strip()
        wind = (w.get("wind") or "").strip()

        if not condition and not temp and not wind:
            return None

        indoor = condition.lower() in ("roof closed", "dome")

        try:
            temp_f = int(temp) if temp else None
        except ValueError:
            temp_f = None

        return {
            "condition": condition or None,
            "temp_f": temp_f,
            "wind_text": wind or None,
            "indoor": indoor,
        }
    except Exception as e:
        print(f"[merge_game_data] weather fetch failed: {e}", file=sys.stderr)
        return None


def resolve_park_factor(venue_name: str | None) -> float:
    """以 venue_name 解析 runs PF（thin wrapper；委派給 park_factors_lib.runs_pf）。"""
    return runs_pf(venue_name)


def extract_meta(game_data: dict, home_pitcher: dict, away_pitcher: dict) -> dict:
    """從各腳本輸出提取 metadata（隊名、投手名、先發場次、場館等）"""
    game = game_data.get("game", {})
    home_season = home_pitcher.get("season", {})
    away_season = away_pitcher.get("season", {})

    return {
        "_meta": {
            "home_team": game.get("home", {}).get("team"),
            "away_team": game.get("away", {}).get("team"),
            "home_sp": home_pitcher.get("name"),
            "away_sp": away_pitcher.get("name"),
            "home_sp_starts": home_season.get("gs"),
            "away_sp_starts": away_season.get("gs"),
            "venue": game.get("venue"),
            "game_pk": game.get("gamePk"),
            "game_date": game.get("date"),
            "official_date": game.get("officialDate"),
        }
    }


def _md_fmt(v, decimals: int = 2) -> str:
    """格式化數值；None → '—'。"""
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        if decimals == 0:
            return f"{v:.0f}"
        return f"{v:.{decimals}f}"
    return str(v)


def format_md(merged: dict, command: str | None = None) -> str:
    """渲染 merged.json 一頁總覽 MD。

    純函數：只讀 merged dict。包含 _meta / 投手 / 打線 / 牛棚 / Park / 多窗口戰績 / 觸發摘要。
    """
    meta = merged.get("_meta", {}) or {}
    home_team = meta.get("home_team", "?")
    away_team = meta.get("away_team", "?")
    home_sp = meta.get("home_sp", "?")
    away_sp = meta.get("away_sp", "?")
    home_gs = meta.get("home_sp_starts")
    away_gs = meta.get("away_sp_starts")
    venue = meta.get("venue", "—")
    game_pk = meta.get("game_pk", "—")
    game_date = meta.get("game_date", "—")

    lines = [
        f"# Merged Game Data — {away_team} @ {home_team}",
        f"**game_pk**: {game_pk} | **date**: {game_date}",
        f"**venue**: {venue} | **park_factor (runs)**: {_md_fmt(merged.get('park_factor'), 1)}",
        f"**先發**: {away_sp} ({away_team}, GS {away_gs}) vs {home_sp} ({home_team}, GS {home_gs})",
        "",
        "---",
        "",
    ]

    # 觸發摘要：從 nested dict 推回 trigger 提示
    trigger_lines = []
    for side, label in [("home", home_team), ("away", away_team)]:
        p = merged.get(f"{side}_pitcher", {}) or {}
        delta = p.get("era_xera_delta")
        if delta is not None and abs(delta) >= 1.5:
            trigger_lines.append(
                f"- **{label} 投手 Flag 8**：ERA - xERA = {delta:+.2f}（|Δ| ≥ 1.5）"
            )
        l = merged.get(f"{side}_lineup", {}) or {}
        recent_babip = l.get("recent_babip")
        if recent_babip is not None:
            try:
                rb = float(recent_babip)
                if rb <= 0.260 or rb >= 0.370:
                    trigger_lines.append(
                        f"- **{label} 打線 Flag 3**：last7 BABIP = {rb:.3f}（極端值，回歸風險）"
                    )
            except (ValueError, TypeError):
                pass
    if trigger_lines:
        lines += ["## 🚨 Triggers", ""] + trigger_lines + ["", "---", ""]

    # 投手對決
    lines += [
        "## Starting Pitchers",
        "",
        "| | Home | Away |",
        "|---|------|------|",
        f"| 投手 | {home_sp} | {away_sp} |",
        f"| FIP | {_md_fmt(merged.get('home_starter_fip'))} | {_md_fmt(merged.get('away_starter_fip'))} |",
        f"| K-BB% | {_md_fmt(merged.get('home_starter_k_bb'), 1)} | {_md_fmt(merged.get('away_starter_k_bb'), 1)} |",
        f"| WHIP | {_md_fmt(merged.get('home_starter_whip'))} | {_md_fmt(merged.get('away_starter_whip'))} |",
        f"| ERA / xERA | {_md_fmt((merged.get('home_pitcher') or {}).get('era'))} / {_md_fmt((merged.get('home_pitcher') or {}).get('xera'))} | {_md_fmt((merged.get('away_pitcher') or {}).get('era'))} / {_md_fmt((merged.get('away_pitcher') or {}).get('xera'))} |",
        f"| era_xera_delta | {_md_fmt((merged.get('home_pitcher') or {}).get('era_xera_delta'), 3)} | {_md_fmt((merged.get('away_pitcher') or {}).get('era_xera_delta'), 3)} |",
        "",
    ]

    # 打線
    lines += [
        "## Lineups",
        "",
        "| | Home | Away |",
        "|---|------|------|",
        f"| xwOBA | {_md_fmt(merged.get('home_batting_xwoba'), 3)} | {_md_fmt(merged.get('away_batting_xwoba'), 3)} |",
        f"| OPS | {_md_fmt(merged.get('home_batting_ops'), 3)} | {_md_fmt(merged.get('away_batting_ops'), 3)} |",
        f"| K% | {_md_fmt(merged.get('home_batting_k_pct'), 1)} | {_md_fmt(merged.get('away_batting_k_pct'), 1)} |",
        f"| last7 BABIP | {_md_fmt((merged.get('home_lineup') or {}).get('recent_babip'), 3)} | {_md_fmt((merged.get('away_lineup') or {}).get('recent_babip'), 3)} |",
        "",
    ]

    # 牛棚 / Park
    lines += [
        "## Bullpen / Park",
        "",
        "| | Home | Away |",
        "|---|------|------|",
        f"| Bullpen ERA | {_md_fmt(merged.get('home_bullpen_era'))} | {_md_fmt(merged.get('away_bullpen_era'))} |",
        "",
    ]

    # 多窗口得失分
    lines += [
        "## Multi-Window RS / RA (per game)",
        "",
        "| 視窗 | Home RS | Home RA | Away RS | Away RA |",
        "|------|---------|---------|---------|---------|",
        f"| 近 10 | {_md_fmt(merged.get('home_recent_rs'))} | {_md_fmt(merged.get('home_recent_ra'))} | {_md_fmt(merged.get('away_recent_rs'))} | {_md_fmt(merged.get('away_recent_ra'))} |",
        f"| 近 30 | {_md_fmt(merged.get('home_recent_30_rs'))} | {_md_fmt(merged.get('home_recent_30_ra'))} | {_md_fmt(merged.get('away_recent_30_rs'))} | {_md_fmt(merged.get('away_recent_30_ra'))} |",
        f"| 本季 ({_md_fmt(merged.get('home_season_games'), 0)}/{_md_fmt(merged.get('away_season_games'), 0)} 場) | {_md_fmt(merged.get('home_season_rs'))} | {_md_fmt(merged.get('home_season_ra'))} | {_md_fmt(merged.get('away_season_rs'))} | {_md_fmt(merged.get('away_season_ra'))} |",
        "",
    ]

    lines += [
        "---",
        "",
        "## Source",
        f"- Generated by: `{command or 'merge_game_data.py'}`",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        "- JSON sibling: see same directory `merged.json`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Merge all script outputs for predict.py")
    parser.add_argument("--game", help="fetch_game_data.py output JSON path")
    parser.add_argument("--home-pitcher", help="pitcher_stats.py output for home starter")
    parser.add_argument("--away-pitcher", help="pitcher_stats.py output for away starter")
    parser.add_argument("--home-lineup", help="lineup_analyzer.py output for home team")
    parser.add_argument("--away-lineup", help="lineup_analyzer.py output for away team")
    parser.add_argument("--home-roster", default=None,
                        help="roster_checker.py output for home team (optional, "
                             "powers core_bullpen_il_count). Skip → count = 0.")
    parser.add_argument("--away-roster", default=None,
                        help="roster_checker.py output for away team (optional, see --home-roster)")
    parser.add_argument("--home-bullpen-era", type=float, default=None,
                        help="Override home bullpen ERA (auto-fetched if omitted)")
    parser.add_argument("--away-bullpen-era", type=float, default=None,
                        help="Override away bullpen ERA (auto-fetched if omitted)")
    parser.add_argument("--park-factor", type=float, default=None,
                        help="Override park factor (auto-resolved from venue if omitted)")
    parser.add_argument("--output", "-o", help="Output file path (default: print to stdout)")
    parser.add_argument("--no-md", action="store_true", help="Skip MD summary output (only write JSON)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "merge_game_data test mode"}, indent=2))
        return

    required = ["game", "home_pitcher", "away_pitcher", "home_lineup", "away_lineup"]
    missing = [f"--{r.replace('_', '-')}" for r in required if getattr(args, r) is None]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    game_data = load_json(args.game)
    home_pitcher_data = load_json(args.home_pitcher)
    away_pitcher_data = load_json(args.away_pitcher)

    # 從 game_data 取得 team_id 和 venue
    game_info = game_data.get("game", {})
    home_team_id = game_info.get("home", {}).get("team_id")
    away_team_id = game_info.get("away", {}).get("team_id")
    venue_name = game_info.get("venue")
    game_year = int(game_info.get("date", "2026")[:4]) if game_info.get("date") else 2026

    # E1: 自動抓牛棚 ERA + weather（可手動 override；3 fetch 並行 — Cleanup #12）
    home_bp_era, away_bp_era, weather = _fetch_merge_runtime_inputs(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        game_year=game_year,
        game_pk=game_info.get("gamePk"),
        home_bp_override=args.home_bullpen_era,
        away_bp_override=args.away_bullpen_era,
    )

    # E2: 自動 Park Factor（可手動 override）
    if args.park_factor is not None:
        park_factor = args.park_factor
    else:
        park_factor = resolve_park_factor(venue_name)
        print(f"Auto-resolved park factor for {venue_name}: {park_factor}", file=sys.stderr)

    home_lineup_data = load_json(args.home_lineup)
    away_lineup_data = load_json(args.away_lineup)

    merged = {}
    merged.update(extract_game_features(game_data))
    merged.update(extract_pitcher_features(home_pitcher_data, "home"))
    merged.update(extract_pitcher_features(away_pitcher_data, "away"))
    merged.update(extract_lineup_features(home_lineup_data, "home"))
    merged.update(extract_lineup_features(away_lineup_data, "away"))
    # nested dict 供 Flag 8 (ERA-xERA gap) 偵測；flat keys 給 dossier 渲染直接讀
    merged.update(extract_pitcher_nested(home_pitcher_data, "home"))
    merged.update(extract_pitcher_nested(away_pitcher_data, "away"))
    merged.update(extract_lineup_nested(home_lineup_data, "home"))
    merged.update(extract_lineup_nested(away_lineup_data, "away"))
    merged["home_bullpen_era"] = home_bp_era
    merged["away_bullpen_era"] = away_bp_era
    merged["park_factor"] = park_factor
    # Core bullpen IL count from roster (optional; counts pitchers with
    # core_role ∈ {Closer, Setup, High-leverage RP, Co-Closer} on IL).
    home_roster_data = load_json(args.home_roster) if args.home_roster else None
    away_roster_data = load_json(args.away_roster) if args.away_roster else None
    merged.update(extract_core_bullpen_il_count(home_roster_data, "home"))
    merged.update(extract_core_bullpen_il_count(away_roster_data, "away"))
    # weather（與 park_factor 同層級的條件修正資料；無資料則 None，AI 在 summary 跳過）
    merged["weather"] = weather
    merged.update(extract_meta(game_data, home_pitcher_data, away_pitcher_data))

    json_output = json.dumps(merged, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Saved to {args.output}", file=sys.stderr)

        if not args.no_md:
            json_path = Path(args.output)
            md_path = json_path.with_name(json_path.stem + "_summary.md")
            try:
                md_path.write_text(format_md(merged, command="merge_game_data.py"), encoding="utf-8")
                print(f"Saved summary to {md_path}", file=sys.stderr)
            except Exception as e:
                print(f"Skipped summary md: {e}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
