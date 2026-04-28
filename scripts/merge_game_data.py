#!/usr/bin/env python3
"""MLB Merge Game Data — 合併 Phase 1/2 腳本輸出為 predict.py 所需的 merged.json"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Park Factor 資料（2023-2025 3 年加權，Baseball Savant；HR PF 暫不啟用）
_PF_DATA_PATH = Path(__file__).parent / "data" / "park_factors.json"
_PF_DATA = json.loads(_PF_DATA_PATH.read_text(encoding="utf-8"))
PARK_FACTORS = _PF_DATA["park_factors"]
PARK_ALIASES = _PF_DATA["_aliases"]


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
    prior_pitcher_data: dict | None,
    prefix: str,
) -> dict:
    """Plan B §4.6：產 nested `{prefix}_pitcher` dict，包含 era/xera/ip + prior_year.era。

    給 predict.py 的 pitcher_triggers_yoy 讀（B7 YoY 觸發判斷）。
    與現有 `extract_pitcher_features` 共存（不動 flat keys，確保 review_stats 等 backward-compat）。
    """
    season = pitcher_data.get("season", {}) if pitcher_data else {}
    if isinstance(season, dict) and "error" in season:
        season = {}
    expected = pitcher_data.get("expected", {}) if pitcher_data else {}
    if isinstance(expected, dict) and "error" in expected:
        expected = {}
    prior_season = (prior_pitcher_data or {}).get("season", {}) if prior_pitcher_data else {}
    if isinstance(prior_season, dict) and "error" in prior_season:
        prior_season = {}

    era = season.get("era")
    # xera 來源優先序：season.xera (legacy/test) → expected.xera (pitcher_stats 實際輸出位置)
    xera = season.get("xera") if season.get("xera") is not None else expected.get("xera")
    ip = season.get("ip")
    delta = None
    if era is not None and xera is not None:
        delta = round(abs(era - xera), 3)

    return {
        f"{prefix}_pitcher": {
            "era": era,
            "xera": xera,
            "ip": ip,
            "era_xera_delta": delta,
            "prior_year": {"era": prior_season.get("era")},
        }
    }


def extract_lineup_nested(lineup_data: dict | None, prefix: str) -> dict:
    """Plan B §4.6：產 nested `{prefix}_lineup` dict，包含 recent_babip。

    給 predict.py 的 lineup_triggers_babip 讀（B10 BABIP 回歸觸發判斷）。
    """
    recent = lineup_data.get("last7_babip") if lineup_data else None
    return {
        f"{prefix}_lineup": {
            "recent_babip": recent,
        }
    }


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


def resolve_park_factor(venue_name: str | None) -> float:
    """以 venue_name 解析 runs PF（HR PF 暫不啟用）。

    舊球場名透過 _aliases 表解析到 canonical 新名（如 Tropicana → Steinbrenner）。
    未知 venue 回傳 100.0（聯盟平均，安全 fallback）。
    """
    if not venue_name:
        return 100.0
    canonical = PARK_ALIASES.get(venue_name, venue_name)
    entry = PARK_FACTORS.get(canonical)
    if entry:
        return float(entry["runs_pf"])
    return 100.0


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
    """渲染 merged.json 一頁總覽 MD（Phase 2 整合）。

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
        f"# Merged Phase 2 — {away_team} @ {home_team}",
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
        if delta is not None and delta >= 1.5:
            trigger_lines.append(
                f"- **{label} 投手 Flag 13**：|ERA - xERA| = {delta:.2f}（≥ 1.5）；"
                f"prior_year ERA = {_md_fmt(p.get('prior_year', {}).get('era'))}"
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
        lines += ["## 🚨 Triggers (Phase 2 整合視角)", ""] + trigger_lines + ["", "---", ""]

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
        f"| prior_year ERA | {_md_fmt(((merged.get('home_pitcher') or {}).get('prior_year') or {}).get('era'))} | {_md_fmt(((merged.get('away_pitcher') or {}).get('prior_year') or {}).get('era'))} |",
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
    parser.add_argument("--home-pitcher-prior", default=None,
                        help="Optional prior-year pitcher_stats.py output for home starter (B7 YoY)")
    parser.add_argument("--away-pitcher-prior", default=None,
                        help="Optional prior-year pitcher_stats.py output for away starter (B7 YoY)")
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
    home_pitcher_prior_data = load_json(args.home_pitcher_prior) if args.home_pitcher_prior else None
    away_pitcher_prior_data = load_json(args.away_pitcher_prior) if args.away_pitcher_prior else None

    # 從 game_data 取得 team_id 和 venue
    game_info = game_data.get("game", {})
    home_team_id = game_info.get("home", {}).get("team_id")
    away_team_id = game_info.get("away", {}).get("team_id")
    venue_name = game_info.get("venue")
    game_year = int(game_info.get("date", "2026")[:4]) if game_info.get("date") else 2026

    # E1: 自動抓牛棚 ERA（可手動 override）
    if args.home_bullpen_era is not None:
        home_bp_era = args.home_bullpen_era
    elif home_team_id:
        home_bp_era = fetch_bullpen_era(home_team_id, game_year)
        print(f"Auto-fetched home bullpen ERA: {home_bp_era}", file=sys.stderr)
    else:
        home_bp_era = 4.00

    if args.away_bullpen_era is not None:
        away_bp_era = args.away_bullpen_era
    elif away_team_id:
        away_bp_era = fetch_bullpen_era(away_team_id, game_year)
        print(f"Auto-fetched away bullpen ERA: {away_bp_era}", file=sys.stderr)
    else:
        away_bp_era = 4.00

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
    # Plan B §4.6: nested dict 供 B7 YoY / B10 BABIP 觸發判斷；flat keys 保留 backward-compat
    merged.update(extract_pitcher_nested(home_pitcher_data, home_pitcher_prior_data, "home"))
    merged.update(extract_pitcher_nested(away_pitcher_data, away_pitcher_prior_data, "away"))
    merged.update(extract_lineup_nested(home_lineup_data, "home"))
    merged.update(extract_lineup_nested(away_lineup_data, "away"))
    merged["home_bullpen_era"] = home_bp_era
    merged["away_bullpen_era"] = away_bp_era
    merged["park_factor"] = park_factor
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
