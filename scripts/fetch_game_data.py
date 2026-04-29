#!/usr/bin/env python3
"""MLB Game Data Fetcher — Phase 1 API 資料一次撈齊"""

import argparse
import json
import sys
from datetime import datetime, timedelta

import requests

from _team_resolver import (
    FULL_NAMES,
    TEAM_ID_TO_ABBR,
    TEAM_MAP,
    resolve_team_id,
    team_abbr,
)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def compute_trend_arrows(rs10: float, ra10: float, rs30: float, ra30: float) -> dict:
    """近10 vs 近30 趨勢箭頭。|Δ| ≥ 0.5 才標箭頭。
    攻↑ = RS 上升；守↓ = RA 上升（防守變差）；守↑ = RA 下降。"""
    off_delta = round(rs10 - rs30, 2)
    def_delta = round(ra10 - ra30, 2)
    if off_delta >= 0.5:
        off_arrow = "↑"
    elif off_delta <= -0.5:
        off_arrow = "↓"
    else:
        off_arrow = "→"
    if def_delta >= 0.5:
        def_arrow = "↓"
    elif def_delta <= -0.5:
        def_arrow = "↑"
    else:
        def_arrow = "→"
    return {
        "off_arrow": off_arrow,
        "def_arrow": def_arrow,
        "off_delta": off_delta,
        "def_delta": def_delta,
    }


def detect_current_series(games: list[dict], current_opp_team_name: str, current_game_date: str) -> list[dict]:
    """從 games[0]（最近一場）往後掃描，連續對 current_opp_team_name 的場次收集為當前系列賽。
    結果按日期升序排列；同日多場（doubleheader）標 (DH-N)。
    games 應是 home_recent 格式（按日期 desc 排序）。

    返回 list[dict]，每個 dict 含原 game 欄位 + "label"（如 "G1" 或 "G2 (DH-2)"）。
    若 games 空或 games[0] 對手不同，返回空 list。
    """
    matched = []
    for g in games:
        if g.get("opponent") == current_opp_team_name:
            matched.append(g)
        else:
            break
    if not matched:
        return []

    # 升序排列；同日內保留原順序
    matched.sort(key=lambda g: g["date"])

    # 偵測 doubleheader：同日 ≥ 2 場
    by_date: dict[str, int] = {}
    for g in matched:
        by_date[g["date"]] = by_date.get(g["date"], 0) + 1

    result = []
    g_num = 1
    dh_counters: dict[str, int] = {}
    for g in matched:
        date = g["date"]
        if by_date[date] > 1:
            dh_counters[date] = dh_counters.get(date, 0) + 1
            label = f"G{g_num} (DH-{dh_counters[date]})"
        else:
            label = f"G{g_num}"
        result.append({**g, "label": label})
        g_num += 1
    return result


def format_streak_context(games: list[dict], streak: int) -> str | None:
    """格式化連勝/連敗對手列表（升序）。streak=0 或 games 空回 None。"""
    if streak == 0 or not games:
        return None
    n = abs(streak)
    label = "連勝對手" if streak > 0 else "連敗對手"
    items = []
    for g in games[:n]:
        abbr = team_abbr(None, g.get("opponent", ""))
        date_short = g.get("date", "")[5:]  # MM-DD
        items.append(f"{abbr} ({date_short})")
    items.reverse()  # games 是 desc → 反轉後為 asc
    return f"{label} → " + ", ".join(items)


def _fmt_signed(n) -> str:
    """格式化有號數值。None → '—'；正數加 +；負數用 '−'（U+2212）"""
    if n is None:
        return "—"
    if n > 0:
        return f"+{n}" if isinstance(n, int) else f"+{n:.2f}"
    if n < 0:
        return f"−{abs(n)}" if isinstance(n, int) else f"−{abs(n):.2f}"
    return "0"


def _fmt_streak(s) -> str:
    if s is None or s == 0:
        return "0"
    return f"+{s}" if s > 0 else f"−{abs(s)}"


def _fmt_num(n) -> str:
    if n is None:
        return "—"
    return f"{n:.2f}"


def _fmt_record_row(d: dict) -> str:
    rec = d.get("record", "—")
    rs = _fmt_num(d.get("rs_per_game"))
    ra = _fmt_num(d.get("ra_per_game"))
    diff = _fmt_signed(d.get("run_diff"))
    streak = _fmt_streak(d.get("streak"))
    return f"{rec}  (RS {rs} / RA {ra} / diff {diff} / streak {streak})"


def _fmt_record_row_no_streak(d: dict) -> str:
    rec = d.get("record", "—")
    rs = _fmt_num(d.get("rs_per_game"))
    ra = _fmt_num(d.get("ra_per_game"))
    diff = _fmt_signed(d.get("run_diff"))
    return f"{rec} (RS {rs} / RA {ra} / diff {diff})"


def format_summary_md(result: dict) -> str:
    """組合 game_data_summary.md 完整內容。
    Hard sections（必出現）：比賽資訊 / 戰績摘要 / 趨勢
    Soft sections（缺資料省略）：當前系列賽 / Streak 脈絡
    Fail-fast：result.game 缺失或雙方 team_id 缺失 → raise ValueError
    """
    if "game" not in result:
        raise ValueError("result.game missing — cannot generate summary")
    game = result["game"]
    home = game.get("home", {})
    away = game.get("away", {})
    if not home.get("team_id") or not away.get("team_id"):
        raise ValueError("home/away team_id missing — cannot generate summary")

    home_abbr = team_abbr(home["team_id"], home.get("team", ""))
    away_abbr = team_abbr(away["team_id"], away.get("team", ""))
    game_date = game.get("date", "")[:10]

    lines = [f"# Game Data Summary — {away_abbr} @ {home_abbr} ({game_date})", ""]

    # ========== 比賽資訊（hard） ==========
    lines += [
        "## 比賽資訊",
        f"- 日期 (ET): {game_date}",
        f"- 開賽 (UTC ISO): {game.get('date', '—')}",
        f"- 球場: {game.get('venue', '—')}",
        f"- 狀態: {game.get('status', '—')}",
        f"- 先發: {away.get('probable_pitcher', 'TBD')} ({away_abbr}, {away.get('probable_pitcher_id') or '—'}) vs {home.get('probable_pitcher', 'TBD')} ({home_abbr}, {home.get('probable_pitcher_id') or '—'})",
        "",
    ]

    # ========== 戰績摘要（hard） ==========
    home_recent = result.get("home_recent", {})
    away_recent = result.get("away_recent", {})
    home_30 = result.get("home_recent_30", {})
    away_30 = result.get("away_recent_30", {})
    home_season = result.get("home_season", {})
    away_season = result.get("away_season", {})
    home_n = result.get("home_season_games_count", 0)
    away_n = result.get("away_season_games_count", 0)

    lines += [
        "## 戰績摘要",
        "",
        f"| 區間 | {home_abbr}（主） | {away_abbr}（客） |",
        "|------|---------|----------|",
        f"| 近 10 場 | {_fmt_record_row(home_recent)} | {_fmt_record_row(away_recent)} |",
        f"| 近 30 場 | {_fmt_record_row_no_streak(home_30)} | {_fmt_record_row_no_streak(away_30)} |",
        f"| 本季 | {home_season.get('record', '—')} ({home_n} 場) | {away_season.get('record', '—')} ({away_n} 場) |",
        "",
    ]

    # ========== 趨勢（hard） ==========
    if (home_recent.get("rs_per_game") is not None
            and home_30.get("rs_per_game") is not None
            and away_recent.get("rs_per_game") is not None
            and away_30.get("rs_per_game") is not None):
        h = compute_trend_arrows(home_recent["rs_per_game"], home_recent["ra_per_game"],
                                 home_30["rs_per_game"], home_30["ra_per_game"])
        a = compute_trend_arrows(away_recent["rs_per_game"], away_recent["ra_per_game"],
                                 away_30["rs_per_game"], away_30["ra_per_game"])
        lines += [
            "## 趨勢（近 10 vs 近 30）",
            f"- {home_abbr}: 攻{h['off_arrow']} (RS {home_recent['rs_per_game']:.2f} vs {home_30['rs_per_game']:.2f}，{_fmt_signed(h['off_delta'])}) | 守{h['def_arrow']} (RA {home_recent['ra_per_game']:.2f} vs {home_30['ra_per_game']:.2f}，{_fmt_signed(h['def_delta'])})",
            f"- {away_abbr}: 攻{a['off_arrow']} (RS {away_recent['rs_per_game']:.2f} vs {away_30['rs_per_game']:.2f}，{_fmt_signed(a['off_delta'])}) | 守{a['def_arrow']} (RA {away_recent['ra_per_game']:.2f} vs {away_30['ra_per_game']:.2f}，{_fmt_signed(a['def_delta'])})",
            "",
            "> 規則：|Δ| ≥ 0.5 才標箭頭。攻↑ = RS 上升；守↓ = RA 上升（防守變差）。",
            "",
        ]
    else:
        lines += ["## 趨勢（近 10 vs 近 30）", "- —（資料不足）", ""]

    # ========== 當前系列賽（soft） ==========
    home_games = home_recent.get("games", [])
    if home_games:
        away_team_name = away.get("team", "")
        series = detect_current_series(home_games, away_team_name, game_date)
        lines.append(f"## 當前系列賽 ({away_abbr} @ {home_abbr})")
        if not series:
            lines += [
                f"- G1 ({game_date[5:]}): 本場",
                "- 系列累計: 本系列首戰，無前場",
                "",
            ]
        else:
            home_wins = 0
            away_wins = 0
            for g in series:
                if g.get("is_home"):
                    home_score, away_score = g.get("team_score", 0), g.get("opp_score", 0)
                    winner_abbr = home_abbr if g.get("is_winner") else away_abbr
                else:
                    home_score, away_score = g.get("opp_score", 0), g.get("team_score", 0)
                    winner_abbr = away_abbr if g.get("is_winner") else home_abbr
                if winner_abbr == home_abbr:
                    home_wins += 1
                else:
                    away_wins += 1
                lines.append(
                    f"- {g['label']} ({g['date'][5:]}): {home_abbr} {home_score}-{away_score} {away_abbr} → {winner_abbr} 勝"
                )
            this_g = f"G{len(series) + 1}"
            lines.append(f"- {this_g} ({game_date[5:]}): 本場")
            lines.append(f"- 系列累計: **{home_abbr} {home_wins}-{away_wins} {away_abbr}**")
            lines.append("")

    # ========== Streak 脈絡（soft） ==========
    h_streak = home_recent.get("streak") or 0
    a_streak = away_recent.get("streak") or 0
    h_ctx = format_streak_context(home_games, h_streak) if home_games else None
    away_games = away_recent.get("games", [])
    a_ctx = format_streak_context(away_games, a_streak) if away_games else None
    if h_ctx or a_ctx:
        lines.append("## Streak 脈絡")
        if h_ctx:
            lines.append(f"- {home_abbr} {_fmt_streak(h_streak)}: {h_ctx}")
        if a_ctx:
            lines.append(f"- {away_abbr} {_fmt_streak(a_streak)}: {a_ctx}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def fetch_schedule(date: str, team_id: int = None, hydrate: str = "probablePitcher(note)"):
    """呼叫 MLB Stats API schedule endpoint"""
    params = {"sportId": 1, "date": date, "hydrate": hydrate}
    if team_id:
        params["teamId"] = team_id
    resp = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def find_game(schedule_data: dict, team_id: int) -> dict | None:
    """從 schedule 中找到指定球隊的比賽"""
    for date_entry in schedule_data.get("dates", []):
        for game in date_entry.get("games", []):
            home_id = game["teams"]["home"]["team"]["id"]
            away_id = game["teams"]["away"]["team"]["id"]
            if team_id in (home_id, away_id):
                return game
    return None


def extract_game_info(game: dict) -> dict:
    """從 game object 提取比賽資訊。

    Returns dict with home / away sub-dicts. Each side has:
        - team (str): team name
        - team_id (int): MLBAM team ID
        - probable_pitcher (str): pitcher full name; "TBD" when unannounced
        - probable_pitcher_id (int | None): MLBAM ID; None when probablePitcher not yet announced
    """
    home = game["teams"]["home"]
    away = game["teams"]["away"]
    return {
        "gamePk": game["gamePk"],
        "date": game["gameDate"],
        "status": game["status"]["abstractGameState"],
        "venue": game["venue"]["name"],
        "home": {
            "team": home["team"]["name"],
            "team_id": home["team"]["id"],
            "probable_pitcher": home.get("probablePitcher", {}).get("fullName", "TBD"),
            "probable_pitcher_id": home.get("probablePitcher", {}).get("id"),
        },
        "away": {
            "team": away["team"]["name"],
            "team_id": away["team"]["id"],
            "probable_pitcher": away.get("probablePitcher", {}).get("fullName", "TBD"),
            "probable_pitcher_id": away.get("probablePitcher", {}).get("id"),
        },
    }


def fetch_recent_games(team_id: int, before_date: str, num_days: int = 20, max_games: int = 10) -> list[dict]:
    """取得指定球隊在 before_date 前 num_days 天內已完成的比賽"""
    end_dt = datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=1)
    start_dt = datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=num_days)

    params = {
        "sportId": 1,
        "teamId": team_id,
        "startDate": start_dt.strftime("%Y-%m-%d"),
        "endDate": end_dt.strftime("%Y-%m-%d"),
        "hydrate": "linescore",
    }
    resp = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game["status"]["abstractGameState"] != "Final":
                continue
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            is_home = home["team"]["id"] == team_id
            team_side = home if is_home else away
            opp_side = away if is_home else home
            games.append({
                "date": game["gameDate"][:10],
                "is_home": is_home,
                "opponent": opp_side["team"]["name"],
                "team_score": team_side.get("score") or 0,
                "opp_score": opp_side.get("score") or 0,
                "is_winner": team_side.get("isWinner", False),
            })

    games.sort(key=lambda g: g["date"], reverse=True)
    return games[:max_games]


def fetch_season_games(team_id: int, before_date: str, season_start: str = None) -> list[dict]:
    """取得指定球隊本季所有已完成的比賽（從開季到 before_date 前一天）"""
    end_dt = datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=1)
    if season_start is None:
        season_start = f"{end_dt.year}-03-20"  # MLB 開季通常在 3 月底
    start_dt = datetime.strptime(season_start, "%Y-%m-%d")

    params = {
        "sportId": 1,
        "teamId": team_id,
        "startDate": start_dt.strftime("%Y-%m-%d"),
        "endDate": end_dt.strftime("%Y-%m-%d"),
        "hydrate": "linescore",
    }
    resp = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game["status"]["abstractGameState"] != "Final":
                continue
            # 僅計算例行賽
            if game.get("gameType", "R") != "R":
                continue
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            is_home = home["team"]["id"] == team_id
            team_side = home if is_home else away
            opp_side = away if is_home else home
            games.append({
                "date": game["gameDate"][:10],
                "is_home": is_home,
                "opponent": opp_side["team"]["name"],
                "team_score": team_side.get("score") or 0,
                "opp_score": opp_side.get("score") or 0,
                "is_winner": team_side.get("isWinner", False),
            })

    games.sort(key=lambda g: g["date"], reverse=True)
    return games


def compute_recent_stats(games: list[dict]) -> dict:
    """計算近期戰績統計"""
    if not games:
        return {"record": "0-0", "wins": 0, "losses": 0, "rs_per_game": 0, "ra_per_game": 0, "run_diff": 0, "streak": 0, "games": []}

    wins = sum(1 for g in games if g["is_winner"])
    losses = len(games) - wins
    total_rs = sum(g["team_score"] for g in games)
    total_ra = sum(g["opp_score"] for g in games)

    # 連勝/連敗
    streak = 0
    streak_win = games[0]["is_winner"]
    for g in games:
        if g["is_winner"] == streak_win:
            streak += 1
        else:
            break
    if not streak_win:
        streak = -streak

    return {
        "record": f"{wins}-{losses}",
        "wins": wins,
        "losses": losses,
        "rs_per_game": round(total_rs / len(games), 2),
        "ra_per_game": round(total_ra / len(games), 2),
        "run_diff": total_rs - total_ra,
        "streak": streak,
        "games": games,
    }


def fetch_series_prev(team_id: int, opponent_id: int, game_date: str) -> dict | None:
    """檢查同系列賽前場比分"""
    prev_date = (datetime.strptime(game_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    params = {
        "sportId": 1,
        "teamId": team_id,
        "date": prev_date,
        "hydrate": "linescore",
    }
    resp = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game["status"]["abstractGameState"] != "Final":
                continue
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            home_id = home["team"]["id"]
            away_id = away["team"]["id"]
            # 確認是同一組對手
            if {home_id, away_id} != {team_id, opponent_id}:
                continue
            return {
                "date": prev_date,
                "home": home["team"]["name"],
                "away": away["team"]["name"],
                "home_score": home.get("score") or 0,
                "away_score": away.get("score") or 0,
                "winner": home["team"]["name"] if home.get("isWinner") else away["team"]["name"],
            }
    return None


def main():
    parser = argparse.ArgumentParser(description="Fetch MLB game data for analysis")
    parser.add_argument("--date", help="Game date (YYYY-MM-DD)")
    parser.add_argument("--team", help="Team name/abbreviation")
    parser.add_argument("--output", "-o", help="Output file path (default: print to stdout)")
    parser.add_argument("--test", action="store_true", help="Run with test data")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({
            "test": "OK",
            "message": "fetch_game_data test mode",
            "team_count": len(set(TEAM_MAP.values())),
            "sample": {"NYY": resolve_team_id("NYY"), "洋基": resolve_team_id("洋基")},
        }, indent=2, ensure_ascii=False))
        return

    if not args.date or not args.team:
        parser.error("--date and --team are required unless --test is specified")

    team_id = resolve_team_id(args.team)
    game_date = args.date

    # 1. 取當日賽程
    schedule = fetch_schedule(game_date)
    game = find_game(schedule, team_id)

    if not game:
        print(json.dumps({"error": f"No game found for team {args.team} on {game_date}"}, indent=2, ensure_ascii=False))
        sys.exit(1)

    game_info = extract_game_info(game)

    # 2. 取雙方多窗口戰績
    home_id = game_info["home"]["team_id"]
    away_id = game_info["away"]["team_id"]

    # 近 10 場（短期手感）
    home_recent = compute_recent_stats(fetch_recent_games(home_id, game_date))
    away_recent = compute_recent_stats(fetch_recent_games(away_id, game_date))

    # 本季全部（賽季水準）
    home_season_games = fetch_season_games(home_id, game_date)
    away_season_games = fetch_season_games(away_id, game_date)
    home_season = compute_recent_stats(home_season_games)
    away_season = compute_recent_stats(away_season_games)

    # 近 30 場（中期趨勢）— 如果不到 30 場則等於本季全部
    home_30_games = home_season_games[:30] if len(home_season_games) >= 30 else home_season_games
    away_30_games = away_season_games[:30] if len(away_season_games) >= 30 else away_season_games
    home_recent_30 = compute_recent_stats(home_30_games)
    away_recent_30 = compute_recent_stats(away_30_games)

    # 3. 檢查系列賽前場
    series_prev = fetch_series_prev(home_id, away_id, game_date)

    result = {
        "game": game_info,
        "home_recent": home_recent,
        "away_recent": away_recent,
        "home_recent_30": home_recent_30,
        "away_recent_30": away_recent_30,
        "home_season": home_season,
        "away_season": away_season,
        "home_season_games_count": len(home_season_games),
        "away_season_games_count": len(away_season_games),
        "series_prev": series_prev,
    }

    json_output = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Saved to {args.output}", file=sys.stderr)

        # 額外輸出 summary md（同目錄 game_data_summary.md）
        from pathlib import Path
        summary_path = Path(args.output).parent / "game_data_summary.md"
        try:
            summary_md = format_summary_md(result)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_md)
            print(f"Saved summary to {summary_path}", file=sys.stderr)
        except ValueError as e:
            print(f"Skipped summary (data incomplete): {e}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
