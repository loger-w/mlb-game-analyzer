#!/usr/bin/env python3
"""MLB Results Fetcher — 抓 MLB Stats API Final 比分 → 寫 per-game result.json

用法：
  python scripts/fetch_results.py --date 2026-05-02
  python scripts/fetch_results.py --month 2026-05
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
ANALYSIS_DATA_DIR = SKILL_ROOT / "analysis-data"

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def fetch_final_scores(date: str) -> list[dict]:
    """Fetch all Final regular-season games on date from MLB Schedule API.

    Returns list of dicts: {game_pk, home_team, away_team, home_score, away_score}.
    """
    params = {"sportId": 1, "date": date, "hydrate": "linescore"}
    resp = requests.get(MLB_SCHEDULE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    out = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            if g.get("gameType") != "R":
                continue
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            out.append({
                "game_pk": g.get("gamePk"),
                "home_team": home.get("team", {}).get("name", ""),
                "away_team": away.get("team", {}).get("name", ""),
                "home_score": home.get("score", 0),
                "away_score": away.get("score", 0),
            })
    return out


def build_result_record(raw: dict) -> dict:
    """Convert MLB API row → result.json schema per spec §2.

    Returns dict with winner ∈ {"HOME", "AWAY", "TIE"}.
    """
    home = raw["home_score"]
    away = raw["away_score"]
    return {
        "game_pk": raw["game_pk"],
        "winner": "HOME" if home > away else ("TIE" if home == away else "AWAY"),
        "final_score": [home, away],
        "home_score": home,
        "away_score": away,
        "total": home + away,
        "status": "Final",
        "postponed": False,
    }


def _read_game_pk(matchup_dir: Path) -> Optional[int]:
    """Read game_pk from game_data.json (old: game.gamePk) or features.json (new: game.game_pk)."""
    for fname, key in (("game_data.json", "gamePk"), ("features.json", "game_pk")):
        f = matchup_dir / fname
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        pk = data.get("game", {}).get(key)
        if pk is not None:
            return pk
    return None


def find_matchup_dir_by_pk(date: str, game_pk: int) -> Optional[Path]:
    """Locate analysis-data/{date}/{matchup}/ by exact game_pk.

    Reads game_data.json (old pipeline) or features.json (new pipeline). Unambiguous
    for doubleheaders. Returns None if no matchup dir has matching game_pk.
    """
    date_dir = ANALYSIS_DATA_DIR / date
    if not date_dir.is_dir():
        return None
    for sub in date_dir.iterdir():
        if sub.is_dir() and _read_game_pk(sub) == game_pk:
            return sub
    return None


def write_result(matchup_dir: Path, record: dict) -> Path:
    out = matchup_dir / "result.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def process_date(date: str) -> dict:
    """Fetch & write all results for one date. Returns {date, fetched, matched, missing} dict.

    Match strictly by game_pk (unambiguous for doubleheaders). Games with no matching
    matchup dir are reported as missing rather than falling back to team-name match
    (which would overwrite the wrong dir for doubleheader games).
    """
    scores = fetch_final_scores(date)
    matched = 0
    missing = []
    for raw in scores:
        matchup_dir = find_matchup_dir_by_pk(date, raw["game_pk"])
        if matchup_dir is None:
            missing.append(f"{raw['away_team']}@{raw['home_team']} (game_pk={raw['game_pk']})")
            continue
        record = build_result_record(raw)
        write_result(matchup_dir, record)
        matched += 1
    return {"date": date, "fetched": len(scores), "matched": matched, "missing": missing}


def process_month(month: str) -> list[dict]:
    """Process every date directory under analysis-data/ matching month prefix."""
    summaries = []
    for d in sorted(ANALYSIS_DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith(month):
            continue
        if d.name.endswith(".local-backup"):
            continue
        summaries.append(process_date(d.name))
    return summaries


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="YYYY-MM-DD")
    g.add_argument("--month", help="YYYY-MM")
    args = ap.parse_args()

    if args.date:
        summaries = [process_date(args.date)]
    else:
        summaries = process_month(args.month)

    for s in summaries:
        miss_note = f" (matchups not found: {', '.join(s['missing'])})" if s["missing"] else ""
        print(f"{s['date']}: fetched={s['fetched']} matched={s['matched']}{miss_note}")


if __name__ == "__main__":
    main()
