"""Find the fixed 12:00 ET snapshot for a single game from flat odds/odds_snapshots/.

Methodology choice: use the 12:00 ET snapshot of the ET game date as the market
baseline. Matches the future production workflow where the user triggers one
on-demand snapshot per game before fundamentals analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

SNAPSHOT_TIME = "12-00-ET"


def _parse_iso_utc(s: str) -> Optional[datetime]:
    """Parse ISO 8601 timestamp. Handle 'Z' suffix and '+00:00'."""
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def find_closing_snapshot_for_game(
    snapshots_dir: Path,
    date: str,
    home_team: str,
    away_team: str,
) -> tuple[Optional[dict], Optional[str]]:
    """Find the 12:00 ET snapshot entry for this matchup on the given ET date.

    Returns (game_dict, snapshot_filename) or (None, None) if the 12:00 ET file
    doesn't exist, the matchup isn't in it, or the game already started before
    12:00 ET (in-play safety).
    """
    snapshots_dir = Path(snapshots_dir)
    snap_file = snapshots_dir / f"{date}_{SNAPSHOT_TIME}.json"
    if not snap_file.exists():
        return None, None

    try:
        data = json.loads(snap_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None

    snap_ts = _parse_iso_utc(data.get("snapshot_time_utc", ""))
    if snap_ts is None:
        return None, None

    for g in data.get("games", []):
        if g.get("game_date_et") != date:
            continue
        if g.get("home_team") != home_team or g.get("away_team") != away_team:
            continue
        commence_ts = _parse_iso_utc(g.get("commence_utc", ""))
        if commence_ts is None or snap_ts >= commence_ts:
            continue  # game started before 12:00 ET (rare)
        g_copy = dict(g)
        g_copy["snapshot_time_et"] = data.get("snapshot_time_et", "")
        g_copy["snapshot_time_utc"] = data.get("snapshot_time_utc", "")
        return g_copy, snap_file.name

    return None, None


def extract_pinnacle_no_vig(game: dict) -> Optional[dict]:
    """Extract Pinnacle ML / Total no-vig probabilities + line from a snapshot game.

    Returns: {
        home_winprob_no_vig: float (0-1),
        away_winprob_no_vig: float (0-1),
        total_line: float,
        over_no_vig: float (0-1),
        under_no_vig: float (0-1),
    } or None if Pinnacle data unavailable.
    """
    pinn = game.get("bookmakers", {}).get("pinnacle")
    if not pinn:
        return None
    ml = pinn.get("ml", {})
    ou = pinn.get("ou", {})

    home_team = game.get("home_team")
    away_team = game.get("away_team")
    if not (home_team and away_team and home_team in ml and away_team in ml):
        return None

    over = ou.get("Over", {})
    under = ou.get("Under", {})
    if "no_vig_pct" not in over or "no_vig_pct" not in under:
        return None
    if "point" not in over:
        return None

    return {
        "home_winprob_no_vig": ml[home_team]["no_vig_pct"] / 100.0,
        "away_winprob_no_vig": ml[away_team]["no_vig_pct"] / 100.0,
        "total_line": float(over["point"]),
        "over_no_vig": over["no_vig_pct"] / 100.0,
        "under_no_vig": under["no_vig_pct"] / 100.0,
    }


def extract_pinnacle_rl_no_vig(game: dict) -> Optional[dict]:
    """抽 Pinnacle RL 的 home/away point + no-vig 機率。缺 → None。"""
    pinn = game.get("bookmakers", {}).get("pinnacle")
    if not pinn:
        return None
    rl = pinn.get("rl", {})
    home = game.get("home_team")
    away = game.get("away_team")
    h = rl.get(home, {})
    a = rl.get(away, {})
    if "no_vig_pct" not in h or "no_vig_pct" not in a:
        return None
    if "point" not in h or "point" not in a:
        return None
    return {
        "home_point": float(h["point"]), "home_no_vig": h["no_vig_pct"] / 100.0,
        "away_point": float(a["point"]), "away_no_vig": a["no_vig_pct"] / 100.0,
    }
