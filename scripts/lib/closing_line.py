"""Find closing-line snapshot for a single game from flat odds/odds_snapshots/.

'Closing' = last Pinnacle pre-game snapshot whose snapshot_time_utc < commence_utc.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


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
    """Find latest pre-game snapshot containing this matchup.

    Returns (game_dict, snapshot_filename) or (None, None) if no pre-game snapshot.
    `game_dict` is the inner `games[]` entry, with `snapshot_time_et` injected.
    """
    snapshots_dir = Path(snapshots_dir)
    candidates: list[tuple[datetime, dict, str]] = []

    for f in sorted(snapshots_dir.glob(f"{date}_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        snap_ts = _parse_iso_utc(data.get("snapshot_time_utc", ""))
        if snap_ts is None:
            continue
        for g in data.get("games", []):
            if g.get("home_team") != home_team or g.get("away_team") != away_team:
                continue
            commence_ts = _parse_iso_utc(g.get("commence_utc", ""))
            if commence_ts is None or snap_ts >= commence_ts:
                continue  # in-play / post-game
            g_copy = dict(g)
            g_copy["snapshot_time_et"] = data.get("snapshot_time_et", "")
            g_copy["snapshot_time_utc"] = data.get("snapshot_time_utc", "")
            candidates.append((snap_ts, g_copy, f.name))

    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    _, game_dict, filename = candidates[-1]
    return game_dict, filename


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
