"""P2 — CLV Infrastructure + M5 line movement.

Pure-function helpers for Closing Line Value computation, snapshot discovery,
and line-movement detection. No file I/O side effects beyond snapshot reads.

Spec: docs/superpowers/specs/2026-04-18-p2-clv-infra-design.md
"""
from __future__ import annotations

import glob
import json
import os
import re
from datetime import datetime
from typing import Optional

from odds_analyzer import decimal_to_american


_SNAPSHOT_FILENAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-ET\.json$")


def compute_clv_cents(rec_decimal: float, close_decimal: float) -> int:
    """American cents difference: rec - close. Positive = beat closing.

    For both sides (favorite negative American, underdog positive American),
    a higher American number means a better price for the bettor, so
    american(rec) - american(close) correctly reports beat (positive) / lose (negative).
    """
    rec_am = decimal_to_american(rec_decimal)
    close_am = decimal_to_american(close_decimal)
    return int(round(rec_am - close_am))


def compute_clv_pct_no_vig(
    rec_side_dec: float,
    rec_other_dec: float,
    close_side_dec: float,
    close_other_dec: float,
) -> float:
    """No-vig implied probability delta (close - rec) in percentage points.

    Positive = rec side was priced below closing's true estimate → beat.

    For each snapshot, compute no-vig prob of the bet side by dividing its raw
    implied by the sum of both sides' raw implied (strips the book's hold).
    """
    rec_raw = (1.0 / rec_side_dec, 1.0 / rec_other_dec)
    rec_no_vig = rec_raw[0] / (rec_raw[0] + rec_raw[1])
    close_raw = (1.0 / close_side_dec, 1.0 / close_other_dec)
    close_no_vig = close_raw[0] / (close_raw[0] + close_raw[1])
    delta_pct = (close_no_vig - rec_no_vig) * 100
    return round(delta_pct, 2)


def _iter_snapshots_for_date(snapshot_dir: str, game_date_et: str):
    """Yield (snap_time_dt, snap_dict) for all snapshots that include game_date_et."""
    if not os.path.isdir(snapshot_dir):
        return
    for path in glob.glob(os.path.join(snapshot_dir, "*.json")):
        name = os.path.basename(path)
        if not _SNAPSHOT_FILENAME_RE.match(name):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                snap = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not any(g.get("game_date_et") == game_date_et for g in snap.get("games", [])):
            continue
        try:
            snap_dt = datetime.fromisoformat(snap["snapshot_time_utc"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        yield snap_dt, snap


def _find_latest_snapshot_before(
    snapshot_dir: str,
    game_date_et: str,
    cutoff_utc: str,
) -> Optional[dict]:
    """Newest snapshot with snapshot_time_utc < cutoff_utc and containing game_date_et."""
    try:
        cutoff_dt = datetime.fromisoformat(cutoff_utc.replace("Z", "+00:00"))
    except ValueError:
        return None
    candidates = [(dt, s) for dt, s in _iter_snapshots_for_date(snapshot_dir, game_date_et) if dt < cutoff_dt]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _find_earliest_snapshot_of_date(
    snapshot_dir: str,
    game_date_et: str,
) -> Optional[dict]:
    """Earliest snapshot containing game_date_et."""
    candidates = list(_iter_snapshots_for_date(snapshot_dir, game_date_et))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def find_closing_snapshot(
    commence_utc: str,
    game_date_et: str,
    snapshot_dir: str = "odds_snapshots",
) -> Optional[dict]:
    """Semantic wrapper: closing = latest snapshot before game start."""
    return _find_latest_snapshot_before(snapshot_dir, game_date_et, commence_utc)


def find_opening_snapshot(
    game_date_et: str,
    snapshot_dir: str = "odds_snapshots",
) -> Optional[dict]:
    """Semantic wrapper: opening = earliest snapshot of the game day."""
    return _find_earliest_snapshot_of_date(snapshot_dir, game_date_et)
