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


def pin_rec_snapshot(
    snapshot_game: dict,
    commence_utc: str,
    source_filename: str,
    snapshot_time_et: str,
    snapshot_time_utc: str,
) -> dict:
    """Convert one game's Pinnacle bookmaker block into canonical 3-market shape.

    Missing markets yield null at that key. Returns the full block
    as documented in spec §5.3.
    """
    pinnacle = snapshot_game.get("bookmakers", {}).get("pinnacle", {}) or {}

    # minutes_before_first_pitch
    try:
        commence_dt = datetime.fromisoformat(commence_utc.replace("Z", "+00:00"))
        snap_dt = datetime.fromisoformat(snapshot_time_utc.replace("Z", "+00:00"))
        minutes_before = int((commence_dt - snap_dt).total_seconds() // 60)
    except (ValueError, TypeError, AttributeError):
        minutes_before = None

    home_name = snapshot_game.get("home_team")
    away_name = snapshot_game.get("away_team")

    def _line(dec: float, implied: float) -> dict:
        return {
            "decimal": round(dec, 4),
            "american": decimal_to_american(dec),
            "implied_pct": round(implied, 2),
        }

    # ML
    ml_block = None
    ml = pinnacle.get("ml") or {}
    if home_name in ml and away_name in ml:
        ml_block = {
            "home": _line(ml[home_name]["odds"], ml[home_name]["implied_pct"]),
            "away": _line(ml[away_name]["odds"], ml[away_name]["implied_pct"]),
        }

    # OU
    ou_block = None
    ou = pinnacle.get("ou") or {}
    if "Over" in ou and "Under" in ou:
        ou_block = {
            "point": ou["Over"].get("point"),
            "over":  _line(ou["Over"]["odds"],  ou["Over"]["implied_pct"]),
            "under": _line(ou["Under"]["odds"], ou["Under"]["implied_pct"]),
        }

    # RL — treat a missing `point` on either side as a malformed market (null the block)
    # rather than defaulting to 0, which would silently mislabel favorite_side.
    rl_block = None
    rl = pinnacle.get("rl") or {}
    if home_name in rl and away_name in rl:
        home_point = rl[home_name].get("point")
        away_point = rl[away_name].get("point")
        if home_point is not None and away_point is not None:
            favorite_side = "HOME" if home_point < 0 else "AWAY"
            home_line = _line(rl[home_name]["odds"], rl[home_name]["implied_pct"])
            home_line["point"] = home_point
            away_line = _line(rl[away_name]["odds"], rl[away_name]["implied_pct"])
            away_line["point"] = away_point
            rl_block = {"favorite_side": favorite_side, "home": home_line, "away": away_line}

    return {
        "source": source_filename,
        "snapshot_time_et": snapshot_time_et,
        "snapshot_time_utc": snapshot_time_utc,
        "commence_utc": commence_utc,
        "minutes_before_first_pitch": minutes_before,
        "ml": ml_block,
        "ou": ou_block,
        "rl": rl_block,
    }


def compute_bet_placed(kelly_market_block: Optional[dict]) -> bool:
    """Kelly market sub-block has units > 0 → True; else False."""
    if not kelly_market_block:
        return False
    units = kelly_market_block.get("units", 0)
    try:
        return float(units) > 0
    except (TypeError, ValueError):
        return False


def _side_cents_delta(start_pin: dict, end_pin: dict, market: str, side_key: str) -> Optional[int]:
    """Return end_american - start_american for the given market/side. None if either missing."""
    if not start_pin or not end_pin:
        return None
    s_m = start_pin.get(market)
    e_m = end_pin.get(market)
    if not s_m or not e_m:
        return None
    s_side = s_m.get(side_key)
    e_side = e_m.get(side_key)
    if not s_side or not e_side:
        return None
    return int(e_side["american"] - s_side["american"])


def _interval_deltas(start_pin: Optional[dict], end_pin: Optional[dict]) -> Optional[dict]:
    if start_pin is None or end_pin is None:
        return None
    ml_home = _side_cents_delta(start_pin, end_pin, "ml", "home")
    rl_home = _side_cents_delta(start_pin, end_pin, "rl", "home")
    ou_cents = _side_cents_delta(start_pin, end_pin, "ou", "over")
    ou_point_delta = 0.0
    if start_pin.get("ou") and end_pin.get("ou"):
        try:
            ou_point_delta = float(end_pin["ou"]["point"]) - float(start_pin["ou"]["point"])
        except (TypeError, ValueError):
            ou_point_delta = 0.0
    return {
        "ml_home_cents": ml_home if ml_home is not None else 0,
        "ou_cents": ou_cents if ou_cents is not None else 0,
        "ou_point_delta": round(ou_point_delta, 1),
        "rl_home_cents": rl_home if rl_home is not None else 0,
    }


def detect_line_movement(
    open_snap: Optional[dict],
    rec_snap: dict,
    close_snap: Optional[dict],
    recommended_direction: dict,
    steam_threshold_cents: int = 5,
) -> dict:
    """Compute open→rec and rec→close cents deltas, plus steam / RLM flags.

    Cents convention: positive = recommended side's American odds increased during
    the interval (price improved for the backer). Negative = price worsened.
    Flag logic uses open_to_rec on the recommended side only.
    """
    open_to_rec = _interval_deltas(open_snap, rec_snap)
    rec_to_close = _interval_deltas(rec_snap, close_snap)

    steam = False
    rlm = False
    if open_to_rec is not None:
        checks = []
        # ML
        if recommended_direction.get("ml") == "HOME":
            checks.append(open_to_rec["ml_home_cents"])
        elif recommended_direction.get("ml") == "AWAY":
            c = _side_cents_delta(open_snap, rec_snap, "ml", "away")
            if c is not None:
                checks.append(c)
        # RL
        if recommended_direction.get("rl") == "HOME":
            checks.append(open_to_rec["rl_home_cents"])
        elif recommended_direction.get("rl") == "AWAY":
            c = _side_cents_delta(open_snap, rec_snap, "rl", "away")
            if c is not None:
                checks.append(c)
        # OU
        if recommended_direction.get("ou") == "OVER":
            checks.append(open_to_rec["ou_cents"])
        elif recommended_direction.get("ou") == "UNDER":
            c = _side_cents_delta(open_snap, rec_snap, "ou", "under")
            if c is not None:
                checks.append(c)

        if checks:
            max_favor = max(checks)
            min_favor = min(checks)
            if max_favor >= steam_threshold_cents:
                steam = True
            if min_favor <= -steam_threshold_cents:
                rlm = True

    warnings = []
    if open_snap is None:
        warnings.append("no_open_snapshot")
    if close_snap is None:
        warnings.append("no_close_snapshot")

    return {
        "open_snapshot": open_snap.get("source") if open_snap else None,
        "rec_snapshot":  rec_snap.get("source"),
        "close_snapshot": close_snap.get("source") if close_snap else None,
        "open_to_rec": open_to_rec,
        "rec_to_close": rec_to_close,
        "flags": {"steam_toward_rec": steam, "rlm_suspected": rlm},
        "granularity_note": "4h snapshot cadence; sub-hour steam not detectable",
        "warnings": warnings,
    }
