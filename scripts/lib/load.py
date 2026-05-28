"""Integrate merged.json + signals.json + closing_line + result.json into a pandas DataFrame.

One row per game. Predictions are computed deterministically from the frozen
merged.json feature set via predict_with_formula() + predict(). Slice flags come
from signals.json. Marks parse_failed / closing_missing / result_missing flags
so downstream metrics can exclude them while CSV retains every row.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
SKILL_ROOT = SCRIPT_DIR.parent
ANALYSIS_DATA_DIR = SKILL_ROOT / "analysis-data"
SNAPSHOTS_DIR = SKILL_ROOT / "odds" / "odds_snapshots"

# Ensure scripts/ is importable so predict and scoring_formula can be imported
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from predict import predict
from scoring_formula import predict_with_formula
from lib.closing_line import find_closing_snapshot_for_game, extract_pinnacle_no_vig

# Matchup dir name: "BAL@NYY", optionally with -1 / -2 / -G2 doubleheader suffix
_MATCHUP_RE = re.compile(r"^([A-Z]{2,4})@([A-Z]{2,4})(?:-(?:G?\d+))?$")

CONFIDENCE_TO_PROB = {"LOW": 0.55, "MEDIUM": 0.62, "HIGH": 0.72}


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _matchup_to_abbrs(matchup_dir_name: str) -> tuple[Optional[str], Optional[str]]:
    """'BAL@NYY' or 'BAL@NYY-1' or 'BAL@NYY-G2' -> ('BAL', 'NYY')."""
    m = _MATCHUP_RE.match(matchup_dir_name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _read_game_data(matchup_dir: Path) -> Optional[dict]:
    gd = matchup_dir / "game_data.json"
    if not gd.exists():
        return None
    try:
        return json.loads(gd.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_result(matchup_dir: Path) -> Optional[dict]:
    rj = matchup_dir / "result.json"
    if not rj.exists():
        return None
    try:
        return json.loads(rj.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_dataframe_for_month(
    month: str,
    days_filter: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Build per-game DataFrame. `days_filter` subset of {"2026-05-02", ...}; None = all."""
    rows = []
    for date_dir in sorted(ANALYSIS_DATA_DIR.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith(month):
            continue
        if date_dir.name.endswith(".local-backup"):
            continue
        if days_filter is not None and date_dir.name not in days_filter:
            continue

        for matchup_dir in sorted(date_dir.iterdir()):
            if not matchup_dir.is_dir():
                continue
            away_abbr, home_abbr = _matchup_to_abbrs(matchup_dir.name)
            if not (away_abbr and home_abbr):
                continue

            row = _build_row(date_dir.name, matchup_dir, home_abbr, away_abbr)
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


def _build_row(date: str, matchup_dir: Path, home_abbr: str, away_abbr: str) -> Optional[dict]:
    game_data = _read_game_data(matchup_dir)
    if game_data is None:
        return None

    game_pk = game_data.get("game", {}).get("gamePk")
    home_team = game_data.get("game", {}).get("home", {}).get("team", "")
    away_team = game_data.get("game", {}).get("away", {}).get("team", "")

    # Determine dossier filename (doubleheader-aware)
    dossier_filename = "dossier.md"
    if not (matchup_dir / "summary.md").exists():
        g_summaries = sorted(matchup_dir.glob("summary-G*.md"))
        if g_summaries:
            suffix = g_summaries[0].stem[len("summary"):]  # e.g. "-G1"
            dossier_filename = f"dossier{suffix}.md"

    # Prediction — computed deterministically from frozen merged.json
    merged_json = _read_json(matchup_dir / "merged.json")
    if merged_json is None:
        return None  # merged.json required for deterministic prediction

    formula_pred = predict_with_formula(merged_json)
    pred = predict(formula_pred["home_score"], formula_pred["away_score"])

    skill_direction = pred["direction"]
    skill_total = pred["total"]
    skill_confidence_pct = pred["confidence_pct"]
    skill_confidence = pred["confidence_bucket"]
    skill_prob_mapped = skill_confidence_pct  # always available from predict()
    parse_failed = False

    # Slice flags — read from signals.json (fallback empty if missing)
    signals_data = _read_json(matchup_dir / "signals.json") or {"signals": []}
    signals = signals_data.get("signals", [])

    has_reverse_platoon = any(
        s["name"] == "reverse_platoon" and s.get("fired")
        for s in signals
    )
    # chain_break: fired + OPS-gap value (field: "value") >= 0.300
    has_chain_break_300 = any(
        s["name"] == "chain_break" and s.get("fired")
        and isinstance(s.get("value"), (int, float)) and s["value"] >= 0.300
        for s in signals
    )
    # core_il_count: fired + count (field: "value") >= 2
    has_bullpen_il_2plus = any(
        s["name"] == "core_il_count" and s.get("fired")
        and isinstance(s.get("value"), (int, float)) and s["value"] >= 2
        for s in signals
    )

    # Closing snapshot
    snap_game, snap_filename = find_closing_snapshot_for_game(
        snapshots_dir=SNAPSHOTS_DIR,
        date=date,
        home_team=home_team,
        away_team=away_team,
    )
    no_vig = extract_pinnacle_no_vig(snap_game) if snap_game else None
    closing_missing = no_vig is None

    # Result
    result = _read_result(matchup_dir)
    result_missing = result is None

    # Market favorite
    market_favorite = None
    market_favorite_winprob = None
    if no_vig:
        if no_vig["home_winprob_no_vig"] >= 0.5:
            market_favorite = "HOME"
            market_favorite_winprob = no_vig["home_winprob_no_vig"]
        else:
            market_favorite = "AWAY"
            market_favorite_winprob = no_vig["away_winprob_no_vig"]

    return {
        "date": date,
        "matchup": matchup_dir.name,
        "game_pk": game_pk,
        # Skill prediction (deterministic from merged.json)
        "skill_direction": skill_direction,
        "skill_total": skill_total,
        "skill_confidence": skill_confidence,
        "skill_confidence_pct": skill_confidence_pct,
        "skill_prob_mapped": skill_prob_mapped,
        # Market
        "market_home_winprob_no_vig": no_vig["home_winprob_no_vig"] if no_vig else None,
        "market_total_line": no_vig["total_line"] if no_vig else None,
        "market_favorite": market_favorite,
        "market_favorite_winprob": market_favorite_winprob,
        # Actual
        "actual_winner": result.get("winner") if result else None,
        "actual_total": result.get("total") if result else None,
        "actual_home_score": result.get("home_score") if result else None,
        "actual_away_score": result.get("away_score") if result else None,
        # Flags (from signals.json)
        "park_factor": merged_json.get("park_factor"),
        "has_reverse_platoon": has_reverse_platoon,
        "has_chain_break_300": has_chain_break_300,
        "has_bullpen_il_2plus": has_bullpen_il_2plus,
        # Status
        "parse_failed": parse_failed,
        "closing_missing": closing_missing,
        "closing_snapshot_ts": snap_filename or "",
        "result_missing": result_missing,
        "dossier_path": f"../{date}/{matchup_dir.name}/{dossier_filename}",
    }
