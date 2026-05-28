"""讀 features.json(v2)+ result.json → 每場一列 DataFrame。預測值已凍結,不重算。"""
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
SKILL_ROOT = SCRIPT_DIR.parent
ANALYSIS_DATA_DIR = SKILL_ROOT / "analysis-data"

_MATCHUP_RE = re.compile(r"^([A-Z]{2,4})@([A-Z]{2,4})(?:-(?:G?\d+))?$")


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _build_row(date: str, matchup_dir: Path) -> Optional[dict]:
    feats = _read_json(matchup_dir / "features.json")
    if feats is None or feats.get("schema_version") != 2:
        return None
    model = feats.get("model", {})
    odds = feats.get("odds") or {}
    edges = feats.get("edges", {})
    result = _read_json(matchup_dir / "result.json")

    rl = odds.get("rl") or {}
    total = odds.get("total") or {}
    actual_margin = actual_total = None
    if result is not None:
        actual_margin = result.get("home_score", 0) - result.get("away_score", 0)
        actual_total = result.get("total")

    return {
        "date": date, "matchup": matchup_dir.name,
        "game_pk": feats.get("game", {}).get("game_pk"),
        # model
        "mu_total": model.get("mu_total"),
        "p_home_cover_rl": model.get("p_home_cover_rl"),
        "p_over": model.get("p_over"),
        "p_home_ml": model.get("p_home_ml"),
        # market
        "rl_home_point": rl.get("home_point"),
        "rl_home_no_vig": rl.get("home_no_vig"),
        "total_line": total.get("line"),
        "over_no_vig": total.get("over_no_vig"),
        # edges
        "home_rl_pp": edges.get("home_rl_pp"),
        "over_pp": edges.get("over_pp"),
        # actual
        "actual_winner": result.get("winner") if result else None,
        "actual_total": actual_total,
        "actual_margin": actual_margin,
        # status
        "odds_missing": not bool(odds),
        "result_missing": result is None,
    }


def build_dataframe_for_month(month: str, days_filter: Optional[set] = None) -> pd.DataFrame:
    rows = []
    for date_dir in sorted(ANALYSIS_DATA_DIR.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith(month):
            continue
        if date_dir.name.endswith(".local-backup"):
            continue
        if days_filter is not None and date_dir.name not in days_filter:
            continue
        for matchup_dir in sorted(date_dir.iterdir()):
            if not matchup_dir.is_dir() or not _MATCHUP_RE.match(matchup_dir.name):
                continue
            row = _build_row(date_dir.name, matchup_dir)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)
