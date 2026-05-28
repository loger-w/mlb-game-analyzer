import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.diagnostic import (
    select_failure_cases,
    extract_main_signal,
)

FIXTURES = Path(__file__).parent / "fixtures" / "backtest"


def _df_with_failures():
    return pd.DataFrame([
        {"date": "2026-05-02", "matchup": "BAL@NYY",
         "skill_direction": "HOME", "skill_confidence": "MEDIUM", "skill_confidence_pct": None,
         "skill_total": 8.5, "actual_winner": "AWAY", "actual_total": 13,
         "closing_missing": False, "result_missing": False,
         "dossier_path": "analysis-data/2026-05-02/BAL@NYY/dossier.md"},
        {"date": "2026-05-02", "matchup": "MIL@WSH",
         "skill_direction": "AWAY", "skill_confidence": "HIGH", "skill_confidence_pct": None,
         "skill_total": 9.0, "actual_winner": "HOME", "actual_total": 8,
         "closing_missing": False, "result_missing": False,
         "dossier_path": "analysis-data/2026-05-02/MIL@WSH/dossier.md"},
        # LOW confidence miss — should NOT appear in direction miss list
        {"date": "2026-05-03", "matchup": "X@Y",
         "skill_direction": "HOME", "skill_confidence": "LOW", "skill_confidence_pct": None,
         "skill_total": 7.0, "actual_winner": "AWAY", "actual_total": 7,
         "closing_missing": False, "result_missing": False,
         "dossier_path": "analysis-data/2026-05-03/X@Y/dossier.md"},
        # Big total miss
        {"date": "2026-05-04", "matchup": "A@B",
         "skill_direction": "HOME", "skill_confidence": "MEDIUM", "skill_confidence_pct": None,
         "skill_total": 7.0, "actual_winner": "HOME", "actual_total": 18,
         "closing_missing": False, "result_missing": False,
         "dossier_path": "analysis-data/2026-05-04/A@B/dossier.md"},
    ])


def test_select_failure_cases_filters_low_confidence_dir_miss():
    df = _df_with_failures()
    cases = select_failure_cases(df, top_total_miss=10)
    matchups = set(cases["matchup"])
    # Direction misses at MED/HIGH conf + top total misses
    assert "BAL@NYY" in matchups  # MED conf direction miss
    assert "MIL@WSH" in matchups  # HIGH conf direction miss
    assert "A@B" in matchups       # huge total miss
    # LOW conf direction miss only enters if it's also a top-N total miss
    # Here |7-7|=0 so it shouldn't enter
    assert "X@Y" not in matchups


def test_select_failure_cases_marks_dual_failures():
    df = _df_with_failures()
    cases = select_failure_cases(df, top_total_miss=10)
    bal_row = cases[cases["matchup"] == "BAL@NYY"].iloc[0]
    # |8.5 - 13| = 4.5 — large total miss + direction miss
    assert bal_row["dual_failure"] == True


def test_extract_main_signal_strips_markdown():
    summary_text = "## 整體判斷\n\n- **方向（基本面）**：**Yankees 中度偏優**。三大核心訊號疊加：...\n"
    signal = extract_main_signal(summary_text)
    assert "Yankees" in signal
    assert "**" not in signal  # markdown stripped
    assert len(signal) <= 50


def test_select_failure_cases_uses_pct_confidence():
    """Rows with skill_confidence_pct but no skill_confidence should still be eligible
    for direction-miss listing if pct maps to MEDIUM/HIGH bucket."""
    df = pd.DataFrame([
        {"date": "2026-05-15", "matchup": "X@Y",
         "skill_direction": "HOME", "skill_confidence": None, "skill_confidence_pct": 0.65,
         "skill_total": 8.0, "actual_winner": "AWAY", "actual_total": 5,
         "closing_missing": False, "result_missing": False,
         "dossier_path": "analysis-data/2026-05-15/X@Y/dossier.md"},
    ])
    cases = select_failure_cases(df, top_total_miss=10)
    # 0.65 → MEDIUM bucket; direction miss should be picked up
    assert "X@Y" in set(cases["matchup"])
