"""Tests for scripts/lib/load.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.load import build_dataframe_for_month, _matchup_to_abbrs


def test_matchup_to_abbrs_basic():
    assert _matchup_to_abbrs("BAL@NYY") == ("BAL", "NYY")
    assert _matchup_to_abbrs("ARI@SF") == ("ARI", "SF")


def test_matchup_to_abbrs_with_suffix():
    # Doubleheader suffix forms: -1, -2, -G2
    assert _matchup_to_abbrs("BAL@NYY-1") == ("BAL", "NYY")
    assert _matchup_to_abbrs("BAL@NYY-G2") == ("BAL", "NYY")
    assert _matchup_to_abbrs("STL@CIN-G2") == ("STL", "CIN")


def test_build_dataframe_real_data_smoke(tmp_path):
    """Smoke test against real 2026-05-02 data (1 day, expect ~12-14 rows)."""
    df = build_dataframe_for_month(month="2026-05", days_filter={"2026-05-02"})
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    required_cols = {
        "date", "matchup", "game_pk",
        "skill_direction", "skill_total", "skill_confidence", "skill_confidence_pct",
        "skill_prob_mapped",
        "market_home_winprob_no_vig", "market_total_line",
        "actual_winner", "actual_total",
        "parse_failed", "closing_missing", "result_missing",
        "park_factor", "has_reverse_platoon",
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_build_dataframe_marks_template_as_parse_failed():
    """All 5/25 summaries are template state; all rows should be parse_failed=True."""
    df = build_dataframe_for_month(month="2026-05", days_filter={"2026-05-25"})
    assert len(df) > 0, "Expected at least 1 row for 5/25"
    assert df["parse_failed"].all(), \
        f"Expected all rows to be parse_failed=True, got {df['parse_failed'].sum()}/{len(df)}"


def test_skill_prob_mapped_prefers_pct_when_available():
    """When confidence_pct is set, skill_prob_mapped should use that value;
    otherwise fall back to bucket mapping (LOW=0.55 / MEDIUM=0.62 / HIGH=0.72)."""
    # Use real data — pick a day where some rows have pct, some have bucket
    df = build_dataframe_for_month(month="2026-05", days_filter={"2026-05-02", "2026-05-15"})
    valid = df[~df["parse_failed"]]

    pct_rows = valid[valid["skill_confidence_pct"].notna()]
    bucket_only = valid[valid["skill_confidence_pct"].isna() & valid["skill_confidence"].notna()]

    # Both paths must be exercised in current data — fail loudly if data changes
    assert len(pct_rows) > 0, "Expected ≥1 row with confidence_pct set (post-5/4 format)"
    assert len(bucket_only) > 0, "Expected ≥1 row with only bucket confidence (pre-5/4 format)"

    # pct path
    for _, r in pct_rows.iterrows():
        assert r["skill_prob_mapped"] == r["skill_confidence_pct"]

    # bucket path
    expected_map = {"LOW": 0.55, "MEDIUM": 0.62, "HIGH": 0.72}
    for _, r in bucket_only.iterrows():
        assert r["skill_prob_mapped"] == expected_map[r["skill_confidence"]]
