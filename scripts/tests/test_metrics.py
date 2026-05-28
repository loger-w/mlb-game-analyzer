"""Tests for scripts/lib/metrics.py"""
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.metrics import (
    compute_direction_metrics,
    compute_total_metrics,
    compute_calibration,
    compute_slice_metrics,
    derive_direction_outcome,
)


def _mini_df():
    """10 games. 6 skill picks home, 4 picks away. Half direction-hit, half miss.
    Designed so hand-calc metrics are easy."""
    rows = []
    # Format: (skill_dir, market_home_p, actual_winner, skill_total, line, actual_total, conf)
    cases = [
        ("HOME", 0.60, "HOME", 9.0, 8.5, 10, "HIGH"),    # dir_hit, ou_hit, |err|=1
        ("HOME", 0.55, "AWAY", 8.0, 8.5, 5, "MEDIUM"),   # dir_miss, ou_hit, |err|=3
        ("HOME", 0.62, "HOME", 7.5, 8.0, 8, "MEDIUM"),   # dir_hit, push (actual=line), excluded
        ("HOME", 0.51, "HOME", 8.5, 8.5, 7, "LOW"),      # dir_hit, skill push, excluded
        ("HOME", 0.65, "AWAY", 9.5, 9.0, 11, "HIGH"),    # dir_miss, ou_hit (skill over, actual over)
        ("HOME", 0.58, "HOME", 10.0, 8.5, 9, "LOW"),     # dir_hit, ou_hit, |err|=1
        ("AWAY", 0.45, "AWAY", 7.0, 7.5, 6, "MEDIUM"),   # dir_hit, ou_hit (both under)
        ("AWAY", 0.48, "HOME", 6.5, 7.0, 9, "LOW"),      # dir_miss, ou_miss (skill under, actual over)
        ("AWAY", 0.40, "AWAY", 8.0, 7.5, 8, "HIGH"),     # dir_hit, ou_hit (skill over 7.5, actual 8 over 7.5 — NOT a push)
        ("AWAY", 0.47, "HOME", 7.0, 7.5, 6, "MEDIUM"),   # dir_miss, ou_hit (both under)
    ]
    for i, (sd, mhp, aw, st, ln, at, cf) in enumerate(cases):
        rows.append({
            "matchup": f"X{i}",
            "skill_direction": sd,
            "market_home_winprob_no_vig": mhp,
            "market_favorite": "HOME" if mhp >= 0.5 else "AWAY",
            "market_favorite_winprob": mhp if mhp >= 0.5 else 1 - mhp,
            "actual_winner": aw,
            "skill_total": st,
            "market_total_line": ln,
            "actual_total": at,
            "skill_confidence": cf,
            "skill_confidence_pct": {"LOW": 0.55, "MEDIUM": 0.62, "HIGH": 0.72}[cf],
            "park_factor": 100.0,
            "has_reverse_platoon": False,
            "has_chain_break_300": False,
            "has_bullpen_il_2plus": False,
            "closing_missing": False,
            "result_missing": False,
        })
    return pd.DataFrame(rows)


def test_direction_outcome_helper():
    df = _mini_df()
    out = derive_direction_outcome(df)
    # Game 0: skill=HOME, actual=HOME → hit
    assert out.loc[0, "direction_hit"] == True
    # Game 1: skill=HOME, actual=AWAY → miss
    assert out.loc[1, "direction_hit"] == False


def test_direction_metrics_skill_hit_rate():
    df = _mini_df()
    m = compute_direction_metrics(df)
    # 10 picks total, hits: rows 0,2,3,5,6,8 = 6 hits → 60%
    assert m["skill_n"] == 10
    assert m["skill_hit_rate"] == pytest.approx(0.6)


def test_direction_metrics_market_favorite_hit_rate():
    df = _mini_df()
    m = compute_direction_metrics(df)
    # Market favorite per row computed above
    assert m["market_n"] == 10
    assert m["market_hit_rate"] == pytest.approx(0.6)
    assert m["edge_pp"] == pytest.approx(0.0)


def test_total_metrics_excludes_pushes():
    df = _mini_df()
    m = compute_total_metrics(df)
    # MAE over all 10 rows
    assert m["total_mae"] == pytest.approx(1.3)
    # OU: exclude push rows (idx 2 actual=line=8.0, idx 3 skill=line=8.5)
    # Note: idx 8 actual=8, line=7.5 → NOT a push (8 != 7.5)
    # 8 remain, 7 hits → 7/8 = 0.875
    assert m["ou_n"] == 8
    assert m["ou_hit_rate"] == pytest.approx(7/8, rel=1e-3)


def test_calibration_returns_reliability_table():
    df = _mini_df()
    cal = compute_calibration(df)
    assert "reliability_table" in cal
    rt = cal["reliability_table"]
    assert set(rt["confidence"]) == {"LOW", "MEDIUM", "HIGH"}
    # 'mapped_p' must match canonical mapping
    for _, row in rt.iterrows():
        if row["confidence"] == "HIGH":
            assert row["mapped_p"] == 0.72
    # Brier score in result
    assert "brier_score" in cal
    assert 0.0 <= cal["brier_score"] <= 1.0


def test_calibration_with_pct_rows():
    """Rows with skill_confidence_pct (post-5/4 format) should be bucketed by pct range:
    LOW <0.58 / MEDIUM 0.58-0.67 / HIGH ≥0.67."""
    df = _mini_df()
    # Convert 3 rows to pct-only format
    df.loc[0, "skill_confidence"] = None
    df.loc[0, "skill_confidence_pct"] = 0.72  # → HIGH bucket

    df.loc[1, "skill_confidence"] = None
    df.loc[1, "skill_confidence_pct"] = 0.55  # → LOW bucket

    df.loc[2, "skill_confidence"] = None
    df.loc[2, "skill_confidence_pct"] = 0.63  # → MEDIUM bucket

    cal = compute_calibration(df)
    rt = cal["reliability_table"]
    # All 3 buckets should still appear (n counts include pct-derived buckets)
    bucket_n = {r["confidence"]: r["n"] for _, r in rt.iterrows()}
    assert bucket_n["LOW"] >= 1
    assert bucket_n["MEDIUM"] >= 1
    assert bucket_n["HIGH"] >= 1


def test_slice_metrics_park_factor_groups():
    df = _mini_df()
    df.loc[df.index[:3], "park_factor"] = 105.0  # 3 high-PF rows
    slices = compute_slice_metrics(df)
    assert "park_factor_high" in slices.index
    assert "park_factor_mid" in slices.index
