"""Unit tests for scripts/clv.py (P2 CLV infrastructure)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import shutil
from pathlib import Path

import pytest
from clv import (
    _find_earliest_snapshot_of_date,
    _find_latest_snapshot_before,
    compute_clv_cents,
    compute_clv_pct_no_vig,
    find_closing_snapshot,
    find_opening_snapshot,
)


class TestComputeClvCents:
    def test_beat_closing_favorite(self):
        # Rec HOME at -135 (1.74), closes at -179 (1.56) → we beat close by +44c
        assert compute_clv_cents(1.74, 1.56) == 44

    def test_beat_closing_underdog(self):
        # Rec AWAY at +128 (2.28), closes at +152 (2.52) → we beat by +24c (plus side, bigger = better? NO)
        # American: +128 vs +152 — the higher American number on plus side IS better (more payout)
        # But WE bet at +128 and line closed at +152, meaning we took the WORSE price → CLV = 128-152 = -24c
        # For beat: rec must be a BETTER price than close. Higher positive = better payout = better price.
        assert compute_clv_cents(2.28, 2.52) == -24

    def test_tied(self):
        assert compute_clv_cents(1.91, 1.91) == 0

    def test_lose_to_closing(self):
        # Rec HOME at -120 (1.83), closes at -110 (1.91) → rec price was worse → CLV negative
        assert compute_clv_cents(1.83, 1.91) == -10

    def test_cross_zero_decimal_2(self):
        # decimal 2.0 is exactly +100
        assert compute_clv_cents(2.0, 2.0) == 0

    def test_invalid_decimal(self):
        with pytest.raises(ValueError):
            compute_clv_cents(1.0, 1.5)


class TestComputeClvPctNoVig:
    def test_beat_ml_home(self):
        # rec: home 1.74, away 2.28 → no_vig_home ≈ 56.72%
        # close: home 1.56, away 2.52 → no_vig_home ≈ 61.77%
        # delta (close - rec) ≈ 5.05 → HOME backer beat market by ~5.05 pp
        result = compute_clv_pct_no_vig(1.74, 2.28, 1.56, 2.52)
        assert result == pytest.approx(5.05, abs=0.15)

    def test_tied(self):
        assert compute_clv_pct_no_vig(1.91, 1.91, 1.91, 1.91) == pytest.approx(0.0, abs=0.01)

    def test_lose(self):
        # Reverse: rec 1.56/2.52, close 1.74/2.28 → -5.05
        result = compute_clv_pct_no_vig(1.56, 2.52, 1.74, 2.28)
        assert result == pytest.approx(-5.05, abs=0.15)

    def test_high_vig_scenario(self):
        # Both books with heavy vig — no-vig extraction should still work
        # rec both -120 (1.83), close both -120 (1.83) → same → 0
        assert compute_clv_pct_no_vig(1.83, 1.83, 1.83, 1.83) == pytest.approx(0.0, abs=0.01)

    def test_rounded_to_2dp(self):
        result = compute_clv_pct_no_vig(1.74, 2.28, 1.56, 2.52)
        # Must be 2 decimal places
        assert abs(result - round(result, 2)) < 1e-9


@pytest.fixture
def temp_snapshot_dir(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    snap_dir = tmp_path / "odds_snapshots"
    snap_dir.mkdir()
    # Name the files to match the real cron convention
    shutil.copy(fixtures / "sample_snapshot_open.json",  snap_dir / "2026-04-18_00-00-ET.json")
    shutil.copy(fixtures / "sample_snapshot.json",       snap_dir / "2026-04-18_16-00-ET.json")
    shutil.copy(fixtures / "sample_snapshot_close.json", snap_dir / "2026-04-18_18-00-ET.json")
    return str(snap_dir)


class TestFindLatestSnapshotBefore:
    def test_picks_latest_before_cutoff(self, temp_snapshot_dir):
        # Open snap_time_utc=04:00, rec=20:00, close=22:10 UTC.
        # cutoff 22:00 UTC -> both open+rec pass, pick rec (latest)
        snap = _find_latest_snapshot_before(temp_snapshot_dir, "2026-04-18", "2026-04-18T22:00:00Z")
        assert snap is not None
        assert snap["snapshot_time_utc"] == "2026-04-18T20:00:00+00:00"

    def test_excludes_after_cutoff(self, temp_snapshot_dir):
        # cutoff 05:00 UTC -> only open (04:00) qualifies
        snap = _find_latest_snapshot_before(temp_snapshot_dir, "2026-04-18", "2026-04-18T05:00:00Z")
        assert snap["snapshot_time_utc"] == "2026-04-18T04:00:00+00:00"

    def test_no_match_returns_none(self, temp_snapshot_dir):
        snap = _find_latest_snapshot_before(temp_snapshot_dir, "2026-04-18", "2026-04-18T03:00:00Z")
        assert snap is None

    def test_wrong_date_returns_none(self, temp_snapshot_dir):
        snap = _find_latest_snapshot_before(temp_snapshot_dir, "2026-04-19", "2026-04-19T23:00:00Z")
        assert snap is None

    def test_missing_dir_returns_none(self, tmp_path):
        assert _find_latest_snapshot_before(str(tmp_path / "nope"), "2026-04-18", "2026-04-18T23:00:00Z") is None


class TestFindEarliestSnapshotOfDate:
    def test_picks_earliest(self, temp_snapshot_dir):
        snap = _find_earliest_snapshot_of_date(temp_snapshot_dir, "2026-04-18")
        assert snap["snapshot_time_utc"] == "2026-04-18T04:00:00+00:00"

    def test_no_snapshots_returns_none(self, tmp_path):
        (tmp_path / "odds_snapshots").mkdir()
        assert _find_earliest_snapshot_of_date(str(tmp_path / "odds_snapshots"), "2026-04-18") is None


class TestFindClosingSnapshot:
    def test_closing_for_cubs_game(self, temp_snapshot_dir):
        # Cubs game commence 23:00 UTC -> closing = latest before that = close snap at 22:10 UTC
        snap = find_closing_snapshot("2026-04-18T23:00:00Z", "2026-04-18", temp_snapshot_dir)
        assert snap["snapshot_time_utc"] == "2026-04-18T22:10:00+00:00"

    def test_closing_for_earlier_game(self, temp_snapshot_dir):
        # Orioles game commence 22:11 UTC -> closing = latest before = 20:00 UTC (rec), NOT 22:10
        snap = find_closing_snapshot("2026-04-18T22:11:00Z", "2026-04-18", temp_snapshot_dir)
        assert snap["snapshot_time_utc"] == "2026-04-18T20:00:00+00:00"


class TestFindOpeningSnapshot:
    def test_opening(self, temp_snapshot_dir):
        snap = find_opening_snapshot("2026-04-18", temp_snapshot_dir)
        assert snap["snapshot_time_utc"] == "2026-04-18T04:00:00+00:00"
