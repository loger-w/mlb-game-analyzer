"""Tests for snapshot loading and team-name resolution in predict.py."""
import sys
import os
import json
import shutil
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predict import load_closest_snapshot

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _make_snapshot_dir(tmpdir):
    """Copy fixtures into a temp snapshot dir with expected filename format."""
    shutil.copy(
        os.path.join(FIXTURES, "sample_snapshot.json"),
        os.path.join(tmpdir, "2026-04-18_16-00-ET.json"),
    )
    shutil.copy(
        os.path.join(FIXTURES, "sample_snapshot_earlier.json"),
        os.path.join(tmpdir, "2026-04-18_12-00-ET.json"),
    )


def test_load_closest_picks_newest_before_gametime():
    """game_start 19:00 ET → should pick 16:00 ET (newest before start)."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_snapshot_dir(tmp)
        snap = load_closest_snapshot(
            game_date_et="2026-04-18",
            game_start_utc="2026-04-18T23:00:00Z",
            snapshot_dir=tmp,
        )
        assert snap is not None
        assert snap["snapshot_time_et"] == "2026-04-18 16:00 ET"


def test_load_closest_ignores_snapshots_after_gametime():
    """game_start 15:00 ET → 16:00 ET snapshot 在 start 之後，只能用 12:00 ET。"""
    with tempfile.TemporaryDirectory() as tmp:
        _make_snapshot_dir(tmp)
        snap = load_closest_snapshot(
            game_date_et="2026-04-18",
            game_start_utc="2026-04-18T19:00:00Z",  # 15:00 ET
            snapshot_dir=tmp,
        )
        assert snap is not None
        assert snap["snapshot_time_et"] == "2026-04-18 12:00 ET"


def test_load_closest_no_snapshots_returns_none():
    """Empty snapshot dir → None."""
    with tempfile.TemporaryDirectory() as tmp:
        snap = load_closest_snapshot(
            game_date_et="2026-04-18",
            game_start_utc="2026-04-18T23:00:00Z",
            snapshot_dir=tmp,
        )
        assert snap is None


def test_load_closest_ignores_other_dates():
    """Snapshot 日期對不上 → None。"""
    with tempfile.TemporaryDirectory() as tmp:
        _make_snapshot_dir(tmp)
        snap = load_closest_snapshot(
            game_date_et="2026-04-19",  # 不是同一天
            game_start_utc="2026-04-19T23:00:00Z",
            snapshot_dir=tmp,
        )
        assert snap is None
