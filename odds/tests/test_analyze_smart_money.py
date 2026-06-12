"""Tests for odds/analyze_smart_money.py — 日期發現的容錯(壞 commence_utc 不可拖垮整批)。"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from snapshot_loader import Snapshot
from analyze_smart_money import _discover_et_dates


def _snap(games):
    return Snapshot(
        snapshot_time_et=datetime(2026, 5, 2, 12, 0),
        snapshot_time_utc=datetime(2026, 5, 2, 16, 0, tzinfo=timezone.utc),
        games=games,
    )


def test_discover_et_dates_converts_utc_to_et_date():
    # 2026-05-03T02:00 UTC = 2026-05-02 22:00 ET → 歸到 5/02
    snaps = [_snap([{"commence_utc": "2026-05-03T02:00:00Z"}])]
    assert _discover_et_dates(snaps) == ["2026-05-02"]


def test_discover_et_dates_skips_unparseable_and_missing():
    snaps = [_snap([
        {"commence_utc": "garbage"},
        {"commence_utc": ""},
        {},
        {"commence_utc": "2026-05-02T23:30:00Z"},   # ET 19:30 → 5/02
    ])]
    assert _discover_et_dates(snaps) == ["2026-05-02"]


def test_discover_et_dates_sorted_unique():
    snaps = [_snap([
        {"commence_utc": "2026-05-04T17:00:00Z"},
        {"commence_utc": "2026-05-02T23:30:00Z"},
        {"commence_utc": "2026-05-02T17:00:00Z"},   # 同日重複
    ])]
    assert _discover_et_dates(snaps) == ["2026-05-02", "2026-05-04"]
