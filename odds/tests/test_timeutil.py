"""Tests for odds.lib.timeutil — odds 樹共用的時區常數與 ISO 解析。"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from timeutil import ET, TW, parse_iso_utc


def test_et_is_utc_minus_4():
    assert ET.utcoffset(None) == timedelta(hours=-4)


def test_tw_is_utc_plus_8():
    assert TW.utcoffset(None) == timedelta(hours=+8)


def test_parse_iso_utc_handles_z_suffix():
    dt = parse_iso_utc("2026-05-02T16:00:00Z")
    assert dt == datetime(2026, 5, 2, 16, 0, tzinfo=timezone.utc)


def test_parse_iso_utc_handles_explicit_offset():
    dt = parse_iso_utc("2026-05-02T16:00:00+00:00")
    assert dt == datetime(2026, 5, 2, 16, 0, tzinfo=timezone.utc)


def test_parse_iso_utc_invalid_returns_none():
    assert parse_iso_utc("not-a-date") is None
    assert parse_iso_utc("") is None
    assert parse_iso_utc(None) is None
