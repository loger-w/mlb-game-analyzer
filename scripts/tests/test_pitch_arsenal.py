"""Tests for pitcher_stats.fetch_pitch_arsenal — Statcast pitch arsenal lookup.

The arsenal data comes from pybaseball.statcast_pitcher_arsenal_stats, which is
a leaderboard returning one row per (player_id, pitch_type). We monkeypatch the
import seam to control the DataFrame returned, mirroring the pattern in
test_pitcher_stats.py.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_arsenal_stub(df):
    """Build a stub for statcast_pitcher_arsenal_stats(year, minPA=...) → df."""
    calls = {"n": 0, "args": []}

    def stub(year, minPA=25):
        calls["n"] += 1
        calls["args"].append({"year": year, "minPA": minPA})
        return df

    return stub, calls


def test_fetch_arsenal_returns_pitches_sorted_by_usage(monkeypatch):
    """Happy path: 3 pitches → returns 3 dicts sorted by usage descending."""
    import pitcher_stats

    df = pd.DataFrame([
        {"player_id": 693821, "pitch_type": "FF", "pitch_name": "4-Seam Fastball",
         "pitch_usage": 23.7, "run_value_per_100": 0.4, "est_woba": 0.310,
         "whiff_percent": 18.2, "put_away": 14.5, "hard_hit_percent": 35.1},
        {"player_id": 693821, "pitch_type": "SL", "pitch_name": "Slider",
         "pitch_usage": 32.1, "run_value_per_100": -1.8, "est_woba": 0.245,
         "whiff_percent": 38.2, "put_away": 22.1, "hard_hit_percent": 28.3},
        {"player_id": 693821, "pitch_type": "SI", "pitch_name": "Sinker",
         "pitch_usage": 22.7, "run_value_per_100": -0.6, "est_woba": 0.290,
         "whiff_percent": 12.4, "put_away": 8.0, "hard_hit_percent": 31.5},
        {"player_id": 999999, "pitch_type": "FF", "pitch_name": "4-Seam Fastball",
         "pitch_usage": 50.0, "run_value_per_100": 0.0, "est_woba": 0.300,
         "whiff_percent": 20.0, "put_away": 15.0, "hard_hit_percent": 30.0},
    ])
    stub, calls = _make_arsenal_stub(df)
    monkeypatch.setattr(pitcher_stats, "_import_arsenal_fn", lambda: stub)

    result = pitcher_stats.fetch_pitch_arsenal(693821, 2025)
    assert isinstance(result, list)
    assert len(result) == 3  # 999999's row excluded
    # usage descending: SL 32.1 > FF 23.7 > SI 22.7
    assert [r["pitch_type"] for r in result] == ["SL", "FF", "SI"]
    assert result[0]["usage"] == 32.1
    assert result[0]["rv_per_100"] == -1.8
    assert result[0]["xwoba_against"] == 0.245
    assert result[0]["whiff_pct"] == 38.2
    assert result[0]["put_away_pct"] == 22.1
    assert result[0]["hard_hit_pct"] == 28.3
    # leaderboard called once with year + default minPA
    assert calls["n"] == 1
    assert calls["args"][0]["year"] == 2025
    assert calls["args"][0]["minPA"] == 25


def test_fetch_arsenal_player_not_in_leaderboard_returns_error(monkeypatch):
    """Pitcher not in df → return [{'error': ...}] (so callers can skip render section)."""
    import pitcher_stats

    df = pd.DataFrame([
        {"player_id": 999999, "pitch_type": "FF", "pitch_name": "4-Seam Fastball",
         "pitch_usage": 50.0, "run_value_per_100": 0.0, "est_woba": 0.300,
         "whiff_percent": 20.0, "put_away": 15.0, "hard_hit_percent": 30.0},
    ])
    stub, _ = _make_arsenal_stub(df)
    monkeypatch.setattr(pitcher_stats, "_import_arsenal_fn", lambda: stub)

    result = pitcher_stats.fetch_pitch_arsenal(693821, 2025)
    assert isinstance(result, list)
    assert len(result) == 1
    assert "error" in result[0]
    assert "693821" in result[0]["error"]


def test_fetch_arsenal_empty_dataframe_returns_error(monkeypatch):
    """Empty leaderboard → return [{'error': ...}]."""
    import pitcher_stats

    stub, _ = _make_arsenal_stub(pd.DataFrame())
    monkeypatch.setattr(pitcher_stats, "_import_arsenal_fn", lambda: stub)

    result = pitcher_stats.fetch_pitch_arsenal(693821, 2025)
    assert isinstance(result, list)
    assert len(result) == 1
    assert "error" in result[0]


def test_fetch_arsenal_handles_pybaseball_exception(monkeypatch):
    """Stub raising arbitrary Exception → caught, return [{'error': msg}]."""
    import pitcher_stats

    def raising_stub(year, minPA=25):
        raise RuntimeError("savant timeout")

    monkeypatch.setattr(pitcher_stats, "_import_arsenal_fn", lambda: raising_stub)

    result = pitcher_stats.fetch_pitch_arsenal(693821, 2025)
    assert isinstance(result, list)
    assert len(result) == 1
    assert "error" in result[0]
    assert "savant timeout" in result[0]["error"]


def test_fetch_arsenal_handles_missing_columns_gracefully(monkeypatch):
    """If Savant ever drops a column (e.g. put_away), the field becomes None
    not crash. Defensive against upstream schema drift."""
    import pitcher_stats

    df = pd.DataFrame([
        {"player_id": 693821, "pitch_type": "FF", "pitch_name": "4-Seam Fastball",
         "pitch_usage": 50.0, "run_value_per_100": 0.0, "est_woba": 0.300,
         "whiff_percent": 20.0, "hard_hit_percent": 30.0},
        # put_away missing intentionally
    ])
    stub, _ = _make_arsenal_stub(df)
    monkeypatch.setattr(pitcher_stats, "_import_arsenal_fn", lambda: stub)

    result = pitcher_stats.fetch_pitch_arsenal(693821, 2025)
    assert len(result) == 1
    assert result[0]["pitch_type"] == "FF"
    assert result[0]["put_away_pct"] is None  # graceful absence
    assert result[0]["whiff_pct"] == 20.0  # other fields still present
