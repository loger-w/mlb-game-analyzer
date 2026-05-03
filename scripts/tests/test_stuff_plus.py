"""Tests for pitcher_stats.fetch_stuff_pitching_plus — FanGraphs Stuff+/Pitching+ lookup.

Stuff+ is FanGraphs IP, only exposed via `pybaseball.pitching_stats(year, qual=...)`
which returns rows keyed by FanGraphs IDfg (NOT MLBAM). The fetch fn resolves
MLBAM → IDfg via `pybaseball.playerid_reverse_lookup([mlbam_id], key_type='mlbam')`
to bypass the J.T./Castillo name-lookup pain.

Mirror the monkeypatch pattern from test_pitch_arsenal.py.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_stuff_stubs(reverse_lookup_df, pitching_stats_df):
    """Build stubs for (playerid_reverse_lookup, pitching_stats) — the 2-tuple
    `_import_stuff_fns()` will return."""
    calls = {"reverse_n": 0, "stats_n": 0, "stats_args": []}

    def reverse_lookup_stub(player_ids, key_type=None):
        calls["reverse_n"] += 1
        return reverse_lookup_df

    def pitching_stats_stub(year, qual=1):
        calls["stats_n"] += 1
        calls["stats_args"].append({"year": year, "qual": qual})
        return pitching_stats_df

    return reverse_lookup_stub, pitching_stats_stub, calls


def test_fetch_stuff_returns_metrics(monkeypatch):
    """Happy path: MLBAM 693821 → IDfg 27495 → row in pitching_stats with all 3
    plus metrics populated."""
    import pitcher_stats

    reverse_df = pd.DataFrame([{"key_mlbam": 693821, "key_fangraphs": 27495}])
    stats_df = pd.DataFrame([
        {"IDfg": 27495, "Stuff+": 122.5, "Location+": 105.3, "Pitching+": 115.8},
        {"IDfg": 99999, "Stuff+": 95.0, "Location+": 100.0, "Pitching+": 96.5},
    ])
    rl_stub, ps_stub, calls = _make_stuff_stubs(reverse_df, stats_df)
    monkeypatch.setattr(pitcher_stats, "_import_stuff_fns",
                        lambda: (rl_stub, ps_stub))

    result = pitcher_stats.fetch_stuff_pitching_plus(693821, 2025)
    assert result["stuff_plus"] == 122.5
    assert result["location_plus"] == 105.3
    assert result["pitching_plus"] == 115.8
    assert result["idfg"] == 27495
    assert "error" not in result
    # Both pybaseball calls happened once with year=2025, qual=1 (small-sample inclusive)
    assert calls["reverse_n"] == 1
    assert calls["stats_n"] == 1
    assert calls["stats_args"][0]["year"] == 2025
    assert calls["stats_args"][0]["qual"] == 1


def test_fetch_stuff_idfg_not_found_returns_error(monkeypatch):
    """Reverse lookup returns empty df → no IDfg → return {'error': ...}.
    Caller pattern: dict.get("stuff_plus") → None → confidence missing_stuff."""
    import pitcher_stats

    reverse_df = pd.DataFrame()  # MLBAM not in Chadwick register
    stats_df = pd.DataFrame([
        {"IDfg": 27495, "Stuff+": 100, "Location+": 100, "Pitching+": 100},
    ])
    rl_stub, ps_stub, _ = _make_stuff_stubs(reverse_df, stats_df)
    monkeypatch.setattr(pitcher_stats, "_import_stuff_fns",
                        lambda: (rl_stub, ps_stub))

    result = pitcher_stats.fetch_stuff_pitching_plus(693821, 2025)
    assert "error" in result
    assert "693821" in result["error"]


def test_fetch_stuff_pitcher_below_qual_returns_error(monkeypatch):
    """IDfg resolved but pitcher not in pitching_stats df (below qual / no IP) →
    return {'error': ...}."""
    import pitcher_stats

    reverse_df = pd.DataFrame([{"key_mlbam": 693821, "key_fangraphs": 27495}])
    stats_df = pd.DataFrame([
        {"IDfg": 99999, "Stuff+": 100, "Location+": 100, "Pitching+": 100},
    ])
    rl_stub, ps_stub, _ = _make_stuff_stubs(reverse_df, stats_df)
    monkeypatch.setattr(pitcher_stats, "_import_stuff_fns",
                        lambda: (rl_stub, ps_stub))

    result = pitcher_stats.fetch_stuff_pitching_plus(693821, 2025)
    assert "error" in result
    # Either the MLBAM or the resolved IDfg in error message helps debugging
    assert "27495" in result["error"] or "693821" in result["error"]


def test_fetch_stuff_handles_pybaseball_exception(monkeypatch):
    """Stub raising arbitrary Exception → caught, return {'error': msg}."""
    import pitcher_stats

    def raising_rl(player_ids, key_type=None):
        raise RuntimeError("FanGraphs timeout")

    def ps_stub(year, qual=1):
        return pd.DataFrame()

    monkeypatch.setattr(pitcher_stats, "_import_stuff_fns",
                        lambda: (raising_rl, ps_stub))

    result = pitcher_stats.fetch_stuff_pitching_plus(693821, 2025)
    assert "error" in result
    assert "FanGraphs timeout" in result["error"]


def test_fetch_stuff_handles_missing_columns_gracefully(monkeypatch):
    """If FanGraphs ever drops a column (e.g. Pitching+), the field becomes None
    not crash. Defensive against upstream schema drift."""
    import pitcher_stats

    reverse_df = pd.DataFrame([{"key_mlbam": 693821, "key_fangraphs": 27495}])
    stats_df = pd.DataFrame([
        {"IDfg": 27495, "Stuff+": 110, "Location+": 95},
        # Pitching+ column missing intentionally
    ])
    rl_stub, ps_stub, _ = _make_stuff_stubs(reverse_df, stats_df)
    monkeypatch.setattr(pitcher_stats, "_import_stuff_fns",
                        lambda: (rl_stub, ps_stub))

    result = pitcher_stats.fetch_stuff_pitching_plus(693821, 2025)
    assert result["stuff_plus"] == 110
    assert result["location_plus"] == 95
    assert result["pitching_plus"] is None  # graceful absence
    assert "error" not in result
