"""Tests for lineup_analyzer.fetch_team_wrc_plus — FanGraphs wRC+ team leaderboard fetch.

wRC+ is FanGraphs IP, only exposed via `pybaseball.batting_stats(year, qual=1)` keyed
by IDfg. The fetch fn filters by Team abbr and resolves IDfg → MLBAM in batch via
`playerid_reverse_lookup(idfg_list, key_type="fangraphs")` so caller can match against
MLB API mlbam_id (the rest of the codebase keys on MLBAM).

Mirrors the monkeypatch pattern from test_stuff_plus.py — stub `_import_wrc_fns()`
to return a (reverse_lookup, batting_stats) 2-tuple of stubs.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_wrc_stubs(reverse_lookup_df, batting_stats_df):
    """Build (reverse_lookup_stub, batting_stats_stub, calls_dict)."""
    calls = {"reverse_n": 0, "stats_n": 0, "stats_args": []}

    def reverse_lookup_stub(player_ids, key_type=None):
        calls["reverse_n"] += 1
        return reverse_lookup_df

    def batting_stats_stub(year, qual=1):
        calls["stats_n"] += 1
        calls["stats_args"].append({"year": year, "qual": qual})
        return batting_stats_df

    return reverse_lookup_stub, batting_stats_stub, calls


def test_fetch_team_wrc_plus_returns_mlbam_keyed_dict(monkeypatch):
    """Happy path: team has 2 batters in batting_stats → return {mlbam: wrc+}."""
    import lineup_analyzer

    rev_df = pd.DataFrame([
        {"key_mlbam": 592450, "key_fangraphs": 19755},
        {"key_mlbam": 519317, "key_fangraphs": 9218},
    ])
    bs_df = pd.DataFrame([
        {"IDfg": 19755, "Team": "NYY", "wRC+": 145.0},
        {"IDfg": 9218,  "Team": "NYY", "wRC+": 132.0},
        {"IDfg": 99999, "Team": "BOS", "wRC+": 110.0},
    ])
    rl, bs, calls = _make_wrc_stubs(rev_df, bs_df)
    monkeypatch.setattr(lineup_analyzer, "_import_wrc_fns", lambda: (rl, bs))

    result = lineup_analyzer.fetch_team_wrc_plus(team_id=147, year=2025)  # NYY = 147
    assert result == {592450: 145.0, 519317: 132.0}
    assert calls["reverse_n"] == 1
    assert calls["stats_n"] == 1
    assert calls["stats_args"][0]["year"] == 2025
    assert calls["stats_args"][0]["qual"] == 1


def test_fetch_team_wrc_plus_filters_by_team(monkeypatch):
    """batting_stats has multiple teams → only target team's batters returned."""
    import lineup_analyzer

    rev_df = pd.DataFrame([
        {"key_mlbam": 1, "key_fangraphs": 100},
        {"key_mlbam": 2, "key_fangraphs": 200},
    ])
    bs_df = pd.DataFrame([
        {"IDfg": 100, "Team": "NYY", "wRC+": 120.0},
        {"IDfg": 200, "Team": "NYY", "wRC+": 110.0},
        {"IDfg": 300, "Team": "BOS", "wRC+": 140.0},  # other team
        {"IDfg": 400, "Team": "TBR", "wRC+": 95.0},   # other team
    ])
    rl, bs, _ = _make_wrc_stubs(rev_df, bs_df)
    monkeypatch.setattr(lineup_analyzer, "_import_wrc_fns", lambda: (rl, bs))

    result = lineup_analyzer.fetch_team_wrc_plus(team_id=147, year=2025)
    assert set(result.keys()) == {1, 2}
    # No 300 (BOS) or 400 (TBR) in result


def test_fetch_team_wrc_plus_empty_leaderboard_returns_empty_dict(monkeypatch):
    """batting_stats df empty (early season / API outage) → return {} silently."""
    import lineup_analyzer

    rl, bs, _ = _make_wrc_stubs(pd.DataFrame(), pd.DataFrame())
    monkeypatch.setattr(lineup_analyzer, "_import_wrc_fns", lambda: (rl, bs))

    result = lineup_analyzer.fetch_team_wrc_plus(team_id=147, year=2025)
    assert result == {}


def test_fetch_team_wrc_plus_no_team_match_warns(monkeypatch, capsys):
    """Team abbr mismatch (FanGraphs uses different abbr) → return {} + stderr warn.

    Common cause: pybaseball/FanGraphs uses "NYY" / "TBR" / "WSN" while MLB API uses
    "NYY" / "TB" / "WSH". When the filter hits zero rows, surface a stderr warning so
    a real-world mismatch doesn't silently produce zero wRC+ data.
    """
    import lineup_analyzer

    rev_df = pd.DataFrame([
        {"key_mlbam": 1, "key_fangraphs": 100},
    ])
    bs_df = pd.DataFrame([
        {"IDfg": 100, "Team": "BOS", "wRC+": 120.0},  # no NYY rows at all
    ])
    rl, bs, _ = _make_wrc_stubs(rev_df, bs_df)
    monkeypatch.setattr(lineup_analyzer, "_import_wrc_fns", lambda: (rl, bs))

    result = lineup_analyzer.fetch_team_wrc_plus(team_id=147, year=2025)
    assert result == {}
    captured = capsys.readouterr()
    # Either "no rows for team" or "team abbr mismatch" — message can phrase either way
    err = captured.err.lower()
    assert "no rows for team" in err or "team abbr mismatch" in err or "nyy" in err


def test_fetch_team_wrc_plus_skips_unmapped_idfg(monkeypatch):
    """IDfg in batting_stats but missing from reverse_lookup → that batter skipped."""
    import lineup_analyzer

    rev_df = pd.DataFrame([
        # Only IDfg 100 maps to MLBAM 1; IDfg 200 has no MLBAM (e.g. minor leaguer
        # call-up not yet in Chadwick register)
        {"key_mlbam": 1, "key_fangraphs": 100},
    ])
    bs_df = pd.DataFrame([
        {"IDfg": 100, "Team": "NYY", "wRC+": 120.0},
        {"IDfg": 200, "Team": "NYY", "wRC+": 110.0},
    ])
    rl, bs, _ = _make_wrc_stubs(rev_df, bs_df)
    monkeypatch.setattr(lineup_analyzer, "_import_wrc_fns", lambda: (rl, bs))

    result = lineup_analyzer.fetch_team_wrc_plus(team_id=147, year=2025)
    assert result == {1: 120.0}


def test_fetch_team_wrc_plus_handles_pybaseball_exception(monkeypatch):
    """Stub raising → return {} silently (caller treats as no wRC+ data)."""
    import lineup_analyzer

    def raising_bs(year, qual=1):
        raise RuntimeError("FanGraphs timeout")

    rev_df = pd.DataFrame()
    monkeypatch.setattr(
        lineup_analyzer, "_import_wrc_fns",
        lambda: ((lambda *a, **k: rev_df), raising_bs),
    )

    result = lineup_analyzer.fetch_team_wrc_plus(team_id=147, year=2025)
    assert result == {}
