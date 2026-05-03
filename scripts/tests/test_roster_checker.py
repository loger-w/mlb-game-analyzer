"""Tests for roster_checker.fetch_pitcher_season_stats_bulk.

Covers behavior (per-pid skip, None filtering, empty-splits handling) and
concurrency (Cleanup #6: parallel fetch via ThreadPoolExecutor).
"""
import os
import sys
import threading
import time
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _stats_payload(saves=0, holds=0, g=0, gs=0, ip="0"):
    return {
        "stats": [{
            "splits": [{
                "stat": {
                    "saves": saves, "holds": holds,
                    "gamesPlayed": g, "gamesStarted": gs,
                    "inningsPitched": ip,
                }
            }]
        }]
    }


def _make_fake_get(payload_by_pid, fail_pids=None):
    """Return a thread-safe fake requests.get that maps URL pid → payload (or raises)."""
    fail_pids = fail_pids or set()

    def fake_get(url, params=None, timeout=10):
        # URL format: ".../people/{pid}/stats"
        pid = int(url.split("/people/")[1].split("/")[0])
        if pid in fail_pids:
            raise RuntimeError(f"simulated network fail pid={pid}")
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: payload_by_pid.get(pid, {"stats": []})
        return resp

    return fake_get


def test_fetch_pitcher_season_stats_bulk_returns_all_valid_pids(monkeypatch):
    """All valid pids → dict keyed by pid with parsed stats."""
    from roster_checker import fetch_pitcher_season_stats_bulk
    payloads = {
        100: _stats_payload(saves=10, holds=2, g=40, gs=0, ip="35.1"),
        200: _stats_payload(saves=0, holds=15, g=50, gs=0, ip="48.2"),
        300: _stats_payload(saves=0, holds=0, g=10, gs=10, ip="60.0"),
    }
    monkeypatch.setattr("roster_checker.requests.get", _make_fake_get(payloads))
    result = fetch_pitcher_season_stats_bulk([100, 200, 300], season=2026)
    assert set(result.keys()) == {100, 200, 300}
    assert result[100]["saves"] == 10
    assert result[100]["holds"] == 2
    assert result[100]["g"] == 40
    assert result[100]["gs"] == 0
    # parse_ip("35.1") = 35 + 1/3 ≈ 35.333
    assert abs(result[100]["ip"] - 35.333) < 0.01
    assert result[200]["holds"] == 15
    assert result[300]["gs"] == 10


def test_fetch_pitcher_season_stats_bulk_skips_failed_pids(monkeypatch):
    """Per-pid network failure → that pid absent from result; rest preserved."""
    from roster_checker import fetch_pitcher_season_stats_bulk
    payloads = {
        100: _stats_payload(saves=5, g=20, ip="20.0"),
        200: _stats_payload(saves=0, holds=10, g=30, ip="30.0"),
        300: _stats_payload(saves=2, g=25, ip="25.0"),
    }
    monkeypatch.setattr(
        "roster_checker.requests.get",
        _make_fake_get(payloads, fail_pids={200}),
    )
    result = fetch_pitcher_season_stats_bulk([100, 200, 300], season=2026)
    assert set(result.keys()) == {100, 300}  # 200 dropped silently
    assert result[100]["saves"] == 5
    assert result[300]["saves"] == 2


def test_fetch_pitcher_season_stats_bulk_skips_empty_splits(monkeypatch):
    """API returns no stats / empty splits → pid silently dropped."""
    from roster_checker import fetch_pitcher_season_stats_bulk
    payloads = {
        100: _stats_payload(saves=5, g=20, ip="20.0"),
        200: {"stats": []},                      # empty stats
        300: {"stats": [{"splits": []}]},        # empty splits
    }
    monkeypatch.setattr("roster_checker.requests.get", _make_fake_get(payloads))
    result = fetch_pitcher_season_stats_bulk([100, 200, 300], season=2026)
    assert set(result.keys()) == {100}


def test_fetch_pitcher_season_stats_bulk_filters_none_pids(monkeypatch):
    """None entries in player_ids list are skipped without API call."""
    from roster_checker import fetch_pitcher_season_stats_bulk
    call_pids = []

    def tracking_get(url, params=None, timeout=10):
        pid = int(url.split("/people/")[1].split("/")[0])
        call_pids.append(pid)
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: _stats_payload(saves=5, g=20, ip="20.0")
        return resp

    monkeypatch.setattr("roster_checker.requests.get", tracking_get)
    result = fetch_pitcher_season_stats_bulk([None, 100, None, 200], season=2026)
    assert set(result.keys()) == {100, 200}
    # None entries must not trigger API call
    assert set(call_pids) == {100, 200}


def test_fetch_pitcher_season_stats_bulk_runs_concurrently(monkeypatch):
    """Cleanup #6: bulk fetch must issue requests concurrently, not strictly sequentially.

    Uses a thread-safe in-flight counter to observe overlapping requests. Sequential
    code keeps max in-flight at 1 (each request finishes before next begins). Parallel
    code (ThreadPoolExecutor with max_workers ≥ 2) lets multiple requests overlap, so
    max in-flight should reach ≥ 2 with 4 pids and a brief per-call hold.
    """
    from roster_checker import fetch_pitcher_season_stats_bulk

    state = {"current": 0, "max": 0}
    lock = threading.Lock()

    def slow_get(url, params=None, timeout=10):
        with lock:
            state["current"] += 1
            if state["current"] > state["max"]:
                state["max"] = state["current"]
        time.sleep(0.05)  # hold the "connection" briefly so peers can ramp up
        with lock:
            state["current"] -= 1
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        # Return empty stats so caller silently skips — this test cares about overlap, not output
        resp.json = lambda: {"stats": []}
        return resp

    monkeypatch.setattr("roster_checker.requests.get", slow_get)

    fetch_pitcher_season_stats_bulk([100, 200, 300, 400], season=2026)

    assert state["max"] >= 2, (
        f"Expected concurrent fetches (≥2 in-flight); observed max {state['max']}. "
        "Cleanup #6 requires ThreadPoolExecutor to overlap requests."
    )
