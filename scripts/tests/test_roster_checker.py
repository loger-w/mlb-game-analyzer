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


def _make_fake_get_seasoned(payload_by_pid_season):
    """Like _make_fake_get but keyed by (pid, season). Lets tests differentiate
    current-season vs prior-season responses for the prior-year fallback (Bug 3).

    payload_by_pid_season: dict[(int, int), dict] — value is the JSON payload.
    Unmapped (pid, season) → returns {"stats": []} (no splits, simulating no data).

    Side: fake_get.calls grows as a list of (pid, season) tuples for assertions
    like "prior year was NOT queried because current G ≥ 5".
    """
    calls: list[tuple[int, int]] = []

    def fake_get(url, params=None, timeout=10):
        pid = int(url.split("/people/")[1].split("/")[0])
        season = (params or {}).get("season")
        calls.append((pid, season))
        payload = payload_by_pid_season.get((pid, season), {"stats": []})
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: payload
        return resp

    fake_get.calls = calls
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


# ---------------------------------------------------------------------------
# Backlog #3 — prior_year fallback (Bug 3: 5/02 BAL@NYY 漏抓 Bautista)
# ---------------------------------------------------------------------------


def test_fetch_falls_back_to_prior_year_when_current_returns_no_splits(monkeypatch):
    """Long-IL all-season case (e.g. Bautista 2026): current season returns 0 splits
    → fall back to prior year stats with from_prior_year=True flag.

    Without this, tag_role gets 0/0/0/0/0 → Unknown → core_il_count drops the player
    from the count (per merge_game_data.extract_core_bullpen_il_count semantics).
    """
    from roster_checker import fetch_pitcher_season_stats_bulk

    payloads = {
        # current season: empty (long-IL the whole year)
        (100, 2026): {"stats": []},
        # prior season: full Closer workload
        (100, 2025): _stats_payload(saves=30, holds=2, g=60, gs=0, ip="60.0"),
    }
    fake_get = _make_fake_get_seasoned(payloads)
    monkeypatch.setattr("roster_checker.requests.get", fake_get)

    result = fetch_pitcher_season_stats_bulk([100], season=2026)
    assert 100 in result
    assert result[100]["saves"] == 30
    assert result[100]["g"] == 60
    assert result[100]["from_prior_year"] is True
    # Both seasons were queried
    assert (100, 2026) in fake_get.calls
    assert (100, 2025) in fake_get.calls


def test_fetch_falls_back_to_prior_year_when_current_g_below_5(monkeypatch):
    """Sparse current (G=2 — long-IL since April / call-up) + robust prior → prior wins.

    G < 5 is the gate per backlog spec; covers both long-IL and April small-sample cases."""
    from roster_checker import fetch_pitcher_season_stats_bulk

    payloads = {
        (200, 2026): _stats_payload(saves=0, holds=0, g=2, gs=0, ip="2.0"),
        (200, 2025): _stats_payload(saves=0, holds=20, g=65, gs=0, ip="62.0"),
    }
    fake_get = _make_fake_get_seasoned(payloads)
    monkeypatch.setattr("roster_checker.requests.get", fake_get)

    result = fetch_pitcher_season_stats_bulk([200], season=2026)
    assert result[200]["holds"] == 20
    assert result[200]["g"] == 65
    assert result[200]["from_prior_year"] is True


def test_fetch_uses_current_when_g_at_or_above_5(monkeypatch):
    """Current season G ≥ 5 → return current; prior year is NOT queried (saves API call).

    This is the fast path: a healthy mid-season pitcher should never trigger the
    prior-year fetch overhead."""
    from roster_checker import fetch_pitcher_season_stats_bulk

    payloads = {
        (300, 2026): _stats_payload(saves=10, holds=5, g=40, gs=0, ip="38.0"),
        # Note: prior year payload exists but should never be fetched
        (300, 2025): _stats_payload(saves=0, holds=0, g=99, gs=0, ip="99.0"),
    }
    fake_get = _make_fake_get_seasoned(payloads)
    monkeypatch.setattr("roster_checker.requests.get", fake_get)

    result = fetch_pitcher_season_stats_bulk([300], season=2026)
    assert result[300]["saves"] == 10
    assert result[300]["g"] == 40
    assert result[300].get("from_prior_year") is not True
    assert (300, 2026) in fake_get.calls
    assert (300, 2025) not in fake_get.calls


def test_fetch_no_prior_fallback_when_player_lacks_prior_data(monkeypatch):
    """Rookie / first-year call-up: current G=2, prior season empty → return sparse
    current (no from_prior_year flag — there's nothing to fall back to)."""
    from roster_checker import fetch_pitcher_season_stats_bulk

    payloads = {
        (400, 2026): _stats_payload(saves=0, holds=1, g=2, gs=0, ip="3.0"),
        (400, 2025): {"stats": []},  # no prior data (rookie)
    }
    fake_get = _make_fake_get_seasoned(payloads)
    monkeypatch.setattr("roster_checker.requests.get", fake_get)

    result = fetch_pitcher_season_stats_bulk([400], season=2026)
    assert result[400]["g"] == 2
    assert result[400].get("from_prior_year") is not True
