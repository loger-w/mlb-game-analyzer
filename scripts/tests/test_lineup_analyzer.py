"""Tests for lineup_analyzer helpers."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_last7_babip_averaged_from_core_lineup():
    """last7_babip = 核心打線近 7 天 BABIP 平均（忽略 None / 非數值）。"""
    from lineup_analyzer import compute_last7_babip

    core_lineup = [
        {"last_7": {"babip": "0.320"}},
        {"last_7": {"babip": "0.280"}},
        {"last_7": {"babip": "0.340"}},
        {"last_7": {"babip": None}},
        {"last_7": {}},
    ]
    result = compute_last7_babip(core_lineup)
    assert result is not None
    # (0.320 + 0.280 + 0.340) / 3 ≈ 0.313
    assert abs(result - 0.313) < 0.002


def test_last7_babip_empty_lineup_returns_none():
    from lineup_analyzer import compute_last7_babip
    assert compute_last7_babip([]) is None


def test_last7_babip_all_missing_returns_none():
    from lineup_analyzer import compute_last7_babip
    assert compute_last7_babip([{"last_7": {}}, {}, {"last_7": {"babip": None}}]) is None


def test_last7_babip_numeric_values_accepted():
    """float 值（非 string）也能處理。"""
    from lineup_analyzer import compute_last7_babip
    core_lineup = [
        {"last_7": {"babip": 0.300}},
        {"last_7": {"babip": 0.320}},
    ]
    result = compute_last7_babip(core_lineup)
    assert result is not None
    assert abs(result - 0.310) < 0.001


def test_last7_babip_invalid_value_ignored():
    """非數值 babip（如 "N/A"）應被忽略。"""
    from lineup_analyzer import compute_last7_babip
    core_lineup = [
        {"last_7": {"babip": "0.280"}},
        {"last_7": {"babip": "N/A"}},
        {"last_7": {"babip": "0.320"}},
    ]
    result = compute_last7_babip(core_lineup)
    assert result is not None
    assert abs(result - 0.300) < 0.001


import json
from pathlib import Path
from unittest.mock import MagicMock


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _mock_requests_get(fixture: dict):
    """Build a MagicMock that emulates requests.get returning fixture JSON."""
    resp = MagicMock()
    resp.json.return_value = fixture
    resp.raise_for_status.return_value = None
    return MagicMock(return_value=resp)


def test_fetch_official_lineup_full(monkeypatch):
    """完整 9 人 → 回傳 list[int] 長度 9，順序保留。"""
    fixture = _load_fixture("feed_live_official_lineup.json")
    monkeypatch.setattr("lineup_analyzer.requests.get", _mock_requests_get(fixture))

    from lineup_analyzer import fetch_official_lineup
    result = fetch_official_lineup(game_pk=778345, team_id=147)
    assert result == [592450, 519317, 624413, 519203, 670541, 543305, 596019, 624577, 656555]


def test_fetch_official_lineup_partial(monkeypatch):
    """5 人 → 直接回傳 list[int] 長度 5（caller 決定 fallback）。"""
    fixture = _load_fixture("feed_live_partial_lineup.json")
    monkeypatch.setattr("lineup_analyzer.requests.get", _mock_requests_get(fixture))

    from lineup_analyzer import fetch_official_lineup
    result = fetch_official_lineup(game_pk=778345, team_id=147)
    assert result == [592450, 519317, 624413, 519203, 670541]


def test_fetch_official_lineup_empty(monkeypatch):
    """battingOrder=[] → 回傳空 list（不是 None）。"""
    fixture = _load_fixture("feed_live_empty_lineup.json")
    monkeypatch.setattr("lineup_analyzer.requests.get", _mock_requests_get(fixture))

    from lineup_analyzer import fetch_official_lineup
    result = fetch_official_lineup(game_pk=778345, team_id=147)
    assert result == []


def test_fetch_official_lineup_team_not_found(monkeypatch, capsys):
    """team_id 不在 home/away → 回 None + stderr 警告。"""
    fixture = _load_fixture("feed_live_official_lineup.json")
    monkeypatch.setattr("lineup_analyzer.requests.get", _mock_requests_get(fixture))

    from lineup_analyzer import fetch_official_lineup
    result = fetch_official_lineup(game_pk=778345, team_id=999)  # 不在 fixture 內
    assert result is None
    captured = capsys.readouterr()
    assert "team_id 999 not in boxscore" in captured.err


def test_fetch_official_lineup_api_fail(monkeypatch, capsys):
    """requests.get 拋例外 → 回 None + stderr 警告。"""
    def _raise(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("lineup_analyzer.requests.get", _raise)

    from lineup_analyzer import fetch_official_lineup
    result = fetch_official_lineup(game_pk=778345, team_id=147)
    assert result is None
    captured = capsys.readouterr()
    assert "feed/live fetch failed" in captured.err


def test_fetch_official_lineup_http_error(monkeypatch, capsys):
    """raise_for_status raises HTTPError → 回 None + stderr 警告。"""
    import requests
    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    monkeypatch.setattr("lineup_analyzer.requests.get", lambda *a, **k: resp)

    from lineup_analyzer import fetch_official_lineup
    result = fetch_official_lineup(game_pk=778345, team_id=147)
    assert result is None
    captured = capsys.readouterr()
    assert "feed/live fetch failed" in captured.err


def test_analyze_team_official_path(monkeypatch):
    """game_pk + 完整 9 人 → lineup_source=official，9 人含 batting_order=1..9。"""
    fixture = _load_fixture("feed_live_official_lineup.json")
    monkeypatch.setattr("lineup_analyzer.requests.get", _mock_requests_get(fixture))

    monkeypatch.setattr(
        "lineup_analyzer.fetch_team_roster",
        lambda team_id, year: [
            {"id": pid, "name": f"P{pid}", "position": "DH"}
            for pid in [592450, 519317, 624413, 519203, 670541, 543305, 596019, 624577, 656555]
        ],
    )
    monkeypatch.setattr(
        "lineup_analyzer.fetch_player_batting",
        lambda pid, year: {
            "mlbam_id": pid, "pa": 100, "avg": 0.250, "obp": 0.330, "slg": 0.420,
            "ops": 0.750, "iso": 0.170, "babip": 0.300, "k_pct": 22.0, "bb_pct": 9.0,
        },
    )
    monkeypatch.setattr("lineup_analyzer.fetch_statcast_batting_leaderboard", lambda y: ({}, {}))
    monkeypatch.setattr("lineup_analyzer.fetch_team_wrc_plus", lambda team_id, year: {})
    monkeypatch.setattr("lineup_analyzer.fetch_player_platoon", lambda pid, y: None)
    monkeypatch.setattr("lineup_analyzer.fetch_player_last7", lambda pid: None)

    from lineup_analyzer import analyze_team
    result = analyze_team("NYY", 2026, opposing_pitcher_id=None, game_pk=778345)
    assert result["lineup_source"] == "official"
    assert len(result["lineup"]) == 9
    assert [b["batting_order"] for b in result["lineup"]] == list(range(1, 10))
    assert result["lineup_source_detail"]["game_pk"] == 778345
    assert "fetched_at" in result["lineup_source_detail"]


def test_analyze_team_partial_falls_back(monkeypatch, capsys):
    """5 人 → fallback projected + stderr。"""
    fixture = _load_fixture("feed_live_partial_lineup.json")
    monkeypatch.setattr("lineup_analyzer.requests.get", _mock_requests_get(fixture))

    monkeypatch.setattr(
        "lineup_analyzer.fetch_team_roster",
        lambda team_id, year: [{"id": 1, "name": "X", "position": "C"}],
    )
    monkeypatch.setattr("lineup_analyzer.fetch_il_names", lambda team_id, year: set())
    monkeypatch.setattr(
        "lineup_analyzer.fetch_player_batting",
        lambda pid, year: {
            "mlbam_id": pid, "pa": 100, "avg": 0.250, "obp": 0.330, "slg": 0.420,
            "ops": 0.750, "iso": 0.170, "babip": 0.300, "k_pct": 22.0, "bb_pct": 9.0,
        },
    )
    monkeypatch.setattr("lineup_analyzer.fetch_statcast_batting_leaderboard", lambda y: ({}, {}))
    monkeypatch.setattr("lineup_analyzer.fetch_team_wrc_plus", lambda team_id, year: {})
    monkeypatch.setattr("lineup_analyzer.fetch_player_platoon", lambda pid, y: None)
    monkeypatch.setattr("lineup_analyzer.fetch_player_last7", lambda pid: None)

    from lineup_analyzer import analyze_team
    result = analyze_team("NYY", 2026, opposing_pitcher_id=None, game_pk=778345)
    assert result["lineup_source"] == "projected"
    assert result["lineup_source_detail"] is None
    captured = capsys.readouterr()
    assert "official lineup partial" in captured.err


def test_analyze_team_no_game_pk(monkeypatch):
    """game_pk=None → 直接走 PA proxy，不打 feed/live。"""
    called = []
    def _get(*a, **k):
        called.append(a)
        raise AssertionError("requests.get should not be called when game_pk=None")
    monkeypatch.setattr("lineup_analyzer.requests.get", _get)

    monkeypatch.setattr(
        "lineup_analyzer.fetch_team_roster",
        lambda team_id, year: [{"id": 1, "name": "X", "position": "C"}],
    )
    monkeypatch.setattr("lineup_analyzer.fetch_il_names", lambda team_id, year: set())
    monkeypatch.setattr(
        "lineup_analyzer.fetch_player_batting",
        lambda pid, year: {
            "mlbam_id": pid, "pa": 100, "avg": 0.250, "obp": 0.330, "slg": 0.420,
            "ops": 0.750, "iso": 0.170, "babip": 0.300, "k_pct": 22.0, "bb_pct": 9.0,
        },
    )
    monkeypatch.setattr("lineup_analyzer.fetch_statcast_batting_leaderboard", lambda y: ({}, {}))
    monkeypatch.setattr("lineup_analyzer.fetch_team_wrc_plus", lambda team_id, year: {})
    monkeypatch.setattr("lineup_analyzer.fetch_player_platoon", lambda pid, y: None)
    monkeypatch.setattr("lineup_analyzer.fetch_player_last7", lambda pid: None)

    from lineup_analyzer import analyze_team
    result = analyze_team("NYY", 2026, opposing_pitcher_id=None, game_pk=None)
    assert result["lineup_source"] == "projected"
    assert not called


def test_analyze_team_api_fail_falls_back(monkeypatch, capsys):
    """feed/live 失敗 → fallback projected。"""
    def _raise(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("lineup_analyzer.requests.get", _raise)

    monkeypatch.setattr(
        "lineup_analyzer.fetch_team_roster",
        lambda team_id, year: [{"id": 1, "name": "X", "position": "C"}],
    )
    monkeypatch.setattr("lineup_analyzer.fetch_il_names", lambda team_id, year: set())
    monkeypatch.setattr(
        "lineup_analyzer.fetch_player_batting",
        lambda pid, year: {
            "mlbam_id": pid, "pa": 100, "avg": 0.250, "obp": 0.330, "slg": 0.420,
            "ops": 0.750, "iso": 0.170, "babip": 0.300, "k_pct": 22.0, "bb_pct": 9.0,
        },
    )
    monkeypatch.setattr("lineup_analyzer.fetch_statcast_batting_leaderboard", lambda y: ({}, {}))
    monkeypatch.setattr("lineup_analyzer.fetch_team_wrc_plus", lambda team_id, year: {})
    monkeypatch.setattr("lineup_analyzer.fetch_player_platoon", lambda pid, y: None)
    monkeypatch.setattr("lineup_analyzer.fetch_player_last7", lambda pid: None)

    from lineup_analyzer import analyze_team
    result = analyze_team("NYY", 2026, opposing_pitcher_id=None, game_pk=778345)
    assert result["lineup_source"] == "projected"
    captured = capsys.readouterr()
    assert "feed/live fetch failed" in captured.err


# ---------------------------------------------------------------------------
# Backlog #2: wRC+ → lineup_analyzer integration tests
# ---------------------------------------------------------------------------

_OFFICIAL_PIDS = [592450, 519317, 624413, 519203, 670541, 543305, 596019, 624577, 656555]


def _setup_official_path_mocks(monkeypatch, wrc_map):
    """Common monkeypatch setup for analyze_team-on-official-path integration tests.

    Returns analyze_team is callable; pass wrc_map to control what fetch_team_wrc_plus
    yields for the wRC+ integration assertions.
    """
    fixture = _load_fixture("feed_live_official_lineup.json")
    monkeypatch.setattr("lineup_analyzer.requests.get", _mock_requests_get(fixture))
    monkeypatch.setattr(
        "lineup_analyzer.fetch_team_roster",
        lambda team_id, year: [
            {"id": pid, "name": f"P{pid}", "position": "DH"} for pid in _OFFICIAL_PIDS
        ],
    )
    monkeypatch.setattr(
        "lineup_analyzer.fetch_player_batting",
        lambda pid, year: {
            "mlbam_id": pid, "pa": 100, "avg": 0.250, "obp": 0.330, "slg": 0.420,
            "ops": 0.750, "iso": 0.170, "babip": 0.300, "k_pct": 22.0, "bb_pct": 9.0,
        },
    )
    monkeypatch.setattr("lineup_analyzer.fetch_statcast_batting_leaderboard", lambda y: ({}, {}))
    monkeypatch.setattr("lineup_analyzer.fetch_player_platoon", lambda pid, y: None)
    monkeypatch.setattr("lineup_analyzer.fetch_player_last7", lambda pid: None)
    monkeypatch.setattr("lineup_analyzer.fetch_team_wrc_plus", lambda team_id, year: wrc_map)


def test_analyze_team_assigns_wrc_plus_per_batter(monkeypatch):
    """Backlog #2: each batter gets wrc_plus from fetch_team_wrc_plus (mlbam-keyed).
    Batters not in the map → wrc_plus = None (early-season / not yet qualified)."""
    wrc_map = {592450: 145.0, 519317: 132.0, 624413: 110.0}
    _setup_official_path_mocks(monkeypatch, wrc_map)

    from lineup_analyzer import analyze_team
    result = analyze_team("NYY", 2026, opposing_pitcher_id=None, game_pk=778345)

    by_pid = {b["mlbam_id"]: b for b in result["lineup"]}
    assert by_pid[592450]["wrc_plus"] == 145.0
    assert by_pid[519317]["wrc_plus"] == 132.0
    assert by_pid[624413]["wrc_plus"] == 110.0
    # Batters not in the wrc_map → wrc_plus is None (graceful absence)
    assert by_pid[519203]["wrc_plus"] is None
    assert by_pid[670541]["wrc_plus"] is None


def test_analyze_team_computes_avg_wrc_plus(monkeypatch):
    """Backlog #2: team avg_wrc_plus = mean of per-batter wrc_plus, None excluded."""
    # 3 of 9 batters have data: 120, 110, 100 → avg = 110.0
    wrc_map = {592450: 120.0, 519317: 110.0, 624413: 100.0}
    _setup_official_path_mocks(monkeypatch, wrc_map)

    from lineup_analyzer import analyze_team
    result = analyze_team("NYY", 2026, opposing_pitcher_id=None, game_pk=778345)

    assert result["avg_wrc_plus"] == 110.0


def test_analyze_team_avg_wrc_plus_none_when_no_data(monkeypatch):
    """Backlog #2: avg_wrc_plus is None when no batters have wrc_plus.

    Common in early season: fetch_team_wrc_plus returns {} (qual=1 still has no
    qualifiers, or team abbr mismatch). Output must surface None — not crash, not
    dummy 0 — so dossier renders "—" for the row."""
    _setup_official_path_mocks(monkeypatch, wrc_map={})

    from lineup_analyzer import analyze_team
    result = analyze_team("NYY", 2026, opposing_pitcher_id=None, game_pk=778345)

    assert result["avg_wrc_plus"] is None
