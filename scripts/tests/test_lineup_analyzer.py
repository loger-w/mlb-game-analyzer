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
