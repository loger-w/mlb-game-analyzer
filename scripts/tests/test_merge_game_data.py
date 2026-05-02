"""Tests for merge_game_data."""
import sys
import os
import json
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _mock_requests_get(fixture: dict):
    resp = MagicMock()
    resp.json.return_value = fixture
    resp.raise_for_status.return_value = None
    return MagicMock(return_value=resp)


def test_nested_pitcher_from_pitcher_stats_output():
    """extract_pitcher_nested 從 pitcher_stats.py 輸出取 era/xera/ip + delta。"""
    from merge_game_data import extract_pitcher_nested
    pitcher_data = {
        "season": {"era": 3.50, "xera": 2.80, "ip": 45.2, "fip": 3.10},
    }
    result = extract_pitcher_nested(pitcher_data, prefix="home")
    assert "home_pitcher" in result
    p = result["home_pitcher"]
    assert p["era"] == 3.50
    assert p["xera"] == 2.80
    assert p["ip"] == 45.2
    assert abs(p["era_xera_delta"] - 0.70) < 0.001
    assert "prior_year" not in p


def test_nested_pitcher_missing_fields_tolerant():
    from merge_game_data import extract_pitcher_nested
    result = extract_pitcher_nested({"season": {}}, prefix="home")
    assert result["home_pitcher"]["era"] is None
    assert result["home_pitcher"]["xera"] is None
    assert result["home_pitcher"]["ip"] is None
    assert result["home_pitcher"]["era_xera_delta"] is None


def test_nested_pitcher_season_error_treats_as_empty():
    """pitcher_stats.py 回 {season: {error: ...}} 時視為缺失。"""
    from merge_game_data import extract_pitcher_nested
    result = extract_pitcher_nested({"season": {"error": "lookup_failed"}}, prefix="away")
    assert result["away_pitcher"]["era"] is None


def test_nested_pitcher_none_data_tolerant():
    from merge_game_data import extract_pitcher_nested
    result = extract_pitcher_nested(None, prefix="home")
    assert result["home_pitcher"]["era"] is None


# ---------------------------------------------------------------------------
# arsenal_top pass-through (PR-1 commit 4)
# ---------------------------------------------------------------------------

def test_nested_pitcher_includes_arsenal_top_3_when_arsenal_present():
    """5 球種 → 取前 3（保留 fetch_pitch_arsenal 已排序的順序）。"""
    from merge_game_data import extract_pitcher_nested
    pitcher_data = {
        "season": {"era": 3.50, "xera": 2.80, "ip": 45.2},
        "arsenal": [
            {"pitch_type": "SL", "usage": 32.1, "rv_per_100": -1.8, "xwoba_against": 0.245,
             "whiff_pct": 38.2, "put_away_pct": 22.1, "hard_hit_pct": 28.3},
            {"pitch_type": "FF", "usage": 23.7, "rv_per_100": 0.4, "xwoba_against": 0.310,
             "whiff_pct": 18.2, "put_away_pct": 14.5, "hard_hit_pct": 35.1},
            {"pitch_type": "SI", "usage": 22.7, "rv_per_100": -0.6, "xwoba_against": 0.290,
             "whiff_pct": 12.4, "put_away_pct": 8.0, "hard_hit_pct": 31.5},
            {"pitch_type": "FC", "usage": 11.7, "rv_per_100": 0.1, "xwoba_against": 0.305,
             "whiff_pct": 15.0, "put_away_pct": 10.0, "hard_hit_pct": 30.0},
            {"pitch_type": "CH", "usage": 9.7, "rv_per_100": -0.2, "xwoba_against": 0.295,
             "whiff_pct": 25.0, "put_away_pct": 18.0, "hard_hit_pct": 25.0},
        ],
    }
    result = extract_pitcher_nested(pitcher_data, prefix="home")
    p = result["home_pitcher"]
    assert "arsenal_top" in p
    assert len(p["arsenal_top"]) == 3
    assert [a["pitch_type"] for a in p["arsenal_top"]] == ["SL", "FF", "SI"]
    # Full schema preserved
    assert p["arsenal_top"][0]["rv_per_100"] == -1.8
    assert p["arsenal_top"][0]["xwoba_against"] == 0.245


def test_nested_pitcher_arsenal_top_3_when_only_2_pitches():
    """少於 3 個球種 → 全部回傳。"""
    from merge_game_data import extract_pitcher_nested
    pitcher_data = {
        "season": {"era": 3.50},
        "arsenal": [
            {"pitch_type": "SL", "usage": 60.0},
            {"pitch_type": "FF", "usage": 40.0},
        ],
    }
    result = extract_pitcher_nested(pitcher_data, prefix="home")
    assert len(result["home_pitcher"]["arsenal_top"]) == 2


def test_nested_pitcher_arsenal_top_skips_error_entries():
    """arsenal 開頭就是 [{'error': ...}] → arsenal_top = []。"""
    from merge_game_data import extract_pitcher_nested
    pitcher_data = {
        "season": {"era": 3.50},
        "arsenal": [{"error": "No arsenal data"}],
    }
    result = extract_pitcher_nested(pitcher_data, prefix="home")
    assert result["home_pitcher"]["arsenal_top"] == []


def test_nested_pitcher_arsenal_top_missing_arsenal_returns_empty_list():
    """pitcher_data 沒有 arsenal key → arsenal_top = []，不 crash。"""
    from merge_game_data import extract_pitcher_nested
    pitcher_data = {"season": {"era": 3.50}}
    result = extract_pitcher_nested(pitcher_data, prefix="home")
    assert result["home_pitcher"]["arsenal_top"] == []


def test_nested_pitcher_arsenal_top_none_data_tolerant():
    """pitcher_data is None → arsenal_top = []。"""
    from merge_game_data import extract_pitcher_nested
    result = extract_pitcher_nested(None, prefix="home")
    assert result["home_pitcher"]["arsenal_top"] == []


def test_nested_lineup_from_lineup_analyzer_output():
    from merge_game_data import extract_lineup_nested
    lineup_data = {"last7_babip": 0.320, "avg_babip": 0.290}
    result = extract_lineup_nested(lineup_data, prefix="home")
    assert result["home_lineup"]["recent_babip"] == 0.320


def test_nested_lineup_missing_last7_babip_returns_none():
    from merge_game_data import extract_lineup_nested
    result = extract_lineup_nested({"avg_babip": 0.290}, prefix="home")
    assert result["home_lineup"]["recent_babip"] is None


def test_nested_lineup_none_data_tolerant():
    from merge_game_data import extract_lineup_nested
    result = extract_lineup_nested(None, prefix="away")
    assert result["away_lineup"]["recent_babip"] is None


# ============================================================================
# Park Factor JSON 化 + alias 解析
# ============================================================================

def test_resolve_park_factor_canonical_name():
    """正式球場名直接命中 JSON 表。"""
    from merge_game_data import resolve_park_factor
    assert resolve_park_factor("Coors Field") == 131.0
    assert resolve_park_factor("T-Mobile Park") == 82.0


def test_resolve_park_factor_alias_old_name():
    """舊球場名透過 alias 解析到新名 — 向後相容。"""
    from merge_game_data import resolve_park_factor
    # Tropicana → Steinbrenner（Rays 臨時主場）
    assert resolve_park_factor("Tropicana Field") == 100.0
    # Oakland Coliseum → Sutter Health Park
    assert resolve_park_factor("Oakland Coliseum") == 109.0
    # Minute Maid → Daikin
    assert resolve_park_factor("Minute Maid Park") == 98.0
    # Dodger Stadium → UNIQLO Field at Dodger Stadium
    assert resolve_park_factor("Dodger Stadium") == 98.0
    # Guaranteed Rate → Rate Field
    assert resolve_park_factor("Guaranteed Rate Field") == 97.0
    # Camden Yards → Oriole Park at Camden Yards
    assert resolve_park_factor("Camden Yards") == 96.0


def test_resolve_park_factor_unknown_returns_default():
    """未知球場名回傳 100.0（聯盟平均，安全 fallback）。"""
    from merge_game_data import resolve_park_factor
    assert resolve_park_factor("Nonexistent Stadium") == 100.0


def test_resolve_park_factor_none_returns_default():
    """None venue 回傳 100.0。"""
    from merge_game_data import resolve_park_factor
    assert resolve_park_factor(None) == 100.0


def test_resolve_park_factor_returns_float():
    """回傳型別必為 float（predict.py 後續做 PF / 100 浮點除法）。"""
    from merge_game_data import resolve_park_factor
    result = resolve_park_factor("Coors Field")
    assert isinstance(result, float)


# ============================================================================
# P7: fetch_weather
# ============================================================================

def test_fetch_weather_full(monkeypatch):
    """三欄齊 → 回傳 dict，indoor=False。"""
    fixture = _load_fixture("feed_live_official_lineup.json")  # weather=Sunny/78/wind
    monkeypatch.setattr("merge_game_data.requests.get", _mock_requests_get(fixture))

    from merge_game_data import fetch_weather
    result = fetch_weather(game_pk=778345)
    assert result == {
        "condition": "Sunny",
        "temp_f": 78,
        "wind_text": "10 mph, Out To CF",
        "indoor": False,
    }


def test_fetch_weather_indoor(monkeypatch):
    """condition='Roof Closed' → indoor=True。"""
    fixture = _load_fixture("feed_live_indoor.json")
    monkeypatch.setattr("merge_game_data.requests.get", _mock_requests_get(fixture))

    from merge_game_data import fetch_weather
    result = fetch_weather(game_pk=778345)
    assert result["indoor"] is True
    assert result["condition"] == "Roof Closed"
    assert result["temp_f"] == 72


def test_fetch_weather_empty(monkeypatch):
    """weather={} → 回傳 None。"""
    fixture = _load_fixture("feed_live_empty_lineup.json")
    monkeypatch.setattr("merge_game_data.requests.get", _mock_requests_get(fixture))

    from merge_game_data import fetch_weather
    assert fetch_weather(game_pk=778345) is None


def test_fetch_weather_partial(monkeypatch):
    """只有 condition、缺 wind/temp → 回傳 dict，缺欄為 None。"""
    fixture = {"gameData": {"weather": {"condition": "Cloudy", "temp": "", "wind": ""}}}
    monkeypatch.setattr("merge_game_data.requests.get", _mock_requests_get(fixture))

    from merge_game_data import fetch_weather
    result = fetch_weather(game_pk=778345)
    assert result == {"condition": "Cloudy", "temp_f": None, "wind_text": None, "indoor": False}


def test_fetch_weather_api_fail(monkeypatch, capsys):
    """API 失敗 → 回 None + stderr 警告。"""
    def _raise(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("merge_game_data.requests.get", _raise)

    from merge_game_data import fetch_weather
    assert fetch_weather(game_pk=778345) is None
    captured = capsys.readouterr()
    assert "weather fetch failed" in captured.err


# ============================================================================
# P8: merged.weather 整合到 main 流程
# ============================================================================

def test_merged_weather_present(monkeypatch, tmp_path):
    """end-to-end mock：weather API 回完整 → merged['weather'] dict 帶 4 欄。"""
    fixture = _load_fixture("feed_live_official_lineup.json")
    monkeypatch.setattr("merge_game_data.requests.get", _mock_requests_get(fixture))

    game_data = {
        "game": {
            "gamePk": 778345,
            "date": "2026-04-30T23:00:00Z",
            "venue": "Yankee Stadium",
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "X", "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "Y", "probable_pitcher_id": 2},
        },
        "home_recent": {}, "away_recent": {},
        "home_recent_30": {}, "away_recent_30": {},
        "home_season": {}, "away_season": {},
        "home_season_games_count": 0, "away_season_games_count": 0,
    }
    home_pitcher = {"name": "X", "season": {"era": 4.0}}
    away_pitcher = {"name": "Y", "season": {"era": 4.0}}
    home_lineup = {"avg_xwoba": 0.315, "avg_ops": 0.710, "avg_k_pct": 22.0,
                   "lineup_source": "official", "lineup_source_detail": {"game_pk": 778345}}
    away_lineup = {"avg_xwoba": 0.315, "avg_ops": 0.710, "avg_k_pct": 22.0,
                   "lineup_source": "projected", "lineup_source_detail": None}

    g_path = tmp_path / "g.json"; g_path.write_text(json.dumps(game_data), encoding="utf-8")
    hp = tmp_path / "hp.json"; hp.write_text(json.dumps(home_pitcher), encoding="utf-8")
    ap = tmp_path / "ap.json"; ap.write_text(json.dumps(away_pitcher), encoding="utf-8")
    hl = tmp_path / "hl.json"; hl.write_text(json.dumps(home_lineup), encoding="utf-8")
    al = tmp_path / "al.json"; al.write_text(json.dumps(away_lineup), encoding="utf-8")
    out = tmp_path / "merged.json"

    import sys as _sys
    _sys.argv = ["merge_game_data.py", "--game", str(g_path),
                 "--home-pitcher", str(hp), "--away-pitcher", str(ap),
                 "--home-lineup", str(hl), "--away-lineup", str(al),
                 "-o", str(out), "--no-md",
                 "--park-factor", "100",
                 "--home-bullpen-era", "4.0", "--away-bullpen-era", "4.0"]
    from merge_game_data import main
    main()

    merged = json.loads(out.read_text(encoding="utf-8"))
    assert merged["weather"] == {
        "condition": "Sunny",
        "temp_f": 78,
        "wind_text": "10 mph, Out To CF",
        "indoor": False,
    }


def test_merged_weather_absent(monkeypatch, tmp_path):
    """weather 欄位全空 → merged['weather'] = None。"""
    fixture = _load_fixture("feed_live_empty_lineup.json")
    monkeypatch.setattr("merge_game_data.requests.get", _mock_requests_get(fixture))

    game_data = {
        "game": {
            "gamePk": 778345,
            "date": "2026-04-30T23:00:00Z",
            "venue": "Yankee Stadium",
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "X", "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "Y", "probable_pitcher_id": 2},
        },
        "home_recent": {}, "away_recent": {},
        "home_recent_30": {}, "away_recent_30": {},
        "home_season": {}, "away_season": {},
        "home_season_games_count": 0, "away_season_games_count": 0,
    }
    home_pitcher = {"name": "X", "season": {"era": 4.0}}
    away_pitcher = {"name": "Y", "season": {"era": 4.0}}
    home_lineup = {"avg_xwoba": 0.315, "avg_ops": 0.710, "avg_k_pct": 22.0}
    away_lineup = {"avg_xwoba": 0.315, "avg_ops": 0.710, "avg_k_pct": 22.0}

    g_path = tmp_path / "g.json"; g_path.write_text(json.dumps(game_data), encoding="utf-8")
    hp = tmp_path / "hp.json"; hp.write_text(json.dumps(home_pitcher), encoding="utf-8")
    ap = tmp_path / "ap.json"; ap.write_text(json.dumps(away_pitcher), encoding="utf-8")
    hl = tmp_path / "hl.json"; hl.write_text(json.dumps(home_lineup), encoding="utf-8")
    al = tmp_path / "al.json"; al.write_text(json.dumps(away_lineup), encoding="utf-8")
    out = tmp_path / "merged.json"

    import sys as _sys
    _sys.argv = ["merge_game_data.py", "--game", str(g_path),
                 "--home-pitcher", str(hp), "--away-pitcher", str(ap),
                 "--home-lineup", str(hl), "--away-lineup", str(al),
                 "-o", str(out), "--no-md",
                 "--park-factor", "100",
                 "--home-bullpen-era", "4.0", "--away-bullpen-era", "4.0"]
    from merge_game_data import main
    main()

    merged = json.loads(out.read_text(encoding="utf-8"))
    assert merged["weather"] is None
