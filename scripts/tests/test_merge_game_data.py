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


# ---------------------------------------------------------------------------
# core_bullpen_il_count from roster data (PR-2 commit 9)
# ---------------------------------------------------------------------------

def test_extract_core_bullpen_il_count_sums_core_roles_only():
    """Closer + Setup + High-leverage RP + Co-Closer count as core; others don't."""
    from merge_game_data import extract_core_bullpen_il_count
    roster = {
        "injured_list": [
            {"name": "Felix Bautista", "position": "Pitcher", "core_role": "Closer"},
            {"name": "Eric Helsley", "position": "Pitcher", "core_role": "Setup"},
            {"name": "Some Guy", "position": "Pitcher", "core_role": "Long RP"},
            {"name": "Other Guy", "position": "Pitcher", "core_role": "Middle RP"},
            {"name": "Position Guy", "position": "Second Base"},  # not pitcher
        ],
    }
    result = extract_core_bullpen_il_count(roster, prefix="home")
    assert "home_core_bullpen_il_count" in result
    assert result["home_core_bullpen_il_count"] == 2  # Bautista + Helsley


def test_extract_core_bullpen_il_count_includes_co_closer():
    from merge_game_data import extract_core_bullpen_il_count
    roster = {
        "injured_list": [
            {"name": "A", "position": "Pitcher", "core_role": "Co-Closer"},
            {"name": "B", "position": "Pitcher", "core_role": "Co-Closer"},
        ],
    }
    result = extract_core_bullpen_il_count(roster, prefix="away")
    assert result["away_core_bullpen_il_count"] == 2


def test_extract_core_bullpen_il_count_zero_when_no_pitcher_il():
    from merge_game_data import extract_core_bullpen_il_count
    roster = {"injured_list": []}
    assert extract_core_bullpen_il_count(roster, prefix="home")["home_core_bullpen_il_count"] == 0


def test_extract_core_bullpen_il_count_zero_when_il_pitchers_lack_role():
    """Pitchers on IL but no core_role tagged (e.g. starter on IL) → not counted."""
    from merge_game_data import extract_core_bullpen_il_count
    roster = {
        "injured_list": [
            {"name": "Starter Guy", "position": "Pitcher", "core_role": "Starter"},
            {"name": "Untagged", "position": "Pitcher"},  # missing core_role
        ],
    }
    assert extract_core_bullpen_il_count(roster, prefix="home")["home_core_bullpen_il_count"] == 0


def test_extract_core_bullpen_il_count_none_roster_returns_zero():
    """None roster (e.g. roster JSON not provided) → 0 (not None)."""
    from merge_game_data import extract_core_bullpen_il_count
    result = extract_core_bullpen_il_count(None, prefix="home")
    assert result["home_core_bullpen_il_count"] == 0


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


# ============================================================================
# Cleanup #12 — _fetch_merge_runtime_inputs concurrent dispatch
# ============================================================================


def test_fetch_merge_runtime_inputs_uses_overrides_skips_fetch(monkeypatch):
    """When home_bp_override / away_bp_override given, fetch_bullpen_era must NOT be called."""
    import merge_game_data

    weather_calls = {"n": 0}

    def boom_bullpen(*a, **k):
        raise AssertionError("fetch_bullpen_era must not be called when override provided")

    def stub_weather(pk):
        weather_calls["n"] += 1
        return {"condition": "Sunny"}

    monkeypatch.setattr(merge_game_data, "fetch_bullpen_era", boom_bullpen)
    monkeypatch.setattr(merge_game_data, "fetch_weather", stub_weather)

    home_bp, away_bp, weather = merge_game_data._fetch_merge_runtime_inputs(
        home_team_id=147, away_team_id=110, game_year=2026, game_pk=778345,
        home_bp_override=3.85, away_bp_override=4.42,
    )
    assert home_bp == 3.85
    assert away_bp == 4.42
    assert weather == {"condition": "Sunny"}
    assert weather_calls["n"] == 1


def test_fetch_merge_runtime_inputs_fetches_when_no_override(monkeypatch):
    """No override → fetch_bullpen_era called for each side; weather fetched."""
    import merge_game_data

    bp_seen = []

    def stub_bullpen(team_id, year):
        bp_seen.append((team_id, year))
        return 3.50 if team_id == 147 else 4.10

    monkeypatch.setattr(merge_game_data, "fetch_bullpen_era", stub_bullpen)
    monkeypatch.setattr(merge_game_data, "fetch_weather", lambda pk: {"temp_f": 70})

    home_bp, away_bp, weather = merge_game_data._fetch_merge_runtime_inputs(
        home_team_id=147, away_team_id=110, game_year=2026, game_pk=778345,
        home_bp_override=None, away_bp_override=None,
    )
    assert home_bp == 3.50
    assert away_bp == 4.10
    assert weather == {"temp_f": 70}
    assert sorted(bp_seen) == [(110, 2026), (147, 2026)]


def test_fetch_merge_runtime_inputs_no_team_id_falls_back_to_4(monkeypatch):
    """team_id=None on either side → that side returns 4.00 fallback without fetch.
    game_pk=None → weather=None without fetch."""
    import merge_game_data

    monkeypatch.setattr(merge_game_data, "fetch_bullpen_era", lambda tid, y: 3.0)
    monkeypatch.setattr(merge_game_data, "fetch_weather", lambda pk: {"temp_f": 1})

    home_bp, away_bp, weather = merge_game_data._fetch_merge_runtime_inputs(
        home_team_id=None, away_team_id=None, game_year=2026, game_pk=None,
        home_bp_override=None, away_bp_override=None,
    )
    assert home_bp == 4.00
    assert away_bp == 4.00
    assert weather is None


def test_fetch_merge_runtime_inputs_runs_concurrently(monkeypatch):
    """Cleanup #12: 3 fetches dispatched in parallel via ThreadPoolExecutor.

    In-flight counter approach: each stub holds for 50ms; sequential keeps max=1,
    parallel reaches max ≥ 2 with workers≥3 + 3 tasks."""
    import threading
    import time
    import merge_game_data

    state = {"current": 0, "max": 0}
    lock = threading.Lock()

    def slow_bullpen(team_id, year):
        with lock:
            state["current"] += 1
            if state["current"] > state["max"]:
                state["max"] = state["current"]
        time.sleep(0.05)
        with lock:
            state["current"] -= 1
        return 4.00

    def slow_weather(pk):
        with lock:
            state["current"] += 1
            if state["current"] > state["max"]:
                state["max"] = state["current"]
        time.sleep(0.05)
        with lock:
            state["current"] -= 1
        return None

    monkeypatch.setattr(merge_game_data, "fetch_bullpen_era", slow_bullpen)
    monkeypatch.setattr(merge_game_data, "fetch_weather", slow_weather)

    merge_game_data._fetch_merge_runtime_inputs(
        home_team_id=147, away_team_id=110, game_year=2026, game_pk=778345,
        home_bp_override=None, away_bp_override=None,
    )
    assert state["max"] >= 2, (
        f"Expected concurrent fetches (≥2 in-flight); observed max {state['max']}. "
        "Cleanup #12 requires ThreadPoolExecutor to overlap merge runtime fetches."
    )
