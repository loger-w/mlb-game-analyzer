import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bullpen


def test_relief_er_ip_sums_only_relievers():
    """Starter (gamesStarted=1) excluded; relievers (gamesStarted=0) summed."""
    side = {
        "pitchers": [10, 20, 30],
        "players": {
            "ID10": {"stats": {"pitching": {"gamesStarted": 1, "earnedRuns": 4, "inningsPitched": "6.0"}}},
            "ID20": {"stats": {"pitching": {"gamesStarted": 0, "earnedRuns": 1, "inningsPitched": "1.2"}}},
            "ID30": {"stats": {"pitching": {"gamesStarted": 0, "earnedRuns": 2, "inningsPitched": "1.0"}}},
        },
    }
    er, ip = bullpen.relief_er_ip(side)
    assert er == 3
    assert abs(ip - (1 + 2/3 + 1.0)) < 1e-9


def test_relief_er_ip_skips_pitchers_without_pitching_line():
    side = {"pitchers": [10], "players": {"ID10": {"stats": {}}}}
    assert bullpen.relief_er_ip(side) == (0.0, 0.0)


def test_relief_era_as_of_counts_strictly_before():
    per_game = [
        {"date": "2026-05-01", "er": 2, "ip": 3.0},
        {"date": "2026-05-02", "er": 1, "ip": 3.0},
        {"date": "2026-05-05", "er": 9, "ip": 1.0},  # on/after as_of → excluded
    ]
    # as_of 2026-05-05 → only 05-01 & 05-02: ER=3, IP=6 → 9*3/6 = 4.50
    assert bullpen.relief_era_as_of(per_game, "2026-05-05") == 4.50


def test_relief_era_as_of_none_when_no_innings():
    assert bullpen.relief_era_as_of([{"date": "2026-05-01", "er": 0, "ip": 0.0}], "2026-05-05") is None
    assert bullpen.relief_era_as_of([], "2026-05-05") is None


def _fake_games(year, through):
    return [
        {"game_pk": 1, "date": "2026-05-01", "home_id": 141, "away_id": 147},
        {"game_pk": 2, "date": "2026-05-02", "home_id": 147, "away_id": 141},
    ]


def _fake_boxscore(game_pk):
    # Every game: home relief ER=1 IP=2.0, away relief ER=3 IP=2.0
    side = lambda er: {"pitchers": [9],
                       "players": {"ID9": {"stats": {"pitching":
                           {"gamesStarted": 0, "earnedRuns": er, "inningsPitched": "2.0"}}}}}
    return {"teams": {"home": side(1), "away": side(3)}}


def test_build_relief_index_accumulates_both_teams():
    idx = bullpen.build_relief_index(2026, "2026-05-31",
                                     games_fetcher=_fake_games, boxscore_fetcher=_fake_boxscore)
    # 141: home in g1 (er1) + away in g2 (er3); 147: away in g1 (er3) + home in g2 (er1)
    assert [r["er"] for r in idx["141"]] == [1, 3]
    assert [r["er"] for r in idx["147"]] == [3, 1]


def test_load_or_build_index_caches_and_reuses(tmp_path, monkeypatch):
    calls = {"n": 0}
    def counting_games(year, through):
        calls["n"] += 1
        return _fake_games(year, through)
    monkeypatch.setattr(bullpen, "_fetch_season_final_games", counting_games)
    monkeypatch.setattr(bullpen, "_fetch_boxscore", _fake_boxscore)

    a = bullpen.load_or_build_index(2026, needed_through="2026-05-10", cache_dir=tmp_path)
    b = bullpen.load_or_build_index(2026, needed_through="2026-05-10", cache_dir=tmp_path)
    assert a == b
    assert calls["n"] == 1  # second call served from cache
    assert (tmp_path / "relief_index_2026.json").exists()


def test_load_or_build_index_rebuilds_when_coverage_insufficient(tmp_path, monkeypatch):
    calls = {"n": 0}
    def counting_games(year, through):
        calls["n"] += 1
        return _fake_games(year, through)
    monkeypatch.setattr(bullpen, "_fetch_season_final_games", counting_games)
    monkeypatch.setattr(bullpen, "_fetch_boxscore", _fake_boxscore)

    bullpen.load_or_build_index(2026, needed_through="2026-05-01", cache_dir=tmp_path)
    bullpen.load_or_build_index(2026, needed_through="2026-05-20", cache_dir=tmp_path)  # beyond built_through
    assert calls["n"] == 2


def test_relief_era_uses_cache_and_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(bullpen, "_fetch_season_final_games", _fake_games)
    monkeypatch.setattr(bullpen, "_fetch_boxscore", _fake_boxscore)
    # team 141, as_of 2026-05-02 → only g1 (er1, ip2.0) → 9*1/2 = 4.50
    assert bullpen.relief_era(141, 2026, "2026-05-02", cache_dir=tmp_path) == 4.50
    # unknown team → no innings → fallback
    assert bullpen.relief_era(999, 2026, "2026-05-02", cache_dir=tmp_path, fallback=4.00) == 4.00


def test_relief_ip_last_k_sums_window():
    idx = {"141": [{"date": "2026-05-01", "er": 0, "ip": 3.0},
                   {"date": "2026-05-02", "er": 0, "ip": 2.0},
                   {"date": "2026-05-04", "er": 0, "ip": 5.0}]}
    # as_of 05-04, k=2 → window [05-02, 05-04) → only 05-02 (2.0); 05-01 outside, 05-04 excluded
    assert bullpen.relief_ip_last_k(141, 2026, "2026-05-04", k=2, index=idx) == 2.0
    # k=3 → [05-01, 05-04) → 3.0 + 2.0 = 5.0
    assert bullpen.relief_ip_last_k(141, 2026, "2026-05-04", k=3, index=idx) == 5.0
    # unknown team → 0.0
    assert bullpen.relief_ip_last_k(999, 2026, "2026-05-04", k=2, index=idx) == 0.0
