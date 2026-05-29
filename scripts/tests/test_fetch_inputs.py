import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fetch_inputs as fi


def test_parse_ip_thirds():
    assert fi.parse_ip("123.1") == 123 + 1/3
    assert fi.parse_ip("50.2") == 50 + 2/3
    assert fi.parse_ip("12.0") == 12.0
    assert fi.parse_ip("0") == 0.0


def test_calc_fip_standard():
    # (13*15 + 3*(40+5) - 2*180)/180 + 3.10
    # = (195 + 135 - 360)/180 + 3.10 = -30/180 + 3.10 = 2.93
    assert fi.calc_fip(hr=15, bb=40, hbp=5, k=180, ip=180.0) == 2.93


def test_calc_fip_min_ip_fallback():
    # IP < MIN_IP(10) → None(呼叫端 fallback 聯盟)
    assert fi.calc_fip(hr=1, bb=2, hbp=0, k=5, ip=4.0) is None


def test_rs_blend():
    # 0.35*5.0 + 0.65*4.0 = 1.75 + 2.6 = 4.35
    assert math.isclose(fi.rs_blend(recent=5.0, season=4.0), 4.35, rel_tol=1e-9)


def test_assemble_inputs_pure():
    raw = {
        "game": {"date": "2026-05-29", "game_pk": 778001, "venue": "Coors Field",
                 "home": {"team": "Colorado Rockies", "team_id": 115,
                          "probable_pitcher": "P A", "probable_pitcher_id": 1},
                 "away": {"team": "Arizona Diamondbacks", "team_id": 109,
                          "probable_pitcher": "P B", "probable_pitcher_id": 2}},
        "home_rs_recent": 5.0, "home_rs_season": 4.0,
        "away_rs_recent": 4.2, "away_rs_season": 4.4,
        "home_ra_recent": 5.5, "home_ra_season": 5.0,
        "away_ra_recent": 4.0, "away_ra_season": 4.1,
        "home_starter": {"name": "P A", "id": 1, "fip": 4.5, "ip": 60.0,
                         "k": 55, "bb": 20, "hbp": 3, "hr": 8},
        "away_starter": {"name": "P B", "id": 2, "fip": 3.6, "ip": 70.0,
                         "k": 80, "bb": 18, "hbp": 2, "hr": 6},
        "home_bullpen_era": 4.2, "away_bullpen_era": 4.0,
        "park_factor": 112.0,
        "lineup_frozen": {"source": "projected", "home": [], "away": []},
    }
    out = fi.assemble_inputs(raw)
    assert out["home_rs_blend"] == round(fi.rs_blend(5.0, 4.0), 3)
    assert out["away_rs_blend"] == round(fi.rs_blend(4.2, 4.4), 3)
    assert out["park_factor"] == 112.0
    # raw 透傳(供 features.json 凍結)
    assert out["raw"]["home_starter"]["fip"] == 4.5


def test_assemble_inputs_fip_none_fallback():
    raw = {
        "home_rs_recent": 4.5, "home_rs_season": 4.5,
        "away_rs_recent": 4.5, "away_rs_season": 4.5,
        "home_starter": {"fip": None}, "away_starter": {"fip": 3.6},
        "home_bullpen_era": 4.0, "away_bullpen_era": 4.0, "park_factor": 100.0,
    }
    out = fi.assemble_inputs(raw)
    import config
    assert out["home_starter_fip"] == config.LEAGUE_RG   # None → fallback
    assert out["away_starter_fip"] == 3.6


def test_stat_from_byrange_splits_takes_first():
    # byDateRange 可能回重複 splits;取第一筆(已是彙總),不可加總
    splits = [
        {"stat": {"inningsPitched": "46.2", "strikeOuts": 43, "baseOnBalls": 9, "hitByPitch": 0, "homeRuns": 5}},
        {"stat": {"inningsPitched": "46.2", "strikeOuts": 43, "baseOnBalls": 9, "hitByPitch": 0, "homeRuns": 5}},
    ]
    s = fi._stat_from_byrange_splits(splits)
    assert s["inningsPitched"] == "46.2"
    assert s["strikeOuts"] == 43


def test_stat_from_byrange_splits_empty_is_none():
    assert fi._stat_from_byrange_splits([]) is None
