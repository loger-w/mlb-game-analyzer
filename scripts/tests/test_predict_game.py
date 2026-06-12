import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import predict_game as pg


def _inputs():
    return {
        "home_rs_blend": 4.8, "away_rs_blend": 4.2,
        "home_starter_fip": 4.5, "away_starter_fip": 3.6,
        "home_bullpen_era": 4.2, "away_bullpen_era": 4.0,
        "park_factor": 100.0,
        "raw": {"game": {"date": "2026-05-29", "game_pk": 1, "venue": "X",
                         "home": {"team": "H"}, "away": {"team": "A"}},
                "home_starter": {"fip": 4.5}, "away_starter": {"fip": 3.6},
                "lineup_frozen": {"source": "projected", "home": [], "away": []},
                "home_ra_recent": 5.0, "home_ra_season": 5.0,
                "away_ra_recent": 4.0, "away_ra_season": 4.0},
    }


def test_run_one_from_inputs_no_market():
    bundle = pg.run_one_from_inputs(_inputs(), market=None, snapshot_file=None)
    assert bundle["model"]["mu_home"] > 0          # μ 有算出來
    assert bundle["edges"]["home_rl_pp"] is None   # 無 market
    assert bundle["model"]["p_home_cover_rl"] is None


def test_run_one_from_inputs_with_market():
    market = {"rl": {"home_point": -1.5, "home_no_vig": 0.41,
                     "away_point": 1.5, "away_no_vig": 0.59},
              "total": {"line": 8.5, "over_no_vig": 0.52, "under_no_vig": 0.48}}
    bundle = pg.run_one_from_inputs(_inputs(), market=market, snapshot_file="s.json")
    m = bundle["model"]; e = bundle["edges"]
    assert 0.0 < m["p_home_cover_rl"] < 1.0
    assert 0.0 < m["p_over"] < 1.0
    # edge = (model 機率 − 市場 no-vig)×100,驗證 orchestrator 接線(不綁 config)
    assert e["home_rl_pp"] == round((m["p_home_cover_rl"] - 0.41) * 100, 2)
    assert e["over_pp"] == round((m["p_over"] - 0.52) * 100, 2)
