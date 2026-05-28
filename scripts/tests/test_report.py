import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import report


def _bundle():
    return {
        "inputs": {"raw": {"game": {"date": "2026-05-29", "game_pk": 778001,
                                    "venue": "Coors Field",
                                    "home": {"team": "Colorado Rockies"},
                                    "away": {"team": "Arizona Diamondbacks"}},
                           "home_starter": {"name": "P A", "fip": 4.5},
                           "away_starter": {"name": "P B", "fip": 3.6},
                           "lineup_frozen": {"source": "official", "home": [], "away": []},
                           "home_rs_recent": 5.0, "home_rs_season": 4.0,
                           "away_rs_recent": 4.2, "away_rs_season": 4.4,
                           "home_ra_recent": 5.5, "home_ra_season": 5.0,
                           "away_ra_recent": 4.0, "away_ra_season": 4.1},
                   "home_rs_blend": 4.8, "away_rs_blend": 4.2,
                   "home_starter_fip": 4.5, "away_starter_fip": 3.6,
                   "home_bullpen_era": 4.2, "away_bullpen_era": 4.0,
                   "park_factor": 112.0},
        "model": {"mu_home": 4.10, "mu_away": 4.18, "mu_margin": -0.08, "mu_total": 8.28,
                  "p_home_ml": 0.492, "p_home_cover_rl": 0.355, "p_over": 0.479},
        "market": {"rl": {"home_point": -1.5, "home_no_vig": 0.41,
                          "away_point": 1.5, "away_no_vig": 0.59},
                   "total": {"line": 8.5, "over_no_vig": 0.52, "under_no_vig": 0.48}},
        "edges": {"home_rl_pp": -5.5, "over_pp": -4.1},
        "snapshot_file": "2026-05-29_15-00-ET.json",
    }


def test_build_features_schema():
    feats = report.build_features(_bundle())
    assert feats["schema_version"] == 2
    assert feats["game"]["home"] == "Colorado Rockies"
    assert feats["inputs"]["home_ra_season"] == 5.0     # RA 凍結但不進模型
    assert feats["lineup_frozen"]["source"] == "official"
    assert feats["model"]["p_over"] == 0.479
    assert feats["odds"]["rl"]["home_point"] == -1.5
    assert feats["edges"]["over_pp"] == -4.1
    assert "constants_used" in feats["model"]


def test_render_prediction_md_has_rl_ou_no_ml():
    md = report.render_prediction_md(_bundle())
    assert "RL HOME" in md and "Over" in md
    assert "35.5%" in md or "0.355" in md
    assert "Money line" not in md and "Moneyline" not in md  # 不得出現 ML


def test_render_prediction_md_no_market():
    b = _bundle()
    b["market"] = None
    b["model"] = {**b["model"], "p_home_cover_rl": None, "p_over": None}
    b["edges"] = {"home_rl_pp": None, "over_pp": None}
    md = report.render_prediction_md(b)
    assert "無盤口可比" in md


def test_write_outputs(tmp_path):
    paths = report.write_outputs(_bundle(), out_dir=tmp_path)
    assert (tmp_path / "features.json").exists()
    assert (tmp_path / "prediction.md").exists()
    data = json.loads((tmp_path / "features.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
