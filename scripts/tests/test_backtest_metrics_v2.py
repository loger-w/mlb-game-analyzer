import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from lib import metrics


def _df():
    return pd.DataFrame([
        {"p_home_cover_rl": 0.355, "rl_home_point": -1.5, "actual_margin": 3,
         "p_over": 0.479, "total_line": 8.5, "actual_total": 9,
         "home_rl_pp": -5.5, "over_pp": -4.1, "result_missing": False, "odds_missing": False},
        {"p_home_cover_rl": 0.60, "rl_home_point": -1.5, "actual_margin": 0,
         "p_over": 0.60, "total_line": 8.5, "actual_total": 7,
         "home_rl_pp": 8.0, "over_pp": 6.0, "result_missing": False, "odds_missing": False},
    ])


def test_rl_hit_rate():
    out = metrics.compute_rl_metrics(_df())
    # g1: model 主過盤 0.355<0.5 → 預測「主不過」;實際 margin 3>1.5 → 主過 → miss
    # g2: model 0.60>0.5 → 預測「主過」;實際 margin 0(<1.5) → 主沒過 → miss
    assert out["n"] == 2
    assert round(out["rl_hit_rate"], 3) == 0.000


def test_ou_hit_rate():
    out = metrics.compute_ou_metrics(_df())
    # g1: p_over 0.479<0.5 → 預測 Under;實際 9>8.5 → Over → miss
    # g2: 0.60>0.5 → 預測 Over;實際 7<8.5 → Under → miss
    assert out["n"] == 2
    assert round(out["ou_hit_rate"], 3) == 0.000


def test_edge_calibration_positive_side():
    out = metrics.compute_edge_calibration(_df())
    assert "rl_pos_edge_n" in out and "ou_pos_edge_n" in out
    # g2 has home_rl_pp>0 and over_pp>0
    assert out["rl_pos_edge_n"] == 1
    assert out["ou_pos_edge_n"] == 1
