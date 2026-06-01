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


def _sweep_df():
    # row0: home edge +8, margin 3 (>1.5 → home covers) → pick home, HIT
    # row1: home edge -5 (pick away), margin 0 (home no-cover → away covers) → HIT; ou edge -4 (under), total 7<8.5 under → HIT
    # row2: home edge +2 (pick home), margin 1 (<1.5 no-cover) → MISS; ou edge +2 (over), total 10>8.5 over → HIT
    # row3: home edge 0 (no pick); ou edge +3 but PUSH (total==line) → excluded from O/U
    return pd.DataFrame([
        {"home_rl_pp": 8.0, "rl_home_point": -1.5, "actual_margin": 3,
         "over_pp": 6.0, "total_line": 8.5, "actual_total": 7,
         "result_missing": False, "odds_missing": False},
        {"home_rl_pp": -5.0, "rl_home_point": -1.5, "actual_margin": 0,
         "over_pp": -4.0, "total_line": 8.5, "actual_total": 7,
         "result_missing": False, "odds_missing": False},
        {"home_rl_pp": 2.0, "rl_home_point": -1.5, "actual_margin": 1,
         "over_pp": 2.0, "total_line": 8.5, "actual_total": 10,
         "result_missing": False, "odds_missing": False},
        {"home_rl_pp": 0.0, "rl_home_point": -1.5, "actual_margin": 5,
         "over_pp": 3.0, "total_line": 8.5, "actual_total": 8.5,
         "result_missing": False, "odds_missing": False},
    ])


def test_threshold_sweep_two_sided_and_filtering():
    out = metrics.compute_threshold_sweep(_sweep_df(), [0, 2, 3])
    assert out["thresholds"] == [0, 2, 3]
    rl = {r["t"]: r for r in out["rl"]}
    # row3 home edge 0 → no pick → excluded everywhere. Candidates: 8, -5, 2
    assert rl[0]["n_bets"] == 3 and round(rl[0]["hit_rate"], 3) == round(2 / 3, 3)
    # t=3 keeps |edge|>=3 → 8 (home pick HIT), -5 (away pick HIT) → 2/2; proves two-sided away pick counts
    assert rl[3]["n_bets"] == 2 and rl[3]["hit_rate"] == 1.0
    ou = {r["t"]: r for r in out["ou"]}
    # push row excluded → 3 O/U bets (6, -4, 2)
    assert ou[0]["n_bets"] == 3
    # t=3 → |over_pp|>=3 → 6 (over MISS), -4 (under HIT) → 1/2
    assert ou[3]["n_bets"] == 2 and ou[3]["hit_rate"] == 0.5


def test_threshold_sweep_empty_bucket_hit_none():
    out = metrics.compute_threshold_sweep(_sweep_df(), [99])
    assert out["rl"][0]["n_bets"] == 0 and out["rl"][0]["hit_rate"] is None
    assert out["ou"][0]["n_bets"] == 0 and out["ou"][0]["hit_rate"] is None


def test_render_threshold_section_present():
    from lib import render
    sweep = {"thresholds": [0, 2],
             "rl": [{"t": 0, "n_bets": 100, "hit_rate": 0.5}, {"t": 2, "n_bets": 40, "hit_rate": 0.55}],
             "ou": [{"t": 0, "n_bets": 90, "hit_rate": 0.52}, {"t": 2, "n_bets": 30, "hit_rate": None}]}
    sweep_clv = {"rl": [{"t": 0, "n": 95, "mean": -0.2, "share_pos": 0.44},
                        {"t": 2, "n": 38, "mean": 0.1, "share_pos": 0.5}],
                 "ou": [{"t": 0, "n": 88, "mean": 0.07, "share_pos": 0.45},
                        {"t": 2, "n": 0, "mean": None, "share_pos": None}]}
    text = render.render_threshold_section(sweep, sweep_clv)
    assert "edge 門檻掃描" in text
    assert "≥0pp" in text and "≥2pp" in text
    assert "RL" in text and "O/U" in text
    assert "—" in text   # None hit_rate / mean / share render as em dash
