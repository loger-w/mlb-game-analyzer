import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_model


def test_pitch_today():
    # 0.6*3.6 + 0.4*4.0 = 3.76
    assert math.isclose(run_model.pitch_today(3.6, 4.0), 3.76, rel_tol=1e-9)
    # 0.6*4.5 + 0.4*4.2 = 4.38
    assert math.isclose(run_model.pitch_today(4.5, 4.2), 4.38, rel_tol=1e-9)


def test_expected_runs_worked_example():
    mu_home, mu_away = run_model.expected_runs(
        home_rs=4.8, away_rs=4.2,
        home_pitch=4.38, away_pitch=3.76, pf=100,
    )
    assert round(mu_home, 2) == 4.10   # 4.8*3.76/4.4
    assert round(mu_away, 2) == 4.18   # 4.2*4.38/4.4


def test_park_factor_scales_both_sides():
    a = run_model.expected_runs(4.5, 4.5, 4.4, 4.4, pf=100)
    b = run_model.expected_runs(4.5, 4.5, 4.4, 4.4, pf=110)
    assert b[0] > a[0] and b[1] > a[1]


def test_probabilities_worked_example():
    mu_margin = 4.10 - 4.18   # -0.08
    mu_total = 4.10 + 4.18    # 8.28
    # 主 -1.5 過盤
    assert round(run_model.cover_prob_home(mu_margin, rl_point_home=-1.5), 3) == 0.355
    # Over 8.5
    assert round(run_model.over_prob(mu_total, total_line=8.5), 3) == 0.479
    # 主 ML
    assert round(run_model.home_ml_prob(mu_margin), 3) == 0.492


def test_cover_prob_home_dog_line():
    # 主 +1.5:margin > -1.5 才 cover,機率應 > 主 -1.5
    mm = 0.0
    assert run_model.cover_prob_home(mm, +1.5) > run_model.cover_prob_home(mm, -1.5)


def test_over_under_complement():
    p_over = run_model.over_prob(8.28, 8.5)
    assert math.isclose(p_over + run_model.over_prob(8.28, 8.5), 2 * p_over)


def test_predict_assembles_output():
    out = run_model.predict(
        home_rs=4.8, away_rs=4.2,
        home_starter_fip=4.5, away_starter_fip=3.6,
        home_bullpen_era=4.2, away_bullpen_era=4.0,
        pf=100, rl_point_home=-1.5, total_line=8.5,
    )
    assert round(out["mu_home"], 2) == 4.10
    assert round(out["mu_away"], 2) == 4.18
    assert round(out["mu_margin"], 2) == -0.08
    assert round(out["mu_total"], 2) == 8.28
    assert round(out["p_home_cover_rl"], 3) == 0.355
    assert round(out["p_over"], 3) == 0.479
    assert round(out["p_home_ml"], 3) == 0.492


def test_predict_deterministic():
    kw = dict(home_rs=4.8, away_rs=4.2, home_starter_fip=4.5, away_starter_fip=3.6,
              home_bullpen_era=4.2, away_bullpen_era=4.0, pf=100,
              rl_point_home=-1.5, total_line=8.5)
    assert run_model.predict(**kw) == run_model.predict(**kw)


def test_predict_no_market_none_probs():
    out = run_model.predict(
        home_rs=4.8, away_rs=4.2, home_starter_fip=4.5, away_starter_fip=3.6,
        home_bullpen_era=4.2, away_bullpen_era=4.0, pf=100,
        rl_point_home=None, total_line=None,
    )
    assert out["p_home_cover_rl"] is None
    assert out["p_over"] is None
    assert out["p_home_ml"] is not None
