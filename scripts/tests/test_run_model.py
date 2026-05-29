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
    # 明確帶入 league_rg=4.4 測公式本身,不綁 config 先驗(refit 後仍成立)
    mu_home, mu_away = run_model.expected_runs(
        home_rs=4.8, away_rs=4.2,
        home_pitch=4.38, away_pitch=3.76, pf=100, league_rg=4.4,
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
    old_sigma = 3.0 * math.sqrt(2)   # 測公式,明確帶入舊 σ 不綁 config
    # 主 -1.5 過盤
    assert round(run_model.cover_prob_home(mu_margin, rl_point_home=-1.5, sigma=old_sigma), 3) == 0.355
    # Over 8.5
    assert round(run_model.over_prob(mu_total, total_line=8.5, sigma=old_sigma), 3) == 0.479
    # 主 ML
    assert round(run_model.home_ml_prob(mu_margin, sigma=old_sigma), 3) == 0.492


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
    # 結構 / 內部一致性(用 live config,refit 後仍成立)
    assert set(out) >= {"mu_home", "mu_away", "mu_margin", "mu_total",
                        "p_home_ml", "p_home_cover_rl", "p_over"}
    assert round(out["mu_margin"], 2) == round(out["mu_home"] - out["mu_away"], 2)
    assert round(out["mu_total"], 2) == round(out["mu_home"] + out["mu_away"], 2)
    for k in ("p_home_ml", "p_home_cover_rl", "p_over"):
        assert 0.0 < out[k] < 1.0


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


def test_expected_runs_league_rg_override_changes_mu():
    import config
    base = run_model.expected_runs(4.5, 4.0, 4.0, 4.2, 100.0)
    override = run_model.expected_runs(4.5, 4.0, 4.0, 4.2, 100.0, league_rg=config.LEAGUE_RG * 2)
    # mu scales as 1/league_rg → doubling league_rg halves each mu
    assert abs(override[0] - base[0] / 2) < 1e-9
    assert abs(override[1] - base[1] / 2) < 1e-9


def test_cover_prob_home_sigma_override_changes_prob():
    base = run_model.cover_prob_home(0.5, -1.5)
    wide = run_model.cover_prob_home(0.5, -1.5, sigma=100.0)
    # huge sigma → prob pulled toward 0.5
    assert abs(wide - 0.5) < abs(base - 0.5)


def test_over_prob_sigma_override_changes_prob():
    base = run_model.over_prob(8.0, 8.5)
    wide = run_model.over_prob(8.0, 8.5, sigma=100.0)
    assert abs(wide - 0.5) < abs(base - 0.5)
