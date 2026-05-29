import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ablation
import fit_config
import config


def _row(**kw):
    base = dict(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                home_ra_recent=4.4, home_ra_season=4.4, away_ra_recent=4.4, away_ra_season=4.4,
                home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0,
                away_bullpen_era=4.0, park_factor=100.0, actual_total=9, actual_margin=1,
                has_odds=False, rl_home_point=None, rl_home_no_vig=None,
                total_line=None, over_no_vig=None, date="2026-05-01", matchup="A@B")
    base.update(kw)
    return base


def test_recompute_mu_ra_w0_equals_baseline():
    r = _row(home_rs_recent=4.8, home_rs_season=4.8, away_rs_recent=4.2, away_rs_season=4.2,
             home_starter_fip=4.5, away_starter_fip=3.6, home_bullpen_era=4.2, away_bullpen_era=4.0,
             home_ra_recent=3.0, home_ra_season=3.0, away_ra_recent=6.0, away_ra_season=6.0,
             park_factor=100.0)
    base = fit_config.recompute_mu(r, league_rg=4.4)
    ra0 = ablation.recompute_mu_ra(r, league_rg=4.4, w_ra=0.0)
    assert abs(ra0[0] - base[0]) < 1e-9
    assert abs(ra0[1] - base[1]) < 1e-9


def test_recompute_mu_ra_blends_in_ra():
    # away gives up more runs (RA 6.0); blending it into away defense raises expected total.
    r = _row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
             home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
             away_ra_recent=6.0, away_ra_season=6.0,
             home_ra_recent=4.0, home_ra_season=4.0, park_factor=100.0)
    mt0, _ = ablation.recompute_mu_ra(r, league_rg=4.4, w_ra=0.0)
    mt5, _ = ablation.recompute_mu_ra(r, league_rg=4.4, w_ra=0.5)
    assert mt5 > mt0


def test_fit_params_returns_league_and_sigma():
    rows = [_row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                 home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
                 home_ra_recent=4.4, home_ra_season=4.4, away_ra_recent=4.4, away_ra_season=4.4,
                 park_factor=100.0, actual_total=8, actual_margin=0) for _ in range(20)]
    p = ablation.fit_params(rows, w_ra=0.0)
    assert p["w_ra"] == 0.0
    assert abs(p["league_rg"] - 4.4) < 0.05    # mean total 8 at L=4.4 (mu_total=8)
    assert p["sigma_team"] == 0.0              # μ predicts actuals exactly


def test_select_w_ra_recovers_signal():
    # actual total tracks away RA: higher away RA → higher actual total.
    # Baseline (w=0) μ ignores RA → residuals; blending RA in reduces residuals → σ drops.
    rows = []
    for ra, tot in [(3.0, 6), (4.0, 8), (5.0, 10), (6.0, 12)]:
        for _ in range(10):
            rows.append(_row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                             home_starter_fip=4.0, away_starter_fip=4.0,
                             home_bullpen_era=4.0, away_bullpen_era=4.0,
                             away_ra_recent=ra, away_ra_season=ra,
                             home_ra_recent=ra, home_ra_season=ra,
                             park_factor=100.0, actual_total=tot, actual_margin=0))
    w_star, table = ablation.select_w_ra(rows, ablation.W_RA_GRID)
    assert w_star > 0.0
    sig0 = dict(table)[0.0]
    assert dict(table)[w_star] <= sig0


def test_select_w_ra_rejects_noise():
    # RA identical across rows (no signal) → w=0 as good as any → argmin tie-breaks to 0.0.
    rows = [_row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                 home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
                 home_ra_recent=4.4, home_ra_season=4.4, away_ra_recent=4.4, away_ra_season=4.4,
                 park_factor=100.0, actual_total=8 + (i % 3), actual_margin=0) for i in range(30)]
    w_star, table = ablation.select_w_ra(rows, ablation.W_RA_GRID)
    assert w_star == 0.0


def test_eval_logloss_returns_per_bet_arrays():
    r = _row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
             home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
             home_ra_recent=4.4, home_ra_season=4.4, away_ra_recent=4.4, away_ra_season=4.4,
             park_factor=100.0, actual_total=9, actual_margin=3, has_odds=True,
             rl_home_point=-1.5, rl_home_no_vig=0.40, total_line=8.5, over_no_vig=0.48)
    out = ablation.eval_logloss([r], {"w_ra": 0.0, "league_rg": 4.4, "sigma_team": 3.0})
    assert len(out["rl"]) == 1 and out["rl"][0] > 0
    assert len(out["ou"]) == 1 and out["ou"][0] > 0
    assert len(out["market_rl"]) == 1 and len(out["market_ou"]) == 1


def test_eval_logloss_skips_push_and_no_odds():
    r_push = _row(has_odds=True, rl_home_point=-1.5, rl_home_no_vig=0.4,
                  total_line=9.0, over_no_vig=0.48, actual_total=9, actual_margin=2)
    r_noodds = _row(has_odds=False)
    out = ablation.eval_logloss([r_push, r_noodds], {"w_ra": 0.0, "league_rg": 4.4, "sigma_team": 3.0})
    assert len(out["ou"]) == 0      # push excluded
    assert len(out["rl"]) == 1      # push row keeps RL; no-odds row excluded


def _odds_row(away_ra, total, margin):
    # May-style test row with odds. Line fixed; actual total/margin vary with away RA.
    return _row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
                away_ra_recent=away_ra, away_ra_season=away_ra, home_ra_recent=4.4, home_ra_season=4.4,
                park_factor=100.0, actual_total=total, actual_margin=margin, has_odds=True,
                rl_home_point=-1.5, rl_home_no_vig=0.41, total_line=8.5, over_no_vig=0.50)


def test_ablate_ra_structure_and_keys():
    # Train has residual spread (σ>0) so the eval's Gaussian probs are well-defined.
    train = [_row(away_ra_recent=4.4, away_ra_season=4.4,
                  actual_total=8 + (i % 3), actual_margin=(i % 3) - 1) for i in range(30)]
    test = [_odds_row(4.4, 9, 1) for _ in range(20)]
    out = ablation.ablate_ra(train, test, ablation.W_RA_GRID)
    for k in ("w_ra_star", "baseline", "candidate", "pooled_improve", "pooled_se",
              "accept", "gap_baseline", "gap_candidate"):
        assert k in out
    assert out["baseline"]["w_ra"] == 0.0
    assert isinstance(out["accept"], bool)


def _sym_odds_row(ra, total, margin):
    # symmetric: home_ra == away_ra == ra → μ_margin unaffected, isolates the total signal.
    r = _odds_row(ra, total, margin)
    r["home_ra_recent"] = ra
    r["home_ra_season"] = ra
    return r


def test_ablate_ra_accepts_when_candidate_clearly_better():
    # Train: actual total tracks RA strongly (total = 2·ra) → RA reduces train σ → w_ra*>0.
    train = []
    for ra, tot in [(2.5, 5), (4.0, 8), (5.5, 11), (7.0, 14)]:
        for _ in range(15):
            train.append(_row(away_ra_recent=ra, away_ra_season=ra, home_ra_recent=ra, home_ra_season=ra,
                              actual_total=tot, actual_margin=0))
    # Test (May): same RA→total relationship, symmetric, with odds; line 8.5 fixed.
    test = []
    for ra, tot in [(2.5, 5), (4.0, 8), (5.5, 11), (7.0, 14)]:
        for _ in range(15):
            test.append(_sym_odds_row(ra, tot, 1))
    out = ablation.ablate_ra(train, test, ablation.W_RA_GRID)
    assert out["w_ra_star"] > 0.0
    assert out["candidate"]["pooled_ll"] < out["baseline"]["pooled_ll"]   # RA helps OOS
    assert out["accept"] is True


def test_render_report_contains_verdict_and_numbers():
    result = {
        "w_ra_star": 0.25, "train_table": [(0.0, 3.5), (0.25, 3.4)],
        "baseline": {"w_ra": 0.0, "league_rg": 4.2, "sigma_team": 3.46,
                     "rl_ll": 0.69, "ou_ll": 0.70, "pooled_ll": 0.695},
        "candidate": {"w_ra": 0.25, "league_rg": 4.1, "sigma_team": 3.40,
                      "rl_ll": 0.68, "ou_ll": 0.69, "pooled_ll": 0.685},
        "pooled_improve": 0.010, "pooled_se": 0.004, "accept": True,
        "market_pooled_ll": 0.690, "gap_baseline": 0.005, "gap_candidate": -0.005,
    }
    text = ablation.render_report(result, train_n=468, test_n=292)
    assert "w_ra*" in text and "0.25" in text
    assert "ACCEPT" in text
    assert "0.685" in text
