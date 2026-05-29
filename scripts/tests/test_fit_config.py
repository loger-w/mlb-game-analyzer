import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fit_config
import config


def _row(**kw):
    base = dict(home_rs_recent=4.5, home_rs_season=4.5, away_rs_recent=4.0, away_rs_season=4.0,
                home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0,
                away_bullpen_era=4.0, park_factor=100.0, actual_total=9, actual_margin=1,
                has_odds=False, rl_home_point=None, rl_home_no_vig=None,
                total_line=None, over_no_vig=None, date="2026-05-01", matchup="A@B")
    base.update(kw)
    return base


def test_recompute_mu_matches_manual():
    r = _row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
             home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
             park_factor=100.0)
    # rs blend = 4.4; pitch = 0.6*4.0+0.4*4.0 = 4.0; mu = 4.4*4.0/4.4*1.0 = 4.0 each
    mt, mm = fit_config.recompute_mu(r, league_rg=4.4)
    assert abs(mt - 8.0) < 1e-9
    assert abs(mm - 0.0) < 1e-9


def test_recompute_mu_null_fip_uses_league_rg_fallback():
    r = _row(home_starter_fip=None, away_starter_fip=None)
    # should not raise; null fip → fallback = league_rg
    mt, mm = fit_config.recompute_mu(r, league_rg=4.4)
    assert mt > 0


def test_load_fit_rows_reads_inputs_and_result(tmp_path):
    d = tmp_path / "2026-05-01" / "A@B"
    d.mkdir(parents=True)
    feats = {"schema_version": 2,
             "inputs": {"home_rs_recent": 4.5, "home_rs_season": 4.4,
                        "away_rs_recent": 4.0, "away_rs_season": 4.1,
                        "home_starter": {"fip": 3.5}, "away_starter": {"fip": None},
                        "home_bullpen_era": 3.9, "away_bullpen_era": 4.1, "park_factor": 101.0},
             "odds": {"rl": {"home_point": -1.5, "home_no_vig": 0.40},
                      "total": {"line": 8.5, "over_no_vig": 0.48}}}
    (d / "features.json").write_text(json.dumps(feats), encoding="utf-8")
    (d / "result.json").write_text(json.dumps({"home_score": 6, "away_score": 3, "total": 9}), encoding="utf-8")

    rows = fit_config.load_fit_rows({"2026-05"}, data_dir=tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["home_starter_fip"] == 3.5
    assert r["away_starter_fip"] is None
    assert r["actual_total"] == 9
    assert r["actual_margin"] == 3
    assert r["has_odds"] is True
    assert r["rl_home_point"] == -1.5
    assert r["total_line"] == 8.5


def test_load_fit_rows_skips_without_result(tmp_path):
    d = tmp_path / "2026-05-02" / "C@D"
    d.mkdir(parents=True)
    (d / "features.json").write_text(json.dumps({"schema_version": 2, "inputs": {}}), encoding="utf-8")
    rows = fit_config.load_fit_rows({"2026-05"}, data_dir=tmp_path)
    assert rows == []


def test_fit_league_rg_recovers_level():
    # All rows identical, no null FIP → mu_total ∝ 1/L exactly.
    # At L0=4.4 mu_total=8.0 (see recompute test). Want mean actual total = 8.0 → L stays ~4.4.
    rows = [_row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                 home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0,
                 away_bullpen_era=4.0, park_factor=100.0, actual_total=8) for _ in range(20)]
    L = fit_config.fit_league_rg(rows)
    assert abs(L - 4.4) < 0.05


def test_fit_league_rg_higher_when_actual_lower():
    # actual total 7.0 < predicted 8.0 at L=4.4 → need higher L to lower mu
    rows = [_row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                 home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0,
                 away_bullpen_era=4.0, park_factor=100.0, actual_total=7) for _ in range(20)]
    L = fit_config.fit_league_rg(rows)
    assert L > 4.4
    # mean predicted at fitted L should ≈ 7.0
    mean_mu = sum(fit_config.recompute_mu(r, L)[0] for r in rows) / len(rows)
    assert abs(mean_mu - 7.0) < 0.02


def test_fit_sigma_team_recovers_spread():
    # At L=4.4 each row predicts mu_total=8.0, mu_margin=0.0.
    # Build rows whose residuals have a known RMS. Use symmetric pairs so totals & margins
    # each have residual magnitude exactly k → SIGMA_TEAM = sqrt((k^2 + k^2)/4) = k/sqrt(2).
    k = 4.0
    rows = []
    for _ in range(10):
        rows.append(_row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                         home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0,
                         away_bullpen_era=4.0, park_factor=100.0,
                         actual_total=int(8 + k), actual_margin=int(0 + k)))
        rows.append(_row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                         home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0,
                         away_bullpen_era=4.0, park_factor=100.0,
                         actual_total=int(8 - k), actual_margin=int(0 - k)))
    s = fit_config.fit_sigma_team(rows, league_rg=4.4)
    assert abs(s - k / math.sqrt(2)) < 1e-3   # fit rounds to 3dp


def test_eval_calibration_basic():
    # One game, with odds. At L=4.4, sigma_team chosen so SIGMA=4.24.
    # mu_margin=0, rl_home_point=-1.5 → p_cover = P(margin>1.5) with mean 0 → <0.5.
    r = _row(home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
             home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
             park_factor=100.0, actual_total=9, actual_margin=3, has_odds=True,
             rl_home_point=-1.5, rl_home_no_vig=0.40, total_line=8.5, over_no_vig=0.48)
    out = fit_config.eval_calibration([r], league_rg=4.4, sigma_team=3.0)
    assert out["n_rl"] == 1
    assert out["n_ou"] == 1                      # 9 != 8.5, not a push
    assert out["rl_log_loss"] > 0
    assert out["ou_log_loss"] > 0


def test_eval_calibration_excludes_push_and_no_odds():
    r_push = _row(has_odds=True, rl_home_point=-1.5, rl_home_no_vig=0.4,
                  total_line=9.0, over_no_vig=0.48, actual_total=9, actual_margin=2)  # total==line → push
    r_noodds = _row(has_odds=False)
    out = fit_config.eval_calibration([r_push, r_noodds], league_rg=4.4, sigma_team=3.0)
    assert out["n_ou"] == 0       # push excluded
    assert out["n_rl"] == 1       # the push row still has RL; no-odds row excluded everywhere


def test_rewrite_config_text_replaces_only_two_values_and_keeps_comments():
    text = (
        "import math\n"
        "LEAGUE_RG = 4.4        # 聯盟每場均分\n"
        "RECENT_W = 0.35        # RS blend\n"
        "SIGMA_TEAM = 3.0       # 單隊單場得分 SD(歷史先驗)\n"
        "SIGMA = SIGMA_TEAM * math.sqrt(2)\n"
    )
    out = fit_config.rewrite_config_text(text, league_rg=4.75, sigma_team=3.42)
    assert "LEAGUE_RG = 4.75        # 聯盟每場均分" in out
    assert "SIGMA_TEAM = 3.42       # 單隊單場得分 SD(歷史先驗)" in out
    assert "RECENT_W = 0.35        # RS blend" in out          # untouched
    assert "SIGMA = SIGMA_TEAM * math.sqrt(2)" in out          # untouched
