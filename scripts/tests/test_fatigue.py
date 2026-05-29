import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fatigue
import fit_config


def _row(**kw):
    base = dict(date="2026-05-04", matchup="A@H",
                home_rs_recent=4.4, home_rs_season=4.4, away_rs_recent=4.4, away_rs_season=4.4,
                home_ra_recent=4.4, home_ra_season=4.4, away_ra_recent=4.4, away_ra_season=4.4,
                home_starter_fip=4.0, away_starter_fip=4.0, home_bullpen_era=4.0, away_bullpen_era=4.0,
                park_factor=100.0, home_fat_ip=0.0, away_fat_ip=0.0)
    base.update(kw)
    return base


def test_recompute_mu_fatigue_w0_equals_baseline():
    r = _row(home_fat_ip=12.0, away_fat_ip=8.0)
    base = fit_config.recompute_mu(r, league_rg=4.4)
    f0 = fatigue.recompute_mu_fatigue(r, league_rg=4.4, w_fat=0.0)
    assert abs(f0[0] - base[0]) < 1e-9 and abs(f0[1] - base[1]) < 1e-9


def test_recompute_mu_fatigue_tired_pen_raises_total():
    # away pen tired (high fat_ip) → away defense worse → home scores more → total up
    r = _row(away_fat_ip=12.0, home_fat_ip=0.0)
    t0, _ = fatigue.recompute_mu_fatigue(r, league_rg=4.4, w_fat=0.0)
    t1, _ = fatigue.recompute_mu_fatigue(r, league_rg=4.4, w_fat=0.05)
    assert t1 > t0


def test_team_ids_from_matchup_strips_doubleheader(monkeypatch):
    monkeypatch.setattr(fatigue, "resolve_team_id", lambda a: {"A": 1, "H": 2}[a])
    assert fatigue.team_ids_from_matchup("A@H") == (1, 2)
    assert fatigue.team_ids_from_matchup("A@H-G2") == (1, 2)   # suffix stripped


def test_add_fatigue_to_rows_enriches(monkeypatch):
    monkeypatch.setattr(fatigue, "resolve_team_id", lambda a: {"A": 100, "H": 200}[a])
    idx = {"100": [{"date": "2026-05-02", "er": 0, "ip": 4.0}],
           "200": [{"date": "2026-05-03", "er": 0, "ip": 13.0}]}
    rows = [dict(date="2026-05-04", matchup="A@H")]
    out = fatigue.add_fatigue_to_rows(rows, 2026, k=2, index=idx)
    assert out[0]["away_fat_ip"] == 4.0    # team A (away), 05-02 in [05-02,05-04)
    assert out[0]["home_fat_ip"] == 13.0   # team H (home), 05-03 in window


def test_add_fatigue_does_not_rebuild_when_cache_exists(tmp_path, monkeypatch):
    import json
    # cache built only through 05-25, but a row dated 05-27 → must NOT trigger a rebuild/fetch
    (tmp_path / "relief_index_2026.json").write_text(json.dumps({
        "built_through": "2026-05-25",
        "index": {"100": [{"date": "2026-05-25", "er": 0, "ip": 3.0}]}}), encoding="utf-8")
    monkeypatch.setattr(fatigue, "resolve_team_id", lambda a: {"A": 100, "H": 200}[a])

    def boom(*a, **k):
        raise AssertionError("must not rebuild / fetch from the network")
    monkeypatch.setattr(fatigue.bullpen, "_fetch_season_final_games", boom)

    rows = [dict(date="2026-05-27", matchup="A@H")]
    out = fatigue.add_fatigue_to_rows(rows, 2026, k=2, cache_dir=tmp_path)  # index=None → read cache
    assert out[0]["away_fat_ip"] == 3.0    # team 100, [05-25,05-27) → 05-25 (3.0)
    assert out[0]["home_fat_ip"] == 0.0    # team 200 not in cached index


import json


def _write_two_snaps(tmp_path, over_entry, over_close):
    for slot, snap_utc, over_nv in [("12-00-ET", "2026-05-04T16:00:00Z", over_entry),
                                     ("18-00-ET", "2026-05-04T21:00:00Z", over_close)]:
        data = {"snapshot_time_utc": snap_utc, "snapshot_time_et": f"2026-05-04 {slot[:2]}:00 ET",
                "games": [{"home_team": "Hh", "away_team": "Aa", "game_date_et": "2026-05-04",
                           "commence_utc": "2026-05-04T22:00:00Z",
                           "bookmakers": {"pinnacle": {
                               "ml": {"Aa": {"no_vig_pct": 39.0}, "Hh": {"no_vig_pct": 61.0}},
                               "ou": {"Over": {"point": 8.5, "no_vig_pct": over_nv},
                                      "Under": {"point": 8.5, "no_vig_pct": 100 - over_nv}},
                               "rl": {"Hh": {"point": -1.5, "no_vig_pct": 40.0},
                                      "Aa": {"point": 1.5, "no_vig_pct": 60.0}}}}}]}
        (tmp_path / f"2026-05-04_{slot}.json").write_text(json.dumps(data), encoding="utf-8")


def _drow(**kw):
    base = dict(date="2026-05-04", matchup="Aa@Hh", home_team="Hh", away_team="Aa",
                home_rl_pp=None, over_pp=2.0, rl_home_point=-1.5, total_line=8.5,
                actual_margin=1, actual_total=10, home_fat_ip=0.0, away_fat_ip=0.0)
    base.update(kw)
    return base


def test_fatigue_filter_splits_tail_and_reports(tmp_path):
    _write_two_snaps(tmp_path, over_entry=50.0, over_close=55.0)  # over CLV +5 in over direction
    rows = [
        _drow(home_fat_ip=13.0, over_pp=2.0, actual_total=10),  # tail (pen≥12), over edge, over hit (10>8.5)
        _drow(away_fat_ip=4.0, over_pp=2.0, actual_total=7),    # non-tail, over edge, over miss (7<8.5)
    ]
    out = fatigue.fatigue_filter_report(rows, tmp_path)
    assert out["tail"]["n"] == 1 and out["tail"]["hit_rate"] == 1.0
    assert out["non_tail"]["n"] == 1 and out["non_tail"]["hit_rate"] == 0.0
    assert out["tail"]["clv_mean"] == 5.0          # over no-vig rose 50→55
    assert out["tail"]["n_clv"] == 1


def test_fatigue_filter_skips_non_positive_edge_and_push():
    # over_pp<=0 → no over bet; home_rl_pp None → no rl bet → empty
    rows = [_drow(over_pp=-1.0, home_rl_pp=None)]
    out = fatigue.fatigue_filter_report(rows, "nonexistent_dir")
    assert out["tail"]["n"] == 0 and out["non_tail"]["n"] == 0


def test_render_report_has_both_paths():
    path_a = {"w_star": 0.0,
              "baseline": {"w": 0.0, "league_rg": 4.2, "sigma_team": 3.46,
                           "rl_ll": 0.69, "ou_ll": 0.70, "pooled_ll": 0.695},
              "candidate": {"w": 0.0, "league_rg": 4.2, "sigma_team": 3.46,
                            "rl_ll": 0.69, "ou_ll": 0.70, "pooled_ll": 0.695},
              "pooled_improve": 0.0, "pooled_se": 0.004, "accept": False}
    path_b = {"tail": {"n": 18, "hit_rate": 0.5, "clv_mean": -0.2, "n_clv": 15},
              "non_tail": {"n": 240, "hit_rate": 0.49, "clv_mean": 0.0, "n_clv": 220}}
    text = fatigue.render_report(path_a, path_b, train_n=468, test_n=292, valid_n=258)
    assert "Path A" in text and "Path B" in text
    assert "REJECT" in text
    assert "18" in text          # tail n surfaced
