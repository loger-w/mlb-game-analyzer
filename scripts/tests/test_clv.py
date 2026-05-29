import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import clv


def _game(rl_home_nv=40.0, rl_away_nv=60.0, over_nv=50.0, total=8.5, rl_home_pt=-1.5):
    return {"home_team": "H", "away_team": "A",
            "bookmakers": {"pinnacle": {
                "ml": {"A": {"no_vig_pct": 39.0}, "H": {"no_vig_pct": 61.0}},
                "ou": {"Over": {"point": total, "no_vig_pct": over_nv},
                       "Under": {"point": total, "no_vig_pct": 100 - over_nv}},
                "rl": {"H": {"point": rl_home_pt, "no_vig_pct": rl_home_nv},
                       "A": {"point": -rl_home_pt, "no_vig_pct": rl_away_nv}},
            }}}


def test_pick_sides_from_edges():
    assert clv.pick_sides({"home_rl_pp": 2.2, "over_pp": -5.4}) == {"rl": "home", "ou": "under"}
    assert clv.pick_sides({"home_rl_pp": -1.0, "over_pp": 3.0}) == {"rl": "away", "ou": "over"}
    assert clv.pick_sides({"home_rl_pp": 0, "over_pp": None}) == {"rl": None, "ou": None}


def test_pick_sides_treats_nan_as_no_pick():
    # df.to_dict('records') turns missing edges into float('nan'), which is != 0 → must NOT pick
    nan = float("nan")
    assert clv.pick_sides({"home_rl_pp": nan, "over_pp": nan}) == {"rl": None, "ou": None}


def test_no_vig_for():
    g = _game(rl_home_nv=40.0, rl_away_nv=60.0, over_nv=52.0)
    assert abs(clv.no_vig_for(g, "rl", "home") - 0.40) < 1e-9
    assert abs(clv.no_vig_for(g, "rl", "away") - 0.60) < 1e-9
    assert abs(clv.no_vig_for(g, "ou", "over") - 0.52) < 1e-9
    assert abs(clv.no_vig_for(g, "ou", "under") - 0.48) < 1e-9
    assert clv.no_vig_for({"bookmakers": {}}, "rl", "home") is None


def test_point_for():
    g = _game(total=9.0, rl_home_pt=-1.5)
    assert clv.point_for(g, "ou") == 9.0
    assert clv.point_for(g, "rl") == -1.5


def test_clv_pp_sign_and_magnitude():
    entry = _game(over_nv=50.0)
    close = _game(over_nv=55.0)   # over no-vig rose 5pp → market moved toward "over"
    assert clv.clv_pp(entry, close, "ou", "over") == 5.0
    assert clv.clv_pp(entry, close, "ou", "under") == -5.0
    assert clv.clv_pp({"bookmakers": {}}, close, "ou", "over") is None


import json


def _write_snaps(tmp_path, slots):
    """slots = [(filename_slot, snap_utc, snap_et, over_nv, rl_home_nv)]; commence fixed 22:00Z."""
    for slot, snap_utc, snap_et, over_nv, rl_home_nv in slots:
        data = {"snapshot_time_utc": snap_utc, "snapshot_time_et": snap_et,
                "games": [{"home_team": "H", "away_team": "A", "game_date_et": "2026-05-02",
                           "commence_utc": "2026-05-02T22:00:00Z",
                           "bookmakers": {"pinnacle": {
                               "ml": {"A": {"no_vig_pct": 39.0}, "H": {"no_vig_pct": 61.0}},
                               "ou": {"Over": {"point": 8.5, "no_vig_pct": over_nv},
                                      "Under": {"point": 8.5, "no_vig_pct": 100 - over_nv}},
                               "rl": {"H": {"point": -1.5, "no_vig_pct": rl_home_nv},
                                      "A": {"point": 1.5, "no_vig_pct": 100 - rl_home_nv}},
                           }}}]}
        (tmp_path / f"2026-05-02_{slot}.json").write_text(json.dumps(data), encoding="utf-8")


def _row(**kw):
    base = dict(date="2026-05-02", matchup="A@H", home_team="H", away_team="A",
                home_rl_pp=2.0, over_pp=3.0)
    base.update(kw)
    return base


def test_compute_clv_rows_headroom_and_clv(tmp_path):
    _write_snaps(tmp_path, [
        ("12-00-ET", "2026-05-02T16:00:00Z", "2026-05-02 12:00 ET", 50.0, 40.0),
        ("18-00-ET", "2026-05-02T21:00:00Z", "2026-05-02 18:00 ET", 55.0, 43.0),
    ])
    rows = clv.compute_clv_rows([_row()], tmp_path)
    r = rows[0]
    assert r["has_headroom"] is True
    assert r["entry_hour"] == 12
    assert r["ou_pick"] == "over" and r["ou_clv"] == 5.0      # 55-50
    assert r["rl_pick"] == "home" and r["rl_clv"] == 3.0      # 43-40
    assert r["rl_point_stable"] is True and r["ou_point_stable"] is True


def test_compute_clv_rows_no_headroom_single_snapshot(tmp_path):
    _write_snaps(tmp_path, [
        ("12-00-ET", "2026-05-02T16:00:00Z", "2026-05-02 12:00 ET", 50.0, 40.0),
    ])
    rows = clv.compute_clv_rows([_row()], tmp_path)
    assert rows[0]["has_headroom"] is False
    assert rows[0]["ou_clv"] is None


def test_aggregate_clv_stats_and_share(tmp_path):
    rows = [
        {"has_headroom": True, "entry_hour": 12, "minutes_gap": 300,
         "rl_pick": "home", "rl_clv": 2.0, "rl_edge_pp": 2.0, "rl_point_stable": True,
         "ou_pick": "over", "ou_clv": 5.0, "ou_edge_pp": 4.0, "ou_point_stable": True},
        {"has_headroom": True, "entry_hour": 12, "minutes_gap": 300,
         "rl_pick": "home", "rl_clv": -1.0, "rl_edge_pp": 1.0, "rl_point_stable": True,
         "ou_pick": "over", "ou_clv": 1.0, "ou_edge_pp": 2.0, "ou_point_stable": False},
        {"has_headroom": True, "entry_hour": 15, "minutes_gap": 200,
         "rl_pick": "away", "rl_clv": 0.5, "rl_edge_pp": -3.0, "rl_point_stable": True,
         "ou_pick": "under", "ou_clv": -2.0, "ou_edge_pp": -1.0, "ou_point_stable": True},
        {"has_headroom": False, "entry_hour": None, "minutes_gap": None,
         "rl_pick": None, "rl_clv": None, "rl_edge_pp": None, "rl_point_stable": None,
         "ou_pick": None, "ou_clv": None, "ou_edge_pp": None, "ou_point_stable": None},
    ]
    agg = clv.aggregate_clv(rows)
    assert agg["n_total"] == 4 and agg["n_headroom"] == 3
    assert agg["ou"]["n"] == 3
    assert abs(agg["ou"]["mean"] - (5 + 1 - 2) / 3) < 1e-3
    assert abs(agg["ou"]["share_pos"] - 2 / 3) < 1e-3
    assert agg["ou"]["pct_point_stable"] == round(2 / 3, 3)
    assert 12 in agg["entry_hour_table"] and agg["entry_hour_table"][12]["n"] == 4


def test_render_clv_section_present():
    from lib import render
    agg = {"n_total": 292, "n_headroom": 120, "median_minutes_gap": 180.0,
           "rl": {"n": 110, "mean": 0.2, "median": 0.1, "share_pos": 0.52, "corr_edge": 0.05, "pct_point_stable": 0.99},
           "ou": {"n": 100, "mean": -0.1, "median": 0.0, "share_pos": 0.48, "corr_edge": -0.02, "pct_point_stable": 0.4},
           "entry_hour_table": {12: {"n": 80, "mean": 0.1}, 15: {"n": 40, "mean": -0.2}}}
    text = render.render_clv_section(agg)
    assert "CLV" in text
    assert "120" in text and "292" in text
    assert "52" in text   # rl share>0 as pct
