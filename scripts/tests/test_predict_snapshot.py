"""Tests for snapshot loading and team-name resolution in predict.py."""
import sys
import os
import json
import shutil
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predict import load_closest_snapshot

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _make_snapshot_dir(tmpdir):
    """Copy fixtures into a temp snapshot dir with expected filename format."""
    shutil.copy(
        os.path.join(FIXTURES, "sample_snapshot.json"),
        os.path.join(tmpdir, "2026-04-18_16-00-ET.json"),
    )
    shutil.copy(
        os.path.join(FIXTURES, "sample_snapshot_earlier.json"),
        os.path.join(tmpdir, "2026-04-18_12-00-ET.json"),
    )


def test_load_closest_picks_newest_before_gametime():
    """game_start 19:00 ET → should pick 16:00 ET (newest before start)."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_snapshot_dir(tmp)
        snap = load_closest_snapshot(
            game_date_et="2026-04-18",
            game_start_utc="2026-04-18T23:00:00Z",
            snapshot_dir=tmp,
        )
        assert snap is not None
        assert snap["snapshot_time_et"] == "2026-04-18 16:00 ET"


def test_load_closest_ignores_snapshots_after_gametime():
    """game_start 15:00 ET → 16:00 ET snapshot 在 start 之後，只能用 12:00 ET。"""
    with tempfile.TemporaryDirectory() as tmp:
        _make_snapshot_dir(tmp)
        snap = load_closest_snapshot(
            game_date_et="2026-04-18",
            game_start_utc="2026-04-18T19:00:00Z",  # 15:00 ET
            snapshot_dir=tmp,
        )
        assert snap is not None
        assert snap["snapshot_time_et"] == "2026-04-18 12:00 ET"


def test_load_closest_no_snapshots_returns_none():
    """Empty snapshot dir → None."""
    with tempfile.TemporaryDirectory() as tmp:
        snap = load_closest_snapshot(
            game_date_et="2026-04-18",
            game_start_utc="2026-04-18T23:00:00Z",
            snapshot_dir=tmp,
        )
        assert snap is None


def test_load_closest_ignores_other_dates():
    """Snapshot 日期對不上 → None。"""
    with tempfile.TemporaryDirectory() as tmp:
        _make_snapshot_dir(tmp)
        snap = load_closest_snapshot(
            game_date_et="2026-04-19",  # 不是同一天
            game_start_utc="2026-04-19T23:00:00Z",
            snapshot_dir=tmp,
        )
        assert snap is None


from predict import resolve_pinnacle_odds


def test_resolve_odds_matches_teams():
    """用 snapshot 找 CHC@NYM 的 Pinnacle odds。"""
    with open(os.path.join(FIXTURES, "sample_snapshot.json")) as f:
        snap = json.load(f)

    result = resolve_pinnacle_odds(snap, home_abbrev="CHC", away_abbrev="NYM")
    assert result is not None
    assert result["ml"]["home_decimal"] == 1.74
    assert result["ml"]["away_decimal"] == 2.24
    assert result["ou"]["line"] == 8.0
    assert result["ou"]["over_decimal"] == 1.93
    assert result["ou"]["under_decimal"] == 1.94
    assert result["rl"]["home_point"] == -1.5
    assert result["rl"]["home_decimal"] == 1.56
    assert result["rl"]["away_decimal"] == 2.58


def test_resolve_odds_team_mismatch_returns_none():
    """隊名對不上 → None。"""
    with open(os.path.join(FIXTURES, "sample_snapshot.json")) as f:
        snap = json.load(f)

    # Miami Marlins 不在 fixture
    result = resolve_pinnacle_odds(snap, home_abbrev="MIA", away_abbrev="ATL")
    assert result is None


def test_resolve_odds_missing_ou_market():
    """早 snapshot 只有 ML 沒 OU/RL → ml 有值但 ou/rl 為 None。"""
    with open(os.path.join(FIXTURES, "sample_snapshot_earlier.json")) as f:
        snap = json.load(f)

    result = resolve_pinnacle_odds(snap, home_abbrev="CHC", away_abbrev="NYM")
    assert result is not None
    assert result["ml"]["home_decimal"] == 1.70
    assert result["ou"] is None
    assert result["rl"] is None


def test_resolve_odds_doubleheader_without_index_errors():
    """同日同兩隊出現 2 場 → 需要 --game-index，未指定應 raise."""
    snap = {
        "snapshot_time_et": "2026-04-18 16:00 ET",
        "games": [
            {
                "game": "NYM @ CHC",
                "home_team": "Chicago Cubs",
                "away_team": "New York Mets",
                "commence_et": "2026-04-18 13:00 ET",
                "game_date_et": "2026-04-18",
                "bookmakers": {"pinnacle": {"ml": {
                    "Chicago Cubs": {"odds": 1.80}, "New York Mets": {"odds": 2.10},
                }, "ou": {}, "rl": {}}},
            },
            {
                "game": "NYM @ CHC",
                "home_team": "Chicago Cubs",
                "away_team": "New York Mets",
                "commence_et": "2026-04-18 19:00 ET",
                "game_date_et": "2026-04-18",
                "bookmakers": {"pinnacle": {"ml": {
                    "Chicago Cubs": {"odds": 1.75}, "New York Mets": {"odds": 2.20},
                }, "ou": {}, "rl": {}}},
            },
        ],
    }
    with pytest.raises(ValueError, match="doubleheader"):
        resolve_pinnacle_odds(snap, home_abbrev="CHC", away_abbrev="NYM")


def test_resolve_odds_doubleheader_with_index():
    """--game-index 指定 G2 → 取第二場 (19:00)。"""
    snap = {
        "snapshot_time_et": "2026-04-18 16:00 ET",
        "games": [
            {
                "game": "NYM @ CHC (G1)",
                "home_team": "Chicago Cubs",
                "away_team": "New York Mets",
                "commence_et": "2026-04-18 13:00 ET",
                "game_date_et": "2026-04-18",
                "bookmakers": {"pinnacle": {"ml": {
                    "Chicago Cubs": {"odds": 1.80}, "New York Mets": {"odds": 2.10},
                }, "ou": {}, "rl": {}}},
            },
            {
                "game": "NYM @ CHC (G2)",
                "home_team": "Chicago Cubs",
                "away_team": "New York Mets",
                "commence_et": "2026-04-18 19:00 ET",
                "game_date_et": "2026-04-18",
                "bookmakers": {"pinnacle": {"ml": {
                    "Chicago Cubs": {"odds": 1.75}, "New York Mets": {"odds": 2.20},
                }, "ou": {}, "rl": {}}},
            },
        ],
    }
    res = resolve_pinnacle_odds(snap, home_abbrev="CHC", away_abbrev="NYM", game_index=2)
    assert res["ml"]["home_decimal"] == 1.75  # G2


import argparse


def _make_args(game_data_path, **overrides):
    """Build argparse.Namespace with all defaults compute_kelly_block expects."""
    ns = argparse.Namespace(
        game_data=str(game_data_path),
        kelly_divisor=4, kelly_cap=3.0, unit_size=1.0,
        no_auto_odds=False, game_index=None,
        ml_odds_home_dec=None, ml_odds_away_dec=None,
        ou_odds_over_dec=None, ou_odds_under_dec=None,
        rl_odds_home_dec=None, rl_odds_away_dec=None,
        ou_line=None,  # Task 11 follow-up 568f416 made this CLI arg a fallback inside compute_kelly_block
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_end_to_end_predict_with_snapshot(tmp_path):
    """Happy path: 路徑含 ET 日期 → 抓 snapshot → kelly block 完整。"""
    game_dir = tmp_path / "2026-04-18" / "NYM@CHC"
    game_dir.mkdir(parents=True)
    merged_path = game_dir / "merged.json"
    shutil.copy(os.path.join(FIXTURES, "sample_merged.json"), merged_path)

    snap_dir = tmp_path / "odds_snapshots"
    snap_dir.mkdir()
    shutil.copy(
        os.path.join(FIXTURES, "sample_snapshot.json"),
        snap_dir / "2026-04-18_16-00-ET.json",
    )

    from predict import compute_kelly_block
    import predict
    orig = predict.load_closest_snapshot
    predict.load_closest_snapshot = lambda gde, gsu, snapshot_dir=None: orig(gde, gsu, snapshot_dir=str(snap_dir))

    try:
        with open(merged_path) as f:
            merged = json.load(f)
        formula_pred = {"total": 9.5, "margin": 0.8, "log5_pct": 60.0}
        args = _make_args(merged_path)
        kelly_block = compute_kelly_block(
            args, merged, formula_pred,
            final_ml_rec="CHC", final_ou_rec="OVER", final_rl_rec="PASS",
        )
    finally:
        predict.load_closest_snapshot = orig

    assert kelly_block is not None
    assert kelly_block["ml"] is not None
    assert kelly_block["ml"]["raw_kelly_pct"] > 0
    assert kelly_block["ml"]["capped_pct"] <= 3.0
    assert kelly_block["ou"] is not None
    assert kelly_block["ou"]["line"] == 8.0
    assert kelly_block["rl"] is None
    assert "rl_guardrail_pass" in kelly_block["warnings"]


def test_c1_west_coast_late_game_finds_snapshot(tmp_path):
    """C1 regression: UTC 2026-04-19T02:00:00Z（ET 22:00 前一天）應仍找到 ET 2026-04-18 的 snapshot。"""
    game_dir = tmp_path / "2026-04-18" / "LAD@SF"
    game_dir.mkdir(parents=True)
    merged_path = game_dir / "merged.json"
    with open(os.path.join(FIXTURES, "sample_merged.json")) as f:
        merged = json.load(f)
    merged["_meta"]["game_date"] = "2026-04-19T02:00:00Z"
    merged["_meta"]["home_team"] = "CHC"
    merged["_meta"]["away_team"] = "NYM"
    with open(merged_path, "w") as f:
        json.dump(merged, f)

    snap_dir = tmp_path / "odds_snapshots"
    snap_dir.mkdir()
    shutil.copy(
        os.path.join(FIXTURES, "sample_snapshot.json"),
        snap_dir / "2026-04-18_20-00-ET.json",
    )

    from predict import compute_kelly_block
    import predict
    orig = predict.load_closest_snapshot
    predict.load_closest_snapshot = lambda gde, gsu, snapshot_dir=None: orig(gde, gsu, snapshot_dir=str(snap_dir))

    try:
        args = _make_args(merged_path)
        kelly_block = compute_kelly_block(
            args, merged,
            formula_prediction={"total": 9.5, "margin": 0.8, "log5_pct": 60.0},
            final_ml_rec="CHC", final_ou_rec="OVER", final_rl_rec="PASS",
        )
    finally:
        predict.load_closest_snapshot = orig

    assert kelly_block["ml"] is not None, "C1 bug: west-coast late game snapshot not found"
    assert kelly_block["snapshot_time_et"] is not None
    assert "no_matching_snapshot" not in kelly_block["warnings"]


def test_c2_c3_model_market_split_uses_market_favorite(tmp_path):
    """C2/C3 regression: model 與 market 熱門方分歧時，RL Kelly 必須查 market bucket。"""
    game_dir = tmp_path / "2026-04-18" / "CHC@NYM"
    game_dir.mkdir(parents=True)
    merged_path = game_dir / "merged.json"
    with open(os.path.join(FIXTURES, "sample_merged.json")) as f:
        merged = json.load(f)
    with open(merged_path, "w") as f:
        json.dump(merged, f)

    # 客製 snapshot：home=CHC 冷門 (+140), away=NYM 熱門 (-150), home_point=+1.5
    snap = {
        "snapshot_time_utc": "2026-04-18T20:00:00+00:00",
        "snapshot_time_et": "2026-04-18 16:00 ET",
        "games": [{
            "game": "New York Mets @ Chicago Cubs",
            "away_team": "New York Mets",
            "home_team": "Chicago Cubs",
            "commence_utc": "2026-04-18T23:00:00Z",
            "commence_et": "2026-04-18 19:00 ET",
            "game_date_et": "2026-04-18",
            "bookmakers": {"pinnacle": {
                "title": "Pinnacle",
                "ml": {
                    "Chicago Cubs": {"odds": 2.40, "implied_pct": 41.7},
                    "New York Mets": {"odds": 1.67, "implied_pct": 59.9},
                },
                "ou": {"Over": {"odds": 1.91, "point": 8.5}, "Under": {"odds": 1.95, "point": 8.5}},
                "rl": {
                    "Chicago Cubs": {"odds": 3.00, "point": 1.5},
                    "New York Mets": {"odds": 1.385, "point": -1.5},
                },
            }},
        }],
    }
    snap_dir = tmp_path / "odds_snapshots"
    snap_dir.mkdir()
    with open(snap_dir / "2026-04-18_16-00-ET.json", "w") as f:
        json.dump(snap, f)

    from predict import compute_kelly_block
    import predict
    orig = predict.load_closest_snapshot
    predict.load_closest_snapshot = lambda gde, gsu, snapshot_dir=None: orig(gde, gsu, snapshot_dir=str(snap_dir))

    try:
        args = _make_args(merged_path)
        kelly_block = compute_kelly_block(
            args, merged,
            formula_prediction={"total": 9.0, "margin": 0.5, "log5_pct": 55.0},
            final_ml_rec="CHC", final_ou_rec="OVER",
            final_rl_rec="NYM",
        )
    finally:
        predict.load_closest_snapshot = orig

    rl = kelly_block["rl"]
    assert rl is not None
    assert rl["favorite_side"] == "AWAY_-1.5"
    assert rl["favorite"]["decimal_odds"] == pytest.approx(1.385, abs=0.01)
    assert rl["favorite"]["raw_kelly_pct"] == 0
    assert rl["underdog"]["raw_kelly_pct"] > 0


def test_i1_divergent_forces_ml_kelly_null(tmp_path):
    """I1 regression: final_ml_rec=PASS → kelly.ml 必為 null + warning 紀錄。"""
    game_dir = tmp_path / "2026-04-18" / "NYM@CHC"
    game_dir.mkdir(parents=True)
    merged_path = game_dir / "merged.json"
    shutil.copy(os.path.join(FIXTURES, "sample_merged.json"), merged_path)

    snap_dir = tmp_path / "odds_snapshots"
    snap_dir.mkdir()
    shutil.copy(
        os.path.join(FIXTURES, "sample_snapshot.json"),
        snap_dir / "2026-04-18_16-00-ET.json",
    )

    from predict import compute_kelly_block
    import predict
    orig = predict.load_closest_snapshot
    predict.load_closest_snapshot = lambda gde, gsu, snapshot_dir=None: orig(gde, gsu, snapshot_dir=str(snap_dir))

    try:
        with open(merged_path) as f:
            merged = json.load(f)
        args = _make_args(merged_path)
        kelly_block = compute_kelly_block(
            args, merged,
            formula_prediction={"total": 9.5, "margin": 0.8, "log5_pct": 60.0},
            final_ml_rec="PASS", final_ou_rec="OVER", final_rl_rec="PASS",
        )
    finally:
        predict.load_closest_snapshot = orig

    assert kelly_block["ml"] is None, "I1: PASS 市場的 kelly 必為 null"
    assert "ml_guardrail_pass" in kelly_block["warnings"]
    assert "rl_guardrail_pass" in kelly_block["warnings"]
    assert kelly_block["ou"] is not None


def test_compute_kelly_block_handles_full_team_names_in_meta(tmp_path):
    """Real merged.json stores _meta.home_team as full names (e.g. 'Chicago Cubs').
    compute_kelly_block must convert via TEAM_ABBREV before snapshot lookup.
    """
    game_dir = tmp_path / "2026-04-18" / "NYM@CHC"
    game_dir.mkdir(parents=True)
    merged_path = game_dir / "merged.json"
    with open(os.path.join(FIXTURES, "sample_merged.json")) as f:
        merged = json.load(f)
    # Override to use FULL names (mimicking real project data)
    merged["_meta"]["home_team"] = "Chicago Cubs"
    merged["_meta"]["away_team"] = "New York Mets"
    with open(merged_path, "w") as f:
        json.dump(merged, f)

    snap_dir = tmp_path / "odds_snapshots"
    snap_dir.mkdir()
    shutil.copy(
        os.path.join(FIXTURES, "sample_snapshot.json"),
        snap_dir / "2026-04-18_16-00-ET.json",
    )

    from predict import compute_kelly_block
    import predict
    orig = predict.load_closest_snapshot
    predict.load_closest_snapshot = lambda gde, gsu, snapshot_dir=None: orig(gde, gsu, snapshot_dir=str(snap_dir))

    try:
        args = _make_args(merged_path)
        kelly_block = compute_kelly_block(
            args, merged,
            formula_prediction={"total": 9.5, "margin": 0.8, "log5_pct": 60.0},
            final_ml_rec="CHC", final_ou_rec="OVER", final_rl_rec="PASS",
        )
    finally:
        predict.load_closest_snapshot = orig

    # Before fix: team_name_mismatch warning + ml/ou/rl all None
    # After fix: full-name converts to abbrev, snapshot matches, ml populated
    assert kelly_block["ml"] is not None, \
        "Full team names in _meta must be converted to abbrevs via TEAM_ABBREV"
    assert "team_name_mismatch" not in " ".join(kelly_block["warnings"])
    assert kelly_block["snapshot_time_et"] is not None


# === Task 1: compute_trend_tags bullpen 擴充 ===
from predict import compute_trend_tags


def test_compute_trend_tags_bullpen_slump_home():
    tags = compute_trend_tags({
        "home_bullpen_era": 5.2, "away_bullpen_era": 4.0,
        "home_recent_rs": 4.5, "home_recent_ra": 4.5,
        "home_season_rs": 4.5, "home_season_ra": 4.5,
        "away_recent_rs": 4.5, "away_recent_ra": 4.5,
        "away_season_rs": 4.5, "away_season_ra": 4.5,
    })
    assert "home-bullpen-slump" in tags
    assert "away-bullpen-slump" not in tags
    assert "home-bullpen-strong" not in tags


def test_compute_trend_tags_bullpen_strong_away():
    tags = compute_trend_tags({
        "home_bullpen_era": 4.0, "away_bullpen_era": 2.8,
        "home_recent_rs": 4.5, "home_recent_ra": 4.5,
        "home_season_rs": 4.5, "home_season_ra": 4.5,
        "away_recent_rs": 4.5, "away_recent_ra": 4.5,
        "away_season_rs": 4.5, "away_season_ra": 4.5,
    })
    assert "away-bullpen-strong" in tags
    assert "home-bullpen-slump" not in tags


def test_compute_trend_tags_bullpen_neutral_no_tag():
    tags = compute_trend_tags({
        "home_bullpen_era": 4.0, "away_bullpen_era": 4.0,
        "home_recent_rs": 4.5, "home_recent_ra": 4.5,
        "home_season_rs": 4.5, "home_season_ra": 4.5,
        "away_recent_rs": 4.5, "away_recent_ra": 4.5,
        "away_season_rs": 4.5, "away_season_ra": 4.5,
    })
    assert not any("bullpen" in t for t in tags)


def test_compute_trend_tags_bullpen_boundary_exact_5_is_slump():
    """5.0 應計入 slump（spec: `>= 5.0`）"""
    tags = compute_trend_tags({
        "home_bullpen_era": 5.0, "away_bullpen_era": 3.0,
        "home_recent_rs": 4.5, "home_recent_ra": 4.5,
        "home_season_rs": 4.5, "home_season_ra": 4.5,
        "away_recent_rs": 4.5, "away_recent_ra": 4.5,
        "away_season_rs": 4.5, "away_season_ra": 4.5,
    })
    assert "home-bullpen-slump" in tags
    assert "away-bullpen-strong" in tags  # 3.0 應計入 strong（spec: `<= 3.0`）


# === Task 4: apply_rl_guardrail 8 fixture tests ===
from predict import apply_rl_guardrail


def _rl(**overrides):
    """Shorthand: build apply_rl_guardrail kwargs with sensible defaults.

    Plan B 2026-04-22（W1）：`user_rl_rec` / `user_rl_stars` kwargs 廢除；
    RL 全走 auto override。
    """
    base = dict(
        adj_home=5.0,
        adj_away=3.0,
        trend_tags=[],
        predicted_winner="HOME",
        home_team="Atlanta Braves",
        away_team="New York Mets",
        kelly_rl_available=False,
    )
    base.update(overrides)
    return base


def test_rl1b_mid_diff_strong_tag_1star():
    """Fixture 1: LOW + diff=1.8 + home-bullpen-slump → mid-diff+strong-tag, ATL 1★"""
    rec, stars, ov = apply_rl_guardrail(**_rl(
        adj_home=5.4, adj_away=3.6,                  # diff = 1.8
        trend_tags=["home-bullpen-slump"],
    ))
    assert rec == "ATL"
    assert stars == 1
    assert ov["active"] is True
    assert ov["path"] == "mid-diff+strong-tag"
    assert ov["diff"] == 1.8
    assert ov["stars"] == 1
    assert ov["tags"] == ["home-bullpen-slump"]
    assert ov["warnings"] == []
    assert ov["thresholds"] == {"diff_min": 1.5, "diff_big": 2.2, "diff_star": 2.0}


def test_rl1b_big_diff_no_tag_2star():
    """Fixture 2: LOW + diff=2.3 + 無強 tag → big-diff, ATL 2★"""
    rec, stars, ov = apply_rl_guardrail(**_rl(
        adj_home=6.0, adj_away=3.7,                  # diff = 2.3
        trend_tags=[],
    ))
    assert rec == "ATL"
    assert stars == 2
    assert ov["active"] is True
    assert ov["path"] == "big-diff"
    assert ov["tags"] == []  # pure-diff 路徑，tags 為空 list


def test_rl1b_mid_diff_strong_tag_just_over_star_boundary():
    """Fixture 3: LOW + diff=2.1 + home-pitching-slump → mid-diff+strong-tag, ATL 2★"""
    rec, stars, ov = apply_rl_guardrail(**_rl(
        adj_home=5.6, adj_away=3.5,                  # diff = 2.1
        trend_tags=["home-pitching-slump"],
    ))
    assert rec == "ATL"
    assert stars == 2
    assert ov["active"] is True
    assert ov["path"] == "mid-diff+strong-tag"
    assert ov["stars"] == 2


def test_rl1b_diff_below_min_not_triggered():
    """Fixture 4: LOW + diff=1.4 + home-bullpen-slump → 不觸發（diff<1.5）"""
    rec, stars, ov = apply_rl_guardrail(**_rl(
        adj_home=5.2, adj_away=3.8,                  # diff = 1.4
        trend_tags=["home-bullpen-slump"],
    ))
    assert rec == "PASS"
    # user_rl_stars is None + rec 已是 PASS → RL-2 不觸發；stars 保持 None
    assert stars is None
    assert ov["active"] is False
    assert ov["path"] is None


def test_rl1b_mid_diff_without_strong_tag_not_triggered():
    """Fixture 5: LOW + diff=1.8 + 無強 tag → 不觸發（中分差需 tag）"""
    rec, stars, ov = apply_rl_guardrail(**_rl(
        adj_home=5.4, adj_away=3.6,                  # diff = 1.8
        trend_tags=["home-hot-offense"],             # 非 strong tag
    ))
    assert rec == "PASS"
    assert ov["active"] is False


def test_rl1b_auto_big_diff_triggers_without_user_input():
    """Fixture 6: diff=2.3 + home-pitching-slump → 觸發 big-diff 2★（自動路徑，無 user input）。"""
    rec, stars, ov = apply_rl_guardrail(**_rl(
        adj_home=6.0, adj_away=3.7,                  # diff = 2.3
        trend_tags=["home-pitching-slump"],
    ))
    assert rec == "ATL"
    assert stars == 2
    assert ov["active"] is True
    assert ov["path"] == "big-diff"


def test_rl1b_defensive_direction_mismatch_still_triggers():
    """Fixture 8: diff_side=HOME 但 predicted_winner=AWAY + diff=2.3
    → 仍觸發 big-diff path，HOME abbr 2★，warnings 記 pw_diff_direction_mismatch。

    Q4 defensive check 保留（spec 2026-04-21 §現況觀察）：方向不一致時記錄
    warning 但不阻擋 override（diff_side 是 adj_home/adj_away 的 ground truth）。
    """
    rec, stars, ov = apply_rl_guardrail(**_rl(
        adj_home=6.0, adj_away=3.7,                  # diff_side=HOME, diff=2.3
        predicted_winner="AWAY",                     # 刻意不一致
    ))
    assert rec == "ATL"                              # 仍按 diff_side 推 HOME abbr
    assert stars == 2
    assert ov["active"] is True
    assert ov["path"] == "big-diff"
    assert "pw_diff_direction_mismatch" in ov["warnings"]


# ============================================================================
# Plan B 2026-04-22 — W1: 廢除 --run-line-rec / --run-line-stars
# ============================================================================
import subprocess


def _predict_py_path():
    return os.path.join(os.path.dirname(__file__), "..", "predict.py")


def test_w1_run_line_rec_arg_removed():
    """W1（Plan B §4.2）：--run-line-rec argparse 已廢除，應 reject。"""
    result = subprocess.run(
        [sys.executable, _predict_py_path(), "--test", "--run-line-rec", "NYY"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    combined = (result.stderr + result.stdout)
    assert "unrecognized arguments" in combined or "--run-line-rec" in combined


def test_w1_run_line_stars_arg_removed():
    """W1（Plan B §4.2）：--run-line-stars argparse 已廢除，應 reject。"""
    result = subprocess.run(
        [sys.executable, _predict_py_path(), "--test", "--run-line-stars", "2"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    combined = (result.stderr + result.stdout)
    assert "unrecognized arguments" in combined or "--run-line-stars" in combined


# ============================================================================
# Plan B 2026-04-22 — W2: --ml-rec schema validation
# ============================================================================

def test_w2_ml_rec_accepts_valid_abbr():
    from predict import validate_ml_rec, TEAM_ABBREV
    validate_ml_rec("NYY", set(TEAM_ABBREV.values()))


def test_w2_ml_rec_accepts_pass():
    from predict import validate_ml_rec, TEAM_ABBREV
    validate_ml_rec("PASS", set(TEAM_ABBREV.values()))


def test_w2_ml_rec_accepts_none():
    from predict import validate_ml_rec, TEAM_ABBREV
    validate_ml_rec(None, set(TEAM_ABBREV.values()))


def test_w2_ml_rec_rejects_literal_home():
    from predict import validate_ml_rec, TEAM_ABBREV
    with pytest.raises(SystemExit):
        validate_ml_rec("HOME", set(TEAM_ABBREV.values()))


def test_w2_ml_rec_rejects_literal_away():
    from predict import validate_ml_rec, TEAM_ABBREV
    with pytest.raises(SystemExit):
        validate_ml_rec("AWAY", set(TEAM_ABBREV.values()))


def test_w2_ml_rec_rejects_bogus():
    from predict import validate_ml_rec, TEAM_ABBREV
    with pytest.raises(SystemExit):
        validate_ml_rec("ZZZ", set(TEAM_ABBREV.values()))


# ============================================================================
# Plan B 2026-04-22 — W4: --game-data path regex validation
# ============================================================================

def test_w4_game_data_valid_unix_path():
    from predict import validate_game_data_path
    validate_game_data_path("analysis-data/2026-04-23/NYY@BOS/merged.json")
    validate_game_data_path("/abs/path/to/analysis-data/2026-04-23/NYY@BOS/merged.json")


def test_w4_game_data_valid_windows_path():
    from predict import validate_game_data_path
    validate_game_data_path(r"C:\projects\analysis-data\2026-04-23\NYY@BOS\merged.json")


def test_w4_game_data_valid_doubleheader():
    from predict import validate_game_data_path
    validate_game_data_path("analysis-data/2026-04-23/NYY@BOS-G1/merged.json")
    validate_game_data_path("analysis-data/2026-04-23/NYY@BOS-G2/merged.json")


def test_w4_game_data_rejects_missing_date():
    from predict import validate_game_data_path
    with pytest.raises(SystemExit):
        validate_game_data_path("analysis-data/NYY@BOS/merged.json")


def test_w4_game_data_rejects_wrong_filename():
    from predict import validate_game_data_path
    with pytest.raises(SystemExit):
        validate_game_data_path("analysis-data/2026-04-23/NYY@BOS/foo.json")


def test_w4_game_data_rejects_bogus_tmp_path():
    from predict import validate_game_data_path
    with pytest.raises(SystemExit):
        validate_game_data_path("/tmp/foo.json")


# ============================================================================
# Plan B 2026-04-22 — W3: signal_adjustments allowlist warning
# ============================================================================

def test_w3_signal_adjustments_known_prefix_silent(capsys):
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({"bullpen_il_home": 0.3, "weather_mild_hr": 0.1})
    captured = capsys.readouterr()
    assert "unknown signal key" not in captured.err.lower()


def test_w3_signal_adjustments_unknown_warns(capsys):
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({"zzz_totally_bogus_xyz": 0.5})
    captured = capsys.readouterr()
    assert "zzz_totally_bogus_xyz" in captured.err


def test_w3_signal_adjustments_mixed(capsys):
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({
        "bullpen_il_home": 0.3,
        "zzz_bogus": 0.2,
        "park_factor_adj": 0.1,
    })
    captured = capsys.readouterr()
    assert "zzz_bogus" in captured.err
    assert "bullpen_il_home" not in captured.err
    assert "park_factor_adj" not in captured.err


def test_w3_signal_adjustments_empty_or_none(capsys):
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys(None)
    warn_unknown_signal_keys({})
    captured = capsys.readouterr()
    assert captured.err == ""


def test_w3_never_exits():
    """W3: 即便全部未知，也只警告不 exit。"""
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({"bogus1": 1, "bogus2": 2, "bogus3": 3})  # should not raise


def test_w3_team_abbr_prefix_known(capsys):
    """現有 94 個 signal key 中，team abbr prefix（如 nyy_, bal_）應被識別。"""
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({"nyy_lhp_disadvantage": 0.2, "bal_bullpen_depth": -0.1})
    captured = capsys.readouterr()
    assert "nyy_lhp_disadvantage" not in captured.err
    assert "bal_bullpen_depth" not in captured.err


def test_w3_pitcher_suffix_known(capsys):
    """pitcher 個人化 signal（如 castillo_velo_decline）應被識別。"""
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({"castillo_velo_decline": 0.3, "rocker_new_arsenal": 0.2})
    captured = capsys.readouterr()
    assert "castillo_velo_decline" not in captured.err
    assert "rocker_new_arsenal" not in captured.err


# ============================================================================
# Plan B 2026-04-22 — Y-new-2: close game (|adj_diff| < 0.5) cap to 1
# ============================================================================

def test_ynew2_close_game_caps_to_1():
    from predict import apply_close_game_cap
    new_cap, reason = apply_close_game_cap(4.2, 4.5, current_cap=5)
    assert new_cap == 1
    assert reason is not None
    assert "近身戰" in reason


def test_ynew2_wide_game_no_cap():
    from predict import apply_close_game_cap
    new_cap, reason = apply_close_game_cap(4.2, 6.8, current_cap=5)
    assert new_cap == 5
    assert reason is None


def test_ynew2_respects_tighter_existing_cap():
    """current_cap 已更緊時不回升。"""
    from predict import apply_close_game_cap
    new_cap, _ = apply_close_game_cap(4.2, 4.5, current_cap=0)
    assert new_cap == 0


def test_ynew2_boundary_0_5_not_triggered():
    """|diff| == 0.5 正好不觸發（strictly less than）。"""
    from predict import apply_close_game_cap
    new_cap, reason = apply_close_game_cap(4.0, 4.5, current_cap=5)
    assert new_cap == 5
    assert reason is None


# ============================================================================
# Plan B 2026-04-22 — Y-new-3: divergent user tag caps to 2
# ============================================================================

def test_ynew3_divergent_caps_to_2():
    from predict import apply_divergent_user_tag_cap
    new_cap, reason = apply_divergent_user_tag_cap(["divergent"], current_cap=5)
    assert new_cap == 2
    assert reason is not None
    assert "divergent" in reason


def test_ynew3_no_divergent_no_cap():
    from predict import apply_divergent_user_tag_cap
    new_cap, reason = apply_divergent_user_tag_cap(["early-season", "weather"], current_cap=5)
    assert new_cap == 5
    assert reason is None


def test_ynew3_mixed_tags_with_divergent():
    from predict import apply_divergent_user_tag_cap
    new_cap, _ = apply_divergent_user_tag_cap(["early-season", "divergent", "bullpen-il"], current_cap=5)
    assert new_cap == 2


def test_ynew3_respects_tighter_cap():
    from predict import apply_divergent_user_tag_cap
    new_cap, _ = apply_divergent_user_tag_cap(["divergent"], current_cap=1)
    assert new_cap == 1


def test_ynew3_empty_user_tags_no_cap():
    from predict import apply_divergent_user_tag_cap
    new_cap, reason = apply_divergent_user_tag_cap([], current_cap=5)
    assert new_cap == 5
    assert reason is None


# ============================================================================
# Plan B 2026-04-22 — Y-new-1: home 2-star audit tag
# ============================================================================

def test_ynew1_home_2star_triggers_tag():
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("HOME", 2, "NYY") is True


def test_ynew1_away_2star_no_tag():
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("AWAY", 2, "BOS") is False


def test_ynew1_home_3star_no_tag():
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("HOME", 3, "NYY") is False


def test_ynew1_home_2star_pass_no_tag():
    """final_ml_rec == PASS → tag 無意義（推薦已消）。"""
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("HOME", 2, "PASS") is False


def test_ynew1_home_2star_none_rec_no_tag():
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("HOME", 2, None) is False


def test_ynew1_home_none_stars_no_tag():
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("HOME", None, "NYY") is False


# ============================================================================
# Plan B 2026-04-22 — B7 YoY / B10 BABIP trigger helpers
# ============================================================================

def test_pitcher_yoy_triggered_by_era_xera_gap():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 3.50, "xera": 1.88, "ip": 45.0, "prior_year": {"era": 4.00}}) is True


def test_pitcher_yoy_triggered_by_small_ip_era_drop():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 2.50, "xera": 2.40, "ip": 25.0, "prior_year": {"era": 3.80}}) is True


def test_pitcher_yoy_not_triggered_normal():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 3.80, "xera": 3.50, "ip": 45.0, "prior_year": {"era": 3.90}}) is False


def test_pitcher_yoy_boundary_1_5_triggers():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 5.00, "xera": 3.50, "ip": 45.0, "prior_year": {"era": 4.00}}) is True


def test_pitcher_yoy_boundary_just_under_not_triggered():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 5.00, "xera": 3.51, "ip": 45.0, "prior_year": {"era": 4.00}}) is False


def test_pitcher_yoy_none_tolerant():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy(None) is False
    assert pitcher_triggers_yoy({}) is False
    assert pitcher_triggers_yoy({"era": None}) is False


def test_pitcher_yoy_no_prior_year_era_gap_still_triggers():
    """era-xera gap ≥ 1.5 路徑獨立於 prior_year。"""
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 5.00, "xera": 3.40, "ip": 45.0, "prior_year": {"era": None}}) is True


def test_pitcher_yoy_no_prior_year_small_ip_not_triggered():
    """小 IP 但無 prior year 比較對象 → 不觸發 IP 路徑（era gap 獨立路徑若不符也 False）。"""
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 2.50, "xera": 2.40, "ip": 25.0, "prior_year": {"era": None}}) is False


def test_lineup_babip_low_extreme_triggers():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.250}) is True


def test_lineup_babip_high_extreme_triggers():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.380}) is True


def test_lineup_babip_normal_no_trigger():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.300}) is False


def test_lineup_babip_boundary_260_triggers():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.260}) is True


def test_lineup_babip_boundary_370_triggers():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.370}) is True


def test_lineup_babip_none_no_trigger():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": None}) is False
    assert lineup_triggers_babip({}) is False
    assert lineup_triggers_babip(None) is False


# ============================================================================
# Plan B 2026-04-22 Tasks 5.2 / 5.3 — B7 YoY file check + phase3_summary grep
# ============================================================================

def _minimal_merged_json(
    home_era=3.8, home_xera=3.5, home_ip=50.0, home_prior_era=3.9,
    away_era=3.8, away_xera=3.5, away_ip=50.0, away_prior_era=3.9,
    home_babip=0.300, away_babip=0.300,
):
    """產最小合法 merged.json content（dict）。
    預設不觸發任何 B7/B10，測試時覆蓋 fields 來觸發。
    """
    return {
        "_meta": {
            "home_team": "New York Yankees", "away_team": "Boston Red Sox",
            "home_sp": "Max Fried", "away_sp": "Garrett Crochet",
            "home_sp_starts": 3, "away_sp_starts": 3,
            "game_date": "2026-04-23T23:05:00Z", "venue": "Yankee Stadium",
            "game_pk": 999,
        },
        "home_starter_fip": 3.50, "home_starter_k_bb": 18.0, "home_starter_whip": 1.20,
        "away_starter_fip": 3.80, "away_starter_k_bb": 22.0, "away_starter_whip": 1.15,
        "home_batting_xwoba": 0.330, "home_batting_ops": 0.780, "home_batting_k_pct": 20.0,
        "away_batting_xwoba": 0.340, "away_batting_ops": 0.800, "away_batting_k_pct": 21.0,
        "home_pitcher": {
            "era": home_era, "xera": home_xera, "ip": home_ip,
            "era_xera_delta": abs(home_era - home_xera) if (home_era and home_xera) else None,
            "prior_year": {"era": home_prior_era},
        },
        "away_pitcher": {
            "era": away_era, "xera": away_xera, "ip": away_ip,
            "era_xera_delta": abs(away_era - away_xera) if (away_era and away_xera) else None,
            "prior_year": {"era": away_prior_era},
        },
        "home_lineup": {"recent_babip": home_babip},
        "away_lineup": {"recent_babip": away_babip},
        "home_bullpen_era": 4.0, "away_bullpen_era": 4.0, "park_factor": 100,
        "home_season_games": 22, "away_season_games": 22,
        "home_recent_rs": 4.5, "home_recent_ra": 4.5,
        "away_recent_rs": 4.5, "away_recent_ra": 4.5,
    }


def _setup_game_dir(tmp_path, date="2026-04-23", matchup="NYY@BOS", **merged_overrides):
    """建 analysis-data/<date>/<matchup>/merged.json，回傳 game_dir + merged_path。"""
    game_dir = tmp_path / "analysis-data" / date / matchup
    game_dir.mkdir(parents=True)
    merged_path = game_dir / "merged.json"
    merged_path.write_text(json.dumps(_minimal_merged_json(**merged_overrides)))
    return game_dir, merged_path


def test_yoy_check_triggered_missing_file_exits(tmp_path):
    """B7 觸發（home era-xera≥1.5）+ 缺 prior year file → sys.exit with hint。"""
    game_dir, merged_path = _setup_game_dir(
        tmp_path, home_era=5.0, home_xera=3.0, home_ip=45.0, home_prior_era=4.0
    )
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "B7" in result.stderr or "YoY" in result.stderr
    assert "pitcher_stats.py" in result.stderr


def test_yoy_check_triggered_with_file_proceeds_past_yoy(tmp_path):
    """B7 觸發 + prior year file 存在 → YoY check 通過（後續可能因 phase3 check 卡住但不在 B7）。"""
    game_dir, merged_path = _setup_game_dir(
        tmp_path, home_era=5.0, home_xera=3.0, home_ip=45.0, home_prior_era=4.0
    )
    # 建 prior year file
    (game_dir / "home_pitcher_2025.json").write_text(json.dumps({"season": {"era": 4.0}}))
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save", "--skip-phase3-check"],
        capture_output=True, text=True, encoding="utf-8",
    )
    # 要麼 returncode == 0 或 stderr 不含 B7 錯誤
    assert "B7 YoY" not in result.stderr, f"should pass B7 check; stderr: {result.stderr}"


def test_yoy_skip_flag_bypasses(tmp_path):
    """--skip-yoy-check 即便觸發 + 缺檔也不阻擋。"""
    game_dir, merged_path = _setup_game_dir(
        tmp_path, home_era=5.0, home_xera=3.0, home_ip=45.0, home_prior_era=4.0
    )
    # 寫一個 phase3_summary.md 讓 phase3 check 通過
    (game_dir / "phase3_summary.md").write_text("# test summary\n")
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save", "--skip-yoy-check"],
        capture_output=True, text=True, encoding="utf-8",
    )
    # 應跑到 predict.py 後半段（可能成功或因其他原因失敗，但不應是 B7）
    assert "B7 YoY" not in result.stderr


def test_yoy_no_trigger_no_check(tmp_path):
    """無 YoY 觸發 + 缺 prior year file → 也不 exit（因 trigger 不成立）。"""
    game_dir, merged_path = _setup_game_dir(tmp_path)  # 預設不觸發
    (game_dir / "phase3_summary.md").write_text("# test summary\n")
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert "B7 YoY" not in result.stderr


def test_phase3_missing_file_exits(tmp_path):
    """phase3_summary.md 不存在 → exit。"""
    game_dir, merged_path = _setup_game_dir(tmp_path)
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "phase3_summary" in result.stderr


def test_phase3_yoy_trigger_missing_section_exits(tmp_path):
    """B7 觸發 + phase3_summary.md 存在但缺 '## YoY 對比結論' → exit。"""
    game_dir, merged_path = _setup_game_dir(
        tmp_path, home_era=5.0, home_xera=3.0, home_ip=45.0, home_prior_era=4.0
    )
    (game_dir / "home_pitcher_2025.json").write_text(json.dumps({"season": {"era": 4.0}}))
    (game_dir / "phase3_summary.md").write_text(
        "# basic summary\n\n## 投打對決\n無 YoY section\n", encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "phase3_summary" in result.stderr
    assert "YoY 對比結論" in result.stderr


def test_phase3_babip_trigger_missing_section_exits(tmp_path):
    """B10 觸發（BABIP≤.260）+ 缺 '## BABIP 回歸判定' → exit。"""
    game_dir, merged_path = _setup_game_dir(tmp_path, home_babip=0.250)
    (game_dir / "phase3_summary.md").write_text("# basic summary\n")
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "BABIP 回歸判定" in result.stderr


def test_phase3_skip_flag_bypasses(tmp_path):
    """--skip-phase3-check 即便缺 phase3_summary.md 也放行。"""
    game_dir, merged_path = _setup_game_dir(tmp_path)
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save", "--skip-phase3-check"],
        capture_output=True, text=True, encoding="utf-8",
    )
    # phase3 check 不會阻擋；可能 returncode 非 0 但錯誤不是 phase3_summary
    assert "phase3_summary" not in result.stderr


def test_phase3_all_sections_present_passes(tmp_path):
    """所有必要 section 都在 → phase3 check 通過。"""
    game_dir, merged_path = _setup_game_dir(
        tmp_path, home_era=5.0, home_xera=3.0, home_ip=45.0, home_prior_era=4.0,
        home_babip=0.250,
    )
    (game_dir / "home_pitcher_2025.json").write_text(json.dumps({"season": {"era": 4.0}}))
    (game_dir / "phase3_summary.md").write_text(
        "# summary\n\n## YoY 對比結論\n OK\n\n## BABIP 回歸判定\n OK\n", encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save"],
        capture_output=True, text=True, encoding="utf-8",
    )
    # phase3 check 不卡；整體可能成功
    assert "phase3_summary.md 缺必要 section" not in result.stderr
