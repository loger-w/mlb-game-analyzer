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
        ml_pred = {"home_win_pct": 60.0}
        formula_pred = {"total": 9.5, "margin": 0.8}
        args = _make_args(merged_path)
        kelly_block = compute_kelly_block(
            args, merged, ml_pred, formula_pred,
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
            ml_prediction={"home_win_pct": 60.0},
            formula_prediction={"total": 9.5, "margin": 0.8},
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
            ml_prediction={"home_win_pct": 55.0},
            formula_prediction={"total": 9.0, "margin": 0.5},
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
            ml_prediction={"home_win_pct": 60.0},
            formula_prediction={"total": 9.5, "margin": 0.8},
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
            ml_prediction={"home_win_pct": 60.0},
            formula_prediction={"total": 9.5, "margin": 0.8},
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
# should_force_ml_pass helper tests (α 實作 — D1 改讀 ml_lean vs formula_lean)
# ============================================================================
from predict import should_force_ml_pass


def test_should_force_ml_pass_direction_mismatch_returns_true():
    """ml 看 HOME、formula 看 AWAY → 方向分歧 → True"""
    assert should_force_ml_pass(
        ml_pred={"home_win_pct": 60},
        formula_pred={"log5_pct": 40},
    ) is True


def test_should_force_ml_pass_both_lean_home_returns_false():
    """ml 看 HOME、formula 看 HOME → 方向一致 → False（不 force PASS）"""
    assert should_force_ml_pass(
        ml_pred={"home_win_pct": 60},
        formula_pred={"log5_pct": 55},
    ) is False


def test_should_force_ml_pass_both_lean_away_returns_false():
    """ml 看 AWAY、formula 看 AWAY → 方向一致 → False"""
    assert should_force_ml_pass(
        ml_pred={"home_win_pct": 40},
        formula_pred={"log5_pct": 45},
    ) is False


def test_should_force_ml_pass_none_ml_pred_returns_false():
    """ml_pred 缺失（NO_ML_MODEL）→ False（無模型不比對）"""
    assert should_force_ml_pass(
        ml_pred=None,
        formula_pred={"log5_pct": 60},
    ) is False


def test_should_force_ml_pass_none_formula_pred_returns_false():
    """formula_pred 缺失 → False"""
    assert should_force_ml_pass(
        ml_pred={"home_win_pct": 60},
        formula_pred=None,
    ) is False


def test_should_force_ml_pass_boundary_50_home_formula_45_returns_true():
    """ml=50.1 (HOME) vs formula=45 (AWAY) → 分歧 → True"""
    assert should_force_ml_pass(
        ml_pred={"home_win_pct": 50.1},
        formula_pred={"log5_pct": 45},
    ) is True


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
# Plan B 2026-04-22 — Y2: xgb_home_lean vs predicted_winner divergent force PASS
# ============================================================================

def test_y2_xgb_diverges_returns_true():
    """xgb 61% HOME but predicted_winner AWAY → True。"""
    from predict import check_xgb_divergent
    assert check_xgb_divergent({"home_win_pct": 61.0}, "AWAY") is True


def test_y2_xgb_aligned_home_returns_false():
    from predict import check_xgb_divergent
    assert check_xgb_divergent({"home_win_pct": 58.0}, "HOME") is False


def test_y2_xgb_aligned_away_returns_false():
    from predict import check_xgb_divergent
    assert check_xgb_divergent({"home_win_pct": 42.0}, "AWAY") is False


def test_y2_ml_pred_none_returns_false():
    from predict import check_xgb_divergent
    assert check_xgb_divergent(None, "HOME") is False


def test_y2_boundary_50_treated_as_away_lean():
    """home_win_pct == 50.0 → AWAY lean（> 50 才算 HOME）。"""
    from predict import check_xgb_divergent
    assert check_xgb_divergent({"home_win_pct": 50.0}, "HOME") is True
    assert check_xgb_divergent({"home_win_pct": 50.0}, "AWAY") is False


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
