import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import load


def _write_game(base: Path, date: str, matchup: str):
    d = base / date / matchup
    d.mkdir(parents=True)
    (d / "features.json").write_text(json.dumps({
        "schema_version": 2,
        "game": {"date": date, "game_pk": 1, "home": "H", "away": "A", "venue": "X"},
        "model": {"mu_total": 8.28, "p_home_cover_rl": 0.355, "p_over": 0.479,
                  "p_home_ml": 0.492},
        "odds": {"snapshot_file": "s.json",
                 "rl": {"home_point": -1.5, "home_no_vig": 0.41,
                        "away_point": 1.5, "away_no_vig": 0.59},
                 "total": {"line": 8.5, "over_no_vig": 0.52, "under_no_vig": 0.48}},
        "edges": {"home_rl_pp": -5.5, "over_pp": -4.1},
    }, ensure_ascii=False), encoding="utf-8")
    (d / "result.json").write_text(json.dumps({
        "winner": "HOME", "home_score": 6, "away_score": 3, "total": 9,
    }, ensure_ascii=False), encoding="utf-8")


def test_load_features_v2(tmp_path, monkeypatch):
    monkeypatch.setattr(load, "ANALYSIS_DATA_DIR", tmp_path)
    _write_game(tmp_path, "2026-05-29", "ARI@COL")
    df = load.build_dataframe_for_month("2026-05")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["p_over"] == 0.479
    assert row["over_pp"] == -4.1
    assert row["actual_total"] == 9
    assert row["actual_margin"] == 3      # home 6 - away 3
    assert row["result_missing"] == False
    assert row["odds_missing"] == False


def test_load_skips_non_v2(tmp_path, monkeypatch):
    monkeypatch.setattr(load, "ANALYSIS_DATA_DIR", tmp_path)
    d = tmp_path / "2026-05-29" / "ARI@COL"
    d.mkdir(parents=True)
    (d / "features.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    df = load.build_dataframe_for_month("2026-05")
    assert len(df) == 0


def test_load_includes_team_names(tmp_path, monkeypatch):
    from lib import load as load_mod
    d = tmp_path / "2026-05-02" / "BAL@NYY"
    d.mkdir(parents=True)
    feats = {"schema_version": 2,
             "game": {"game_pk": 1, "date": "2026-05-02",
                      "home": "New York Yankees", "away": "Baltimore Orioles"},
             "model": {}, "odds": None, "edges": {}}
    (d / "features.json").write_text(json.dumps(feats), encoding="utf-8")
    monkeypatch.setattr(load_mod, "ANALYSIS_DATA_DIR", tmp_path)
    df = load_mod.build_dataframe_for_month("2026-05")
    assert df.iloc[0]["home_team"] == "New York Yankees"
    assert df.iloc[0]["away_team"] == "Baltimore Orioles"


def test_load_skips_unknown_schema_versions(tmp_path, monkeypatch):
    """schema_version 嚴格 ==2(int):3、'2'(字串)、缺欄全部跳過。"""
    monkeypatch.setattr(load, "ANALYSIS_DATA_DIR", tmp_path)
    for abbr, sv in zip(("ARI", "ATL", "BOS"),
                        ({"schema_version": 3}, {"schema_version": "2"}, {})):
        d = tmp_path / "2026-05-29" / f"{abbr}@COL"
        d.mkdir(parents=True)
        (d / "features.json").write_text(json.dumps({
            **sv, "game": {"date": "2026-05-29", "game_pk": 1, "home": "H", "away": "A"},
            "model": {"p_home_cover_rl": 0.5, "p_over": 0.5},
        }), encoding="utf-8")
    df = load.build_dataframe_for_month("2026-05")
    assert len(df) == 0
