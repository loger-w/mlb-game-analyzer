import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.render import render_report, render_details_csv


def _sample_data():
    df = pd.DataFrame([{
        "date": "2026-05-02", "matchup": "BAL@NYY", "game_pk": 1,
        "skill_direction": "HOME", "skill_total": 8.5, "skill_confidence": "MEDIUM",
        "skill_confidence_pct": 0.62,
        "market_home_winprob_no_vig": 0.60, "market_total_line": 8.5,
        "market_favorite": "HOME", "market_favorite_winprob": 0.60,
        "actual_winner": "AWAY", "actual_total": 13, "actual_home_score": 5, "actual_away_score": 8,
        "park_factor": 105.0,
        "has_reverse_platoon": True, "has_chain_break_300": False, "has_bullpen_il_2plus": True,
        "closing_missing": False, "closing_snapshot_ts": "2026-05-02_12-00-ET.json",
        "result_missing": False,
        "dossier_path": "analysis-data/2026-05-02/BAL@NYY/dossier.md",
    }])
    direction_metrics = {
        "skill_n": 1, "skill_hit_rate": 0.0, "skill_ci": (0.0, 0.95),
        "market_n": 1, "market_hit_rate": 0.0, "market_ci": (0.0, 0.95),
        "edge_pp": 0.0,
        "skill_aligned_n": 1, "skill_aligned_hit_rate": 0.0,
        "skill_against_n": 0, "skill_against_hit_rate": None,
    }
    total_metrics = {"n": 1, "total_mae": 4.5, "total_bias": -4.5, "ou_n": 1, "ou_hit_rate": 0.0}
    calibration = {
        "reliability_table": pd.DataFrame([
            {"confidence": "MEDIUM", "n": 1, "mapped_p": 0.62, "actual_hit_rate": 0.0, "delta": -0.62, "ci_low": 0, "ci_high": 0.95},
        ]),
        "brier_score": 0.38, "log_loss": 0.95,
        "brier_baseline_market": 0.36, "log_loss_baseline_market": 0.92,
    }
    slices = pd.DataFrame({"n": [1], "dir_n": [1], "dir_hit": [0.0], "ou_hit": [0.0], "mae": [4.5], "bias": [-4.5]},
                         index=["direction_HOME"])
    failures = df.copy()
    failures["dual_failure"] = True
    failures["is_direction_miss"] = True
    failures["is_top_total_miss"] = True
    failures["main_signal"] = "Yankees 中度偏優"
    return df, direction_metrics, total_metrics, calibration, slices, failures


def test_render_report_contains_required_sections(tmp_path):
    df, dm, tm, cal, sl, fc = _sample_data()
    out = tmp_path / "report.md"
    render_report(df=df, direction_metrics=dm, total_metrics=tm,
                  calibration=cal, slices=sl, failure_cases=fc,
                  month="2026-05", out_path=out)
    content = out.read_text(encoding="utf-8")
    assert "# MLB Skill 回測 — 2026 年 5 月" in content
    assert "## 資料健康度" in content
    assert "## TL;DR" in content
    assert "## 1. 方向類指標" in content
    assert "## 2. 總分類指標" in content
    assert "## 3. 信心 Calibration" in content
    assert "## 4. 分組切片" in content
    assert "## 5. 失敗案例" in content
    assert "BAL@NYY" in content


def test_render_details_csv_has_all_columns(tmp_path):
    df, *_ = _sample_data()
    out = tmp_path / "details.csv"
    render_details_csv(df, out_path=out)
    csv_text = out.read_text(encoding="utf-8")
    header = csv_text.splitlines()[0]
    # Required columns (must include skill_confidence_pct from Task 6 amendment)
    for col in ["date", "matchup", "game_pk", "skill_direction", "skill_confidence_pct",
                "actual_winner", "park_factor", "has_reverse_platoon", "closing_snapshot_ts"]:
        assert col in header
