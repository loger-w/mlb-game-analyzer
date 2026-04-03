#!/usr/bin/env python3
"""MLB Game Predictor — XGBoost 預測 + Log5 交叉驗證 + 信號計分表"""

import argparse
import json
import math
import os
import sys

import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
WIN_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_win_model.pkl")
TOTAL_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_total_model.pkl")

FEATURE_COLS = [
    "home_starter_fip", "home_starter_k_bb", "home_starter_whip",
    "away_starter_fip", "away_starter_k_bb", "away_starter_whip",
    "home_batting_wrc", "home_batting_ops", "home_batting_k_pct",
    "away_batting_wrc", "away_batting_ops", "away_batting_k_pct",
    "home_bullpen_era", "away_bullpen_era",
    "home_recent_rs", "home_recent_ra",
    "away_recent_rs", "away_recent_ra",
    "park_factor",
]


def log5(home_pct: float, away_pct: float) -> float:
    """Log5 勝率公式"""
    p = (home_pct * (1 - away_pct)) / (home_pct * (1 - away_pct) + away_pct * (1 - home_pct))
    return p


def pythagorean_runs(rs: float, ra: float, exponent: float = 1.83) -> float:
    """Pythagorean 期望勝率"""
    if rs + ra == 0:
        return 0.5
    return (rs ** exponent) / (rs ** exponent + ra ** exponent)


def compute_signal_table(data: dict) -> dict:
    """計算大小分信號計分表"""
    over_signals = []
    under_signals = []

    # 打線近期火力
    home_rs = data.get("home_recent_rs", 4.5)
    away_rs = data.get("away_recent_rs", 4.5)
    if home_rs >= 5 and away_rs >= 5:
        over_signals.append({"signal": "雙方打線近期 Hot（場均 ≥ 5 分）", "score": 2})
    if home_rs <= 2 and away_rs <= 2:
        under_signals.append({"signal": "雙方打線近期 Cold（場均 ≤ 2 分）", "score": -2})

    # 牛棚
    home_bp = data.get("home_bullpen_era", 4.0)
    away_bp = data.get("away_bullpen_era", 4.0)
    if home_bp >= 5.0 or away_bp >= 5.0:
        over_signals.append({"signal": "牛棚 ERA ≥ 5.0", "score": 2})

    # 球場
    pf = data.get("park_factor", 100)
    if pf >= 105:
        over_signals.append({"signal": f"Park Factor {pf} ≥ 105", "score": 1})
    if pf <= 95:
        under_signals.append({"signal": f"Park Factor {pf} ≤ 95", "score": -1})

    # 先發投手等級
    home_fip = data.get("home_starter_fip", 4.0)
    away_fip = data.get("away_starter_fip", 4.0)
    if home_fip >= 5.0 and away_fip >= 5.0:
        over_signals.append({"signal": "雙方先發 FIP ≥ 5.0（Back-end 以下）", "score": 1})
    if home_fip <= 3.0 and away_fip <= 3.0:
        under_signals.append({"signal": "雙方先發 FIP ≤ 3.0（Ace 級）", "score": -2})

    # 打線 K%
    home_k = data.get("home_batting_k_pct", 20)
    away_k = data.get("away_batting_k_pct", 20)
    if home_k >= 25 and away_k >= 25:
        under_signals.append({"signal": "雙方打線 K% ≥ 25%", "score": -1})

    # 打線 wRC+
    home_wrc = data.get("home_batting_wrc", 100)
    away_wrc = data.get("away_batting_wrc", 100)
    if home_wrc >= 110 and away_wrc >= 110:
        over_signals.append({"signal": "雙方打線 wRC+ ≥ 110", "score": 1})

    over_total = sum(s["score"] for s in over_signals)
    under_total = sum(s["score"] for s in under_signals)
    net_score = over_total + under_total  # under_signals 已經是負數

    return {
        "over_signals": over_signals,
        "under_signals": under_signals,
        "over_total": over_total,
        "under_total": under_total,
        "net_score": net_score,
    }


def predict_with_ml(features: list[float]) -> dict | None:
    """用 XGBoost 模型預測"""
    if not os.path.exists(WIN_MODEL_PATH) or not os.path.exists(TOTAL_MODEL_PATH):
        return None

    win_model = joblib.load(WIN_MODEL_PATH)
    total_model = joblib.load(TOTAL_MODEL_PATH)

    X = np.array([features])
    win_prob = float(win_model.predict_proba(X)[0][1])  # 主隊勝率
    total_runs = float(total_model.predict(X)[0])

    # 分配得分（基於勝率比例）
    home_ratio = win_prob / (win_prob + (1 - win_prob)) * 1.05  # 微調主場
    home_score = round(total_runs * home_ratio / 2, 1)
    away_score = round(total_runs - home_score, 1)

    return {
        "home_win_pct": round(win_prob * 100, 1),
        "home_score": home_score,
        "away_score": away_score,
        "total": round(home_score + away_score, 1),
    }


def predict_with_formula(data: dict) -> dict:
    """用 Log5 + Pythagorean 公式預測"""
    home_rs = data.get("home_recent_rs", 4.5)
    home_ra = data.get("home_recent_ra", 4.5)
    away_rs = data.get("away_recent_rs", 4.5)
    away_ra = data.get("away_recent_ra", 4.5)
    pf = data.get("park_factor", 100)

    home_pct = pythagorean_runs(home_rs, home_ra)
    away_pct = pythagorean_runs(away_rs, away_ra)
    log5_pct = log5(home_pct, away_pct)
    # 主場優勢 +3%
    log5_pct = min(log5_pct + 0.03, 0.95)

    # Park Factor 修正
    pf_mult = pf / 100
    home_score = round(home_rs * pf_mult, 1)
    away_score = round(away_rs * (2 - pf_mult), 1)  # 反向修正客隊

    return {
        "log5_pct": round(log5_pct * 100, 1),
        "pythag_home_pct": round(home_pct * 100, 1),
        "pythag_away_pct": round(away_pct * 100, 1),
        "home_score": home_score,
        "away_score": away_score,
        "total": round(home_score + away_score, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict MLB game outcome")
    parser.add_argument("--game-data", required=True, help="Path to JSON with merged game data")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "predict test mode"}))
        return

    with open(args.game_data, "r") as f:
        data = json.load(f)

    # 建構特徵向量
    features = [data.get(col, 0) for col in FEATURE_COLS]

    # ML 預測
    ml_pred = predict_with_ml(features)

    # 公式預測
    formula_pred = predict_with_formula(data)

    # 信號計分表
    signal_table = compute_signal_table(data)

    # 交叉驗證
    cross_validation = "NO_ML_MODEL"
    if ml_pred:
        ml_lean = "HOME" if ml_pred["home_win_pct"] > 50 else "AWAY"
        formula_lean = "HOME" if formula_pred["log5_pct"] > 50 else "AWAY"
        pct_diff = abs(ml_pred["home_win_pct"] - formula_pred["log5_pct"])
        cross_validation = "CONSISTENT" if ml_lean == formula_lean and pct_diff < 15 else "DIVERGENT"

    # 最終推薦
    # 勝率：有 ML 時用 ML（XGBoost 勝率預測可靠）
    # 比分：一律用 formula（ML 的 total_model 訓練資料有結構性缺陷，比分不可靠）
    if ml_pred:
        final_pct = ml_pred["home_win_pct"]
    else:
        final_pct = formula_pred["log5_pct"]
    final_home = formula_pred["home_score"]
    final_away = formula_pred["away_score"]

    result = {
        "ml_prediction": ml_pred,
        "formula_prediction": formula_pred,
        "cross_validation": cross_validation,
        "signal_table": signal_table,
        "final": {
            "recommended_winner": "HOME" if final_pct > 50 else "AWAY",
            "home_win_pct": round(final_pct, 1),
            "confidence": "HIGH" if cross_validation == "CONSISTENT" else ("MEDIUM" if cross_validation == "NO_ML_MODEL" else "LOW"),
            "predicted_home_score": final_home,
            "predicted_away_score": final_away,
            "predicted_total": round(final_home + final_away, 1),
            "over_under_lean": "OVER" if signal_table["net_score"] > 0 else ("UNDER" if signal_table["net_score"] < 0 else "NEUTRAL"),
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
