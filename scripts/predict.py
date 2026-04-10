#!/usr/bin/env python3
"""MLB Game Predictor — XGBoost 預測 + Log5 交叉驗證 + 信號計分表"""

import argparse
import json
import math
import os
import sys

import joblib
import numpy as np

# Fix Windows encoding for emoji output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
WIN_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_win_model.pkl")
TOTAL_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_total_model.pkl")

FEATURE_COLS = [
    "home_starter_fip", "home_starter_k_bb", "home_starter_whip",
    "away_starter_fip", "away_starter_k_bb", "away_starter_whip",
    "home_batting_xwoba", "home_batting_ops", "home_batting_k_pct",
    "away_batting_xwoba", "away_batting_ops", "away_batting_k_pct",
    "home_bullpen_era", "away_bullpen_era",
    "home_recent_rs", "home_recent_ra",
    "away_recent_rs", "away_recent_ra",
    "park_factor",
]


def log5(home_pct: float, away_pct: float) -> float:
    """Log5 勝率公式"""
    p = (home_pct * (1 - away_pct)) / (home_pct * (1 - away_pct) + away_pct * (1 - home_pct))
    return p


def pythagorean_runs(rs: float, ra: float, g: float = 10) -> float:
    """Pythagenport 動態指數公式（與 reference/teams-and-api.md 一致）

    exponent = 1.50 × log10[(RS + RA) / G] + 0.45
    Pythagenport RMSE = 3.991 勝（優於固定指數 1.83 的 4.126）
    """
    if rs + ra == 0:
        return 0.5
    exponent = 1.50 * math.log10((rs + ra) / g) + 0.45
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

    # 打線 xwOBA
    home_xwoba = data.get("home_batting_xwoba", 0.320)
    away_xwoba = data.get("away_batting_xwoba", 0.320)
    if home_xwoba >= 0.350 and away_xwoba >= 0.350:
        over_signals.append({"signal": "雙方打線 xwOBA ≥ .350", "score": 1})

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
    """用 XGBoost 模型預測（若模型特徵不匹配則 graceful fallback）"""
    if not os.path.exists(WIN_MODEL_PATH) or not os.path.exists(TOTAL_MODEL_PATH):
        return None

    try:
        win_model = joblib.load(WIN_MODEL_PATH)
        total_model = joblib.load(TOTAL_MODEL_PATH)

        X = np.array([features])
        win_prob = float(win_model.predict_proba(X)[0][1])  # 主隊勝率
        total_runs = float(total_model.predict(X)[0])
    except Exception:
        # 模型特徵欄位不匹配時 graceful fallback
        return None

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
    parser.add_argument("--save", action="store_true",
                        help="Append prediction to scripts/predictions.jsonl")
    parser.add_argument("--test", action="store_true")
    # 分析後手動傳入的欄位
    parser.add_argument("--adjusted-home", type=float, help="分析後調整的主隊得分")
    parser.add_argument("--adjusted-away", type=float, help="分析後調整的客隊得分")
    parser.add_argument("--ou-line", type=float, help="大小分線（有效線，如四分球取中位 9.75）")
    parser.add_argument("--ou-rec", choices=["OVER", "UNDER", "PASS"], help="大小分推薦")
    parser.add_argument("--ml-rec", help="獨贏推薦（隊伍縮寫或 PASS）")
    parser.add_argument("--ml-stars", type=int, choices=[0, 1, 2, 3, 4, 5], help="獨贏星級")
    parser.add_argument("--run-line-rec", help="讓分推薦（隊伍縮寫或 PASS）")
    parser.add_argument("--signal-adjustments", type=json.loads,
                        help='信號修正 JSON，例如 \'{"puk_il":0.3}\'')
    parser.add_argument("--tags", help="逗號分隔標籤，例如 divergent,early-season")
    parser.add_argument("--temperature", type=float, help="氣溫（°F）")
    parser.add_argument("--wind-mph", type=float, help="風速（mph）")
    parser.add_argument("--wind-direction", help="風向")
    parser.add_argument("--umpire", help="主審姓名")
    parser.add_argument("--umpire-ou-rate", type=float, help="主審 Over% 歷史")
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

    # 自動存檔到 predictions.jsonl
    if args.save:
        from datetime import datetime as _dt
        meta = data.get("_meta", {})
        home_team = meta.get("home_team") or "HOME"
        away_team = meta.get("away_team") or "AWAY"

        adj_home = args.adjusted_home if args.adjusted_home is not None else final_home
        adj_away = args.adjusted_away if args.adjusted_away is not None else final_away
        adj_total = round(adj_home + adj_away, 1)

        raw_date = (meta.get("game_date") or "")[:10]
        record_date = raw_date if raw_date else _dt.now().strftime("%Y-%m-%d")

        record = {
            "date": record_date,
            "game": f"{away_team} vs {home_team}",
            "home_team": home_team,
            "away_team": away_team,
            "home_sp": meta.get("home_sp"),
            "away_sp": meta.get("away_sp"),
            "home_sp_starts": meta.get("home_sp_starts"),
            "away_sp_starts": meta.get("away_sp_starts"),
            "venue": meta.get("venue"),
            "game_pk": meta.get("game_pk"),
            "predicted_winner": result["final"]["recommended_winner"],
            "predicted_home_pct": result["final"]["home_win_pct"],
            "predicted_home_score": adj_home,
            "predicted_away_score": adj_away,
            "predicted_total": adj_total,
            "formula_home_score": final_home,
            "formula_away_score": final_away,
            "adjusted_total": adj_total,
            "signal_adjustments": args.signal_adjustments if args.signal_adjustments is not None else {},
            "ou_line": args.ou_line,
            "ou_rec": args.ou_rec if args.ou_rec is not None else result["final"]["over_under_lean"],
            "run_line_rec": args.run_line_rec if args.run_line_rec is not None else "PASS",
            "ml_rec": args.ml_rec,
            "ml_stars": args.ml_stars,
            "confidence": result["final"]["confidence"],
            "cross_validation": result["cross_validation"],
            "tags": [t.strip() for t in args.tags.split(",")] if args.tags else [],
            "park_factor": data.get("park_factor"),
            "temperature_f": args.temperature,
            "wind_mph": args.wind_mph,
            "wind_direction": args.wind_direction,
            "umpire_name": args.umpire,
            "umpire_ou_rate": args.umpire_ou_rate,
            "actual_winner": None,
            "actual_home_score": None,
            "actual_away_score": None,
            "actual_total": None,
            "verified": False,
        }
        jsonl_path = os.path.join(os.path.dirname(__file__), "predictions.jsonl")
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved to {jsonl_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
