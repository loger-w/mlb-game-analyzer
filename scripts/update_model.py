#!/usr/bin/env python3
"""MLB Model Updater — 每週追加訓練最新比賽結果"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
WIN_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_win_model.pkl")
TOTAL_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_total_model.pkl")


def main():
    parser = argparse.ArgumentParser(description="Update MLB prediction models with recent data")
    parser.add_argument("--since", help="Start date (YYYY-MM-DD), default: 7 days ago")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "update_model test mode"}))
        return

    if not os.path.exists(WIN_MODEL_PATH):
        print(json.dumps({"error": "No existing model found. Run train.py first."}))
        sys.exit(1)

    since = args.since
    if not since:
        since = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    print(f"Updating models with data since {since}...", file=sys.stderr)

    # 重新跑 train.py 的邏輯（含最新年份）
    # 簡化版：直接重新訓練當年度
    current_year = datetime.now().year
    from train import build_season_data, create_training_features

    team_stats = build_season_data(current_year)
    X_new, y_win_new, y_total_new = create_training_features(team_stats)

    # 載入現有模型，用新資料追加訓練
    win_model = joblib.load(WIN_MODEL_PATH)
    total_model = joblib.load(TOTAL_MODEL_PATH)

    win_model.fit(X_new, y_win_new, xgb_model=win_model.get_booster())
    total_model.fit(X_new, y_total_new, xgb_model=total_model.get_booster())

    joblib.dump(win_model, WIN_MODEL_PATH)
    joblib.dump(total_model, TOTAL_MODEL_PATH)

    print(json.dumps({
        "status": "OK",
        "updated_since": since,
        "new_samples": X_new.shape[0],
    }, indent=2))


if __name__ == "__main__":
    main()
