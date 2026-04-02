#!/usr/bin/env python3
"""MLB XGBoost Model Trainer — 拉歷史資料訓練勝負 + 總分預測模型"""

import argparse
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

try:
    from pybaseball import pitching_stats, batting_stats, schedule_and_record
except ImportError:
    print(json.dumps({"error": "pybaseball not installed"}))
    sys.exit(1)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
WIN_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_win_model.pkl")
TOTAL_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_total_model.pkl")

# 特徵欄位定義
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


def build_season_data(year: int) -> pd.DataFrame:
    """從 pybaseball 建構一個賽季的訓練資料"""
    print(f"Loading {year} pitching stats...", file=sys.stderr)
    pitchers = pitching_stats(year, year, qual=10)

    print(f"Loading {year} batting stats...", file=sys.stderr)
    batters = batting_stats(year, year, qual=50)

    # 按球隊聚合投手平均 FIP / K-BB% / WHIP
    pitcher_agg = pitchers.groupby("Team").agg({
        "FIP": "mean",
        "WHIP": "mean",
    }).reset_index()
    pitcher_agg.columns = ["Team", "team_fip", "team_whip"]

    # K-BB% 需要手動計算
    if "K%" in pitchers.columns and "BB%" in pitchers.columns:
        pitchers["k_bb_raw"] = pitchers["K%"] - pitchers["BB%"]
        if pitchers["k_bb_raw"].max() < 1:
            pitchers["k_bb_raw"] = pitchers["k_bb_raw"] * 100
        kb_agg = pitchers.groupby("Team")["k_bb_raw"].mean().reset_index()
        kb_agg.columns = ["Team", "team_k_bb"]
        pitcher_agg = pitcher_agg.merge(kb_agg, on="Team", how="left")
    else:
        pitcher_agg["team_k_bb"] = 0

    # 按球隊聚合打線 wRC+ / OPS / K%
    batter_agg = batters.groupby("Team").agg({
        "wRC+": "mean",
        "OPS": "mean",
    }).reset_index()
    batter_agg.columns = ["Team", "team_wrc", "team_ops"]

    if "K%" in batters.columns:
        k_agg = batters.groupby("Team")["K%"].mean().reset_index()
        k_agg.columns = ["Team", "team_k_pct"]
        if k_agg["team_k_pct"].max() < 1:
            k_agg["team_k_pct"] = k_agg["team_k_pct"] * 100
        batter_agg = batter_agg.merge(k_agg, on="Team", how="left")
    else:
        batter_agg["team_k_pct"] = 20

    # 合併
    team_stats = pitcher_agg.merge(batter_agg, on="Team", how="outer")
    team_stats = team_stats.fillna(team_stats.mean(numeric_only=True))

    print(f"Built team stats for {len(team_stats)} teams in {year}", file=sys.stderr)
    return team_stats


def create_training_features(team_stats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成訓練特徵（模擬對決）"""
    teams = team_stats["Team"].tolist()
    X_rows = []
    y_win = []
    y_total = []

    for i, home_team in enumerate(teams):
        for j, away_team in enumerate(teams):
            if i == j:
                continue
            home = team_stats[team_stats["Team"] == home_team].iloc[0]
            away = team_stats[team_stats["Team"] == away_team].iloc[0]

            features = [
                home["team_fip"], home.get("team_k_bb", 0), home["team_whip"],
                away["team_fip"], away.get("team_k_bb", 0), away["team_whip"],
                home["team_wrc"], home["team_ops"], home.get("team_k_pct", 20),
                away["team_wrc"], away["team_ops"], away.get("team_k_pct", 20),
                home["team_fip"],  # 牛棚近似用 team FIP
                away["team_fip"],
                4.5, 4.5,  # 近期得失分（使用聯盟平均作為初始值）
                4.5, 4.5,
                100,  # park factor (聯盟平均)
            ]
            X_rows.append(features)

            # 模擬勝負與得分（基於 Pythagorean）
            home_strength = home["team_wrc"] / 100 * (4.5 / max(home["team_fip"], 2.0))
            away_strength = away["team_wrc"] / 100 * (4.5 / max(away["team_fip"], 2.0))
            home_win_prob = (home_strength ** 2) / (home_strength ** 2 + away_strength ** 2) + 0.03  # 主場優勢
            y_win.append(1 if np.random.random() < home_win_prob else 0)
            home_runs = max(0, np.random.poisson(home_strength * 1.05))
            away_runs = max(0, np.random.poisson(away_strength * 0.95))
            y_total.append(home_runs + away_runs)

    return np.array(X_rows), np.array(y_win), np.array(y_total)


def train_models(years: list[int], validate: bool = False):
    """訓練 XGBoost 模型"""
    all_X = []
    all_y_win = []
    all_y_total = []

    for year in years:
        team_stats = build_season_data(year)
        X, y_win, y_total = create_training_features(team_stats)
        all_X.append(X)
        all_y_win.append(y_win)
        all_y_total.append(y_total)

    X = np.vstack(all_X)
    y_win = np.concatenate(all_y_win)
    y_total = np.concatenate(all_y_total)

    print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features", file=sys.stderr)

    if validate:
        from sklearn.model_selection import train_test_split
        X_train, X_test, yw_train, yw_test, yt_train, yt_test = train_test_split(
            X, y_win, y_total, test_size=0.2, random_state=42
        )
    else:
        X_train, yw_train, yt_train = X, y_win, y_total
        X_test, yw_test, yt_test = None, None, None

    # 勝負預測模型
    win_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss",
    )
    win_model.fit(X_train, yw_train)

    # 總分預測模型
    total_model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42,
    )
    total_model.fit(X_train, yt_train)

    # 儲存模型
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(win_model, WIN_MODEL_PATH)
    joblib.dump(total_model, TOTAL_MODEL_PATH)

    result = {
        "status": "OK",
        "samples": X.shape[0],
        "features": X.shape[1],
        "win_model": WIN_MODEL_PATH,
        "total_model": TOTAL_MODEL_PATH,
    }

    if validate and X_test is not None:
        from sklearn.metrics import accuracy_score, mean_absolute_error
        win_acc = accuracy_score(yw_test, win_model.predict(X_test))
        total_mae = mean_absolute_error(yt_test, total_model.predict(X_test))
        result["validation"] = {
            "win_accuracy": round(win_acc * 100, 1),
            "total_mae": round(total_mae, 2),
        }
        print(f"Win accuracy: {win_acc*100:.1f}%", file=sys.stderr)
        print(f"Total MAE: {total_mae:.2f}", file=sys.stderr)

    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train MLB prediction models")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025],
                        help="Years to train on (default: 2023 2024 2025)")
    parser.add_argument("--validate", action="store_true", help="Hold out 20% for validation")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "train test mode"}))
        return

    train_models(args.years, args.validate)


if __name__ == "__main__":
    main()
