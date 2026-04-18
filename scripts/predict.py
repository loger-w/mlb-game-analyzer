#!/usr/bin/env python3
"""MLB Game Predictor — XGBoost 預測 + Log5 交叉驗證 + 信號計分表"""

import argparse
import glob
import json
import math
import os
import re
import sys
from datetime import datetime

import joblib
import numpy as np

# Fix Windows encoding for emoji output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
WIN_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_win_model.pkl")

# F2: 完整 30 隊隊名 → 縮寫映射（用於方向矛盾檢查）
TEAM_ABBREV = {
    "New York Yankees": "NYY", "New York Mets": "NYM", "Boston Red Sox": "BOS",
    "Los Angeles Dodgers": "LAD", "Los Angeles Angels": "LAA", "Houston Astros": "HOU",
    "Atlanta Braves": "ATL", "Philadelphia Phillies": "PHI", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "St. Louis Cardinals": "STL", "Milwaukee Brewers": "MIL",
    "Pittsburgh Pirates": "PIT", "Arizona Diamondbacks": "ARI", "Colorado Rockies": "COL",
    "Baltimore Orioles": "BAL", "Tampa Bay Rays": "TB", "Toronto Blue Jays": "TOR",
    "Minnesota Twins": "MIN", "Kansas City Royals": "KC", "Detroit Tigers": "DET",
    "Cleveland Guardians": "CLE", "Seattle Mariners": "SEA", "Athletics": "OAK",
    "Texas Rangers": "TEX", "Miami Marlins": "MIA", "Washington Nationals": "WSH",
}

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


_SNAPSHOT_FILENAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})-00-ET\.json$")


def load_closest_snapshot(
    game_date_et: str,
    game_start_utc: str,
    snapshot_dir: str = None,
) -> dict | None:
    """Find newest Pinnacle snapshot with snapshot_time < game_start_utc
    and containing games on game_date_et.

    Returns None if no match.
    """
    if snapshot_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        snapshot_dir = os.path.join(base, "odds_snapshots")

    if not os.path.isdir(snapshot_dir):
        return None

    try:
        game_start_dt = datetime.fromisoformat(game_start_utc.replace("Z", "+00:00"))
    except ValueError:
        return None

    candidates = []
    for path in glob.glob(os.path.join(snapshot_dir, "*.json")):
        name = os.path.basename(path)
        m = _SNAPSHOT_FILENAME_RE.match(name)
        if not m:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                snap = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        snap_time = datetime.fromisoformat(snap["snapshot_time_utc"].replace("Z", "+00:00"))
        if snap_time >= game_start_dt:
            continue

        has_date = any(g.get("game_date_et") == game_date_et for g in snap.get("games", []))
        if not has_date:
            continue

        candidates.append((snap_time, snap))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


_NAME_TO_ABBREV = dict(TEAM_ABBREV)


def resolve_pinnacle_odds(
    snapshot: dict,
    home_abbrev: str,
    away_abbrev: str,
    game_index: int = None,
) -> dict | None:
    """Extract Pinnacle decimal odds. For doubleheaders, game_index (1 or 2) required."""
    matches = []
    for g in snapshot.get("games", []):
        home_full = g.get("home_team")
        away_full = g.get("away_team")
        gh = _NAME_TO_ABBREV.get(home_full)
        ga = _NAME_TO_ABBREV.get(away_full)
        if gh != home_abbrev or ga != away_abbrev:
            continue
        matches.append(g)

    if not matches:
        return None

    if len(matches) > 1:
        if game_index is None:
            raise ValueError(
                f"doubleheader detected for {away_abbrev}@{home_abbrev}; "
                f"pass game_index=1 or 2"
            )
        # 按 commence_et 排序，game_index 1 基底
        matches.sort(key=lambda g: g.get("commence_et", ""))
        if game_index < 1 or game_index > len(matches):
            raise ValueError(f"game_index {game_index} out of range (have {len(matches)} games)")
        g = matches[game_index - 1]
    else:
        g = matches[0]

    pin = g.get("bookmakers", {}).get("pinnacle")
    if not pin:
        return None

    ml = pin.get("ml", {})
    ou = pin.get("ou", {})
    rl = pin.get("rl", {})

    home_full = g["home_team"]
    away_full = g["away_team"]

    result = {
        "snapshot_time_et": snapshot.get("snapshot_time_et"),
        "ml": None,
        "ou": None,
        "rl": None,
    }

    if home_full in ml and away_full in ml:
        result["ml"] = {
            "home_decimal": ml[home_full]["odds"],
            "away_decimal": ml[away_full]["odds"],
        }

    if "Over" in ou and "Under" in ou:
        result["ou"] = {
            "line": ou["Over"].get("point"),
            "over_decimal": ou["Over"]["odds"],
            "under_decimal": ou["Under"]["odds"],
        }

    if home_full in rl and away_full in rl:
        result["rl"] = {
            "home_point": rl[home_full].get("point"),
            "home_decimal": rl[home_full]["odds"],
            "away_point": rl[away_full].get("point"),
            "away_decimal": rl[away_full]["odds"],
        }

    return result


def pythagorean_runs(rs: float, ra: float, g: float = 10) -> float:
    """Pythagenport 動態指數公式（與 reference/teams-and-api.md 一致）

    exponent = 1.50 × log10[(RS + RA) / G] + 0.45
    Pythagenport RMSE = 3.991 勝（優於固定指數 1.83 的 4.126）
    """
    if rs + ra == 0:
        return 0.5
    exponent = 1.50 * math.log10((rs + ra) / g) + 0.45
    return (rs ** exponent) / (rs ** exponent + ra ** exponent)


def compute_trend_tags(data: dict) -> list[str]:
    """比較近 10 場 vs 本季平均，產出趨勢標籤"""
    tags = []
    for side in ("home", "away"):
        rs_10 = data.get(f"{side}_recent_rs", 4.5)
        ra_10 = data.get(f"{side}_recent_ra", 4.5)
        rs_season = data.get(f"{side}_season_rs", rs_10)
        ra_season = data.get(f"{side}_season_ra", ra_10)

        if rs_season > 0 and rs_10 > rs_season * 1.2:
            tags.append(f"{side}-hot-offense")
        elif rs_season > 0 and rs_10 < rs_season * 0.8:
            tags.append(f"{side}-cold-offense")

        if ra_season > 0 and ra_10 > ra_season * 1.2:
            tags.append(f"{side}-pitching-slump")
        elif ra_season > 0 and ra_10 < ra_season * 0.8:
            tags.append(f"{side}-pitching-hot")

    return tags


def compute_signal_table(data: dict) -> dict:
    """F1: 信號計分表 — 使用 Run Value 修正（對齊 reference/prediction.md）

    每個信號轉為 ±run 修正值，最終加總到預測比分上。
    正值 = 得分上升（Over 方向），負值 = 得分下降（Under 方向）。
    """
    signals = []

    # --- 總分上修信號 ---

    # 雙方打線近期 Hot（場均 ≥ 5）
    home_rs = data.get("home_recent_rs", 4.5)
    away_rs = data.get("away_recent_rs", 4.5)
    if home_rs >= 5 and away_rs >= 5:
        signals.append({"signal": "雙方打線近期 Hot（場均 ≥ 5 分）", "run_value": +0.5})
    if home_rs <= 2 and away_rs <= 2:
        signals.append({"signal": "雙方打線近期 Cold（場均 ≤ 2 分）", "run_value": -0.5})

    # 牛棚 ERA ≥ 5.0（對手得分上升）
    home_bp = data.get("home_bullpen_era", 4.0)
    away_bp = data.get("away_bullpen_era", 4.0)
    if home_bp >= 5.0:
        signals.append({"signal": f"主隊牛棚 ERA {home_bp:.2f} ≥ 5.0", "run_value": +0.5})
    if away_bp >= 5.0:
        signals.append({"signal": f"客隊牛棚 ERA {away_bp:.2f} ≥ 5.0", "run_value": +0.5})

    # Park Factor 修正：(PF - 100) × 0.05
    pf = data.get("park_factor", 100)
    pf_adj = round((pf - 100) * 0.05, 2)
    if abs(pf_adj) >= 0.1:
        signals.append({"signal": f"Park Factor {pf}（修正 {pf_adj:+.2f}）", "run_value": pf_adj})

    # 雙方先發投手等級
    home_fip = data.get("home_starter_fip", 4.0)
    away_fip = data.get("away_starter_fip", 4.0)
    if home_fip <= 3.0 and away_fip <= 3.0:
        signals.append({"signal": "雙方先發 FIP ≤ 3.0（Ace 級）", "run_value": -1.0})
    elif home_fip <= 3.2 and away_fip <= 3.2:
        signals.append({"signal": "雙方先發 FIP ≤ 3.2（Strong Ace+）", "run_value": -0.5})

    # 打線 xwOBA
    home_xwoba = data.get("home_batting_xwoba", 0.320)
    away_xwoba = data.get("away_batting_xwoba", 0.320)
    if home_xwoba >= 0.350 and away_xwoba >= 0.350:
        signals.append({"signal": "雙方打線 xwOBA ≥ .350", "run_value": +0.5})

    # 打線 K%
    home_k = data.get("home_batting_k_pct", 20)
    away_k = data.get("away_batting_k_pct", 20)
    if home_k >= 25 and away_k >= 25:
        signals.append({"signal": "雙方打線 K% ≥ 25%", "run_value": -0.3})

    total_run_adj = round(sum(s["run_value"] for s in signals), 2)

    return {
        "signals": signals,
        "total_run_adjustment": total_run_adj,
    }


def predict_with_ml(features: list[float]) -> dict | None:
    """用 XGBoost 模型預測主隊勝率（若模型不存在或特徵不匹配則 graceful fallback）"""
    if not os.path.exists(WIN_MODEL_PATH):
        return None

    try:
        win_model = joblib.load(WIN_MODEL_PATH)
        X = np.array([features])
        win_prob = float(win_model.predict_proba(X)[0][1])  # 主隊勝率
    except Exception:
        # 模型特徵欄位不匹配時 graceful fallback
        return None

    return {
        "home_win_pct": round(win_prob * 100, 1),
    }


def predict_with_formula(data: dict) -> dict:
    """F3: 用 Log5 + 期望得分公式預測（納入對方投手壓制力）

    E[R] = 聯盟平均得分 × (打線 xwOBA / 聯盟 xwOBA) × (對方投手 ERA / 聯盟 ERA) × (PF / 100)
    聯盟平均（2024-2025 基準）：R/G ≈ 4.5, xwOBA ≈ 0.315, ERA ≈ 4.20
    """
    LEAGUE_RPG = 4.5
    LEAGUE_XWOBA = 0.315
    LEAGUE_ERA = 4.20

    home_rs = data.get("home_recent_rs", 4.5)
    home_ra = data.get("home_recent_ra", 4.5)
    away_rs = data.get("away_recent_rs", 4.5)
    away_ra = data.get("away_recent_ra", 4.5)
    pf = data.get("park_factor", 100)

    # Log5 勝率（用 Pythagorean）
    home_pct = pythagorean_runs(home_rs, home_ra)
    away_pct = pythagorean_runs(away_rs, away_ra)
    log5_pct = log5(home_pct, away_pct)
    log5_pct = min(log5_pct + 0.03, 0.95)  # 主場優勢 +3%

    # 期望得分（納入打線 xwOBA + 對方投手 FIP）
    home_xwoba = data.get("home_batting_xwoba", LEAGUE_XWOBA)
    away_xwoba = data.get("away_batting_xwoba", LEAGUE_XWOBA)
    # 用 FIP 代替 ERA（FIP 更能反映投手真實能力）
    away_pitcher_fip = data.get("away_starter_fip", LEAGUE_ERA)
    home_pitcher_fip = data.get("home_starter_fip", LEAGUE_ERA)

    pf_mult = pf / 100
    home_score = round(
        LEAGUE_RPG * (home_xwoba / LEAGUE_XWOBA) * (away_pitcher_fip / LEAGUE_ERA) * pf_mult, 1
    )
    away_score = round(
        LEAGUE_RPG * (away_xwoba / LEAGUE_XWOBA) * (home_pitcher_fip / LEAGUE_ERA) * pf_mult, 1
    )

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
    parser.add_argument("--game-data", help="Path to JSON with merged game data")
    parser.add_argument("--save", action="store_true",
                        help="Save prediction.json into the game's analysis-data folder (parent dir of --game-data)")
    parser.add_argument("-o", "--output",
                        help="Explicit output path for prediction.json (overrides --save auto-path)")
    parser.add_argument("--test", action="store_true")
    # Post-analysis manual fields
    parser.add_argument("--adjusted-home", type=float, help="Adjusted home runs scored")
    parser.add_argument("--adjusted-away", type=float, help="Adjusted away runs scored")
    parser.add_argument("--ou-line", type=float, help="O/U line (effective, e.g. 9.75 for quarter-ball)")
    parser.add_argument("--ou-rec", choices=["OVER", "UNDER", "PASS"], help="O/U recommendation")
    parser.add_argument("--ml-rec", help="ML recommendation (team abbr or PASS)")
    parser.add_argument("--ml-stars", type=int, choices=[0, 1, 2, 3, 4, 5], help="ML star rating")
    parser.add_argument("--run-line-rec", help="Run line recommendation (team abbr or PASS)")
    parser.add_argument("--run-line", help="Run line value, e.g. '-1.5' or '1+50'")
    parser.add_argument("--ou-stars", type=int, choices=[0, 1, 2, 3, 4, 5], help="O/U star rating")
    parser.add_argument("--run-line-stars", type=int, choices=[0, 1, 2, 3, 4, 5], help="Run line star rating")
    parser.add_argument("--signal-adjustments", type=json.loads,
                        help='Signal adjustments JSON, e.g. \'{"puk_il":0.3}\'')
    parser.add_argument("--tags", help="Comma-separated tags, e.g. divergent,early-season")
    parser.add_argument("--temperature", type=float, help="Temperature (F)")
    parser.add_argument("--wind-mph", type=float, help="Wind speed (mph)")
    parser.add_argument("--wind-direction", help="Wind direction")
    parser.add_argument("--umpire", help="Home plate umpire name")
    parser.add_argument("--umpire-ou-rate", type=float, help="Umpire career Over pct")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "predict test mode"}))
        return

    if not args.game_data:
        parser.error("--game-data is required unless --test is specified")

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

    # 用 30 場 Formula 做交叉驗證（比 10 場更穩定）
    home_season_games = data.get("home_season_games", 0)
    away_season_games = data.get("away_season_games", 0)
    min_season_games = min(home_season_games, away_season_games)

    formula_30_pred = None
    if min_season_games >= 30:
        # 用 30 場數據計算 Formula
        data_30 = {
            "home_recent_rs": data.get("home_recent_30_rs", data.get("home_recent_rs", 4.5)),
            "home_recent_ra": data.get("home_recent_30_ra", data.get("home_recent_ra", 4.5)),
            "away_recent_rs": data.get("away_recent_30_rs", data.get("away_recent_rs", 4.5)),
            "away_recent_ra": data.get("away_recent_30_ra", data.get("away_recent_ra", 4.5)),
            "park_factor": data.get("park_factor", 100),
        }
        formula_30_pred = predict_with_formula(data_30)

    # 交叉驗證
    cross_validation = "NO_ML_MODEL"
    if ml_pred:
        if min_season_games < 30:
            cross_validation = "INSUFFICIENT_SAMPLE"
        else:
            ml_lean = "HOME" if ml_pred["home_win_pct"] > 50 else "AWAY"
            xval_formula = formula_30_pred if formula_30_pred else formula_pred
            formula_lean = "HOME" if xval_formula["log5_pct"] > 50 else "AWAY"
            pct_diff = abs(ml_pred["home_win_pct"] - xval_formula["log5_pct"])
            cross_validation = "CONSISTENT" if ml_lean == formula_lean else "DIVERGENT"

    # 最終推薦
    # 勝率：有 ML 時用 ML（XGBoost 勝率預測可靠）
    # 比分：一律用 formula（ML 的 total_model 訓練資料有結構性缺陷，比分不可靠）
    if ml_pred:
        final_pct = ml_pred["home_win_pct"]
    else:
        final_pct = formula_pred["log5_pct"]
    final_home = formula_pred["home_score"]
    final_away = formula_pred["away_score"]

    # 計算 adjusted 比分（用於決定最終方向）
    adj_home = args.adjusted_home if args.adjusted_home is not None else final_home
    adj_away = args.adjusted_away if args.adjusted_away is not None else final_away
    adj_total = round(adj_home + adj_away, 1)

    # 決定最終方向：adjusted 比分優先於 XGBoost
    has_adjusted = args.adjusted_home is not None or args.adjusted_away is not None
    if has_adjusted and (adj_home > adj_away) != (final_pct > 50):
        # adjusted 比分方向與 XGBoost 相反 → 使用 Log5 勝率
        adjusted_winner = "HOME" if adj_home > adj_away else "AWAY"
        adjusted_pct = formula_pred["log5_pct"] if adjusted_winner == "HOME" else round(100 - formula_pred["log5_pct"], 1)
        display_home_pct = round(formula_pred["log5_pct"], 1)
    else:
        adjusted_winner = "HOME" if final_pct > 50 else "AWAY"
        display_home_pct = round(final_pct, 1)

    result = {
        "ml_prediction": ml_pred,
        "formula_prediction": formula_pred,
        "cross_validation": cross_validation,
        "signal_table": signal_table,
        "final": {
            "recommended_winner": adjusted_winner,
            "home_win_pct": display_home_pct,
            "confidence": "HIGH" if cross_validation == "CONSISTENT" else ("MEDIUM" if cross_validation == "NO_ML_MODEL" else "LOW"),
            "predicted_home_score": adj_home,
            "predicted_away_score": adj_away,
            "predicted_total": adj_total,
            "signal_run_adjustment": signal_table["total_run_adjustment"],
            "over_under_lean": "OVER" if signal_table["total_run_adjustment"] > 0 else ("UNDER" if signal_table["total_run_adjustment"] < 0 else "NEUTRAL"),
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 自動存檔到 predictions.jsonl
    if args.save:
        from datetime import datetime as _dt
        meta = data.get("_meta", {})
        home_team = meta.get("home_team") or "HOME"
        away_team = meta.get("away_team") or "AWAY"

        raw_date = (meta.get("game_date") or "")[:10]
        record_date = raw_date if raw_date else _dt.now().strftime("%Y-%m-%d")

        # === 護欄機制：星級自動上限 ===
        original_ml_stars = args.ml_stars
        ml_stars_cap = 5  # 預設無上限
        force_ml_pass = False  # 強制 ml_rec = PASS 旗標
        cap_reasons = []

        # F4: 規則 1：DIVERGENT → 強制 PASS（ml_stars = 0）
        if cross_validation == "DIVERGENT":
            ml_stars_cap = 0
            force_ml_pass = True
            cap_reasons.append("DIVERGENT 強制 PASS")

        # D1.5: 規則 2：INSUFFICIENT_SAMPLE → 方向檢查
        # ml_pred 與 formula log5 方向分歧 → 強制 PASS（比照 DIVERGENT）
        # 方向一致 → 上限 2（樣本不足時保守）
        if cross_validation == "INSUFFICIENT_SAMPLE" and ml_pred and formula_pred:
            ml_lean = "HOME" if ml_pred["home_win_pct"] > 50 else "AWAY"
            formula_lean = "HOME" if formula_pred["log5_pct"] > 50 else "AWAY"
            if ml_lean != formula_lean:
                ml_stars_cap = 0
                force_ml_pass = True
                cap_reasons.append("INSUFFICIENT_SAMPLE + 方向分歧 強制 PASS")
            else:
                ml_stars_cap = min(ml_stars_cap, 2)
                cap_reasons.append("INSUFFICIENT_SAMPLE 方向一致 上限 2")
        elif cross_validation == "INSUFFICIENT_SAMPLE":
            # fallback：缺 ml_pred 或 formula_pred 時套舊規則
            ml_stars_cap = min(ml_stars_cap, 3)
            cap_reasons.append("INSUFFICIENT_SAMPLE 上限 3（無方向資料）")

        # 規則 3：開季（先發場次 < 5）→ 上限 3
        home_sp_starts = meta.get("home_sp_starts") or 0
        away_sp_starts = meta.get("away_sp_starts") or 0
        if home_sp_starts < 5 or away_sp_starts < 5:
            ml_stars_cap = min(ml_stars_cap, 3)
            cap_reasons.append("開季（先發場次 < 5）上限 3")

        # 規則 4：XGBoost 勝率 50-55% → 上限 2
        rec_side_pct = final_pct if result["final"]["recommended_winner"] == "HOME" else (100 - final_pct)
        if 50 <= rec_side_pct < 55:
            ml_stars_cap = min(ml_stars_cap, 2)
            cap_reasons.append(f"XGBoost 勝率 {rec_side_pct:.1f}%（50-55%）上限 2")

        # F2: 規則 5：方向矛盾（ml_rec 與 predicted_winner 不一致）→ 上限 2
        direction_override = False
        if args.ml_rec and args.ml_rec != "PASS":
            predicted_winner = result["final"]["recommended_winner"]
            home_abbr = TEAM_ABBREV.get(home_team, "")
            away_abbr = TEAM_ABBREV.get(away_team, "")
            rec_is_home = args.ml_rec == home_abbr
            rec_is_away = args.ml_rec == away_abbr
            if (predicted_winner == "HOME" and rec_is_away) or (predicted_winner == "AWAY" and rec_is_home):
                direction_override = True
                ml_stars_cap = min(ml_stars_cap, 2)
                cap_reasons.append(f"ml_rec={args.ml_rec} 與 XGBoost predicted_winner={predicted_winner}({home_abbr if predicted_winner == 'HOME' else away_abbr}) 方向矛盾，上限 2")

        # 套用星級上限
        final_ml_stars = args.ml_stars
        if final_ml_stars is not None and final_ml_stars > ml_stars_cap:
            print(f"⚠️ ml_stars 從 {final_ml_stars} 降為 {ml_stars_cap}（原因：{'; '.join(cap_reasons)}）", file=sys.stderr)
            final_ml_stars = ml_stars_cap

        # 套用 ml_rec 強制 PASS（DIVERGENT / INSUFFICIENT_SAMPLE 方向分歧）
        final_ml_rec = args.ml_rec
        if force_ml_pass and final_ml_rec and final_ml_rec != "PASS":
            print(f"⚠️ ml_rec 從 {final_ml_rec} 改為 PASS（原因：{'; '.join(cap_reasons)}）", file=sys.stderr)
            final_ml_rec = "PASS"

        # === 趨勢標籤 ===
        trend_tags = compute_trend_tags(data)

        # 合併使用者 tags + 趨勢 tags + direction-override
        user_tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
        if direction_override and "direction-override" not in user_tags:
            user_tags.append("direction-override")
        all_tags = list(dict.fromkeys(user_tags + trend_tags))  # 去重保序

        # === 護欄機制：O/U 自動 PASS ===
        final_ou_rec = args.ou_rec if args.ou_rec is not None else result["final"]["over_under_lean"]
        if final_ou_rec == "NEUTRAL":
            final_ou_rec = "PASS"
        final_ou_stars = args.ou_stars

        # OU-1: 差距 < 1.5 run → PASS（SD ≈ 4.5，< 1.5 在噪音範圍）
        if final_ou_rec != "PASS" and args.ou_line is not None:
            ou_gap = abs(adj_total - args.ou_line)
            if ou_gap < 1.5:
                print(f"⚠️ O/U 從 {final_ou_rec} 改為 PASS（差距 {ou_gap:.1f} < 1.5 run）", file=sys.stderr)
                final_ou_rec = "PASS"
                final_ou_stars = 0

        # OU-2: 方向與調整後比分矛盾 → PASS
        if final_ou_rec == "OVER" and args.ou_line is not None and adj_total <= args.ou_line:
            print(f"⚠️ O/U 從 OVER 改為 PASS（調整後總分 {adj_total} ≤ 線 {args.ou_line}）", file=sys.stderr)
            final_ou_rec = "PASS"
            final_ou_stars = 0
        elif final_ou_rec == "UNDER" and args.ou_line is not None and adj_total >= args.ou_line:
            print(f"⚠️ O/U 從 UNDER 改為 PASS（調整後總分 {adj_total} ≥ 線 {args.ou_line}）", file=sys.stderr)
            final_ou_rec = "PASS"
            final_ou_stars = 0

        # OU-3: 非 PASS 但 stars 未指定 → PASS（防止 upload 套 default 3 星）
        if final_ou_rec != "PASS" and final_ou_stars is None:
            print(f"⚠️ O/U 從 {final_ou_rec} 改為 PASS（未指定 --ou-stars）", file=sys.stderr)
            final_ou_rec = "PASS"
            final_ou_stars = 0

        # === 護欄機制：讓分盤自動 PASS ===
        final_rl_rec = args.run_line_rec if args.run_line_rec is not None else "PASS"
        final_rl_stars = args.run_line_stars

        # RL-1: LOW confidence → PASS（規則：ML 信心 < MEDIUM 時不推受讓）
        if result["final"]["confidence"] == "LOW" and final_rl_rec != "PASS":
            print(f"⚠️ 讓分盤從 {final_rl_rec} 改為 PASS（confidence = LOW）", file=sys.stderr)
            final_rl_rec = "PASS"
            final_rl_stars = 0

        # RL-2: 非 PASS 但 stars 未指定 → PASS
        if final_rl_rec != "PASS" and final_rl_stars is None:
            print(f"⚠️ 讓分盤從 {final_rl_rec} 改為 PASS（未指定 --run-line-stars）", file=sys.stderr)
            final_rl_rec = "PASS"
            final_rl_stars = 0

        record = {
            "date": record_date,
            "game_time": meta.get("game_date"),
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
            "ou_rec": final_ou_rec,
            "ou_stars": final_ou_stars,
            "run_line_rec": final_rl_rec,
            "run_line": args.run_line,
            "run_line_stars": final_rl_stars,
            "ml_rec": final_ml_rec,
            "ml_stars": final_ml_stars,
            "original_ml_stars": original_ml_stars,
            "confidence": result["final"]["confidence"],
            "cross_validation": result["cross_validation"],
            "tags": all_tags,
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
        # 寫入 per-game prediction.json（放在 --game-data 所在資料夾）
        if args.output:
            prediction_path = args.output
        else:
            game_dir = os.path.dirname(os.path.abspath(args.game_data))
            prediction_path = os.path.join(game_dir, "prediction.json")

        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
        with open(prediction_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"Saved to {prediction_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
