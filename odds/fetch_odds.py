#!/usr/bin/env python3
"""MLB Odds Fetcher — Pinnacle only

由 Windows Task Scheduler 每天 5 次觸發，抓 Pinnacle MLB 賠率快照。
Cron 設計用 ET 思考（對齊 MLB 賽程與 Pinnacle 市場節奏），TW 換算僅為排程器的
實作細節：

    ET 22 (prev day) — opener / 早期市場 anchor
    ET 12             — 早場 / 日場前 1-2h
    ET 15             — 早晚場前 1-3h
    ET 18             — 主流晚場前 1h（Pinnacle 限額峰值）
    ET 21             — 西岸前 1h

對應 Windows Task Scheduler（TW = ET + 12h）：
    TW 10 / 00 / 03 / 06 / 09

Credit 消耗：3 credits / 次（h2h + totals + spreads，eu region），5 次/天 = 15/天，
月均 450 credits，在 500 免費額度內，剩 50 buffer。

只儲存 Pinnacle 賠率（Sharp book，隱含勝率最準確）。比賽開打後 Pinnacle 會暫停
盤口，loader 端再以 snapshot_time_utc >= commence_utc 過濾賽中污染。

每個 outcome（ml/ou/rl 各邊）落盤時除 raw `implied_pct` 外，亦寫入 vig 剝除後
的 `no_vig_pct`（雙邊合計 = 100），供下游 movement / report 使用。
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

# 讓 lib/ 子模組可被 import（與 analyze_smart_money.py 對齊）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

from odds_math import no_vig_two_way
from timeutil import ET, parse_iso_utc

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ── 路徑 ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE     = os.path.join(BASE_DIR, "API_KEY.env")
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "odds_snapshots")
LOG_FILE     = os.path.join(BASE_DIR, "odds_fetch.log")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def load_api_key() -> str:
    if not os.path.exists(ENV_FILE):
        raise FileNotFoundError(f"找不到 API_KEY.env：{ENV_FILE}")
    with open(ENV_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise ValueError("API_KEY.env 中找不到 ODDS_API_KEY")


def implied_prob(decimal_odds: float) -> float:
    """小數賠率 → 隱含勝率（%）"""
    if decimal_odds <= 0:
        return 0.0
    return round(1 / decimal_odds * 100, 1)


def fetch_odds(api_key: str) -> tuple:
    """呼叫 API，回傳 (games_list, credits_remaining, credits_used)"""
    url = (
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
        f"?apiKey={api_key}"
        "&regions=eu"
        "&markets=h2h,totals,spreads"
        "&bookmakers=pinnacle"
        "&oddsFormat=decimal"
        "&dateFormat=iso"
    )
    with urllib.request.urlopen(url, timeout=15) as resp:
        games = json.loads(resp.read())
        remaining = resp.headers.get("x-requests-remaining", "?")
        used      = resp.headers.get("x-requests-used", "?")
    return games, remaining, used


def parse_game(game: dict) -> dict | None:
    """解析單場比賽，提取各莊家的 ML / O/U / RL 賠率與隱含勝率。

    commence_time 缺或無法解析 → None（單場跳過，不可炸掉整輪快照）。
    """
    if not game.get("home_team") or not game.get("away_team"):
        return None
    utc_dt = parse_iso_utc(game.get("commence_time"))
    if utc_dt is None:
        return None
    et_dt  = utc_dt.astimezone(ET)

    parsed = {
        "game":         f"{game['away_team']} @ {game['home_team']}",
        "away_team":    game["away_team"],
        "home_team":    game["home_team"],
        "commence_utc": game["commence_time"],
        "commence_et":  et_dt.strftime("%Y-%m-%d %H:%M ET"),   # 美東時間（可直接對照 MLB 官網賽程）
        "game_date_et": et_dt.strftime("%Y-%m-%d"),            # 美東日期，用於篩選「哪天的比賽」
        "bookmakers":   {},
    }

    for bk in game.get("bookmakers", []):
        bk_data = {"title": bk["title"], "ml": {}, "ou": {}, "rl": {}}

        for mkt in bk.get("markets", []):
            if mkt["key"] == "h2h":
                for o in mkt["outcomes"]:
                    bk_data["ml"][o["name"]] = {
                        "odds":        o["price"],
                        "implied_pct": implied_prob(o["price"]),
                    }
            elif mkt["key"] == "totals":
                for o in mkt["outcomes"]:
                    bk_data["ou"][o["name"]] = {
                        "odds":        o["price"],
                        "point":       o.get("point"),
                        "implied_pct": implied_prob(o["price"]),
                    }
            elif mkt["key"] == "spreads":
                for o in mkt["outcomes"]:
                    bk_data["rl"][o["name"]] = {
                        "odds":        o["price"],
                        "point":       o.get("point"),
                        "implied_pct": implied_prob(o["price"]),
                    }

        _attach_no_vig(bk_data, parsed["home_team"], parsed["away_team"])
        parsed["bookmakers"][bk["key"]] = bk_data

    return parsed


def _attach_no_vig(bk_data: dict, home: str, away: str) -> None:
    """對 ml / ou / rl 三個雙邊市場各自計算 no-vig fair %，並寫回 outcome dict。

    任一邊缺 implied_pct 或 ≤ 0 → 該市場不寫 no_vig_pct（呼叫端可自行 fallback 重算）。
    """
    # ML：home / away
    ml = bk_data.get("ml", {})
    _pair_no_vig(ml.get(home), ml.get(away))

    # Total juice：Over / Under（point 不動，僅 juice 雙邊）
    ou = bk_data.get("ou", {})
    _pair_no_vig(ou.get("Over"), ou.get("Under"))

    # RL：home / away
    rl = bk_data.get("rl", {})
    _pair_no_vig(rl.get(home), rl.get(away))


def parse_games(raw_games: list) -> tuple[list, int]:
    """整批解析。回 (parsed, skipped);skipped = 必要欄位缺/壞而被丟棄的場數。"""
    parsed = [p for p in (parse_game(g) for g in raw_games) if p is not None]
    return parsed, len(raw_games) - len(parsed)


def _pair_no_vig(side_a: dict | None, side_b: dict | None) -> None:
    """讀兩個 outcome dict 的 implied_pct，計 no-vig 後 in-place 寫回。"""
    if not side_a or not side_b:
        return
    fair_a, fair_b = no_vig_two_way(side_a.get("implied_pct"), side_b.get("implied_pct"))
    if fair_a is None or fair_b is None:
        return
    side_a["no_vig_pct"] = fair_a
    side_b["no_vig_pct"] = fair_b


def write_log(msg: str):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    write_log("=== fetch_odds.py 開始執行 ===")

    # 讀取 API key
    try:
        api_key = load_api_key()
    except Exception as e:
        write_log(f"ERROR 讀取 API key 失敗：{e}")
        sys.exit(1)

    # 呼叫 API
    try:
        raw_games, remaining, used = fetch_odds(api_key)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        write_log(f"ERROR API 請求失敗 HTTP {e.code}：{body}")
        sys.exit(1)
    except Exception as e:
        write_log(f"ERROR 網路錯誤：{e}")
        sys.exit(1)

    # 無比賽時直接記錄並結束（節省 credits 分析時間）
    if not raw_games:
        write_log(f"INFO 今日無 MLB 比賽 | credits 剩餘 {remaining}")
        return

    # 解析並儲存快照（檔名用美東時間，與 MLB 官網賽程日期一致）
    now_utc  = datetime.now(timezone.utc)
    now_et   = now_utc.astimezone(ET)
    ts_label = now_et.strftime("%Y-%m-%d_%H-00-ET")   # 例：2026-04-14_18-00-ET
    parsed, skipped = parse_games(raw_games)
    if skipped:
        write_log(f"WARN {skipped} 場因必要欄位缺/壞被跳過(API 回傳 {len(raw_games)} 場)")

    snapshot = {
        "snapshot_time_utc": now_utc.isoformat(),
        "snapshot_time_et":  now_et.strftime("%Y-%m-%d %H:%M ET"),
        "credits_remaining": remaining,
        "credits_used":      used,
        "game_count":        len(parsed),
        "games":             parsed,
    }

    out_path = os.path.join(SNAPSHOT_DIR, f"{ts_label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    write_log(
        f"OK  {len(parsed)} 場比賽 | "
        f"credits 剩餘 {remaining} / 已用 {used} | "
        f"存至 odds_snapshots/{ts_label}.json"
    )

    # 簡要摘要（Pinnacle，開打時間用美東）
    print()
    print(f"{'比賽':<42} {'開打(ET)':>14} {'主隊 ML':>9} {'客隊 ML':>9} {'O/U':>6} {'主隊%':>7} {'客隊%':>7}")
    print("-" * 102)
    for g in parsed:
        bk        = g["bookmakers"].get("pinnacle", {})
        home_odds = bk.get("ml", {}).get(g["home_team"], {}).get("odds", "-")
        away_odds = bk.get("ml", {}).get(g["away_team"], {}).get("odds", "-")
        home_pct  = bk.get("ml", {}).get(g["home_team"], {}).get("implied_pct", "-")
        away_pct  = bk.get("ml", {}).get(g["away_team"], {}).get("implied_pct", "-")
        ou_vals   = bk.get("ou", {})
        ou_point  = next((v["point"] for v in ou_vals.values() if v.get("point")), "-")
        # commence_et 格式："2026-04-14 18:36 ET" → 只取 HH:MM 日期部分
        et_label  = g.get("commence_et", "-")
        print(f"{g['game']:<42} {et_label:>14} {str(home_odds):>9} {str(away_odds):>9} {str(ou_point):>6} {str(home_pct):>7} {str(away_pct):>7}")

    print()
    write_log("=== 執行完畢 ===")


if __name__ == "__main__":
    main()
