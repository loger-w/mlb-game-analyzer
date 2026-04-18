#!/usr/bin/env python3
"""MLB Results Uploader — 計算賽後結果、更新 jsonl、推送至後端 API"""

import argparse
import json
import os
import sys

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_ROOT = os.path.dirname(SCRIPT_DIR)
ANALYSIS_DATA_DIR = os.path.join(SKILL_ROOT, "analysis-data")
ENV_PATH = os.path.join(SKILL_ROOT, "API_KEY.env")

from review_stats import judge_ml, judge_ou, judge_rl, team_to_abbr, daily_jsonl_path
from clv import (
    find_closing_snapshot, pin_rec_snapshot,
    compute_clv_cents, compute_clv_pct_no_vig,
    detect_line_movement, compute_bet_placed,
)
from predict import TEAM_ABBREV, _abbrev_to_full_name


def find_game_folder(date: str, home_team: str, away_team: str) -> str | None:
    """在 analysis-data/{date}/ 下尋找 per-game 資料夾（匹配隊名關鍵字）"""
    date_dir = os.path.join(ANALYSIS_DATA_DIR, date)
    if not os.path.isdir(date_dir):
        return None
    home_kw = home_team.split()[-1].lower()
    away_kw = away_team.split()[-1].lower()
    for name in os.listdir(date_dir):
        folder = os.path.join(date_dir, name)
        pred_path = os.path.join(folder, "prediction.json")
        if not os.path.exists(pred_path):
            continue
        try:
            with open(pred_path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            if home_kw in (rec.get("home_team") or "").lower() and \
               away_kw in (rec.get("away_team") or "").lower():
                return folder
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _parse_env_file(path: str) -> dict[str, str]:
    """簡易 .env 解析（不依賴 python-dotenv）"""
    result = {}
    if not os.path.exists(path):
        return result
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip()
    return result


def load_env():
    env = _parse_env_file(ENV_PATH)
    base_url = env.get("API_BASE_URL") or os.environ.get("API_BASE_URL")
    secret = env.get("CRON_SECRET") or os.environ.get("CRON_SECRET")
    if not base_url or not secret:
        print(f"ERROR: API_BASE_URL 或 CRON_SECRET 未設定（{ENV_PATH}）", file=sys.stderr)
        sys.exit(1)
    return base_url, secret


def load_records(date: str) -> list[dict]:
    """讀取指定日期的 analysis-data/{date}/predictions.jsonl"""
    path = daily_jsonl_path(date)
    if not os.path.exists(path):
        print(f"ERROR: 找不到 {path}（先跑 summarize_predictions.py --date {date}）", file=sys.stderr)
        sys.exit(1)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def result_code(judge_result, rec_field: str) -> str | None:
    """將 judge_*() 回傳值轉為 API 結果代碼。
    - judge 回傳 None 且 rec 為 PASS → "PASS"
    - judge 回傳 None 且 rec 有值 → None（比賽未結束）
    - 其他 → 原值（WIN / LOSS / PUSH）
    """
    if judge_result is not None:
        return judge_result  # WIN / LOSS / PUSH
    if not rec_field or str(rec_field).upper() == "PASS":
        return "PASS"
    return None  # 比賽未結束


def compute_results(r: dict) -> tuple[str | None, str | None, str | None]:
    """回傳 (ml_result, ou_result, run_line_result)"""
    ml = result_code(judge_ml(r), r.get("ml_rec", "PASS"))
    ou = result_code(judge_ou(r), r.get("ou_rec", "PASS"))
    rl = result_code(judge_rl(r), r.get("run_line_rec", "PASS"))
    return ml, ou, rl


def _market_clv(rec_market, close_market, direction, bet_placed):
    """Compute CLV for one market (ml/ou/rl). Returns dict or None if direction/data missing."""
    if not rec_market or not close_market or direction is None:
        return None
    if direction in ("HOME", "AWAY"):
        side_key = "home" if direction == "HOME" else "away"
        other_key = "away" if direction == "HOME" else "home"
    elif direction in ("OVER", "UNDER"):
        side_key = "over" if direction == "OVER" else "under"
        other_key = "under" if direction == "OVER" else "over"
    else:
        return None
    rec_side = rec_market.get(side_key)
    close_side = close_market.get(side_key)
    rec_other = rec_market.get(other_key)
    close_other = close_market.get(other_key)
    if not (rec_side and close_side and rec_other and close_other):
        return None
    cents = compute_clv_cents(rec_side["decimal"], close_side["decimal"])
    pct = compute_clv_pct_no_vig(rec_side["decimal"], rec_other["decimal"],
                                  close_side["decimal"], close_other["decimal"])
    result = {"cents": cents, "pct_no_vig": pct, "direction": direction, "bet_placed": bet_placed}
    if side_key in ("over", "under"):
        result["point_delta"] = round(float(close_market.get("point", 0)) - float(rec_market.get("point", 0)), 1)
    return result


def _enrich_record_with_clv(record: dict, snap_dir: str, home_full: str, away_full: str, force: bool = False):
    """Mutate `record` in place. Adds closing_line_*, clv, rec_to_close, clv_warnings.

    Idempotent: returns early if record already has `clv` unless force=True.
    Works against the FLAT jsonl record shape (top-level date/game_time/ml_rec/etc.).
    """
    if "clv" in record and not force:
        return
    record.setdefault("clv_warnings", [])

    rec_snap = record.get("recommendation_snapshot")
    kelly = record.get("kelly")
    if not kelly:
        record["clv"] = None
        record["clv_warnings"].append("no_kelly_block")
        return
    if not rec_snap:
        record["clv"] = None
        record["clv_warnings"].append("no_rec_snapshot")
        return

    commence_utc = record.get("game_time") or rec_snap.get("commence_utc")
    game_date_et = record.get("date")
    if not commence_utc or not game_date_et:
        record["clv"] = None
        record["clv_warnings"].append("missing_meta")
        return

    raw_close = find_closing_snapshot(commence_utc, game_date_et, snap_dir)
    if raw_close is None:
        record["clv"] = None
        record["clv_warnings"].append("no_closing_snapshot")
        return

    target_game = None
    for g in raw_close.get("games", []):
        if g.get("home_team") == home_full and g.get("away_team") == away_full:
            target_game = g
            break
    if target_game is None:
        record["clv"] = None
        record["clv_warnings"].append("team_resolve_failed")
        return

    close_source = None
    for name in os.listdir(snap_dir):
        try:
            with open(os.path.join(snap_dir, name), encoding="utf-8") as f:
                other = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if other.get("snapshot_time_utc") == raw_close["snapshot_time_utc"]:
            close_source = name
            break

    close_pinned = pin_rec_snapshot(
        target_game, commence_utc,
        close_source or "unknown",
        raw_close.get("snapshot_time_et"), raw_close.get("snapshot_time_utc"),
    )

    record["closing_line_source"] = close_pinned["source"]
    record["closing_line_minutes_before_first_pitch"] = close_pinned["minutes_before_first_pitch"]
    record["closing_line"] = {k: close_pinned[k] for k in ("ml", "ou", "rl")}

    def _clean_dir(v):
        if v in (None, "PASS", "NEUTRAL"):
            return None
        return v

    ml_dir = _clean_dir(record.get("ml_rec"))
    ou_dir = _clean_dir(record.get("ou_rec"))
    rl_dir = rec_snap.get("rl", {}).get("favorite_side") if rec_snap.get("rl") else None

    record["clv"] = {
        "ml": _market_clv(rec_snap.get("ml"), close_pinned.get("ml"), ml_dir,
                          compute_bet_placed(kelly.get("ml"))),
        "ou": _market_clv(rec_snap.get("ou"), close_pinned.get("ou"), ou_dir,
                          compute_bet_placed(kelly.get("ou"))) if ou_dir else None,
        "rl": _market_clv(rec_snap.get("rl"), close_pinned.get("rl"), rl_dir,
                          compute_bet_placed(kelly.get("rl"))),
    }

    lm = detect_line_movement(None, rec_snap, close_pinned,
                              {"ml": ml_dir, "ou": ou_dir, "rl": rl_dir})
    record["rec_to_close"] = lm["rec_to_close"]

    mb = close_pinned.get("minutes_before_first_pitch")
    if mb is not None and mb > 240:
        record["clv_warnings"].append(f"closing_stale:{mb}min")


def update_records(records: list[dict], force: bool = False) -> list[dict]:
    """為已驗證紀錄補上 result 欄位 + CLV enrichment"""
    snap_dir = os.environ.get("MLB_SNAPSHOT_DIR_OVERRIDE") or os.path.join(SKILL_ROOT, "odds_snapshots")
    updated = []
    for r in records:
        if r.get("verified"):
            ml, ou, rl = compute_results(r)
            r = {**r, "ml_result": ml, "ou_result": ou, "run_line_result": rl}
            home_raw = r.get("home_team") or ""
            away_raw = r.get("away_team") or ""
            home_full = home_raw if home_raw in TEAM_ABBREV else (_abbrev_to_full_name(home_raw) or home_raw)
            away_full = away_raw if away_raw in TEAM_ABBREV else (_abbrev_to_full_name(away_raw) or away_raw)
            try:
                _enrich_record_with_clv(r, snap_dir, home_full, away_full, force=force)
            except (KeyError, IOError, json.JSONDecodeError, TypeError, AttributeError) as e:
                print(f"⚠️ CLV enrich failed for {home_raw} vs {away_raw}: {e}", file=sys.stderr)
        updated.append(r)
    return updated


def save_daily_jsonl(date: str, records: list[dict]):
    """重寫 analysis-data/{date}/predictions.jsonl"""
    path = daily_jsonl_path(date)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_per_game_predictions(date: str, records: list[dict]) -> int:
    """同步更新每場 prediction.json（真相來源）"""
    count = 0
    for r in records:
        folder = find_game_folder(date, r.get("home_team", ""), r.get("away_team", ""))
        if not folder:
            print(f"WARNING: 找不到 {r.get('away_team')} @ {r.get('home_team')} 的資料夾，略過 per-game 寫入", file=sys.stderr)
            continue
        with open(os.path.join(folder, "prediction.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        count += 1
    return count


def build_payload(daily_records: list[dict], date: str) -> dict:
    """組裝 /api/ingest/results 的 payload"""
    daily = [r for r in daily_records if r.get("verified")]
    results = []
    for r in daily:
        ml, ou, rl = compute_results(r)
        results.append({
            "home_team": team_to_abbr(r.get("home_team", "")),
            "away_team": team_to_abbr(r.get("away_team", "")),
            "game_time": r.get("game_time") or r.get("date"),
            "actual_home_score": r.get("actual_home_score"),
            "actual_away_score": r.get("actual_away_score"),
            "ml_result": ml,
            "ou_result": ou,
            "run_line_result": rl,
        })
    return {"date": date, "results": results}


def upload(base_url: str, secret: str, payload: dict):
    url = f"{base_url.rstrip('/')}/api/ingest/results"
    headers = {
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    count = len(payload.get("results", []))
    if resp.status_code >= 200 and resp.status_code < 300:
        print(f"✅ {payload['date']}: {count} 筆結果上傳成功（{resp.status_code}）")
    else:
        print(f"ERROR: {payload['date']} 上傳失敗 {resp.status_code} — {resp.text}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload game results to backend API")
    parser.add_argument("--date", required=True, help="日期 YYYY-MM-DD")
    parser.add_argument("--test", action="store_true", help="印出 payload 但不實際送出、不更新 jsonl")
    parser.add_argument("--force", action="store_true", help="強制重新計算 CLV 欄位（覆蓋已存在值）")
    args = parser.parse_args()

    base_url, secret = load_env()
    daily_records = load_records(args.date)

    daily_verified = [r for r in daily_records if r.get("verified")]
    if not daily_verified:
        print(f"ERROR: {args.date} 沒有已驗證的比賽紀錄", file=sys.stderr)
        sys.exit(1)

    payload = build_payload(daily_records, args.date)

    if args.test:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    # 1. 更新 per-date jsonl + per-game prediction.json
    updated = update_records(daily_records, force=args.force)
    save_daily_jsonl(args.date, updated)
    n_per_game = save_per_game_predictions(args.date, updated)
    print(f"✅ analysis-data/{args.date}/predictions.jsonl 已更新（{len(daily_verified)} 筆 result）")
    print(f"✅ per-game prediction.json 同步 {n_per_game} 筆")

    # 2. 推送到後端
    upload(base_url, secret, payload)


if __name__ == "__main__":
    main()
