#!/usr/bin/env python3
"""Orchestrator:單場 (--matchup) 或當日全部 (--all)。串 fetch_inputs → run_model
→ odds_compare → report。"""
import argparse
import sys
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_model
import report as report_mod
from fetch_inputs import fetch_inputs, MLB_API_BASE
from odds_compare import find_latest_snapshot_for_game, market_from_snapshot, compute_edges
from _team_resolver import resolve_team_id, team_abbr

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def run_one_from_inputs(inputs: dict, market: dict | None, snapshot_file: str | None) -> dict:
    """純組裝:inputs(+ 可選 market) → 完整 bundle(model/market/edges/snapshot)。"""
    rl_point = market["rl"]["home_point"] if market else None
    total_line = market["total"]["line"] if market else None
    model = run_model.predict(
        home_rs=inputs["home_rs_blend"], away_rs=inputs["away_rs_blend"],
        home_starter_fip=inputs["home_starter_fip"], away_starter_fip=inputs["away_starter_fip"],
        home_bullpen_era=inputs["home_bullpen_era"], away_bullpen_era=inputs["away_bullpen_era"],
        pf=inputs["park_factor"], rl_point_home=rl_point, total_line=total_line,
    )
    edges = compute_edges(model, market)
    return {"inputs": inputs, "model": model, "market": market,
            "edges": edges, "snapshot_file": snapshot_file}


def predict_one(date: str, away: str, home: str, suffix: str = "") -> Path:
    """完整單場:抓資料 + 盤口 + 算 + 寫檔。回傳 out_dir。"""
    inputs = fetch_inputs(date, away, home)
    home_team = inputs["raw"]["game"]["home"]["team"]
    away_team = inputs["raw"]["game"]["away"]["team"]
    snap_game, snap_file = find_latest_snapshot_for_game(date, home_team, away_team)
    market = market_from_snapshot(snap_game) if snap_game else None
    bundle = run_one_from_inputs(inputs, market, snap_file)
    out_dir = SKILL_ROOT / "analysis-data" / date / f"{away}@{home}{suffix}"
    report_mod.write_outputs(bundle, out_dir)
    print(f"[OK] {away}@{home}{suffix} → {out_dir}", file=sys.stderr)
    return out_dir


def predict_all(date: str) -> list[Path]:
    """當日全部例行賽。逐場跑,單場失敗只記錄不中斷。"""
    params = {"sportId": 1, "date": date, "hydrate": "probablePitcher(note)"}
    r = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=15)
    r.raise_for_status()
    out = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            if g.get("gameType") != "R":
                continue
            home_id = g["teams"]["home"]["team"]["id"]
            away_id = g["teams"]["away"]["team"]["id"]
            home = team_abbr(home_id, g["teams"]["home"]["team"]["name"])
            away = team_abbr(away_id, g["teams"]["away"]["team"]["name"])
            try:
                out.append(predict_one(date, away, home))
            except Exception as e:
                print(f"[SKIP] {away}@{home}:{e}", file=sys.stderr)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="MLB team-level 預測")
    p.add_argument("--date", required=True, help="ET 開打日 YYYY-MM-DD")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--matchup", help="AWAY@HOME(縮寫)")
    grp.add_argument("--all", action="store_true", help="當日全部例行賽")
    args = p.parse_args(argv)
    if args.all:
        predict_all(args.date)
    else:
        away, _, home = args.matchup.partition("@")
        predict_one(args.date, away.strip(), home.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
