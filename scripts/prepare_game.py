#!/usr/bin/env python3
"""prepare_game.py：Phase 1+2 一鍵整合腳本（spec 2026-04-28-prepare-game-script）。

Step 順序（spec §3.2）：
  A) fetch_game_data → game_data.json + summary
  B) roster_checker × 2（雙隊平行）
  C) pitcher_stats × 2（用 Step A 的 mlbam_id，雙隊平行）
  D) lineup_analyzer × 2（用 Step A 的 mlbam_id，雙隊平行）
  E) merge_game_data → merged.json
  F) dossier_renderer → dossier.md
  G) phase3_skeleton_renderer → phase3_skeleton.md

不再做 Step C-prior（YoY 補跑）— spec §3.2.

Exit codes（spec §3.1）：
  0 = success
  2 = gameType ≠ "R"
  3 = 雙隊未對戰
  4 = doubleheader 未指定 --game-suffix
  5 = 先發不在 active roster
  6 = （保留給 predict.py --ou-stars 必填錯誤）
  7 = API 失敗
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1+2 一鍵整合（spec 2026-04-28）")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--away", required=True, help="客隊縮寫，如 TB")
    parser.add_argument("--home", required=True, help="主隊縮寫，如 CLE")
    parser.add_argument("--output-dir", default=None,
                        help="覆蓋預設目錄（analysis-data/{date}/{away}@{home}[-Gn]）")
    parser.add_argument("--season", type=int, default=None,
                        help="預設 = year of --date")
    parser.add_argument("--game-suffix", choices=["G1", "G2"], default=None,
                        help="Doubleheader 用")
    parser.add_argument("--force", action="store_true", help="覆蓋既有輸出檔")
    args = parser.parse_args(argv)
    if args.season is None:
        args.season = int(args.date[:4])
    return args


def compute_output_dir(*, date: str, away: str, home: str,
                       game_suffix: str | None, override: str | None) -> Path:
    if override:
        return Path(override)
    suffix = f"-{game_suffix}" if game_suffix else ""
    return Path(f"analysis-data/{date}/{away}@{home}{suffix}")


def dossier_filename(suffix: str | None) -> str:
    return f"dossier-{suffix}.md" if suffix else "dossier.md"


def skeleton_filename(suffix: str | None) -> str:
    return f"phase3_skeleton-{suffix}.md" if suffix else "phase3_skeleton.md"


def run_step(label: str, cmd: list[str]) -> str:
    """跑單一子步驟。失敗 → propagate exit code + stderr。回傳 stdout。

    Caller responsible for building absolute paths into cmd（例：[..., "-o", str(output_dir / "x.json")]）。
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    except FileNotFoundError as e:
        print(f"[{label}] ⛔ 找不到腳本：{e}", file=sys.stderr)
        sys.exit(1)
    if result.returncode != 0:
        print(f"[{label}] ⛔ exit {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout


# ---------------------------------------------------------------------------
# Step A: fetch_game_data
# ---------------------------------------------------------------------------

def step_a(*, date: str, team_abbr: str, output_dir: Path) -> dict:
    """Step A: 跑 fetch_game_data.py，讀回 JSON，提取 IDs。

    Returns:
        {home_id, away_id, home_name, away_name, home_team_id, away_team_id}
    """
    out_path = output_dir / "game_data.json"
    run_step("A", [
        PYTHON,
        str(SCRIPT_DIR / "fetch_game_data.py"),
        "--date", date,
        "--team", team_abbr,
        "-o", str(out_path),
    ])
    print(f"[A] game_data        ✓", file=sys.stderr)

    game_data = json.loads(out_path.read_text(encoding="utf-8"))

    # Support both real structure (game.home.*) and test stub (home.*)
    game_section = game_data.get("game", game_data)

    # Validate gameType (only reject if explicitly non-R)
    game_type = game_section.get("gameType") or game_data.get("_meta", {}).get("gameType")
    if game_type is not None and game_type != "R":
        print(f"[A] ⛔ gameType={game_type!r}（非例行賽，exit 2）", file=sys.stderr)
        sys.exit(2)

    home = game_section.get("home", {})
    away = game_section.get("away", {})

    return {
        "home_id": home.get("probable_pitcher_id"),
        "away_id": away.get("probable_pitcher_id"),
        "home_name": home.get("probable_pitcher"),
        "away_name": away.get("probable_pitcher"),
        "home_team_id": home.get("team_id"),
        "away_team_id": away.get("team_id"),
    }


# ---------------------------------------------------------------------------
# Step B: roster_checker × 2 parallel
# ---------------------------------------------------------------------------

def step_b(*, home: str, away: str, season: int,
           home_pitcher: str, away_pitcher: str, output_dir: Path) -> None:
    """Step B: 雙隊 roster_checker 平行跑。"""
    sides = [
        ("home", home, home_pitcher, output_dir / "home_roster.json"),
        ("away", away, away_pitcher, output_dir / "away_roster.json"),
    ]

    def _run_side(side_tuple):
        side, team, pitcher, out_path = side_tuple
        cmd = [
            PYTHON,
            str(SCRIPT_DIR / "roster_checker.py"),
            "--team", team,
            "--season", str(season),
            "--expected-starter", pitcher,
            "-o", str(out_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        except FileNotFoundError as e:
            return side, -1, "", str(e)
        return side, result.returncode, result.stdout, result.stderr

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(_run_side, s): s for s in sides}
        results = {}
        for future in concurrent.futures.as_completed(futures):
            side, code, stdout, stderr = future.result()
            results[side] = (code, stdout, stderr)

    for side, (code, stdout, stderr) in results.items():
        if code != 0:
            combined = (stdout or "") + (stderr or "")
            if "STARTER_NOT_ACTIVE" in combined:
                print(f"[B] ⛔ {side} STARTER_NOT_ACTIVE（exit 5）", file=sys.stderr)
                if stderr:
                    print(stderr, file=sys.stderr)
                sys.exit(5)
            print(f"[B] ⛔ {side} exit {code}", file=sys.stderr)
            if stderr:
                print(stderr, file=sys.stderr)
            sys.exit(code)

    print(f"[B] roster (home+away) ✓", file=sys.stderr)


# ---------------------------------------------------------------------------
# Step C: pitcher_stats × 2 parallel
# ---------------------------------------------------------------------------

def step_c(*, home_id: int | None, away_id: int | None,
           home_name: str | None, away_name: str | None,
           season: int, output_dir: Path) -> None:
    """Step C: 雙隊 pitcher_stats 平行跑（用 --mlbam-id 略過 name lookup）。"""
    sides = [
        ("home", home_id, home_name or "Home Pitcher", output_dir / "home_pitcher.json"),
        ("away", away_id, away_name or "Away Pitcher", output_dir / "away_pitcher.json"),
    ]

    def _run_side(side_tuple):
        side, mlbam_id, name, out_path = side_tuple
        cmd = [
            PYTHON,
            str(SCRIPT_DIR / "pitcher_stats.py"),
            "--name", name,
            "--year", str(season),
            "-o", str(out_path),
        ]
        if mlbam_id:
            cmd += ["--mlbam-id", str(mlbam_id)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        except FileNotFoundError as e:
            return side, -1, "", str(e)
        return side, result.returncode, result.stdout, result.stderr

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(_run_side, s): s for s in sides}
        results = {}
        for future in concurrent.futures.as_completed(futures):
            side, code, stdout, stderr = future.result()
            results[side] = (code, stdout, stderr)

    for side, (code, stdout, stderr) in results.items():
        if code != 0:
            print(f"[C] ⛔ {side} pitcher_stats exit {code}", file=sys.stderr)
            if stderr:
                print(stderr, file=sys.stderr)
            sys.exit(code)

    print(f"[C] pitcher_stats (home+away) ✓", file=sys.stderr)


# ---------------------------------------------------------------------------
# Step D: lineup_analyzer × 2 parallel (vs opposing pitcher)
# ---------------------------------------------------------------------------

def step_d(*, home: str, away: str,
           home_id: int | None, away_id: int | None,
           season: int, output_dir: Path) -> None:
    """Step D: 雙隊 lineup_analyzer 平行跑。
    home 打線 vs away 投手（opposing_id = away_id）
    away 打線 vs home 投手（opposing_id = home_id）
    """
    sides = [
        ("home", home, away_id, output_dir / "home_lineup.json"),
        ("away", away, home_id, output_dir / "away_lineup.json"),
    ]

    def _run_side(side_tuple):
        side, team, opposing_id, out_path = side_tuple
        cmd = [
            PYTHON,
            str(SCRIPT_DIR / "lineup_analyzer.py"),
            "--team", team,
            "--year", str(season),
            "-o", str(out_path),
        ]
        if opposing_id:
            cmd += ["--opposing-pitcher-id", str(opposing_id)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        except FileNotFoundError as e:
            return side, -1, "", str(e)
        return side, result.returncode, result.stdout, result.stderr

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(_run_side, s): s for s in sides}
        results = {}
        for future in concurrent.futures.as_completed(futures):
            side, code, stdout, stderr = future.result()
            results[side] = (code, stdout, stderr)

    for side, (code, stdout, stderr) in results.items():
        if code != 0:
            print(f"[D] ⛔ {side} lineup_analyzer exit {code}", file=sys.stderr)
            if stderr:
                print(stderr, file=sys.stderr)
            sys.exit(code)

    print(f"[D] lineup (home+away) ✓", file=sys.stderr)


# ---------------------------------------------------------------------------
# Step E: merge_game_data
# ---------------------------------------------------------------------------

def step_e(*, output_dir: Path) -> None:
    """Step E: merge_game_data.py → merged.json（sequential）。"""
    out_path = output_dir / "merged.json"
    cmd = [
        PYTHON,
        str(SCRIPT_DIR / "merge_game_data.py"),
        "--game", str(output_dir / "game_data.json"),
        "--home-pitcher", str(output_dir / "home_pitcher.json"),
        "--away-pitcher", str(output_dir / "away_pitcher.json"),
        "--home-lineup", str(output_dir / "home_lineup.json"),
        "--away-lineup", str(output_dir / "away_lineup.json"),
        "-o", str(out_path),
    ]
    run_step("E", cmd)
    print(f"[E] merged.json      ✓", file=sys.stderr)



# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = compute_output_dir(
        date=args.date, away=args.away, home=args.home,
        game_suffix=args.game_suffix, override=args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Steps A-G 後續實作
    raise NotImplementedError("Steps A-G integrating...")


if __name__ == "__main__":
    sys.exit(main())
