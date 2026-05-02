#!/usr/bin/env python3
"""prepare_game.py：步驟 1 一鍵整合腳本。

Step 順序：
  A) fetch_game_data → game_data.json + summary
  B) roster_checker × 2（雙隊平行）
  C) pitcher_stats × 2（用 Step A 的 mlbam_id，雙隊平行）
  D) lineup_analyzer × 2（用 Step A 的 mlbam_id，雙隊平行）
  E) merge_game_data → merged.json
  F) dossier_renderer → dossier.md
  G) summary_renderer → summary.md（含 AI 填空 placeholder）

Exit codes：
  0 = success
  1 = 子腳本找不到 / FileNotFoundError
  2 = gameType ≠ "R"
  3 = 雙隊未對戰
  4 = doubleheader 未指定 --game-suffix
  5 = 先發不在 active roster
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
    parser = argparse.ArgumentParser(description="步驟 1：MLB 單場資料收集")
    parser.add_argument("--date", required=True,
                        help="YYYY-MM-DD（ET 開打日，與 MLB Stats API 對齊）")
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


def summary_filename(suffix: str | None) -> str:
    return f"summary-{suffix}.md" if suffix else "summary.md"


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

def step_a(*, date: str, team_abbr: str, output_dir: Path,
           home_abbr: str | None = None, away_abbr: str | None = None,
           game_suffix: str | None = None) -> dict:
    """Step A: 跑 fetch_game_data.py，讀回 JSON，提取 IDs。

    Returns:
        {home_id, away_id, home_name, away_name, home_team_id, away_team_id}

    Exit codes enforced here:
        2 = gameType ≠ "R"
        3 = 雙隊未對戰（fetched teams don't match --home/--away）
        4 = Doubleheader 未指定 --game-suffix
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

    # exit 4: Doubleheader 未指定 --game-suffix
    total_games = game_data.get("_games_on_date_for_team", 1)
    if total_games >= 2 and game_suffix is None:
        gamePk = game_section.get("gamePk", "?")
        print(
            f"[A] ⛔ exit 4: 當日雙重賽（{total_games} 場），請加 --game-suffix G1 或 G2（gamePk={gamePk}）",
            file=sys.stderr,
        )
        sys.exit(4)

    # exit 3: 雙隊未對戰
    if home_abbr is not None and away_abbr is not None:
        # Import lazily to avoid circular import at module level
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
        from _team_resolver import resolve_team_id
        try:
            expected_home = resolve_team_id(home_abbr)
            expected_away = resolve_team_id(away_abbr)
            actual_home = home.get("team_id")
            actual_away = away.get("team_id")
            if actual_home != expected_home or actual_away != expected_away:
                print(
                    f"[A] ⛔ exit 3: 雙隊未對戰 — 期待 {away_abbr}@{home_abbr}"
                    f"（IDs {expected_away}@{expected_home}），"
                    f"實際 fetch 到 ID {actual_away}@{actual_home}",
                    file=sys.stderr,
                )
                sys.exit(3)
        except ValueError:
            pass  # unknown abbr — 讓後續流程處理

    return {
        "home_id": home.get("probable_pitcher_id"),
        "away_id": away.get("probable_pitcher_id"),
        "home_name": home.get("probable_pitcher"),
        "away_name": away.get("probable_pitcher"),
        "home_team_id": home.get("team_id"),
        "away_team_id": away.get("team_id"),
        "game_pk": game_section.get("gamePk"),
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
           season: int, output_dir: Path,
           game_pk: int | None = None) -> None:
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
        if game_pk:
            cmd += ["--game-pk", str(game_pk)]
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
# Step F + G helpers
# ---------------------------------------------------------------------------

def _load_bundle(output_dir: Path) -> dict:
    """Read all JSON outputs from output_dir into a bundle dict."""
    bundle = {}
    for key, fname in [
        ("game_data", "game_data.json"),
        ("home_roster", "home_roster.json"),
        ("away_roster", "away_roster.json"),
        ("home_pitcher", "home_pitcher.json"),
        ("away_pitcher", "away_pitcher.json"),
        ("home_lineup", "home_lineup.json"),
        ("away_lineup", "away_lineup.json"),
        ("merged", "merged.json"),
    ]:
        path = output_dir / fname
        if path.exists():
            bundle[key] = json.loads(path.read_text(encoding="utf-8"))
    return bundle


def step_f(*, output_dir: Path, dossier_path: Path, game_suffix: str | None = None) -> None:
    """Step F: 渲染 dossier.md（File 索引段帶 DH suffix-aware summary 檔名）。"""
    from dossier_renderer import render_dossier
    print(f"[F] dossier.md       → {dossier_path}", file=sys.stderr)
    bundle = _load_bundle(output_dir)
    md = render_dossier(
        bundle,
        game_dir=str(output_dir),
        summary_filename=summary_filename(game_suffix),
    )
    dossier_path.write_text(md, encoding="utf-8")


def step_g(*, output_dir: Path, summary_path: Path, force: bool = False) -> None:
    """Step G: 渲染 summary.md template（含 AI 填空 placeholder）。

    防呆：若 summary 檔已存在且不含 `<!-- AI 補` placeholder，視為「分析師已寫完」，
    保留不覆蓋；用 --force 強制重產（會覆蓋分析師的內容）。
    """
    if summary_path.exists() and not force:
        existing = summary_path.read_text(encoding="utf-8")
        if "<!-- AI 補" not in existing:
            print(
                f"[G] ⚠️  {summary_path.name} 已被編輯（無 AI placeholder），"
                f"保留不覆蓋；用 --force 強制重產",
                file=sys.stderr,
            )
            return

    from summary_renderer import render_summary
    from scoring_formula import predict_with_formula
    print(f"[G] summary  → {summary_path}", file=sys.stderr)
    bundle = _load_bundle(output_dir)
    formula_pred = predict_with_formula(bundle.get("merged", {}))
    md = render_summary(bundle, formula_pred)
    summary_path.write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# Risk Notes
# ---------------------------------------------------------------------------

def _print_risk_notes(output_dir: Path) -> None:
    """從 merged.json 偵測 Flag 3/8，印到 stderr。

    無論是否有 Flag 觸發，header 都必須輸出；無 Flag 時輸出「（無）」。
    """
    print("⚠️  Risk Notes (AI 在 summary 風險提示段處理):", file=sys.stderr)

    merged_path = output_dir / "merged.json"
    if not merged_path.exists():
        print("  （無）", file=sys.stderr)
        return
    merged = json.loads(merged_path.read_text(encoding="utf-8"))

    risk_lines = []
    for side, label in [("home", "主隊"), ("away", "客隊")]:
        # Flag 8: |ERA - xERA| ≥ 1.5（signed delta；abs() 比閾值）
        p = merged.get(f"{side}_pitcher", {}) or {}
        delta = p.get("era_xera_delta")
        if delta is not None:
            try:
                d = float(delta)
                if abs(d) >= 1.5:
                    risk_lines.append(
                        f"  ⚠️  Flag 8 ({label}投手): ERA-xERA = {d:+.2f}（|Δ| ≥ 1.5）"
                    )
            except (ValueError, TypeError):
                pass

        # Flag 3: last7 BABIP ≤ .260 or ≥ .370
        lu = merged.get(f"{side}_lineup", {}) or {}
        recent_babip = lu.get("recent_babip")
        if recent_babip is not None:
            try:
                rb = float(recent_babip)
                if rb <= 0.260 or rb >= 0.370:
                    risk_lines.append(
                        f"  ⚠️  Flag 3  ({label}打線): last7 BABIP = {rb:.3f}（極端值）"
                    )
            except (ValueError, TypeError):
                pass

    if risk_lines:
        for line in risk_lines:
            print(line, file=sys.stderr)
    else:
        print("  （無）", file=sys.stderr)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    # Ensure SCRIPT_DIR is importable for renderers + predict (called inside step_f/step_g)
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    args = parse_args(argv)
    output_dir = compute_output_dir(
        date=args.date, away=args.away, home=args.home,
        game_suffix=args.game_suffix, override=args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step A — sequential; must complete first
    # --date 為 ET 開打日，直接傳給 fetch_game_data.py / MLB Stats API
    ids = step_a(
        date=args.date, team_abbr=args.away, output_dir=output_dir,
        home_abbr=args.home, away_abbr=args.away, game_suffix=args.game_suffix,
    )

    # Steps B/C/D — can run concurrently but each pair is internally parallel
    # Sequential between steps per spec
    step_b(
        home=args.home, away=args.away,
        season=args.season,
        home_pitcher=ids["home_name"] or "",
        away_pitcher=ids["away_name"] or "",
        output_dir=output_dir,
    )

    step_c(
        home_id=ids["home_id"],
        away_id=ids["away_id"],
        home_name=ids["home_name"],
        away_name=ids["away_name"],
        season=args.season,
        output_dir=output_dir,
    )

    step_d(
        home=args.home, away=args.away,
        home_id=ids["home_id"],
        away_id=ids["away_id"],
        season=args.season,
        output_dir=output_dir,
        game_pk=ids.get("game_pk"),
    )

    # Step E — sequential; waits for A+C+D
    step_e(output_dir=output_dir)

    # Steps F + G — sequential; wait for E
    dossier_path = output_dir / dossier_filename(args.game_suffix)
    summary_path = output_dir / summary_filename(args.game_suffix)
    step_f(output_dir=output_dir, dossier_path=dossier_path, game_suffix=args.game_suffix)
    step_g(output_dir=output_dir, summary_path=summary_path, force=args.force)

    # Risk Notes to stderr
    _print_risk_notes(output_dir)

    print(f"\n[Done] output_dir: {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
