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


# ---- Step A 後續實作於 Task 11（先把 CLI / helpers 上 commit） ----


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = compute_output_dir(
        date=args.date, away=args.away, home=args.home,
        game_suffix=args.game_suffix, override=args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 後續實作於 Task 11
    raise NotImplementedError("Steps A-G 實作於 Task 11")


if __name__ == "__main__":
    sys.exit(main())
