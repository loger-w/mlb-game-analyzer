#!/usr/bin/env python3
"""Smart Money Tracker — 從 odds_snapshots 讀多份快照，輸出 per-TW-game-date Markdown 報告。

由 Windows Task Scheduler 在 TW 12 / 15 / 18 / 21 觸發，跑於 fetch_odds.py 之後。
報告以台灣時區 (TW, UTC+8) 為主：檔名 = TW 開球日，所有顯示時間皆為 TW。

CLI:
  python odds/analyze_smart_money.py                     # 預設今日 TW
  python odds/analyze_smart_money.py --date 2026-04-30   # TW 開球日；ET 4/29 開打 = TW 4/30
  python odds/analyze_smart_money.py --snapshot-dir /custom/path
  python odds/analyze_smart_money.py --reports-dir  /custom/out
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 讓 lib/ 子模組可被 import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

from snapshot_loader import (
    load_snapshots_for_et_date,
    collect_game_timeline,
)
from movement import compute_game_movement, GameMovementReport
from md_renderer import render

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ── 路徑 ─────────────────────────────────────────────────────────────────────

BASE_DIR             = Path(__file__).resolve().parent.parent   # mlb-game-analyzer/
DEFAULT_SNAPSHOT_DIR = BASE_DIR / "odds_snapshots"
DEFAULT_REPORTS_DIR  = Path(__file__).resolve().parent / "reports"

# MLB 球季固定 EDT = UTC-4（與 fetch_odds.py 對齊）
ET = timezone(timedelta(hours=-4))
TW = timezone(timedelta(hours=+8))


# ── 主程式 ────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart Money Tracker")
    p.add_argument(
        "--date",
        help=("TW game date (YYYY-MM-DD); 預設 = 現在 TW 的日期。"
              "注意：此為 TW 開球日；ET 4/29 開打的場次應使用 --date 2026-04-30。"),
    )
    p.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR),
                   help=f"snapshot JSON 目錄;預設 {DEFAULT_SNAPSHOT_DIR}")
    p.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR),
                   help=f"輸出 md 目錄;預設 {DEFAULT_REPORTS_DIR}")
    return p.parse_args(argv)


def _now_tw() -> datetime:
    return datetime.now(timezone.utc).astimezone(TW)


def _format_rendered_at(now_utc: datetime) -> str:
    et = now_utc.astimezone(ET)
    tw = now_utc.astimezone(TW)
    return f"{tw.strftime('%Y-%m-%d %H:%M TW')}(ET {et.strftime('%H:%M')})"


def _summarize_to_stdout(reports: list[GameMovementReport], out_path: Path) -> None:
    by_tier: dict[str, int] = {"major": 0, "significant": 0, "watch": 0, "quiet": 0}
    for r in reports:
        by_tier[r.tier] = by_tier.get(r.tier, 0) + 1
    print(
        f"OK  寫入 {out_path}  |  "
        f"major {by_tier['major']} / significant {by_tier['significant']} / "
        f"watch {by_tier['watch']} / quiet {by_tier['quiet']}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    now_utc = datetime.now(timezone.utc)
    tw_date = args.date or _now_tw().strftime("%Y-%m-%d")
    snapshot_dir = Path(args.snapshot_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"{tw_date}.md"

    rendered_at = _format_rendered_at(now_utc)

    # TW 開球日 = ET 開球前一日；於 STDOUT 提示語意
    et_prev = (datetime.strptime(tw_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"INFO 分析 TW 日期 {tw_date}（含 ET {et_prev} 開打的場次）")

    # 1. 載入 snapshots（loader 不再依日期過濾，全目錄載入）
    snapshots = load_snapshots_for_et_date(tw_date, snapshot_dir)
    snapshot_times_tw = [s.snapshot_time_tw.strftime("%m-%d %H:%M") for s in snapshots]

    if not snapshots:
        md = render(
            tw_date=tw_date,
            snapshot_count=0,
            snapshot_times_tw=[],
            reports=[],
            rendered_at=rendered_at,
        )
        out_path.write_text(md, encoding="utf-8")
        print(f"INFO {tw_date} 無 snapshot;寫入空白 md → {out_path}")
        return 0

    # 2. 組 timeline
    timelines = collect_game_timeline(snapshots, tw_date)

    # 3. 計算 movements
    reports: list[GameMovementReport] = []
    for timeline in timelines.values():
        try:
            reports.append(compute_game_movement(timeline, now_utc))
        except Exception as e:
            print(
                f"[analyze_smart_money] WARN 計算失敗 game={timeline[0].away}@{timeline[0].home}: {e}",
                file=sys.stderr,
            )

    # 4. tier 排序：major → significant → watch → quiet;同 tier 內按開球時間升冪
    tier_rank = {"major": 0, "significant": 1, "watch": 2, "quiet": 3}
    reports.sort(key=lambda r: (tier_rank.get(r.tier, 9), r.commence_tw))

    # 5. 渲染並寫檔
    md = render(
        tw_date=tw_date,
        snapshot_count=len(snapshots),
        snapshot_times_tw=snapshot_times_tw,
        reports=reports,
        rendered_at=rendered_at,
    )
    out_path.write_text(md, encoding="utf-8")

    _summarize_to_stdout(reports, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
