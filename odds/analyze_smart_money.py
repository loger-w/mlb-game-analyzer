#!/usr/bin/env python3
"""Smart Money Tracker — 從 odds_snapshots 讀多份快照，輸出 per-ET-date Markdown 報告。

由 Windows Task Scheduler 在 TW 10 / 00 / 03 / 06 / 09 觸發（= ET 22 prev / 12 / 15 / 18 / 21），
跑於 fetch_odds.py 之後。報告以 ET 為主時區（與 MLB 賽程、Pinnacle 市場節奏一致）；
TW 僅作為 OS 排程器的實作細節，不出現在程式碼或報告內容。

CLI:
  python odds/analyze_smart_money.py                     # 預設今日 ET
  python odds/analyze_smart_money.py --date 2026-04-29   # ET 開球日
  python odds/analyze_smart_money.py --snapshot-dir /custom/path
  python odds/analyze_smart_money.py --reports-dir  /custom/out
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# 讓 lib/ 子模組可被 import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

from snapshot_loader import (
    load_snapshots_for_et_date,
    collect_game_timeline,
)
from movement import compute_game_movement, GameMovementReport
from md_renderer import render
from timeutil import ET, TW, parse_iso_utc

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ── 路徑 ─────────────────────────────────────────────────────────────────────

BASE_DIR             = Path(__file__).resolve().parent.parent   # mlb-game-analyzer/
DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent / "odds_snapshots"
DEFAULT_REPORTS_DIR  = Path(__file__).resolve().parent / "reports"

# ── 主程式 ────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart Money Tracker")
    p.add_argument(
        "--date",
        help="ET game date (YYYY-MM-DD); 預設 = 現在 ET 的日期。",
    )
    p.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR),
                   help=f"snapshot JSON 目錄;預設 {DEFAULT_SNAPSHOT_DIR}")
    p.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR),
                   help=f"輸出 md 目錄;預設 {DEFAULT_REPORTS_DIR}")
    return p.parse_args(argv)


def _now_et() -> datetime:
    return datetime.now(timezone.utc).astimezone(ET)


def _format_rendered_at(now_utc: datetime) -> str:
    """ET 為主、TW 副欄附在後面（給 TW 使用者對時用）。"""
    et = now_utc.astimezone(ET)
    tw = now_utc.astimezone(TW)
    return f"{et.strftime('%Y-%m-%d %H:%M ET')} (TW {tw.strftime('%Y-%m-%d %H:%M')})"


def _summarize_to_stdout(reports: list[GameMovementReport], out_path: Path) -> None:
    by_tier: dict[str, int] = {"major": 0, "significant": 0, "watch": 0, "quiet": 0}
    for r in reports:
        by_tier[r.tier] = by_tier.get(r.tier, 0) + 1
    print(
        f"OK  寫入 {out_path}  |  "
        f"major {by_tier['major']} / significant {by_tier['significant']} / "
        f"watch {by_tier['watch']} / quiet {by_tier['quiet']}"
    )


def _discover_et_dates(snapshots: list) -> list[str]:
    """掃描所有 snapshot 的 commence_utc，回傳出現過的 ET 日期列表（升冪排序）。"""
    dates: set[str] = set()
    for snap in snapshots:
        for g in snap.games:
            utc = parse_iso_utc(g.get("commence_utc"))
            if utc is None:
                continue
            dates.add(utc.astimezone(ET).strftime("%Y-%m-%d"))
    return sorted(dates)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    now_utc = datetime.now(timezone.utc)
    snapshot_dir = Path(args.snapshot_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rendered_at = _format_rendered_at(now_utc)

    # 1. 載入全部 snapshots（loader 不依日期過濾）
    snapshots = load_snapshots_for_et_date("", snapshot_dir)
    snapshot_times_et = [s.snapshot_time_et.strftime("%m-%d %H:%M") for s in snapshots]

    if not snapshots:
        et_date = args.date or _now_et().strftime("%Y-%m-%d")
        out_path = reports_dir / f"{et_date}.md"
        md = render(
            et_date=et_date,
            snapshot_count=0,
            snapshot_times_et=[],
            reports=[],
            rendered_at=rendered_at,
        )
        out_path.write_text(md, encoding="utf-8")
        print(f"INFO {et_date} 無 snapshot;寫入空白 md → {out_path}")
        return 0

    # 2. 決定要產生哪些 ET 日期的報告
    #    --date 指定 → 只產生那一天；否則自動掃描所有 snapshot 內出現的 ET 日期
    et_dates = [args.date] if args.date else _discover_et_dates(snapshots)

    tier_rank = {"major": 0, "significant": 1, "watch": 2, "quiet": 3}

    for et_date in et_dates:
        out_path = reports_dir / f"{et_date}.md"
        print(f"INFO 分析 ET 日期 {et_date}")

        # 3. 組 timeline（只取該 ET 日的場次）
        timelines = collect_game_timeline(snapshots, et_date)

        # 4. 計算 movements
        reports: list[GameMovementReport] = []
        for timeline in timelines.values():
            try:
                reports.append(compute_game_movement(timeline, now_utc))
            except Exception as e:
                print(
                    f"[analyze_smart_money] WARN 計算失敗 game={timeline[0].away}@{timeline[0].home}: {e}",
                    file=sys.stderr,
                )

        # 5. tier 排序：major → significant → watch → quiet;同 tier 內按開球時間升冪
        reports.sort(key=lambda r: (tier_rank.get(r.tier, 9), r.commence_et))

        # 6. 渲染並寫檔
        md = render(
            et_date=et_date,
            snapshot_count=len(snapshots),
            snapshot_times_et=snapshot_times_et,
            reports=reports,
            rendered_at=rendered_at,
        )
        out_path.write_text(md, encoding="utf-8")
        _summarize_to_stdout(reports, out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
