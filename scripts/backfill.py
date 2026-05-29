#!/usr/bin/env python3
"""回填過去日期的 point-in-time 預測(features.json),供回測。

用法:
  python scripts/backfill.py --start 2026-05-01 --end 2026-05-25
  python scripts/backfill.py --start 2026-05-01 --end 2026-05-25 --refresh-cache
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import predict_game
import bullpen

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def daterange(start: str, end: str) -> list[str]:
    """含頭含尾的日期清單(YYYY-MM-DD)。"""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    out = []
    d = s
    while d <= e:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="回填 point-in-time 預測供回測")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--refresh-cache", action="store_true", help="強制重建 relief index")
    args = p.parse_args(argv)

    year = int(args.start[:4])
    print(f"預建 relief index(through {args.end})…可能需數分鐘", file=sys.stderr)
    bullpen.load_or_build_index(year, needed_through=args.end, refresh=args.refresh_cache)

    for d in daterange(args.start, args.end):
        try:
            outs = predict_game.predict_all(d)
            print(f"[{d}] {len(outs)} 場", file=sys.stderr)
        except Exception as e:
            print(f"[{d}] 失敗:{e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
