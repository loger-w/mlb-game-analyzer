#!/usr/bin/env python3
"""One-shot CLV backfill for historical predictions.jsonl records.

Usage:
  python scripts/backfill_clv.py --date 2026-04-18                      # dry-run by default
  python scripts/backfill_clv.py --date 2026-04-18 --no-dry-run         # actually write
  python scripts/backfill_clv.py --all --no-dry-run                     # all dates
  python scripts/backfill_clv.py --date 2026-04-18 --force --no-dry-run # overwrite existing clv

Writes the same fields as upload_results.py (closing_line, clv, rec_to_close).
Only processes records with verified=true. Idempotent unless --force.
"""
import argparse
import json
import os
import sys

from upload_results import _enrich_record_with_clv
from predict import TEAM_ABBREV, _abbrev_to_full_name


def _analysis_root() -> str:
    override = os.environ.get("MLB_ANALYSIS_ROOT_OVERRIDE")
    if override:
        return override
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "analysis-data")


def _snap_dir() -> str:
    override = os.environ.get("MLB_SNAPSHOT_DIR_OVERRIDE")
    if override:
        return override
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "odds_snapshots")


def _resolve_full_team_name(raw):
    if not raw:
        return None
    if raw in TEAM_ABBREV:
        return raw  # already a full name (TEAM_ABBREV keys are full names)
    return _abbrev_to_full_name(raw)


def process_date(date: str, dry_run: bool, force: bool) -> dict:
    path = os.path.join(_analysis_root(), date, "predictions.jsonl")
    if not os.path.isfile(path):
        return {"processed": 0, "updated": 0, "skipped": 0, "errors": 0}

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    snap_dir = _snap_dir()
    counts = {"processed": 0, "updated": 0, "skipped": 0, "errors": 0}
    for rec in records:
        counts["processed"] += 1
        if not rec.get("verified"):
            counts["skipped"] += 1
            continue
        if "clv" in rec and not force:
            counts["skipped"] += 1
            continue
        home_full = _resolve_full_team_name(rec.get("home_team"))
        away_full = _resolve_full_team_name(rec.get("away_team"))
        if not home_full or not away_full:
            counts["errors"] += 1
            continue
        before = json.dumps(rec, sort_keys=True)
        try:
            _enrich_record_with_clv(rec, snap_dir, home_full, away_full, force=force)
        except Exception as exc:
            sys.stderr.write(f"error enriching {home_full} vs {away_full}: {exc}\n")
            counts["errors"] += 1
            continue
        after = json.dumps(rec, sort_keys=True)
        if after != before:
            counts["updated"] += 1

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        print(f"[dry-run] would update {counts['updated']} records in {path}")

    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="Single date YYYY-MM-DD")
    p.add_argument("--all", action="store_true", help="All dates under analysis-data/")
    p.add_argument("--force", action="store_true", help="Overwrite existing clv fields")
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                   help="Actually write (default: dry-run)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                   help="Preview only (default)")
    args = p.parse_args()

    if not args.date and not args.all:
        p.error("must specify --date or --all")

    dates = []
    if args.date:
        dates = [args.date]
    else:
        for entry in sorted(os.listdir(_analysis_root())):
            if len(entry) == 10 and entry[4] == "-" and entry[7] == "-":
                dates.append(entry)

    total = {"processed": 0, "updated": 0, "skipped": 0, "errors": 0}
    for d in dates:
        c = process_date(d, dry_run=args.dry_run, force=args.force)
        for k, v in c.items():
            total[k] += v

    print(f"summary: processed={total['processed']} updated={total['updated']} "
          f"skipped={total['skipped']} errors={total['errors']} "
          f"{'(dry-run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
