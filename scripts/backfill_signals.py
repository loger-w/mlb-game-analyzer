"""為既有比賽重算補 signals.json（best-effort）。

TTO3 / pitch_mix 依賴 Statcast 逐球資料，當時若未凍結則不 fire（樣本受限），
core_il / reverse_platoon / chain_break / platoon 用 lineup/roster 算，應可補回。
用法：python scripts/backfill_signals.py --month 2026-05
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _load_bundle(game_dir: Path) -> dict:
    bundle = {}
    for key, fname in [
        ("home_pitcher", "home_pitcher.json"),
        ("away_pitcher", "away_pitcher.json"),
        ("home_lineup", "home_lineup.json"),
        ("away_lineup", "away_lineup.json"),
        ("merged", "merged.json"),
    ]:
        p = game_dir / fname
        if p.exists():
            try:
                bundle[key] = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                bundle[key] = None
    return bundle


def backfill_one(game_dir: Path) -> bool:
    """重算單場 signals.json。回傳是否成功（無 merged.json → False）。"""
    if not (game_dir / "merged.json").exists():
        return False
    from signals_lib import signals_for_bundle
    bundle = _load_bundle(game_dir)
    result = signals_for_bundle(bundle)
    (game_dir / "signals.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="YYYY-MM")
    args = ap.parse_args(argv)
    data_dir = SKILL_ROOT / "analysis-data"
    done = skipped = 0
    for date_dir in sorted(data_dir.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith(args.month):
            continue
        if date_dir.name.endswith(".local-backup"):
            continue
        for game_dir in sorted(date_dir.iterdir()):
            if not game_dir.is_dir():
                continue
            if backfill_one(game_dir):
                done += 1
            else:
                skipped += 1
    print(f"backfill signals: done={done} skipped={skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
