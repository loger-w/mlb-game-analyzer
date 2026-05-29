import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backfill


def test_daterange_inclusive():
    assert backfill.daterange("2026-05-01", "2026-05-03") == [
        "2026-05-01", "2026-05-02", "2026-05-03"]


def test_daterange_single_day():
    assert backfill.daterange("2026-05-07", "2026-05-07") == ["2026-05-07"]
