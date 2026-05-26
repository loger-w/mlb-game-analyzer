"""End-to-end smoke test: run backtest on one day, verify outputs exist."""
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent.parent


def test_backtest_run_single_day(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = subprocess.run(
        [
            sys.executable, "scripts/backtest.py", "run",
            "--month", "2026-05",
            "--days", "2026-05-02",
            "--out", str(out_dir),
        ],
        cwd=SKILL_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (out_dir / "2026-05-report.md").exists()
    assert (out_dir / "2026-05-details.csv").exists()
    report = (out_dir / "2026-05-report.md").read_text(encoding="utf-8")
    assert "## 資料健康度" in report
    csv_text = (out_dir / "2026-05-details.csv").read_text(encoding="utf-8")
    header = csv_text.splitlines()[0]
    assert "date" in header and "matchup" in header
    assert len(csv_text.splitlines()) > 1  # at least one data row
