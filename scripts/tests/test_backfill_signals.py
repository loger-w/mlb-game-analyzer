import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backfill_signals import backfill_one


def test_backfill_one_writes_signals(tmp_path, monkeypatch):
    import types
    fake_sig = types.ModuleType("signals_lib")
    fake_sig.signals_for_bundle = lambda bundle: {"signals": [], "fired_count": 0}
    monkeypatch.setitem(sys.modules, "signals_lib", fake_sig)

    (tmp_path / "merged.json").write_text("{}", encoding="utf-8")
    ok = backfill_one(tmp_path)
    assert ok is True
    assert (tmp_path / "signals.json").exists()


def test_backfill_one_skips_when_no_merged(tmp_path):
    ok = backfill_one(tmp_path)
    assert ok is False
    assert not (tmp_path / "signals.json").exists()
