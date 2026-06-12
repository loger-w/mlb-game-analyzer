"""Tests for scripts/park_factors_lib.py — runs_pf 是 fetch_inputs 用的唯一入口。"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from park_factors_lib import resolve_canonical, runs_pf


def test_resolve_canonical_maps_alias_to_new_name():
    assert resolve_canonical("Tropicana Field") == "George M. Steinbrenner Field"


def test_resolve_canonical_passes_through_unknown():
    assert resolve_canonical("Unknown Park") == "Unknown Park"


def test_resolve_canonical_none_and_empty():
    assert resolve_canonical(None) is None
    assert resolve_canonical("") is None


def test_runs_pf_known_venue():
    assert runs_pf("Coors Field") == 131.0


def test_runs_pf_resolves_alias():
    # Oakland Coliseum → Sutter Health Park(runs_pf=109,≠default 100,
    # 能區分「真的走了 alias」和「fallback 到 default」)
    assert runs_pf("Oakland Coliseum") == 109.0
    assert runs_pf("Oakland Coliseum") == runs_pf("Sutter Health Park")


def test_runs_pf_unknown_returns_default():
    assert runs_pf("FakePark") == 100.0
    assert runs_pf("FakePark", default=99.0) == 99.0


def test_runs_pf_none_returns_default():
    assert runs_pf(None) == 100.0
