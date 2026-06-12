"""scripts/lib/render.py 格式化 helper 的邊界測試 — NaN/inf 必須輸出 '—' 而非 'nan%'。"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.render import _pct, _f, _clvpp


def test_pct_formats_number_and_none():
    assert _pct(0.512) == "51.2%"
    assert _pct(None) == "—"


def test_f_formats_number_and_none():
    assert _f(1.2345) == "1.23"
    assert _f(1.2345, nd=1) == "1.2"
    assert _f(None) == "—"


def test_clvpp_signed_format():
    assert _clvpp(1.5) == "+1.50pp"
    assert _clvpp(-0.25) == "-0.25pp"
    assert _clvpp(None) == "—"


def test_helpers_render_nan_inf_as_em_dash():
    """0/0 等鏈式計算可能產生 NaN/inf;報告不得出現 'nan%' / '+infpp'。"""
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert _pct(bad) == "—"
        assert _f(bad) == "—"
        assert _clvpp(bad) == "—"
