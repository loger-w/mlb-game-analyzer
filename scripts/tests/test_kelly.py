"""Unit tests for fractional Kelly helpers."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from odds_analyzer import calc_fractional_kelly


def test_positive_edge_quarter_kelly_no_cap():
    """p=0.55, ml=-110, quarter Kelly, no cap engaged."""
    result = calc_fractional_kelly(0.55, -110, divisor=4, cap_pct=3.0, unit_size_pct=1.0)
    # b = 100/110 ≈ 0.909
    # raw = (0.55 * 1.909 - 1) / 0.909 * 100 ≈ 5.5
    assert result["raw_kelly_pct"] == 5.5
    # fractional = 5.5 / 4 = 1.375
    assert result["fractional_pct"] == 1.375
    # cap not hit
    assert result["capped_pct"] == 1.375
    # units = 1.375 / 1.0 rounded to nearest 0.5 = 1.5
    assert result["units"] == 1.5


def test_zero_edge_returns_zero():
    """p at implied prob — no edge."""
    # at -110, implied = 110/210 ≈ 0.5238; use exactly that
    result = calc_fractional_kelly(0.5238, -110, divisor=4, cap_pct=3.0, unit_size_pct=1.0)
    # raw should round to ~0 (boundary)
    assert result["raw_kelly_pct"] <= 0.1
    assert result["fractional_pct"] <= 0.1
    assert result["capped_pct"] <= 0.1
    assert result["units"] == 0.0


def test_negative_edge_returns_zero():
    """p < implied — no bet."""
    result = calc_fractional_kelly(0.45, -110, divisor=4, cap_pct=3.0, unit_size_pct=1.0)
    assert result["raw_kelly_pct"] == 0
    assert result["fractional_pct"] == 0
    assert result["capped_pct"] == 0
    assert result["units"] == 0.0


def test_cap_engaged_high_edge_long_odds():
    """p=0.50, ml=+250 → big raw Kelly, cap should trigger."""
    result = calc_fractional_kelly(0.50, +250, divisor=4, cap_pct=3.0, unit_size_pct=1.0)
    # b=2.5, raw = (0.50 * 3.5 - 1) / 2.5 * 100 = 30
    assert result["raw_kelly_pct"] == 30.0
    # fractional = 30 / 4 = 7.5
    assert result["fractional_pct"] == 7.5
    # cap at 3.0
    assert result["capped_pct"] == 3.0
    # units = 3.0 / 1.0 = 3.0
    assert result["units"] == 3.0


def test_half_kelly_divisor():
    """divisor=2 doubles fractional output."""
    result = calc_fractional_kelly(0.55, -110, divisor=2, cap_pct=3.0, unit_size_pct=1.0)
    # raw same = 5.5
    assert result["raw_kelly_pct"] == 5.5
    # fractional = 5.5 / 2 = 2.75
    assert result["fractional_pct"] == 2.75
    assert result["capped_pct"] == 2.75
    # units = 2.75, round to nearest 0.5 = 3.0
    assert result["units"] == 3.0


from odds_analyzer import decimal_to_american


def test_decimal_to_american_favorite():
    """dec=1.83 → American -120."""
    assert decimal_to_american(1.83) == -120


def test_decimal_to_american_underdog():
    """dec=2.50 → American +150."""
    assert decimal_to_american(2.50) == 150


def test_decimal_to_american_even():
    """dec=2.00 → American +100."""
    assert decimal_to_american(2.00) == 100


def test_decimal_to_american_invalid():
    """dec<=1.0 should raise ValueError."""
    import pytest
    with pytest.raises(ValueError):
        decimal_to_american(1.0)
    with pytest.raises(ValueError):
        decimal_to_american(0.5)
