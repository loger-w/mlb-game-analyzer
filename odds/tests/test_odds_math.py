"""Tests for odds.lib.odds_math."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from odds_math import decimal_to_implied, no_vig_two_way


# ── decimal_to_implied (regression smoke) ────────────────────────────────────

def test_decimal_to_implied_basic():
    assert decimal_to_implied(2.0) == 50.0
    assert decimal_to_implied(1.91) == 52.4
    assert decimal_to_implied(1.0) == 0.0
    assert decimal_to_implied(0.0) == 0.0


# ── no_vig_two_way ────────────────────────────────────────────────────────────

def test_no_vig_standard_market():
    """52.4 + 51.0 = 103.4 raw → fair 50.7 / 49.3"""
    fair_home, fair_away = no_vig_two_way(52.4, 51.0)
    assert fair_home == 50.7
    assert fair_away == 49.3
    assert round(fair_home + fair_away, 1) == 100.0


def test_no_vig_symmetric_market():
    """雙邊已對稱 → 不變"""
    a, b = no_vig_two_way(50.0, 50.0)
    assert a == 50.0
    assert b == 50.0


def test_no_vig_skewed_favorite():
    """重炮場：80.0 + 23.5 = 103.5 → fair 77.3 / 22.7"""
    fav, dog = no_vig_two_way(80.0, 23.5)
    assert fav == 77.3
    assert dog == 22.7
    assert round(fav + dog, 1) == 100.0


def test_no_vig_one_side_missing():
    """單邊 None → 兩邊都回 None"""
    assert no_vig_two_way(None, 51.0) == (None, None)
    assert no_vig_two_way(52.4, None) == (None, None)
    assert no_vig_two_way(None, None) == (None, None)


def test_no_vig_zero_or_negative():
    """0 / 負值視為無效 → (None, None)"""
    assert no_vig_two_way(0.0, 51.0) == (None, None)
    assert no_vig_two_way(52.4, 0.0) == (None, None)
    assert no_vig_two_way(-1.0, 51.0) == (None, None)
