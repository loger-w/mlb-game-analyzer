"""Unit tests for scripts/clv.py (P2 CLV infrastructure)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from clv import compute_clv_cents, compute_clv_pct_no_vig


class TestComputeClvCents:
    def test_beat_closing_favorite(self):
        # Rec HOME at -135 (1.74), closes at -179 (1.56) → we beat close by +44c
        assert compute_clv_cents(1.74, 1.56) == 44

    def test_beat_closing_underdog(self):
        # Rec AWAY at +128 (2.28), closes at +152 (2.52) → we beat by +24c (plus side, bigger = better? NO)
        # American: +128 vs +152 — the higher American number on plus side IS better (more payout)
        # But WE bet at +128 and line closed at +152, meaning we took the WORSE price → CLV = 128-152 = -24c
        # For beat: rec must be a BETTER price than close. Higher positive = better payout = better price.
        assert compute_clv_cents(2.28, 2.52) == -24

    def test_tied(self):
        assert compute_clv_cents(1.91, 1.91) == 0

    def test_lose_to_closing(self):
        # Rec HOME at -120 (1.83), closes at -110 (1.91) → rec price was worse → CLV negative
        assert compute_clv_cents(1.83, 1.91) == -10

    def test_cross_zero_decimal_2(self):
        # decimal 2.0 is exactly +100
        assert compute_clv_cents(2.0, 2.0) == 0

    def test_invalid_decimal(self):
        with pytest.raises(ValueError):
            compute_clv_cents(1.0, 1.5)


class TestComputeClvPctNoVig:
    def test_beat_ml_home(self):
        # rec: home 1.74, away 2.28 → no_vig_home ≈ 56.72%
        # close: home 1.56, away 2.52 → no_vig_home ≈ 61.77%
        # delta (close - rec) ≈ 5.05 → HOME backer beat market by ~5.05 pp
        result = compute_clv_pct_no_vig(1.74, 2.28, 1.56, 2.52)
        assert result == pytest.approx(5.05, abs=0.15)

    def test_tied(self):
        assert compute_clv_pct_no_vig(1.91, 1.91, 1.91, 1.91) == pytest.approx(0.0, abs=0.01)

    def test_lose(self):
        # Reverse: rec 1.56/2.52, close 1.74/2.28 → -5.05
        result = compute_clv_pct_no_vig(1.56, 2.52, 1.74, 2.28)
        assert result == pytest.approx(-5.05, abs=0.15)

    def test_high_vig_scenario(self):
        # Both books with heavy vig — no-vig extraction should still work
        # rec both -120 (1.83), close both -120 (1.83) → same → 0
        assert compute_clv_pct_no_vig(1.83, 1.83, 1.83, 1.83) == pytest.approx(0.0, abs=0.01)

    def test_rounded_to_2dp(self):
        result = compute_clv_pct_no_vig(1.74, 2.28, 1.56, 2.52)
        # Must be 2 decimal places
        assert abs(result - round(result, 2)) < 1e-9
