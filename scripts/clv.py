"""P2 — CLV Infrastructure + M5 line movement.

Pure-function helpers for Closing Line Value computation, snapshot discovery,
and line-movement detection. No file I/O side effects beyond snapshot reads.

Spec: docs/superpowers/specs/2026-04-18-p2-clv-infra-design.md
"""
from __future__ import annotations

from typing import Optional

from odds_analyzer import decimal_to_american


def compute_clv_cents(rec_decimal: float, close_decimal: float) -> int:
    """American cents difference: rec - close. Positive = beat closing.

    For both sides (favorite negative American, underdog positive American),
    a higher American number means a better price for the bettor, so
    american(rec) - american(close) correctly reports beat (positive) / lose (negative).
    """
    rec_am = decimal_to_american(rec_decimal)
    close_am = decimal_to_american(close_decimal)
    return int(round(rec_am - close_am))


def compute_clv_pct_no_vig(
    rec_side_dec: float,
    rec_other_dec: float,
    close_side_dec: float,
    close_other_dec: float,
) -> float:
    """No-vig implied probability delta (close - rec) in percentage points.

    Positive = rec side was priced below closing's true estimate → beat.

    For each snapshot, compute no-vig prob of the bet side by dividing its raw
    implied by the sum of both sides' raw implied (strips the book's hold).
    """
    rec_raw = (1.0 / rec_side_dec, 1.0 / rec_other_dec)
    rec_no_vig = rec_raw[0] / (rec_raw[0] + rec_raw[1])
    close_raw = (1.0 / close_side_dec, 1.0 / close_other_dec)
    close_no_vig = close_raw[0] / (close_raw[0] + close_raw[1])
    delta_pct = (close_no_vig - rec_no_vig) * 100
    return round(delta_pct, 2)
