"""Decimal odds → 隱含勝率 — 與 scripts/odds_analyzer.py（American odds）獨立。"""
from __future__ import annotations


def decimal_to_implied(decimal_odds: float) -> float:
    """Decimal odds 轉隱含勝率（百分點，1 位小數）。

    無效輸入（≤1.0、None、非數值）一律回 0.0，呼叫端自行過濾。
    """
    try:
        d = float(decimal_odds)
    except (TypeError, ValueError):
        return 0.0
    if d <= 1.0:
        return 0.0
    return round(100.0 / d, 1)


def no_vig_two_way(
    p1: float | None,
    p2: float | None,
) -> tuple[float | None, float | None]:
    """雙邊互斥市場 vig 剝除：fair_p_i = p_i / (p1 + p2) * 100。

    輸入為 raw implied 百分點（含 vig，雙邊合計常 102-104）。
    任一邊 None / ≤ 0 → 回 (None, None)；呼叫端據此決定不寫 no_vig_pct。
    """
    if p1 is None or p2 is None:
        return (None, None)
    try:
        a = float(p1)
        b = float(p2)
    except (TypeError, ValueError):
        return (None, None)
    if a <= 0.0 or b <= 0.0:
        return (None, None)
    total = a + b
    return (round(a / total * 100.0, 1), round(b / total * 100.0, 1))
