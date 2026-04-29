"""Decimal odds → 隱含勝率 — 與 scripts/odds_analyzer.py（American odds）獨立。"""


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
