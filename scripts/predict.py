"""確定性預測：得分差 → 勝率 → 方向 / 信心。

設計見 docs/superpowers/specs/2026-05-28-deterministic-prediction-design.md。
信心 = 預測那一側的單場勝率，由 winprob 曲線換算，無 AI 介入、無信號 penalty。
"""
from statistics import NormalDist

MARGIN_SD = 4.0   # 單場 run-margin 標準差，歷史 MLB 先驗，非 fit 回測樣本
PUSH_FLOOR = 0.53  # 勝率低於此 → 持平（無方向）

_NORM = NormalDist()


def winprob(gap: float) -> float:
    """P(主隊勝) = Φ(gap / S)，gap = home_score − away_score。"""
    return _NORM.cdf(gap / MARGIN_SD)


def confidence_bucket(p: float) -> str:
    """勝率 → LOW / MEDIUM / HIGH（沿用既有 _effective_confidence_bucket 邊界）。"""
    if p < 0.58:
        return "LOW"
    if p < 0.67:
        return "MEDIUM"
    return "HIGH"
