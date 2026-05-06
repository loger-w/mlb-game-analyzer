"""跨模組共用的 None-safe 格式化 / 數值轉換 helper。

故意命名為 module-private（底線開頭），但提供 import 路徑給內部其他腳本使用。
"""
from __future__ import annotations


def md_fmt(v, decimals: int = 2) -> str:
    """格式化數值；None → '—'。"""
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        if decimals == 0:
            return f"{v:.0f}"
        return f"{v:.{decimals}f}"
    return str(v)


def safe_float(v) -> float | None:
    """轉 float；None / 空字串 / 非數值 → None。"""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def safe_round(value, decimals: int):
    """safe_float + round；任何失敗回原值。"""
    f = safe_float(value)
    if f is None:
        return value
    return round(f, decimals)
