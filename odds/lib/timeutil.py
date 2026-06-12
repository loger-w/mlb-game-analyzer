"""odds 樹共用的時區常數與 ISO 8601 解析。

ET 固定 UTC-4(MLB 賽季在 EDT 內);TW 僅供 OS 排程器換算說明用。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

ET = timezone(timedelta(hours=-4))
TW = timezone(timedelta(hours=+8))


def parse_iso_utc(s) -> Optional[datetime]:
    """ISO 8601 字串 → aware datetime。容忍 'Z' 後綴;None/空/壞格式 → None。"""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
