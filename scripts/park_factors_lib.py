"""Park Factor 共用 lookup helper（merge_game_data + dossier_renderer 共用）。

資料源：scripts/data/park_factors.json（2023-2025 3 年加權，Baseball Savant）。
別名表：把舊球場名（Tropicana / Oakland Coliseum / Minute Maid Park / Dodger Stadium /
Guaranteed Rate Field / Camden Yards）對到 canonical 新名。
"""
from __future__ import annotations

import json
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "park_factors.json"
_PF_DATA = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
PARK_FACTORS: dict = _PF_DATA["park_factors"]
PARK_ALIASES: dict = _PF_DATA["_aliases"]


def resolve_canonical(venue_name: str | None) -> str | None:
    """venue → canonical name（套 alias 表）；None → None。"""
    if not venue_name:
        return None
    return PARK_ALIASES.get(venue_name, venue_name)


def get_park_factor(venue_name: str | None) -> dict:
    """venue → {runs_pf, hr_pf}；未知或 None 回 {}（方便 .get）。

    例：
      get_park_factor("Tropicana Field") → {runs_pf: 100, hr_pf: 109}（透過 alias）
      get_park_factor("Coors Field")     → {runs_pf: 131, hr_pf: 111}
      get_park_factor("Unknown")         → {}
    """
    canonical = resolve_canonical(venue_name)
    if not canonical:
        return {}
    return PARK_FACTORS.get(canonical, {})


def runs_pf(venue_name: str | None, default: float = 100.0) -> float:
    """venue → runs_pf；未知回 default（聯盟平均 100）。"""
    pf = get_park_factor(venue_name)
    runs = pf.get("runs_pf")
    return float(runs) if runs is not None else default
