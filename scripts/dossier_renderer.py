"""Dossier renderer：將 Phase 1+2 各 JSON 整合為 ~250 行 markdown，作 AI 主入口。

設計原則（spec §4）：
- 純函式：輸入 dict bundle，輸出 markdown str
- 無 side effect、無 I/O（除 render_dossier 之外）
- 子函式逐節獨立可測

Bundle keys:
  game_data, home_roster, away_roster, home_pitcher, away_pitcher,
  home_lineup, away_lineup, merged
"""
from __future__ import annotations


PA_FLOOR = 30  # spec §4.2 Top 5 候選池下限


def _il_names_from_roster(roster: dict | None) -> set[str]:
    """從 roster_checker.py 輸出取出 IL'd 球員名字。"""
    if not roster:
        return set()
    return {p.get("name") for p in roster.get("injured_list", []) if p.get("name")}


def select_top5_vs_pitcher(lineup: dict | None, il_names: set[str]) -> list[dict]:
    """從 lineup（lineup_analyzer.py 輸出）選 Top 5 vs 對方先發。

    規則（spec §4.2）：
    - active && PA ≥ 30 && !IL'd
    - 按 PA 降序
    - 最多 5 人，候選池 < 5 就少
    """
    candidates = [
        p for p in (lineup or {}).get("lineup", []) or []
        if (p.get("pa") or 0) >= PA_FLOOR and p.get("name") not in il_names
    ]
    candidates.sort(key=lambda p: p.get("pa") or 0, reverse=True)
    return candidates[:5]


def find_last7_top1_outside_pa_top5(
    lineup: dict | None,
    pa_top5_names: list[str],
    il_names: set[str],
) -> dict | None:
    """找出 last7 OPS top1 球員，若不在 PA top 5 內則回傳；否則 None。

    候選池套用同樣的 IL 過濾與 PA ≥ 30。
    """
    candidates = [
        p for p in (lineup or {}).get("lineup", []) or []
        if (p.get("pa") or 0) >= PA_FLOOR and p.get("name") not in il_names
    ]
    candidates_with_last7 = [p for p in candidates if p.get("last7_ops") is not None]
    if not candidates_with_last7:
        return None
    top1 = max(candidates_with_last7, key=lambda p: p["last7_ops"])
    if top1.get("name") in pa_top5_names:
        return None
    return top1


def render_dossier(bundle: dict, *, game_dir: str = "") -> str:
    """主入口：渲染整份 dossier.md。

    後續子節 render 函式由 Task 8b 起逐節補。
    """
    raise NotImplementedError("實作於 Task 8b")
