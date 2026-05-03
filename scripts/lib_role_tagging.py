"""lib_role_tagging — heuristic pitcher role inference.

MLB Stats API gives season totals (saves, holds, gamesPlayed, gamesStarted,
inningsPitched) but no `role` field. We infer Closer / Setup / High-leverage
RP / etc. from those counters via first-match-wins rules. Output is consumed
by:
  - dossier_renderer (## 牛棚 / Park section)
  - merge_game_data.bullpen_core_il_count (PR-2 commit 9)
  - signals_lib.core_il_count (PR-3)

Pure functions; no I/O at call time.
"""

from __future__ import annotations

# Roles considered "core bullpen" for IL impact assessment (matchup-factors.md).
# Closer + Setup + High-leverage RP + Co-Closer roll up to the 1/2/3+ ladder.
CORE_BULLPEN_ROLES = frozenset({"Closer", "Setup", "High-leverage RP", "Co-Closer"})


def _result(role: str, confidence: str, evidence: dict, small_sample: bool) -> dict:
    return {
        "core_role": role,
        "core_role_confidence": confidence,  # "data" | "heuristic" | "insufficient"
        "core_role_small_sample": small_sample,
        "core_role_evidence": evidence,
    }


def tag_role(pitcher_stats: dict, team_total_games: int | None = None) -> dict:
    """Infer pitcher's core_role from saves / holds / G / GS / IP.

    First-match-wins rules:
        GS ≥ 5 and GS ≥ 0.6 × G   → Starter (or Opener if avg IP/GS < 3.0)
        SV ≥ 8                    → Closer
        HLD ≥ 8                   → Setup
        HLD ≥ 3 or SV ≥ 2         → High-leverage RP
        IP/G ≥ 2.0 and G ≥ 5      → Long RP
        G ≥ 10                    → Middle RP
        else                      → Unknown

    Args:
        pitcher_stats: {saves, holds, g, gs, ip}; missing keys default to 0.
        team_total_games: total games team has played; if not None and < 30,
            sets `core_role_small_sample` True (April-noise warning).

    Returns:
        {core_role, core_role_confidence, core_role_small_sample, core_role_evidence}
    """
    saves = pitcher_stats.get("saves", 0) or 0
    holds = pitcher_stats.get("holds", 0) or 0
    g = pitcher_stats.get("g", 0) or 0
    gs = pitcher_stats.get("gs", 0) or 0
    ip = pitcher_stats.get("ip", 0.0) or 0.0

    evidence = {"saves": saves, "holds": holds, "g": g, "gs": gs, "ip": round(ip, 1)}
    small_sample = team_total_games is not None and team_total_games < 30

    # Starter / Opener (special starter case with short outings)
    if gs >= 5 and gs >= 0.6 * g:
        if gs > 0 and (ip / gs) < 3.0:
            return _result("Opener", "heuristic", evidence, small_sample)
        return _result("Starter", "data", evidence, small_sample)

    if saves >= 8:
        return _result("Closer", "data", evidence, small_sample)
    if holds >= 8:
        return _result("Setup", "data", evidence, small_sample)
    if holds >= 3 or saves >= 2:
        return _result("High-leverage RP", "heuristic", evidence, small_sample)
    if g >= 5 and (ip / g) >= 2.0:
        return _result("Long RP", "heuristic", evidence, small_sample)
    if g >= 10:
        return _result("Middle RP", "heuristic", evidence, small_sample)
    return _result("Unknown", "insufficient", evidence, small_sample)


def detect_committee_closer(roles: list[dict]) -> list[dict]:
    """Relabel two High-leverage RPs (each with SV ≥ 4) as Co-Closer.

    Closer-by-committee pattern (e.g. 2024 Mets early, 2025 Royals): the team
    has no single dominant SV leader, but two relievers each rack 4+ saves.
    `tag_role` treats each as High-leverage RP individually; this aggregator
    re-tags both as Co-Closer when the pattern fires.

    Mutates `roles` in place AND returns it (chainable).
    """
    high_leverage_with_saves = [
        r for r in roles
        if r.get("core_role") == "High-leverage RP"
        and r.get("core_role_evidence", {}).get("saves", 0) >= 4
    ]
    if len(high_leverage_with_saves) >= 2:
        # Top 2 by saves — these are the de-facto co-closers
        sorted_two = sorted(
            high_leverage_with_saves,
            key=lambda r: r["core_role_evidence"]["saves"],
            reverse=True,
        )[:2]
        for r in sorted_two:
            r["core_role"] = "Co-Closer"
            # Confidence stays "heuristic" (it already was)
    return roles
