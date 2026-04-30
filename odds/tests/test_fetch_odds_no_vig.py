"""Tests for fetch_odds.parse_game no-vig attachment."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fetch_odds import parse_game


def _raw_game(
    *,
    ml_home_odds=1.91,
    ml_away_odds=2.05,
    ou_over_odds=1.92,
    ou_under_odds=2.00,
    ou_point=8.5,
    rl_home_odds=1.95,
    rl_away_odds=1.95,
    rl_home_point=-1.5,
    rl_away_point=1.5,
    include_ou_under=True,
    include_rl=True,
):
    """合成 the-odds-api 風格的 raw game dict（最小可解析）。"""
    home = "Cleveland Guardians"
    away = "Tampa Bay Rays"

    ou_outcomes = [{"name": "Over", "price": ou_over_odds, "point": ou_point}]
    if include_ou_under:
        ou_outcomes.append({"name": "Under", "price": ou_under_odds, "point": ou_point})

    markets = [
        {
            "key": "h2h",
            "outcomes": [
                {"name": home, "price": ml_home_odds},
                {"name": away, "price": ml_away_odds},
            ],
        },
        {"key": "totals", "outcomes": ou_outcomes},
    ]
    if include_rl:
        markets.append({
            "key": "spreads",
            "outcomes": [
                {"name": home, "price": rl_home_odds, "point": rl_home_point},
                {"name": away, "price": rl_away_odds, "point": rl_away_point},
            ],
        })

    return {
        "home_team": home,
        "away_team": away,
        "commence_time": "2026-04-30T23:11:00Z",
        "bookmakers": [{"key": "pinnacle", "title": "Pinnacle", "markets": markets}],
    }


# ── 基本 attach ───────────────────────────────────────────────────────────────

def test_parse_game_attaches_no_vig_to_ml():
    parsed = parse_game(_raw_game())
    pinn = parsed["bookmakers"]["pinnacle"]
    home_ml = pinn["ml"]["Cleveland Guardians"]
    away_ml = pinn["ml"]["Tampa Bay Rays"]
    assert "no_vig_pct" in home_ml
    assert "no_vig_pct" in away_ml
    assert round(home_ml["no_vig_pct"] + away_ml["no_vig_pct"], 1) == 100.0


def test_parse_game_attaches_no_vig_to_ou():
    parsed = parse_game(_raw_game())
    ou = parsed["bookmakers"]["pinnacle"]["ou"]
    assert "no_vig_pct" in ou["Over"]
    assert "no_vig_pct" in ou["Under"]
    assert round(ou["Over"]["no_vig_pct"] + ou["Under"]["no_vig_pct"], 1) == 100.0


def test_parse_game_attaches_no_vig_to_rl():
    parsed = parse_game(_raw_game())
    rl = parsed["bookmakers"]["pinnacle"]["rl"]
    home = rl["Cleveland Guardians"]
    away = rl["Tampa Bay Rays"]
    assert "no_vig_pct" in home
    assert "no_vig_pct" in away
    assert round(home["no_vig_pct"] + away["no_vig_pct"], 1) == 100.0


# ── 邊界：缺對邊就不該有 no_vig_pct ──────────────────────────────────────────

def test_parse_game_skips_no_vig_when_under_missing():
    parsed = parse_game(_raw_game(include_ou_under=False))
    over = parsed["bookmakers"]["pinnacle"]["ou"]["Over"]
    assert "no_vig_pct" not in over


def test_parse_game_skips_no_vig_when_rl_missing():
    parsed = parse_game(_raw_game(include_rl=False))
    pinn = parsed["bookmakers"]["pinnacle"]
    assert pinn["rl"] == {}
    # ml 仍應該有 no_vig_pct（與 rl 缺漏無關）
    assert "no_vig_pct" in pinn["ml"]["Cleveland Guardians"]


# ── implied_pct 不變（向後兼容）─────────────────────────────────────────────

def test_parse_game_preserves_raw_implied_pct():
    parsed = parse_game(_raw_game())
    home_ml = parsed["bookmakers"]["pinnacle"]["ml"]["Cleveland Guardians"]
    assert home_ml["implied_pct"] == 52.4   # 1 / 1.91 * 100
    # raw 含 vig，故 ≠ no_vig
    assert home_ml["implied_pct"] != home_ml["no_vig_pct"]
