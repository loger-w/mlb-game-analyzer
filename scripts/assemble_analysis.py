#!/usr/bin/env python3
"""assemble_analysis.py — Assemble GameAnalysis JSON (schema v1.0) from script outputs.

Reads all Phase 1-4 JSON outputs and the latest prediction record, then
assembles a structured analysis for the prediction detail page.

Usage:
    python assemble_analysis.py \
      --game game_data.json \
      --home-pitcher home_pitcher.json \
      --away-pitcher away_pitcher.json \
      --home-lineup home_lineup.json \
      --away-lineup away_lineup.json \
      --home-roster home_roster.json \
      --away-roster away_roster.json \
      --merged merged.json \
      [--scenarios '{"BLOWOUT_FAV":10,...}'] \
      [--betting-json '{"moneyline":{...},...}'] \
      [--log5-pct 45.0] [--pythag-pct 42.0] \
      -o analysis.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Legacy: 不再使用 master jsonl。prediction.json 從 --merged 同目錄讀取。

# ── Emoji → Enum mappings ───────────────────────────────────────────────────


def _strip_emoji(s: str) -> str:
    """Strip leading emoji + whitespace from label strings."""
    return re.sub(
        r"^[\U0001f300-\U0001f9ff\u2600-\u27bf⚪⚡📈📉🔄⚖🔥🔺🔻❄🔴🟠🟡🟢🔵⭐️]+\s*",
        "",
        s,
    ).strip()


_PITCHER_TIER = {
    "ace": "ACE", "strong ace": "STRONG_ACE", "solid starter": "SOLID_STARTER",
    "back end": "BACK_END", "below average": "BELOW_AVERAGE",
}
_AGE_PHASE = {
    "growth": "GROWTH", "成長期": "GROWTH",
    "prime": "PRIME", "巔峰期": "PRIME",
    "veteran": "VETERAN", "老將期": "VETERAN",
    "declining": "DECLINING", "衰退期": "DECLINING",
}
_LINEUP_TIER = {
    "elite": "ELITE", "strong": "STRONG", "above average": "STRONG",
    "average": "AVERAGE", "below average": "BELOW_AVERAGE", "weak": "WEAK",
}
_HEAT = {
    "on fire": "ON_FIRE", "hot": "HOT", "normal": "NORMAL",
    "cold": "COLD", "ice cold": "ICE_COLD",
}


def _map(table: dict, raw: str, default: str) -> str:
    s = _strip_emoji(raw)
    return table.get(s.lower(), table.get(s, default))


# ── Signal key → SignalCode ──────────────────────────────────────────────────

_SIGNAL_PATTERNS: list[tuple[str, str]] = [
    ("park_factor", "PARK_FACTOR"),
    ("bullpen.*3.*il|3.*core.*il|bullpen_il_3", "BULLPEN_IL_3PLUS"),
    ("bullpen.*2.*il|bullpen_il_2", "BULLPEN_IL_2PLUS"),
    ("bullpen.*heavy", "BULLPEN_HEAVY_USE"),
    ("k_pct.*high|both.*k.*high|high.*k_pct", "BOTH_K_PCT_HIGH"),
    ("lineup.*hot|both.*hot", "BOTH_LINEUP_HOT"),
    ("lineup.*cold|both.*cold", "BOTH_LINEUP_COLD"),
    ("sp_strong|both.*sp.*strong", "BOTH_SP_STRONG"),
    ("sp_solid|both.*sp.*solid", "BOTH_SP_SOLID_PLUS"),
    ("temp.*high|high.*temp", "TEMP_HIGH"),
    ("temp.*low|low.*temp", "TEMP_LOW"),
    ("wind.*out", "WIND_OUT"),
    ("wind.*in", "WIND_IN"),
    ("umpire.*over|ump.*over", "UMPIRE_OVER"),
    ("umpire.*under|ump.*under", "UMPIRE_UNDER"),
    ("doubleheader|dh_g2", "DOUBLEHEADER_G2"),
    ("platoon", "PLATOON_DISADVANTAGE"),
    ("sp_rest|rest.*adjust", "SP_REST_ADJUSTED"),
    ("bullpen.*il", "BULLPEN_IL_2PLUS"),
]


def _map_signal(key: str) -> str | None:
    k = key.lower()
    for pattern, code in _SIGNAL_PATTERNS:
        if re.search(pattern, k):
            return code
    return None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_prediction(merged_path: str) -> dict:
    """Load prediction.json from the same folder as merged.json."""
    game_dir = os.path.dirname(os.path.abspath(merged_path))
    pred_path = os.path.join(game_dir, "prediction.json")
    if not os.path.exists(pred_path):
        return {}
    with open(pred_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pythag(rs: float, ra: float, exp: float = 1.83) -> float:
    if rs + ra == 0:
        return 50.0
    return round(100 * (rs**exp) / (rs**exp + ra**exp), 1)


def _log5(hw: float, aw: float) -> float:
    if hw + aw == 0:
        return 50.0
    p = (hw * (1 - aw)) / (hw * (1 - aw) + aw * (1 - hw))
    return round(p * 100, 1)


def _injury_impact(il: list[dict]) -> str:
    if not il:
        return "NONE"
    n = len(il)
    has_60 = any("60" in (p.get("status") or "") for p in il)
    if n >= 3 or (n >= 2 and has_60):
        return "CRITICAL"
    if n >= 2 or has_60:
        return "SIGNIFICANT"
    return "MINOR"


def _classify_il(p: dict) -> dict:
    status = p.get("status", "")
    impact = "CRITICAL" if "60" in status else ("SIGNIFICANT" if "15" in status else "MINOR")
    return {"name": p["name"], "status": status, "position": p.get("position", ""), "impact": impact}


def _abbr_to_dir(abbr: str, home_abbr: str, away_abbr: str) -> str:
    """Convert team abbreviation to HOME/AWAY/PASS."""
    u = (abbr or "").upper()
    if u == "PASS" or not u:
        return "PASS"
    if u == home_abbr.upper():
        return "HOME"
    if u == away_abbr.upper():
        return "AWAY"
    # Might already be HOME/AWAY
    if u in ("HOME", "AWAY", "OVER", "UNDER"):
        return u
    return "PASS"


# ── Section builders ─────────────────────────────────────────────────────────


def build_meta(game: dict, pred: dict, h_lu: dict, a_lu: dict) -> dict:
    g = game.get("game", {})
    # Series game # = count consecutive recent games vs same opponent + 1
    recent = game.get("home_recent", {}).get("games", [])
    opp = g.get("away", {}).get("team", "")
    sg = 1
    for gm in recent:
        if gm.get("opponent") == opp:
            sg += 1
        else:
            break
    return {
        "game_pk": g.get("gamePk") or pred.get("game_pk", 0),
        "date": pred.get("game_time") or g.get("date", ""),
        "venue": g.get("venue") or pred.get("venue", ""),
        "home_team": h_lu.get("team", ""),
        "away_team": a_lu.get("team", ""),
        "home_sp": g.get("home", {}).get("probable_pitcher", ""),
        "away_sp": g.get("away", {}).get("probable_pitcher", ""),
        "series_game": sg if sg > 1 else None,
    }


def build_recent_form(game: dict) -> dict:
    def _tf(rec: dict) -> dict:
        games = rec.get("games", [])
        return {
            "record_10": rec.get("record", "0-0"),
            "streak": rec.get("streak", 0),
            "rs_per_game": round(rec.get("rs_per_game", 0), 1),
            "ra_per_game": round(rec.get("ra_per_game", 0), 1),
            "run_diff": rec.get("run_diff", 0),
            "last_5_results": [("W" if g.get("is_winner") else "L") for g in games[:5]],
        }

    hr = game.get("home_recent", {})
    ar = game.get("away_recent", {})
    # Series prev
    sp = None
    hg = hr.get("games", [])
    opp = game.get("game", {}).get("away", {}).get("team", "")
    if hg and hg[0].get("opponent") == opp:
        g0 = hg[0]
        if g0.get("is_home"):
            sp = {"home_score": g0["team_score"], "away_score": g0["opp_score"],
                  "winner": "home" if g0["is_winner"] else "away"}
        else:
            sp = {"home_score": g0["opp_score"], "away_score": g0["team_score"],
                  "winner": "away" if g0["is_winner"] else "home"}

    return {"home": _tf(hr), "away": _tf(ar), "series_prev": sp}


def build_pitcher(p: dict) -> dict:
    s = p.get("season", {})
    e = p.get("expected", {})
    sc = p.get("statcast", {})
    ps = p.get("platoon_splits", {})

    def _split(d: dict) -> dict:
        return {
            "avg": str(d.get("avg", ".000")), "obp": str(d.get("obp", ".000")),
            "slg": str(d.get("slg", ".000")),
            "k_pct": d.get("k_pct", 0), "bb_pct": d.get("bb_pct", 0), "bf": d.get("bf", 0),
        }

    return {
        "name": p.get("name", ""), "mlbam_id": p.get("mlbam_id", 0),
        "age": p.get("age", 0), "pitch_hand": p.get("pitch_hand", "R"),
        "tier": _map(_PITCHER_TIER, p.get("tier", ""), "BELOW_AVERAGE"),
        "age_phase": _map(_AGE_PHASE, p.get("age_assessment", ""), "PRIME"),
        "season": {k: s.get(k, 0) for k in
                   ("era", "fip", "xfip", "whip", "k_pct", "bb_pct", "k_bb_pct", "hr_per_9", "gb_pct", "ip", "gs")},
        "expected": {"xera": e.get("xera"), "xwoba": e.get("xwoba"), "xba": e.get("xba")},
        "statcast": {
            "avg_velo": sc.get("avg_velo"), "max_velo": sc.get("max_velo"),
            "hard_hit_pct": sc.get("hard_hit_pct"), "barrel_pct": sc.get("barrel_pct"),
            "whiff_pct": sc.get("whiff_pct"), "csw_pct": sc.get("csw_pct"),
            "ev95percent": sc.get("ev95percent"),
            "pitch_types": sc.get("pitch_types", {}),
        },
        "platoon_splits": {"vs_left": _split(ps.get("vs_left", {})), "vs_right": _split(ps.get("vs_right", {}))},
        "game_log": [
            {k: g.get(k) for k in ("date", "opponent", "ip", "era", "k", "bb", "h", "er", "pitches", "strikes")}
            for g in p.get("game_log", [])
        ],
        "prior_year": p.get("prior_year"),
    }


def build_pitching_matchup(hp: dict, ap: dict) -> dict:
    h_fip = hp.get("season", {}).get("fip", 99)
    a_fip = ap.get("season", {}).get("fip", 99)
    diff = h_fip - a_fip
    adv = "even" if abs(diff) < 0.5 else ("home" if diff < 0 else "away")
    return {"home_sp": build_pitcher(hp), "away_sp": build_pitcher(ap), "advantage": adv}


def build_hitter(h: dict) -> dict:
    return {
        "name": h.get("name", ""), "mlbam_id": h.get("mlbam_id", 0),
        "position": h.get("position", ""),
        "ops": h.get("ops", 0), "xwoba": h.get("xwoba"), "babip": h.get("babip", 0),
        "k_pct": h.get("k_pct", 0), "bb_pct": h.get("bb_pct", 0),
        "xba": h.get("xba"), "xslg": h.get("xslg"),
        "ev95pct": h.get("ev95pct"), "barrel_pct": h.get("barrel_pct"),
        "platoon": {
            "vs_lhp": h.get("platoon", {}).get("vs_lhp", {"avg": ".000", "obp": ".000", "slg": ".000", "ops": ".000", "pa": 0}),
            "vs_rhp": h.get("platoon", {}).get("vs_rhp", {"avg": ".000", "obp": ".000", "slg": ".000", "ops": ".000", "pa": 0}),
        },
        "last_7": h.get("last_7", {"avg": ".000", "obp": ".000", "slg": ".000", "ops": ".000", "babip": ".000", "pa": 0}),
        "bvp": h.get("bvp") if h.get("bvp") and h["bvp"].get("pa", 0) > 0 else None,
    }


def build_lineup(h_lu: dict, a_lu: dict) -> dict:
    def _ts(lu: dict) -> dict:
        return {
            "team": lu.get("team", ""),
            "tier": _map(_LINEUP_TIER, lu.get("tier", ""), "AVERAGE"),
            "recent_heat": _map(_HEAT, lu.get("recent_heat", ""), "NORMAL"),
            "avg_ops": lu.get("avg_ops", 0), "avg_xwoba": lu.get("avg_xwoba"),
            "avg_babip": lu.get("avg_babip", 0),
            "avg_k_pct": lu.get("avg_k_pct", 0), "avg_bb_pct": lu.get("avg_bb_pct", 0),
            "chain": lu.get("chain", {"obp_top3": None, "slg_mid": None}),
            "lineup": [build_hitter(h) for h in lu.get("lineup", [])],
        }

    h_ops = h_lu.get("avg_ops", 0)
    a_ops = a_lu.get("avg_ops", 0)
    diff = h_ops - a_ops
    adv = "even" if abs(diff) < 0.020 else ("home" if diff > 0 else "away")
    return {"home": _ts(h_lu), "away": _ts(a_lu), "advantage": adv}


def build_bullpen(h_ros: dict, a_ros: dict, merged: dict, h_abbr: str, a_abbr: str) -> dict:
    def _team(ros: dict, side: str, abbr: str) -> dict:
        il = ros.get("injured_list", [])
        il_p = [_classify_il(p) for p in il if "pitcher" in p.get("position", "").lower()]
        il_pos = [_classify_il(p) for p in il if "pitcher" not in p.get("position", "").lower()]
        return {
            "team": abbr,
            "bullpen_era": merged.get(f"{side}_bullpen_era"),
            "il_pitchers": il_p, "il_position_players": il_pos,
            "injury_impact_summary": _injury_impact(il),
        }

    return {"home": _team(h_ros, "home", h_abbr), "away": _team(a_ros, "away", a_abbr)}


def build_environment(game: dict, merged: dict, pred: dict) -> dict:
    venue = game.get("game", {}).get("venue", "")
    _RETRACTABLE = {"Chase Field", "T-Mobile Park", "Rogers Centre", "Globe Life Field",
                    "loanDepot park", "Minute Maid Park", "American Family Field"}
    _CLOSED = {"Tropicana Field"}
    roof = "retractable" if venue in _RETRACTABLE else ("closed" if venue in _CLOSED else None)
    return {
        "venue": venue,
        "park_factor": merged.get("park_factor", 100),
        "temperature_f": pred.get("temperature_f"),
        "wind_mph": pred.get("wind_mph"),
        "wind_direction": pred.get("wind_direction"),
        "roof": roof,
    }


def build_signals(pred: dict) -> dict:
    raw = pred.get("signal_adjustments", {})
    signals, total, seen = [], 0.0, set()
    for key, val in raw.items():
        code = _map_signal(key)
        if code and code not in seen:
            signals.append({"code": code, "run_value": round(val, 1)})
            total += val
            seen.add(code)
        elif code is None:
            print(f"WARNING: unmapped signal '{key}', skipping", file=sys.stderr)
    return {"signals": signals, "total_run_adjustment": round(total, 1)}


def build_win_prob(pred: dict, game: dict, log5_pct: float | None, pythag_pct: float | None) -> dict:
    xgb = pred.get("predicted_home_pct", 50)
    if pythag_pct is None:
        pythag_pct = _pythag(
            game.get("home_recent", {}).get("rs_per_game", 4),
            game.get("home_recent", {}).get("ra_per_game", 4),
        )
    if log5_pct is None:
        hr = game.get("home_recent", {})
        ar = game.get("away_recent", {})
        hw = hr.get("wins", 5) / max(hr.get("wins", 5) + hr.get("losses", 5), 1)
        aw = ar.get("wins", 5) / max(ar.get("wins", 5) + ar.get("losses", 5), 1)
        log5_pct = _log5(hw, aw)
    cv = pred.get("cross_validation", "INSUFFICIENT_SAMPLE").upper().replace(" ", "_")
    if cv not in ("CONSISTENT", "DIVERGENT", "INSUFFICIENT_SAMPLE"):
        cv = "INSUFFICIENT_SAMPLE"
    conf = pred.get("confidence", "LOW").upper()
    if conf not in ("HIGH", "MEDIUM", "LOW"):
        conf = "LOW"
    return {
        "xgboost_home_pct": round(xgb, 1), "log5_home_pct": round(log5_pct, 1),
        "pythag_home_pct": round(pythag_pct, 1), "cross_validation": cv, "confidence": conf,
    }


def build_score(pred: dict, scenarios_arg: str | None) -> dict:
    # 優先使用 Phase 3 修正後比分，其次公式原始值
    home = pred.get("predicted_home_score", pred.get("formula_home_score", 4))
    away = pred.get("predicted_away_score", pred.get("formula_away_score", 4))
    total = round(home + away, 1)

    def _ex(st: str) -> str:
        f, d = max(home, away), min(home, away)
        m = {"BLOWOUT_FAV": (f + 3, max(0, d - 1)), "COMFORTABLE_FAV": (f + 1, d),
             "CLOSE_FAV": (f, max(1, d - 1)), "CLOSE_DOG": (d, f - 1),
             "COMFORTABLE_DOG": (d - 1, f + 1), "BLOWOUT_DOG": (max(0, d - 2), f + 3)}
        hi, lo = m.get(st, (4, 3))
        hi, lo = round(hi), round(lo)
        if hi == lo:
            hi += 1
        return f"{lo}-{hi}" if home >= away else f"{hi}-{lo}"

    types = ["BLOWOUT_FAV", "COMFORTABLE_FAV", "CLOSE_FAV", "CLOSE_DOG", "COMFORTABLE_DOG", "BLOWOUT_DOG"]
    if scenarios_arg:
        raw = json.loads(scenarios_arg) if isinstance(scenarios_arg, str) else scenarios_arg
        scenarios = [{"type": t, "pct": raw.get(t, 0), "example_score": _ex(t)} for t in types]
    else:
        diff = abs(home - away)
        dist = ([15, 25, 20, 18, 14, 8] if diff >= 3
                else [8, 20, 25, 22, 16, 9] if diff >= 1.5
                else [6, 15, 24, 24, 18, 13])
        scenarios = [{"type": t, "pct": dist[i], "example_score": _ex(t)} for i, t in enumerate(types)]

    return {
        "home_score": round(home), "away_score": round(away), "total": total,
        "home_range": [max(0, round(home - 2)), round(home + 2)],
        "away_range": [max(0, round(away - 2)), round(away + 2)],
        "scenarios": scenarios,
    }


def build_betting(pred: dict, h_abbr: str, a_abbr: str, betting_arg: str | None) -> dict:
    if betting_arg:
        return json.loads(betting_arg) if isinstance(betting_arg, str) else betting_arg

    home_pct = pred.get("predicted_home_pct", 50)

    def _mk(dir_raw: str, stars_raw, line_val, default_reasons: list[str]) -> dict:
        direction = _abbr_to_dir(str(dir_raw), h_abbr, a_abbr)
        stars = int(stars_raw or 0)
        if direction == "PASS" or stars == 0:
            return {"direction": "PASS", "stars": 0, "line": line_val,
                    "model_pct": 0, "implied_pct": 0, "edge": 0, "reasons": [], "risk": "HIGH"}
        mp = home_pct if direction == "HOME" else (100 - home_pct)
        ip = round(mp * 0.88, 1)
        return {
            "direction": direction, "stars": stars, "line": line_val,
            "model_pct": round(mp, 1), "implied_pct": ip, "edge": round(mp - ip, 1),
            "reasons": default_reasons,
            "risk": "LOW" if stars >= 4 else ("MEDIUM" if stars >= 2 else "HIGH"),
        }

    ml = _mk(pred.get("ml_rec", "PASS"), pred.get("ml_stars"), None, ["CONSISTENT_MODELS"])
    ou_dir = (pred.get("ou_rec") or "PASS").upper()
    ou = _mk(ou_dir, pred.get("ou_stars"), pred.get("ou_line"),
             ["HIGH_TOTAL_OFFENSE"] if ou_dir == "OVER" else ["LOW_TOTAL_PITCHING"])
    rl = _mk(pred.get("run_line_rec", "PASS"), pred.get("run_line_stars"),
             pred.get("run_line", -1.5), ["PITCHER_MISMATCH"])

    return {"moneyline": ml, "run_line": rl, "over_under": ou}


# ── Main ─────────────────────────────────────────────────────────────────────


def assemble(args) -> dict:
    game = _load(args.game)
    hp = _load(args.home_pitcher)
    ap = _load(args.away_pitcher)
    h_lu = _load(args.home_lineup)
    a_lu = _load(args.away_lineup)
    h_ros = _load(args.home_roster)
    a_ros = _load(args.away_roster)
    merged = _load(args.merged)
    pred = _load_prediction(args.merged)

    h_abbr = h_lu.get("team", "")
    a_abbr = a_lu.get("team", "")

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": build_meta(game, pred, h_lu, a_lu),
        "recent_form": build_recent_form(game),
        "pitching_matchup": build_pitching_matchup(hp, ap),
        "lineup_analysis": build_lineup(h_lu, a_lu),
        "bullpen_and_injuries": build_bullpen(h_ros, a_ros, merged, h_abbr, a_abbr),
        "environment": build_environment(game, merged, pred),
        "signal_adjustments": build_signals(pred),
        "win_probability": build_win_prob(pred, game, args.log5_pct, args.pythag_pct),
        "score_prediction": build_score(pred, args.scenarios),
        "betting_recommendations": build_betting(pred, h_abbr, a_abbr, args.betting_json),
    }


def validate(analysis: dict) -> list[str]:
    """Light validation — returns list of issues (empty = OK)."""
    issues = []
    if analysis.get("schema_version") != "1.0":
        issues.append("schema_version must be '1.0'")
    required = [
        "meta", "recent_form", "pitching_matchup", "lineup_analysis",
        "bullpen_and_injuries", "environment", "signal_adjustments",
        "win_probability", "score_prediction", "betting_recommendations",
    ]
    for k in required:
        if k not in analysis:
            issues.append(f"missing top-level key: {k}")
    bet = analysis.get("betting_recommendations", {})
    for mkt in ("moneyline", "run_line", "over_under"):
        if mkt not in bet:
            issues.append(f"missing betting_recommendations.{mkt}")
    return issues


def main():
    p = argparse.ArgumentParser(description="Assemble GameAnalysis JSON v1.0")
    p.add_argument("--game", required=True, help="game_data.json path")
    p.add_argument("--home-pitcher", required=True, help="home_pitcher.json path")
    p.add_argument("--away-pitcher", required=True, help="away_pitcher.json path")
    p.add_argument("--home-lineup", required=True, help="home_lineup.json path")
    p.add_argument("--away-lineup", required=True, help="away_lineup.json path")
    p.add_argument("--home-roster", required=True, help="home_roster.json path")
    p.add_argument("--away-roster", required=True, help="away_roster.json path")
    p.add_argument("--merged", required=True, help="merged.json path")
    p.add_argument("--scenarios", default=None,
                   help='JSON: {"BLOWOUT_FAV":10,"COMFORTABLE_FAV":20,...}')
    p.add_argument("--betting-json", default=None,
                   help="Full BettingRecommendations JSON (overrides auto-generation)")
    p.add_argument("--log5-pct", type=float, default=None, help="Override Log5 home win %%")
    p.add_argument("--pythag-pct", type=float, default=None, help="Override Pythagorean home win %%")
    p.add_argument("-o", "--output", required=True, help="Output analysis.json path")
    p.add_argument("--validate", action="store_true", help="Validate output and exit")
    args = p.parse_args()

    analysis = assemble(args)

    issues = validate(analysis)
    if issues:
        for iss in issues:
            print(f"VALIDATION: {iss}", file=sys.stderr)
        if args.validate:
            sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    if args.validate:
        print(f"✅ analysis.json validated OK ({len(json.dumps(analysis))} bytes)")
    else:
        print(f"✅ {args.output} generated ({len(json.dumps(analysis))} bytes)")


if __name__ == "__main__":
    main()
