"""CLV(closing-line value)量尺:模型 pick 那一側的 no-vig 機率,從最早到最晚 pre-commence
快照移動了多少(pp)。讀 odds_snapshots 直接算,對線上模型唯讀。"""
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.closing_line import (
    _parse_iso_utc, find_entry_close_snapshots,
    extract_pinnacle_no_vig, extract_pinnacle_rl_no_vig,
)


def _finite_nonzero(x) -> bool:
    """有限且非零的數值才算一個 edge。nan(pandas 缺值)/None/0 → False。"""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x != 0


def pick_sides(row: dict) -> dict:
    """模型在每個市場的 pick 側(正 edge 那邊)。edge=0/缺/nan → None。"""
    rl = None
    e = row.get("home_rl_pp")
    if _finite_nonzero(e):
        rl = "home" if e > 0 else "away"
    ou = None
    o = row.get("over_pp")
    if _finite_nonzero(o):
        ou = "over" if o > 0 else "under"
    return {"rl": rl, "ou": ou}


def no_vig_for(game: dict, market: str, side: str):
    """該 snapshot game 裡,某市場某側的 no-vig 機率(0-1)。缺 → None。"""
    if market == "rl":
        d = extract_pinnacle_rl_no_vig(game)
        if d is None:
            return None
        return d["home_no_vig"] if side == "home" else d["away_no_vig"]
    d = extract_pinnacle_no_vig(game)
    if d is None:
        return None
    return d["over_no_vig"] if side == "over" else d["under_no_vig"]


def point_for(game: dict, market: str):
    """市場的線:rl → home_point;ou → total_line。缺 → None。"""
    if market == "rl":
        d = extract_pinnacle_rl_no_vig(game)
        return d["home_point"] if d else None
    d = extract_pinnacle_no_vig(game)
    return d["total_line"] if d else None


def clv_pp(entry_game: dict, close_game: dict, market: str, side: str):
    """(close − entry) no-vig × 100,pick 方向。任一缺 → None。正值=線往我們方向跑。"""
    e = no_vig_for(entry_game, market, side)
    c = no_vig_for(close_game, market, side)
    if e is None or c is None:
        return None
    return round((c - e) * 100, 2)


def _et_hour(snapshot_time_et):
    """'2026-05-02 12:00 ET' → 12。解析失敗 → None。"""
    try:
        return int(snapshot_time_et.split()[1].split(":")[0])
    except (AttributeError, IndexError, ValueError):
        return None


def compute_clv_rows(rows, snapshots_dir) -> list:
    """每場一列:找 entry/close 快照,算 pick 方向的 CLV。rows = df.to_dict('records') 或測試 dict。"""
    out = []
    for r in rows:
        picks = pick_sides(r)
        rec = {"date": r.get("date"), "matchup": r.get("matchup"),
               "has_headroom": False, "entry_hour": None, "minutes_gap": None,
               "rl_pick": picks["rl"], "rl_clv": None, "rl_edge_pp": r.get("home_rl_pp"), "rl_point_stable": None,
               "ou_pick": picks["ou"], "ou_clv": None, "ou_edge_pp": r.get("over_pp"), "ou_point_stable": None}
        entry, close = find_entry_close_snapshots(snapshots_dir, r.get("date"), r.get("home_team"), r.get("away_team"))
        if entry is None or close is None:
            out.append(rec)
            continue
        e_ts = _parse_iso_utc(entry.get("snapshot_time_utc", ""))
        c_ts = _parse_iso_utc(close.get("snapshot_time_utc", ""))
        if e_ts is None or c_ts is None or not (e_ts < c_ts):
            out.append(rec)   # single snapshot / unparseable → no headroom
            continue
        rec["has_headroom"] = True
        rec["entry_hour"] = _et_hour(entry.get("snapshot_time_et", ""))
        rec["minutes_gap"] = round((c_ts - e_ts).total_seconds() / 60, 1)
        if picks["rl"]:
            rec["rl_clv"] = clv_pp(entry, close, "rl", picks["rl"])
            pe, pc = point_for(entry, "rl"), point_for(close, "rl")
            rec["rl_point_stable"] = (pe == pc) if (pe is not None and pc is not None) else None
        if picks["ou"]:
            rec["ou_clv"] = clv_pp(entry, close, "ou", picks["ou"])
            pe, pc = point_for(entry, "ou"), point_for(close, "ou")
            rec["ou_point_stable"] = (pe == pc) if (pe is not None and pc is not None) else None
        out.append(rec)
    return out


def _stats(vals: list) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "share_pos": None}
    n = len(vals); s = sorted(vals)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"n": n, "mean": round(sum(vals) / n, 3), "median": round(median, 3),
            "share_pos": round(sum(1 for v in vals if v > 0) / n, 3)}


def _corr(xs: list, ys: list):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    n = len(pairs)
    ax = sum(p[0] for p in pairs) / n; ay = sum(p[1] for p in pairs) / n
    sxy = sum((p[0] - ax) * (p[1] - ay) for p in pairs)
    sxx = sum((p[0] - ax) ** 2 for p in pairs); syy = sum((p[1] - ay) ** 2 for p in pairs)
    if sxx == 0 or syy == 0:
        return None
    return round(sxy / (sxx ** 0.5 * syy ** 0.5), 3)


def aggregate_clv(rows: list) -> dict:
    """headroom 子集上彙總 RL / O-U 的 CLV(mean/median/share>0/corr)、entry-hour 表、診斷。"""
    hr = [r for r in rows if r["has_headroom"]]
    rl = [r for r in hr if r["rl_clv"] is not None]
    ou = [r for r in hr if r["ou_clv"] is not None]

    by_hour = {}
    for r in hr:
        for c in (r["rl_clv"], r["ou_clv"]):
            if c is not None:
                by_hour.setdefault(r["entry_hour"], []).append(c)
    hour_table = {h: {"n": len(v), "mean": round(sum(v) / len(v), 3)}
                  for h, v in sorted(by_hour.items(), key=lambda x: (x[0] is None, x[0]))}

    gaps = sorted(r["minutes_gap"] for r in hr if r["minutes_gap"] is not None)
    med_gap = gaps[len(gaps) // 2] if gaps else None

    def _pct_stable(subset, key):
        vals = [r[key] for r in subset if r[key] is not None]
        return round(sum(1 for v in vals if v) / len(vals), 3) if vals else None

    return {
        "n_total": len(rows), "n_headroom": len(hr), "median_minutes_gap": med_gap,
        "rl": {**_stats([r["rl_clv"] for r in rl]),
               "corr_edge": _corr([abs(r["rl_edge_pp"]) if r["rl_edge_pp"] is not None else None for r in rl],
                                  [r["rl_clv"] for r in rl]),
               "pct_point_stable": _pct_stable(rl, "rl_point_stable")},
        "ou": {**_stats([r["ou_clv"] for r in ou]),
               "corr_edge": _corr([abs(r["ou_edge_pp"]) if r["ou_edge_pp"] is not None else None for r in ou],
                                  [r["ou_clv"] for r in ou]),
               "pct_point_stable": _pct_stable(ou, "ou_point_stable")},
        "entry_hour_table": hour_table,
    }
