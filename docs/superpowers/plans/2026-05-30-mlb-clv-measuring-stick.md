# MLB CLV Measuring Stick Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only CLV (closing-line value) measuring stick that, for each backfilled game, reports whether Pinnacle's no-vig line for the model's pick-side moved toward us between the earliest and latest pre-commence snapshot.

**Architecture:** Read the existing `odds/odds_snapshots/` files directly (not the snapshot `features.json` froze). For each game: entry = earliest pre-commence snapshot, close = latest; CLV_pp = signed no-vig move of the pick side (close − entry); aggregate on the headroom subset (entry strictly earlier than close), stratified by entry hour, with honest diagnostics. Wired into the existing backtest report as an additive section. No model/config/live-flow changes, no new fetching.

**Tech Stack:** Python 3, `pytest`, `pandas` (existing), stdlib only for new logic. Spec: `docs/superpowers/specs/2026-05-30-mlb-clv-measuring-stick-design.md`. Builds on `scripts/lib/closing_line.py`.

**Commit policy:** Repo owner commits manually. Treat `Commit` steps as checkpoints — run only if the owner asks; otherwise leave staged.

---

## File Structure

- **`scripts/lib/closing_line.py`** — ADD `find_entry_close_snapshots(...)` (earliest & latest pre-commence snapshot for a matchup). Existing `find_closing_snapshot_for_game` / `extract_*` untouched.
- **`scripts/lib/clv.py`** (new) — pure core (`pick_sides`, `no_vig_for`, `point_for`, `clv_pp`, `_et_hour`) + driving fns (`compute_clv_rows`, `aggregate_clv`).
- **`scripts/lib/load.py`** — add `home_team`, `away_team` (full names) to the row dict.
- **`scripts/backtest.py`** — compute CLV, pass to render. `SNAPSHOTS_DIR = SKILL_ROOT/"odds"/"odds_snapshots"`.
- **`scripts/lib/render.py`** — `render_report` gains a `clv` kwarg + a "CLV(領先指標)" section.
- **Tests** — new `scripts/tests/test_clv.py`; extend `scripts/tests/test_closing_line.py`, `scripts/tests/test_backtest_load_v2.py`.

**Snapshot game dict shape** (real, verified): `{home_team, away_team, game_date_et, commence_utc, bookmakers.pinnacle.{ml,ou,rl}}`; top-level snapshot has `snapshot_time_utc`, `snapshot_time_et`. `extract_pinnacle_rl_no_vig(game) -> {home_point, home_no_vig, away_point, away_no_vig}`; `extract_pinnacle_no_vig(game) -> {total_line, over_no_vig, under_no_vig, ...}`.

**clv row dict** (produced by `compute_clv_rows`, consumed by `aggregate_clv`): `{date, matchup, has_headroom, entry_hour, minutes_gap, rl_pick, rl_clv, rl_edge_pp, rl_point_stable, ou_pick, ou_clv, ou_edge_pp, ou_point_stable}`.

---

## Task 1: `find_entry_close_snapshots`

**Files:**
- Modify: `scripts/lib/closing_line.py`
- Test: `scripts/tests/test_closing_line.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts/tests/test_closing_line.py`:

```python
def _snap(snap_utc, snap_et, commence_utc, date_et="2026-05-02",
          home="New York Yankees", away="Baltimore Orioles", over_nv=51.0):
    return {
        "snapshot_time_utc": snap_utc, "snapshot_time_et": snap_et,
        "games": [{
            "away_team": away, "home_team": home, "game_date_et": date_et,
            "commence_utc": commence_utc,
            "bookmakers": {"pinnacle": {
                "ml": {away: {"no_vig_pct": 39.2}, home: {"no_vig_pct": 60.8}},
                "ou": {"Over": {"point": 8.5, "no_vig_pct": over_nv},
                       "Under": {"point": 8.5, "no_vig_pct": 100 - over_nv}},
                "rl": {home: {"point": -1.5, "no_vig_pct": 40.0},
                       away: {"point": 1.5, "no_vig_pct": 60.0}},
            }},
        }],
    }


def test_find_entry_close_picks_earliest_and_latest(tmp_path):
    from lib.closing_line import find_entry_close_snapshots
    import json
    # three pre-commence snapshots (commence 22:00 UTC)
    (tmp_path / "2026-05-02_12-00-ET.json").write_text(json.dumps(
        _snap("2026-05-02T16:00:00Z", "2026-05-02 12:00 ET", "2026-05-02T22:00:00Z", over_nv=50.0)), encoding="utf-8")
    (tmp_path / "2026-05-02_15-00-ET.json").write_text(json.dumps(
        _snap("2026-05-02T19:00:00Z", "2026-05-02 15:00 ET", "2026-05-02T22:00:00Z", over_nv=53.0)), encoding="utf-8")
    (tmp_path / "2026-05-02_18-00-ET.json").write_text(json.dumps(
        _snap("2026-05-02T21:00:00Z", "2026-05-02 18:00 ET", "2026-05-02T22:00:00Z", over_nv=55.0)), encoding="utf-8")
    entry, close = find_entry_close_snapshots(tmp_path, "2026-05-02", "New York Yankees", "Baltimore Orioles")
    assert entry["snapshot_time_utc"] == "2026-05-02T16:00:00Z"   # earliest
    assert close["snapshot_time_utc"] == "2026-05-02T21:00:00Z"   # latest


def test_find_entry_close_excludes_post_commence(tmp_path):
    from lib.closing_line import find_entry_close_snapshots
    import json
    # one valid pre-commence + one AFTER commence (22:00 ET = next-day UTC, snap >= commence)
    (tmp_path / "2026-05-02_12-00-ET.json").write_text(json.dumps(
        _snap("2026-05-02T16:00:00Z", "2026-05-02 12:00 ET", "2026-05-02T22:00:00Z")), encoding="utf-8")
    (tmp_path / "2026-05-02_22-00-ET.json").write_text(json.dumps(
        _snap("2026-05-03T02:00:00Z", "2026-05-02 22:00 ET", "2026-05-02T22:00:00Z")), encoding="utf-8")
    entry, close = find_entry_close_snapshots(tmp_path, "2026-05-02", "New York Yankees", "Baltimore Orioles")
    # post-commence excluded → only one qualifies → entry == close (same snapshot)
    assert entry["snapshot_time_utc"] == "2026-05-02T16:00:00Z"
    assert close["snapshot_time_utc"] == "2026-05-02T16:00:00Z"


def test_find_entry_close_none_when_no_match(tmp_path):
    from lib.closing_line import find_entry_close_snapshots
    assert find_entry_close_snapshots(tmp_path, "2026-05-02", "X", "Y") == (None, None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/tests/test_closing_line.py -k entry_close -q`
Expected: FAIL with `ImportError: cannot import name 'find_entry_close_snapshots'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/lib/closing_line.py`, add at end:

```python
def find_entry_close_snapshots(snapshots_dir, date, home_team, away_team):
    """Earliest & latest pre-commence Pinnacle snapshot for this matchup on the ET date.

    Scans {date}_*-ET.json, keeps games with matching teams, game_date_et==date, and
    snapshot_time < commence (strict — excludes the 22:00-ET post-commence trap).
    Returns (earliest_game, latest_game) by snapshot_time, each with snapshot_time_utc/et
    attached; (None, None) if none qualify. The two may be the same dict when only one qualifies.
    """
    snapshots_dir = Path(snapshots_dir)
    cands = []  # (snap_ts, game_copy)
    for f in sorted(snapshots_dir.glob(f"{date}_*-ET.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        snap_ts = _parse_iso_utc(data.get("snapshot_time_utc", ""))
        if snap_ts is None:
            continue
        for g in data.get("games", []):
            if g.get("game_date_et") != date:
                continue
            if g.get("home_team") != home_team or g.get("away_team") != away_team:
                continue
            commence = _parse_iso_utc(g.get("commence_utc", ""))
            if commence is None or snap_ts >= commence:
                continue
            gc = dict(g)
            gc["snapshot_time_utc"] = data.get("snapshot_time_utc", "")
            gc["snapshot_time_et"] = data.get("snapshot_time_et", "")
            cands.append((snap_ts, gc))
    if not cands:
        return None, None
    cands.sort(key=lambda x: x[0])
    return cands[0][1], cands[-1][1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_closing_line.py -q`
Expected: PASS (all — existing closing-line tests unaffected).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/lib/closing_line.py scripts/tests/test_closing_line.py
git commit -m "feat(closing_line): find_entry_close_snapshots (earliest & latest pre-commence)"
```

---

## Task 2: `clv.py` pure functions

**Files:**
- Create: `scripts/lib/clv.py`
- Test: `scripts/tests/test_clv.py`

- [ ] **Step 1: Write the failing tests**

Create `scripts/tests/test_clv.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import clv


def _game(rl_home_nv=40.0, rl_away_nv=60.0, over_nv=50.0, total=8.5, rl_home_pt=-1.5):
    return {"home_team": "H", "away_team": "A",
            "bookmakers": {"pinnacle": {
                "ml": {"A": {"no_vig_pct": 39.0}, "H": {"no_vig_pct": 61.0}},
                "ou": {"Over": {"point": total, "no_vig_pct": over_nv},
                       "Under": {"point": total, "no_vig_pct": 100 - over_nv}},
                "rl": {"H": {"point": rl_home_pt, "no_vig_pct": rl_home_nv},
                       "A": {"point": -rl_home_pt, "no_vig_pct": rl_away_nv}},
            }}}


def test_pick_sides_from_edges():
    assert clv.pick_sides({"home_rl_pp": 2.2, "over_pp": -5.4}) == {"rl": "home", "ou": "under"}
    assert clv.pick_sides({"home_rl_pp": -1.0, "over_pp": 3.0}) == {"rl": "away", "ou": "over"}
    assert clv.pick_sides({"home_rl_pp": 0, "over_pp": None}) == {"rl": None, "ou": None}


def test_no_vig_for():
    g = _game(rl_home_nv=40.0, rl_away_nv=60.0, over_nv=52.0)
    assert abs(clv.no_vig_for(g, "rl", "home") - 0.40) < 1e-9
    assert abs(clv.no_vig_for(g, "rl", "away") - 0.60) < 1e-9
    assert abs(clv.no_vig_for(g, "ou", "over") - 0.52) < 1e-9
    assert abs(clv.no_vig_for(g, "ou", "under") - 0.48) < 1e-9
    assert clv.no_vig_for({"bookmakers": {}}, "rl", "home") is None


def test_point_for():
    g = _game(total=9.0, rl_home_pt=-1.5)
    assert clv.point_for(g, "ou") == 9.0
    assert clv.point_for(g, "rl") == -1.5


def test_clv_pp_sign_and_magnitude():
    entry = _game(over_nv=50.0)
    close = _game(over_nv=55.0)   # over no-vig rose 5pp → market moved toward "over"
    assert clv.clv_pp(entry, close, "ou", "over") == 5.0
    assert clv.clv_pp(entry, close, "ou", "under") == -5.0
    # missing side → None
    assert clv.clv_pp({"bookmakers": {}}, close, "ou", "over") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/tests/test_clv.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'lib.clv'` (or `ImportError`).

- [ ] **Step 3: Write minimal implementation**

Create `scripts/lib/clv.py`:

```python
"""CLV(closing-line value)量尺:模型 pick 那一側的 no-vig 機率,從最早到最晚 pre-commence
快照移動了多少(pp)。讀 odds_snapshots 直接算,對線上模型唯讀。"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.closing_line import (
    _parse_iso_utc, find_entry_close_snapshots,
    extract_pinnacle_no_vig, extract_pinnacle_rl_no_vig,
)


def pick_sides(row: dict) -> dict:
    """模型在每個市場的 pick 側(正 edge 那邊)。edge=0/缺 → None。"""
    rl = None
    e = row.get("home_rl_pp")
    if isinstance(e, (int, float)) and e != 0:
        rl = "home" if e > 0 else "away"
    ou = None
    o = row.get("over_pp")
    if isinstance(o, (int, float)) and o != 0:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_clv.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/lib/clv.py scripts/tests/test_clv.py
git commit -m "feat(clv): pure CLV helpers (pick_sides, no_vig_for, clv_pp)"
```

---

## Task 3: `compute_clv_rows` + `aggregate_clv`

**Files:**
- Modify: `scripts/lib/clv.py`
- Test: `scripts/tests/test_clv.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts/tests/test_clv.py`:

```python
import json


def _write_snaps(tmp_path, slots):
    """slots = [(filename_slot, snap_utc, snap_et, over_nv, rl_home_nv)]; commence fixed 22:00Z."""
    for slot, snap_utc, snap_et, over_nv, rl_home_nv in slots:
        data = {"snapshot_time_utc": snap_utc, "snapshot_time_et": snap_et,
                "games": [{"home_team": "H", "away_team": "A", "game_date_et": "2026-05-02",
                           "commence_utc": "2026-05-02T22:00:00Z",
                           "bookmakers": {"pinnacle": {
                               "ml": {"A": {"no_vig_pct": 39.0}, "H": {"no_vig_pct": 61.0}},
                               "ou": {"Over": {"point": 8.5, "no_vig_pct": over_nv},
                                      "Under": {"point": 8.5, "no_vig_pct": 100 - over_nv}},
                               "rl": {"H": {"point": -1.5, "no_vig_pct": rl_home_nv},
                                      "A": {"point": 1.5, "no_vig_pct": 100 - rl_home_nv}},
                           }}}]}
        (tmp_path / f"2026-05-02_{slot}.json").write_text(json.dumps(data), encoding="utf-8")


def _row(**kw):
    base = dict(date="2026-05-02", matchup="A@H", home_team="H", away_team="A",
                home_rl_pp=2.0, over_pp=3.0)
    base.update(kw)
    return base


def test_compute_clv_rows_headroom_and_clv(tmp_path):
    _write_snaps(tmp_path, [
        ("12-00-ET", "2026-05-02T16:00:00Z", "2026-05-02 12:00 ET", 50.0, 40.0),
        ("18-00-ET", "2026-05-02T21:00:00Z", "2026-05-02 18:00 ET", 55.0, 43.0),
    ])
    rows = clv.compute_clv_rows([_row()], tmp_path)
    r = rows[0]
    assert r["has_headroom"] is True
    assert r["entry_hour"] == 12
    assert r["ou_pick"] == "over" and r["ou_clv"] == 5.0      # 55-50
    assert r["rl_pick"] == "home" and r["rl_clv"] == 3.0      # 43-40
    assert r["rl_point_stable"] is True and r["ou_point_stable"] is True


def test_compute_clv_rows_no_headroom_single_snapshot(tmp_path):
    _write_snaps(tmp_path, [
        ("12-00-ET", "2026-05-02T16:00:00Z", "2026-05-02 12:00 ET", 50.0, 40.0),
    ])
    rows = clv.compute_clv_rows([_row()], tmp_path)
    assert rows[0]["has_headroom"] is False
    assert rows[0]["ou_clv"] is None


def test_aggregate_clv_stats_and_share(tmp_path):
    # 3 headroom games: ou_clv +5, +1, -2  → mean ~1.33, share>0 = 2/3
    rows = [
        {"has_headroom": True, "entry_hour": 12, "minutes_gap": 300,
         "rl_pick": "home", "rl_clv": 2.0, "rl_edge_pp": 2.0, "rl_point_stable": True,
         "ou_pick": "over", "ou_clv": 5.0, "ou_edge_pp": 4.0, "ou_point_stable": True},
        {"has_headroom": True, "entry_hour": 12, "minutes_gap": 300,
         "rl_pick": "home", "rl_clv": -1.0, "rl_edge_pp": 1.0, "rl_point_stable": True,
         "ou_pick": "over", "ou_clv": 1.0, "ou_edge_pp": 2.0, "ou_point_stable": False},
        {"has_headroom": True, "entry_hour": 15, "minutes_gap": 200,
         "rl_pick": "away", "rl_clv": 0.5, "rl_edge_pp": -3.0, "rl_point_stable": True,
         "ou_pick": "under", "ou_clv": -2.0, "ou_edge_pp": -1.0, "ou_point_stable": True},
        {"has_headroom": False, "entry_hour": None, "minutes_gap": None,
         "rl_pick": None, "rl_clv": None, "rl_edge_pp": None, "rl_point_stable": None,
         "ou_pick": None, "ou_clv": None, "ou_edge_pp": None, "ou_point_stable": None},
    ]
    agg = clv.aggregate_clv(rows)
    assert agg["n_total"] == 4 and agg["n_headroom"] == 3
    assert agg["ou"]["n"] == 3
    assert abs(agg["ou"]["mean"] - (5 + 1 - 2) / 3) < 1e-3
    assert abs(agg["ou"]["share_pos"] - 2 / 3) < 1e-3
    assert agg["ou"]["pct_point_stable"] == round(2 / 3, 3)   # 2 of 3 stable
    assert 12 in agg["entry_hour_table"] and agg["entry_hour_table"][12]["n"] == 4  # 2 rl + 2 ou clv at hr 12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/tests/test_clv.py -k "compute_clv_rows or aggregate" -q`
Expected: FAIL with `AttributeError: module 'lib.clv' has no attribute 'compute_clv_rows'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/lib/clv.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_clv.py -q`
Expected: PASS (all clv tests).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/lib/clv.py scripts/tests/test_clv.py
git commit -m "feat(clv): compute_clv_rows + aggregate_clv (headroom subset, diagnostics)"
```

---

## Task 4: Add team names to load rows

**Files:**
- Modify: `scripts/lib/load.py` (the `return { ... }` dict in `_build_row`)
- Test: `scripts/tests/test_backtest_load_v2.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_backtest_load_v2.py` (uses the file's existing imports/helpers; this test builds its own minimal v2 dir):

```python
def test_load_includes_team_names(tmp_path, monkeypatch):
    import json
    from lib import load as load_mod
    d = tmp_path / "2026-05-02" / "BAL@NYY"
    d.mkdir(parents=True)
    feats = {"schema_version": 2,
             "game": {"game_pk": 1, "date": "2026-05-02",
                      "home": "New York Yankees", "away": "Baltimore Orioles"},
             "model": {}, "odds": None, "edges": {}}
    (d / "features.json").write_text(json.dumps(feats), encoding="utf-8")
    monkeypatch.setattr(load_mod, "ANALYSIS_DATA_DIR", tmp_path)
    df = load_mod.build_dataframe_for_month("2026-05")
    assert df.iloc[0]["home_team"] == "New York Yankees"
    assert df.iloc[0]["away_team"] == "Baltimore Orioles"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_backtest_load_v2.py -k team_names -q`
Expected: FAIL with `KeyError: 'home_team'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/lib/load.py`, in `_build_row`'s returned dict, add (right after the `"game_pk"` line):

```python
        "game_pk": feats.get("game", {}).get("game_pk"),
        "home_team": feats.get("game", {}).get("home"),
        "away_team": feats.get("game", {}).get("away"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/tests/test_backtest_load_v2.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit** (checkpoint)

```bash
git add scripts/lib/load.py scripts/tests/test_backtest_load_v2.py
git commit -m "feat(load): expose home_team/away_team for CLV snapshot matching"
```

---

## Task 5: Wire CLV into the backtest report

**Files:**
- Modify: `scripts/lib/render.py` (`render_report` signature + new section)
- Modify: `scripts/backtest.py` (compute CLV, pass to render)
- Test: `scripts/tests/test_clv.py` (render section smoke test)

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_clv.py`:

```python
def test_render_clv_section_present():
    import pandas as pd
    from lib import render
    agg = {"n_total": 292, "n_headroom": 120, "median_minutes_gap": 180.0,
           "rl": {"n": 110, "mean": 0.2, "median": 0.1, "share_pos": 0.52, "corr_edge": 0.05, "pct_point_stable": 0.99},
           "ou": {"n": 100, "mean": -0.1, "median": 0.0, "share_pos": 0.48, "corr_edge": -0.02, "pct_point_stable": 0.4},
           "entry_hour_table": {12: {"n": 80, "mean": 0.1}, 15: {"n": 40, "mean": -0.2}}}
    text = render.render_clv_section(agg)
    assert "CLV" in text
    assert "120" in text and "292" in text     # headroom / total
    assert "0.52" in text or "52" in text        # rl share>0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/tests/test_clv.py -k render_clv -q`
Expected: FAIL with `AttributeError: module 'lib.render' has no attribute 'render_clv_section'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/lib/render.py`, add a section renderer and call it from `render_report`:

```python
def _f(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def render_clv_section(clv: dict) -> str:
    rl, ou = clv["rl"], clv["ou"]
    hours = "".join(f"  - {h}:00 ET｜n={v['n']}｜mean {_f(v['mean'])}pp\n"
                    for h, v in clv["entry_hour_table"].items())
    return (
        "## CLV(領先指標:線往模型 pick 那側移動了多少 pp)\n"
        f"_headroom 子集(entry 嚴格早於 close):{clv['n_headroom']} / {clv['n_total']}"
        f"｜entry→close 中位 {_f(clv['median_minutes_gap'], 0)} 分_\n\n"
        f"- RL:n={rl['n']}｜mean {_f(rl['mean'])}pp｜median {_f(rl['median'])}pp"
        f"｜往我方比例 {_pct(rl['share_pos'])}｜corr(|edge|,CLV) {_f(rl['corr_edge'])}"
        f"｜點數穩定 {_pct(rl['pct_point_stable'])}\n"
        f"- O/U:n={ou['n']}｜mean {_f(ou['mean'])}pp｜median {_f(ou['median'])}pp"
        f"｜往我方比例 {_pct(ou['share_pos'])}｜corr(|edge|,CLV) {_f(ou['corr_edge'])}"
        f"｜點數穩定 {_pct(ou['pct_point_stable'])}\n"
        f"- 依 entry 時段:\n{hours}\n"
        "> mean CLV 顯著 >0 才代表模型 edge 訊號有領先力;≈0 = 與無 alpha 結論一致。\n"
        "> ⚠️ 存檔『close』為軟代理(中位約開賽前 71 分)、4/28 前無盤中 odds、"
        "點數移動時 CLV 為近似(看點數穩定比例)。\n"
    )


def render_report(*, df: pd.DataFrame, rl: dict, ou: dict, edge: dict,
                  month: str, out_path: Path, clv: dict | None = None) -> None:
    valid = 0
    if len(df):
        valid = int(((~df["odds_missing"]) & (~df["result_missing"])).sum())
    lines = [
        f"# MLB 回測(v2)— {month}",
        "",
        f"_有效樣本(odds + result 皆有):{valid} / {len(df)}_",
        "",
        "## RL 過盤(model p>0.5 預測主過盤是否命中)",
        f"- n = {rl['n']}｜命中率 = {_pct(rl['rl_hit_rate'])}",
        "",
        "## O/U(model p>0.5 預測 Over 是否命中,排除 push)",
        f"- n = {ou['n']}｜命中率 = {_pct(ou['ou_hit_rate'])}",
        "",
        "## edge 校準(正 edge 那側實際命中率)",
        f"- RL 正 edge:n = {edge['rl_pos_edge_n']}｜命中 = {_pct(edge['rl_pos_edge_hit'])}",
        f"- O/U 正 edge:n = {edge['ou_pos_edge_n']}｜命中 = {_pct(edge['ou_pos_edge_hit'])}",
        "",
    ]
    if clv is not None:
        lines += [render_clv_section(clv), ""]
    lines += [
        "> σ_team / 權重未經回測重新擬合前,edge 命中僅供觀察,不可當下注依據。",
        "",
        "<!-- 結論待人工填 -->",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Wire it in `scripts/backtest.py`**

Add the import and snapshot dir near the top imports:

```python
from lib.load import build_dataframe_for_month
from lib.metrics import compute_rl_metrics, compute_ou_metrics, compute_edge_calibration
from lib.render import render_report, render_details_csv
from lib.clv import compute_clv_rows, aggregate_clv

SNAPSHOTS_DIR = SKILL_ROOT / "odds" / "odds_snapshots"
```

In `cmd_run`, after `edge = compute_edge_calibration(df)`:

```python
    edge = compute_edge_calibration(df)
    clv = aggregate_clv(compute_clv_rows(df.to_dict("records"), SNAPSHOTS_DIR)) if len(df) else None

    report_path = out_dir / f"{args.month}-report.md"
    csv_path = out_dir / f"{args.month}-details.csv"
    render_report(df=df, rl=rl, ou=ou, edge=edge, month=args.month, out_path=report_path, clv=clv)
```

(`SKILL_ROOT` is already defined in `backtest.py`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest scripts/tests -q`
Expected: PASS (all — existing backtest/render tests still green since `clv` defaults to `None`).

- [ ] **Step 6: Commit** (checkpoint)

```bash
git add scripts/lib/render.py scripts/backtest.py scripts/tests/test_clv.py
git commit -m "feat(backtest): wire CLV measuring stick into the report"
```

---

## Task 6: Run it + read the verdict (operational, not TDD)

**Files:** none (operational). Reads existing snapshots + features.json; updates `analysis-data/backtest/2026-05-report.md` with the CLV section.

- [ ] **Step 1: Full suite green**

Run: `python -m pytest scripts/tests -q`
Expected: PASS (all).

- [ ] **Step 2: Run the backtest**

Run: `python scripts/backtest.py run --month 2026-05`
Expected: same RL/OU/edge as before, plus the report now has a CLV section. Fast (reads local files only).

- [ ] **Step 3: Read the CLV section**

Read `analysis-data/backtest/2026-05-report.md`. Note: `n_headroom / n_total` (expect headroom well below total — the ~120/292 the workflow estimated, since pre-4/28 has no intra-day odds and many games froze at one slot), RL & O-U mean CLV (pp), share>0, corr(|edge|, CLV), entry-hour table, median minutes gap, % point-stable.

- [ ] **Step 4: Sanity checks**

Confirm: `n_headroom ≤ n_total` and is plausibly in the low hundreds or less; `median_minutes_gap` is positive (entry strictly before close); O-U `pct_point_stable` is lower than RL's (totals move, RL fixed ±1.5) — if RL point-stable isn't ~1.0, investigate. If `n_headroom` is ~0, the headroom subset is too thin to read — report that honestly (it means the archive can't measure CLV, which is itself the finding motivating forward capture / subsystem B).

- [ ] **Step 5: Report to user**

Summarize honestly: how many games actually had headroom, whether the model's pick-side line moved toward us on average (mean CLV vs ~0), whether bigger edge correlated with bigger favorable move, and the point-stability caveat. Frame mean CLV ≈ 0 as *consistent with* the established no-alpha finding, not a new failure. Note that thin headroom strengthens the case for subsystem B (forward near-close capture).

> **Known limitations (document, do not fix here):** the archived close is a soft proxy (~71 min pre-commence); no intra-day odds before 2026-04-28; CLV across a moved point is approximate (surfaced via % point-stable); this measures the line, not realized bets.

---

## Self-Review

**Spec coverage:** entry=earliest / close=latest pre-commence via `find_entry_close_snapshots` with strict `snap_ts<commence` (Task 1) ✓; CLV = signed no-vig pp in pick direction, headroom subset (Tasks 2–3) ✓; pick from edges (Task 2 `pick_sides`) ✓; aggregation mean/median/share>0/corr/entry-hour/diagnostics (Task 3 `aggregate_clv`) ✓; point-move approximation surfaced via `pct_point_stable` (Tasks 3, 5) ✓; team names for matching (Task 4) ✓; wired into report additively, `clv` defaults None so existing tests pass (Task 5) ✓; read-only / no model mutation / no fetching (Tasks 5–6) ✓; honest caveats in the rendered section (Task 5) ✓; run + read verdict (Task 6) ✓. B (forward capture), alt-line CLV, bet selection — correctly out of scope.

**Placeholder scan:** none — every code/test step is complete.

**Type/name consistency:** `find_entry_close_snapshots(snapshots_dir, date, home_team, away_team) -> (game|None, game|None)` (Task 1) called in `compute_clv_rows` (Task 3). clv row keys produced in Task 3 match those consumed by `aggregate_clv` (Task 3) and the Task 3 aggregate-test. `pick_sides`/`no_vig_for`/`point_for`/`clv_pp`/`_et_hour` (Task 2) used by `compute_clv_rows` (Task 3). `aggregate_clv` result keys (`n_total, n_headroom, median_minutes_gap, rl{n,mean,median,share_pos,corr_edge,pct_point_stable}, ou{...}, entry_hour_table`) produced in Task 3 and consumed by `render_clv_section` (Task 5) — consistent. `render_report(..., clv=None)` (Task 5) matches the `backtest.py` call passing `clv=clv` (Task 5). `load._build_row` adds `home_team`/`away_team` (Task 4) consumed by `compute_clv_rows` via `df.to_dict("records")` (Task 5).
