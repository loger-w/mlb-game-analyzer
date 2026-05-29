# MLB Point-in-Time Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the refactored team-level prediction inputs point-in-time (no look-ahead), so the new RL/OU model can be honestly backtested by backfilling features.json over past dates (May 2026).

**Architecture:** Three inputs feed `run_model`: team RS/RA (already point-in-time via `before_date`), starter FIP, and bullpen relief ERA. We fix the two with look-ahead: (1) starter FIP via MLB `byDateRange` with `endDate = game_date − 1`; (2) bullpen relief ERA — which has **no clean point-in-time endpoint** — by reconstructing relief lines (pitchers with `gamesStarted==0`) from per-game boxscores, building a league-wide cached index once, then summing games strictly before the game date. A `backfill.py` runner pre-builds the index then loops `predict_game.predict_all` across a date range. Existing `fetch_results` (already fixed) + `backtest` close the loop.

**Tech Stack:** Python 3, `requests`, `pytest`, MLB Stats API (`statsapi.mlb.com/api/v1`).

**Commit policy:** This repo's owner commits manually. Treat the `Commit` steps as checkpoints — run them only if the owner has asked for commits; otherwise skip and leave changes staged for review.

**Already done (prerequisite, do not redo):** `fetch_results.find_matchup_dir_by_pk` was fixed to read `game_pk` from either `game_data.json` (old) or `features.json` (new) via `_read_game_pk`. Test: `scripts/tests/test_fetch_results.py::test_find_matchup_dir_by_pk_reads_features_json`.

---

## File Structure

- **Create** `scripts/bullpen.py` — relief-ERA reconstruction: pure parsers (`relief_er_ip`, `relief_era_as_of`) + I/O index builder + disk cache + `relief_era` query. Lives in `scripts/` (not `lib/`) because it feeds prediction inputs, alongside `fetch_inputs.py`. Self-contained `_parse_ip` to avoid an import cycle with `fetch_inputs`.
- **Create** `scripts/tests/test_bullpen.py` — unit tests for the pure parsers, the index builder (injected fetchers), the cache, and `relief_era`.
- **Modify** `scripts/fetch_inputs.py` — `fetch_starter` → `byDateRange` point-in-time (new `end_date` arg); `fetch_bullpen_era` → delegate to `bullpen.relief_era` (new `as_of` arg, lazy import); `fetch_inputs` → compute `cutoff = date − 1 day`, thread it. Add pure helper `_stat_from_byrange_splits`.
- **Modify** `scripts/tests/test_fetch_inputs.py` — add tests for `_stat_from_byrange_splits`.
- **Create** `scripts/backfill.py` — CLI runner: pure `daterange` + main loop that pre-builds the relief index then calls `predict_game.predict_all` per date.
- **Create** `scripts/tests/test_backfill.py` — unit test for `daterange`.

**Alignment invariant (all three inputs use "through the day before the game"):**
- RS/RA: `_team_rs_ra(before_date=date)` → end = `date − 1` (unchanged).
- Starter: `byDateRange` `endDate = date − 1` (inclusive).
- Bullpen: `relief_era_as_of(per_game, as_of=date)` counts games with `date_str < date`.

---

## Task 1: Bullpen pure functions (parsers, no I/O)

**Files:**
- Create: `scripts/bullpen.py`
- Test: `scripts/tests/test_bullpen.py`

- [ ] **Step 1: Write the failing tests**

Create `scripts/tests/test_bullpen.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bullpen


def test_relief_er_ip_sums_only_relievers():
    """Starter (gamesStarted=1) excluded; relievers (gamesStarted=0) summed."""
    side = {
        "pitchers": [10, 20, 30],
        "players": {
            "ID10": {"stats": {"pitching": {"gamesStarted": 1, "earnedRuns": 4, "inningsPitched": "6.0"}}},
            "ID20": {"stats": {"pitching": {"gamesStarted": 0, "earnedRuns": 1, "inningsPitched": "1.2"}}},
            "ID30": {"stats": {"pitching": {"gamesStarted": 0, "earnedRuns": 2, "inningsPitched": "1.0"}}},
        },
    }
    er, ip = bullpen.relief_er_ip(side)
    assert er == 3
    assert abs(ip - (1 + 2/3 + 1.0)) < 1e-9


def test_relief_er_ip_skips_pitchers_without_pitching_line():
    side = {"pitchers": [10], "players": {"ID10": {"stats": {}}}}
    assert bullpen.relief_er_ip(side) == (0.0, 0.0)


def test_relief_era_as_of_counts_strictly_before():
    per_game = [
        {"date": "2026-05-01", "er": 2, "ip": 3.0},
        {"date": "2026-05-02", "er": 1, "ip": 3.0},
        {"date": "2026-05-05", "er": 9, "ip": 1.0},  # on/after as_of → excluded
    ]
    # as_of 2026-05-05 → only 05-01 & 05-02: ER=3, IP=6 → 9*3/6 = 4.50
    assert bullpen.relief_era_as_of(per_game, "2026-05-05") == 4.50


def test_relief_era_as_of_none_when_no_innings():
    assert bullpen.relief_era_as_of([{"date": "2026-05-01", "er": 0, "ip": 0.0}], "2026-05-05") is None
    assert bullpen.relief_era_as_of([], "2026-05-05") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/tests/test_bullpen.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bullpen'` (or `AttributeError`).

- [ ] **Step 3: Write minimal implementation**

Create `scripts/bullpen.py`:

```python
"""Point-in-time 牛棚 relief ERA:從每場 boxscore 重建後援(該場 gamesStarted==0)ER/IP,
依日期截止累計查詢。線上模型用 relief-only(sitCodes=rp);MLB API 無乾淨的 point-in-time
relief 端點(byDateRange+rp 數值錯亂、statSplits+日期被忽略),故由 boxscore 重建。"""
import json
import sys
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
DEFAULT_CACHE_DIR = SKILL_ROOT / "analysis-data" / "backtest" / "cache"


def _parse_ip(ip_str) -> float:
    """MLB inningsPitched 字串 → float(.1=1/3, .2=2/3)。自帶以避免與 fetch_inputs 互相 import。"""
    whole, _, frac = str(ip_str).partition(".")
    thirds = {"1": 1/3, "2": 2/3}.get(frac, 0.0)
    return int(whole or 0) + thirds


def relief_er_ip(side: dict) -> tuple[float, float]:
    """單場單隊:加總後援(該場 gamesStarted==0)的 ER 與 IP。side = boxscore.teams[home|away]。"""
    players = side.get("players", {})
    er = 0.0
    ip = 0.0
    for pid in side.get("pitchers", []):
        ps = players.get(f"ID{pid}", {}).get("stats", {}).get("pitching", {})
        if not ps:
            continue
        if int(ps.get("gamesStarted", 0) or 0) != 0:
            continue
        er += int(ps.get("earnedRuns", 0) or 0)
        ip += _parse_ip(ps.get("inningsPitched", "0"))
    return er, ip


def relief_era_as_of(per_game: list[dict], as_of: str) -> float | None:
    """per_game=[{date,er,ip}]。加總 date < as_of 的場次 → ERA=9*ER/IP。IP=0 → None。"""
    tot_er = 0.0
    tot_ip = 0.0
    for g in per_game:
        if g["date"] < as_of:
            tot_er += g["er"]
            tot_ip += g["ip"]
    if tot_ip == 0:
        return None
    return round(9.0 * tot_er / tot_ip, 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_bullpen.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit** (checkpoint — see Commit policy)

```bash
git add scripts/bullpen.py scripts/tests/test_bullpen.py
git commit -m "feat(bullpen): relief ER/IP parsers for point-in-time ERA"
```

---

## Task 2: Bullpen index builder + cache + relief_era (I/O, injectable)

**Files:**
- Modify: `scripts/bullpen.py`
- Test: `scripts/tests/test_bullpen.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts/tests/test_bullpen.py`:

```python
def _fake_games(year, through):
    return [
        {"game_pk": 1, "date": "2026-05-01", "home_id": 141, "away_id": 147},
        {"game_pk": 2, "date": "2026-05-02", "home_id": 147, "away_id": 141},
    ]


def _fake_boxscore(game_pk):
    # Every game: home relief ER=1 IP=2.0, away relief ER=3 IP=2.0
    side = lambda er: {"pitchers": [9],
                       "players": {"ID9": {"stats": {"pitching":
                           {"gamesStarted": 0, "earnedRuns": er, "inningsPitched": "2.0"}}}}}
    return {"teams": {"home": side(1), "away": side(3)}}


def test_build_relief_index_accumulates_both_teams():
    idx = bullpen.build_relief_index(2026, "2026-05-31",
                                     games_fetcher=_fake_games, boxscore_fetcher=_fake_boxscore)
    # 141: home in g1 (er1) + away in g2 (er3); 147: away in g1 (er3) + home in g2 (er1)
    assert [r["er"] for r in idx["141"]] == [1, 3]
    assert [r["er"] for r in idx["147"]] == [3, 1]


def test_load_or_build_index_caches_and_reuses(tmp_path, monkeypatch):
    calls = {"n": 0}
    def counting_games(year, through):
        calls["n"] += 1
        return _fake_games(year, through)
    monkeypatch.setattr(bullpen, "_fetch_season_final_games", counting_games)
    monkeypatch.setattr(bullpen, "_fetch_boxscore", _fake_boxscore)

    a = bullpen.load_or_build_index(2026, needed_through="2026-05-10", cache_dir=tmp_path)
    b = bullpen.load_or_build_index(2026, needed_through="2026-05-10", cache_dir=tmp_path)
    assert a == b
    assert calls["n"] == 1  # second call served from cache
    assert (tmp_path / "relief_index_2026.json").exists()


def test_load_or_build_index_rebuilds_when_coverage_insufficient(tmp_path, monkeypatch):
    calls = {"n": 0}
    def counting_games(year, through):
        calls["n"] += 1
        return _fake_games(year, through)
    monkeypatch.setattr(bullpen, "_fetch_season_final_games", counting_games)
    monkeypatch.setattr(bullpen, "_fetch_boxscore", _fake_boxscore)

    bullpen.load_or_build_index(2026, needed_through="2026-05-01", cache_dir=tmp_path)
    bullpen.load_or_build_index(2026, needed_through="2026-05-20", cache_dir=tmp_path)  # beyond built_through
    assert calls["n"] == 2


def test_relief_era_uses_cache_and_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(bullpen, "_fetch_season_final_games", _fake_games)
    monkeypatch.setattr(bullpen, "_fetch_boxscore", _fake_boxscore)
    # team 141, as_of 2026-05-02 → only g1 (er1, ip2.0) → 9*1/2 = 4.50
    assert bullpen.relief_era(141, 2026, "2026-05-02", cache_dir=tmp_path) == 4.50
    # unknown team → no innings → fallback
    assert bullpen.relief_era(999, 2026, "2026-05-02", cache_dir=tmp_path, fallback=4.00) == 4.00
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/tests/test_bullpen.py -q`
Expected: FAIL with `AttributeError: module 'bullpen' has no attribute 'build_relief_index'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/bullpen.py`:

```python
def _fetch_season_final_games(year: int, through: str) -> list[dict]:
    """賽季 Final 例行賽:回 [{game_pk,date,home_id,away_id}]。一次 range 查詢全聯盟。"""
    params = {"sportId": 1, "startDate": f"{year}-03-01", "endDate": through, "gameType": "R"}
    r = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=30)
    r.raise_for_status()
    out = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final" or g.get("gameType") != "R":
                continue
            out.append({"game_pk": g["gamePk"], "date": g["gameDate"][:10],
                        "home_id": g["teams"]["home"]["team"]["id"],
                        "away_id": g["teams"]["away"]["team"]["id"]})
    return out


def _fetch_boxscore(game_pk: int) -> dict:
    r = requests.get(f"{MLB_API_BASE}/game/{game_pk}/boxscore", timeout=15)
    r.raise_for_status()
    return r.json()


def build_relief_index(year: int, through: str,
                       games_fetcher=None, boxscore_fetcher=None) -> dict:
    """一次掃全季 Final 賽,回 {team_id(str): [{date,er,ip}]}。fetchers 預設真連線,可注入供測試。"""
    games_fetcher = games_fetcher or _fetch_season_final_games
    boxscore_fetcher = boxscore_fetcher or _fetch_boxscore
    index: dict[str, list] = {}
    for g in games_fetcher(year, through):
        box = boxscore_fetcher(g["game_pk"])
        teams = box.get("teams", {})
        for side_key, tid in (("home", g["home_id"]), ("away", g["away_id"])):
            er, ip = relief_er_ip(teams.get(side_key, {}))
            index.setdefault(str(tid), []).append({"date": g["date"], "er": er, "ip": ip})
    return index


def _cache_path(year: int, cache_dir) -> Path:
    return Path(cache_dir) / f"relief_index_{year}.json"


def load_or_build_index(year: int, needed_through: str,
                        cache_dir=DEFAULT_CACHE_DIR, refresh: bool = False) -> dict:
    """載入快取(覆蓋足夠才用)否則重建。covers iff built_through >= needed_through。"""
    p = _cache_path(year, cache_dir)
    if p.exists() and not refresh:
        cached = json.loads(p.read_text(encoding="utf-8"))
        if cached.get("built_through", "") >= needed_through:
            return cached["index"]
    index = build_relief_index(year, needed_through)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"built_through": needed_through, "index": index}, ensure_ascii=False),
                 encoding="utf-8")
    return index


def relief_era(team_id: int, year: int, as_of: str,
               cache_dir=DEFAULT_CACHE_DIR, fallback: float = 4.00) -> float:
    """team_id 在 as_of(不含當日)之前的 point-in-time relief ERA。無資料 → fallback。"""
    index = load_or_build_index(year, needed_through=as_of, cache_dir=cache_dir)
    val = relief_era_as_of(index.get(str(team_id), []), as_of)
    return val if val is not None else fallback
```

> **Note on the rebuild guard:** during backfill the runner pre-builds with `needed_through = end_date` once (Task 5), so every per-game `relief_era(as_of ≤ end_date)` finds `built_through ≥ as_of` and hits the cache — no mid-loop rebuilds. The guard only triggers rebuilds for ad-hoc/live calls past current coverage.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_bullpen.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit** (checkpoint — see Commit policy)

```bash
git add scripts/bullpen.py scripts/tests/test_bullpen.py
git commit -m "feat(bullpen): league relief index build + cache + point-in-time relief_era"
```

---

## Task 3: Starter FIP → point-in-time via byDateRange

**Files:**
- Modify: `scripts/fetch_inputs.py` (add `_stat_from_byrange_splits`; change `fetch_starter` signature + body)
- Test: `scripts/tests/test_fetch_inputs.py`

**Context:** `byDateRange` returns `stats[0].splits`, and the splits list can contain **duplicate** aggregate entries (observed: 2 identical splits each equal to the full total). `splits[0]` is the correct aggregate — take it, do **not** sum.

- [ ] **Step 1: Write the failing test**

Append to `scripts/tests/test_fetch_inputs.py`:

```python
def test_stat_from_byrange_splits_takes_first():
    # byDateRange 可能回重複 splits;取第一筆(已是彙總),不可加總
    splits = [
        {"stat": {"inningsPitched": "46.2", "strikeOuts": 43, "baseOnBalls": 9, "hitByPitch": 0, "homeRuns": 5}},
        {"stat": {"inningsPitched": "46.2", "strikeOuts": 43, "baseOnBalls": 9, "hitByPitch": 0, "homeRuns": 5}},
    ]
    s = fi._stat_from_byrange_splits(splits)
    assert s["inningsPitched"] == "46.2"
    assert s["strikeOuts"] == 43


def test_stat_from_byrange_splits_empty_is_none():
    assert fi._stat_from_byrange_splits([]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/tests/test_fetch_inputs.py -q`
Expected: FAIL with `AttributeError: module 'fetch_inputs' has no attribute '_stat_from_byrange_splits'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/fetch_inputs.py`, add the pure helper just above `fetch_starter`:

```python
def _stat_from_byrange_splits(splits: list) -> dict | None:
    """byDateRange 的 splits 可能重複;取第一筆彙總(IP 已等於整段總和)。空 → None。"""
    if not splits:
        return None
    return splits[0].get("stat", {})
```

Then replace `fetch_starter` (current signature `fetch_starter(mlbam_id, name, year)`, season stats) with the point-in-time `byDateRange` version:

```python
def fetch_starter(mlbam_id: int | None, name: str, year: int, end_date: str) -> dict:
    """先發 point-in-time 成績(賽季起 → end_date,含)並算 FIP。id 缺/無成績 → fip=None。"""
    base = {"name": name, "id": mlbam_id, "fip": None,
            "ip": None, "k": None, "bb": None, "hbp": None, "hr": None}
    if not mlbam_id:
        return base
    try:
        r = requests.get(f"{MLB_API_BASE}/people/{mlbam_id}/stats",
                         params={"stats": "byDateRange", "group": "pitching", "season": year,
                                 "startDate": f"{year}-03-01", "endDate": end_date},
                         timeout=10)
        r.raise_for_status()
        splits = (r.json().get("stats") or [{}])[0].get("splits") or []
        s = _stat_from_byrange_splits(splits)
        if not s:
            return base
        ip = parse_ip(s.get("inningsPitched", "0"))
        k = int(s.get("strikeOuts", 0)); bb = int(s.get("baseOnBalls", 0))
        hbp = int(s.get("hitByPitch", 0)); hr = int(s.get("homeRuns", 0))
        base.update(ip=ip, k=k, bb=bb, hbp=hbp, hr=hr,
                    fip=calc_fip(hr=hr, bb=bb, hbp=hbp, k=k, ip=ip))
        return base
    except Exception as e:
        print(f"[fetch_inputs] starter {mlbam_id} 失敗:{e}", file=sys.stderr)
        return base
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_fetch_inputs.py -q`
Expected: PASS (all, including the 2 new).

- [ ] **Step 5: Commit** (checkpoint — see Commit policy)

```bash
git add scripts/fetch_inputs.py scripts/tests/test_fetch_inputs.py
git commit -m "feat(fetch_inputs): starter FIP point-in-time via byDateRange"
```

---

## Task 4: Bullpen delegation + thread cutoff in fetch_inputs

**Files:**
- Modify: `scripts/fetch_inputs.py` (`fetch_bullpen_era` body/signature; `fetch_inputs` threading)

**Note:** This task wires modules together (no new pure logic), so it has no unit test of its own; coverage comes from Task 1–3 unit tests plus the Task 6 end-to-end run. After editing, run the full suite to confirm nothing regressed.

- [ ] **Step 1: Replace `fetch_bullpen_era`**

Replace the current `fetch_bullpen_era(team_id, year)` (season `statSplits sitCodes=rp`) with a delegate to the point-in-time accumulator. Lazy-import `bullpen` inside the function to avoid any import cycle:

```python
def fetch_bullpen_era(team_id: int, year: int, as_of: str) -> float:
    """牛棚 relief ERA(point-in-time,不含 as_of 當日)。委派 bullpen.relief_era;無資料 → 4.00。"""
    import bullpen
    return bullpen.relief_era(team_id, year, as_of)
```

- [ ] **Step 2: Thread the cutoff in `fetch_inputs`**

In `fetch_inputs`, after `year = int(date[:4])`, add the cutoff (day before the game), and pass it through. The required edits:

```python
    year = int(date[:4])
    cutoff = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
```

Change the two starter calls to pass `cutoff`:

```python
    home_starter = fetch_starter(home_pp.get("id"), home_pp.get("fullName", "TBD"), year, cutoff)
    away_starter = fetch_starter(away_pp.get("id"), away_pp.get("fullName", "TBD"), year, cutoff)
```

Change the two bullpen calls in the `raw = {...}` dict to pass `as_of=date` (relief_era_as_of counts games `< date` = through cutoff):

```python
        "home_bullpen_era": fetch_bullpen_era(home_id, year, date),
        "away_bullpen_era": fetch_bullpen_era(away_id, year, date),
```

(`datetime` and `timedelta` are already imported at the top of `fetch_inputs.py`.)

- [ ] **Step 3: Run the full suite to verify no regressions**

Run: `python -m pytest scripts/tests -q`
Expected: PASS (all). If any test calls `fetch_starter`/`fetch_bullpen_era` with the old arity, update it to the new signature.

- [ ] **Step 4: Smoke-check one live prediction (point-in-time path end-to-end)**

Run:
```bash
python scripts/predict_game.py --date 2026-05-20 --matchup <AWAY>@<HOME>
```
(Pick any real matchup from that date.) Expected: writes `analysis-data/2026-05-20/<AWAY>@<HOME>/features.json` with `inputs.home_starter.ip` **less than** the pitcher's full-season IP (proving the cutoff applied), and a non-null `inputs.home_bullpen_era`. First run builds the relief index (slow); note that.

- [ ] **Step 5: Commit** (checkpoint — see Commit policy)

```bash
git add scripts/fetch_inputs.py
git commit -m "feat(fetch_inputs): point-in-time cutoff for starter + bullpen (no look-ahead)"
```

---

## Task 5: Backfill runner

**Files:**
- Create: `scripts/backfill.py`
- Test: `scripts/tests/test_backfill.py`

- [ ] **Step 1: Write the failing test**

Create `scripts/tests/test_backfill.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backfill


def test_daterange_inclusive():
    assert backfill.daterange("2026-05-01", "2026-05-03") == [
        "2026-05-01", "2026-05-02", "2026-05-03"]


def test_daterange_single_day():
    assert backfill.daterange("2026-05-07", "2026-05-07") == ["2026-05-07"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/tests/test_backfill.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'backfill'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/backfill.py`:

```python
#!/usr/bin/env python3
"""回填過去日期的 point-in-time 預測(features.json),供回測。

用法:
  python scripts/backfill.py --start 2026-05-01 --end 2026-05-25
  python scripts/backfill.py --start 2026-05-01 --end 2026-05-25 --refresh-cache
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import predict_game
import bullpen

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def daterange(start: str, end: str) -> list[str]:
    """含頭含尾的日期清單(YYYY-MM-DD)。"""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    out = []
    d = s
    while d <= e:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="回填 point-in-time 預測供回測")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--refresh-cache", action="store_true", help="強制重建 relief index")
    args = p.parse_args(argv)

    year = int(args.start[:4])
    print(f"預建 relief index(through {args.end})…可能需數分鐘", file=sys.stderr)
    bullpen.load_or_build_index(year, needed_through=args.end, refresh=args.refresh_cache)

    for d in daterange(args.start, args.end):
        try:
            outs = predict_game.predict_all(d)
            print(f"[{d}] {len(outs)} 場", file=sys.stderr)
        except Exception as e:
            print(f"[{d}] 失敗:{e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/tests/test_backfill.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit** (checkpoint — see Commit policy)

```bash
git add scripts/backfill.py scripts/tests/test_backfill.py
git commit -m "feat(backfill): date-range runner that pre-builds relief index then predicts"
```

---

## Task 6: Execute the backfill + backtest (end-to-end run, not TDD)

**Files:** none (operational). Produces `analysis-data/2026-05-*/**/features.json`, `result.json`, and `analysis-data/backtest/2026-05-report.md` + `2026-05-details.csv`.

- [ ] **Step 1: Full suite green before running**

Run: `python -m pytest scripts/tests -q`
Expected: PASS (all).

- [ ] **Step 2: Run the backfill** (long — ~1000 boxscore fetches for the index, then ~15 games/day; consider running in background)

Run: `python scripts/backfill.py --start 2026-05-01 --end 2026-05-25`
Expected: stderr shows `預建 relief index…` then one `[YYYY-MM-DD] N 場` line per day. Each game writes `analysis-data/<date>/<AWAY>@<HOME>/features.json` (+ `prediction.md`).

- [ ] **Step 3: Fetch results for the backfilled range**

Run: `python scripts/fetch_results.py --month 2026-05`
Expected: per-day `fetched=N matched=M` lines; `matched` should now cover the backfilled matchups (uses the already-fixed `_read_game_pk`).

- [ ] **Step 4: Run the backtest**

Run: `python scripts/backtest.py run --month 2026-05`
Expected: `Loaded <N> rows.` with N in the low hundreds; `Valid (odds+result): <V> / <N>` where V is the games that also had a usable Pinnacle snapshot. Report + CSV written.

- [ ] **Step 5: Sanity-check the report**

Read `analysis-data/backtest/2026-05-report.md`. Confirm: valid-sample count is plausible (≫ 1), RL/OU hit rates are populated, edge-calibration n > 0. Do **not** over-interpret coefficients — `config.py` σ/weights are still un-refit (the report already prints this caveat).

- [ ] **Step 6: Commit** (checkpoint — see Commit policy)

```bash
git add scripts/ analysis-data/backtest/2026-05-report.md analysis-data/backtest/2026-05-details.csv
git commit -m "chore(backtest): point-in-time May backfill + v2 backtest report"
```

> **Known limitations (document, do not fix here):**
> - Doubleheaders: `predict_all` writes both games to the same `<AWAY>@<HOME>` dir (no `-G1/-G2` suffix), so the second overwrites the first. Rare; affects a handful of May games.
> - Probable pitcher provenance: past-date schedule returns the announced probable (≈ actual starter); treated as known pre-game (acceptable, not look-ahead).
> - Odds coverage: days with no usable pre-commence Pinnacle snapshot yield `odds=null` → excluded from valid sample (handled gracefully, just lowers N).
> - The single legacy `2026-05-27/MIA@TOR/features.json` predates this fix (season-stat starter/bullpen); it is outside the 5/1–5/25 backfill range and does not affect this backtest.

---

## Self-Review

**Spec coverage:** starter point-in-time (Task 3) ✓; bullpen point-in-time via boxscore reconstruction (Tasks 1–2, wired in 4) ✓; cutoff alignment to "day before game" (Task 4) ✓; backfill runner (Task 5) ✓; results + backtest (Task 6, plus the already-done `fetch_results` fix) ✓; RS/RA unchanged (already point-in-time) ✓.

**Type/name consistency:** `relief_er_ip(side)->(er,ip)`, `relief_era_as_of(per_game, as_of)->float|None`, `build_relief_index(year, through, games_fetcher, boxscore_fetcher)->{str:[{date,er,ip}]}`, `load_or_build_index(year, needed_through, cache_dir, refresh)->index`, `relief_era(team_id, year, as_of, cache_dir, fallback)->float`, `fetch_starter(mlbam_id, name, year, end_date)`, `fetch_bullpen_era(team_id, year, as_of)`, `_stat_from_byrange_splits(splits)->dict|None`, `daterange(start, end)->[str]` — consistent across tasks.

**Placeholder scan:** none — every code/test step contains full content.
