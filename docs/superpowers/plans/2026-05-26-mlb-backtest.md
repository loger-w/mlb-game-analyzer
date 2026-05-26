# MLB Skill 回測 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 為 2026 年 5 月 mlb-game-analyzer skill 預測（軌 A：基本面方向 + 總分 + 信心）建構回測 pipeline，產出 Markdown 報告 + CSV，作為 skill 後續迭代燃料。

**Architecture:** 五階段 pipeline。Stage 1 用獨立 `scripts/fetch_results.py` 從 MLB Stats API 抓 final score、寫 per-game `result.json`；Stage 2-5 由 `scripts/backtest.py` 統整：load (parse summary + closing line + result) → metrics → diagnostic → render。所有解析 / 計算 / 渲染 helper 解構到 `scripts/lib/`，純函式好測。

**Tech Stack:** Python 3.x, `requests`（HTTP）, `pandas`（DataFrame）, `pytest`（測試）, `python-dateutil`（UTC parsing）。對齊現有 `scripts/` pattern（同目錄 import、`_team_resolver`、`scripts/tests/`）。

**Spec：** `docs/superpowers/specs/2026-05-26-mlb-backtest-design.md`

---

## File Structure

```
scripts/
  fetch_results.py           [新建] MLB API 抓 final score → result.json
  backtest.py                [新建] CLI 入口 + argparse
  lib/                       [新建目錄]
    __init__.py
    parse_summary.py         summary.md → {direction, total, confidence, flags, park_factor}
    closing_line.py          odds_snapshots/ → 對應 game 的 Pinnacle pre-game 最後 snapshot + no-vig 機率
    load.py                  整合上面兩個 + result.json → pd.DataFrame
    metrics.py               方向 / 總分 / Calibration / 切片 指標
    diagnostic.py            失敗案例選取 + 主訊號抽取
    render.py                Markdown 報告 + CSV 輸出
  tests/
    fixtures/
      backtest/              [新建子目錄]
        sample_snapshot.json   合成 odds snapshot fixture
        sample_summary.md      合成 summary fixture
        sample_result.json     合成 result fixture
    test_fetch_results.py    [新建]
    test_parse_summary.py    [新建]
    test_closing_line.py     [新建]
    test_load.py             [新建]
    test_metrics.py          [新建]
    test_diagnostic.py       [新建]
    test_render.py           [新建]
    test_backtest_e2e.py     [新建]

analysis-data/
  {date}/{matchup}/
    result.json              [新增，由 Task 2 產生 22×~13 ≈ 271 個]
  backtest/                  [新建，由 Task 11 產生]
    2026-05-report.md
    2026-05-details.csv
```

`scripts/lib/` 是新目錄，符合「small focused files」原則 — 既有 `scripts/` 已有不少根層級 lib (`_utils.py`, `_team_resolver.py`, `signals_lib.py` 等)，但這次 backtest 形成獨立模組組，集中到 `lib/` 子目錄避免污染根層級。

---

## Task 1: `scripts/fetch_results.py` — 從 MLB API 抓 final score 寫 result.json

**Files:**
- Create: `scripts/fetch_results.py`
- Create: `scripts/tests/test_fetch_results.py`
- Create: `scripts/tests/fixtures/backtest/sample_mlb_schedule.json`

**Note:** spec §2 提到從 git commit `3c1cd89` 撈回 — 但該版本 import `review_stats` 已被刪。改採「保留 `fetch_final_scores()` 抓資料邏輯精神，但重寫 main + I/O 對齊本 spec 的 result.json schema」。

- [ ] **Step 1: 從 git 撈出 3c1cd89 版本當參考**

```bash
git show 3c1cd89:scripts/fetch_results.py > /tmp/fetch_results_old.py
```

只看 `fetch_final_scores()` 函式（讀 MLB Schedule API、過濾 Final + Regular Season、抽 home/away score）。

- [ ] **Step 2: 建立 fixture 檔**

`scripts/tests/fixtures/backtest/sample_mlb_schedule.json`（精簡的 MLB API 響應）：

```json
{
  "dates": [
    {
      "date": "2026-05-02",
      "games": [
        {
          "gamePk": 823554,
          "gameType": "R",
          "status": {"abstractGameState": "Final", "detailedState": "Final"},
          "teams": {
            "home": {"team": {"id": 147, "name": "New York Yankees"}, "score": 5},
            "away": {"team": {"id": 110, "name": "Baltimore Orioles"}, "score": 8}
          }
        },
        {
          "gamePk": 823555,
          "gameType": "R",
          "status": {"abstractGameState": "Preview", "detailedState": "Postponed"},
          "teams": {
            "home": {"team": {"id": 142, "name": "Minnesota Twins"}, "score": 0},
            "away": {"team": {"id": 141, "name": "Toronto Blue Jays"}, "score": 0}
          }
        }
      ]
    }
  ]
}
```

- [ ] **Step 3: 寫 failing test**

`scripts/tests/test_fetch_results.py`:

```python
"""Tests for scripts/fetch_results.py"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from fetch_results import fetch_final_scores, build_result_record


FIXTURE = Path(__file__).parent / "fixtures" / "backtest" / "sample_mlb_schedule.json"


def test_fetch_final_scores_filters_to_final_regular_season():
    fake_resp = json.loads(FIXTURE.read_text(encoding="utf-8"))
    with patch("fetch_results.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = fake_resp
        mock_get.return_value.raise_for_status = lambda: None
        results = fetch_final_scores("2026-05-02")
    assert len(results) == 1
    r = results[0]
    assert r["game_pk"] == 823554
    assert r["home_team"] == "New York Yankees"
    assert r["away_team"] == "Baltimore Orioles"
    assert r["home_score"] == 5
    assert r["away_score"] == 8


def test_build_result_record_winner_away():
    record = build_result_record({
        "game_pk": 823554,
        "home_team": "New York Yankees",
        "away_team": "Baltimore Orioles",
        "home_score": 5,
        "away_score": 8,
    })
    assert record["game_pk"] == 823554
    assert record["winner"] == "AWAY"
    assert record["home_score"] == 5
    assert record["away_score"] == 8
    assert record["total"] == 13
    assert record["status"] == "Final"
    assert record["postponed"] is False


def test_build_result_record_winner_home():
    record = build_result_record({
        "game_pk": 1,
        "home_team": "H",
        "away_team": "A",
        "home_score": 7,
        "away_score": 4,
    })
    assert record["winner"] == "HOME"
    assert record["total"] == 11
```

- [ ] **Step 4: 跑 test 確認失敗**

```bash
python -m pytest scripts/tests/test_fetch_results.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'fetch_results'`

- [ ] **Step 5: 寫 `scripts/fetch_results.py` 實作**

```python
#!/usr/bin/env python3
"""MLB Results Fetcher — 抓 MLB Stats API Final 比分 → 寫 per-game result.json

用法：
  python scripts/fetch_results.py --date 2026-05-02
  python scripts/fetch_results.py --month 2026-05
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
ANALYSIS_DATA_DIR = SKILL_ROOT / "analysis-data"

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def fetch_final_scores(date: str) -> list[dict]:
    """Fetch all Final regular-season games on date from MLB Schedule API.

    Returns list of dicts: {game_pk, home_team, away_team, home_score, away_score}.
    """
    params = {"sportId": 1, "date": date, "hydrate": "linescore"}
    resp = requests.get(MLB_SCHEDULE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    out = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            if g.get("gameType") != "R":
                continue
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            out.append({
                "game_pk": g.get("gamePk"),
                "home_team": home.get("team", {}).get("name", ""),
                "away_team": away.get("team", {}).get("name", ""),
                "home_score": home.get("score", 0),
                "away_score": away.get("score", 0),
            })
    return out


def build_result_record(raw: dict) -> dict:
    """Convert MLB API row → result.json schema per spec §2."""
    home = raw["home_score"]
    away = raw["away_score"]
    return {
        "game_pk": raw["game_pk"],
        "winner": "HOME" if home > away else "AWAY",
        "final_score": [home, away],
        "home_score": home,
        "away_score": away,
        "total": home + away,
        "status": "Final",
        "postponed": False,
    }


def find_matchup_dir(date: str, home_team: str, away_team: str) -> Optional[Path]:
    """Locate analysis-data/{date}/{AWAY_ABBR}@{HOME_ABBR}/ by matching team names.

    Matchup dirs are like 'BAL@NYY' (away@home, both English abbr). We resolve by
    reading each subdir's game_data.json `game.home.team` / `game.away.team`.
    """
    date_dir = ANALYSIS_DATA_DIR / date
    if not date_dir.is_dir():
        return None
    for sub in date_dir.iterdir():
        if not sub.is_dir():
            continue
        gd = sub / "game_data.json"
        if not gd.exists():
            continue
        try:
            data = json.loads(gd.read_text(encoding="utf-8"))
            g = data.get("game", {})
            if g.get("home", {}).get("team") == home_team and g.get("away", {}).get("team") == away_team:
                return sub
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def write_result(matchup_dir: Path, record: dict) -> Path:
    out = matchup_dir / "result.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def process_date(date: str) -> dict:
    """Fetch & write all results for one date. Returns {matched, missing, postponed} counts."""
    scores = fetch_final_scores(date)
    matched = 0
    missing = []
    for raw in scores:
        matchup_dir = find_matchup_dir(date, raw["home_team"], raw["away_team"])
        if matchup_dir is None:
            missing.append(f"{raw['away_team']}@{raw['home_team']}")
            continue
        record = build_result_record(raw)
        write_result(matchup_dir, record)
        matched += 1
    return {"date": date, "fetched": len(scores), "matched": matched, "missing": missing}


def process_month(month: str) -> list[dict]:
    """Process every date directory under analysis-data/ matching month prefix."""
    summaries = []
    for d in sorted(ANALYSIS_DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith(month):
            continue
        if d.name.endswith(".local-backup"):
            continue
        summaries.append(process_date(d.name))
    return summaries


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="YYYY-MM-DD")
    g.add_argument("--month", help="YYYY-MM")
    args = ap.parse_args()

    if args.date:
        summaries = [process_date(args.date)]
    else:
        summaries = process_month(args.month)

    for s in summaries:
        miss_note = f" (matchups not found: {', '.join(s['missing'])})" if s["missing"] else ""
        print(f"{s['date']}: fetched={s['fetched']} matched={s['matched']}{miss_note}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 跑 test 確認通過**

```bash
python -m pytest scripts/tests/test_fetch_results.py -v
```

Expected: PASS — 3 passed

- [ ] **Step 7: Commit**

```bash
git add scripts/fetch_results.py scripts/tests/test_fetch_results.py scripts/tests/fixtures/backtest/sample_mlb_schedule.json
git commit -m "feat(backtest): fetch_results.py 從 MLB API 抓 final score 寫 result.json

- 邏輯參考 git 3c1cd89 fetch_final_scores
- 移除 review_stats 依賴 (該檔已隨 1e35517 刪除)
- 新 result.json schema 對齊 spec §2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: 跑 `fetch_results.py` 補全 5 月比賽結果

**Files:**
- Create (by script): `analysis-data/{date}/{matchup}/result.json` × ~271 個

不是 TDD task（純 data 操作），但有 verification step。

- [ ] **Step 1: 跑單日 dry-check（5/02 一天）確認腳本可運作**

```bash
python scripts/fetch_results.py --date 2026-05-02
```

Expected stdout:
```
2026-05-02: fetched=14 matched=14
```

如果 `matched < fetched`，stdout 會印出 missing matchups — 通常是 team name spelling 不一致（例：MLB API 「Athletics」 vs analysis-data 「OAK / Athletics」）。需排查 team name mapping。

- [ ] **Step 2: 驗證 5/02 任一場 result.json**

```bash
python -c "import json; d=json.load(open('analysis-data/2026-05-02/BAL@NYY/result.json')); print(json.dumps(d, indent=2))"
```

Expected: 含 `game_pk`, `winner`, `final_score`, `total`, `status="Final"`, `postponed=False`。

- [ ] **Step 3: 跑全 5 月**

```bash
python scripts/fetch_results.py --month 2026-05
```

Expected stdout (一行一天)：
```
2026-05-01: fetched=N matched=N
2026-05-02: fetched=14 matched=14
...
2026-05-25: fetched=13 matched=13
```

預期總 matched ≈ 271。若 fetched != matched，stdout 列 missing。

- [ ] **Step 4: 驗證涵蓋率**

```bash
echo "預期 result.json 數量 (應接近 271):" && find analysis-data/2026-05-* -maxdepth 2 -name "result.json" -not -path "*.local-backup*" | wc -l
echo "" && echo "缺少 result.json 的 matchup 目錄:" && for d in analysis-data/2026-05-*/; do [[ "$d" == *backup* ]] && continue; for m in "$d"*/; do [ -d "$m" ] && [ ! -f "${m}result.json" ] && echo "MISSING: $m"; done; done | head -20
```

若有少量 MISSING（postponed / 抓不到 final），記錄但接受 — 它們會在 Task 6 `result_missing=True` 標記後剔出指標。

- [ ] **Step 5: Commit**

```bash
git add analysis-data/2026-05-*/*/result.json
git commit -m "data(backtest): 5/01-5/25 比賽結果 result.json (~271 場)

由 scripts/fetch_results.py 從 MLB Stats API 抓 Final 比分產生。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `scripts/lib/__init__.py` + 共用常數

**Files:**
- Create: `scripts/lib/__init__.py`

- [ ] **Step 1: 建立空 `__init__.py`**

```bash
mkdir -p scripts/lib && type nul > scripts/lib/__init__.py
```

（Windows PowerShell；類 Unix 用 `touch`）

實際內容空白即可（純 package marker）。

- [ ] **Step 2: Commit**

```bash
git add scripts/lib/__init__.py
git commit -m "feat(backtest): scripts/lib/ package marker

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `scripts/lib/parse_summary.py` — 從 summary.md 抽預測欄位

**Files:**
- Create: `scripts/lib/parse_summary.py`
- Create: `scripts/tests/test_parse_summary.py`
- Create: `scripts/tests/fixtures/backtest/sample_summary_finished.md`
- Create: `scripts/tests/fixtures/backtest/sample_summary_template.md`

5 月實證樣本：109 場 finished、162 場 template。Finished phrasing 多樣（純隊伍 last name、HOME/AWAY 標記、team 縮寫混合）— parser 需 fuzzy match 多策略。

- [ ] **Step 1: 建 fixture 檔（finished + template 兩種）**

`scripts/tests/fixtures/backtest/sample_summary_finished.md`：複製真實的 `analysis-data/2026-05-02/BAL@NYY/summary.md` 完整內容（避免重抄；用 `cp`）。

```bash
cp analysis-data/2026-05-02/BAL@NYY/summary.md scripts/tests/fixtures/backtest/sample_summary_finished.md
```

`scripts/tests/fixtures/backtest/sample_summary_template.md`：複製 5/25 的某場 template summary。

```bash
cp analysis-data/2026-05-25/ARI@SF/summary.md scripts/tests/fixtures/backtest/sample_summary_template.md
```

- [ ] **Step 2: 寫 failing test**

`scripts/tests/test_parse_summary.py`:

```python
"""Tests for scripts/lib/parse_summary.py"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.parse_summary import parse_summary

FIXTURES = Path(__file__).parent / "fixtures" / "backtest"


def test_parse_finished_extracts_direction_total_confidence():
    result = parse_summary(
        FIXTURES / "sample_summary_finished.md",
        home_team_abbr="NYY",
        away_team_abbr="BAL",
    )
    assert result is not None
    assert result["direction"] == "HOME"  # "Yankees 中度偏優" → home (NYY)
    assert result["total"] == 8.5
    assert result["confidence"] == "MEDIUM"
    assert result["parse_failed"] is False


def test_parse_template_returns_parse_failed():
    result = parse_summary(
        FIXTURES / "sample_summary_template.md",
        home_team_abbr="SF",
        away_team_abbr="ARI",
    )
    assert result is not None
    assert result["parse_failed"] is True
    assert result["direction"] is None


def test_parse_extracts_flags():
    result = parse_summary(
        FIXTURES / "sample_summary_finished.md",
        home_team_abbr="NYY",
        away_team_abbr="BAL",
    )
    assert isinstance(result.get("has_reverse_platoon"), bool)
    assert isinstance(result.get("has_chain_break_300"), bool)
    assert isinstance(result.get("has_bullpen_il_2plus"), bool)


def test_parse_extracts_park_factor():
    result = parse_summary(
        FIXTURES / "sample_summary_finished.md",
        home_team_abbr="NYY",
        away_team_abbr="BAL",
    )
    assert result["park_factor"] is not None
    assert isinstance(result["park_factor"], float)


def test_direction_phrasing_pure_team_name():
    """ '**Yankees 中度偏優**' → HOME (when home=NYY)"""
    from lib.parse_summary import _resolve_direction
    assert _resolve_direction("**Yankees 中度偏優**。三大核心訊號", "NYY", "BAL") == "HOME"


def test_direction_phrasing_with_marker():
    """ '**AWAY (ATL) 顯著有利**' → AWAY"""
    from lib.parse_summary import _resolve_direction
    assert _resolve_direction("**AWAY (ATL) 顯著有利**。Quintana 崩盤", "COL", "ATL") == "AWAY"


def test_direction_phrasing_abbr_only():
    """ 'CHC 略佔優' → away/home depending on which is CHC"""
    from lib.parse_summary import _resolve_direction
    assert _resolve_direction("CHC 略佔優。三條獨立邊", "CHC", "ARI") == "HOME"
    assert _resolve_direction("CHC 略佔優。三條獨立邊", "ARI", "CHC") == "AWAY"


def test_direction_phrasing_pingpan():
    """ '持平' / '勢均力敵' / '無明顯方向' → 持平"""
    from lib.parse_summary import _resolve_direction
    assert _resolve_direction("持平 — 兩邊投打勢均", "NYY", "BAL") == "持平"
```

- [ ] **Step 3: 跑 test 確認失敗**

```bash
python -m pytest scripts/tests/test_parse_summary.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'lib.parse_summary'`

- [ ] **Step 4: 寫 `scripts/lib/parse_summary.py` 實作**

```python
"""Parse mlb-game-analyzer summary.md into structured prediction fields.

Schema (per spec §2 / §7):
  {direction: "HOME"|"AWAY"|"持平"|None,
   total: float|None,
   confidence: "LOW"|"MEDIUM"|"HIGH"|None,
   park_factor: float|None,
   has_reverse_platoon: bool,
   has_chain_break_300: bool,
   has_bullpen_il_2plus: bool,
   parse_failed: bool}
"""

import re
from pathlib import Path
from typing import Optional

# Team last-name (English) lookup. Built from common usage; not exhaustive but
# covers all 30 MLB teams' single-word last name as appears in finished summaries.
TEAM_LAST_NAME_TO_ABBR = {
    "yankees": "NYY", "mets": "NYM", "red sox": "BOS", "dodgers": "LAD",
    "angels": "LAA", "astros": "HOU", "braves": "ATL", "phillies": "PHI",
    "padres": "SD", "giants": "SF", "cubs": "CHC", "white sox": "CWS",
    "reds": "CIN", "cardinals": "STL", "brewers": "MIL", "pirates": "PIT",
    "diamondbacks": "ARI", "rockies": "COL", "orioles": "BAL", "rays": "TB",
    "blue jays": "TOR", "twins": "MIN", "royals": "KC", "tigers": "DET",
    "guardians": "CLE", "mariners": "SEA", "athletics": "ATH", "rangers": "TEX",
    "marlins": "MIA", "nationals": "WSH",
}

PINGPAN_KEYWORDS = ("持平", "勢均", "無明顯方向", "中性", "難以判定")


def _resolve_direction(phrase: str, home_abbr: str, away_abbr: str) -> Optional[str]:
    """Resolve direction from phrasing in '方向（基本面）' line.

    Strategy chain (first match wins):
      1. Explicit HOME / AWAY marker → use that
      2. 持平/勢均/中性 keyword → "持平"
      3. Team abbreviation (NYY, BAL, ...) → map to HOME/AWAY
      4. Team last name (Yankees, Orioles, ...) → map to abbr → HOME/AWAY
      5. Return None (couldn't resolve)
    """
    p = phrase.lower()

    # 1. Explicit marker
    if "home" in p and "away" not in p:
        return "HOME"
    if "away" in p and "home" not in p:
        return "AWAY"
    # Both present → fall through (rare; usually phrasing like "AWAY (ATL) vs HOME")
    # In that case first occurrence wins:
    if "home" in p and "away" in p:
        return "HOME" if p.index("home") < p.index("away") else "AWAY"

    # 2. 持平 keyword
    for kw in PINGPAN_KEYWORDS:
        if kw in phrase:
            return "持平"

    # 3. Team abbreviation
    for token in re.findall(r"\b([A-Z]{2,4})\b", phrase):
        if token == home_abbr:
            return "HOME"
        if token == away_abbr:
            return "AWAY"

    # 4. Team last name
    for name, abbr in TEAM_LAST_NAME_TO_ABBR.items():
        if name in p:
            if abbr == home_abbr:
                return "HOME"
            if abbr == away_abbr:
                return "AWAY"

    return None


_DIR_LINE_RE = re.compile(r"^-\s+\*\*方向（基本面）\*\*[:：]\s*(.+?)$", re.MULTILINE)
_TOTAL_LINE_RE = re.compile(
    r"^-\s+\*\*總分（基本面）\*\*[:：].*?adjusted\s+([0-9]+(?:\.[0-9]+)?)",
    re.MULTILINE | re.IGNORECASE,
)
_TOTAL_LINE_FALLBACK_RE = re.compile(
    r"^-\s+\*\*總分（基本面）\*\*[:：]\s*\*?\*?\s*([0-9]+(?:\.[0-9]+)?)",
    re.MULTILINE,
)
_CONFIDENCE_RE = re.compile(
    r"^-\s+\*\*信心\*\*[:：]\s*\*?\*?\s*(LOW|MEDIUM|MED|HIGH)",
    re.MULTILINE | re.IGNORECASE,
)
_PARK_FACTOR_RE = re.compile(
    r"Park Factor[:：]\s*([0-9]+(?:\.[0-9]+)?)",
)


def _is_template_line(line: str) -> bool:
    return "<!-- AI" in line


def parse_summary(path: Path, home_team_abbr: str, away_team_abbr: str) -> dict:
    """Parse summary.md → prediction dict (see module docstring for schema).

    If '方向（基本面）' line is template, returns dict with parse_failed=True
    and all fields None / False.
    """
    text = Path(path).read_text(encoding="utf-8")

    result: dict = {
        "direction": None,
        "total": None,
        "confidence": None,
        "park_factor": None,
        "has_reverse_platoon": False,
        "has_chain_break_300": False,
        "has_bullpen_il_2plus": False,
        "parse_failed": True,
    }

    # Direction
    dir_match = _DIR_LINE_RE.search(text)
    if not dir_match:
        return result
    dir_phrase = dir_match.group(1).strip()
    if _is_template_line(dir_phrase):
        return result
    direction = _resolve_direction(dir_phrase, home_team_abbr, away_team_abbr)
    if direction is None:
        return result
    result["direction"] = direction

    # Total
    tot_match = _TOTAL_LINE_RE.search(text)
    if not tot_match:
        tot_match = _TOTAL_LINE_FALLBACK_RE.search(text)
    if tot_match:
        try:
            result["total"] = float(tot_match.group(1))
        except ValueError:
            pass

    # Confidence
    conf_match = _CONFIDENCE_RE.search(text)
    if conf_match:
        c = conf_match.group(1).upper()
        result["confidence"] = "MEDIUM" if c == "MED" else c

    # Park Factor
    pf_match = _PARK_FACTOR_RE.search(text)
    if pf_match:
        try:
            result["park_factor"] = float(pf_match.group(1))
        except ValueError:
            pass

    # Flags
    result["has_reverse_platoon"] = bool(re.search(r"reverse platoon", text, re.IGNORECASE))
    chain_break_match = re.search(r"chain breaks? at #.*OPS 落差\s+([0-9]+\.[0-9]+)", text)
    if chain_break_match:
        try:
            result["has_chain_break_300"] = float(chain_break_match.group(1)) >= 0.300
        except ValueError:
            pass
    result["has_bullpen_il_2plus"] = bool(re.search(r"牛棚 core IL [×x](?:2|3)", text))

    # parse_failed only if direction couldn't be resolved (other fields are best-effort)
    result["parse_failed"] = False
    return result
```

- [ ] **Step 5: 跑 test 確認通過**

```bash
python -m pytest scripts/tests/test_parse_summary.py -v
```

Expected: PASS — 8 passed

如果有任何單 test 失敗，最可能是 fixture 內容跟 parser regex 不對齊。檢查 `sample_summary_finished.md` 的「**信心**：」、「**總分（基本面）**：」實際行內容，調 regex。

- [ ] **Step 6: Commit**

```bash
git add scripts/lib/parse_summary.py scripts/tests/test_parse_summary.py scripts/tests/fixtures/backtest/sample_summary_finished.md scripts/tests/fixtures/backtest/sample_summary_template.md
git commit -m "feat(backtest): parse_summary — summary.md → prediction dict

多策略 fuzzy match direction (HOME/AWAY marker, 持平, 縮寫, last name).
含 fixture-based test (8 cases).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `scripts/lib/closing_line.py` — 找對應 closing snapshot

**Files:**
- Create: `scripts/lib/closing_line.py`
- Create: `scripts/tests/test_closing_line.py`
- Create: `scripts/tests/fixtures/backtest/sample_snapshot_pregame.json`
- Create: `scripts/tests/fixtures/backtest/sample_snapshot_inplay.json`

odds snapshot 是 flat 結構 `odds/odds_snapshots/YYYY-MM-DD_HH-MM-ET.json`，內含 `snapshot_time_utc` + `games[].commence_utc` + `games[].bookmakers.pinnacle.{ml, ou, rl}`，no-vig 機率 (`no_vig_pct`) 已預算好。

- [ ] **Step 1: 建 fixture 檔（pre-game / in-play 各一）**

`scripts/tests/fixtures/backtest/sample_snapshot_pregame.json`：

```json
{
  "snapshot_time_utc": "2026-05-02T16:00:00+00:00",
  "snapshot_time_et": "2026-05-02 12:00 ET",
  "game_count": 1,
  "games": [
    {
      "game": "Baltimore Orioles @ New York Yankees",
      "away_team": "Baltimore Orioles",
      "home_team": "New York Yankees",
      "commence_utc": "2026-05-02T17:36:00Z",
      "commence_et": "2026-05-02 13:36 ET",
      "game_date_et": "2026-05-02",
      "bookmakers": {
        "pinnacle": {
          "title": "Pinnacle",
          "ml": {
            "Baltimore Orioles": {"odds": 2.50, "implied_pct": 40.0, "no_vig_pct": 39.2},
            "New York Yankees": {"odds": 1.61, "implied_pct": 62.1, "no_vig_pct": 60.8}
          },
          "ou": {
            "Over": {"odds": 1.90, "point": 8.5, "implied_pct": 52.6, "no_vig_pct": 51.0},
            "Under": {"odds": 1.99, "point": 8.5, "implied_pct": 50.3, "no_vig_pct": 49.0}
          },
          "rl": {
            "Baltimore Orioles": {"odds": 1.77, "point": 1.5, "implied_pct": 56.5, "no_vig_pct": 55.2},
            "New York Yankees": {"odds": 2.18, "point": -1.5, "implied_pct": 45.9, "no_vig_pct": 44.8}
          }
        }
      }
    }
  ]
}
```

`scripts/tests/fixtures/backtest/sample_snapshot_inplay.json`：相同 schema 但 `snapshot_time_utc` 改為 `"2026-05-02T19:00:00+00:00"`（晚於 commence_utc 17:36）。

- [ ] **Step 2: 寫 failing test**

`scripts/tests/test_closing_line.py`:

```python
"""Tests for scripts/lib/closing_line.py"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.closing_line import (
    find_closing_snapshot_for_game,
    extract_pinnacle_no_vig,
)

FIXTURES = Path(__file__).parent / "fixtures" / "backtest"


def test_finds_latest_pregame_snapshot_excludes_inplay(tmp_path):
    # Two snapshots: one pre-game @ 12:00 ET, one in-play @ 15:00 ET
    pre = (FIXTURES / "sample_snapshot_pregame.json").read_text(encoding="utf-8")
    inp = (FIXTURES / "sample_snapshot_inplay.json").read_text(encoding="utf-8")
    (tmp_path / "2026-05-02_09-00-ET.json").write_text(pre, encoding="utf-8")
    (tmp_path / "2026-05-02_15-00-ET.json").write_text(inp, encoding="utf-8")

    snap, snap_ts = find_closing_snapshot_for_game(
        snapshots_dir=tmp_path,
        date="2026-05-02",
        home_team="New York Yankees",
        away_team="Baltimore Orioles",
    )
    assert snap is not None
    # Must pick the pre-game one (snapshot_time_utc < commence_utc)
    assert "12:00 ET" in snap.get("snapshot_time_et", "") or "09-00" in str(snap_ts)


def test_returns_none_when_no_pregame_snapshot(tmp_path):
    inp = (FIXTURES / "sample_snapshot_inplay.json").read_text(encoding="utf-8")
    (tmp_path / "2026-05-02_15-00-ET.json").write_text(inp, encoding="utf-8")
    snap, snap_ts = find_closing_snapshot_for_game(
        snapshots_dir=tmp_path,
        date="2026-05-02",
        home_team="New York Yankees",
        away_team="Baltimore Orioles",
    )
    assert snap is None


def test_extract_pinnacle_no_vig_returns_complete_dict():
    pre = (FIXTURES / "sample_snapshot_pregame.json")
    import json
    data = json.loads(pre.read_text(encoding="utf-8"))
    game = data["games"][0]
    line = extract_pinnacle_no_vig(game)
    assert abs(line["home_winprob_no_vig"] - 0.608) < 0.001
    assert abs(line["away_winprob_no_vig"] - 0.392) < 0.001
    assert line["total_line"] == 8.5
    assert abs(line["over_no_vig"] - 0.510) < 0.001
    assert abs(line["under_no_vig"] - 0.490) < 0.001


def test_extract_returns_none_if_pinnacle_missing(tmp_path):
    line = extract_pinnacle_no_vig({"bookmakers": {}})
    assert line is None
```

- [ ] **Step 3: 跑 test 確認失敗**

```bash
python -m pytest scripts/tests/test_closing_line.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: 寫 `scripts/lib/closing_line.py` 實作**

```python
"""Find closing-line snapshot for a single game from flat odds/odds_snapshots/.

'Closing' = last Pinnacle pre-game snapshot whose snapshot_time_utc < commence_utc.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def _parse_iso_utc(s: str) -> Optional[datetime]:
    """Parse ISO 8601 timestamp. Handle 'Z' suffix and '+00:00'."""
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def find_closing_snapshot_for_game(
    snapshots_dir: Path,
    date: str,
    home_team: str,
    away_team: str,
) -> tuple[Optional[dict], Optional[str]]:
    """Find latest pre-game snapshot containing this matchup.

    Returns (game_dict, snapshot_filename) or (None, None) if no pre-game snapshot.
    `game_dict` is the inner `games[]` entry, with `snapshot_time_et` injected.
    """
    snapshots_dir = Path(snapshots_dir)
    candidates: list[tuple[datetime, dict, str]] = []

    for f in sorted(snapshots_dir.glob(f"{date}_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        snap_ts = _parse_iso_utc(data.get("snapshot_time_utc", ""))
        if snap_ts is None:
            continue
        for g in data.get("games", []):
            if g.get("home_team") != home_team or g.get("away_team") != away_team:
                continue
            commence_ts = _parse_iso_utc(g.get("commence_utc", ""))
            if commence_ts is None or snap_ts >= commence_ts:
                continue  # in-play / post-game
            g_copy = dict(g)
            g_copy["snapshot_time_et"] = data.get("snapshot_time_et", "")
            g_copy["snapshot_time_utc"] = data.get("snapshot_time_utc", "")
            candidates.append((snap_ts, g_copy, f.name))

    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    _, game_dict, filename = candidates[-1]
    return game_dict, filename


def extract_pinnacle_no_vig(game: dict) -> Optional[dict]:
    """Extract Pinnacle ML / Total no-vig probabilities + line from a snapshot game.

    Returns: {
        home_winprob_no_vig: float (0-1),
        away_winprob_no_vig: float (0-1),
        total_line: float,
        over_no_vig: float (0-1),
        under_no_vig: float (0-1),
    } or None if Pinnacle data unavailable.
    """
    pinn = game.get("bookmakers", {}).get("pinnacle")
    if not pinn:
        return None
    ml = pinn.get("ml", {})
    ou = pinn.get("ou", {})

    home_team = game.get("home_team")
    away_team = game.get("away_team")
    if not (home_team and away_team and home_team in ml and away_team in ml):
        return None

    over = ou.get("Over", {})
    under = ou.get("Under", {})
    if "no_vig_pct" not in over or "no_vig_pct" not in under:
        return None
    if "point" not in over:
        return None

    return {
        "home_winprob_no_vig": ml[home_team]["no_vig_pct"] / 100.0,
        "away_winprob_no_vig": ml[away_team]["no_vig_pct"] / 100.0,
        "total_line": float(over["point"]),
        "over_no_vig": over["no_vig_pct"] / 100.0,
        "under_no_vig": under["no_vig_pct"] / 100.0,
    }
```

- [ ] **Step 5: 跑 test 確認通過**

```bash
python -m pytest scripts/tests/test_closing_line.py -v
```

Expected: PASS — 4 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/lib/closing_line.py scripts/tests/test_closing_line.py scripts/tests/fixtures/backtest/sample_snapshot_pregame.json scripts/tests/fixtures/backtest/sample_snapshot_inplay.json
git commit -m "feat(backtest): closing_line — 找對應 game 的 Pinnacle pre-game 最後 snapshot

從 odds/odds_snapshots/ flat 結構撈該日 snapshot, 篩 snapshot_time_utc <
commence_utc, 取最後一個. no_vig_pct 已預算, 直接讀.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `scripts/lib/load.py` — 整合 parse + closing + result 成 DataFrame

**Files:**
- Create: `scripts/lib/load.py`
- Create: `scripts/tests/test_load.py`

- [ ] **Step 1: 寫 failing test**

`scripts/tests/test_load.py`:

```python
"""Tests for scripts/lib/load.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.load import build_dataframe_for_month, _matchup_to_abbrs


def test_matchup_to_abbrs_basic():
    assert _matchup_to_abbrs("BAL@NYY") == ("BAL", "NYY")
    assert _matchup_to_abbrs("ARI@SF") == ("ARI", "SF")


def test_matchup_to_abbrs_with_suffix():
    # If doubleheader ever uses BAL@NYY-1 / -2 suffix
    assert _matchup_to_abbrs("BAL@NYY-1") == ("BAL", "NYY")


def test_build_dataframe_real_data_smoke(tmp_path):
    """Smoke test against real 2026-05-02 data (1 day, expect 14 rows)."""
    df = build_dataframe_for_month(month="2026-05", days_filter={"2026-05-02"})
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    required_cols = {
        "date", "matchup", "game_pk",
        "skill_direction", "skill_total", "skill_confidence",
        "market_home_winprob_no_vig", "market_total_line",
        "actual_winner", "actual_total",
        "parse_failed", "closing_missing", "result_missing",
        "park_factor", "has_reverse_platoon",
    }
    assert required_cols.issubset(set(df.columns))


def test_build_dataframe_marks_template_as_parse_failed(tmp_path):
    """Template summaries should have parse_failed=True."""
    df = build_dataframe_for_month(month="2026-05", days_filter={"2026-05-25"})
    # 5/25 batch is all template state
    assert df["parse_failed"].sum() > 0
```

注意：此 test 依賴 Task 1+2 (`result.json` 已生成) 和真實 `analysis-data/2026-05-*`，是 integration-ish。`days_filter` 是參數讓 smoke test 跑得快。

- [ ] **Step 2: 跑 test 確認失敗**

```bash
python -m pytest scripts/tests/test_load.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 寫 `scripts/lib/load.py` 實作**

```python
"""Integrate parse_summary + closing_line + result.json into a pandas DataFrame.

One row per game. Marks parse_failed / closing_missing / result_missing flags
so downstream metrics can exclude them while CSV retains every row.
"""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
SKILL_ROOT = SCRIPT_DIR.parent
ANALYSIS_DATA_DIR = SKILL_ROOT / "analysis-data"
SNAPSHOTS_DIR = SKILL_ROOT / "odds" / "odds_snapshots"

from lib.parse_summary import parse_summary
from lib.closing_line import find_closing_snapshot_for_game, extract_pinnacle_no_vig

_MATCHUP_RE = re.compile(r"^([A-Z]{2,4})@([A-Z]{2,4})(?:-\d+)?$")

CONFIDENCE_TO_PROB = {"LOW": 0.55, "MEDIUM": 0.62, "HIGH": 0.72}


def _matchup_to_abbrs(matchup_dir_name: str) -> tuple[Optional[str], Optional[str]]:
    """'BAL@NYY' or 'BAL@NYY-1' → ('BAL', 'NYY')."""
    m = _MATCHUP_RE.match(matchup_dir_name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _read_game_data(matchup_dir: Path) -> Optional[dict]:
    gd = matchup_dir / "game_data.json"
    if not gd.exists():
        return None
    try:
        return json.loads(gd.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_result(matchup_dir: Path) -> Optional[dict]:
    rj = matchup_dir / "result.json"
    if not rj.exists():
        return None
    try:
        return json.loads(rj.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_dataframe_for_month(
    month: str,
    days_filter: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Build per-game DataFrame. `days_filter` ⊆ {"2026-05-02", ...}; None = all."""
    rows = []
    for date_dir in sorted(ANALYSIS_DATA_DIR.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith(month):
            continue
        if date_dir.name.endswith(".local-backup"):
            continue
        if days_filter is not None and date_dir.name not in days_filter:
            continue

        for matchup_dir in sorted(date_dir.iterdir()):
            if not matchup_dir.is_dir():
                continue
            away_abbr, home_abbr = _matchup_to_abbrs(matchup_dir.name)
            if not (away_abbr and home_abbr):
                continue

            row = _build_row(date_dir.name, matchup_dir, home_abbr, away_abbr)
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


def _build_row(date: str, matchup_dir: Path, home_abbr: str, away_abbr: str) -> Optional[dict]:
    game_data = _read_game_data(matchup_dir)
    if game_data is None:
        return None

    game_pk = game_data.get("game", {}).get("gamePk")
    home_team = game_data.get("game", {}).get("home", {}).get("team", "")
    away_team = game_data.get("game", {}).get("away", {}).get("team", "")

    # Parse summary
    summary_path = matchup_dir / "summary.md"
    if not summary_path.exists():
        return None
    pred = parse_summary(summary_path, home_team_abbr=home_abbr, away_team_abbr=away_abbr)

    # Closing snapshot
    snap_game, snap_filename = find_closing_snapshot_for_game(
        snapshots_dir=SNAPSHOTS_DIR,
        date=date,
        home_team=home_team,
        away_team=away_team,
    )
    no_vig = extract_pinnacle_no_vig(snap_game) if snap_game else None
    closing_missing = no_vig is None

    # Result
    result = _read_result(matchup_dir)
    result_missing = result is None

    # Confidence → mapped probability
    skill_conf = pred.get("confidence")
    skill_prob_mapped = CONFIDENCE_TO_PROB.get(skill_conf) if skill_conf else None

    # Market favorite
    market_favorite = None
    market_favorite_winprob = None
    if no_vig:
        if no_vig["home_winprob_no_vig"] >= 0.5:
            market_favorite = "HOME"
            market_favorite_winprob = no_vig["home_winprob_no_vig"]
        else:
            market_favorite = "AWAY"
            market_favorite_winprob = no_vig["away_winprob_no_vig"]

    return {
        "date": date,
        "matchup": matchup_dir.name,
        "game_pk": game_pk,
        # Skill prediction
        "skill_direction": pred.get("direction"),
        "skill_total": pred.get("total"),
        "skill_confidence": skill_conf,
        "skill_prob_mapped": skill_prob_mapped,
        # Market
        "market_home_winprob_no_vig": no_vig["home_winprob_no_vig"] if no_vig else None,
        "market_total_line": no_vig["total_line"] if no_vig else None,
        "market_favorite": market_favorite,
        "market_favorite_winprob": market_favorite_winprob,
        # Actual
        "actual_winner": result.get("winner") if result else None,
        "actual_total": result.get("total") if result else None,
        "actual_home_score": result.get("home_score") if result else None,
        "actual_away_score": result.get("away_score") if result else None,
        # Flags
        "park_factor": pred.get("park_factor"),
        "has_reverse_platoon": pred.get("has_reverse_platoon", False),
        "has_chain_break_300": pred.get("has_chain_break_300", False),
        "has_bullpen_il_2plus": pred.get("has_bullpen_il_2plus", False),
        # Status
        "parse_failed": pred.get("parse_failed", True),
        "closing_missing": closing_missing,
        "closing_snapshot_ts": snap_filename or "",
        "result_missing": result_missing,
        "dossier_path": str((matchup_dir / "dossier.md").relative_to(SKILL_ROOT)),
    }
```

- [ ] **Step 4: 跑 test 確認通過**

```bash
python -m pytest scripts/tests/test_load.py -v
```

Expected: PASS — 4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/load.py scripts/tests/test_load.py
git commit -m "feat(backtest): load — 整合 parse_summary + closing_line + result 成 DataFrame

一場一列, 含 parse_failed/closing_missing/result_missing flag.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `scripts/lib/metrics.py` — 計算所有指標

**Files:**
- Create: `scripts/lib/metrics.py`
- Create: `scripts/tests/test_metrics.py`

- [ ] **Step 1: 寫 failing test (合成 10 場 mini dataset)**

`scripts/tests/test_metrics.py`:

```python
"""Tests for scripts/lib/metrics.py"""
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.metrics import (
    compute_direction_metrics,
    compute_total_metrics,
    compute_calibration,
    compute_slice_metrics,
    derive_direction_outcome,
)


def _mini_df():
    """10 games. 6 skill picks home, 4 picks away. Half direction-hit, half miss.
    Designed so hand-calc metrics are easy."""
    rows = []
    # Format: (skill_dir, market_home_p, actual_winner, skill_total, line, actual_total, conf)
    cases = [
        ("HOME", 0.60, "HOME", 9.0, 8.5, 10, "HIGH"),    # dir_hit, ou_hit, |err|=1
        ("HOME", 0.55, "AWAY", 8.0, 8.5, 5, "MEDIUM"),   # dir_miss, ou_hit, |err|=3
        ("HOME", 0.62, "HOME", 7.5, 8.0, 8, "MEDIUM"),   # dir_hit, push (skill=7.5 < line, actual=push), excluded
        ("HOME", 0.51, "HOME", 8.5, 8.5, 7, "LOW"),      # dir_hit, skill push, excluded
        ("HOME", 0.65, "AWAY", 9.5, 9.0, 11, "HIGH"),    # dir_miss, ou_hit (skill over, actual over)
        ("HOME", 0.58, "HOME", 10.0, 8.5, 9, "LOW"),     # dir_hit, ou_hit, |err|=1
        ("AWAY", 0.45, "AWAY", 7.0, 7.5, 6, "MEDIUM"),   # dir_hit, ou_hit (both under)
        ("AWAY", 0.48, "HOME", 6.5, 7.0, 9, "LOW"),      # dir_miss, ou_miss (skill under, actual over)
        ("AWAY", 0.40, "AWAY", 8.0, 7.5, 8, "HIGH"),     # dir_hit, ou_miss (skill over, actual push, excluded)
        ("AWAY", 0.47, "HOME", 7.0, 7.5, 6, "MEDIUM"),   # dir_miss, ou_hit (both under)
    ]
    for i, (sd, mhp, aw, st, ln, at, cf) in enumerate(cases):
        rows.append({
            "matchup": f"X{i}",
            "skill_direction": sd,
            "market_home_winprob_no_vig": mhp,
            "market_favorite": "HOME" if mhp >= 0.5 else "AWAY",
            "market_favorite_winprob": mhp if mhp >= 0.5 else 1 - mhp,
            "actual_winner": aw,
            "skill_total": st,
            "market_total_line": ln,
            "actual_total": at,
            "skill_confidence": cf,
            "skill_prob_mapped": {"LOW": 0.55, "MEDIUM": 0.62, "HIGH": 0.72}[cf],
            "park_factor": 100.0,
            "has_reverse_platoon": False,
            "has_chain_break_300": False,
            "has_bullpen_il_2plus": False,
            "parse_failed": False,
            "closing_missing": False,
            "result_missing": False,
        })
    return pd.DataFrame(rows)


def test_direction_outcome_helper():
    df = _mini_df()
    out = derive_direction_outcome(df)
    # Game 0: skill=HOME, actual=HOME → hit
    assert out.loc[0, "direction_hit"] == True
    # Game 1: skill=HOME, actual=AWAY → miss
    assert out.loc[1, "direction_hit"] == False


def test_direction_metrics_skill_hit_rate():
    df = _mini_df()
    m = compute_direction_metrics(df)
    # 10 picks total, hits: rows 0,2,3,5,6,8 = 6 hits → 60%
    assert m["skill_n"] == 10
    assert m["skill_hit_rate"] == pytest.approx(0.6)


def test_direction_metrics_market_favorite_hit_rate():
    df = _mini_df()
    m = compute_direction_metrics(df)
    # Market favorite per row: HOME, HOME, HOME, HOME, HOME, HOME, HOME (0.55>0.5: idx 0-5 are HOME-fav, idx 1 is 0.55 still home),
    # idx 6,7,8,9 home<0.5 → AWAY fav.
    # Actual winners: HOME, AWAY, HOME, HOME, AWAY, HOME, AWAY, HOME, AWAY, HOME
    # Market favs:    HOME, HOME, HOME, HOME, HOME, HOME, AWAY, AWAY, AWAY, AWAY
    # Hits:            T,    F,    T,    T,    F,    T,    T,    F,    T,    F  → 6/10 = 60%
    assert m["market_n"] == 10
    assert m["market_hit_rate"] == pytest.approx(0.6)
    assert m["edge_pp"] == pytest.approx(0.0)


def test_total_metrics_excludes_pushes():
    df = _mini_df()
    m = compute_total_metrics(df)
    # MAE = mean(|skill_total - actual_total|) over all 10 rows
    # |9-10|+|8-5|+|7.5-8|+|8.5-7|+|9.5-11|+|10-9|+|7-6|+|6.5-9|+|8-8|+|7-6| = 1+3+0.5+1.5+1.5+1+1+2.5+0+1 = 13
    assert m["total_mae"] == pytest.approx(1.3)
    # OU hit: exclude row 2 (actual==line: 8==8 NO actual=8 line=8 → actually line=8 actual=8 push)
    #         exclude row 3 (skill==line: 8.5==8.5 push)
    #         exclude row 8 (actual==line: 7.5 line, actual 8, no; but skill==line? skill=8 line=7.5, no)
    #   Actually: push = actual_total==line OR skill_total==line.
    #   Row indices with push: 2 (actual=8 line=8.0 → push) and 3 (skill=8.5 line=8.5 push)
    # OU hits among remaining 8: skill_dir = sign(skill_total - line), actual_dir = sign(actual_total - line)
    # Row 0: skill 9-8.5=+, actual 10-8.5=+ → hit
    # Row 1: skill 8-8.5=-, actual 5-8.5=- → hit
    # Row 4: skill 9.5-9=+, actual 11-9=+ → hit
    # Row 5: skill 10-8.5=+, actual 9-8.5=+ → hit
    # Row 6: skill 7-7.5=-, actual 6-7.5=- → hit
    # Row 7: skill 6.5-7=-, actual 9-7=+ → miss
    # Row 8: skill 8-7.5=+, actual 8-7.5=+ → hit
    # Row 9: skill 7-7.5=-, actual 6-7.5=- → hit
    # 7/8 = 87.5%
    assert m["ou_n"] == 8
    assert m["ou_hit_rate"] == pytest.approx(0.875)


def test_calibration_returns_reliability_table():
    df = _mini_df()
    cal = compute_calibration(df)
    assert "reliability_table" in cal
    rt = cal["reliability_table"]
    assert set(rt["confidence"]) == {"LOW", "MEDIUM", "HIGH"}
    # 'mapped_p' must match canonical mapping
    for _, row in rt.iterrows():
        if row["confidence"] == "HIGH":
            assert row["mapped_p"] == 0.72
    # Brier score in result
    assert "brier_score" in cal
    assert 0.0 <= cal["brier_score"] <= 1.0


def test_slice_metrics_park_factor_groups():
    df = _mini_df()
    df.loc[df.index[:3], "park_factor"] = 105.0  # 3 high-PF rows
    slices = compute_slice_metrics(df)
    assert "park_factor_high" in slices.index
    assert "park_factor_mid" in slices.index
```

- [ ] **Step 2: 跑 test 確認失敗**

```bash
python -m pytest scripts/tests/test_metrics.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 寫 `scripts/lib/metrics.py` 實作**

```python
"""Compute backtest metrics from per-game DataFrame.

All functions operate on the DataFrame produced by lib.load.build_dataframe_for_month.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd


def _valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for indicator computation (excludes parse/closing/result failures)."""
    return df[
        (~df["parse_failed"])
        & (~df["closing_missing"])
        & (~df["result_missing"])
    ].copy()


def derive_direction_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Add `direction_hit` column to df. Operates on a copy."""
    out = df.copy()
    out["direction_hit"] = out["skill_direction"] == out["actual_winner"]
    return out


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_direction_metrics(df: pd.DataFrame) -> dict:
    valid = _valid_rows(df)
    # Exclude '持平' skill picks from direction analysis
    direction_valid = valid[valid["skill_direction"].isin(["HOME", "AWAY"])]

    n = len(direction_valid)
    if n == 0:
        return {"skill_n": 0, "skill_hit_rate": None, "market_n": 0, "market_hit_rate": None, "edge_pp": None}

    skill_hits = (direction_valid["skill_direction"] == direction_valid["actual_winner"]).sum()
    skill_rate = skill_hits / n
    skill_ci = _wilson_ci(skill_rate, n)

    market_hits = (direction_valid["market_favorite"] == direction_valid["actual_winner"]).sum()
    market_rate = market_hits / n
    market_ci = _wilson_ci(market_rate, n)

    # Aligned / against-market splits
    aligned = direction_valid[direction_valid["skill_direction"] == direction_valid["market_favorite"]]
    against = direction_valid[direction_valid["skill_direction"] != direction_valid["market_favorite"]]
    aligned_rate = (aligned["skill_direction"] == aligned["actual_winner"]).mean() if len(aligned) else None
    against_rate = (against["skill_direction"] == against["actual_winner"]).mean() if len(against) else None

    return {
        "skill_n": n,
        "skill_hit_rate": skill_rate,
        "skill_ci": skill_ci,
        "market_n": n,
        "market_hit_rate": market_rate,
        "market_ci": market_ci,
        "edge_pp": skill_rate - market_rate,
        "skill_aligned_n": len(aligned),
        "skill_aligned_hit_rate": aligned_rate,
        "skill_against_n": len(against),
        "skill_against_hit_rate": against_rate,
    }


def compute_total_metrics(df: pd.DataFrame) -> dict:
    valid = _valid_rows(df)
    valid = valid[
        valid["skill_total"].notna()
        & valid["actual_total"].notna()
        & valid["market_total_line"].notna()
    ]

    if len(valid) == 0:
        return {"n": 0, "total_mae": None, "total_bias": None, "ou_n": 0, "ou_hit_rate": None}

    abs_err = (valid["skill_total"] - valid["actual_total"]).abs()
    signed_err = valid["skill_total"] - valid["actual_total"]

    # Exclude pushes for OU hit-rate
    line = valid["market_total_line"]
    no_push = valid[
        (valid["actual_total"] != line)
        & (valid["skill_total"] != line)
    ]
    if len(no_push) > 0:
        skill_side = np.sign(no_push["skill_total"] - no_push["market_total_line"])
        actual_side = np.sign(no_push["actual_total"] - no_push["market_total_line"])
        ou_hit_rate = (skill_side == actual_side).mean()
    else:
        ou_hit_rate = None

    return {
        "n": len(valid),
        "total_mae": float(abs_err.mean()),
        "total_bias": float(signed_err.mean()),
        "ou_n": len(no_push),
        "ou_hit_rate": float(ou_hit_rate) if ou_hit_rate is not None else None,
    }


def compute_calibration(df: pd.DataFrame) -> dict:
    valid = _valid_rows(df)
    valid = valid[valid["skill_direction"].isin(["HOME", "AWAY"])]
    valid = valid[valid["skill_confidence"].isin(["LOW", "MEDIUM", "HIGH"])]

    if len(valid) == 0:
        return {"reliability_table": pd.DataFrame(), "brier_score": None, "log_loss": None,
                "brier_baseline_market": None, "log_loss_baseline_market": None}

    valid = valid.copy()
    valid["outcome"] = (valid["skill_direction"] == valid["actual_winner"]).astype(float)

    # Reliability table
    rows = []
    for conf in ["LOW", "MEDIUM", "HIGH"]:
        sub = valid[valid["skill_confidence"] == conf]
        n_sub = len(sub)
        hit = float(sub["outcome"].mean()) if n_sub > 0 else None
        mapped = sub["skill_prob_mapped"].iloc[0] if n_sub > 0 else None
        ci = _wilson_ci(hit, n_sub) if n_sub > 0 and hit is not None else (None, None)
        rows.append({
            "confidence": conf,
            "n": n_sub,
            "mapped_p": mapped,
            "actual_hit_rate": hit,
            "delta": (hit - mapped) if (hit is not None and mapped is not None) else None,
            "ci_low": ci[0],
            "ci_high": ci[1],
        })
    reliability = pd.DataFrame(rows)

    # Brier / log-loss with skill mapping
    p = valid["skill_prob_mapped"].astype(float)
    y = valid["outcome"]
    brier = float(((p - y) ** 2).mean())
    eps = 1e-12
    log_loss = float(-(y * np.log(p.clip(eps, 1 - eps)) + (1 - y) * np.log((1 - p).clip(eps, 1 - eps))).mean())

    # Baseline: market no-vig prob on the side skill picked
    # If skill picked HOME, market_p = market_home_winprob_no_vig; else 1 - that
    market_p_on_skill_side = np.where(
        valid["skill_direction"] == "HOME",
        valid["market_home_winprob_no_vig"],
        1 - valid["market_home_winprob_no_vig"],
    )
    valid["_market_p_skill_side"] = market_p_on_skill_side
    valid_market = valid[valid["_market_p_skill_side"].notna()]
    if len(valid_market) > 0:
        mp = valid_market["_market_p_skill_side"].astype(float)
        my = valid_market["outcome"]
        brier_mkt = float(((mp - my) ** 2).mean())
        log_loss_mkt = float(
            -(my * np.log(mp.clip(eps, 1 - eps)) + (1 - my) * np.log((1 - mp).clip(eps, 1 - eps))).mean()
        )
    else:
        brier_mkt = None
        log_loss_mkt = None

    return {
        "reliability_table": reliability,
        "brier_score": brier,
        "log_loss": log_loss,
        "brier_baseline_market": brier_mkt,
        "log_loss_baseline_market": log_loss_mkt,
    }


def compute_slice_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-slice metrics: dir_hit, ou_hit, mae, bias, n."""
    valid = _valid_rows(df)

    def _slice_stats(sub: pd.DataFrame) -> dict:
        if len(sub) == 0:
            return {"n": 0, "dir_hit": None, "ou_hit": None, "mae": None, "bias": None}
        dir_sub = sub[sub["skill_direction"].isin(["HOME", "AWAY"])]
        dir_hit = (dir_sub["skill_direction"] == dir_sub["actual_winner"]).mean() if len(dir_sub) else None
        # OU
        tot_sub = sub[sub["skill_total"].notna() & sub["actual_total"].notna() & sub["market_total_line"].notna()]
        no_push = tot_sub[(tot_sub["actual_total"] != tot_sub["market_total_line"])
                         & (tot_sub["skill_total"] != tot_sub["market_total_line"])]
        if len(no_push):
            ss = np.sign(no_push["skill_total"] - no_push["market_total_line"])
            sa = np.sign(no_push["actual_total"] - no_push["market_total_line"])
            ou_hit = (ss == sa).mean()
        else:
            ou_hit = None
        mae = float((tot_sub["skill_total"] - tot_sub["actual_total"]).abs().mean()) if len(tot_sub) else None
        bias = float((tot_sub["skill_total"] - tot_sub["actual_total"]).mean()) if len(tot_sub) else None
        return {"n": len(sub), "dir_hit": dir_hit, "ou_hit": ou_hit, "mae": mae, "bias": bias}

    slices: dict[str, dict] = {}

    # Direction
    slices["direction_HOME"] = _slice_stats(valid[valid["skill_direction"] == "HOME"])
    slices["direction_AWAY"] = _slice_stats(valid[valid["skill_direction"] == "AWAY"])

    # Confidence
    for c in ["LOW", "MEDIUM", "HIGH"]:
        slices[f"confidence_{c}"] = _slice_stats(valid[valid["skill_confidence"] == c])

    # Park factor
    slices["park_factor_high"] = _slice_stats(valid[valid["park_factor"] > 102])
    slices["park_factor_mid"] = _slice_stats(
        valid[(valid["park_factor"] >= 98) & (valid["park_factor"] <= 102)]
    )
    slices["park_factor_low"] = _slice_stats(valid[valid["park_factor"] < 98])

    # Flag-based
    slices["has_reverse_platoon"] = _slice_stats(valid[valid["has_reverse_platoon"] == True])
    slices["no_reverse_platoon"] = _slice_stats(valid[valid["has_reverse_platoon"] == False])
    slices["has_chain_break_300"] = _slice_stats(valid[valid["has_chain_break_300"] == True])
    slices["has_bullpen_il_2plus"] = _slice_stats(valid[valid["has_bullpen_il_2plus"] == True])

    return pd.DataFrame.from_dict(slices, orient="index")
```

- [ ] **Step 4: 跑 test 確認通過**

```bash
python -m pytest scripts/tests/test_metrics.py -v
```

Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/metrics.py scripts/tests/test_metrics.py
git commit -m "feat(backtest): metrics — 方向/總分/Calibration/切片 指標計算

含 Wilson CI, Brier, log-loss, reliability table, slice frames.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `scripts/lib/diagnostic.py` — 失敗案例 + 主訊號抽取

**Files:**
- Create: `scripts/lib/diagnostic.py`
- Create: `scripts/tests/test_diagnostic.py`

- [ ] **Step 1: 寫 failing test**

`scripts/tests/test_diagnostic.py`:

```python
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.diagnostic import (
    select_failure_cases,
    extract_main_signal,
)

FIXTURES = Path(__file__).parent / "fixtures" / "backtest"


def _df_with_failures():
    return pd.DataFrame([
        {"date": "2026-05-02", "matchup": "BAL@NYY",
         "skill_direction": "HOME", "skill_confidence": "MEDIUM",
         "skill_total": 8.5, "actual_winner": "AWAY", "actual_total": 13,
         "parse_failed": False, "closing_missing": False, "result_missing": False,
         "dossier_path": "analysis-data/2026-05-02/BAL@NYY/dossier.md"},
        {"date": "2026-05-02", "matchup": "MIL@WSH",
         "skill_direction": "AWAY", "skill_confidence": "HIGH",
         "skill_total": 9.0, "actual_winner": "HOME", "actual_total": 8,
         "parse_failed": False, "closing_missing": False, "result_missing": False,
         "dossier_path": "analysis-data/2026-05-02/MIL@WSH/dossier.md"},
        # LOW confidence miss — should NOT appear in direction miss list
        {"date": "2026-05-03", "matchup": "X@Y",
         "skill_direction": "HOME", "skill_confidence": "LOW",
         "skill_total": 7.0, "actual_winner": "AWAY", "actual_total": 7,
         "parse_failed": False, "closing_missing": False, "result_missing": False,
         "dossier_path": "analysis-data/2026-05-03/X@Y/dossier.md"},
        # Big total miss
        {"date": "2026-05-04", "matchup": "A@B",
         "skill_direction": "HOME", "skill_confidence": "MEDIUM",
         "skill_total": 7.0, "actual_winner": "HOME", "actual_total": 18,
         "parse_failed": False, "closing_missing": False, "result_missing": False,
         "dossier_path": "analysis-data/2026-05-04/A@B/dossier.md"},
    ])


def test_select_failure_cases_filters_low_confidence_dir_miss():
    df = _df_with_failures()
    cases = select_failure_cases(df, top_total_miss=10)
    matchups = set(cases["matchup"])
    # Direction misses at MED/HIGH conf + top total misses
    assert "BAL@NYY" in matchups  # MED conf direction miss
    assert "MIL@WSH" in matchups  # HIGH conf direction miss
    assert "A@B" in matchups       # huge total miss
    # LOW conf direction miss only enters if it's also a top-N total miss
    # Here |7-7|=0 so it shouldn't enter
    assert "X@Y" not in matchups


def test_select_failure_cases_marks_dual_failures():
    df = _df_with_failures()
    cases = select_failure_cases(df, top_total_miss=10)
    bal_row = cases[cases["matchup"] == "BAL@NYY"].iloc[0]
    # |8.5 - 13| = 4.5 — large total miss + direction miss
    assert bal_row["dual_failure"] == True


def test_extract_main_signal_strips_markdown():
    summary_text = "## 整體判斷\n\n- **方向（基本面）**：**Yankees 中度偏優**。三大核心訊號疊加：...\n"
    signal = extract_main_signal(summary_text)
    assert "Yankees" in signal
    assert "**" not in signal  # markdown stripped
    assert len(signal) <= 50
```

- [ ] **Step 2: 跑 test 確認失敗**

```bash
python -m pytest scripts/tests/test_diagnostic.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 寫 `scripts/lib/diagnostic.py` 實作**

```python
"""Select failure cases for the report's diagnostic section."""

import re
from pathlib import Path

import pandas as pd


_DIR_LINE_RE = re.compile(r"^-\s+\*\*方向（基本面）\*\*[:：]\s*(.+?)$", re.MULTILINE)
_MD_STRIP = re.compile(r"\*+")


def extract_main_signal(summary_text: str, max_chars: int = 50) -> str:
    """Extract first ~50 chars of the '方向（基本面）' content, markdown stripped.

    Used in failure case table to give human reader the skill's stated rationale.
    """
    m = _DIR_LINE_RE.search(summary_text)
    if not m:
        return ""
    phrase = m.group(1).strip()
    phrase = _MD_STRIP.sub("", phrase)
    # Trim at first sentence terminator (。/，) if any within max_chars
    for term in ["。", "，", "—"]:
        idx = phrase.find(term)
        if 0 < idx <= max_chars:
            return phrase[:idx].strip()
    return phrase[:max_chars].strip()


def select_failure_cases(df: pd.DataFrame, top_total_miss: int = 10) -> pd.DataFrame:
    """Two rankings merged:
       1. direction_miss at confidence ≥ MEDIUM (all)
       2. top N rows by |skill_total - actual_total|
       Merged + deduped by matchup. Flags `dual_failure` for rows in both lists.
    """
    valid = df[
        (~df["parse_failed"]) & (~df["closing_missing"]) & (~df["result_missing"])
    ].copy()

    # 1. Direction misses (MED/HIGH only)
    dir_miss = valid[
        (valid["skill_direction"].isin(["HOME", "AWAY"]))
        & (valid["skill_direction"] != valid["actual_winner"])
        & (valid["skill_confidence"].isin(["MEDIUM", "HIGH"]))
    ].copy()
    dir_miss["is_direction_miss"] = True

    # 2. Top total misses
    with_err = valid[valid["skill_total"].notna() & valid["actual_total"].notna()].copy()
    with_err["total_abs_error"] = (with_err["skill_total"] - with_err["actual_total"]).abs()
    top_total = with_err.nlargest(top_total_miss, "total_abs_error").copy()
    top_total["is_top_total_miss"] = True

    # Merge
    combined = pd.concat([dir_miss, top_total], ignore_index=False)
    combined["is_direction_miss"] = combined.get("is_direction_miss", False).fillna(False)
    combined["is_top_total_miss"] = combined.get("is_top_total_miss", False).fillna(False)

    # Dedupe by matchup+date (keep flags from both)
    combined = combined.groupby(["date", "matchup"], as_index=False).agg({
        **{c: "first" for c in combined.columns if c not in ("is_direction_miss", "is_top_total_miss", "date", "matchup")},
        "is_direction_miss": "max",
        "is_top_total_miss": "max",
    })
    combined["dual_failure"] = combined["is_direction_miss"] & combined["is_top_total_miss"]

    # Sort: direction miss first by date, then total miss by abs error
    combined = combined.sort_values(
        by=["is_direction_miss", "date"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return combined
```

- [ ] **Step 4: 跑 test 確認通過**

```bash
python -m pytest scripts/tests/test_diagnostic.py -v
```

Expected: PASS — 3 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/diagnostic.py scripts/tests/test_diagnostic.py
git commit -m "feat(backtest): diagnostic — 失敗案例選取 + 主訊號抽取

方向誤判(信心≥MED) + 總分大失誤top10 合併, dual_failure 標記.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: `scripts/lib/render.py` — 寫 Markdown 報告 + CSV

**Files:**
- Create: `scripts/lib/render.py`
- Create: `scripts/tests/test_render.py`

- [ ] **Step 1: 寫 failing test**

`scripts/tests/test_render.py`:

```python
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.render import render_report, render_details_csv


def _sample_data():
    df = pd.DataFrame([{
        "date": "2026-05-02", "matchup": "BAL@NYY", "game_pk": 1,
        "skill_direction": "HOME", "skill_total": 8.5, "skill_confidence": "MEDIUM",
        "skill_prob_mapped": 0.62,
        "market_home_winprob_no_vig": 0.60, "market_total_line": 8.5,
        "market_favorite": "HOME", "market_favorite_winprob": 0.60,
        "actual_winner": "AWAY", "actual_total": 13, "actual_home_score": 5, "actual_away_score": 8,
        "park_factor": 105.0,
        "has_reverse_platoon": True, "has_chain_break_300": False, "has_bullpen_il_2plus": True,
        "parse_failed": False, "closing_missing": False, "closing_snapshot_ts": "2026-05-02_12-00-ET.json",
        "result_missing": False,
        "dossier_path": "analysis-data/2026-05-02/BAL@NYY/dossier.md",
    }])
    direction_metrics = {
        "skill_n": 1, "skill_hit_rate": 0.0, "skill_ci": (0.0, 0.95),
        "market_n": 1, "market_hit_rate": 0.0, "market_ci": (0.0, 0.95),
        "edge_pp": 0.0,
        "skill_aligned_n": 1, "skill_aligned_hit_rate": 0.0,
        "skill_against_n": 0, "skill_against_hit_rate": None,
    }
    total_metrics = {"n": 1, "total_mae": 4.5, "total_bias": -4.5, "ou_n": 1, "ou_hit_rate": 0.0}
    calibration = {
        "reliability_table": pd.DataFrame([
            {"confidence": "MEDIUM", "n": 1, "mapped_p": 0.62, "actual_hit_rate": 0.0, "delta": -0.62, "ci_low": 0, "ci_high": 0.95},
        ]),
        "brier_score": 0.38, "log_loss": 0.95,
        "brier_baseline_market": 0.36, "log_loss_baseline_market": 0.92,
    }
    slices = pd.DataFrame({"n": [1], "dir_hit": [0.0], "ou_hit": [0.0], "mae": [4.5], "bias": [-4.5]},
                         index=["direction_HOME"])
    failures = df.copy()
    failures["dual_failure"] = True
    failures["is_direction_miss"] = True
    failures["is_top_total_miss"] = True
    failures["main_signal"] = "Yankees 中度偏優"
    return df, direction_metrics, total_metrics, calibration, slices, failures


def test_render_report_contains_required_sections(tmp_path):
    df, dm, tm, cal, sl, fc = _sample_data()
    out = tmp_path / "report.md"
    render_report(df=df, direction_metrics=dm, total_metrics=tm,
                  calibration=cal, slices=sl, failure_cases=fc,
                  month="2026-05", out_path=out)
    content = out.read_text(encoding="utf-8")
    assert "# MLB Skill 回測 — 2026 年 5 月" in content
    assert "## 資料健康度" in content
    assert "## TL;DR" in content
    assert "## 1. 方向類指標" in content
    assert "## 2. 總分類指標" in content
    assert "## 3. 信心 Calibration" in content
    assert "## 4. 分組切片" in content
    assert "## 5. 失敗案例" in content
    assert "BAL@NYY" in content


def test_render_details_csv_has_all_columns(tmp_path):
    df, *_ = _sample_data()
    out = tmp_path / "details.csv"
    render_details_csv(df, out_path=out)
    csv_text = out.read_text(encoding="utf-8")
    header = csv_text.splitlines()[0]
    for col in ["date", "matchup", "game_pk", "skill_direction", "actual_winner",
                "park_factor", "has_reverse_platoon", "closing_snapshot_ts"]:
        assert col in header
```

- [ ] **Step 2: 跑 test 確認失敗**

```bash
python -m pytest scripts/tests/test_render.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 寫 `scripts/lib/render.py` 實作**

```python
"""Render backtest output: Markdown report + details CSV."""

from datetime import date as _date
from pathlib import Path
from typing import Optional

import pandas as pd


CSV_COLUMNS = [
    "date", "matchup", "game_pk",
    "skill_direction", "skill_total", "skill_confidence", "skill_prob_mapped",
    "market_home_winprob_no_vig", "market_total_line", "market_favorite", "market_favorite_winprob",
    "actual_winner", "actual_total", "actual_home_score", "actual_away_score",
    "direction_hit", "ou_hit", "total_abs_error", "total_signed_error",
    "brier_score", "log_loss",
    "park_factor", "has_reverse_platoon", "has_chain_break_300", "has_bullpen_il_2plus",
    "closing_snapshot_ts", "closing_missing", "result_missing", "parse_failed",
    "dossier_path",
]


def _enrich_with_per_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add direction_hit / ou_hit / total_abs_error / total_signed_error / brier / log_loss
    per-row (NaN where not applicable)."""
    import numpy as np
    out = df.copy()
    eps = 1e-12

    out["direction_hit"] = (out["skill_direction"] == out["actual_winner"]).where(
        out["skill_direction"].isin(["HOME", "AWAY"]) & out["actual_winner"].notna()
    )
    out["total_abs_error"] = (out["skill_total"] - out["actual_total"]).abs()
    out["total_signed_error"] = out["skill_total"] - out["actual_total"]

    line = out["market_total_line"]
    no_push_mask = (out["actual_total"] != line) & (out["skill_total"] != line) & line.notna()
    ss = np.sign(out["skill_total"] - line)
    sa = np.sign(out["actual_total"] - line)
    out["ou_hit"] = (ss == sa).where(no_push_mask)

    # Brier / log-loss per row (NaN if skill_prob_mapped missing)
    p = out["skill_prob_mapped"]
    y = out["direction_hit"].astype(float)
    out["brier_score"] = ((p - y) ** 2).where(p.notna() & y.notna())
    log_term = -(y * np.log(p.clip(eps, 1 - eps)) + (1 - y) * np.log((1 - p).clip(eps, 1 - eps)))
    out["log_loss"] = log_term.where(p.notna() & y.notna())

    return out


def render_details_csv(df: pd.DataFrame, out_path: Path):
    enriched = _enrich_with_per_row_metrics(df)
    # Ensure all CSV_COLUMNS exist, fill missing with NaN
    for c in CSV_COLUMNS:
        if c not in enriched.columns:
            enriched[c] = None
    enriched = enriched[CSV_COLUMNS]
    # Convert bool/NaN combo cleanly
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False, encoding="utf-8")


def _fmt_pct(x: Optional[float], digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:.{digits}f}%"


def _fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def render_report(
    df: pd.DataFrame,
    direction_metrics: dict,
    total_metrics: dict,
    calibration: dict,
    slices: pd.DataFrame,
    failure_cases: pd.DataFrame,
    month: str,
    out_path: Path,
):
    n_input = len(df)
    n_parse_ok = int((~df["parse_failed"]).sum())
    n_closing_ok = int((~df["closing_missing"]).sum())
    n_result_ok = int((~df["result_missing"]).sum())
    n_valid = int(((~df["parse_failed"]) & (~df["closing_missing"]) & (~df["result_missing"])).sum())

    today = _date.today().isoformat()
    lines = []
    lines.append(f"# MLB Skill 回測 — {month[:4]} 年 {int(month[5:7])} 月")
    lines.append("")
    lines.append(f"_樣本：{n_valid} 場有效（{month} 全月） ｜ baseline: Pinnacle no-vig 收盤線 ｜ 生成於 {today}_")
    lines.append("")

    # 資料健康度
    lines.append("## 資料健康度")
    lines.append(f"- 輸入 summary.md：{n_input}")
    lines.append(f"- 通過解析：{n_parse_ok}（剔出 parse_failed {n_input - n_parse_ok}）")
    lines.append(f"- 通過 closing snapshot 匹配：{n_closing_ok}（剔出 closing_missing {n_input - n_closing_ok}）")
    lines.append(f"- 通過 result 取得：{n_result_ok}（剔出 result_missing {n_input - n_result_ok}）")
    lines.append(f"- **有效樣本：{n_valid} 場**")
    lines.append("")

    # TL;DR
    lines.append("## TL;DR")
    edge_str = _fmt_pct(direction_metrics.get("edge_pp"), 1) if direction_metrics.get("edge_pp") is not None else "—"
    lines.append(f"- 方向命中率：skill {_fmt_pct(direction_metrics.get('skill_hit_rate'))} vs market {_fmt_pct(direction_metrics.get('market_hit_rate'))}，edge {edge_str}")
    lines.append(f"- 總分 MAE：{_fmt_num(total_metrics.get('total_mae'))}，bias {_fmt_num(total_metrics.get('total_bias'))}")
    lines.append(f"- 反市場時：skill 命中 {_fmt_pct(direction_metrics.get('skill_against_hit_rate'))}（n={direction_metrics.get('skill_against_n')}）")
    lines.append(f"- Brier: skill {_fmt_num(calibration.get('brier_score'))} vs market {_fmt_num(calibration.get('brier_baseline_market'))}")
    lines.append("")

    # 方向類
    lines.append("## 1. 方向類指標")
    lines.append("")
    lines.append("| 指標 | 值 | n | 95% CI |")
    lines.append("|---|---|---|---|")
    sci = direction_metrics.get("skill_ci")
    mci = direction_metrics.get("market_ci")
    lines.append(f"| skill 命中率 | {_fmt_pct(direction_metrics.get('skill_hit_rate'))} | {direction_metrics.get('skill_n')} | "
                 f"{_fmt_pct(sci[0]) if sci else '—'} – {_fmt_pct(sci[1]) if sci else '—'} |")
    lines.append(f"| market favorite 命中率 | {_fmt_pct(direction_metrics.get('market_hit_rate'))} | {direction_metrics.get('market_n')} | "
                 f"{_fmt_pct(mci[0]) if mci else '—'} – {_fmt_pct(mci[1]) if mci else '—'} |")
    lines.append(f"| **skill edge (pp)** | {edge_str} | — | — |")
    lines.append(f"| skill 同意市場時命中率 | {_fmt_pct(direction_metrics.get('skill_aligned_hit_rate'))} | {direction_metrics.get('skill_aligned_n')} | — |")
    lines.append(f"| skill 反市場時命中率 | {_fmt_pct(direction_metrics.get('skill_against_hit_rate'))} | {direction_metrics.get('skill_against_n')} | — |")
    lines.append("")

    # 總分類
    lines.append("## 2. 總分類指標")
    lines.append("")
    lines.append("| 指標 | 值 | n |")
    lines.append("|---|---|---|")
    lines.append(f"| MAE | {_fmt_num(total_metrics.get('total_mae'))} | {total_metrics.get('n')} |")
    lines.append(f"| bias (skill − actual) | {_fmt_num(total_metrics.get('total_bias'))} | {total_metrics.get('n')} |")
    lines.append(f"| Over/Under 命中率 | {_fmt_pct(total_metrics.get('ou_hit_rate'))} | {total_metrics.get('ou_n')} |")
    lines.append("")

    # Calibration
    lines.append("## 3. 信心 Calibration")
    lines.append("")
    lines.append("| 信心 | n | mapped p | 實際命中率 | 落差 | CI |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in calibration["reliability_table"].iterrows():
        ci_str = f"{_fmt_pct(r['ci_low'])} – {_fmt_pct(r['ci_high'])}" if r.get("ci_low") is not None else "—"
        lines.append(f"| {r['confidence']} | {r['n']} | {_fmt_num(r['mapped_p'])} | "
                     f"{_fmt_pct(r['actual_hit_rate'])} | {_fmt_pct(r['delta'])} | {ci_str} |")
    lines.append("")
    lines.append(f"- Brier (skill / market): {_fmt_num(calibration.get('brier_score'))} / "
                 f"{_fmt_num(calibration.get('brier_baseline_market'))}")
    lines.append(f"- log-loss (skill / market): {_fmt_num(calibration.get('log_loss'))} / "
                 f"{_fmt_num(calibration.get('log_loss_baseline_market'))}")
    lines.append("")
    lines.append("<!-- 結論待人工填 -->")
    lines.append("")

    # 切片
    lines.append("## 4. 分組切片")
    lines.append("")
    lines.append("| 切片 | n | dir_hit | ou_hit | mae | bias |")
    lines.append("|---|---|---|---|---|---|")
    for idx, r in slices.iterrows():
        lines.append(f"| {idx} | {r['n']} | {_fmt_pct(r['dir_hit'])} | {_fmt_pct(r['ou_hit'])} | "
                     f"{_fmt_num(r['mae'])} | {_fmt_num(r['bias'])} |")
    lines.append("")

    # 失敗案例
    lines.append("## 5. 失敗案例")
    lines.append("")
    if len(failure_cases) == 0:
        lines.append("_無_")
    else:
        lines.append("| date | matchup | skill 方向 (信心) | 實際勝方 | skill total | 實際 total | 主訊號 | dossier |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in failure_cases.iterrows():
            dual = " ⚠️" if r.get("dual_failure") else ""
            dossier = r.get("dossier_path", "")
            sig = r.get("main_signal", "")
            lines.append(
                f"| {r['date']} | {r['matchup']}{dual} | {r['skill_direction']} ({r['skill_confidence']}) | "
                f"{r['actual_winner']} | {_fmt_num(r['skill_total'])} | {_fmt_num(r['actual_total'], 0)} | "
                f"{sig} | [link]({dossier}) |"
            )
    lines.append("")

    # 結論 stub
    lines.append("## 6. 結論與下一步")
    lines.append("")
    lines.append("<!-- 結論待人工填 -->")
    lines.append("")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: 跑 test 確認通過**

```bash
python -m pytest scripts/tests/test_render.py -v
```

Expected: PASS — 2 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/render.py scripts/tests/test_render.py
git commit -m "feat(backtest): render — Markdown 報告 + CSV 輸出

依 spec §6 報告骨架 + §7 26-column CSV.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: `scripts/backtest.py` — CLI 入口 + E2E smoke

**Files:**
- Create: `scripts/backtest.py`
- Create: `scripts/tests/test_backtest_e2e.py`

- [ ] **Step 1: 寫 e2e smoke test**

`scripts/tests/test_backtest_e2e.py`:

```python
"""End-to-end smoke test: run backtest on one day, verify outputs exist."""
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent.parent


def test_backtest_run_single_day(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = subprocess.run(
        [
            sys.executable, "scripts/backtest.py", "run",
            "--month", "2026-05",
            "--days", "2026-05-02",
            "--out", str(out_dir),
        ],
        cwd=SKILL_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (out_dir / "2026-05-report.md").exists()
    assert (out_dir / "2026-05-details.csv").exists()
    report = (out_dir / "2026-05-report.md").read_text(encoding="utf-8")
    assert "## 資料健康度" in report
    csv_text = (out_dir / "2026-05-details.csv").read_text(encoding="utf-8")
    header = csv_text.splitlines()[0]
    assert "date" in header and "matchup" in header
    assert len(csv_text.splitlines()) > 1  # at least one data row
```

- [ ] **Step 2: 跑 test 確認失敗**

```bash
python -m pytest scripts/tests/test_backtest_e2e.py -v
```

Expected: FAIL — `scripts/backtest.py: No such file or directory` or similar

- [ ] **Step 3: 寫 `scripts/backtest.py`**

```python
#!/usr/bin/env python3
"""MLB Skill Backtest — entry point.

用法：
  python scripts/backtest.py run --month 2026-05
  python scripts/backtest.py run --month 2026-05 --days 2026-05-02,2026-05-03
  python scripts/backtest.py run --month 2026-05 --out /tmp/out
"""

import argparse
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.load import build_dataframe_for_month
from lib.metrics import (
    compute_direction_metrics, compute_total_metrics,
    compute_calibration, compute_slice_metrics,
)
from lib.diagnostic import select_failure_cases, extract_main_signal
from lib.render import render_report, render_details_csv


def _attach_main_signal(failure_cases, df):
    """For each failure row, read summary.md and extract main signal."""
    if len(failure_cases) == 0:
        failure_cases["main_signal"] = []
        return failure_cases
    signals = []
    for _, r in failure_cases.iterrows():
        summary_path = SKILL_ROOT / "analysis-data" / r["date"] / r["matchup"] / "summary.md"
        if summary_path.exists():
            signals.append(extract_main_signal(summary_path.read_text(encoding="utf-8")))
        else:
            signals.append("")
    failure_cases["main_signal"] = signals
    return failure_cases


def cmd_run(args):
    days_filter = set(args.days.split(",")) if args.days else None
    out_dir = Path(args.out) if args.out else SKILL_ROOT / "analysis-data" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for month={args.month}, days={days_filter or 'all'}...")
    df = build_dataframe_for_month(month=args.month, days_filter=days_filter)
    print(f"Loaded {len(df)} rows.")

    print("Computing metrics...")
    dm = compute_direction_metrics(df)
    tm = compute_total_metrics(df)
    cal = compute_calibration(df)
    slices = compute_slice_metrics(df)

    print("Selecting failure cases...")
    failures = select_failure_cases(df, top_total_miss=10)
    failures = _attach_main_signal(failures, df)

    print("Rendering...")
    report_path = out_dir / f"{args.month}-report.md"
    csv_path = out_dir / f"{args.month}-details.csv"
    render_report(df=df, direction_metrics=dm, total_metrics=tm,
                  calibration=cal, slices=slices, failure_cases=failures,
                  month=args.month, out_path=report_path)
    render_details_csv(df, out_path=csv_path)

    print(f"Report: {report_path}")
    print(f"CSV:    {csv_path}")
    print(f"Valid:  {((~df['parse_failed']) & (~df['closing_missing']) & (~df['result_missing'])).sum()} / {len(df)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--month", required=True, help="YYYY-MM")
    p_run.add_argument("--days", help="comma-separated YYYY-MM-DD, optional")
    p_run.add_argument("--out", help="output directory (default: analysis-data/backtest/)")
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑 e2e test 確認通過**

```bash
python -m pytest scripts/tests/test_backtest_e2e.py -v
```

Expected: PASS — 1 passed

- [ ] **Step 5: 跑全套測試確認沒打壞別的**

```bash
python -m pytest scripts/tests/test_fetch_results.py scripts/tests/test_parse_summary.py scripts/tests/test_closing_line.py scripts/tests/test_load.py scripts/tests/test_metrics.py scripts/tests/test_diagnostic.py scripts/tests/test_render.py scripts/tests/test_backtest_e2e.py -v
```

Expected: PASS — all green (合計 ~30 cases)

- [ ] **Step 6: Commit**

```bash
git add scripts/backtest.py scripts/tests/test_backtest_e2e.py
git commit -m "feat(backtest): backtest.py 入口 + e2e smoke

argparse subcommand 'run' 串起 load → metrics → diagnostic → render.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: 跑全 5 月、檢視產出、commit 報告

**Files:**
- Create: `analysis-data/backtest/2026-05-report.md`
- Create: `analysis-data/backtest/2026-05-details.csv`

- [ ] **Step 1: 跑全月**

```bash
python scripts/backtest.py run --month 2026-05
```

Expected stdout:
```
Loading data for month=2026-05, days=all...
Loaded ~271 rows.
Computing metrics...
Selecting failure cases...
Rendering...
Report: analysis-data/backtest/2026-05-report.md
CSV:    analysis-data/backtest/2026-05-details.csv
Valid:  ~100 / ~271
```

「Valid」是通過 parse + closing + result 三關的場次，預期接近 109（finished summaries）但會被 closing_missing / result_missing 進一步剔出，可能落在 90-105 之間。

- [ ] **Step 2: 開報告檢視**

```bash
cat analysis-data/backtest/2026-05-report.md | head -60
```

確認：
- 資料健康度數字看起來合理（n_valid > 80）
- TL;DR 有實際數字（不是 `—`）
- 方向 / 總分 / Calibration 三表都有值
- 失敗案例至少有幾條

如果某段大量出現 `—`（missing），檢查 parse_failed / closing_missing / result_missing 是否異常高。常見問題：
- `closing_missing > 50%`：odds snapshot 命名 / 隊伍名對不上 → 看 `closing_line.py` 邏輯
- `parse_failed > 50%`：summary.md 「整體判斷」欄式樣 parse 邏輯有 bug → 看 `parse_summary.py`

- [ ] **Step 3: 開 CSV 用 pandas 看健康度**

```bash
python -c "
import pandas as pd
df = pd.read_csv('analysis-data/backtest/2026-05-details.csv')
print('Total rows:', len(df))
print('parse_failed True:', df['parse_failed'].sum())
print('closing_missing True:', df['closing_missing'].sum())
print('result_missing True:', df['result_missing'].sum())
print('valid:', ((~df['parse_failed']) & (~df['closing_missing']) & (~df['result_missing'])).sum())
print()
print('skill_direction value counts:')
print(df['skill_direction'].value_counts(dropna=False))
print()
print('skill_confidence value counts:')
print(df['skill_confidence'].value_counts(dropna=False))
"
```

- [ ] **Step 4: 視覺檢查報告完整性**

人眼開 `analysis-data/backtest/2026-05-report.md` 過一次，確認每個 section 都有實際內容（不是 stub）。「結論待人工填」段是 expected stub。

- [ ] **Step 5: Commit**

```bash
git add analysis-data/backtest/
git commit -m "data(backtest): 2026-05 軌 A 回測首次跑 — report + details

樣本 N 場有效, baseline Pinnacle no-vig 收盤. 詳見
analysis-data/backtest/2026-05-report.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Checklist (寫完 plan 後檢查)

1. **Spec coverage**:
   - §3 Pipeline 5 stages ↔ Tasks 1-2 (Stage 1), 4-6 (Stage 2), 7 (Stage 3), 8 (Stage 4), 9 (Stage 5) ✓
   - §4.1-4.4 所有指標 ↔ Task 7 metrics.py ✓
   - §5 失敗案例 ↔ Task 8 diagnostic.py ✓
   - §6 報告骨架 + §7 CSV ↔ Task 9 render.py ✓
   - §8 邊界處理 ↔ load.py 的 parse_failed/closing_missing/result_missing flags ✓
   - §9 檔案組織 ↔ File Structure 章節 ✓
   - §10 測試 ↔ 每 task 都有 TDD ✓

2. **Placeholder scan**：完整 code、exact paths、specific commands. 無 TBD / "implement later"。

3. **Type consistency**：
   - `parse_summary` 簽名：`(path, home_team_abbr, away_team_abbr)` — 全 plan 一致 ✓
   - `find_closing_snapshot_for_game` 簽名：`(snapshots_dir, date, home_team, away_team)` — 全 plan 一致 ✓
   - `CONFIDENCE_TO_PROB` mapping `{"LOW": 0.55, "MEDIUM": 0.62, "HIGH": 0.72}` — 在 load.py 與 metrics.py 一致 ✓
   - DataFrame 欄位名 (`skill_direction`, `market_favorite`, `parse_failed` 等) — 全 plan 一致 ✓

4. **Cross-task dependencies**：
   - Task 2 依 Task 1 (fetch_results 完成才能跑 data)
   - Task 6 依 Task 1+2 (load 用 result.json) + Task 3+4+5 (parse_summary, closing_line)
   - Task 7 依 Task 6
   - Task 10 依所有 lib/* 都完成
   - Task 11 依 Task 10
   - Plan 任務順序符合 ✓
