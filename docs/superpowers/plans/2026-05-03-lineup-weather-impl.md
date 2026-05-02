# Official Lineup + Weather 整合 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `mlb-game-analyzer` skill 的 pipeline 中，自動優先採用球隊公布的當日打序（feed/live battingOrder），無資料時 silent fallback 至現行 PA proxy；同步在 merge 階段補抓 `gameData.weather`，無資料時跳過天氣分析。

**Architecture:** 兩條改動共用 MLB Stats API 的 `feed/live` 端點。`lineup_analyzer.py` 加 `fetch_official_lineup` + `analyze_team` 分支；`merge_game_data.py` 加 `fetch_weather` 補欄到 `merged.json`；`dossier_renderer.py` / `summary_renderer.py` 增 source 標記與 weather 三狀態渲染。失敗永遠 fallback，不 abort pipeline。`scoring_formula.py` 與 Flag 體系完全不動。

**Tech Stack:** Python 3.11+、`requests`、`pytest`（既有）、MLB Stats API v1.1。

**Spec reference:** `docs/superpowers/specs/2026-05-02-lineup-weather-design.md`

---

## File Structure

**新增**：
- `scripts/tests/fixtures/feed_live_official_lineup.json` — 完整 9 人 + sunny 天氣
- `scripts/tests/fixtures/feed_live_partial_lineup.json` — battingOrder 5 人
- `scripts/tests/fixtures/feed_live_empty_lineup.json` — battingOrder = [] 且 weather = {}
- `scripts/tests/fixtures/feed_live_indoor.json` — condition = "Roof Closed"
- `scripts/tests/fixtures/feed_live_weather_only.json` — battingOrder = [] 但 weather 有值

**修改**：
- `scripts/lineup_analyzer.py` — 新 `fetch_official_lineup`、`build_lineup_from_official`、`build_lineup_from_pa_proxy`（既有邏輯抽出）、`_select_lineup_9` helper、`analyze_team` 重構、CLI `--game-pk`
- `scripts/prepare_game.py` — `step_a` ids 加 `game_pk`、`step_d` cmd 加 `--game-pk`
- `scripts/merge_game_data.py` — 新 `fetch_weather`、`main()` 加 `merged["weather"]`
- `scripts/dossier_renderer.py` — `_render_lineup_overview` 分支、新 `_render_full9_vs_pitcher` helper、`_render_top5_vs_pitcher` helper（從現有 inline 抽出）、weather row 加進 header section
- `scripts/summary_renderer.py` — `_render_lineup_section` source 標記、`_render_conditional_section` weather 三狀態
- `scripts/tests/test_lineup_analyzer.py` — 新測試 9 case
- `scripts/tests/test_merge_game_data.py` — 新測試 7 case
- `scripts/tests/test_dossier_renderer.py` — 新測試 7 case
- `scripts/tests/test_summary_renderer.py` — 新測試 4 case
- `scripts/tests/test_prepare_game_steps.py` — 既有測試擴充 2 case
- `reference/matchup-factors.md` — `## 打線分析` 段首加 source 說明、`## 球場 & 天氣` 加 `### 天氣修正`
- `SKILL.md` — Quick Reference 第 1 步描述、新增「條件式資料」段、步驟 1 ℹ️ 補一條、步驟 2.3 補「天氣」

**不動**：`scoring_formula.py`、`flags-checklist.md`、`fetch_game_data.py`、`pitcher_stats.py`、`roster_checker.py`、`park_factors_lib.py`

---

## Test Execution

從 `scripts/` 目錄跑：
```bash
cd scripts
python -m pytest tests/ -v
```

或單檔：
```bash
python -m pytest tests/test_lineup_analyzer.py -v
```

---

## Task 1: Fixtures

**Files:**
- Create: `scripts/tests/fixtures/feed_live_official_lineup.json`
- Create: `scripts/tests/fixtures/feed_live_partial_lineup.json`
- Create: `scripts/tests/fixtures/feed_live_empty_lineup.json`
- Create: `scripts/tests/fixtures/feed_live_indoor.json`
- Create: `scripts/tests/fixtures/feed_live_weather_only.json`

- [ ] **Step 1: 建立 fixtures 目錄並寫 5 個 fixture**

`scripts/tests/fixtures/feed_live_official_lineup.json`:
```json
{
  "gameData": {
    "weather": {"condition": "Sunny", "temp": "78", "wind": "10 mph, Out To CF"}
  },
  "liveData": {
    "boxscore": {
      "teams": {
        "home": {
          "team": {"id": 147},
          "battingOrder": [592450, 519317, 624413, 519203, 670541, 543305, 596019, 624577, 656555],
          "players": {
            "ID592450": {"position": {"abbreviation": "DH"}},
            "ID519317": {"position": {"abbreviation": "RF"}},
            "ID624413": {"position": {"abbreviation": "1B"}},
            "ID519203": {"position": {"abbreviation": "LF"}},
            "ID670541": {"position": {"abbreviation": "C"}},
            "ID543305": {"position": {"abbreviation": "SS"}},
            "ID596019": {"position": {"abbreviation": "3B"}},
            "ID624577": {"position": {"abbreviation": "2B"}},
            "ID656555": {"position": {"abbreviation": "CF"}}
          }
        },
        "away": {
          "team": {"id": 110},
          "battingOrder": [],
          "players": {}
        }
      }
    }
  }
}
```

`scripts/tests/fixtures/feed_live_partial_lineup.json`:
```json
{
  "gameData": {"weather": {"condition": "Cloudy", "temp": "65", "wind": "5 mph, L To R"}},
  "liveData": {
    "boxscore": {
      "teams": {
        "home": {
          "team": {"id": 147},
          "battingOrder": [592450, 519317, 624413, 519203, 670541],
          "players": {}
        },
        "away": {"team": {"id": 110}, "battingOrder": [], "players": {}}
      }
    }
  }
}
```

`scripts/tests/fixtures/feed_live_empty_lineup.json`:
```json
{
  "gameData": {"weather": {}},
  "liveData": {
    "boxscore": {
      "teams": {
        "home": {"team": {"id": 147}, "battingOrder": [], "players": {}},
        "away": {"team": {"id": 110}, "battingOrder": [], "players": {}}
      }
    }
  }
}
```

`scripts/tests/fixtures/feed_live_indoor.json`:
```json
{
  "gameData": {"weather": {"condition": "Roof Closed", "temp": "72", "wind": "0 mph, None"}},
  "liveData": {
    "boxscore": {
      "teams": {
        "home": {"team": {"id": 141}, "battingOrder": [], "players": {}},
        "away": {"team": {"id": 110}, "battingOrder": [], "players": {}}
      }
    }
  }
}
```

`scripts/tests/fixtures/feed_live_weather_only.json`:
```json
{
  "gameData": {"weather": {"condition": "Partly Cloudy", "temp": "82", "wind": "12 mph, In From RF"}},
  "liveData": {
    "boxscore": {
      "teams": {
        "home": {"team": {"id": 147}, "battingOrder": [], "players": {}},
        "away": {"team": {"id": 110}, "battingOrder": [], "players": {}}
      }
    }
  }
}
```

- [ ] **Step 2: 驗證 fixtures 是合法 JSON**

Run:
```bash
cd /c/Users/USER/.agents/skills/mlb-game-analyzer
python -c "import json, glob; [json.loads(open(p, encoding='utf-8').read()) for p in glob.glob('scripts/tests/fixtures/feed_live_*.json')]; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/tests/fixtures/
git commit -m "test(fixtures): feed/live 5 變體 (lineup/weather 整合用)"
```

---

## Task 2: `fetch_official_lineup` function

**Files:**
- Modify: `scripts/lineup_analyzer.py`
- Test: `scripts/tests/test_lineup_analyzer.py`

- [ ] **Step 1: 加 fixture loader helper（test 共用）**

在 `scripts/tests/test_lineup_analyzer.py` 檔尾加（既有 import 區塊保留）：

```python
import json
from pathlib import Path
from unittest.mock import MagicMock


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _mock_requests_get(fixture: dict):
    """Build a MagicMock that emulates requests.get returning fixture JSON."""
    resp = MagicMock()
    resp.json.return_value = fixture
    resp.raise_for_status.return_value = None
    return MagicMock(return_value=resp)
```

- [ ] **Step 2: 寫 5 個 failing test**

加到 `scripts/tests/test_lineup_analyzer.py`：

```python
def test_fetch_official_lineup_full(monkeypatch):
    """完整 9 人 → 回傳 list[int] 長度 9，順序保留。"""
    fixture = _load_fixture("feed_live_official_lineup.json")
    monkeypatch.setattr("lineup_analyzer.requests.get", _mock_requests_get(fixture))

    from lineup_analyzer import fetch_official_lineup
    result = fetch_official_lineup(game_pk=778345, team_id=147)
    assert result == [592450, 519317, 624413, 519203, 670541, 543305, 596019, 624577, 656555]


def test_fetch_official_lineup_partial(monkeypatch):
    """5 人 → 直接回傳 list[int] 長度 5（caller 決定 fallback）。"""
    fixture = _load_fixture("feed_live_partial_lineup.json")
    monkeypatch.setattr("lineup_analyzer.requests.get", _mock_requests_get(fixture))

    from lineup_analyzer import fetch_official_lineup
    result = fetch_official_lineup(game_pk=778345, team_id=147)
    assert result == [592450, 519317, 624413, 519203, 670541]


def test_fetch_official_lineup_empty(monkeypatch):
    """battingOrder=[] → 回傳空 list（不是 None）。"""
    fixture = _load_fixture("feed_live_empty_lineup.json")
    monkeypatch.setattr("lineup_analyzer.requests.get", _mock_requests_get(fixture))

    from lineup_analyzer import fetch_official_lineup
    result = fetch_official_lineup(game_pk=778345, team_id=147)
    assert result == []


def test_fetch_official_lineup_team_not_found(monkeypatch, capsys):
    """team_id 不在 home/away → 回 None + stderr 警告。"""
    fixture = _load_fixture("feed_live_official_lineup.json")
    monkeypatch.setattr("lineup_analyzer.requests.get", _mock_requests_get(fixture))

    from lineup_analyzer import fetch_official_lineup
    result = fetch_official_lineup(game_pk=778345, team_id=999)  # 不在 fixture 內
    assert result is None
    captured = capsys.readouterr()
    assert "team_id 999 not in boxscore" in captured.err


def test_fetch_official_lineup_api_fail(monkeypatch, capsys):
    """requests.get 拋例外 → 回 None + stderr 警告。"""
    def _raise(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("lineup_analyzer.requests.get", _raise)

    from lineup_analyzer import fetch_official_lineup
    result = fetch_official_lineup(game_pk=778345, team_id=147)
    assert result is None
    captured = capsys.readouterr()
    assert "feed/live fetch failed" in captured.err
```

- [ ] **Step 3: 跑測試確認 5 個都失敗**

Run:
```bash
cd scripts
python -m pytest tests/test_lineup_analyzer.py::test_fetch_official_lineup_full tests/test_lineup_analyzer.py::test_fetch_official_lineup_partial tests/test_lineup_analyzer.py::test_fetch_official_lineup_empty tests/test_lineup_analyzer.py::test_fetch_official_lineup_team_not_found tests/test_lineup_analyzer.py::test_fetch_official_lineup_api_fail -v
```
Expected: 5 FAIL（`ImportError: cannot import name 'fetch_official_lineup'`）

- [ ] **Step 4: 在 `lineup_analyzer.py` 實作 `fetch_official_lineup`**

加在 `MLB_API_BASE` 常數之後、`fetch_team_roster` 之前（讓相關函式集中在頂部）：

```python
def fetch_official_lineup(game_pk: int, team_id: int) -> list[int] | None:
    """從 feed/live 取該隊公布打序的 player_id list（按 1-9 棒順序）。

    回傳：
      - list[int] 長度 9：官方公布完整打序
      - list[int] 長度 0~8：部分公布（caller 自行決定 fallback）
      - None：API 失敗 / team_id 不在 boxscore

    side 自動判斷：比對 boxscore.teams.{home|away}.team.id。
    """
    try:
        resp = requests.get(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        boxscore = data.get("liveData", {}).get("boxscore", {})
        for side in ("home", "away"):
            t = boxscore.get("teams", {}).get(side, {})
            if t.get("team", {}).get("id") == team_id:
                return list(t.get("battingOrder", []))
        print(
            f"[lineup_analyzer] team_id {team_id} not in boxscore (game_pk={game_pk})",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(f"[lineup_analyzer] feed/live fetch failed: {e}", file=sys.stderr)
        return None
```

- [ ] **Step 5: 跑測試確認 5 個全 pass**

Run（同 Step 3 命令）。
Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/lineup_analyzer.py scripts/tests/test_lineup_analyzer.py
git commit -m "feat(lineup): fetch_official_lineup 從 feed/live 取打序"
```

---

## Task 3: `analyze_team` 重構（抽 helper + official 分支）

**Files:**
- Modify: `scripts/lineup_analyzer.py`
- Test: `scripts/tests/test_lineup_analyzer.py`

**重點**：保持 `analyze_team` 既有 API 不破壞、現行回傳 schema 全保留並新增 2 欄。

- [ ] **Step 1: 寫 4 個 failing test（official path / partial fallback / no game_pk / API fail fallback）**

在 `test_lineup_analyzer.py` 加：

```python
def test_analyze_team_official_path(monkeypatch):
    """game_pk + 完整 9 人 → lineup_source=official，9 人含 batting_order=1..9。"""
    fixture = _load_fixture("feed_live_official_lineup.json")
    monkeypatch.setattr("lineup_analyzer.requests.get", _mock_requests_get(fixture))

    # Stub: roster + per-player batting + statcast
    monkeypatch.setattr(
        "lineup_analyzer.fetch_team_roster",
        lambda team_id, year: [
            {"id": pid, "name": f"P{pid}", "position": "DH"}
            for pid in [592450, 519317, 624413, 519203, 670541, 543305, 596019, 624577, 656555]
        ],
    )
    monkeypatch.setattr(
        "lineup_analyzer.fetch_player_batting",
        lambda pid, year: {
            "mlbam_id": pid, "pa": 100, "avg": 0.250, "obp": 0.330, "slg": 0.420,
            "ops": 0.750, "iso": 0.170, "babip": 0.300, "k_pct": 22.0, "bb_pct": 9.0,
        },
    )
    monkeypatch.setattr("lineup_analyzer.fetch_statcast_batting_leaderboard", lambda y: ({}, {}))
    monkeypatch.setattr("lineup_analyzer.fetch_player_platoon", lambda pid, y: None)
    monkeypatch.setattr("lineup_analyzer.fetch_player_last7", lambda pid: None)

    from lineup_analyzer import analyze_team
    result = analyze_team("NYY", 2026, opposing_pitcher_id=None, game_pk=778345)
    assert result["lineup_source"] == "official"
    assert len(result["lineup"]) == 9
    assert [b["batting_order"] for b in result["lineup"]] == list(range(1, 10))
    assert result["lineup_source_detail"]["game_pk"] == 778345
    assert "fetched_at" in result["lineup_source_detail"]


def test_analyze_team_partial_falls_back(monkeypatch, capsys):
    """5 人 → fallback projected + stderr。"""
    fixture = _load_fixture("feed_live_partial_lineup.json")
    monkeypatch.setattr("lineup_analyzer.requests.get", _mock_requests_get(fixture))

    monkeypatch.setattr(
        "lineup_analyzer.fetch_team_roster",
        lambda team_id, year: [{"id": 1, "name": "X", "position": "C"}],
    )
    monkeypatch.setattr("lineup_analyzer.fetch_il_names", lambda team_id, year: set())
    monkeypatch.setattr(
        "lineup_analyzer.fetch_player_batting",
        lambda pid, year: {
            "mlbam_id": pid, "pa": 100, "avg": 0.250, "obp": 0.330, "slg": 0.420,
            "ops": 0.750, "iso": 0.170, "babip": 0.300, "k_pct": 22.0, "bb_pct": 9.0,
        },
    )
    monkeypatch.setattr("lineup_analyzer.fetch_statcast_batting_leaderboard", lambda y: ({}, {}))
    monkeypatch.setattr("lineup_analyzer.fetch_player_platoon", lambda pid, y: None)
    monkeypatch.setattr("lineup_analyzer.fetch_player_last7", lambda pid: None)

    from lineup_analyzer import analyze_team
    result = analyze_team("NYY", 2026, opposing_pitcher_id=None, game_pk=778345)
    assert result["lineup_source"] == "projected"
    assert result["lineup_source_detail"] is None
    captured = capsys.readouterr()
    assert "official lineup partial" in captured.err


def test_analyze_team_no_game_pk(monkeypatch):
    """game_pk=None → 直接走 PA proxy，不打 feed/live。"""
    called = []
    def _get(*a, **k):
        called.append(a)
        raise AssertionError("requests.get should not be called when game_pk=None")
    monkeypatch.setattr("lineup_analyzer.requests.get", _get)

    monkeypatch.setattr(
        "lineup_analyzer.fetch_team_roster",
        lambda team_id, year: [{"id": 1, "name": "X", "position": "C"}],
    )
    monkeypatch.setattr("lineup_analyzer.fetch_il_names", lambda team_id, year: set())
    monkeypatch.setattr(
        "lineup_analyzer.fetch_player_batting",
        lambda pid, year: {
            "mlbam_id": pid, "pa": 100, "avg": 0.250, "obp": 0.330, "slg": 0.420,
            "ops": 0.750, "iso": 0.170, "babip": 0.300, "k_pct": 22.0, "bb_pct": 9.0,
        },
    )
    monkeypatch.setattr("lineup_analyzer.fetch_statcast_batting_leaderboard", lambda y: ({}, {}))
    monkeypatch.setattr("lineup_analyzer.fetch_player_platoon", lambda pid, y: None)
    monkeypatch.setattr("lineup_analyzer.fetch_player_last7", lambda pid: None)

    from lineup_analyzer import analyze_team
    result = analyze_team("NYY", 2026, opposing_pitcher_id=None, game_pk=None)
    assert result["lineup_source"] == "projected"
    # called list should be empty since requests.get raised assertion which is caught? No, AssertionError propagates if called.
    assert not called


def test_analyze_team_api_fail_falls_back(monkeypatch, capsys):
    """feed/live 失敗 → fallback projected。"""
    def _raise(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("lineup_analyzer.requests.get", _raise)

    monkeypatch.setattr(
        "lineup_analyzer.fetch_team_roster",
        lambda team_id, year: [{"id": 1, "name": "X", "position": "C"}],
    )
    monkeypatch.setattr("lineup_analyzer.fetch_il_names", lambda team_id, year: set())
    monkeypatch.setattr(
        "lineup_analyzer.fetch_player_batting",
        lambda pid, year: {
            "mlbam_id": pid, "pa": 100, "avg": 0.250, "obp": 0.330, "slg": 0.420,
            "ops": 0.750, "iso": 0.170, "babip": 0.300, "k_pct": 22.0, "bb_pct": 9.0,
        },
    )
    monkeypatch.setattr("lineup_analyzer.fetch_statcast_batting_leaderboard", lambda y: ({}, {}))
    monkeypatch.setattr("lineup_analyzer.fetch_player_platoon", lambda pid, y: None)
    monkeypatch.setattr("lineup_analyzer.fetch_player_last7", lambda pid: None)

    from lineup_analyzer import analyze_team
    result = analyze_team("NYY", 2026, opposing_pitcher_id=None, game_pk=778345)
    assert result["lineup_source"] == "projected"
    captured = capsys.readouterr()
    assert "feed/live fetch failed" in captured.err
```

- [ ] **Step 2: 跑測試確認 4 個都失敗**

Run:
```bash
cd scripts
python -m pytest tests/test_lineup_analyzer.py::test_analyze_team_official_path tests/test_lineup_analyzer.py::test_analyze_team_partial_falls_back tests/test_lineup_analyzer.py::test_analyze_team_no_game_pk tests/test_lineup_analyzer.py::test_analyze_team_api_fail_falls_back -v
```
Expected: 4 FAIL（`TypeError: analyze_team() got an unexpected keyword argument 'game_pk'` 或 `KeyError: 'lineup_source'`）

- [ ] **Step 3: 重構 `analyze_team` — 抽 PA proxy 為 helper、加 `_select_lineup_9`、加 official 分支**

整段替換 `lineup_analyzer.py` 既有的 `analyze_team`（約 304-430 行）為以下內容。新增 import：在頂部 `from datetime import datetime, timezone` 之外，已經有了；不用改 import。

```python
def build_lineup_from_pa_proxy(team_id: int, year: int) -> list[dict]:
    """既有 PA-排序 top 9 邏輯抽出。
    
    Roster → IL filter → fetch_player_batting → 按 PA 降序 → top 9
    """
    roster = fetch_team_roster(team_id, year)
    if not roster:
        return []
    il_names = fetch_il_names(team_id, year)

    batters = []
    for player in roster:
        if player["name"] in il_names:
            continue
        stats = fetch_player_batting(player["id"], year)
        if stats:
            stats["name"] = player["name"]
            stats["position"] = player["position"]
            stats["batting_order"] = None  # projected path 無實際打序
            batters.append(stats)

    batters.sort(key=lambda b: b["pa"], reverse=True)
    return batters[:9]


def build_lineup_from_official(official_ids: list[int], team_id: int, year: int) -> list[dict]:
    """用 official 9 個 player_id 直接組打線（不過濾 IL — 上場了就是上場）。
    
    Position 與 name 從 fetch_team_roster 拿；極少數查不到（剛升上來）就空字串。
    """
    roster = fetch_team_roster(team_id, year)
    pos_map = {p["id"]: p["position"] for p in roster}
    name_map = {p["id"]: p["name"] for p in roster}

    core_lineup = []
    for i, pid in enumerate(official_ids, start=1):
        stats = fetch_player_batting(pid, year)
        if not stats:
            # Rookie / 0 PA — 補骨架，下游聚合會忽略 0 值
            stats = {
                "mlbam_id": pid, "pa": 0, "avg": 0.0, "obp": 0.0, "slg": 0.0,
                "ops": 0.0, "iso": 0.0, "babip": 0.0, "k_pct": 0.0, "bb_pct": 0.0,
            }
        stats["name"] = name_map.get(pid, f"Player {pid}")
        stats["position"] = pos_map.get(pid, "")
        stats["batting_order"] = i
        core_lineup.append(stats)
    return core_lineup


def _select_lineup_9(team_id: int, year: int, game_pk: int | None):
    """Try official → fallback PA proxy. 回傳 (core_lineup, source, detail)。"""
    if game_pk:
        official_ids = fetch_official_lineup(game_pk, team_id)
        if official_ids is not None:
            if len(official_ids) == 9:
                print("[lineup_analyzer] official lineup fetched (9)", file=sys.stderr)
                core = build_lineup_from_official(official_ids, team_id, year)
                detail = {
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "game_pk": game_pk,
                }
                return core, "official", detail
            elif len(official_ids) == 0:
                print(
                    "[lineup_analyzer] official lineup not yet posted, fallback to PA proxy",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[lineup_analyzer] official lineup partial (N={len(official_ids)}), fallback to PA proxy",
                    file=sys.stderr,
                )
        # official_ids is None：fetch_official_lineup 已自行 stderr 警告
    return build_lineup_from_pa_proxy(team_id, year), "projected", None


def analyze_team(team: str, year: int, opposing_pitcher_id: int | None = None,
                 game_pk: int | None = None) -> dict:
    """完整的球隊打線分析。
    
    若 game_pk 提供且球隊已公布完整 9 人打序，採 official 路徑；
    否則 fallback 至 PA proxy（active roster 排除 IL，按 PA 降序前 9）。
    """
    team_id = resolve_team_id(team)

    core_lineup, lineup_source, lineup_source_detail = _select_lineup_9(team_id, year, game_pk)

    if not core_lineup:
        return {"error": f"No active roster found for {team}"}

    # 3. Statcast leaderboard（一次拉全聯盟，記憶體內 merge）
    expected_map, barrels_map = fetch_statcast_batting_leaderboard(year)

    # 4. Merge Statcast + Platoon + Last7 + BvP
    for batter in core_lineup:
        pid = str(batter["mlbam_id"])
        exp = expected_map.get(pid, {})
        bar = barrels_map.get(pid, {})
        batter["xwoba"] = exp.get("xwoba")
        batter["xba"] = exp.get("xba")
        batter["xslg"] = exp.get("xslg")
        batter["ev95pct"] = bar.get("ev95pct")
        batter["barrel_pct"] = bar.get("barrel_pct")
        batter["platoon"] = fetch_player_platoon(batter["mlbam_id"], year)
        batter["last_7"] = fetch_player_last7(batter["mlbam_id"])
        if opposing_pitcher_id:
            batter["bvp"] = fetch_bvp(batter["mlbam_id"], opposing_pitcher_id)

    # 5. 整體指標（既有，不變）
    avg_ops = sum(b["ops"] for b in core_lineup) / len(core_lineup)
    avg_babip = sum(b["babip"] for b in core_lineup) / len(core_lineup)
    avg_k_pct = sum(b["k_pct"] for b in core_lineup) / len(core_lineup)
    avg_bb_pct = sum(b["bb_pct"] for b in core_lineup) / len(core_lineup)

    xwoba_values = [b["xwoba"] for b in core_lineup if b.get("xwoba") is not None]
    avg_xwoba = sum(xwoba_values) / len(xwoba_values) if xwoba_values else None

    # 6. 打線評級（既有）
    tier = "🟢 Weak"
    if avg_xwoba is not None:
        for tier_name, check_fn in TIER_MAP:
            if check_fn(avg_xwoba):
                tier = tier_name
                break
    else:
        for tier_name, check_fn in TIER_MAP_OPS:
            if check_fn(avg_ops):
                tier = tier_name
                break

    # 7. 大小分傾向（既有）
    over_under_lean = 0
    if avg_babip <= 0.270:
        over_under_lean += 1
    if avg_babip >= 0.320:
        over_under_lean -= 1
    if avg_k_pct >= 25:
        over_under_lean -= 1
    if avg_xwoba is not None and avg_xwoba >= 0.350:
        over_under_lean += 1

    # 8. 串聯分析（既有）
    chain = {}
    if len(core_lineup) >= 3:
        chain["obp_top3"] = round(sum(b["obp"] for b in core_lineup[:3]) / 3, 3)
    if len(core_lineup) >= 5:
        chain["slg_mid"] = round(sum(b["slg"] for b in core_lineup[3:5]) / 2, 3)

    # 整體近 7 場熱度（既有）
    last7_ops_values = []
    for b in core_lineup:
        if b.get("last_7") and b["last_7"].get("ops"):
            try:
                last7_ops_values.append(float(b["last_7"]["ops"]))
            except (ValueError, TypeError):
                pass
    recent_heat = None
    if last7_ops_values:
        avg_last7_ops = sum(last7_ops_values) / len(last7_ops_values)
        if avg_last7_ops >= 0.830:
            recent_heat = "🔥 Hot"
        elif avg_last7_ops <= 0.600:
            recent_heat = "🥶 Cold"
        else:
            recent_heat = "⚖️ Normal"

    return {
        "team": team,
        "team_id": team_id,
        "tier": tier,
        "avg_ops": round(avg_ops, 3),
        "avg_xwoba": round(avg_xwoba, 3) if avg_xwoba else None,
        "avg_babip": round(avg_babip, 3),
        "avg_k_pct": round(avg_k_pct, 1),
        "avg_bb_pct": round(avg_bb_pct, 1),
        "over_under_lean": over_under_lean,
        "recent_heat": recent_heat,
        "last7_babip": compute_last7_babip(core_lineup),
        "chain": chain,
        "lineup_source": lineup_source,
        "lineup_source_detail": lineup_source_detail,
        "lineup": core_lineup,
    }
```

- [ ] **Step 4: 跑 4 個 analyze_team 測試確認 pass**

Run（同 Step 2 命令）。
Expected: 4 PASS

- [ ] **Step 5: 跑全部 lineup_analyzer 測試確認既有不破壞**

Run:
```bash
cd scripts
python -m pytest tests/test_lineup_analyzer.py -v
```
Expected: ALL PASS（既有 5 個 + Task 2 新加 5 個 + 本 Task 4 個 = 14 個）

- [ ] **Step 6: Commit**

```bash
git add scripts/lineup_analyzer.py scripts/tests/test_lineup_analyzer.py
git commit -m "feat(lineup): analyze_team 加 game_pk 分支 + 抽 helper"
```

---

## Task 4: lineup_analyzer CLI `--game-pk` 參數

**Files:**
- Modify: `scripts/lineup_analyzer.py`（main 函式）

- [ ] **Step 1: 改 `main()`**

定位 `lineup_analyzer.py` 的 `main()` 函式（檔末），找到 argparse 區塊。在 `--opposing-pitcher-id` 行下方加：

```python
    parser.add_argument("--game-pk", type=int, default=None,
                        help="MLB Stats API gamePk for fetching official lineup (optional)")
```

接著 `analyze_team` 呼叫從：
```python
    result = analyze_team(args.team, args.year, args.opposing_pitcher_id)
```
改為：
```python
    result = analyze_team(args.team, args.year, args.opposing_pitcher_id, game_pk=args.game_pk)
```

並把 `command` 字串 build 也加 game_pk：
```python
            command = f"lineup_analyzer.py --team {args.team} --year {args.year}" + (
                f" --opposing-pitcher-id {args.opposing_pitcher_id}" if args.opposing_pitcher_id else ""
            ) + (
                f" --game-pk {args.game_pk}" if args.game_pk else ""
            )
```

- [ ] **Step 2: 手動驗證 CLI**

Run:
```bash
cd scripts
python lineup_analyzer.py --help | grep -A1 "game-pk"
```
Expected: 看到 `--game-pk GAME_PK` 出現於 help 文字。

- [ ] **Step 3: Commit**

```bash
git add scripts/lineup_analyzer.py
git commit -m "feat(lineup): CLI --game-pk 參數"
```

---

## Task 5: `prepare_game.py` step_a 回傳 game_pk

**Files:**
- Modify: `scripts/prepare_game.py`（step_a 函式）
- Test: `scripts/tests/test_prepare_game_steps.py`

- [ ] **Step 1: 寫 failing test**

加到 `scripts/tests/test_prepare_game_steps.py`（檔內既有 `test_step_a_extracts_pitcher_ids` 之後）：

```python
def test_step_a_returns_game_pk(monkeypatch, tmp_path):
    """step_a 回傳值應包含 game_pk（從 game_data.json 既有的 gamePk 讀出）。"""
    from prepare_game import step_a

    game_data_path = tmp_path / "game_data.json"
    game_data_path.write_text(json.dumps({
        "_meta": {},
        "gamePk": 778345,
        "home": {"team": "CLE", "team_id": 114, "probable_pitcher": "Tanner Bibee", "probable_pitcher_id": 676440},
        "away": {"team": "TB", "team_id": 139, "probable_pitcher": "Nick Martínez", "probable_pitcher_id": 607259},
    }), encoding="utf-8")

    monkeypatch.setattr("prepare_game.subprocess.run", make_fake_run())
    result = step_a(date="2026-04-28", team_abbr="TB", output_dir=tmp_path)
    assert result["game_pk"] == 778345
```

- [ ] **Step 2: 跑測試確認失敗**

Run:
```bash
cd scripts
python -m pytest tests/test_prepare_game_steps.py::test_step_a_returns_game_pk -v
```
Expected: FAIL（`KeyError: 'game_pk'`）

- [ ] **Step 3: 改 `step_a`（在檔案約 161-168 行的 return 區塊）**

定位現有 return：
```python
    return {
        "home_id": home.get("probable_pitcher_id"),
        "away_id": away.get("probable_pitcher_id"),
        "home_name": home.get("probable_pitcher"),
        "away_name": away.get("probable_pitcher"),
        "home_team_id": home.get("team_id"),
        "away_team_id": away.get("team_id"),
    }
```

改為：
```python
    return {
        "home_id": home.get("probable_pitcher_id"),
        "away_id": away.get("probable_pitcher_id"),
        "home_name": home.get("probable_pitcher"),
        "away_name": away.get("probable_pitcher"),
        "home_team_id": home.get("team_id"),
        "away_team_id": away.get("team_id"),
        "game_pk": game_section.get("gamePk"),
    }
```

- [ ] **Step 4: 跑測試確認 pass**

Run（同 Step 2 命令）。
Expected: PASS

- [ ] **Step 5: 跑全部 test_prepare_game_steps 確認既有不破壞**

Run:
```bash
cd scripts
python -m pytest tests/test_prepare_game_steps.py -v
```
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_game.py scripts/tests/test_prepare_game_steps.py
git commit -m "feat(prepare): step_a 回傳值加 game_pk"
```

---

## Task 6: `prepare_game.py` step_d 傳遞 `--game-pk`

**Files:**
- Modify: `scripts/prepare_game.py`（step_d 函式 + main 呼叫處）
- Test: `scripts/tests/test_prepare_game_steps.py`

- [ ] **Step 1: 寫 failing test**

```python
def test_step_d_passes_game_pk(monkeypatch, tmp_path):
    """step_d 接到 game_pk 時，cmd 內必含 --game-pk。"""
    from prepare_game import step_d

    captured_cmds = []

    def fake_run(*args, **kwargs):
        captured_cmds.append(args[0])
        return FakeResult(returncode=0)

    monkeypatch.setattr("prepare_game.subprocess.run", fake_run)
    step_d(home="CLE", away="TB", home_id=676440, away_id=607259,
           season=2026, output_dir=tmp_path, game_pk=778345)

    # 兩次呼叫（home + away）每個 cmd 都該帶 --game-pk
    assert len(captured_cmds) == 2
    for cmd in captured_cmds:
        assert "--game-pk" in cmd
        idx = cmd.index("--game-pk")
        assert cmd[idx + 1] == "778345"


def test_step_d_no_game_pk_omits_arg(monkeypatch, tmp_path):
    """step_d 接到 game_pk=None 時，cmd 內不出現 --game-pk。"""
    from prepare_game import step_d

    captured_cmds = []

    def fake_run(*args, **kwargs):
        captured_cmds.append(args[0])
        return FakeResult(returncode=0)

    monkeypatch.setattr("prepare_game.subprocess.run", fake_run)
    step_d(home="CLE", away="TB", home_id=676440, away_id=607259,
           season=2026, output_dir=tmp_path, game_pk=None)

    for cmd in captured_cmds:
        assert "--game-pk" not in cmd
```

- [ ] **Step 2: 跑測試確認失敗**

Run:
```bash
cd scripts
python -m pytest tests/test_prepare_game_steps.py::test_step_d_passes_game_pk tests/test_prepare_game_steps.py::test_step_d_no_game_pk_omits_arg -v
```
Expected: 2 FAIL（`TypeError: step_d() got an unexpected keyword argument 'game_pk'`）

- [ ] **Step 3: 改 `step_d` signature + 內部 cmd build**

定位 `prepare_game.py` 的 `step_d`（約 273-316 行）。signature 改為：

```python
def step_d(*, home: str, away: str,
           home_id: int | None, away_id: int | None,
           season: int, output_dir: Path,
           game_pk: int | None = None) -> None:
```

`_run_side` 內 cmd build 段：

```python
    def _run_side(side_tuple):
        side, team, opposing_id, out_path = side_tuple
        cmd = [
            PYTHON,
            str(SCRIPT_DIR / "lineup_analyzer.py"),
            "--team", team,
            "--year", str(season),
            "-o", str(out_path),
        ]
        if opposing_id:
            cmd += ["--opposing-pitcher-id", str(opposing_id)]
        if game_pk:
            cmd += ["--game-pk", str(game_pk)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        except FileNotFoundError as e:
            return side, -1, "", str(e)
        return side, result.returncode, result.stdout, result.stderr
```

定位 `main()` 內呼叫 `step_d` 處（約 494-500 行），改為：

```python
    step_d(
        home=args.home, away=args.away,
        home_id=ids["home_id"],
        away_id=ids["away_id"],
        season=args.season,
        output_dir=output_dir,
        game_pk=ids.get("game_pk"),
    )
```

- [ ] **Step 4: 跑測試確認 pass**

Run（同 Step 2 命令）。
Expected: 2 PASS

- [ ] **Step 5: 跑全部 prepare_game_steps 確認既有不破壞**

Run:
```bash
cd scripts
python -m pytest tests/test_prepare_game_steps.py -v
```
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_game.py scripts/tests/test_prepare_game_steps.py
git commit -m "feat(prepare): step_d 傳 --game-pk 給 lineup_analyzer"
```

---

## Task 7: `fetch_weather` function

**Files:**
- Modify: `scripts/merge_game_data.py`
- Test: `scripts/tests/test_merge_game_data.py`

- [ ] **Step 1: 加 fixture loader 與 mock helper（test 共用）**

在 `scripts/tests/test_merge_game_data.py` 檔頭 import 區之後加：

```python
import json
from pathlib import Path
from unittest.mock import MagicMock

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _mock_requests_get(fixture: dict):
    resp = MagicMock()
    resp.json.return_value = fixture
    resp.raise_for_status.return_value = None
    return MagicMock(return_value=resp)
```

（如果該檔已有同名 helper，就跳過此 step。）

- [ ] **Step 2: 寫 5 個 failing test**

```python
def test_fetch_weather_full(monkeypatch):
    """三欄齊 → 回傳 dict，indoor=False。"""
    fixture = _load_fixture("feed_live_official_lineup.json")  # weather=Sunny/78/wind
    monkeypatch.setattr("merge_game_data.requests.get", _mock_requests_get(fixture))

    from merge_game_data import fetch_weather
    result = fetch_weather(game_pk=778345)
    assert result == {
        "condition": "Sunny",
        "temp_f": 78,
        "wind_text": "10 mph, Out To CF",
        "indoor": False,
    }


def test_fetch_weather_indoor(monkeypatch):
    """condition='Roof Closed' → indoor=True。"""
    fixture = _load_fixture("feed_live_indoor.json")
    monkeypatch.setattr("merge_game_data.requests.get", _mock_requests_get(fixture))

    from merge_game_data import fetch_weather
    result = fetch_weather(game_pk=778345)
    assert result["indoor"] is True
    assert result["condition"] == "Roof Closed"
    assert result["temp_f"] == 72


def test_fetch_weather_empty(monkeypatch):
    """weather={} → 回傳 None。"""
    fixture = _load_fixture("feed_live_empty_lineup.json")
    monkeypatch.setattr("merge_game_data.requests.get", _mock_requests_get(fixture))

    from merge_game_data import fetch_weather
    assert fetch_weather(game_pk=778345) is None


def test_fetch_weather_partial(monkeypatch):
    """只有 condition、缺 wind/temp → 回傳 dict，缺欄為 None。"""
    fixture = {"gameData": {"weather": {"condition": "Cloudy", "temp": "", "wind": ""}}}
    monkeypatch.setattr("merge_game_data.requests.get", _mock_requests_get(fixture))

    from merge_game_data import fetch_weather
    result = fetch_weather(game_pk=778345)
    assert result == {"condition": "Cloudy", "temp_f": None, "wind_text": None, "indoor": False}


def test_fetch_weather_api_fail(monkeypatch, capsys):
    """API 失敗 → 回 None + stderr 警告。"""
    def _raise(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("merge_game_data.requests.get", _raise)

    from merge_game_data import fetch_weather
    assert fetch_weather(game_pk=778345) is None
    captured = capsys.readouterr()
    assert "weather fetch failed" in captured.err
```

- [ ] **Step 3: 跑測試確認失敗**

Run:
```bash
cd scripts
python -m pytest tests/test_merge_game_data.py -k "fetch_weather" -v
```
Expected: 5 FAIL（`ImportError: cannot import name 'fetch_weather'`）

- [ ] **Step 4: 在 `merge_game_data.py` 實作 `fetch_weather`**

加在 `fetch_bullpen_era` 函式之後（約 170 行附近）：

```python
def fetch_weather(game_pk: int) -> dict | None:
    """從 feed/live 取 gameData.weather。

    回傳：
      - dict：{condition, temp_f, wind_text, indoor}
      - None：API 失敗 / weather 欄位不存在或全空
    """
    try:
        resp = requests.get(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
            timeout=10,
        )
        resp.raise_for_status()
        w = resp.json().get("gameData", {}).get("weather", {}) or {}
        condition = (w.get("condition") or "").strip()
        temp = (w.get("temp") or "").strip()
        wind = (w.get("wind") or "").strip()

        if not condition and not temp and not wind:
            return None

        indoor = condition.lower() in ("roof closed", "dome")

        try:
            temp_f = int(temp) if temp else None
        except ValueError:
            temp_f = None

        return {
            "condition": condition or None,
            "temp_f": temp_f,
            "wind_text": wind or None,
            "indoor": indoor,
        }
    except Exception as e:
        print(f"[merge_game_data] weather fetch failed: {e}", file=sys.stderr)
        return None
```

- [ ] **Step 5: 跑測試確認 5 個 pass**

Run（同 Step 3 命令）。
Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/merge_game_data.py scripts/tests/test_merge_game_data.py
git commit -m "feat(merge): fetch_weather 從 feed/live 取天氣"
```

---

## Task 8: merge_game_data 整合 weather 到 merged.json

**Files:**
- Modify: `scripts/merge_game_data.py`（main 函式）
- Test: `scripts/tests/test_merge_game_data.py`

- [ ] **Step 1: 寫 2 個 failing test**

```python
def test_merged_weather_present(monkeypatch, tmp_path):
    """end-to-end mock：weather API 回完整 → merged['weather'] dict 帶 4 欄。"""
    fixture = _load_fixture("feed_live_official_lineup.json")
    monkeypatch.setattr("merge_game_data.requests.get", _mock_requests_get(fixture))
    # bullpen ERA fetch 也會打 API；簡化用同 fixture（API 不對等但回傳會被 fetch_bullpen_era 視為 4.00 fallback）
    # 不重要：本測試只看 weather

    # 準備最小可運行的 input JSON
    game_data = {
        "game": {
            "gamePk": 778345,
            "date": "2026-04-30T23:00:00Z",
            "venue": "Yankee Stadium",
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "X", "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "Y", "probable_pitcher_id": 2},
        },
        "home_recent": {}, "away_recent": {},
        "home_recent_30": {}, "away_recent_30": {},
        "home_season": {}, "away_season": {},
        "home_season_games_count": 0, "away_season_games_count": 0,
    }
    home_pitcher = {"name": "X", "season": {"era": 4.0}}
    away_pitcher = {"name": "Y", "season": {"era": 4.0}}
    home_lineup = {"avg_xwoba": 0.315, "avg_ops": 0.710, "avg_k_pct": 22.0,
                   "lineup_source": "official", "lineup_source_detail": {"game_pk": 778345}}
    away_lineup = {"avg_xwoba": 0.315, "avg_ops": 0.710, "avg_k_pct": 22.0,
                   "lineup_source": "projected", "lineup_source_detail": None}

    g_path = tmp_path / "g.json"; g_path.write_text(json.dumps(game_data), encoding="utf-8")
    hp = tmp_path / "hp.json"; hp.write_text(json.dumps(home_pitcher), encoding="utf-8")
    ap = tmp_path / "ap.json"; ap.write_text(json.dumps(away_pitcher), encoding="utf-8")
    hl = tmp_path / "hl.json"; hl.write_text(json.dumps(home_lineup), encoding="utf-8")
    al = tmp_path / "al.json"; al.write_text(json.dumps(away_lineup), encoding="utf-8")
    out = tmp_path / "merged.json"

    import sys as _sys
    _sys.argv = ["merge_game_data.py", "--game", str(g_path),
                 "--home-pitcher", str(hp), "--away-pitcher", str(ap),
                 "--home-lineup", str(hl), "--away-lineup", str(al),
                 "-o", str(out), "--no-md",
                 "--park-factor", "100",
                 "--home-bullpen-era", "4.0", "--away-bullpen-era", "4.0"]
    from merge_game_data import main
    main()

    merged = json.loads(out.read_text(encoding="utf-8"))
    assert merged["weather"] == {
        "condition": "Sunny",
        "temp_f": 78,
        "wind_text": "10 mph, Out To CF",
        "indoor": False,
    }


def test_merged_weather_absent(monkeypatch, tmp_path):
    """weather 欄位全空 → merged['weather'] = None。"""
    fixture = _load_fixture("feed_live_empty_lineup.json")
    monkeypatch.setattr("merge_game_data.requests.get", _mock_requests_get(fixture))

    game_data = {
        "game": {
            "gamePk": 778345,
            "date": "2026-04-30T23:00:00Z",
            "venue": "Yankee Stadium",
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "X", "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "Y", "probable_pitcher_id": 2},
        },
        "home_recent": {}, "away_recent": {},
        "home_recent_30": {}, "away_recent_30": {},
        "home_season": {}, "away_season": {},
        "home_season_games_count": 0, "away_season_games_count": 0,
    }
    home_pitcher = {"name": "X", "season": {"era": 4.0}}
    away_pitcher = {"name": "Y", "season": {"era": 4.0}}
    home_lineup = {"avg_xwoba": 0.315, "avg_ops": 0.710, "avg_k_pct": 22.0}
    away_lineup = {"avg_xwoba": 0.315, "avg_ops": 0.710, "avg_k_pct": 22.0}

    g_path = tmp_path / "g.json"; g_path.write_text(json.dumps(game_data), encoding="utf-8")
    hp = tmp_path / "hp.json"; hp.write_text(json.dumps(home_pitcher), encoding="utf-8")
    ap = tmp_path / "ap.json"; ap.write_text(json.dumps(away_pitcher), encoding="utf-8")
    hl = tmp_path / "hl.json"; hl.write_text(json.dumps(home_lineup), encoding="utf-8")
    al = tmp_path / "al.json"; al.write_text(json.dumps(away_lineup), encoding="utf-8")
    out = tmp_path / "merged.json"

    import sys as _sys
    _sys.argv = ["merge_game_data.py", "--game", str(g_path),
                 "--home-pitcher", str(hp), "--away-pitcher", str(ap),
                 "--home-lineup", str(hl), "--away-lineup", str(al),
                 "-o", str(out), "--no-md",
                 "--park-factor", "100",
                 "--home-bullpen-era", "4.0", "--away-bullpen-era", "4.0"]
    from merge_game_data import main
    main()

    merged = json.loads(out.read_text(encoding="utf-8"))
    assert merged["weather"] is None
```

- [ ] **Step 2: 跑測試確認失敗**

Run:
```bash
cd scripts
python -m pytest tests/test_merge_game_data.py::test_merged_weather_present tests/test_merge_game_data.py::test_merged_weather_absent -v
```
Expected: 2 FAIL（`KeyError: 'weather'`）

- [ ] **Step 3: 改 `merge_game_data.main()`**

定位 `main()` 內 `merged["park_factor"] = park_factor` 那一行（約 399 行附近）。在它之後加：

```python
    # weather（與 park_factor 同層級的條件修正資料；無資料則 None，AI 在 summary 跳過）
    merged["weather"] = fetch_weather(game_info.get("gamePk")) if game_info.get("gamePk") else None
```

- [ ] **Step 4: 跑測試確認 pass**

Run（同 Step 2 命令）。
Expected: 2 PASS

- [ ] **Step 5: 跑全部 test_merge_game_data 確認既有不破壞**

Run:
```bash
cd scripts
python -m pytest tests/test_merge_game_data.py -v
```
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/merge_game_data.py scripts/tests/test_merge_game_data.py
git commit -m "feat(merge): merged.weather 整合到 main 流程"
```

---

## Task 9: dossier_renderer — lineup source label + 9 棒 vs 對方先發

**Files:**
- Modify: `scripts/dossier_renderer.py`
- Test: `scripts/tests/test_dossier_renderer.py`

- [ ] **Step 1: 讀現行 lineup section 邏輯**

```bash
cd /c/Users/USER/.agents/skills/mlb-game-analyzer
sed -n '520,650p' scripts/dossier_renderer.py
```

留意 `_render_lineup_overview` 周圍的 helper（`_render_top5_block` 等）。

- [ ] **Step 2: 寫 4 個 failing test**

加到 `scripts/tests/test_dossier_renderer.py`：

```python
def _make_lineup(source="projected", batters=None):
    if batters is None:
        batters = [
            {"mlbam_id": 100 + i, "name": f"P{i}", "position": "DH",
             "pa": 200 - i * 10, "avg": 0.250, "obp": 0.330, "slg": 0.420,
             "ops": 0.750, "iso": 0.170, "babip": 0.300, "k_pct": 22.0, "bb_pct": 9.0,
             "xwoba": 0.330, "xba": 0.250, "xslg": 0.420,
             "ev95pct": 50.0, "barrel_pct": 8.0,
             "platoon": None, "last_7": None, "bvp": None,
             "batting_order": (i + 1) if source == "official" else None}
            for i in range(9)
        ]
    return {
        "team": "NYY", "team_id": 147, "tier": "🟡 Average",
        "avg_ops": 0.750, "avg_xwoba": 0.330, "avg_babip": 0.300,
        "avg_k_pct": 22.0, "avg_bb_pct": 9.0, "over_under_lean": 0,
        "recent_heat": "⚖️ Normal", "last7_babip": 0.300, "chain": {},
        "lineup_source": source, "lineup_source_detail": None, "lineup": batters,
    }


def test_dossier_lineup_section_official():
    """home/away 都 official → 標題出現「打線來源：🟢 official」、9 棒 vs 對方先發 table。"""
    from dossier_renderer import render_dossier
    bundle = {
        "game_data": {"game": {
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "HP",
                     "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "AP",
                     "probable_pitcher_id": 2},
            "venue": "Yankee Stadium",
            "officialDate": "2026-04-30",
            "date": "2026-04-30T23:00:00Z",
        }},
        "home_lineup": _make_lineup("official"),
        "away_lineup": _make_lineup("official"),
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "season": {}},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "season": {}},
        "merged": {"park_factor": 100, "home_bullpen_era": 4.0, "away_bullpen_era": 4.0},
    }
    md = render_dossier(bundle, game_dir="/tmp", summary_filename="summary.md")
    assert "🟢 official" in md
    # 9 棒 vs 對方先發應出現
    assert "9 棒 vs" in md or "1-9 棒 vs" in md or "All 9 vs" in md  # 依實作命名


def test_dossier_lineup_section_projected():
    """home/away 都 projected → 標題「🟡 projected」、Top 5 sub-block 維持。"""
    from dossier_renderer import render_dossier
    bundle = {
        "game_data": {"game": {
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "HP",
                     "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "AP",
                     "probable_pitcher_id": 2},
            "venue": "Yankee Stadium",
            "officialDate": "2026-04-30",
            "date": "2026-04-30T23:00:00Z",
        }},
        "home_lineup": _make_lineup("projected"),
        "away_lineup": _make_lineup("projected"),
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "season": {}},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "season": {}},
        "merged": {"park_factor": 100, "home_bullpen_era": 4.0, "away_bullpen_era": 4.0},
    }
    md = render_dossier(bundle, game_dir="/tmp", summary_filename="summary.md")
    assert "🟡 projected" in md
    # 既有 Top 5 sub-block 標題仍在
    assert "Top 5" in md or "PA top" in md or "對方先發" in md  # 依現行命名


def test_dossier_lineup_section_no_source_field():
    """缺 lineup_source（舊 merged.json） → 預設 projected，向下相容。"""
    from dossier_renderer import render_dossier
    home_l = _make_lineup("projected")
    home_l.pop("lineup_source")
    home_l.pop("lineup_source_detail")
    bundle = {
        "game_data": {"game": {
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "HP",
                     "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "AP",
                     "probable_pitcher_id": 2},
            "venue": "Yankee Stadium",
            "officialDate": "2026-04-30",
            "date": "2026-04-30T23:00:00Z",
        }},
        "home_lineup": home_l,
        "away_lineup": _make_lineup("projected"),
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "season": {}},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "season": {}},
        "merged": {"park_factor": 100, "home_bullpen_era": 4.0, "away_bullpen_era": 4.0},
    }
    # 不該 raise KeyError
    md = render_dossier(bundle, game_dir="/tmp", summary_filename="summary.md")
    assert "🟡 projected" in md
```

- [ ] **Step 3: 跑測試確認失敗**

Run:
```bash
cd scripts
python -m pytest tests/test_dossier_renderer.py::test_dossier_lineup_section_official tests/test_dossier_renderer.py::test_dossier_lineup_section_projected tests/test_dossier_renderer.py::test_dossier_lineup_section_no_source_field -v
```
Expected: 3 FAIL（assertion 找不到 `🟢 official` / `🟡 projected`）

- [ ] **Step 4: 改 `_render_lineup_overview` 加 source label + 分支**

定位 `_render_lineup_overview`（檔約 555 行起）。在現有打線 table 之前插入 source 標記，並把現有「Top 5 vs 對方先發」段抽出為 helper、新增 `_render_full9_vs_pitcher` helper。

實作骨架（具體 inline 邏輯依現行檔案調整）：

```python
def _source_label(source: str) -> str:
    if source == "official":
        return "🟢 official"
    return "🟡 projected（PA 排序近似 — 打線尚未公布）"


def _render_full9_vs_pitcher(lineup: dict, opposing_hand: str, opposing_pitcher_name: str,
                             team_label: str) -> list[str]:
    """All-9-batter table vs 對方先發（用於 official 路徑）。
    
    每棒按 batting_order 排序，顯示 vs LHP/RHP OPS、Last 7 OPS、BvP（PA≥15 才標）。
    """
    lines = [
        f"### {team_label} 1–9 棒 vs {opposing_pitcher_name} ({opposing_hand}HP)",
        "",
        "| # | Name | Pos | PA | vs OPS | Last7 OPS | BvP (PA / OPS) |",
        "|---|------|-----|----|--------|-----------|----------------|",
    ]
    batters = lineup.get("lineup", []) or []
    for b in batters:
        order = b.get("batting_order") or "?"
        name = b.get("name", "?")
        pos = b.get("position", "")
        pa = b.get("pa", "—")
        vs_ops = _lineup_vs_hand_ops(b, opposing_hand)
        last7 = (b.get("last_7") or {}).get("ops") or "—"
        bvp = b.get("bvp") or {}
        bvp_str = (
            f"{bvp.get('pa')} / {bvp.get('ops')}"
            if bvp and bvp.get("sample_sufficient") else "—"
        )
        lines.append(f"| {order} | {name} | {pos} | {pa} | {vs_ops} | {last7} | {bvp_str} |")
    lines.append("")
    return lines


def _render_lineup_overview(bundle: dict) -> list[str]:
    home_lu = bundle.get("home_lineup") or {}
    away_lu = bundle.get("away_lineup") or {}
    home_source = home_lu.get("lineup_source", "projected")
    away_source = away_lu.get("lineup_source", "projected")

    # 取對方先發資訊（既有邏輯）
    home_p = bundle.get("home_pitcher") or {}
    away_p = bundle.get("away_pitcher") or {}
    home_p_hand = (home_p.get("pitch_hand") or "R").upper()
    away_p_hand = (away_p.get("pitch_hand") or "R").upper()

    # IL names（既有）
    home_il = _il_names_from_roster(bundle.get("home_roster"))
    away_il = _il_names_from_roster(bundle.get("away_roster"))

    lines = ["## 打線", ""]
    lines.append(f"- HOME 打線來源：{_source_label(home_source)}")
    lines.append(f"- AWAY 打線來源：{_source_label(away_source)}")
    lines.append("")

    # 既有的整體 9 人 table（保留）
    lines += _render_lineup_team_block(home_lu, "HOME", ...)  # 依現行函式名整合
    lines += _render_lineup_team_block(away_lu, "AWAY", ...)

    # vs 對方先發 sub-block 分支
    if home_source == "official":
        lines += _render_full9_vs_pitcher(home_lu, away_p_hand, away_p.get("name", "?"), "HOME")
    else:
        lines += _render_top5_vs_pitcher(home_lu, home_il, away_p_hand, away_p.get("name", "?"), "HOME")

    if away_source == "official":
        lines += _render_full9_vs_pitcher(away_lu, home_p_hand, home_p.get("name", "?"), "AWAY")
    else:
        lines += _render_top5_vs_pitcher(away_lu, away_il, home_p_hand, home_p.get("name", "?"), "AWAY")

    return lines
```

> **Note**：`_render_top5_vs_pitcher` 是把現行 `_render_lineup_overview` 內的 inline Top 5 邏輯抽出（用 `select_top5_vs_pitcher` + `find_last7_top1_outside_pa_top5`，現有 helper 不動）。函式 signature 須與被抽前的呼叫端對齊。`_render_lineup_team_block` 是現行整體 9 人 table 渲染段的抽取（如果原本是 inline，建議抽出讓 home/away 共用）。

實作順序建議：
1. 先實作 `_source_label`、`_render_full9_vs_pitcher`、`_render_lineup_team_block`
2. 再修改 `_render_lineup_overview` 加 source 標、改用新 helper
3. 把現有 inline Top 5 邏輯抽進 `_render_top5_vs_pitcher`

- [ ] **Step 5: 跑測試確認 3 個 pass**

Run（同 Step 3 命令）。
Expected: 3 PASS

- [ ] **Step 6: 跑全部 dossier renderer 測試確認既有不破壞**

Run:
```bash
cd scripts
python -m pytest tests/test_dossier_renderer.py -v
```
Expected: ALL PASS（既有測試使用的 fixture 若無 `lineup_source` 欄位，會被預設成 `projected`）

- [ ] **Step 7: Commit**

```bash
git add scripts/dossier_renderer.py scripts/tests/test_dossier_renderer.py
git commit -m "feat(dossier): lineup source 標記 + official 路徑全 9 棒 vs 對方先發"
```

---

## Task 10: dossier_renderer — weather row

**Files:**
- Modify: `scripts/dossier_renderer.py`
- Test: `scripts/tests/test_dossier_renderer.py`

- [ ] **Step 1: 寫 3 個 failing test**

```python
def test_dossier_weather_row_present():
    """merged.weather 三欄齊 → dossier 出現 weather row。"""
    from dossier_renderer import render_dossier
    bundle = {
        "game_data": {"game": {
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "HP",
                     "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "AP",
                     "probable_pitcher_id": 2},
            "venue": "Yankee Stadium",
            "officialDate": "2026-04-30",
            "date": "2026-04-30T23:00:00Z",
        }},
        "home_lineup": _make_lineup("projected"),
        "away_lineup": _make_lineup("projected"),
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "season": {}},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "season": {}},
        "merged": {
            "park_factor": 100, "home_bullpen_era": 4.0, "away_bullpen_era": 4.0,
            "weather": {"condition": "Sunny", "temp_f": 78,
                        "wind_text": "10 mph, Out To CF", "indoor": False},
        },
    }
    md = render_dossier(bundle, game_dir="/tmp", summary_filename="summary.md")
    assert "Sunny" in md and "78°F" in md and "Out To CF" in md


def test_dossier_weather_row_indoor():
    """indoor=True → 顯示「室內（Roof Closed，不適用天氣分析）」。"""
    from dossier_renderer import render_dossier
    bundle = {
        "game_data": {"game": {
            "home": {"team": "TOR", "team_id": 141, "probable_pitcher": "HP",
                     "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "AP",
                     "probable_pitcher_id": 2},
            "venue": "Rogers Centre",
            "officialDate": "2026-04-30",
            "date": "2026-04-30T23:00:00Z",
        }},
        "home_lineup": _make_lineup("projected"),
        "away_lineup": _make_lineup("projected"),
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "season": {}},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "season": {}},
        "merged": {
            "park_factor": 100, "home_bullpen_era": 4.0, "away_bullpen_era": 4.0,
            "weather": {"condition": "Roof Closed", "temp_f": 72,
                        "wind_text": None, "indoor": True},
        },
    }
    md = render_dossier(bundle, game_dir="/tmp", summary_filename="summary.md")
    assert "室內" in md
    assert "不適用天氣分析" in md


def test_dossier_weather_row_absent():
    """merged.weather=None → 整行省略（dossier 不應有 'weather:' / '室內' / '未公布' 字樣）。"""
    from dossier_renderer import render_dossier
    bundle = {
        "game_data": {"game": {
            "home": {"team": "NYY", "team_id": 147, "probable_pitcher": "HP",
                     "probable_pitcher_id": 1},
            "away": {"team": "BOS", "team_id": 110, "probable_pitcher": "AP",
                     "probable_pitcher_id": 2},
            "venue": "Yankee Stadium",
            "officialDate": "2026-04-30",
            "date": "2026-04-30T23:00:00Z",
        }},
        "home_lineup": _make_lineup("projected"),
        "away_lineup": _make_lineup("projected"),
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "season": {}},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "season": {}},
        "merged": {
            "park_factor": 100, "home_bullpen_era": 4.0, "away_bullpen_era": 4.0,
            "weather": None,
        },
    }
    md = render_dossier(bundle, game_dir="/tmp", summary_filename="summary.md")
    # weather 整行不應出現
    assert "**weather**:" not in md
    assert "室內" not in md
    assert "未公布" not in md
```

- [ ] **Step 2: 跑測試確認失敗**

Run:
```bash
cd scripts
python -m pytest tests/test_dossier_renderer.py::test_dossier_weather_row_present tests/test_dossier_renderer.py::test_dossier_weather_row_indoor tests/test_dossier_renderer.py::test_dossier_weather_row_absent -v
```
Expected: 3 FAIL（找不到 weather 字樣）

- [ ] **Step 3: 在 dossier header section 加 weather row**

定位 dossier header（含 venue / park_factor 那段）。在 venue line 之後加邏輯：

```python
def _render_weather_line(weather: dict | None) -> str | None:
    """Return formatted weather line, or None if absent (caller skips line)."""
    if not weather:
        return None
    if weather.get("indoor"):
        cond = weather.get("condition", "Indoor")
        return f"**weather**: 室內（{cond}，不適用天氣分析）"
    parts = []
    if weather.get("condition"):
        parts.append(weather["condition"])
    if weather.get("temp_f") is not None:
        parts.append(f"{weather['temp_f']}°F")
    if weather.get("wind_text"):
        parts.append(f"wind {weather['wind_text']}")
    if not parts:
        return None
    return f"**weather**: {', '.join(parts)}"
```

整合進 header 渲染：

```python
# 既有：
# lines.append(f"**venue**: {venue} | **park_factor (runs)**: {pf}")

weather_line = _render_weather_line(merged.get("weather"))
if weather_line:
    lines.append(weather_line)
```

> 具體插入位置依現行 dossier header 段。

- [ ] **Step 4: 跑測試確認 3 個 pass**

Run（同 Step 2 命令）。
Expected: 3 PASS

- [ ] **Step 5: 跑全部 dossier 測試確認既有不破壞**

Run:
```bash
cd scripts
python -m pytest tests/test_dossier_renderer.py -v
```
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/dossier_renderer.py scripts/tests/test_dossier_renderer.py
git commit -m "feat(dossier): weather row（三狀態，缺資料整行省略）"
```

---

## Task 11: summary_renderer — `## 打線評級` 加 source 標

**Files:**
- Modify: `scripts/summary_renderer.py`
- Test: `scripts/tests/test_summary_renderer.py`

- [ ] **Step 1: 寫 1 個 failing test**

加到 `scripts/tests/test_summary_renderer.py`：

```python
def test_summary_lineup_section_marks_source():
    """home=official → summary `## 打線評級` HOME 段含「打線來源：🟢 official」。"""
    from summary_renderer import render_summary

    bundle = {
        "home_lineup": {"tier": "🟡 Average", "recent_heat": "⚖️ Normal",
                        "lineup_source": "official"},
        "away_lineup": {"tier": "🟡 Average", "recent_heat": "⚖️ Normal",
                        "lineup_source": "projected"},
        "home_pitcher": {"name": "HP", "pitch_hand": "R", "age": 28},
        "away_pitcher": {"name": "AP", "pitch_hand": "R", "age": 30},
        "merged": {"park_factor": 100,
                   "home_bullpen_era": 4.0, "away_bullpen_era": 4.0},
        "home_roster": None, "away_roster": None,
    }
    formula_pred = {"home_score": 4.5, "away_score": 4.0}

    md = render_summary(bundle, formula_pred)
    # HOME 段含 official 標記
    assert "🟢 official" in md
    # AWAY 段含 projected 標記
    assert "🟡 projected" in md
```

- [ ] **Step 2: 跑測試確認失敗**

Run:
```bash
cd scripts
python -m pytest tests/test_summary_renderer.py::test_summary_lineup_section_marks_source -v
```
Expected: FAIL

- [ ] **Step 3: 改 `_render_lineup_section`**

定位 `summary_renderer.py` 約 46-60 行的 `_render_lineup_section`。改為：

```python
def _render_lineup_section(bundle: dict) -> list[str]:
    home_l = bundle.get("home_lineup") or {}
    away_l = bundle.get("away_lineup") or {}
    home_source = home_l.get("lineup_source", "projected")
    away_source = away_l.get("lineup_source", "projected")

    def _label(src):
        return "🟢 official" if src == "official" else "🟡 projected（PA 排序近似 — 打線尚未公布）"

    return [
        "## 打線評級",
        "",
        f"### HOME — {home_l.get('tier', '?')} / {home_l.get('recent_heat', '?')}",
        f"- 打線來源：{_label(home_source)}",
        "- **Tier 覆寫**：<!-- AI 補 -->",
        "",
        f"### AWAY — {away_l.get('tier', '?')} / {away_l.get('recent_heat', '?')}",
        f"- 打線來源：{_label(away_source)}",
        "- **Tier 覆寫**：<!-- AI 補 -->",
        "",
    ]
```

- [ ] **Step 4: 跑測試確認 pass**

Run（同 Step 2 命令）。
Expected: PASS

- [ ] **Step 5: 跑全部 summary_renderer 測試**

Run:
```bash
cd scripts
python -m pytest tests/test_summary_renderer.py -v
```
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/summary_renderer.py scripts/tests/test_summary_renderer.py
git commit -m "feat(summary): ## 打線評級 加 lineup_source 標記"
```

---

## Task 12: summary_renderer — `## 條件修正` weather 三狀態

**Files:**
- Modify: `scripts/summary_renderer.py`
- Test: `scripts/tests/test_summary_renderer.py`

- [ ] **Step 1: 寫 3 個 failing test**

```python
def test_summary_conditional_weather_present():
    """merged.weather 三欄齊 → ## 條件修正 出現「天氣：Sunny, 78°F, ...」+ AI 影響判讀 placeholder。"""
    from summary_renderer import render_summary

    bundle = {
        "home_lineup": {"tier": "?", "recent_heat": "?", "lineup_source": "projected"},
        "away_lineup": {"tier": "?", "recent_heat": "?", "lineup_source": "projected"},
        "home_pitcher": {"name": "?", "pitch_hand": "R"}, "away_pitcher": {"name": "?", "pitch_hand": "R"},
        "merged": {
            "park_factor": 100,
            "weather": {"condition": "Sunny", "temp_f": 78,
                        "wind_text": "10 mph, Out To CF", "indoor": False},
        },
        "home_roster": None, "away_roster": None,
    }
    md = render_summary(bundle, {"home_score": 0, "away_score": 0})
    assert "天氣：Sunny, 78°F, wind 10 mph, Out To CF" in md
    # AI 影響判讀 placeholder 應在
    assert "AI 補：對得分 / HR 影響判讀" in md or "影響判讀" in md


def test_summary_conditional_weather_indoor():
    """indoor=True → 顯示「天氣：室內（Roof Closed，不適用）」、不出現 AI placeholder。"""
    from summary_renderer import render_summary

    bundle = {
        "home_lineup": {"tier": "?", "recent_heat": "?", "lineup_source": "projected"},
        "away_lineup": {"tier": "?", "recent_heat": "?", "lineup_source": "projected"},
        "home_pitcher": {"name": "?"}, "away_pitcher": {"name": "?"},
        "merged": {
            "park_factor": 100,
            "weather": {"condition": "Roof Closed", "temp_f": 72,
                        "wind_text": None, "indoor": True},
        },
        "home_roster": None, "away_roster": None,
    }
    md = render_summary(bundle, {"home_score": 0, "away_score": 0})
    assert "天氣：室內" in md
    assert "Roof Closed" in md
    # 室內不應有「對得分 / HR 影響判讀」placeholder
    assert "對得分 / HR 影響判讀" not in md


def test_summary_conditional_weather_absent():
    """weather=None → 顯示「天氣：未公布（跳過天氣分析）」、不出現 AI placeholder。"""
    from summary_renderer import render_summary

    bundle = {
        "home_lineup": {"tier": "?", "recent_heat": "?", "lineup_source": "projected"},
        "away_lineup": {"tier": "?", "recent_heat": "?", "lineup_source": "projected"},
        "home_pitcher": {"name": "?"}, "away_pitcher": {"name": "?"},
        "merged": {"park_factor": 100, "weather": None},
        "home_roster": None, "away_roster": None,
    }
    md = render_summary(bundle, {"home_score": 0, "away_score": 0})
    assert "天氣：未公布" in md
    assert "對得分 / HR 影響判讀" not in md
```

- [ ] **Step 2: 跑測試確認失敗**

Run:
```bash
cd scripts
python -m pytest tests/test_summary_renderer.py -k "conditional_weather" -v
```
Expected: 3 FAIL

- [ ] **Step 3: 改 `_render_conditional_section`**

定位 `summary_renderer.py` 約 128-137 行的 `_render_conditional_section`。改為：

```python
def _render_weather_state_line(weather: dict | None) -> list[str]:
    """Return summary 的天氣狀態列（含可能的 AI placeholder 子行）。
    
    三狀態：
    - 有資料：「天氣：{condition}, {temp}°F, wind {wind}」+ AI 影響判讀子行
    - 室內：「天氣：室內（{condition}，不適用）」（無 AI 子行）
    - 缺資料：「天氣：未公布（跳過天氣分析）」（無 AI 子行）
    """
    if not weather:
        return ["- 天氣：未公布（跳過天氣分析）"]
    if weather.get("indoor"):
        cond = weather.get("condition", "Indoor")
        return [f"- 天氣：室內（{cond}，不適用）"]
    parts = []
    if weather.get("condition"):
        parts.append(weather["condition"])
    if weather.get("temp_f") is not None:
        parts.append(f"{weather['temp_f']}°F")
    if weather.get("wind_text"):
        parts.append(f"wind {weather['wind_text']}")
    if not parts:
        return ["- 天氣：未公布（跳過天氣分析）"]
    return [
        f"- 天氣：{', '.join(parts)}",
        "  - 影響判讀：<!-- AI 補：對得分 / HR 影響判讀 -->",
    ]


def _render_conditional_section(bundle: dict) -> list[str]:
    merged = bundle.get("merged") or {}
    pf = merged.get("park_factor", 100)
    pf_correction = (pf - 100) * 0.05
    lines = [
        "## 條件修正",
        "",
        f"- Park Factor: {pf} → {pf_correction:+.2f} run",
    ]
    lines += _render_weather_state_line(merged.get("weather"))
    lines += [
        "- 先發 tier / doubleheader：<!-- AI 補 -->",
        "",
    ]
    return lines
```

- [ ] **Step 4: 跑測試確認 3 個 pass**

Run（同 Step 2 命令）。
Expected: 3 PASS

- [ ] **Step 5: 跑全部 summary_renderer 測試**

Run:
```bash
cd scripts
python -m pytest tests/test_summary_renderer.py -v
```
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/summary_renderer.py scripts/tests/test_summary_renderer.py
git commit -m "feat(summary): ## 條件修正 weather 三狀態 pre-fill"
```

---

## Task 13: `reference/matchup-factors.md` 異動

**Files:**
- Modify: `reference/matchup-factors.md`

- [ ] **Step 1: 在 `## 打線分析` 段首加 source 說明**

定位 `## 打線分析` 段（約 39 行）。在 H2 標題之後、`對打線核心（1-9 棒）查詢...` 那行之前插入：

```markdown
**打線來源**（由 `lineup_analyzer.py` 自動偵測）：
- 🟢 **official**：球隊已公布今日打序（賽前 ~2-4 小時 API 才填），9 人 1-9 棒順序為實際打序
- 🟡 **projected**：打序未公布，採 active roster（排除 IL）按 PA 降序取前 9 人作近似

**評級邏輯不分 source**：tier / chain / over_under_lean / 觸發條件對兩種來源一致。
**差異**：official 路徑下 `chain.obp_top3` / `slg_mid` 是真實 1-3 棒 / 4-5 棒；projected 是 PA 排序近似。

```

（注意保留原段尾後續內容如「對打線核心（1-9 棒）查詢...」段不變。）

- [ ] **Step 2: 在 `## 球場 & 天氣` 段尾加 `### 天氣修正`**

定位 `## 球場 & 天氣` 段（約 138 行起）。在現有 `### Park Factor` 段尾、整節結束之前插入：

```markdown
### 天氣修正

資料源：MLB Stats API `feed/live` 的 `gameData.weather`，由 `merge_game_data.py` 自動撈取。
**未公布或室內球場 → 不分析**（merged.weather = None 或 indoor=true）。

> ⛔ 天氣**不進 scoring formula**（與 BABIP / ERA-xERA gap 同等級——研究存在但 noisy）。
> AI 在 summary `## 條件修正` 段以敘事方式判讀，**不自動 ±run value**。

#### 風（wind）

MLB API wind 欄位已含風向解讀（球場 orientation 已換算），形式：

| 文字 | 意義 |
|------|------|
| `Out To CF / LF / RF` | 順風出去（利 HR / 飛球） |
| `In From CF / LF / RF` | 逆風進來（壓 HR / 利投手） |
| `L To R` / `R To L` | 橫風（影響有限） |
| `Calm` / `Varies` | 無顯著影響 |

風速門檻（敘事用）：

| 速度 | 影響 |
|------|------|
| < 8 mph | 噪音，可忽略 |
| 8–15 mph | 輕度，順風略利攻 / 逆風略利投 |
| 15–20 mph | 中度，HR 機率明顯偏移 |
| > 20 mph | 強，**summary 風險段必提** |

#### 溫度

聯盟基準 ~70°F；偏離越多影響越大（球的飛行距離與空氣密度 / 球皮含水量相關）。

| 溫度 | 影響 |
|------|------|
| > 85°F | ⬆️ 球易飛，輕度利攻 |
| 60–85°F | 中性 |
| 50–60°F | 輕度利投 |
| < 50°F | ⬆️ 利投，球員肌肉表現也受影響 |

> Coors / Yankee Stadium / Wrigley 對風更敏感（球場 orientation + 大氣條件交互）。
> 球員適應性差異大（北方球隊冷天表現相對好）— **AI 判讀時優先看相對強度**，不直接套表。
```

- [ ] **Step 3: 視覺驗證 markdown 結構**

Run:
```bash
cd /c/Users/USER/.agents/skills/mlb-game-analyzer
grep -E "^#" reference/matchup-factors.md
```
Expected: 看到 `## 打線分析` 與 `### 天氣修正` / `#### 風（wind）` / `#### 溫度` 都在；標題層級 (#) 正確。

- [ ] **Step 4: Commit**

```bash
git add reference/matchup-factors.md
git commit -m "docs(matchup): 加 §天氣修正 + 打線分析段首 source 說明"
```

---

## Task 14: `SKILL.md` 異動

**Files:**
- Modify: `SKILL.md`

- [ ] **Step 1: 改 Quick Reference 第 1 步描述**

定位 `## Quick Reference` 表（約 21-26 行）。第 1 步那行：

```markdown
| 1. 資料收集 | `merged.json` + `dossier.md` + `summary.md`（含 AI 填空 placeholder） | `prepare_game.py` |
```

改為：

```markdown
| 1. 資料收集 | `merged.json` + `dossier.md` + `summary.md`（含 AI 填空 placeholder）<br>**自動偵測**：official lineup（公布後）/ 天氣（公布後） | `prepare_game.py` |
```

- [ ] **Step 2: 在「資料來源優先順序」段後新增「條件式資料」段**

定位 `### 資料來源優先順序`（約 50 行附近）。在該段尾、`---` 分隔線之前插入：

```markdown
### 條件式資料（公布後才有）

| 資料 | 來源 | 缺資料行為 |
|------|------|-----------|
| 公布打線（battingOrder） | feed/live | fallback 至 PA proxy（lineup_source = "projected"） |
| 天氣（condition / temp / wind） | feed/live `gameData.weather` | summary 標「未公布（跳過天氣分析）」 |

**公布時機**：打線通常開賽前 2–4 小時、天氣前 1 小時 ~ 開賽後填齊。
**重跑取最新**：`prepare_game.py --force` 才會覆蓋已編輯的 summary.md（dossier 永遠重產）。
```

- [ ] **Step 3: 步驟 1 後續動作 ℹ️ 區補一條**

定位 `## 步驟 1：資料收集` 段尾的 `ℹ️` 區（約 79-81 行）。在既有 `ℹ️` 那條之後新增：

```markdown
ℹ️ **打線來源 / 天氣**：dossier 與 summary 都會標記。official 與 projected 分析架構相同，差異僅在 9 人組成是真實打序還是 PA 近似（見 `matchup-factors.md` §打線分析）。
```

- [ ] **Step 4: 步驟 2.3 條件修正描述補「天氣」**

定位 `## 步驟 2：綜合分析` 內的「2.1-2.4 順序執行」表（約 92-99 行）。第 2.3 列：

```markdown
| 2.3 條件修正 | 傷病/TJ/角色轉換/年齡/球場 | `matchup-factors.md` |
```

改為：

```markdown
| 2.3 條件修正 | 傷病/TJ/角色轉換/年齡/球場/**天氣** | `matchup-factors.md` §天氣修正 |
```

- [ ] **Step 5: 視覺驗證**

Run:
```bash
cd /c/Users/USER/.agents/skills/mlb-game-analyzer
grep -nE "official|天氣|條件式資料" SKILL.md
```
Expected: 看到上面 4 處異動字樣全在。

- [ ] **Step 6: Commit**

```bash
git add SKILL.md
git commit -m "docs(skill): Quick Reference + 條件式資料段 + 步驟提示更新"
```

---

## Task 15: End-to-end smoke test

**Files:**
- 跑真實 pipeline 驗證 + 暫存產出（不 commit 產出資料）

- [ ] **Step 1: 全測試套件回歸**

Run:
```bash
cd /c/Users/USER/.agents/skills/mlb-game-analyzer/scripts
python -m pytest tests/ -v
```
Expected: ALL PASS（既有 + 新增約 25 case）

- [ ] **Step 2: 跑當日一場真實比賽**

挑當天有公布打線的場次（賽前 ~2 小時後跑最穩）。例：
```bash
cd /c/Users/USER/.agents/skills/mlb-game-analyzer
python scripts/prepare_game.py --date 2026-05-03 --away BOS --home NYY
```

成功 exit code 0，看 stderr 是否出現以下其中一行（依當下狀態）：
- `[lineup_analyzer] official lineup fetched (9)` ← 公布後
- `[lineup_analyzer] official lineup not yet posted, fallback to PA proxy` ← 未公布
- 其他 fallback 訊息

- [ ] **Step 3: 檢查產出**

```bash
GAME_DIR=analysis-data/2026-05-03/BOS@NYY
cat "$GAME_DIR/merged.json" | python -c "import sys, json; m=json.load(sys.stdin); print('weather:', m.get('weather')); print('home_lineup_source:', m.get('home_lineup', {}).get('lineup_source')); print('away_lineup_source:', m.get('away_lineup', {}).get('lineup_source'))"
```
Expected: 看到 `weather: ...` 與兩隊 lineup_source 標籤。

```bash
grep -E "打線來源|weather|🟢|🟡" "$GAME_DIR/dossier.md" | head -20
grep -E "打線來源|天氣" "$GAME_DIR/summary.md" | head -20
```
Expected: dossier 與 summary 都看到 source 標、weather row（依當下狀態）。

- [ ] **Step 4: 不 commit 產出**

`analysis-data/` 內容是運行產物，不入版控（除非已被 git 追蹤過；確認 `.gitignore` 或現有 commit 規範）。若有 dossier/summary 變更需在 spec 中記錄，建議獨立另開一個 commit 對「sample 場次」更新，與 implementation commit 隔離。

- [ ] **Step 5: 整體 PR/branch 整理**

Run:
```bash
cd /c/Users/USER/.agents/skills/mlb-game-analyzer
git log --oneline -20
```
Expected: 看到 14 個本次 implementation commits（順序與 Task 對應）。

---

## 完工驗收

- [ ] 全測試套件 PASS（pytest tests/）
- [ ] `prepare_game.py` 跑真實場次 exit 0，無 abort
- [ ] dossier.md 顯示 `打線來源` 與（如有）weather row
- [ ] summary.md `## 打線評級` 各隊有 source 標、`## 條件修正` 有天氣狀態列
- [ ] merged.json 含 `weather` 區塊（dict 或 None）與兩隊 `lineup_source`
- [ ] 14 commits 結構清晰，每個 commit 與 task 對應

---

## Self-Review

**Spec coverage**：
- §5 Lineup → Tasks 2-6 ✓
- §6 Weather → Tasks 7-8 ✓
- §7 Reference + SKILL → Tasks 13-14 ✓
- §8 Testing → 各 Task 內含測試 ✓
- §10 Out of Scope → 不需 task（僅文件聲明）

**Placeholder scan**：所有 step 內含實際代碼 / 命令 / 預期輸出，無 TBD ✓

**Type consistency**：
- `lineup_source` 字串值 `"official" | "projected"` 全 plan 一致 ✓
- `lineup_source_detail` 在 official 是 dict、projected 是 None ✓
- `weather` 在 merged.json 是 dict 或 None ✓
- `fetch_official_lineup` 回傳 `list[int] | None`、`fetch_weather` 回傳 `dict | None` 全 plan 一致 ✓
- `batting_order` 在 official 是 1-9 int、projected 是 None ✓

**Scope check**：14 tasks + 1 smoke test，單一 implementation 計畫覆蓋。
