# Phase 1 game_data_summary.md Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓 Phase 1 的 `fetch_game_data.py` 在輸出 JSON 同時，額外產生 ~30-50 行 markdown summary，使 Claude 能以低 context 成本讀取 Phase 1 摘要；下游腳本不變。

**Architecture:** 在 `scripts/fetch_game_data.py` 既有腳本內新增 5 個純函式 + 1 個 assembler；`main()` 寫完 JSON 後額外寫 `game_data_summary.md`；新增測試檔涵蓋邊界條件；最後改 SKILL.md / workflow.md SOP（建議性措辭）。

**Tech Stack:** Python 3, pytest（既有專案慣例：`sys.path.insert` import + plain `def test_*` 函式）。

**Spec:** `docs/superpowers/specs/2026-04-26-game-data-summary-md-design.md`

---

### Task 1: `team_abbr` pure function

**Files:**
- Create: `scripts/tests/test_fetch_game_data.py`
- Modify: `scripts/fetch_game_data.py`（在 `FULL_NAMES` 區段後新增 `TEAM_ID_TO_ABBR` 常數 + `team_abbr()` 函式）

- [ ] **Step 1: Write the failing test**

建立 `scripts/tests/test_fetch_game_data.py`：

```python
"""Tests for fetch_game_data summary helpers (Phase 1 context slimming)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_team_abbr_known_team_id():
    from fetch_game_data import team_abbr
    assert team_abbr(118, "Kansas City Royals") == "KC"


def test_team_abbr_team_id_priority_over_name():
    """team_id 優先於 team_name；name 不影響結果"""
    from fetch_game_data import team_abbr
    assert team_abbr(108, "Wrong Name") == "LAA"


def test_team_abbr_team_id_none_lookup_full_name():
    from fetch_game_data import team_abbr
    assert team_abbr(None, "Los Angeles Angels") == "LAA"


def test_team_abbr_unknown_fallback():
    from fetch_game_data import team_abbr
    assert team_abbr(None, "Unknown Team Name") == "UNK"


def test_team_abbr_empty_name_fallback():
    from fetch_game_data import team_abbr
    assert team_abbr(None, "") == ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:/Users/USER/.claude/skills/mlb-game-analyzer
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: FAIL with `ImportError: cannot import name 'team_abbr'`

- [ ] **Step 3: Write minimal implementation**

在 `scripts/fetch_game_data.py` 的 `FULL_NAMES` dict 結束後（約 line 42 之後）插入：

```python

# Reverse lookup: team_id → English abbreviation (for summary md output).
# Excludes Chinese keys via isascii() filter.
TEAM_ID_TO_ABBR = {tid: abbr for abbr, tid in TEAM_MAP.items() if abbr.isascii() and abbr.isupper()}


def team_abbr(team_id: int | None, team_name: str) -> str:
    """team_id 優先反查 TEAM_ID_TO_ABBR；team_id 為 None 時用 team_name 透過 FULL_NAMES
    反查；都失敗 fallback 用 team_name 前 3 字大寫。"""
    if team_id is not None and team_id in TEAM_ID_TO_ABBR:
        return TEAM_ID_TO_ABBR[team_id]
    name_lower = (team_name or "").lower()
    if name_lower in FULL_NAMES:
        tid = FULL_NAMES[name_lower]
        if tid in TEAM_ID_TO_ABBR:
            return TEAM_ID_TO_ABBR[tid]
    return (team_name or "")[:3].upper()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_game_data.py scripts/tests/test_fetch_game_data.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 1 summary - 新增 team_abbr 純函式

team_id 優先反查 TEAM_ID_TO_ABBR；fallback 用 FULL_NAMES 經 team_name
反查；最終 fallback 取前 3 字大寫。為 game_data_summary.md 顯示縮寫使用。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `compute_trend_arrows` pure function

**Files:**
- Modify: `scripts/fetch_game_data.py`（在 `team_abbr` 後新增）
- Modify: `scripts/tests/test_fetch_game_data.py`（追加測試）

- [ ] **Step 1: Write the failing test**

在 `scripts/tests/test_fetch_game_data.py` 末尾追加：

```python


def test_compute_trend_arrows_offense_up_defense_worse():
    """KC 範例：RS 上升 → 攻↑；RA 上升 → 守↓"""
    from fetch_game_data import compute_trend_arrows
    result = compute_trend_arrows(5.10, 6.00, 3.79, 4.54)
    assert result["off_arrow"] == "↑"
    assert result["def_arrow"] == "↓"
    assert abs(result["off_delta"] - 1.31) < 0.01
    assert abs(result["def_delta"] - 1.46) < 0.01


def test_compute_trend_arrows_offense_down_flat_defense():
    """LAA 範例：RS −0.64 → 攻↓；RA −0.29 → 守→（未達 0.5）"""
    from fetch_game_data import compute_trend_arrows
    result = compute_trend_arrows(4.00, 4.50, 4.64, 4.79)
    assert result["off_arrow"] == "↓"
    assert result["def_arrow"] == "→"


def test_compute_trend_arrows_offense_down_defense_better():
    """RS 下降 → 攻↓；RA 下降 → 守↑"""
    from fetch_game_data import compute_trend_arrows
    result = compute_trend_arrows(3.50, 3.50, 4.50, 4.50)
    assert result["off_arrow"] == "↓"
    assert result["def_arrow"] == "↑"


def test_compute_trend_arrows_threshold_exact_50():
    """Δ = 0.5 邊界值應觸發箭頭（≥ 0.5）"""
    from fetch_game_data import compute_trend_arrows
    result = compute_trend_arrows(5.00, 4.50, 4.50, 5.00)
    assert result["off_arrow"] == "↑"  # +0.50
    assert result["def_arrow"] == "↑"  # RA −0.50 → 守↑


def test_compute_trend_arrows_threshold_just_below():
    """Δ = ±0.49 應為 →"""
    from fetch_game_data import compute_trend_arrows
    result = compute_trend_arrows(4.99, 4.50, 4.50, 4.99)
    assert result["off_arrow"] == "→"
    assert result["def_arrow"] == "→"


def test_compute_trend_arrows_zero_delta():
    from fetch_game_data import compute_trend_arrows
    result = compute_trend_arrows(4.50, 4.50, 4.50, 4.50)
    assert result["off_arrow"] == "→"
    assert result["def_arrow"] == "→"
    assert result["off_delta"] == 0.00
    assert result["def_delta"] == 0.00
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: 6 new tests FAIL with `ImportError: cannot import name 'compute_trend_arrows'`（既有 5 個 PASS）

- [ ] **Step 3: Write minimal implementation**

在 `scripts/fetch_game_data.py` 的 `team_abbr` 函式後新增：

```python


def compute_trend_arrows(rs10: float, ra10: float, rs30: float, ra30: float) -> dict:
    """近10 vs 近30 趨勢箭頭。|Δ| ≥ 0.5 才標箭頭。
    攻↑ = RS 上升；守↓ = RA 上升（防守變差）；守↑ = RA 下降。"""
    off_delta = round(rs10 - rs30, 2)
    def_delta = round(ra10 - ra30, 2)
    if off_delta >= 0.5:
        off_arrow = "↑"
    elif off_delta <= -0.5:
        off_arrow = "↓"
    else:
        off_arrow = "→"
    if def_delta >= 0.5:
        def_arrow = "↓"
    elif def_delta <= -0.5:
        def_arrow = "↑"
    else:
        def_arrow = "→"
    return {
        "off_arrow": off_arrow,
        "def_arrow": def_arrow,
        "off_delta": off_delta,
        "def_delta": def_delta,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: 11 PASS（5 previous + 6 new）

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_game_data.py scripts/tests/test_fetch_game_data.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 1 summary - 新增 compute_trend_arrows

近10 vs 近30 RS/RA 差距 ≥ 0.5 標箭頭；雙箭頭（攻 + 守）保留兩維資訊。
攻↑ = RS 上升；守↓ = RA 上升（防守變差）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `detect_current_series` pure function

**Files:**
- Modify: `scripts/fetch_game_data.py`（在 `compute_trend_arrows` 後新增）
- Modify: `scripts/tests/test_fetch_game_data.py`

- [ ] **Step 1: Write the failing test**

在 `scripts/tests/test_fetch_game_data.py` 末尾追加：

```python


def test_detect_current_series_g3_two_prev_games():
    """前 2 場連續對 LAA → 返回 [G1, G2]，升序排列"""
    from fetch_game_data import detect_current_series
    games = [
        {"date": "2026-04-25", "is_home": True, "opponent": "Los Angeles Angels",
         "team_score": 12, "opp_score": 1, "is_winner": True},
        {"date": "2026-04-24", "is_home": True, "opponent": "Los Angeles Angels",
         "team_score": 6, "opp_score": 3, "is_winner": True},
        {"date": "2026-04-22", "is_home": True, "opponent": "Baltimore Orioles",
         "team_score": 6, "opp_score": 8, "is_winner": False},
    ]
    result = detect_current_series(games, "Los Angeles Angels", "2026-04-26")
    assert len(result) == 2
    assert result[0]["date"] == "2026-04-24"
    assert result[0]["label"] == "G1"
    assert result[1]["date"] == "2026-04-25"
    assert result[1]["label"] == "G2"


def test_detect_current_series_first_game():
    """games[0] 對手不同 → 返回空 list（本系列首戰）"""
    from fetch_game_data import detect_current_series
    games = [
        {"date": "2026-04-25", "is_home": True, "opponent": "Detroit Tigers",
         "team_score": 5, "opp_score": 3, "is_winner": True},
    ]
    result = detect_current_series(games, "Los Angeles Angels", "2026-04-26")
    assert result == []


def test_detect_current_series_empty_games():
    from fetch_game_data import detect_current_series
    result = detect_current_series([], "Los Angeles Angels", "2026-04-26")
    assert result == []


def test_detect_current_series_doubleheader():
    """同日對同對手 2 場 → label 包含 (DH-1) / (DH-2)，G 編號連續遞增"""
    from fetch_game_data import detect_current_series
    games = [
        {"date": "2026-04-25", "is_home": True, "opponent": "Los Angeles Angels",
         "team_score": 12, "opp_score": 1, "is_winner": True},
        {"date": "2026-04-25", "is_home": True, "opponent": "Los Angeles Angels",
         "team_score": 6, "opp_score": 3, "is_winner": True},
        {"date": "2026-04-22", "is_home": True, "opponent": "Detroit Tigers",
         "team_score": 5, "opp_score": 3, "is_winner": True},
    ]
    result = detect_current_series(games, "Los Angeles Angels", "2026-04-26")
    assert len(result) == 2
    # 兩場同日，皆有 DH 標記；G 編號連續
    labels = [g["label"] for g in result]
    assert "G1 (DH-1)" in labels
    assert "G2 (DH-2)" in labels
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: 4 new tests FAIL with `ImportError: cannot import name 'detect_current_series'`

- [ ] **Step 3: Write minimal implementation**

在 `compute_trend_arrows` 後新增：

```python


def detect_current_series(games: list[dict], current_opp_team_name: str, current_game_date: str) -> list[dict]:
    """從 games[0]（最近一場）往後掃描，連續對 current_opp_team_name 的場次收集為當前系列賽。
    結果按日期升序排列；同日多場（doubleheader）標 (DH-N)。
    games 應是 home_recent 格式（按日期 desc 排序）。

    返回 list[dict]，每個 dict 含原 game 欄位 + "label"（如 "G1" 或 "G2 (DH-2)"）。
    若 games 空或 games[0] 對手不同，返回空 list。
    """
    matched = []
    for g in games:
        if g.get("opponent") == current_opp_team_name:
            matched.append(g)
        else:
            break
    if not matched:
        return []

    # 升序排列；同日內保留原順序
    matched.sort(key=lambda g: g["date"])

    # 偵測 doubleheader：同日 ≥ 2 場
    by_date: dict[str, int] = {}
    for g in matched:
        by_date[g["date"]] = by_date.get(g["date"], 0) + 1

    result = []
    g_num = 1
    dh_counters: dict[str, int] = {}
    for g in matched:
        date = g["date"]
        if by_date[date] > 1:
            dh_counters[date] = dh_counters.get(date, 0) + 1
            label = f"G{g_num} (DH-{dh_counters[date]})"
        else:
            label = f"G{g_num}"
        result.append({**g, "label": label})
        g_num += 1
    return result
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: 15 PASS（11 previous + 4 new）

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_game_data.py scripts/tests/test_fetch_game_data.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 1 summary - 新增 detect_current_series

從 games[0] 往後連續同對手收集為當前系列賽，升序排列；同日多場
標 (DH-N)，G 編號連續遞增。系列首戰返回空 list。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `format_streak_context` pure function

**Files:**
- Modify: `scripts/fetch_game_data.py`（在 `detect_current_series` 後新增）
- Modify: `scripts/tests/test_fetch_game_data.py`

- [ ] **Step 1: Write the failing test**

在 `scripts/tests/test_fetch_game_data.py` 末尾追加：

```python


def test_format_streak_context_winning_streak():
    """連勝 → '連勝對手 → ABBR (MM-DD), ...'，升序排列"""
    from fetch_game_data import format_streak_context
    games = [
        {"date": "2026-04-25", "opponent": "Los Angeles Angels", "is_winner": True},
        {"date": "2026-04-24", "opponent": "Los Angeles Angels", "is_winner": True},
        {"date": "2026-04-22", "opponent": "Baltimore Orioles", "is_winner": False},
    ]
    result = format_streak_context(games, 2)
    assert result is not None
    assert "連勝對手" in result
    assert "LAA" in result
    assert "04-24" in result
    assert "04-25" in result
    # 升序：04-24 應在 04-25 前
    assert result.index("04-24") < result.index("04-25")


def test_format_streak_context_losing_streak():
    """連敗 → '連敗對手 → ABBR (MM-DD), ...'"""
    from fetch_game_data import format_streak_context
    games = [
        {"date": "2026-04-25", "opponent": "Kansas City Royals", "is_winner": False},
        {"date": "2026-04-24", "opponent": "Kansas City Royals", "is_winner": False},
        {"date": "2026-04-22", "opponent": "Toronto Blue Jays", "is_winner": False},
    ]
    result = format_streak_context(games, -3)
    assert result is not None
    assert "連敗對手" in result
    assert "KC" in result
    assert "TOR" in result
    # TOR (04-22) 應排在最前（升序）
    assert result.index("TOR") < result.index("KC")


def test_format_streak_context_streak_zero_returns_none():
    from fetch_game_data import format_streak_context
    games = [{"date": "2026-04-25", "opponent": "X", "is_winner": True}]
    assert format_streak_context(games, 0) is None


def test_format_streak_context_empty_games_returns_none():
    from fetch_game_data import format_streak_context
    assert format_streak_context([], 2) is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: 4 new tests FAIL with `ImportError: cannot import name 'format_streak_context'`

- [ ] **Step 3: Write minimal implementation**

在 `detect_current_series` 後新增：

```python


def format_streak_context(games: list[dict], streak: int) -> str | None:
    """格式化連勝/連敗對手列表（升序）。streak=0 或 games 空回 None。"""
    if streak == 0 or not games:
        return None
    n = abs(streak)
    label = "連勝對手" if streak > 0 else "連敗對手"
    items = []
    for g in games[:n]:
        abbr = team_abbr(None, g.get("opponent", ""))
        date_short = g.get("date", "")[5:]  # MM-DD
        items.append(f"{abbr} ({date_short})")
    items.reverse()  # games 是 desc → 反轉後為 asc
    return f"{label} → " + ", ".join(items)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: 19 PASS（15 previous + 4 new）

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_game_data.py scripts/tests/test_fetch_game_data.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 1 summary - 新增 format_streak_context

按 streak 正負取 abs(streak) 場對手，反向後升序輸出。
streak=0 或 games 空時返回 None（讓 caller 省略整個 section）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `format_summary_md` assembler

**Files:**
- Modify: `scripts/fetch_game_data.py`（在 `format_streak_context` 後新增格式化輔助 + assembler）
- Modify: `scripts/tests/test_fetch_game_data.py`

- [ ] **Step 1: Write the failing test**

在 `scripts/tests/test_fetch_game_data.py` 末尾追加（smoke + edge case）：

```python


def _make_minimal_result(home_games=None, away_games=None, series_prev=None):
    """測試用 result_dict 工廠"""
    return {
        "game": {
            "gamePk": 824122,
            "date": "2026-04-26T23:20:00Z",
            "status": "Preview",
            "venue": "Kauffman Stadium",
            "home": {"team": "Kansas City Royals", "team_id": 118, "probable_pitcher": "Seth Lugo"},
            "away": {"team": "Los Angeles Angels", "team_id": 108, "probable_pitcher": "Reid Detmers"},
        },
        "home_recent": {"record": "3-7", "rs_per_game": 5.10, "ra_per_game": 6.00,
                        "run_diff": -9, "streak": 2, "games": home_games or []},
        "away_recent": {"record": "3-7", "rs_per_game": 4.00, "ra_per_game": 4.50,
                        "run_diff": -5, "streak": -3, "games": away_games or []},
        "home_recent_30": {"record": "10-18", "rs_per_game": 3.79, "ra_per_game": 4.54,
                           "run_diff": -21, "streak": 2, "games": []},
        "away_recent_30": {"record": "12-16", "rs_per_game": 4.64, "ra_per_game": 4.79,
                           "run_diff": -4, "streak": -3, "games": []},
        "home_season": {"record": "10-18", "rs_per_game": 3.79, "ra_per_game": 4.54,
                        "run_diff": -21, "streak": 2, "games": []},
        "away_season": {"record": "12-16", "rs_per_game": 4.64, "ra_per_game": 4.79,
                        "run_diff": -4, "streak": -3, "games": []},
        "home_season_games_count": 28,
        "away_season_games_count": 28,
        "series_prev": series_prev,
    }


def test_format_summary_md_smoke_full_game():
    """完整 result_dict → markdown 含所有 hard sections + 標題"""
    from fetch_game_data import format_summary_md
    home_games = [
        {"date": "2026-04-25", "is_home": True, "opponent": "Los Angeles Angels",
         "team_score": 12, "opp_score": 1, "is_winner": True},
        {"date": "2026-04-24", "is_home": True, "opponent": "Los Angeles Angels",
         "team_score": 6, "opp_score": 3, "is_winner": True},
    ]
    away_games = [
        {"date": "2026-04-25", "is_home": False, "opponent": "Kansas City Royals",
         "team_score": 1, "opp_score": 12, "is_winner": False},
        {"date": "2026-04-24", "is_home": False, "opponent": "Kansas City Royals",
         "team_score": 3, "opp_score": 6, "is_winner": False},
        {"date": "2026-04-22", "is_home": True, "opponent": "Toronto Blue Jays",
         "team_score": 2, "opp_score": 4, "is_winner": False},
    ]
    md = format_summary_md(_make_minimal_result(home_games, away_games))
    assert "# Game Data Summary — LAA @ KC (2026-04-26)" in md
    assert "## 比賽資訊" in md
    assert "## 戰績摘要" in md
    assert "## 趨勢" in md
    assert "## 當前系列賽" in md
    assert "## Streak 脈絡" in md
    assert "Reid Detmers" in md
    assert "Seth Lugo" in md
    # 系列累計：KC 2-0 LAA
    assert "KC 2-0 LAA" in md or "**KC 2-0 LAA**" in md


def test_format_summary_md_first_game_of_series():
    """無前場 → 系列賽 section 顯示「本系列首戰」"""
    from fetch_game_data import format_summary_md
    home_games = [
        {"date": "2026-04-25", "is_home": True, "opponent": "Detroit Tigers",
         "team_score": 5, "opp_score": 3, "is_winner": True},
    ]
    md = format_summary_md(_make_minimal_result(home_games=home_games))
    assert "本系列首戰" in md


def test_format_summary_md_empty_games_omits_soft_sections():
    """games 空 → 系列賽 + Streak 脈絡 sections 整段省略；hard sections 仍存在"""
    from fetch_game_data import format_summary_md
    md = format_summary_md(_make_minimal_result())
    assert "## 戰績摘要" in md  # hard section 保留
    assert "## 當前系列賽" not in md  # soft section 省略
    assert "## Streak 脈絡" not in md


def test_format_summary_md_raises_on_missing_game():
    from fetch_game_data import format_summary_md
    import pytest
    with pytest.raises(ValueError):
        format_summary_md({})


def test_format_summary_md_raises_on_missing_team_id():
    from fetch_game_data import format_summary_md
    import pytest
    bad = _make_minimal_result()
    bad["game"]["home"]["team_id"] = None
    with pytest.raises(ValueError):
        format_summary_md(bad)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: 5 new tests FAIL with `ImportError: cannot import name 'format_summary_md'`

- [ ] **Step 3: Write minimal implementation**

在 `format_streak_context` 後新增：

```python


def _fmt_signed(n) -> str:
    """格式化有號數值。None → '—'；正數加 +；負數用 '−'（U+2212）"""
    if n is None:
        return "—"
    if n > 0:
        return f"+{n}" if isinstance(n, int) else f"+{n:.2f}"
    if n < 0:
        return f"−{abs(n)}" if isinstance(n, int) else f"−{abs(n):.2f}"
    return "0"


def _fmt_streak(s) -> str:
    if s is None or s == 0:
        return "0"
    return f"+{s}" if s > 0 else f"−{abs(s)}"


def _fmt_num(n) -> str:
    if n is None:
        return "—"
    return f"{n:.2f}"


def _fmt_record_row(d: dict) -> str:
    rec = d.get("record", "—")
    rs = _fmt_num(d.get("rs_per_game"))
    ra = _fmt_num(d.get("ra_per_game"))
    diff = _fmt_signed(d.get("run_diff"))
    streak = _fmt_streak(d.get("streak"))
    return f"{rec}  (RS {rs} / RA {ra} / diff {diff} / streak {streak})"


def _fmt_record_row_no_streak(d: dict) -> str:
    rec = d.get("record", "—")
    rs = _fmt_num(d.get("rs_per_game"))
    ra = _fmt_num(d.get("ra_per_game"))
    diff = _fmt_signed(d.get("run_diff"))
    return f"{rec} (RS {rs} / RA {ra} / diff {diff})"


def format_summary_md(result: dict) -> str:
    """組合 game_data_summary.md 完整內容。
    Hard sections（必出現）：比賽資訊 / 戰績摘要 / 趨勢
    Soft sections（缺資料省略）：當前系列賽 / Streak 脈絡
    Fail-fast：result.game 缺失或雙方 team_id 缺失 → raise ValueError
    """
    if "game" not in result:
        raise ValueError("result.game missing — cannot generate summary")
    game = result["game"]
    home = game.get("home", {})
    away = game.get("away", {})
    if not home.get("team_id") or not away.get("team_id"):
        raise ValueError("home/away team_id missing — cannot generate summary")

    home_abbr = team_abbr(home["team_id"], home.get("team", ""))
    away_abbr = team_abbr(away["team_id"], away.get("team", ""))
    game_date = game.get("date", "")[:10]

    lines = [f"# Game Data Summary — {away_abbr} @ {home_abbr} ({game_date})", ""]

    # ========== 比賽資訊（hard） ==========
    lines += [
        "## 比賽資訊",
        f"- 日期 (ET): {game_date}",
        f"- 開賽 (UTC ISO): {game.get('date', '—')}",
        f"- 球場: {game.get('venue', '—')}",
        f"- 狀態: {game.get('status', '—')}",
        f"- 先發: {away.get('probable_pitcher', 'TBD')} ({away_abbr}) vs {home.get('probable_pitcher', 'TBD')} ({home_abbr})",
        "",
    ]

    # ========== 戰績摘要（hard） ==========
    home_recent = result.get("home_recent", {})
    away_recent = result.get("away_recent", {})
    home_30 = result.get("home_recent_30", {})
    away_30 = result.get("away_recent_30", {})
    home_season = result.get("home_season", {})
    away_season = result.get("away_season", {})
    home_n = result.get("home_season_games_count", 0)
    away_n = result.get("away_season_games_count", 0)

    lines += [
        "## 戰績摘要",
        "",
        f"| 區間 | {home_abbr}（主） | {away_abbr}（客） |",
        "|------|---------|----------|",
        f"| 近 10 場 | {_fmt_record_row(home_recent)} | {_fmt_record_row(away_recent)} |",
        f"| 近 30 場 | {_fmt_record_row_no_streak(home_30)} | {_fmt_record_row_no_streak(away_30)} |",
        f"| 本季 | {home_season.get('record', '—')} ({home_n} 場) | {away_season.get('record', '—')} ({away_n} 場) |",
        "",
    ]

    # ========== 趨勢（hard） ==========
    if (home_recent.get("rs_per_game") is not None
            and home_30.get("rs_per_game") is not None
            and away_recent.get("rs_per_game") is not None
            and away_30.get("rs_per_game") is not None):
        h = compute_trend_arrows(home_recent["rs_per_game"], home_recent["ra_per_game"],
                                 home_30["rs_per_game"], home_30["ra_per_game"])
        a = compute_trend_arrows(away_recent["rs_per_game"], away_recent["ra_per_game"],
                                 away_30["rs_per_game"], away_30["ra_per_game"])
        lines += [
            "## 趨勢（近 10 vs 近 30）",
            f"- {home_abbr}: 攻{h['off_arrow']} (RS {home_recent['rs_per_game']:.2f} vs {home_30['rs_per_game']:.2f}，{_fmt_signed(h['off_delta'])}) | 守{h['def_arrow']} (RA {home_recent['ra_per_game']:.2f} vs {home_30['ra_per_game']:.2f}，{_fmt_signed(h['def_delta'])})",
            f"- {away_abbr}: 攻{a['off_arrow']} (RS {away_recent['rs_per_game']:.2f} vs {away_30['rs_per_game']:.2f}，{_fmt_signed(a['off_delta'])}) | 守{a['def_arrow']} (RA {away_recent['ra_per_game']:.2f} vs {away_30['ra_per_game']:.2f}，{_fmt_signed(a['def_delta'])})",
            "",
            "> 規則：|Δ| ≥ 0.5 才標箭頭。攻↑ = RS 上升；守↓ = RA 上升（防守變差）。",
            "",
        ]
    else:
        lines += ["## 趨勢（近 10 vs 近 30）", "- —（資料不足）", ""]

    # ========== 當前系列賽（soft） ==========
    home_games = home_recent.get("games", [])
    if home_games:
        away_team_name = away.get("team", "")
        series = detect_current_series(home_games, away_team_name, game_date)
        lines.append(f"## 當前系列賽 ({away_abbr} @ {home_abbr})")
        if not series:
            lines += [
                f"- G1 ({game_date[5:]}): 本場",
                "- 系列累計: 本系列首戰，無前場",
                "",
            ]
        else:
            home_wins = 0
            away_wins = 0
            for g in series:
                if g.get("is_home"):
                    home_score, away_score = g.get("team_score", 0), g.get("opp_score", 0)
                    winner_abbr = home_abbr if g.get("is_winner") else away_abbr
                else:
                    home_score, away_score = g.get("opp_score", 0), g.get("team_score", 0)
                    winner_abbr = away_abbr if g.get("is_winner") else home_abbr
                if winner_abbr == home_abbr:
                    home_wins += 1
                else:
                    away_wins += 1
                lines.append(
                    f"- {g['label']} ({g['date'][5:]}): {home_abbr} {home_score}-{away_score} {away_abbr} → {winner_abbr} 勝"
                )
            this_g = f"G{len(series) + 1}"
            lines.append(f"- {this_g} ({game_date[5:]}): 本場")
            lines.append(f"- 系列累計: **{home_abbr} {home_wins}-{away_wins} {away_abbr}**")
            lines.append("")

    # ========== Streak 脈絡（soft） ==========
    h_streak = home_recent.get("streak") or 0
    a_streak = away_recent.get("streak") or 0
    h_ctx = format_streak_context(home_games, h_streak) if home_games else None
    away_games = away_recent.get("games", [])
    a_ctx = format_streak_context(away_games, a_streak) if away_games else None
    if h_ctx or a_ctx:
        lines.append("## Streak 脈絡")
        if h_ctx:
            lines.append(f"- {home_abbr} {_fmt_streak(h_streak)}: {h_ctx}")
        if a_ctx:
            lines.append(f"- {away_abbr} {_fmt_streak(a_streak)}: {a_ctx}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: 24 PASS（19 previous + 5 new）

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_game_data.py scripts/tests/test_fetch_game_data.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 1 summary - 新增 format_summary_md assembler

組合 game_data_summary.md 完整內容。Hard sections（比賽資訊 / 戰績
摘要 / 趨勢）必出現；Soft sections（系列賽 / Streak 脈絡）缺資料省略。
Fail-fast on missing game / team_id。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `main()` 整合 — 寫出 summary 檔

**Files:**
- Modify: `scripts/fetch_game_data.py`（修改 `main()` 末段）

無單元測試（純 I/O 整合，靠下一步手動執行驗證）。

- [ ] **Step 1: 修改 `main()`**

在 `scripts/fetch_game_data.py` 的 `main()` 中，找到這段：

```python
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(json_output)
```

替換為：

```python
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Saved to {args.output}", file=sys.stderr)

        # 額外輸出 summary md（同目錄 game_data_summary.md）
        from pathlib import Path
        summary_path = Path(args.output).parent / "game_data_summary.md"
        try:
            summary_md = format_summary_md(result)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_md)
            print(f"Saved summary to {summary_path}", file=sys.stderr)
        except ValueError as e:
            print(f"Skipped summary (data incomplete): {e}", file=sys.stderr)
    else:
        print(json_output)
```

- [ ] **Step 2: 手動驗證 — 用本場 LAA@KC fixture**

```bash
cd C:/Users/USER/.claude/skills/mlb-game-analyzer
python scripts/fetch_game_data.py --date 2026-04-26 --team KC -o analysis-data/2026-04-26/LAA@KC/game_data.json
```

Expected stderr:
```
Saved to analysis-data/2026-04-26/LAA@KC/game_data.json
Saved summary to analysis-data/2026-04-26/LAA@KC/game_data_summary.md
```

- [ ] **Step 3: 檢查 summary 內容**

```bash
cat analysis-data/2026-04-26/LAA@KC/game_data_summary.md
```

Expected：30-50 行 markdown，包含 5 個 sections（比賽資訊 / 戰績摘要 / 趨勢 / 當前系列賽（KC 2-0 LAA）/ Streak 脈絡），數值與 spec 範例對齊。

- [ ] **Step 4: 全測試回歸**

```bash
python -m pytest scripts/tests/ -v
```
Expected: 全部 PASS（含舊測試 + 新增 24 個）

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_game_data.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 1 summary - main() 整合輸出 summary md

寫完 JSON 後額外輸出 game_data_summary.md 至同目錄。
ValueError 時 stderr warning 但不 fail（保留 JSON 輸出）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: SOP 更新 — `SKILL.md` + `reference/workflow.md`

**Files:**
- Modify: `SKILL.md`（Quick Reference 表）
- Modify: `reference/workflow.md`（Phase 1 章節）

無單元測試（純文件變更）。

- [ ] **Step 1: 修改 `SKILL.md` Quick Reference 表**

找到 Quick Reference 區段：

```markdown
| Phase | 主要產出 |
|-------|---------|
| 1. 資料收集 | `game_data.json`（`fetch_game_data.py`，例行賽） |
```

改為：

```markdown
| Phase | 主要產出 |
|-------|---------|
| 1. 資料收集 | `game_data.json` + `game_data_summary.md`（`fetch_game_data.py`，例行賽） |
```

- [ ] **Step 2: 修改 `reference/workflow.md` Phase 1.2**

找到 Phase 1.2 末尾「> 僅使用 `gameType = "R"` 例行賽，排除春訓。」

在這行之後加：

```markdown

> 腳本同時輸出 `game_data_summary.md` 至同目錄（~30-50 行 markdown，含戰績 / 趨勢 / 當前系列賽 / Streak 脈絡）。
```

- [ ] **Step 3: 修改 `reference/workflow.md` Phase 1.4 標題下**

找到 `### 1.4 輸出確認`，在標題下、` ``` ` 之前加：

```markdown

✅ Read `$GAME_DIR/game_data_summary.md`，依其內容填入下方輸出模板。

ℹ️ 一般情況下無需 Read `game_data.json`；僅在 summary 缺漏 / 使用者明確要求查驗 / 除錯時 Read 完整 JSON。
```

- [ ] **Step 4: 修改 `reference/workflow.md` Phase 1.5 閘門**

找到 `### 1.5 Phase 1 閘門` 區段，在第一條 checkbox 下增加：

```markdown
- [ ] `game_data_summary.md` 已輸出
```

具體插入位置：

```markdown
### 1.5 Phase 1 閘門

- [ ] `game_data.json` 已輸出
- [ ] `game_data_summary.md` 已輸出
- [ ] `gameType == "R"`（例行賽）
...
```

- [ ] **Step 5: 驗證 SOP 內容**

```bash
grep -n "summary" SKILL.md reference/workflow.md
```
Expected：`SKILL.md` 至少 1 條、`reference/workflow.md` 至少 3 條（Phase 1.2 / 1.4 / 1.5）。

- [ ] **Step 6: Commit**

```bash
git add SKILL.md reference/workflow.md
git commit -m "$(cat <<'EOF'
docs(mlb-skill): Phase 1 SOP 對齊 game_data_summary.md

SKILL.md Quick Reference 補 summary 為主要產出之一；workflow.md Phase
1.2 補腳本同時輸出 summary、Phase 1.4 改 Read summary 而非完整 JSON
（建議性措辭，無 ⛔，保留例外指引）、Phase 1.5 閘門新增 summary 檢查。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- ✅ Section 2 規格 → Task 5 (`format_summary_md`)
- ✅ Section 2 趨勢規則 → Task 2 (`compute_trend_arrows`)
- ✅ Section 2 系列賽偵測 → Task 3 (`detect_current_series`)
- ✅ Section 2 Streak 脈絡 → Task 4 (`format_streak_context`)
- ✅ Section 2 球隊縮寫 → Task 1 (`team_abbr`)
- ✅ Section 3 邊界條件 → Task 5 hard/soft section 邏輯 + edge case 測試
- ✅ Section 4.1 main() 整合 → Task 6
- ✅ Section 4.2 測試 → Tasks 1-5 各自含測試
- ✅ Section 4.3 SKILL.md / workflow.md → Task 7
- ✅ Section 4.4 不動的部分 → 計畫不觸碰

**Placeholder scan:**
- ✅ 無 TBD / TODO
- ✅ 每個 step 含完整可執行的代碼或命令

**Type consistency:**
- ✅ `team_abbr(team_id: int | None, team_name: str) → str` 在 Task 1 定義，Task 4 / 5 呼叫一致
- ✅ `compute_trend_arrows` 返回 dict 鍵 `off_arrow / def_arrow / off_delta / def_delta` 在 Task 5 使用一致
- ✅ `detect_current_series` 返回 list[dict]，每 dict 含 `label` 鍵；Task 5 取 `g["label"]` 一致
- ✅ `format_streak_context` 返回 `str | None`；Task 5 用 `if h_ctx or a_ctx:` 判斷一致
