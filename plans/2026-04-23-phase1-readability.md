# Phase 1 Readability Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓 Phase 1 輸出在不破壞下游 JSON 契約的前提下，變得對 LLM 和人類都更好讀 — 透過 `fetch_game_data.py` 自動印出人類可讀摘要、自動落地 `phase1_summary.md`，並新增早季冗餘標記。

**Architecture:** 所有改動集中在 `scripts/fetch_game_data.py`（本層），輸出契約（JSON 欄位）僅做加法（新增 `equals_season` 旗標）不做減法，下游 `merge_game_data.py` / `predict.py` 無需配合修改。`reference/workflow.md` 同步更新使其敘述與實際行為一致。`phase1_summary.md` 由腳本自動生成，不由 LLM 手寫（與 `phase3_summary.md` 由 LLM 手寫的原因不同 — Phase 1 是資料摘要，無需分析判斷）。

**Tech Stack:** Python 3（既有），pytest（既有 `scripts/tests/` 慣例）。

---

## 背景與 Spec（為什麼做這件事）

2026-04-23 分析 NYY@BOS 一場時發現：

1. **痛點 1 — stdout 極簡**：`fetch_game_data.py --output <file>` 只印 `Saved to ...` 到 stderr，LLM 必須額外 Read 一份 1022 行的 JSON 才能拿到 20 行的摘要資訊。workflow.md 1.4 節**有規範**一份 `📅📊` 人類可讀摘要模板，但腳本**沒有實作**。

2. **痛點 2 — 跨對話 context 流失**：Phase 1 結論只存在於對話 transcript 中。若對話壓縮或下次對話續接，LLM 需要重新讀 JSON 重建 context。相比之下，Phase 3 有 `phase3_summary.md` 落地這個防線，Phase 1 沒有。

3. **痛點 3 — 早季資料冗餘**：賽季開打前 30 場內，`home_recent_30` 與 `home_season` 欄位完全一致（都是全季 N≤30 場），但沒有旗標標示。LLM 讀 JSON 時浪費 token 處理重複資料，或誤以為兩個窗口能各自提供獨立訊號。

**不在本次 scope 內：**
- 對話內我對 BOS 近 10 趨勢字詞錯誤（"略好" 應為 "略差"）— 是敘述層瑕疵，非 code 問題
- 改 JSON schema（刪欄位、改欄位名）— 會破壞下游，成本過高
- Phase 3 summary 由 LLM 手寫的流程 — 不動

---

## File Structure

| 檔案 | 動作 | 責任 |
|---|---|---|
| `scripts/fetch_game_data.py` | Modify | 新增 summary 產生器、輸出邏輯重構、`equals_season` 旗標 |
| `scripts/tests/test_fetch_game_data.py` | **Create** | 本腳本目前沒有 test file，本 plan 順便補上 |
| `reference/workflow.md` | Modify | 更新 1.4 節敘述與實際行為同步，新增 `phase1_summary.md` 落地條目 |

---

## Task 1：抽出 `build_summary_lines()` helper

將 JSON → 人類可讀 summary 的轉換邏輯抽成純函式，便於測試，也讓後續 Task 2 的 output 邏輯乾淨。

**Files:**
- Modify: `scripts/fetch_game_data.py`（新增函式，在 `main()` 之前，約 line 258 前插入）
- Create: `scripts/tests/test_fetch_game_data.py`

**設計決策：**
- 輸入：`build_summary_lines(result: dict) -> list[str]`，`result` 就是目前 line 314-325 組出的 dict
- 輸出：`list[str]`，呼叫端自行決定用 `"\n".join(...)` 還是逐行寫
- 內容遵循 workflow.md 1.4 模板（🗓️📅⚾🕐📊📈），另加一行 `series_prev` 若存在

- [ ] **Step 1：新增 test file 並寫第一個失敗測試**

建立 `scripts/tests/test_fetch_game_data.py`（import 方式**沿用 repo 既有慣例** — 見 `test_merge_game_data.py` line 2-11：`sys.path.insert` 後直接 `from fetch_game_data import ...`，**不要用** `from scripts.fetch_game_data import ...`）：

```python
"""Tests for fetch_game_data.py summary/output helpers."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fetch_game_data import build_summary_lines


def _sample_result() -> dict:
    """最小可行 result dict（欄位結構與 main() 輸出一致）"""
    return {
        "game": {
            "gamePk": 824770,
            "date": "2026-04-23T22:10:00Z",
            "status": "Preview",
            "venue": "Fenway Park",
            "home": {"team": "Boston Red Sox", "team_id": 111, "probable_pitcher": "Payton Tolle"},
            "away": {"team": "New York Yankees", "team_id": 147, "probable_pitcher": "Cam Schlittler"},
        },
        "home_recent": {"record": "4-6", "rs_per_game": 3.7, "ra_per_game": 5.1, "streak": -2, "games": []},
        "away_recent": {"record": "7-3", "rs_per_game": 5.7, "ra_per_game": 4.4, "streak": 5, "games": []},
        "home_recent_30": {"record": "9-15", "rs_per_game": 3.75, "ra_per_game": 4.58, "streak": -2, "games": []},
        "away_recent_30": {"record": "15-9", "rs_per_game": 4.92, "ra_per_game": 3.46, "streak": 5, "games": []},
        "home_season": {"record": "9-15", "rs_per_game": 3.75, "ra_per_game": 4.58, "streak": -2, "games": []},
        "away_season": {"record": "15-9", "rs_per_game": 4.92, "ra_per_game": 3.46, "streak": 5, "games": []},
        "home_season_games_count": 24,
        "away_season_games_count": 24,
        "series_prev": {
            "date": "2026-04-22",
            "home": "Boston Red Sox",
            "away": "New York Yankees",
            "home_score": 1,
            "away_score": 4,
            "winner": "New York Yankees",
        },
    }


def test_build_summary_lines_contains_teams_pitchers_and_window_stats():
    lines = build_summary_lines(_sample_result())
    joined = "\n".join(lines)

    # 基本欄位
    assert "New York Yankees" in joined
    assert "Boston Red Sox" in joined
    assert "Cam Schlittler" in joined
    assert "Payton Tolle" in joined
    assert "Fenway Park" in joined
    assert "Preview" in joined
    # 三窗口戰績（主客各三組）= 六組 "N-M" 字串
    for record in ("4-6", "7-3", "9-15", "15-9"):
        assert record in joined
    # RS/RA 關鍵數字
    assert "3.7" in joined and "5.1" in joined  # BOS 近 10
    assert "5.7" in joined and "4.4" in joined  # NYY 近 10
    # series_prev 存在時列出
    assert "2026-04-22" in joined


def test_build_summary_lines_without_series_prev():
    result = _sample_result()
    result["series_prev"] = None
    lines = build_summary_lines(result)
    joined = "\n".join(lines)
    # 不該出現「同系列前場」區塊的關鍵字（避免誤印 None）
    assert "None" not in joined
```

- [ ] **Step 2：跑測試確認失敗**

Run:
```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: FAIL with `ImportError` / `AttributeError: module has no attribute 'build_summary_lines'`

- [ ] **Step 3：在 `fetch_game_data.py` 新增 `build_summary_lines()`**

在 `main()` 定義之前（約 line 258 之前）插入：

```python
def build_summary_lines(result: dict) -> list[str]:
    """將 result dict 轉為 workflow.md 1.4 規範的人類可讀摘要行。

    Returns:
        list[str]: 每行一段，呼叫端自行 join。
    """
    g = result["game"]
    home = g["home"]
    away = g["away"]

    lines: list[str] = []
    lines.append(f"📅 {g['date']} — {away['team']} @ {home['team']}（{g['venue']}）")
    lines.append(f"⚾ 先發：{away['probable_pitcher']} (客) vs {home['probable_pitcher']} (主)")
    lines.append(f"🕐 狀態：{g['status']} | gamePk：{g['gamePk']}")
    lines.append("")

    def _fmt_window(label: str, stats: dict) -> str:
        streak = stats.get("streak", 0)
        streak_str = f"連勝 {streak}" if streak > 0 else (f"連敗 {abs(streak)}" if streak < 0 else "—")
        return (
            f"📊 {label}：{stats['record']}"
            f"（RS/G {stats['rs_per_game']} | RA/G {stats['ra_per_game']} | {streak_str}）"
        )

    lines.append(f"【{home['team']}】")
    lines.append(_fmt_window("近 10 場", result["home_recent"]))
    lines.append(_fmt_window("近 30 場", result["home_recent_30"]))
    lines.append(_fmt_window(f"本季（{result['home_season_games_count']} 場）", result["home_season"]))
    lines.append("")
    lines.append(f"【{away['team']}】")
    lines.append(_fmt_window("近 10 場", result["away_recent"]))
    lines.append(_fmt_window("近 30 場", result["away_recent_30"]))
    lines.append(_fmt_window(f"本季（{result['away_season_games_count']} 場）", result["away_season"]))

    sp = result.get("series_prev")
    if sp:
        lines.append("")
        lines.append(
            f"🔁 同系列前場（{sp['date']}）：{sp['away']} @ {sp['home']} "
            f"→ {sp['away_score']}-{sp['home_score']}，勝者：{sp['winner']}"
        )

    return lines
```

- [ ] **Step 4：跑測試確認通過**

Run:
```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: PASS（2 passed）

- [ ] **Step 5：Commit**

```bash
git add scripts/fetch_game_data.py scripts/tests/test_fetch_game_data.py
git commit -m "feat(fetch_game_data): 新增 build_summary_lines() helper

將 JSON → workflow.md 1.4 模板的轉換邏輯抽成純函式，
供後續 stdout summary 與 phase1_summary.md 落地共用。"
```

---

## Task 2：整合輸出 — stderr summary + `phase1_summary.md` 自動落地

把寫檔、印 summary 的邏輯從 `main()` 抽出到 `write_outputs()`，同時實作兩個新行為：
1. 有 `--output` 時，自動印 summary 到 stderr（與既有「Saved to …」一致方向，不污染 stdout 的 JSON 模式）
2. 有 `--output` 時，自動在**同目錄**產生 `phase1_summary.md`

**為何選 stderr 而非 stdout？** 無 `--output` 時 stdout 已是 JSON，不能污染；有 `--output` 時 stdout 為空但 stderr 已有 `Saved to` — 保持同通道一致且安全。

**為何 summary 放在同目錄而非自訂路徑？** `$GAME_DIR/` 是既有慣例（phase3_summary.md 也在裡面），統一好找。檔名固定為 `phase1_summary.md`。

**Files:**
- Modify: `scripts/fetch_game_data.py`（新增 `write_outputs()`，修改 `main()`）
- Modify: `scripts/tests/test_fetch_game_data.py`（新增測試）

- [ ] **Step 1：寫失敗測試**

在 `scripts/tests/test_fetch_game_data.py` 加入：

```python
def test_write_outputs_creates_json_summary_file_and_prints_to_stderr(tmp_path, capsys):
    from fetch_game_data import write_outputs

    result = _sample_result()
    output_path = tmp_path / "game_data.json"

    write_outputs(result, str(output_path))

    # JSON 檔有寫
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["game"]["gamePk"] == 824770

    # phase1_summary.md 在同目錄
    summary_path = output_path.parent / "phase1_summary.md"
    assert summary_path.exists()
    md = summary_path.read_text(encoding="utf-8")
    assert "New York Yankees" in md
    assert "Cam Schlittler" in md
    assert "Fenway Park" in md

    # stderr 有印 summary（至少包含 `Saved to` 與某個隊名）
    captured = capsys.readouterr()
    assert "Saved to" in captured.err
    assert "New York Yankees" in captured.err


def test_write_outputs_without_output_path_prints_json_to_stdout(capsys):
    from fetch_game_data import write_outputs

    result = _sample_result()
    write_outputs(result, None)

    captured = capsys.readouterr()
    # stdout 該有完整 JSON
    parsed = json.loads(captured.out)
    assert parsed["game"]["gamePk"] == 824770
    # stderr 不該有 summary（無 --output 模式保持原樣）
    assert "Saved to" not in captured.err
```

- [ ] **Step 2：跑測試確認失敗**

Run:
```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: FAIL（`write_outputs` not defined）

- [ ] **Step 3：實作 `write_outputs()` 並重構 `main()`**

在 `build_summary_lines` 之後、`main()` 之前新增：

```python
def write_outputs(result: dict, output_path: str | None) -> None:
    """將 result 寫出：JSON 檔 + phase1_summary.md（若有 output_path）+ stderr 人類摘要。

    無 output_path 時：僅印 JSON 到 stdout（保持向下相容的 pipe-friendly 模式）。
    """
    json_output = json.dumps(result, indent=2, ensure_ascii=False)

    if output_path is None:
        print(json_output)
        return

    # 1. 寫 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_output)

    # 2. 寫 phase1_summary.md（同目錄，固定檔名）
    from pathlib import Path

    summary_lines = build_summary_lines(result)
    summary_md_body = "# Phase 1 Summary\n\n" + "\n".join(summary_lines) + "\n"
    summary_path = Path(output_path).parent / "phase1_summary.md"
    summary_path.write_text(summary_md_body, encoding="utf-8")

    # 3. 印 stderr（Saved to + summary 本文）
    print(f"Saved to {output_path}", file=sys.stderr)
    print(f"Summary written to {summary_path}", file=sys.stderr)
    print("", file=sys.stderr)
    for line in summary_lines:
        print(line, file=sys.stderr)
```

然後修改 `main()` 的末段（line 327-334 之後），把原本的：

```python
    json_output = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(json_output)
```

整段**替換為**：

```python
    write_outputs(result, args.output)
```

- [ ] **Step 4：跑測試確認通過**

Run:
```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: PASS（4 passed 累計）

- [ ] **Step 5：人工驗證（非自動化）**

跑一次實際 fetch 看 summary 實際長相，確認不會有亂碼或 emoji 在 Windows cmd 破圖：

```bash
python scripts/fetch_game_data.py --date 2026-04-23 --team NYY -o analysis-data/2026-04-23/NYY@BOS/game_data.json
```

Expected:
- stderr 印出 `Saved to ...`、`Summary written to ...\phase1_summary.md`、接著 workflow 1.4 模板的人類摘要
- `analysis-data/2026-04-23/NYY@BOS/phase1_summary.md` 存在且可讀

- [ ] **Step 6：Commit**

```bash
git add scripts/fetch_game_data.py scripts/tests/test_fetch_game_data.py
git commit -m "feat(fetch_game_data): 有 --output 時自動產出 phase1_summary.md + stderr summary

- write_outputs() 整合三路輸出（JSON 檔、phase1_summary.md、stderr 摘要）
- 無 --output 模式維持原 stdout JSON 行為（pipe-friendly）
- 解除 LLM 需要額外 Read 1000+ 行 JSON 的痛點"
```

---

## Task 3：`equals_season` 旗標 — 減少早季冗餘

當 `len(season_games) <= 30` 時，近 30 與本季窗口必然內容相同。在 `home_recent_30` / `away_recent_30` 兩個區塊加 `"equals_season": true/false`，讓 LLM 讀 JSON 時能快速跳過重複內容。

**Files:**
- Modify: `scripts/fetch_game_data.py`（line 305-309 附近）
- Modify: `scripts/tests/test_fetch_game_data.py`

- [ ] **Step 1：寫失敗測試**

在 test 檔加入（這個測試不依賴 network，用 `_sample_result` 無法驗證 — 要用更底層的 assertion，改測 `compute_recent_stats` 無法覆蓋；所以改測 `build_recent_30_block` 這個新 helper 的行為）：

先決定實作方式：抽一個 `build_recent_30_block(season_games: list[dict]) -> dict`，輸出 `compute_recent_stats(subset)` 結果 + `equals_season` 旗標。

```python
def test_build_recent_30_block_flags_equals_season_when_under_30():
    from fetch_game_data import build_recent_30_block

    games = [
        {"date": f"2026-04-{22-i:02d}", "is_home": True, "opponent": "X",
         "team_score": 3, "opp_score": 2, "is_winner": True}
        for i in range(24)
    ]
    block = build_recent_30_block(games)
    assert block["equals_season"] is True
    assert block["wins"] == 24


def test_build_recent_30_block_flags_false_when_over_30():
    from fetch_game_data import build_recent_30_block

    games = [
        {"date": f"2026-0{4 if i < 22 else 5}-{((i % 30) + 1):02d}",
         "is_home": True, "opponent": "X",
         "team_score": 3, "opp_score": 2, "is_winner": True}
        for i in range(45)
    ]
    block = build_recent_30_block(games)
    assert block["equals_season"] is False
    # 應只取前 30 場統計
    assert block["wins"] == 30


def test_build_recent_30_block_flags_true_at_exactly_30():
    from fetch_game_data import build_recent_30_block

    games = [
        {"date": f"2026-04-{(i%28)+1:02d}", "is_home": True, "opponent": "X",
         "team_score": 3, "opp_score": 2, "is_winner": True}
        for i in range(30)
    ]
    block = build_recent_30_block(games)
    # 邊界：剛好 30 場時，近 30 = 本季，應為 True
    assert block["equals_season"] is True
```

- [ ] **Step 2：跑測試確認失敗**

Run:
```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: FAIL（`build_recent_30_block` not defined）

- [ ] **Step 3：實作 `build_recent_30_block()` 並更新 `main()`**

在 `compute_recent_stats` 之後（約 line 221 之後）加入：

```python
def build_recent_30_block(season_games: list[dict]) -> dict:
    """從本季全部比賽產出「近 30 場」統計區塊，並標記是否與本季完全重疊。

    Args:
        season_games: 本季全部已完成比賽（由 fetch_season_games 產出，已按日期新到舊排序）。

    Returns:
        dict: compute_recent_stats 結果 + "equals_season" 旗標。
    """
    subset = season_games[:30]
    block = compute_recent_stats(subset)
    block["equals_season"] = len(season_games) <= 30
    return block
```

修改 `main()` 中的 line 306-309：

```python
    # 近 30 場（中期趨勢）— 如果不到 30 場則等於本季全部
    home_30_games = home_season_games[:30] if len(home_season_games) >= 30 else home_season_games
    away_30_games = away_season_games[:30] if len(away_season_games) >= 30 else away_season_games
    home_recent_30 = compute_recent_stats(home_30_games)
    away_recent_30 = compute_recent_stats(away_30_games)
```

**替換為：**

```python
    # 近 30 場（中期趨勢）— 附 equals_season 旗標，方便 LLM 跳過早季冗餘
    home_recent_30 = build_recent_30_block(home_season_games)
    away_recent_30 = build_recent_30_block(away_season_games)
```

- [ ] **Step 4：跑測試確認通過**

Run:
```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```
Expected: PASS（7 passed 累計）

- [ ] **Step 5：下游回歸驗證 — 跑 merge 測試**

`merge_game_data.py` 會讀 `home_recent_30` 區塊，確保新增欄位不破壞既有行為：

```bash
python -m pytest scripts/tests/test_merge_game_data.py -v
```
Expected: PASS（既有測試全通過，證明新增欄位為加法，下游忽略）

- [ ] **Step 6：Commit**

```bash
git add scripts/fetch_game_data.py scripts/tests/test_fetch_game_data.py
git commit -m "feat(fetch_game_data): 早季近 30 = 本季時加 equals_season 旗標

當 len(season_games) <= 30 時（賽季前一個月），
近 30 場窗口與本季窗口內容完全相同，LLM 讀 JSON 時浪費 token。
新增 equals_season: true/false 旗標讓下游消費者可快速跳過。
下游契約為加法，merge_game_data.py 回歸測試通過。"
```

---

## Task 4：更新 `reference/workflow.md` Phase 1 章節

讓文件敘述與新行為同步。現在 1.4 的 `📅📊` 模板不再只是「應該長這樣」的期待，而是 `fetch_game_data.py` 的實際 stderr 輸出；同時新增 `phase1_summary.md` 的落地描述。

**Files:**
- Modify: `reference/workflow.md`（line 74-96 附近）

- [ ] **Step 1：編輯 1.4 章節**

把現行 workflow.md 的 `### 1.4 輸出確認` 區塊（line 74-87）替換為：

```markdown
### 1.4 輸出確認

**有 `--output` 時，`fetch_game_data.py` 會自動做三件事：**

1. 寫 JSON 到 `--output` 指定路徑（既有行為）
2. 在同目錄產生 `phase1_summary.md`（Phase 1 資料快照，跨對話可讀取）
3. 將人類可讀摘要印到 stderr，格式如下：

```
📅 {日期} — {客隊} @ {主隊}（{球場}）
⚾ 先發：{客隊投手} (客) vs {主隊投手} (主)
🕐 狀態：{Preview/Live/Final} | gamePk：{gamePk}

【{主隊}】
📊 近 10 場：{W}-{L}（RS/G {X} | RA/G {Y} | 連勝/連敗 N）
📊 近 30 場：{W}-{L}（RS/G {X} | RA/G {Y} | 連勝/連敗 N）
📊 本季（{N} 場）：{W}-{L}（RS/G {X} | RA/G {Y} | 連勝/連敗 N）

【{客隊}】
📊 近 10 場：...
📊 近 30 場：...
📊 本季（{N} 場）：...

🔁 同系列前場（{日期}）：{客} @ {主} → {比分}，勝者：{隊名}（若有）
```

> **早季注意**：當 `home_recent_30` / `away_recent_30` 的 `equals_season == true`（本季 ≤ 30 場），近 30 窗口與本季完全重疊，不提供獨立訊號，做趨勢分析時應視為單一窗口。
```

然後**刪除舊的 `### 1.5 Phase 1 閘門`**（line 89-96）前面的分隔符號（該節保留），並**在 1.4 與 1.5 之間**新增：

```markdown
### 1.5 phase1_summary.md 落地

⛔ 進入 Phase 2 前必須：

- [ ] `$GAME_DIR/phase1_summary.md` 存在（由 `fetch_game_data.py --output` 自動產生）
- [ ] md 檔內容與 `game_data.json` 對得起來（無產出失敗殘留）

> 此檔案確保 Phase 2+ 開始時，即使對話壓縮，LLM 仍可透過 Read `phase1_summary.md` 快速回復比賽基本面 context，無需重跑腳本或解 JSON。
```

**原本的 `### 1.5 Phase 1 閘門`** 重新編號為 `### 1.6 Phase 1 閘門`。

- [ ] **Step 2：Commit**

```bash
git add reference/workflow.md
git commit -m "docs(workflow): Phase 1.4 同步 fetch_game_data 新輸出，新增 1.5 summary md 落地閘門

- 1.4 stderr 模板從「應該長這樣」改為腳本實際輸出
- 新增 1.5 phase1_summary.md 落地作為 Phase 2 前置閘門
- Phase 1 閘門重編為 1.6
- 備註 equals_season 旗標對早季趨勢分析的意義"
```

---

## 完成條件（Definition of Done）

- [ ] `python -m pytest scripts/tests/test_fetch_game_data.py -v` 全 PASS（至少 7 test cases）
- [ ] `python -m pytest scripts/tests/test_merge_game_data.py -v` 全 PASS（下游無破壞）
- [ ] 實跑 `python scripts/fetch_game_data.py --date 2026-04-23 --team NYY -o <path>`：
  - stderr 出現 `📅 ...` 人類摘要
  - `<path> 同目錄/phase1_summary.md` 存在且可讀
  - JSON 中 `home_recent_30.equals_season` / `away_recent_30.equals_season` 為 bool
- [ ] `reference/workflow.md` 1.4-1.6 章節與實際行為一致
- [ ] 4 個 commit 歷史清晰（一個 task 一個 commit）

---

## 風險與反制

| 風險 | 反制 |
|---|---|
| 新增 `equals_season` 欄位意外破壞 `merge_game_data.py` | Task 3 Step 5 強制跑下游回歸測試 |
| stderr summary emoji 在 Windows cmd 亂碼 | Task 2 Step 5 人工驗證；若真亂碼，fallback 可改用 ASCII 前綴（`[G]`/`[P]`/`[S]`），但優先保留 emoji 因為 PowerShell/Windows Terminal 支援無虞 |
| `phase1_summary.md` 跟 `game_data.json` 漂移（不同步） | 兩者由**同一個函式呼叫同時寫出**（`write_outputs`），不給漂移機會；測試覆蓋兩者一致性 |
| 未來 workflow.md 新增 Phase 1.5 但腳本沒同步改 `build_summary_lines` | 這是未來事；現階段 workflow.md 的模板由測試間接驗證（test 確認隊名、投手、venue、戰績都入列） |
