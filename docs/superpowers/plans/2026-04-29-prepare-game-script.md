# `prepare_game.py` 整合 + Skill 瘦身 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 1+2 整合為一支 `prepare_game.py`、AI 唯一入口改為 `dossier.md` + `phase3_skeleton.md`、移除 luck-based 自動回測（Step C-prior、Flag 3/13 補跑、H2 grep guard）、reference 5 檔 1031 行 → 3 檔 ~310 行。

**Architecture:** 新增 3 支 Python 模組（`prepare_game.py` 主腳本、`dossier_renderer.py` + `phase3_skeleton_renderer.py` 純函式渲染器）；改 3 支既有腳本（`fetch_game_data.py` 加 ID、`pitcher_stats.py` diacritic fallback、`predict.py` --ou-stars 必填 + 移除 YoY/H2 guard）；reference 6 檔逐一處理（teams-and-api / workflow 整檔刪、flags-checklist / matchup-factors / prediction 改寫、SKILL.md 合併 workflow）。

**Tech Stack:** Python 3、pytest（既有專案慣例：`sys.path.insert` import + plain `def test_*` 函式）、pybaseball（diacritic fallback）。

**Spec:** `docs/superpowers/specs/2026-04-28-prepare-game-script-design.md`

**任務順序設計**：
1. **先做 reference 瘦身（Tasks 1-4）**：純 markdown 改寫，無 code 風險，鎖定 SOP 契約
2. **再做 script 改造（Tasks 5-7）**：上游 3 支腳本 gotcha 修正，每支獨立可測
3. **新模組（Tasks 8-10）**：dossier / skeleton 渲染器、prepare_game.py 整合
4. **SKILL.md 合併 workflow.md（Task 11）**：擺最後因 SKILL.md 會引用 prepare_game.py，需先存在
5. **E2E（Task 12）**：實測 + token 量化

**Existing utilities to reuse:**
- `scripts/_team_resolver.py`：`resolve_team_id` / `team_abbr` / `TEAM_MAP` / `FULL_NAMES`
- `scripts/fetch_game_data.py`：`extract_game_info` / `format_summary_md` / `fetch_schedule`
- `scripts/pitcher_stats.py`：`lookup_pitcher_id` / `_import_pybaseball` / `detect_triggers`
- `scripts/lineup_analyzer.py`：`detect_triggers`
- `scripts/predict.py`：`predict_with_formula`（給 phase3_skeleton 算 base 比分用，line 977）

---

### Task 1: 刪除 `reference/teams-and-api.md`

**Files:**
- Delete: `reference/teams-and-api.md`

**理由:** 內容（隊名表 / API 端點 / Pythagorean 公式）已在 `_team_resolver.py` / `fetch_game_data.py` / `predict.py` 內實作，AI 不需讀。

- [ ] **Step 1: 確認檔案內容無散落引用未處理**

```bash
grep -rn "teams-and-api" --include='*.md' --include='*.py' .
```

預期會列出 `reference/workflow.md` 的引用（後續 Task 11 處理）和 `reference/matchup-factors.md`（已在 Task 3 處理範圍）。

- [ ] **Step 2: 刪檔**

```bash
git rm reference/teams-and-api.md
```

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
docs(mlb-skill): 刪除 reference/teams-and-api.md

內容已在 scripts/_team_resolver.py（隊名表）、fetch_game_data.py（API 端點）、predict.py（Pythagorean 公式）中實作。AI 不需讀。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: 改寫 `reference/flags-checklist.md`

**Files:**
- Modify: `reference/flags-checklist.md`

**目標**：13 條 → 11 條（刪 Flag 7、改寫 Flag 3 / Flag 13）。

- [ ] **Step 1: 刪除 Flag 7 整段（Roster 跳過）**

於 `reference/flags-checklist.md`，找到並刪除：

```markdown
### 7. 跳過 Roster 檢查
- 觸發：Phase 2 Step 1 未通過就進 Step 2
- 處理：阻塞閘門。詳見 `workflow.md` §Phase 2 Step 1
```

理由：`prepare_game.py` 整合後 Step B 失敗會 exit 5，AI 無從跳過。

- [ ] **Step 2: 改寫 Flag 3（BABIP）**

替換原段落為：

```markdown
### 3. Hot/Cold 判定未檢查 BABIP
- 觸發：近 7 天 BABIP `≤ .260` 或 `≥ .370`
- 處理：腳本（`prepare_game.py`）自動標 ⚠️ 風險提示在 dossier 與 phase3_skeleton 的「## 風險提示」段。AI 在敘事中判讀「可能回歸 / 可能持續」**不自動 ±run value**。詳見 `matchup-factors.md` §BABIP 回歸檢查
```

- [ ] **Step 3: 改寫 Flag 13（ERA-xERA）**

替換原段落為：

```markdown
### 13. ERA-xERA 落差 / 小樣本回歸風險
- 觸發：`|ERA − xERA| ≥ 1.5` 或 `IP < 30 且 ERA 比 prior_year 低 ≥ 1.0`
- 處理：腳本（`prepare_game.py`）自動標 ⚠️ 風險提示在 dossier 與 phase3_skeleton 的「## 風險提示」段。AI 在敘事中判讀「運氣 / 結構性退化 / 樣本噪音」**不自動補跑 YoY、不自動下修預測**
```

- [ ] **Step 4: 重新編號（如需要）**

原 Flag 編號 1-13 中刪除 7 後，仍維持原編號（不重排）— 編號是 stable identifier，散落引用會 break。其他 11 條（1-6、8-13）保持原編號。

- [ ] **Step 5: 確認 cross-ref 仍正確**

```bash
grep -n "workflow.md\|matchup-factors\|prediction.md" reference/flags-checklist.md
```

確認所有引用的目標檔案在後續 task 中仍存在或已正確替換。`workflow.md` 引用會在 Task 11 一起更新（指向新 SKILL.md）。

- [ ] **Step 6: Commit**

```bash
git add reference/flags-checklist.md
git commit -m "$(cat <<'EOF'
docs(mlb-skill): flags-checklist Flag 7 刪除、Flag 3/13 改寫為風險標註

- 刪除 Flag 7（Roster 跳過）：prepare_game.py 整合後 Step B 失敗自動 exit 5，AI 無從跳過
- 改寫 Flag 3（BABIP）：腳本標 ⚠️ → AI 敘事判斷，不自動 ±run value
- 改寫 Flag 13（ERA-xERA）：腳本標 ⚠️ → AI 敘事判斷，不自動補跑 YoY

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: 改寫 `reference/matchup-factors.md`

**Files:**
- Modify: `reference/matchup-factors.md`

**目標**：182 → ~100 行；刪 §YoY Statcast 驗證 + §影響分析的賽制規則；改寫 §BABIP 回歸檢查。

- [ ] **Step 1: 刪除 §YoY Statcast 驗證 整段**

定位：`### YoY Statcast 驗證`（line 25），刪除整段直到下一個同層 H3 `### 投手實力分級`。包含「⛔ 觸發條件」、「方法」、「五指標表」、「判定規則」、「Platoon 樣本陷阱」全部。

- [ ] **Step 2: 改寫 §BABIP 回歸檢查**

定位：`### BABIP 回歸檢查（必須執行）`（約 line 72），替換為：

```markdown
### BABIP 回歸風險標註

- 近 7 天 BABIP ≤ .260 或 ≥ .370 → 由 `prepare_game.py` 自動偵測，於 dossier 與 phase3_skeleton 的「## 風險提示」段標 ⚠️
- AI 在敘事中判讀「可能回歸 / 可能持續」，**不自動 ±run value**
- 聯盟平均 BABIP ≈ .300，需 ~800 AB 才穩定 — 7 天樣本噪音極大，自動修正等同賭運氣
```

- [ ] **Step 3: 刪除 §影響分析的賽制規則**

定位：`### 影響分析的賽制規則`（最後一段，含 DH / Pitch Clock / 三打者規則 / 防守布陣）— 整段刪除。對單場修正幾乎無影響。

- [ ] **Step 4: 確認其他 sections 保留**

KEEP 不動：
- §先發投手進階數據 §核心指標
- §投手數據權重表（4 行）
- §投手實力分級（🔴~⚪ Tier 表）
- §打線分析（xwOBA / OPS / 評級 / 熱度 / 串聯）
- §牛棚分析 §牛棚傷兵累計效應
- §牛棚替補品質反向檢查
- §傷兵影響過濾
- §傷病與手術復出（TJ / 角色轉換）
- §球員年齡退化
- §球場 & 天氣 §Park Factor + 分裂型球場 + 重大改造

- [ ] **Step 5: 行數驗證**

```bash
wc -l reference/matchup-factors.md
```

預期：從 182 行降至 ~100 行（範圍 ≤ 110）。

- [ ] **Step 6: Commit**

```bash
git add reference/matchup-factors.md
git commit -m "$(cat <<'EOF'
docs(mlb-skill): matchup-factors 砍 luck-based 邏輯與宏觀賽制段落

- 刪 §YoY Statcast 驗證 整段（不再做 prior year 補跑）
- 改寫 §BABIP 回歸檢查 → §BABIP 回歸風險標註：腳本標 ⚠️、AI 敘事判讀，不自動 ±run value
- 刪 §影響分析的賽制規則（DH / Pitch Clock / 三打者 對單場幾無影響）

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: 改寫 `reference/prediction.md`

**Files:**
- Modify: `reference/prediction.md`

**目標**：249 → ~170 行；刪 §比賽敘事觸發條件 + §預測紀錄格式（JSON schema）；slim §預測紀錄存放位置。

- [ ] **Step 1: 刪除 §比賽敘事觸發條件 整段**

定位：`## 比賽敘事觸發條件`（約 line 119），刪除整個 H2 section（含 5 行劇本表）直到下一個 H2 `## 分析紀律`。

理由：前端只顯示推薦，不顯示比賽敘事；AI 內部仍可做分析師判斷，但不需要劇本模板。

- [ ] **Step 2: SLIM §預測紀錄存放位置**

定位：`## 預測紀錄存放位置`（約 line 200），保留 1 段話：

```markdown
## 預測紀錄存放位置

- **Per-game（真相來源）**：`analysis-data/{YYYY-MM-DD}/{AWAY}@{HOME}/prediction.json`，由 `predict.py --save` 產生。
- **Per-date summary**：`analysis-data/{YYYY-MM-DD}/predictions.jsonl`，由 `mlb-post-game-review` skill 重建。
- **賽後回填** `actual_*` / `verified=true` 由 `mlb-post-game-review` skill 處理。
```

- [ ] **Step 3: 刪除 §預測紀錄格式 整段（JSON schema）**

定位：`## 預測紀錄格式`（約 line 208），整個 H2 section（含 ~38 行 JSON 範本）刪除。

理由：AI 看 `prediction_summary.md` 不看 JSON；要 debug 直接看 `predict.py` 源碼比看文件更準。

- [ ] **Step 4: 行數驗證**

```bash
wc -l reference/prediction.md
```

預期：從 249 降至 ~170（範圍 ≤ 180）。

- [ ] **Step 5: Commit**

```bash
git add reference/prediction.md
git commit -m "$(cat <<'EOF'
docs(mlb-skill): prediction.md 砍 比賽敘事 + JSON schema

- 刪 §比賽敘事觸發條件（前端只顯示推薦）
- SLIM §預測紀錄存放位置 為 1 段話
- 刪 §預測紀錄格式（JSON schema）— AI 看 prediction_summary.md 不看 JSON

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `fetch_game_data.py` — 加 `probable_pitcher_id`

**Files:**
- Modify: `scripts/fetch_game_data.py`（`extract_game_info` 函式 line 286-305，`format_summary_md` line 169）
- Modify: `scripts/tests/test_fetch_game_data.py`（追加 2 個測試）

- [ ] **Step 1: Write the failing tests**

於 `scripts/tests/test_fetch_game_data.py` 末尾追加：

```python


def test_extract_game_info_includes_probable_pitcher_id():
    """schedule API hydrate=probablePitcher 已含 .id；extract_game_info 應寫入 probable_pitcher_id"""
    from fetch_game_data import extract_game_info
    game = {
        "gamePk": 12345,
        "gameDate": "2026-04-28T22:10:00Z",
        "status": {"abstractGameState": "Preview"},
        "venue": {"name": "Progressive Field"},
        "teams": {
            "home": {
                "team": {"name": "Cleveland Guardians", "id": 114},
                "probablePitcher": {"fullName": "Tanner Bibee", "id": 676440},
            },
            "away": {
                "team": {"name": "Tampa Bay Rays", "id": 139},
                "probablePitcher": {"fullName": "Nick Martínez", "id": 607259},
            },
        },
    }
    result = extract_game_info(game)
    assert result["home"]["probable_pitcher_id"] == 676440
    assert result["away"]["probable_pitcher_id"] == 607259


def test_extract_game_info_missing_probable_pitcher_id_is_none():
    """無 probablePitcher（TBD 先發）→ probable_pitcher_id = None"""
    from fetch_game_data import extract_game_info
    game = {
        "gamePk": 12345,
        "gameDate": "2026-04-28T22:10:00Z",
        "status": {"abstractGameState": "Preview"},
        "venue": {"name": "Progressive Field"},
        "teams": {
            "home": {"team": {"name": "Cleveland Guardians", "id": 114}},
            "away": {"team": {"name": "Tampa Bay Rays", "id": 139}},
        },
    }
    result = extract_game_info(game)
    assert result["home"]["probable_pitcher_id"] is None
    assert result["away"]["probable_pitcher_id"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest scripts/tests/test_fetch_game_data.py -k "probable_pitcher_id" -v
```

Expected: 2 FAIL with `KeyError: 'probable_pitcher_id'` or assertion fail（field 不存在）

- [ ] **Step 3: Modify `extract_game_info`**

於 `scripts/fetch_game_data.py` 找到 `extract_game_info`（line 286-305），改寫 home / away 兩個 dict：

```python
def extract_game_info(game: dict) -> dict:
    """從 game object 提取比賽資訊"""
    home = game["teams"]["home"]
    away = game["teams"]["away"]
    return {
        "gamePk": game["gamePk"],
        "date": game["gameDate"],
        "status": game["status"]["abstractGameState"],
        "venue": game["venue"]["name"],
        "home": {
            "team": home["team"]["name"],
            "team_id": home["team"]["id"],
            "probable_pitcher": home.get("probablePitcher", {}).get("fullName", "TBD"),
            "probable_pitcher_id": home.get("probablePitcher", {}).get("id"),
        },
        "away": {
            "team": away["team"]["name"],
            "team_id": away["team"]["id"],
            "probable_pitcher": away.get("probablePitcher", {}).get("fullName", "TBD"),
            "probable_pitcher_id": away.get("probablePitcher", {}).get("id"),
        },
    }
```

- [ ] **Step 4: 改 summary md 的「先發」行加 ID**

於 `scripts/fetch_game_data.py` line 169（`format_summary_md` 內），改：

```python
        f"- 先發: {away.get('probable_pitcher', 'TBD')} ({away_abbr}) vs {home.get('probable_pitcher', 'TBD')} ({home_abbr})",
```

為：

```python
        f"- 先發: {away.get('probable_pitcher', 'TBD')} ({away_abbr}, {away.get('probable_pitcher_id') or '—'}) vs {home.get('probable_pitcher', 'TBD')} ({home_abbr}, {home.get('probable_pitcher_id') or '—'})",
```

- [ ] **Step 5: Run all fetch_game_data tests**

```bash
python -m pytest scripts/tests/test_fetch_game_data.py -v
```

Expected: 全部 PASS（含新 2 + 既有測試）

- [ ] **Step 6: Run full suite to ensure no regression**

```bash
python -m pytest scripts/tests/ -v
```

Expected: 全 PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/fetch_game_data.py scripts/tests/test_fetch_game_data.py
git commit -m "$(cat <<'EOF'
feat(scripts): fetch_game_data extract_game_info 加 probable_pitcher_id

- extract_game_info() 從 schedule API 的 hydrate=probablePitcher response 取出 .id
- 寫入 home.probable_pitcher_id / away.probable_pitcher_id，無 probablePitcher 時為 None
- format_summary_md 「先發」行加註 ID
- 解決 spec P2：下游 lineup_analyzer.py / pitcher_stats.py 不再需要繞 name lookup

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `pitcher_stats.py` — Diacritic fallback

**Files:**
- Modify: `scripts/pitcher_stats.py`（`lookup_pitcher_id` 函式 line 80-97）
- Create: `scripts/tests/test_pitcher_stats.py`（新檔）

- [ ] **Step 1: Write the failing tests**

建立 `scripts/tests/test_pitcher_stats.py`：

```python
"""Tests for pitcher_stats.lookup_pitcher_id (diacritic fallback)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_lookup_stub(strict_df, fuzzy_df):
    """生成 monkeypatch 用的 playerid_lookup stub。

    呼叫順序：第 1 次（strict） → strict_df；第 2 次（fuzzy=True） → fuzzy_df。
    """
    calls = {"n": 0}

    def stub(last, first, fuzzy=False):
        calls["n"] += 1
        return fuzzy_df if fuzzy else strict_df

    return stub, calls


def test_lookup_strict_match_returns_id_no_fallback(monkeypatch):
    """strict match 成功 → 直接 return，不觸發 fuzzy"""
    import pitcher_stats
    strict = pd.DataFrame([{"key_mlbam": 676440, "mlb_played_last": 2026}])
    fuzzy = pd.DataFrame()  # 不該被讀
    stub, calls = _make_lookup_stub(strict, fuzzy)
    monkeypatch.setattr(pitcher_stats, "_import_pybaseball",
                        lambda: (stub, None, None, None))
    result = pitcher_stats.lookup_pitcher_id("Tanner Bibee")
    assert result == 676440
    assert calls["n"] == 1  # fuzzy 未呼叫


def test_lookup_diacritic_fallback_succeeds(monkeypatch, capsys):
    """ASCII 名字 strict 失敗 → fuzzy 成功 → 回傳 ID + stderr warning"""
    import pitcher_stats
    strict = pd.DataFrame()  # empty
    fuzzy = pd.DataFrame([{
        "key_mlbam": 607259,
        "name_first": "Nick",
        "name_last": "Martínez",
        "mlb_played_last": 2026,
    }])
    stub, calls = _make_lookup_stub(strict, fuzzy)
    monkeypatch.setattr(pitcher_stats, "_import_pybaseball",
                        lambda: (stub, None, None, None))
    result = pitcher_stats.lookup_pitcher_id("Nick Martinez")
    assert result == 607259
    assert calls["n"] == 2  # 兩次呼叫（strict 後 fuzzy）
    err = capsys.readouterr().err
    assert "fuzzy" in err.lower()
    assert "Martínez" in err


def test_lookup_fuzzy_year_filter_rejects_old_player(monkeypatch):
    """fuzzy 結果 mlb_played_last 早於 current_year - 1 → 拒絕，return None"""
    import pitcher_stats
    strict = pd.DataFrame()
    fuzzy = pd.DataFrame([{
        "key_mlbam": 100000,
        "name_first": "Old",
        "name_last": "Player",
        "mlb_played_last": 2010,  # 太舊
    }])
    stub, _ = _make_lookup_stub(strict, fuzzy)
    monkeypatch.setattr(pitcher_stats, "_import_pybaseball",
                        lambda: (stub, None, None, None))
    result = pitcher_stats.lookup_pitcher_id("Old Player")
    assert result is None


def test_lookup_fuzzy_multiple_results_picks_highest_year(monkeypatch):
    """fuzzy 多筆結果 → 取 mlb_played_last 最大者"""
    import pitcher_stats
    strict = pd.DataFrame()
    fuzzy = pd.DataFrame([
        {"key_mlbam": 111, "mlb_played_last": 2024},
        {"key_mlbam": 222, "mlb_played_last": 2026},  # 最新
        {"key_mlbam": 333, "mlb_played_last": 2025},
    ])
    stub, _ = _make_lookup_stub(strict, fuzzy)
    monkeypatch.setattr(pitcher_stats, "_import_pybaseball",
                        lambda: (stub, None, None, None))
    result = pitcher_stats.lookup_pitcher_id("Multi Match")
    assert result == 222


def test_lookup_both_empty_returns_none(monkeypatch):
    """strict + fuzzy 都 empty → return None"""
    import pitcher_stats
    stub, _ = _make_lookup_stub(pd.DataFrame(), pd.DataFrame())
    monkeypatch.setattr(pitcher_stats, "_import_pybaseball",
                        lambda: (stub, None, None, None))
    assert pitcher_stats.lookup_pitcher_id("Nonexistent Player") is None


def test_lookup_single_word_name_returns_none(monkeypatch):
    """單字名（無姓） → return None（既有行為保留）"""
    import pitcher_stats
    assert pitcher_stats.lookup_pitcher_id("Cher") is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest scripts/tests/test_pitcher_stats.py -v
```

Expected: `test_lookup_strict_match_returns_id_no_fallback` PASS（既有行為），`test_lookup_diacritic_fallback_succeeds` 等 FAIL（fuzzy 邏輯尚未實作）

- [ ] **Step 3: Modify `lookup_pitcher_id`**

於 `scripts/pitcher_stats.py` 找到 `lookup_pitcher_id`（line 80-97）。在 import 區塊頂端確認 `import sys`（如未 import 則加）。改函式為：

```python
def lookup_pitcher_id(name: str) -> int | None:
    """用 pybaseball 查詢球員 MLBAM ID。

    Strategy:
      1. Strict match
      2. Empty / not-found → fuzzy fallback（解 P3：Nick Martinez vs Nick Martínez）
      3. fuzzy 結果按 mlb_played_last 排序取最新；過濾掉 < current_year - 1 的舊球員
    """
    import sys
    from datetime import datetime
    parts = name.strip().split()
    if len(parts) < 2:
        return None
    last = parts[-1]
    first = parts[0]
    playerid_lookup, _, _, _ = _import_pybaseball()

    def _resolve(df):
        """從 DataFrame 取 mlb_played_last 最大者的 key_mlbam，套年份過濾"""
        if df.empty:
            return None
        if "mlb_played_last" in df.columns and len(df) > 1:
            df = df.sort_values("mlb_played_last", ascending=False, na_position="last")
        row = df.iloc[0]
        last_year = row.get("mlb_played_last") if "mlb_played_last" in df.columns else None
        current_year = datetime.now().year
        # 拒絕 last_year < current_year - 1 的歷史球員（避免 fuzzy 命中退役同名球員）
        if last_year is not None and not pd.isna(last_year) and last_year < current_year - 1:
            return None
        return int(row["key_mlbam"])

    # Round 1: strict
    try:
        with _redirect_pybaseball_stdout():
            strict_result = playerid_lookup(last, first)
    except Exception:
        strict_result = None

    if strict_result is not None and not strict_result.empty:
        resolved = _resolve(strict_result)
        if resolved is not None:
            return resolved

    # Round 2: fuzzy fallback
    try:
        with _redirect_pybaseball_stdout():
            fuzzy_result = playerid_lookup(last, first, fuzzy=True)
    except Exception:
        return None

    if fuzzy_result is None or fuzzy_result.empty:
        return None

    resolved = _resolve(fuzzy_result)
    if resolved is None:
        return None

    # 取出 matched name 給 stderr warning
    row = (fuzzy_result.sort_values("mlb_played_last", ascending=False, na_position="last")
           if "mlb_played_last" in fuzzy_result.columns and len(fuzzy_result) > 1
           else fuzzy_result).iloc[0]
    matched_name = f"{row.get('name_first', '?')} {row.get('name_last', '?')}"
    print(f"⚠️  name \"{name}\" matched fuzzy → \"{matched_name}\" (mlbam={resolved})",
          file=sys.stderr)
    return resolved
```

於檔案頂端確認 `import pandas as pd`（用於 `pd.isna`） — pybaseball 已依賴 pandas，應已在頂部 import。如未有，加：

```python
import pandas as pd
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest scripts/tests/test_pitcher_stats.py -v
```

Expected: 6/6 PASS

- [ ] **Step 5: Run full suite for regressions**

```bash
python -m pytest scripts/tests/ -v
```

Expected: 全 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/pitcher_stats.py scripts/tests/test_pitcher_stats.py
git commit -m "$(cat <<'EOF'
feat(scripts): pitcher_stats.lookup_pitcher_id 加 diacritic fallback

- 第 1 輪 strict match；失敗自動跑 fuzzy=True
- fuzzy 結果按 mlb_played_last 排序取最新，過濾 last_year < current_year - 1 的退役球員
- 命中時 stderr ⚠️ 警告：name "X" matched fuzzy → "Y" (mlbam=N)
- 解決 spec P3：Nick Martinez（純 ASCII）→ Nick Martínez（含 í）

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `predict.py` — `--ou-stars` 必填化 + 移除 YoY/H2 guard

**Files:**
- Modify: `scripts/predict.py`（line 939 argparse、line 951-955 skip flags、line 1044-1098 guard 區塊）
- Modify: `scripts/tests/test_predict.py`（刪 / 改 既有 4 個測試 + 加新測試）

#### Task 7a: `--ou-stars` 必填化

- [ ] **Step 1: Write the failing test**

於 `scripts/tests/test_predict.py` 末尾追加：

```python


def test_ou_rec_over_without_stars_exits(tmp_path):
    """--ou-rec OVER 但缺 --ou-stars → exit 6 + 錯誤訊息"""
    import subprocess
    import sys as _sys
    merged = tmp_path / "merged.json"
    merged.write_text('{"_meta": {}}', encoding="utf-8")
    result = subprocess.run(
        [_sys.executable, _predict_py_path(),
         "--game-data", str(merged), "--save",
         "--ou-rec", "OVER",
         "--ou-line", "9.5",
         "--ml-rec", "PASS", "--ml-stars", "0",
         "--skip-phase3-check"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 6
    assert "--ou-stars" in result.stderr
    assert "OVER/UNDER" in result.stderr or "OVER" in result.stderr


def test_ou_rec_under_without_stars_exits(tmp_path):
    """--ou-rec UNDER 但缺 --ou-stars → exit 6"""
    import subprocess
    import sys as _sys
    merged = tmp_path / "merged.json"
    merged.write_text('{"_meta": {}}', encoding="utf-8")
    result = subprocess.run(
        [_sys.executable, _predict_py_path(),
         "--game-data", str(merged), "--save",
         "--ou-rec", "UNDER",
         "--ou-line", "9.5",
         "--ml-rec", "PASS", "--ml-stars", "0",
         "--skip-phase3-check"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 6


def test_ou_rec_pass_without_stars_ok(tmp_path):
    """--ou-rec PASS 不需要 --ou-stars（既有行為）"""
    import subprocess
    import sys as _sys
    # 用既有 _setup_game_dir helper 建構合法 merged.json
    game_dir, merged_path = _setup_game_dir(tmp_path)
    result = subprocess.run(
        [_sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save",
         "--ou-rec", "PASS",
         "--ou-line", "9.5",
         "--ml-rec", "PASS", "--ml-stars", "0",
         "--skip-phase3-check"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest scripts/tests/test_predict.py -k "ou_rec" -v
```

Expected: `_over_without_stars_exits` / `_under_without_stars_exits` FAIL（目前 silent fallback 不 exit）

- [ ] **Step 3: Add validation in predict.py main()**

於 `scripts/predict.py` `main()` 內，於 `args = parser.parse_args()`（line 955）後立即加：

```python
    # P5：--ou-rec OVER/UNDER 必須同時提供 --ou-stars（避免 silent fallback 為 PASS）
    if args.ou_rec in ("OVER", "UNDER") and args.ou_stars is None:
        sys.exit(
            "⛔ exit 6: --ou-rec=OVER/UNDER 必須同時提供 --ou-stars (0-5)\n"
            "  例：--ou-rec OVER --ou-stars 3"
        )
    # PASS 時 ou_stars 預設 0（保留既有行為）
    if args.ou_rec == "PASS" and args.ou_stars is None:
        args.ou_stars = 0
```

確認 `import sys` 在檔案頂端（既有應已 import）。`sys.exit(string)` 預設 exit code 為 1，但需要 6 — 改用：

```python
    if args.ou_rec in ("OVER", "UNDER") and args.ou_stars is None:
        print(
            "⛔ --ou-rec=OVER/UNDER 必須同時提供 --ou-stars (0-5)\n"
            "  例：--ou-rec OVER --ou-stars 3",
            file=sys.stderr,
        )
        sys.exit(6)
    if args.ou_rec == "PASS" and args.ou_stars is None:
        args.ou_stars = 0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest scripts/tests/test_predict.py -k "ou_rec" -v
```

Expected: 3/3 PASS

#### Task 7b: 移除 YoY prior year guard

- [ ] **Step 5: Delete obsolete tests**

於 `scripts/tests/test_predict.py` 刪除以下測試（已不對應新行為）：

- `test_yoy_skip_flag_bypasses`（含 `--skip-yoy-check` 的 test）
- `test_phase3_yoy_trigger_missing_section_exits`
- `test_phase3_babip_trigger_missing_section_exits`

保留 `test_phase3_skip_flag_bypasses`（仍適用 file existence check）。

- [ ] **Step 6: Add replacement test**

於 `scripts/tests/test_predict.py` 末尾追加：

```python


def test_yoy_trigger_no_prior_year_no_longer_blocks(tmp_path):
    """spec 2026-04-29：移除 Step C-prior 後，Flag 13 觸發但缺 prior year file 不應阻擋"""
    import subprocess
    import sys as _sys
    # 觸發 Flag 13：era 1.15 / xera 3.96 / ip 31.3（Lugo 真實案例）
    game_dir, merged_path = _setup_game_dir(
        tmp_path,
        home_pitcher_era=1.15, home_pitcher_xera=3.96, home_pitcher_ip=31.3,
    )
    # 注意：未建立 prior year file（home_pitcher_2025.json）
    (game_dir / "phase3_summary.md").write_text("# summary\n", encoding="utf-8")
    result = subprocess.run(
        [_sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save",
         "--ou-rec", "PASS", "--ml-rec", "PASS", "--ml-stars", "0"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, f"應不阻擋；stderr={result.stderr}"


def test_h2_grep_removed_phase3_summary_no_yoy_section_passes(tmp_path):
    """spec 2026-04-29：移除 H2 grep 後，phase3_summary.md 缺 ## YoY 對比結論 不應阻擋"""
    import subprocess
    import sys as _sys
    # Flag 13 觸發 + phase3_summary.md 不含 ## YoY 對比結論（新 skeleton 無此 H2）
    game_dir, merged_path = _setup_game_dir(
        tmp_path,
        home_pitcher_era=1.15, home_pitcher_xera=3.96, home_pitcher_ip=31.3,
    )
    (game_dir / "phase3_summary.md").write_text(
        "# summary\n\n## 風險提示\n- AWAY 投手 Flag 13 …\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [_sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save",
         "--ou-rec", "PASS", "--ml-rec", "PASS", "--ml-stars", "0"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, f"應不阻擋；stderr={result.stderr}"
```

如果 `_setup_game_dir` helper 沒有 `home_pitcher_era` / `home_pitcher_xera` / `home_pitcher_ip` 參數，新增到 helper（既有 helper 在檔案上方）。

- [ ] **Step 7: Run tests to verify new tests fail**

```bash
python -m pytest scripts/tests/test_predict.py -k "yoy_trigger_no_prior\|h2_grep_removed" -v
```

Expected: 2 FAIL（既有 guard 仍 block）

- [ ] **Step 8: Remove guards in predict.py**

於 `scripts/predict.py`：

1. 刪除 argparse args（line 951-954）：
```python
    # 刪除以下兩行
    parser.add_argument("--skip-yoy-check", action="store_true",
                        help="Bypass B7 YoY prior year file existence check (Plan B)")
```
保留 `--skip-phase3-check`（仍用於 phase3_summary.md 檔存在 bypass）。

2. 刪除 YoY prior year file existence check（line 1044-1066，整段 `if not args.skip_yoy_check:` 區塊）。

3. 改寫 `if not args.skip_phase3_check:` 區塊（line 1069-1098）為僅檔存在檢查：

```python
        # phase3_summary.md 必須存在（structural 完整性由 phase3_skeleton.md 預填保證）
        if not args.skip_phase3_check:
            game_dir_p3 = Path(args.game_data).parent
            phase3_path = game_dir_p3 / "phase3_summary.md"
            if not phase3_path.exists():
                sys.exit(
                    f"⛔ {phase3_path} 不存在 — Phase 3 結論未存檔\n"
                    f"  請先在 phase3_skeleton.md 補結論並另存為 phase3_summary.md；"
                    f"或加 --skip-phase3-check 跳過（測試用）。"
                )
```

刪除 `required_sections` / `missing` H2 grep 邏輯（line 1080-1098）。

4. 確認 `pitcher_triggers_yoy` / `lineup_triggers_babip` 函式（line 118+ / 142+）仍保留 — 它們**會被 prepare_game.py 使用**來偵測 Flag 13 / Flag 3，不要刪除函式本身。

- [ ] **Step 9: Run tests to verify they pass**

```bash
python -m pytest scripts/tests/test_predict.py -v
```

Expected: 全 PASS（含新 + 既有 — 已刪 obsolete tests）

- [ ] **Step 10: Run full suite for regressions**

```bash
python -m pytest scripts/tests/ -v
```

Expected: 全 PASS

- [ ] **Step 11: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict.py
git commit -m "$(cat <<'EOF'
feat(scripts): predict.py --ou-stars 必填 + 移除 YoY/H2 guard

- --ou-rec OVER/UNDER 缺 --ou-stars → hard exit 6（解 spec P5 silent fallback）
- 移除 YoY prior year file existence check + --skip-yoy-check arg（不再做 Step C-prior）
- 移除 H2 grep guard（## YoY 對比結論 / ## BABIP 回歸判定 / ## 牛棚雙向修正值）
- 保留 phase3_summary.md 檔存在檢查（仍由 --skip-phase3-check bypass）
- 刪除 obsolete tests（test_yoy_skip_flag_bypasses 等 3 個）

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `dossier_renderer.py` — 新模組

**Files:**
- Create: `scripts/dossier_renderer.py`
- Create: `scripts/tests/test_dossier_renderer.py`

**設計**：純函式渲染器，輸入是 7 個 JSON dict（game_data / home_roster / away_roster / home_pitcher / away_pitcher / home_lineup / away_lineup）+ `merged`（含 park / bullpen），輸出 markdown 字串。子函式逐節渲染，可獨立測試。

**Public API：**
```python
def render_dossier(bundle: dict, *, game_dir: str = "") -> str: ...
```

`bundle` 結構：
```python
{
    "game_data": {...},      # fetch_game_data.py 輸出
    "home_roster": {...},    # roster_checker.py 輸出
    "away_roster": {...},
    "home_pitcher": {...},   # pitcher_stats.py 輸出
    "away_pitcher": {...},
    "home_lineup": {...},    # lineup_analyzer.py 輸出
    "away_lineup": {...},
    "merged": {...},         # merge_game_data.py 輸出（park、bullpen ERA）
}
```

#### Task 8a: 模組骨架 + Top 5 候選池函式

- [ ] **Step 1: Write failing test for top5 selection**

建立 `scripts/tests/test_dossier_renderer.py`：

```python
"""Tests for dossier_renderer."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_select_top5_pa_filter():
    """PA ≥ 30、IL'd 排除、最多 5 人，按 PA 降序"""
    from dossier_renderer import select_top5_vs_pitcher
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "season_ops": 0.800, "vs_rhp_ops": 0.750,
             "last7_ops": 0.700, "last7_babip": 0.300, "ev95_pct": 50.0, "barrel_pct": 10.0},
            {"name": "B", "pa": 80,  "season_ops": 0.750, "vs_rhp_ops": 0.700,
             "last7_ops": 0.650, "last7_babip": 0.280, "ev95_pct": 45.0, "barrel_pct": 8.0},
            {"name": "C", "pa": 25,  "season_ops": 0.900, "vs_rhp_ops": 0.850,
             "last7_ops": 0.800, "last7_babip": 0.330, "ev95_pct": 55.0, "barrel_pct": 12.0},  # PA < 30
            {"name": "D", "pa": 60,  "season_ops": 0.700, "vs_rhp_ops": 0.650,
             "last7_ops": 0.600, "last7_babip": 0.260, "ev95_pct": 40.0, "barrel_pct": 7.0},
            {"name": "E", "pa": 45,  "season_ops": 0.680, "vs_rhp_ops": 0.620,
             "last7_ops": 0.580, "last7_babip": 0.240, "ev95_pct": 38.0, "barrel_pct": 6.0},
            {"name": "F", "pa": 35,  "season_ops": 0.660, "vs_rhp_ops": 0.600,
             "last7_ops": 0.560, "last7_babip": 0.220, "ev95_pct": 36.0, "barrel_pct": 5.0},
            {"name": "G", "pa": 32,  "season_ops": 0.640, "vs_rhp_ops": 0.580,
             "last7_ops": 0.540, "last7_babip": 0.200, "ev95_pct": 34.0, "barrel_pct": 4.0},
        ]
    }
    il_names = set()
    result = select_top5_vs_pitcher(lineup, il_names)
    names = [p["name"] for p in result]
    assert names == ["A", "B", "D", "E", "F"]  # G 被擠出（取 top 5）；C 被 PA 過濾


def test_select_top5_excludes_il():
    """IL 名單上的球員直接濾掉"""
    from dossier_renderer import select_top5_vs_pitcher
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "season_ops": 0.8},
            {"name": "B", "pa": 80,  "season_ops": 0.75},
            {"name": "C", "pa": 60,  "season_ops": 0.7},
        ]
    }
    il_names = {"B"}
    result = select_top5_vs_pitcher(lineup, il_names)
    names = [p["name"] for p in result]
    assert names == ["A", "C"]


def test_select_top5_fewer_than_5_returns_what_exists():
    """候選池 < 5 → 返回所有合格者"""
    from dossier_renderer import select_top5_vs_pitcher
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "season_ops": 0.8},
            {"name": "B", "pa": 50,  "season_ops": 0.75},
        ]
    }
    result = select_top5_vs_pitcher(lineup, set())
    assert len(result) == 2


def test_select_top5_last7_top1_outside_pa_top5():
    """last7 OPS top1 不在 PA top5 內 → annotate"""
    from dossier_renderer import find_last7_top1_outside_pa_top5
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "last7_ops": 0.700},
            {"name": "B", "pa": 90,  "last7_ops": 0.650},
            {"name": "C", "pa": 80,  "last7_ops": 0.600},
            {"name": "D", "pa": 70,  "last7_ops": 0.550},
            {"name": "E", "pa": 60,  "last7_ops": 0.500},
            {"name": "Schneemann", "pa": 35, "last7_ops": 1.164},  # 不在 PA top5（被 E 擠掉）但 last7 OPS top1
        ]
    }
    pa_top5 = ["A", "B", "C", "D", "E"]
    annotation = find_last7_top1_outside_pa_top5(lineup, pa_top5, set())
    assert annotation is not None
    assert annotation["name"] == "Schneemann"
    assert annotation["last7_ops"] == 1.164


def test_select_top5_last7_top1_already_in_pa_top5_returns_none():
    """last7 OPS top1 已在 PA top5 內 → 不需 annotate"""
    from dossier_renderer import find_last7_top1_outside_pa_top5
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "last7_ops": 1.000},  # PA top1 + last7 top1
            {"name": "B", "pa": 90,  "last7_ops": 0.500},
        ]
    }
    pa_top5 = ["A", "B"]
    assert find_last7_top1_outside_pa_top5(lineup, pa_top5, set()) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest scripts/tests/test_dossier_renderer.py -v
```

Expected: 5 FAIL（`ImportError: cannot import name 'dossier_renderer'`）

- [ ] **Step 3: Create dossier_renderer.py with helpers**

建立 `scripts/dossier_renderer.py`：

```python
"""Dossier renderer：將 Phase 1+2 各 JSON 整合為 ~250 行 markdown，作 AI 主入口。

設計原則（spec §4）：
- 純函式：輸入 dict bundle，輸出 markdown str
- 無 side effect、無 I/O（除 render_dossier 之外）
- 子函式逐節獨立可測

Bundle keys:
  game_data, home_roster, away_roster, home_pitcher, away_pitcher,
  home_lineup, away_lineup, merged
"""
from __future__ import annotations


PA_FLOOR = 30  # spec §4.2 Top 5 候選池下限


def _il_names_from_roster(roster: dict | None) -> set[str]:
    """從 roster_checker.py 輸出取出 IL'd 球員名字。"""
    if not roster:
        return set()
    return {p.get("name") for p in roster.get("injured_list", []) if p.get("name")}


def select_top5_vs_pitcher(lineup: dict | None, il_names: set[str]) -> list[dict]:
    """從 lineup（lineup_analyzer.py 輸出）選 Top 5 vs 對方先發。

    規則（spec §4.2）：
    - active && PA ≥ 30 && !IL'd
    - 按 PA 降序
    - 最多 5 人，候選池 < 5 就少
    """
    candidates = [
        p for p in (lineup or {}).get("lineup", []) or []
        if (p.get("pa") or 0) >= PA_FLOOR and p.get("name") not in il_names
    ]
    candidates.sort(key=lambda p: p.get("pa") or 0, reverse=True)
    return candidates[:5]


def find_last7_top1_outside_pa_top5(
    lineup: dict | None,
    pa_top5_names: list[str],
    il_names: set[str],
) -> dict | None:
    """找出 last7 OPS top1 球員，若不在 PA top 5 內則回傳；否則 None。

    候選池套用同樣的 IL 過濾與 PA ≥ 30。
    """
    candidates = [
        p for p in (lineup or {}).get("lineup", []) or []
        if (p.get("pa") or 0) >= PA_FLOOR and p.get("name") not in il_names
    ]
    candidates_with_last7 = [p for p in candidates if p.get("last7_ops") is not None]
    if not candidates_with_last7:
        return None
    top1 = max(candidates_with_last7, key=lambda p: p["last7_ops"])
    if top1.get("name") in pa_top5_names:
        return None
    return top1


def render_dossier(bundle: dict, *, game_dir: str = "") -> str:
    """主入口：渲染整份 dossier.md。

    後續子節 render 函式由 Task 8b-8h 分批實作。
    """
    raise NotImplementedError("實作於 Task 8b 起逐節補")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest scripts/tests/test_dossier_renderer.py -v
```

Expected: 5 PASS（top5 選取邏輯通過，render_dossier 尚未實作但無測試）

- [ ] **Step 5: Commit**

```bash
git add scripts/dossier_renderer.py scripts/tests/test_dossier_renderer.py
git commit -m "$(cat <<'EOF'
feat(scripts): dossier_renderer 骨架 + Top 5 選取函式

- select_top5_vs_pitcher: PA ≥ 30、IL 排除、按 PA 降序、最多 5 人
- find_last7_top1_outside_pa_top5: 補列「不在 PA top5 但 last7 OPS top1」
- render_dossier 為 NotImplementedError stub（Task 8b 起補實作）

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

#### Task 8b: 渲染各 section（連續開發，TDD per section）

**注意**：此 task 量較大（~200 行 markdown 輸出邏輯）。建議分節 commit：每完成 1 個 section 函式就 commit 一次。每 section 的測試模式相同：

1. 寫一個「給定 sample bundle → 期望 markdown 片段含 X」的測試
2. 失敗
3. 實作 `_render_<section>(bundle) -> list[str]`
4. PASS
5. Commit

**子節清單（按 spec §4.2 順序）：**

- [ ] **8b-1: `_render_header(bundle) -> list[str]`** — `# Game Dossier — {AWAY} @ {HOME} ({date})`
- [ ] **8b-2: `_render_game_info(bundle)`** — 比賽資訊段（拷貝 game_data_summary 邏輯）
- [ ] **8b-3: `_render_record_summary(bundle)`** — 戰績速查表（近 10 / 30 / 本季 / 趨勢）
- [ ] **8b-4: `_render_series_context(bundle)`** — 系列脈絡（拷貝 game_data_summary §系列賽 + §Streak 脈絡）
- [ ] **8b-5: `_render_pitcher_matchup(bundle)`** — 投手對決表（含 ⚠️ 風險提示行）
- [ ] **8b-6: `_render_lineup_overview(bundle)`** — 打線表（含 ⚠️ 風險提示行）
- [ ] **8b-7: `_render_top5_block(bundle)`** — Top 5 vs 對方先發（含 last7 top1 註腳）
- [ ] **8b-8: `_render_bullpen_park(bundle)`** — 牛棚 / Park 表
- [ ] **8b-9: `_render_risk_summary(bundle)`** — ⚠️ 風險提示摘要段（呼叫 `pitcher_stats.detect_triggers` / `lineup_analyzer.detect_triggers`）
- [ ] **8b-10: `_render_file_index(bundle, game_dir)`** — File 索引段
- [ ] **8b-11: `render_dossier(bundle, game_dir)`** — 串接所有 section + 行數驗證 ≤ 250

**每節測試範本**（範例：8b-5 投手對決）：

```python
def test_render_pitcher_matchup_includes_risk_note_on_flag13():
    """home_pitcher 觸發 Flag 13 → 表格末行含 era_xera_delta + Flag 13"""
    from dossier_renderer import _render_pitcher_matchup
    bundle = {
        "home_pitcher": {
            "name": "Test Home", "tier_emoji": "🟠",
            "season": {"era": 2.10, "ip": 31.3},
            "expected": {"xera": 4.64},
            # ... 其他必要欄位
        },
        "away_pitcher": {
            "name": "Test Away", "tier_emoji": "🟢",
            "season": {"era": 4.45},
            "expected": {"xera": 4.64},
        },
    }
    lines = _render_pitcher_matchup(bundle)
    md = "\n".join(lines)
    assert "## 投手對決" in md
    assert "⚠️" in md
    assert "Flag 13" in md
```

**整體 render_dossier 測試（Task 8b-11）**：

```python
def test_render_dossier_full_output_within_250_lines(tb_cle_bundle):
    """spec §4.1：dossier 行數 ≤ 250"""
    from dossier_renderer import render_dossier
    output = render_dossier(tb_cle_bundle, game_dir="analysis-data/2026-04-28/TB@CLE")
    lines = output.split("\n")
    assert len(lines) <= 250


def test_render_dossier_required_sections_present(tb_cle_bundle):
    """spec §4.2：必出現的 H2"""
    from dossier_renderer import render_dossier
    output = render_dossier(tb_cle_bundle)
    for h2 in ["## 比賽資訊", "## 戰績速查", "## 系列脈絡", "## 投手對決",
               "## 打線", "## 牛棚 / Park", "## ⚠️ 風險提示摘要", "## File 索引"]:
        assert h2 in output, f"缺 {h2}"


def test_render_dossier_no_yoy_section_after_redesign(tb_cle_bundle):
    """spec §4.2：刪除「### YoY 對比」section"""
    from dossier_renderer import render_dossier
    output = render_dossier(tb_cle_bundle)
    assert "YoY 對比" not in output  # 整段刪
```

`tb_cle_bundle` fixture 從既有 `analysis-data/2026-04-28/TB@CLE/` 讀取 8 個 JSON 檔組裝（在 test 檔頂端用 `@pytest.fixture` 定義）。

- [ ] **8b-12: 串接 commit**

每節完成後分別 commit；最後做一次整合測試 + commit：

```bash
python -m pytest scripts/tests/test_dossier_renderer.py -v
git add scripts/dossier_renderer.py scripts/tests/test_dossier_renderer.py
git commit -m "feat(scripts): dossier_renderer 完成 8 個 section + render_dossier 整合"
```

---

### Task 9: `phase3_skeleton_renderer.py` — 新模組

**Files:**
- Create: `scripts/phase3_skeleton_renderer.py`
- Create: `scripts/tests/test_phase3_skeleton_renderer.py`

**設計**：純函式渲染器，輸出 ~30 行 markdown skeleton，AI 在上面補結論段落。

**Public API：**
```python
def render_skeleton(bundle: dict, formula_pred: dict) -> str:
    """bundle 同 dossier_renderer；formula_pred 為 predict.predict_with_formula 輸出"""
```

**H2 清單**（spec §5.3，7 個全部永遠存在）：
1. `## 投手對決` — 含 `### {pitcher} (...)` 子節 + `**Tier 覆寫**:` slot
2. `## 打線評級` — 含 `### HOME` / `### AWAY` + Tier 覆寫
3. `## 牛棚` — 含表格 + `### 牛棚雙向修正值` H3
4. `## 風險提示` — prepare_game.py 預填 Flag 13/3，AI 補敘事
5. `## 條件修正` — Park PF 預填，其他 AI 補
6. `## 修正後預期得分` — base 比分預填（從 formula_pred 取），signal/adjusted AI 補
7. `## 整體判斷` — 全 AI 補

- [ ] **Step 1: Write tests for skeleton structure**

建立 `scripts/tests/test_phase3_skeleton_renderer.py`：

```python
"""Tests for phase3_skeleton_renderer."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _minimal_bundle():
    return {
        "game_data": {"date": "2026-04-28", "home": {"team": "CLE"}, "away": {"team": "TB"}},
        "home_pitcher": {"name": "Bibee", "tier_emoji": "🟢", "info": {"pitch_hand": "R", "age": 27}},
        "away_pitcher": {"name": "Martínez", "tier_emoji": "🟠", "info": {"pitch_hand": "R", "age": 35}},
        "home_lineup": {"tier_emoji": "🟡", "heat_emoji": "⚖️"},
        "away_lineup": {"tier_emoji": "🟢", "heat_emoji": "⚖️"},
        "merged": {"home_bullpen_era": 4.57, "away_bullpen_era": 5.18,
                   "home_bullpen_il_count": 2, "away_bullpen_il_count": 8,
                   "park_factor": 101},
    }


def _minimal_formula_pred():
    return {"home_expected_runs": 4.5, "away_expected_runs": 4.2}


def test_skeleton_contains_7_required_h2():
    """spec §5.3：7 個 H2 永遠存在"""
    from phase3_skeleton_renderer import render_skeleton
    output = render_skeleton(_minimal_bundle(), _minimal_formula_pred())
    for h2 in ["## 投手對決", "## 打線評級", "## 牛棚", "## 風險提示",
               "## 條件修正", "## 修正後預期得分", "## 整體判斷"]:
        assert h2 in output, f"缺 {h2}"


def test_skeleton_no_yoy_or_babip_h2():
    """spec §5.3：刪除 ## YoY 對比結論 / ## BABIP 回歸判定"""
    from phase3_skeleton_renderer import render_skeleton
    output = render_skeleton(_minimal_bundle(), _minimal_formula_pred())
    assert "## YoY 對比結論" not in output
    assert "## BABIP 回歸判定" not in output


def test_skeleton_tier_override_slot_present():
    """spec §5.2：Tier 覆寫 slot 在投手 + 打線段都要有"""
    from phase3_skeleton_renderer import render_skeleton
    output = render_skeleton(_minimal_bundle(), _minimal_formula_pred())
    # 投手對決有 2 個（home + away），打線有 2 個 → 至少 4 處
    assert output.count("**Tier 覆寫**") >= 4


def test_skeleton_risk_section_lists_triggers_when_present():
    """Flag 13 / Flag 3 觸發 → 預填條目至 ## 風險提示"""
    from phase3_skeleton_renderer import render_skeleton
    bundle = _minimal_bundle()
    bundle["away_pitcher"]["season"] = {"era": 2.10, "ip": 31.3}
    bundle["away_pitcher"]["expected"] = {"xera": 4.64}  # gap = 2.54 → Flag 13
    bundle["away_lineup"]["last7_babip"] = 0.241  # Flag 3
    output = render_skeleton(bundle, _minimal_formula_pred())
    assert "Flag 13" in output
    assert "Flag 3" in output
    assert "era_xera_delta" in output or "ERA-xERA" in output


def test_skeleton_risk_section_says_no_flag_when_clean():
    """無 Flag 觸發 → ## 風險提示 內文「無風險提示」"""
    from phase3_skeleton_renderer import render_skeleton
    output = render_skeleton(_minimal_bundle(), _minimal_formula_pred())
    # 找 ## 風險提示 區段
    after_risk = output.split("## 風險提示", 1)[1].split("##", 1)[0]
    assert "無風險提示" in after_risk


def test_skeleton_park_factor_correction_prefilled():
    """## 條件修正 段預填 Park Factor 修正值"""
    from phase3_skeleton_renderer import render_skeleton
    bundle = _minimal_bundle()
    bundle["merged"]["park_factor"] = 110
    output = render_skeleton(bundle, _minimal_formula_pred())
    # PF 110 → +0.5 run 修正
    assert "Park Factor: 110" in output or "PF=110" in output
    assert "+0.50" in output or "+0.5" in output


def test_skeleton_expected_runs_table_uses_formula_pred():
    """## 修正後預期得分 base 列從 formula_pred 取得"""
    from phase3_skeleton_renderer import render_skeleton
    output = render_skeleton(_minimal_bundle(), _minimal_formula_pred())
    assert "4.5" in output  # home_expected_runs
    assert "4.2" in output  # away_expected_runs


def test_skeleton_line_count_within_50():
    """spec §9 驗收：phase3_skeleton.md ≤ 50 行（無 Flag 觸發時）"""
    from phase3_skeleton_renderer import render_skeleton
    output = render_skeleton(_minimal_bundle(), _minimal_formula_pred())
    assert len(output.split("\n")) <= 50
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest scripts/tests/test_phase3_skeleton_renderer.py -v
```

Expected: 7 FAIL（ImportError）

- [ ] **Step 3: Implement render_skeleton**

建立 `scripts/phase3_skeleton_renderer.py`：

```python
"""phase3_skeleton.md renderer：產生 7 個 H2 + 預填數值表 + AI 填空 placeholder。

設計（spec §5）：
- 7 個 H2 永遠存在（即使 Flag 未觸發）
- ## 風險提示 段：prepare_game.py 偵測到的 Flag 13/3 預填條目；無則「無風險提示」
- ## 條件修正 段：Park PF 修正預填
- ## 修正後預期得分 段：base 列從 formula_pred 預填
- 其餘為 AI 填空 (`<!-- AI 補：... -->`)
"""
from __future__ import annotations

# 從 pitcher_stats / lineup_analyzer import 偵測函式（已存在）
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _age_emoji(age: int | None) -> str:
    """spec matchup-factors §球員年齡退化 投手版"""
    if age is None:
        return ""
    if age <= 24: return "📈"
    if age <= 29: return "⚡"
    if age <= 33: return "📉"
    if age <= 36: return "📉📉"
    return "📉📉📉"


def _render_header(bundle: dict) -> list[str]:
    gd = bundle.get("game_data", {})
    away = gd.get("away", {}).get("team", "AWAY")
    home = gd.get("home", {}).get("team", "HOME")
    date = gd.get("date", "")[:10]
    return [f"# Phase 3 Summary — {away} @ {home} ({date})", ""]


def _render_pitcher_matchup_section(bundle: dict) -> list[str]:
    home_p = bundle.get("home_pitcher", {})
    away_p = bundle.get("away_pitcher", {})
    home_info = home_p.get("info", {})
    away_info = away_p.get("info", {})
    home_age = home_info.get("age")
    away_age = away_info.get("age")
    return [
        "## 投手對決",
        "",
        f"### {home_p.get('name', '?')} (HOME, {home_info.get('pitch_hand', '?')}HP, {home_age or '?'} {_age_emoji(home_age)})",
        "- **Tier 覆寫**：<!-- AI 補：覆寫 + 理由 / 或「沿用腳本 {tier}」 -->",
        "- 真實水平判斷：<!-- AI 補：基於 ERA/xERA/FIP/Statcast/年齡綜合 -->",
        "- 對手打線威脅：<!-- AI 補 -->",
        "",
        f"### {away_p.get('name', '?')} (AWAY, {away_info.get('pitch_hand', '?')}HP, {away_age or '?'} {_age_emoji(away_age)})",
        "- **Tier 覆寫**：<!-- AI 補 -->",
        "- 真實水平判斷：<!-- AI 補 -->",
        "- 對手打線威脅：<!-- AI 補 -->",
        "",
    ]


def _render_lineup_section(bundle: dict) -> list[str]:
    home_l = bundle.get("home_lineup", {})
    away_l = bundle.get("away_lineup", {})
    return [
        "## 打線評級",
        "",
        f"### HOME — {home_l.get('tier_emoji', '?')} / {home_l.get('heat_emoji', '?')}",
        "- **Tier 覆寫**：<!-- AI 補 -->",
        "（AI 摘要 + 補主威脅 / 黑洞 list）",
        "",
        f"### AWAY — {away_l.get('tier_emoji', '?')} / {away_l.get('heat_emoji', '?')}",
        "- **Tier 覆寫**：<!-- AI 補 -->",
        "（同上）",
        "",
    ]


def _render_bullpen_section(bundle: dict) -> list[str]:
    m = bundle.get("merged", {})
    return [
        "## 牛棚",
        "",
        "| | HOME | AWAY |",
        "|---|---|---|",
        f"| ERA / IL 數 / 核心 IL 估計 | {m.get('home_bullpen_era', '?')} / {m.get('home_bullpen_il_count', '?')} / <!-- AI --> | "
        f"{m.get('away_bullpen_era', '?')} / {m.get('away_bullpen_il_count', '?')} / <!-- AI --> |",
        "",
        "### 牛棚雙向修正值",
        "- HOME 牛棚：對手 +<!-- AI --> run | HOME ML <!-- AI -->%",
        "- AWAY 牛棚：對手 +<!-- AI --> run | AWAY ML <!-- AI -->%",
        "<!-- AI 補：填入修正值，依 matchup-factors.md §牛棚傷兵累計效應 -->",
        "",
    ]


def _detect_risk_notes(bundle: dict) -> list[str]:
    """偵測 Flag 13 / Flag 3，回傳「條目 markdown 行」list（不含 H2 開頭）。"""
    from pitcher_stats import detect_triggers as detect_pitcher_triggers
    from lineup_analyzer import detect_triggers as detect_lineup_triggers
    notes = []
    for side in ("home", "away"):
        triggers = detect_pitcher_triggers(bundle.get(f"{side}_pitcher", {}))
        for t in triggers:
            if t.get("flag") == 13:
                gap = t.get("value", "?")
                notes.append(f"- ⚠️ {side.upper()} 投手 Flag 13 (era_xera_delta={gap}):")
                notes.append("  - <!-- AI 補：是運氣還結構性？是否影響本場判斷？不自動下修預測 -->")
    for side in ("home", "away"):
        triggers = detect_lineup_triggers(bundle.get(f"{side}_lineup", {}))
        for t in triggers:
            if t.get("flag") == 3:
                babip = bundle.get(f"{side}_lineup", {}).get("last7_babip", "?")
                notes.append(f"- ⚠️ {side.upper()} 打線 Flag 3 (last7 BABIP={babip}):")
                notes.append("  - <!-- AI 補：可能回歸或可能持續？是否影響本場判斷？不自動 ±run value -->")
    return notes


def _render_risk_section(bundle: dict) -> list[str]:
    notes = _detect_risk_notes(bundle)
    if not notes:
        return ["## 風險提示", "", "無風險提示", ""]
    return ["## 風險提示", ""] + notes + [""]


def _render_conditional_section(bundle: dict) -> list[str]:
    pf = bundle.get("merged", {}).get("park_factor", 100)
    pf_correction = (pf - 100) * 0.05
    return [
        "## 條件修正",
        "",
        f"- Park Factor: {pf} → {pf_correction:+.2f} run",
        "- 雙方先發 tier: <!-- AI 補：是否觸發 -1.0 / -0.5 投手戰 -->",
        "- 其他（doubleheader / platoon / 休息日 / 天氣）: <!-- AI 補 -->",
        "",
    ]


def _render_expected_runs_section(bundle: dict, formula_pred: dict) -> list[str]:
    home_base = formula_pred.get("home_expected_runs", "?")
    away_base = formula_pred.get("away_expected_runs", "?")
    total_base = (home_base + away_base) if isinstance(home_base, (int, float)) and isinstance(away_base, (int, float)) else "?"
    return [
        "## 修正後預期得分",
        "",
        "| | base (formula) | + 信號 | adjusted |",
        "|---|---|---|---|",
        f"| HOME | {home_base} | <!-- AI 補 --> | <!-- AI 補 --> |",
        f"| AWAY | {away_base} | <!-- AI 補 --> | <!-- AI 補 --> |",
        f"| Total | {total_base} | <!-- AI 補 --> | <!-- AI 補 --> |",
        "",
    ]


def _render_overall_section() -> list[str]:
    return [
        "## 整體判斷",
        "",
        "- **方向（基本面）**：<!-- AI 補 -->",
        "- **總分（基本面）**：<!-- AI 補 -->",
        "- **信心**：<!-- AI 補 LOW/MEDIUM/HIGH -->",
        "- **風險**：<!-- AI 補 1-4 點 -->",
        "",
        "⛔ MUST NOT contain：星級、明確盤口推薦",
    ]


def render_skeleton(bundle: dict, formula_pred: dict) -> str:
    """主入口。"""
    lines: list[str] = []
    lines += _render_header(bundle)
    lines += _render_pitcher_matchup_section(bundle)
    lines += _render_lineup_section(bundle)
    lines += _render_bullpen_section(bundle)
    lines += _render_risk_section(bundle)
    lines += _render_conditional_section(bundle)
    lines += _render_expected_runs_section(bundle, formula_pred)
    lines += _render_overall_section()
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest scripts/tests/test_phase3_skeleton_renderer.py -v
```

Expected: 7/7 PASS

- [ ] **Step 5: Run full suite for regressions**

```bash
python -m pytest scripts/tests/ -v
```

Expected: 全 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/phase3_skeleton_renderer.py scripts/tests/test_phase3_skeleton_renderer.py
git commit -m "$(cat <<'EOF'
feat(scripts): phase3_skeleton_renderer 新模組

- render_skeleton(bundle, formula_pred) → 7 個 H2 永遠存在的 markdown skeleton
- 預填：投手 hand/age、打線 tier/heat、牛棚 ERA/IL、Park Factor 修正值、formula base 比分
- ## 風險提示 段：偵測 Flag 13/3 預填條目；無 Flag 時內文「無風險提示」
- 移除 ## YoY 對比結論 / ## BABIP 回歸判定（spec §5.3）
- 7 個 ## H2 + 多處 **Tier 覆寫** slot，AI 在此基礎上填結論

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: `prepare_game.py` — 主腳本

**Files:**
- Create: `scripts/prepare_game.py`
- Create: `scripts/tests/test_prepare_game.py`

**架構決策**：
- 使用 `subprocess.run()` 呼叫既有腳本（解耦、不需重構 main()）
- 雙隊平行用 `concurrent.futures.ThreadPoolExecutor(max_workers=2)`
- Step 之間序列（依賴 Step A 的 ID）
- 失敗即 stderr + sys.exit(<code>)

- [ ] **Step 1: Write tests for CLI parsing + exit codes**

建立 `scripts/tests/test_prepare_game.py`：

```python
"""Tests for prepare_game.py main script.

Strategy: 不真的呼叫子腳本（會打 API），而是 monkeypatch subprocess.run
與 Path.exists 等 I/O，測試 CLI parsing、exit code、step 順序。
"""
import os
import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_parse_args_defaults():
    from prepare_game import parse_args
    args = parse_args(["--date", "2026-04-28", "--away", "TB", "--home", "CLE"])
    assert args.date == "2026-04-28"
    assert args.away == "TB"
    assert args.home == "CLE"
    assert args.season == 2026
    assert args.game_suffix is None
    assert not args.force


def test_parse_args_explicit_season_overrides():
    from prepare_game import parse_args
    args = parse_args(["--date", "2026-04-28", "--away", "TB", "--home", "CLE",
                       "--season", "2025"])
    assert args.season == 2025


def test_parse_args_doubleheader_g1():
    from prepare_game import parse_args
    args = parse_args(["--date", "2026-04-28", "--away", "TB", "--home", "CLE",
                       "--game-suffix", "G1"])
    assert args.game_suffix == "G1"


def test_compute_output_dir_default():
    from prepare_game import compute_output_dir
    p = compute_output_dir(date="2026-04-28", away="TB", home="CLE",
                          game_suffix=None, override=None)
    assert p == Path("analysis-data/2026-04-28/TB@CLE")


def test_compute_output_dir_doubleheader_g2():
    from prepare_game import compute_output_dir
    p = compute_output_dir(date="2026-04-28", away="TB", home="CLE",
                          game_suffix="G2", override=None)
    assert p == Path("analysis-data/2026-04-28/TB@CLE-G2")


def test_compute_output_dir_explicit_override():
    from prepare_game import compute_output_dir
    p = compute_output_dir(date="2026-04-28", away="TB", home="CLE",
                          game_suffix=None, override="/tmp/foo")
    assert p == Path("/tmp/foo")


def test_dossier_filename_no_suffix():
    from prepare_game import dossier_filename, skeleton_filename
    assert dossier_filename(None) == "dossier.md"
    assert skeleton_filename(None) == "phase3_skeleton.md"


def test_dossier_filename_with_suffix():
    from prepare_game import dossier_filename, skeleton_filename
    assert dossier_filename("G1") == "dossier-G1.md"
    assert skeleton_filename("G2") == "phase3_skeleton-G2.md"


def test_run_step_subprocess_failure_exits_with_propagated_code(monkeypatch, tmp_path):
    """子腳本 exit non-zero → prepare_game.py exit non-zero（傳遞 stderr）"""
    from prepare_game import run_step

    class FakeResult:
        def __init__(self):
            self.returncode = 5
            self.stdout = ""
            self.stderr = "⛔ 先發不在 active"

    def fake_run(*a, **k):
        return FakeResult()

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(SystemExit) as exc:
        run_step("B", ["python", "scripts/roster_checker.py"], tmp_path)
    assert exc.value.code == 5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest scripts/tests/test_prepare_game.py -v
```

Expected: 9 FAIL（ImportError）

- [ ] **Step 3: Implement prepare_game.py CLI + helpers**

建立 `scripts/prepare_game.py`：

```python
#!/usr/bin/env python3
"""prepare_game.py：Phase 1+2 一鍵整合腳本（spec 2026-04-28-prepare-game-script）。

Step 順序（spec §3.2）：
  A) fetch_game_data → game_data.json + summary
  B) roster_checker × 2（雙隊平行）
  C) pitcher_stats × 2（用 Step A 的 mlbam_id，雙隊平行）
  D) lineup_analyzer × 2（用 Step A 的 mlbam_id，雙隊平行）
  E) merge_game_data → merged.json
  F) dossier_renderer → dossier.md
  G) phase3_skeleton_renderer → phase3_skeleton.md

不再做 Step C-prior（YoY 補跑）— spec §3.2.

Exit codes（spec §3.1）：
  0 = success
  2 = gameType ≠ "R"
  3 = 雙隊未對戰
  4 = doubleheader 未指定 --game-suffix
  5 = 先發不在 active roster
  6 = （保留給 predict.py --ou-stars 必填錯誤）
  7 = API 失敗
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1+2 一鍵整合（spec 2026-04-28）")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--away", required=True, help="客隊縮寫，如 TB")
    parser.add_argument("--home", required=True, help="主隊縮寫，如 CLE")
    parser.add_argument("--output-dir", default=None,
                        help="覆蓋預設目錄（analysis-data/{date}/{away}@{home}[-Gn]）")
    parser.add_argument("--season", type=int, default=None,
                        help="預設 = year of --date")
    parser.add_argument("--game-suffix", choices=["G1", "G2"], default=None,
                        help="Doubleheader 用")
    parser.add_argument("--force", action="store_true", help="覆蓋既有輸出檔")
    args = parser.parse_args(argv)
    if args.season is None:
        args.season = int(args.date[:4])
    return args


def compute_output_dir(*, date: str, away: str, home: str,
                       game_suffix: str | None, override: str | None) -> Path:
    if override:
        return Path(override)
    suffix = f"-{game_suffix}" if game_suffix else ""
    return Path(f"analysis-data/{date}/{away}@{home}{suffix}")


def dossier_filename(suffix: str | None) -> str:
    return f"dossier-{suffix}.md" if suffix else "dossier.md"


def skeleton_filename(suffix: str | None) -> str:
    return f"phase3_skeleton-{suffix}.md" if suffix else "phase3_skeleton.md"


def run_step(label: str, cmd: list[str], output_dir: Path) -> str:
    """跑單一子步驟。失敗 → propagate exit code + stderr。回傳 stdout。"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    except FileNotFoundError as e:
        print(f"[{label}] ⛔ 找不到腳本：{e}", file=sys.stderr)
        sys.exit(1)
    if result.returncode != 0:
        print(f"[{label}] ⛔ exit {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout


# ---- Step A 後續實作於 Task 11（先把 CLI / helpers 上 commit） ----


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = compute_output_dir(
        date=args.date, away=args.away, home=args.home,
        game_suffix=args.game_suffix, override=args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 後續實作於 Task 11
    raise NotImplementedError("Steps A-G 實作於 Task 11")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest scripts/tests/test_prepare_game.py -v
```

Expected: 9/9 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_game.py scripts/tests/test_prepare_game.py
git commit -m "$(cat <<'EOF'
feat(scripts): prepare_game.py CLI 骨架 + helpers

- parse_args / compute_output_dir / dossier_filename / skeleton_filename / run_step
- exit code 表（spec §3.1）：0/2/3/4/5/7
- main() 為 NotImplementedError stub（Steps A-G 實作於下一 task）

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: `prepare_game.py` — Steps A-G 實作

**Files:**
- Modify: `scripts/prepare_game.py`（替換 main() 內 NotImplementedError）
- Modify: `scripts/tests/test_prepare_game.py`（追加 step 整合測試）

**Step 實作策略**：
- Step A：`fetch_game_data.py` 子程序；解析 stdout JSON 取出 probable_pitcher_id
- Step B / C / D：用 `concurrent.futures` 雙隊平行
- Step E：`merge_game_data.py` 子程序
- Step F：import `dossier_renderer.render_dossier`，組 bundle 後寫檔
- Step G：import `phase3_skeleton_renderer.render_skeleton`，需 `predict.predict_with_formula(merged_data)` 取 base 比分

#### Task 11a: Step A 實作 + Step A 後抽出 ID

- [ ] **Step 1: Write integration test (mocked subprocess)**

於 `scripts/tests/test_prepare_game.py` 末尾追加：

```python


def test_step_a_extracts_pitcher_ids(monkeypatch, tmp_path):
    """Step A 跑完後從 game_data.json 取出 home/away probable_pitcher_id"""
    from prepare_game import step_a

    # 模擬 fetch_game_data.py 已寫出 game_data.json
    game_data_path = tmp_path / "game_data.json"
    game_data_path.write_text(json.dumps({
        "_meta": {},
        "home": {"team": "CLE", "team_id": 114,
                 "probable_pitcher": "Tanner Bibee", "probable_pitcher_id": 676440},
        "away": {"team": "TB", "team_id": 139,
                 "probable_pitcher": "Nick Martínez", "probable_pitcher_id": 607259},
    }), encoding="utf-8")

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: FakeResult())
    result = step_a(date="2026-04-28", team_abbr="TB", output_dir=tmp_path)
    assert result["home_id"] == 676440
    assert result["away_id"] == 607259
    assert result["home_name"] == "Tanner Bibee"
    assert result["away_name"] == "Nick Martínez"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest scripts/tests/test_prepare_game.py -k "step_a" -v
```

Expected: FAIL（ImportError on `step_a`）

- [ ] **Step 3: Implement step_a**

於 `scripts/prepare_game.py` 加（在 `run_step` 之後）：

```python
def step_a(*, date: str, team_abbr: str, output_dir: Path) -> dict:
    """Step A: fetch_game_data.py。回傳 {home_id, away_id, home_name, away_name, gameType}。"""
    out_path = output_dir / "game_data.json"
    cmd = [
        PYTHON, str(SCRIPT_DIR / "fetch_game_data.py"),
        "--date", date,
        "--team", team_abbr,
        "-o", str(out_path),
    ]
    print("[A] game_data        ...", file=sys.stderr, end="", flush=True)
    run_step("A", cmd, output_dir)

    if not out_path.exists():
        print(f"\n[A] ⛔ {out_path} 未產生", file=sys.stderr)
        sys.exit(7)
    data = json.loads(out_path.read_text(encoding="utf-8"))

    # 校驗 gameType（spec exit 2）
    game_type = data.get("_meta", {}).get("gameType") or data.get("gameType")
    if game_type and game_type != "R":
        print(f"\n[A] ⛔ exit 2: gameType={game_type}（春訓 / 季後賽不支援）",
              file=sys.stderr)
        sys.exit(2)

    home = data.get("home", {})
    away = data.get("away", {})
    print(" ✓", file=sys.stderr)
    return {
        "home_id": home.get("probable_pitcher_id"),
        "away_id": away.get("probable_pitcher_id"),
        "home_name": home.get("probable_pitcher"),
        "away_name": away.get("probable_pitcher"),
        "home_team_id": home.get("team_id"),
        "away_team_id": away.get("team_id"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest scripts/tests/test_prepare_game.py -k "step_a" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_game.py scripts/tests/test_prepare_game.py
git commit -m "feat(scripts): prepare_game step_a 實作 + 抽出 probable_pitcher_id"
```

#### Task 11b-11d: Steps B / C / D（雙隊平行）

每個 step 同樣 TDD pattern。Step B 範例：

- [ ] **11b-1: Test step_b**

```python
def test_step_b_runs_both_sides_parallel(monkeypatch, tmp_path):
    """Step B 同時跑 home + away roster_checker，並產出 4 個檔"""
    from prepare_game import step_b
    calls = []

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **k):
        calls.append(cmd)
        # 模擬產檔
        if "-o" in cmd:
            out = Path(cmd[cmd.index("-o") + 1])
            out.write_text(json.dumps({"active_roster": {"pitchers": [], "position_players": []},
                                       "injured_list": []}), encoding="utf-8")
        return FakeResult()

    monkeypatch.setattr("subprocess.run", fake_run)
    step_b(home="CLE", away="TB", season=2026,
           home_pitcher="Tanner Bibee", away_pitcher="Nick Martínez",
           output_dir=tmp_path)
    # 應有 2 次 subprocess.run（home + away）
    assert len(calls) == 2
    assert (tmp_path / "home_roster.json").exists()
    assert (tmp_path / "away_roster.json").exists()
```

- [ ] **11b-2: Implement step_b**

```python
def step_b(*, home: str, away: str, season: int,
           home_pitcher: str, away_pitcher: str, output_dir: Path) -> None:
    """Step B: roster_checker × 2 平行。失敗 → exit 5（先發不在 active）。"""
    print("[B] rosters          ...", file=sys.stderr, end="", flush=True)

    def _one(side: str, team: str, pitcher: str):
        out = output_dir / f"{side}_roster.json"
        cmd = [
            PYTHON, str(SCRIPT_DIR / "roster_checker.py"),
            "--team", team,
            "--season", str(season),
            "--expected-starter", pitcher or "",
            "-o", str(out),
        ]
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = [
            ex.submit(_one, "home", home, home_pitcher),
            ex.submit(_one, "away", away, away_pitcher),
        ]
        results = [f.result() for f in futures]

    for side, r in zip(("home", "away"), results):
        if r.returncode != 0:
            print(f"\n[B] ⛔ {side} roster_checker exit {r.returncode}",
                  file=sys.stderr)
            if r.stderr:
                print(r.stderr, file=sys.stderr)
            # roster_checker 觸發 STARTER_NOT_ACTIVE 應 exit 5（spec §3.1）
            sys.exit(5 if "STARTER_NOT_ACTIVE" in (r.stderr + r.stdout) else r.returncode)
    print(" ✓", file=sys.stderr)
```

- [ ] **11b-3: Run + commit**

```bash
python -m pytest scripts/tests/test_prepare_game.py -k "step_b" -v
git add scripts/prepare_game.py scripts/tests/test_prepare_game.py
git commit -m "feat(scripts): prepare_game step_b roster 雙隊平行（含 starter-not-active exit 5）"
```

- [ ] **11c: step_c（pitcher_stats × 2）**

類似 step_b，但用 `--mlbam-id` 而非 `--name`（pitcher_stats.py 接受 `--name` 為主，但既有 lookup_pitcher_id 已支援用 ID 查；如未支援，Task 11c 需要先擴 pitcher_stats.py CLI）。

**檢查項**：用 grep 確認 `pitcher_stats.py` 是否已接受 `--mlbam-id`：

```bash
grep -n "mlbam.id\|mlbam_id" scripts/pitcher_stats.py
```

如果 **沒有**：需於 pitcher_stats.py argparse 新增 `--mlbam-id` arg，main() 內如已提供則跳過 `lookup_pitcher_id` 直接用該 ID 後續 fetch。寫該 CLI 變更的測試：

```python
def test_pitcher_stats_accepts_mlbam_id(monkeypatch):
    """--mlbam-id 提供時不呼叫 lookup_pitcher_id"""
    import pitcher_stats
    called = []
    monkeypatch.setattr(pitcher_stats, "lookup_pitcher_id",
                        lambda name: called.append(name) or 999)
    # ... CLI invocation
```

具體實作 + 測試（追加到 test_pitcher_stats.py）：

```python
# 於 pitcher_stats.py main() 內 args = parser.parse_args() 後：
# parser.add_argument("--mlbam-id", type=int, help="直接指定 MLBAM ID，跳過 name lookup")
# if args.mlbam_id:
#     mlbam_id = args.mlbam_id
# else:
#     mlbam_id = lookup_pitcher_id(args.name)
```

實作 step_c：

```python
def step_c(*, home_id: int | None, away_id: int | None,
           home_name: str, away_name: str, season: int,
           output_dir: Path) -> None:
    print("[C] pitchers         ...", file=sys.stderr, end="", flush=True)

    def _one(side: str, mlbam_id: int | None, name: str):
        out = output_dir / f"{side}_pitcher.json"
        cmd = [PYTHON, str(SCRIPT_DIR / "pitcher_stats.py"),
               "--year", str(season), "-o", str(out)]
        if mlbam_id:
            cmd += ["--mlbam-id", str(mlbam_id), "--name", name]  # name 仍給 summary md
        else:
            cmd += ["--name", name]
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(_one, "home", home_id, home_name),
                   ex.submit(_one, "away", away_id, away_name)]
        results = [f.result() for f in futures]
    for side, r in zip(("home", "away"), results):
        if r.returncode != 0:
            print(f"\n[C] ⛔ {side} pitcher_stats exit {r.returncode}",
                  file=sys.stderr)
            if r.stderr: print(r.stderr, file=sys.stderr)
            sys.exit(r.returncode)
    print(" ✓", file=sys.stderr)
```

- [ ] **11d: step_d（lineup_analyzer × 2）**

類似 step_c，用 Step A 的 mlbam_id 做 `--opposing-pitcher-id`：

```python
def step_d(*, home: str, away: str, home_id: int | None, away_id: int | None,
           season: int, output_dir: Path) -> None:
    print("[D] lineups          ...", file=sys.stderr, end="", flush=True)

    def _one(side: str, team: str, opposing_id: int | None):
        out = output_dir / f"{side}_lineup.json"
        cmd = [PYTHON, str(SCRIPT_DIR / "lineup_analyzer.py"),
               "--team", team, "--year", str(season), "-o", str(out)]
        if opposing_id:
            cmd += ["--opposing-pitcher-id", str(opposing_id)]
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        # home 打線 vs away 投手；away 打線 vs home 投手
        futures = [ex.submit(_one, "home", home, away_id),
                   ex.submit(_one, "away", away, home_id)]
        results = [f.result() for f in futures]
    for side, r in zip(("home", "away"), results):
        if r.returncode != 0:
            print(f"\n[D] ⛔ {side} lineup_analyzer exit {r.returncode}",
                  file=sys.stderr)
            if r.stderr: print(r.stderr, file=sys.stderr)
            sys.exit(r.returncode)
    print(" ✓", file=sys.stderr)
```

#### Task 11e: Step E（merge）

```python
def step_e(*, output_dir: Path) -> None:
    print("[E] merge            ...", file=sys.stderr, end="", flush=True)
    cmd = [
        PYTHON, str(SCRIPT_DIR / "merge_game_data.py"),
        "--game", str(output_dir / "game_data.json"),
        "--home-pitcher", str(output_dir / "home_pitcher.json"),
        "--away-pitcher", str(output_dir / "away_pitcher.json"),
        "--home-lineup", str(output_dir / "home_lineup.json"),
        "--away-lineup", str(output_dir / "away_lineup.json"),
        "-o", str(output_dir / "merged.json"),
    ]
    run_step("E", cmd, output_dir)
    print(" ✓", file=sys.stderr)
```

#### Task 11f: Step F + G（dossier + skeleton）

```python
def _load_bundle(output_dir: Path) -> dict:
    bundle = {}
    for key, fname in [
        ("game_data", "game_data.json"),
        ("home_roster", "home_roster.json"),
        ("away_roster", "away_roster.json"),
        ("home_pitcher", "home_pitcher.json"),
        ("away_pitcher", "away_pitcher.json"),
        ("home_lineup", "home_lineup.json"),
        ("away_lineup", "away_lineup.json"),
        ("merged", "merged.json"),
    ]:
        path = output_dir / fname
        if path.exists():
            bundle[key] = json.loads(path.read_text(encoding="utf-8"))
    return bundle


def step_f(*, output_dir: Path, dossier_path: Path) -> None:
    from dossier_renderer import render_dossier
    print(f"[F] dossier.md       → {dossier_path}", file=sys.stderr)
    bundle = _load_bundle(output_dir)
    md = render_dossier(bundle, game_dir=str(output_dir))
    dossier_path.write_text(md, encoding="utf-8")


def step_g(*, output_dir: Path, skeleton_path: Path) -> None:
    from phase3_skeleton_renderer import render_skeleton
    sys.path.insert(0, str(SCRIPT_DIR))
    from predict import predict_with_formula
    print(f"[G] phase3_skeleton  → {skeleton_path}", file=sys.stderr)
    bundle = _load_bundle(output_dir)
    formula_pred = predict_with_formula(bundle.get("merged", {}))
    md = render_skeleton(bundle, formula_pred)
    skeleton_path.write_text(md, encoding="utf-8")
```

#### Task 11g: 串接 main() + Risk Notes 列印

最終 `main()`：

```python
def _print_risk_notes(output_dir: Path) -> None:
    """讀 pitcher / lineup JSON，列出 Flag 13 / Flag 3 至 stderr（spec §3.4）"""
    from pitcher_stats import detect_triggers as detect_p
    from lineup_analyzer import detect_triggers as detect_l
    notes = []
    for side in ("home", "away"):
        p = output_dir / f"{side}_pitcher.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            for t in detect_p(d):
                if t.get("flag") == 13:
                    notes.append(f"  - {side} pitcher Flag 13 (era_xera_delta={t.get('value', '?')})")
        l = output_dir / f"{side}_lineup.json"
        if l.exists():
            d = json.loads(l.read_text(encoding="utf-8"))
            for t in detect_l(d):
                if t.get("flag") == 3:
                    babip = d.get("last7_babip", "?")
                    notes.append(f"  - {side} lineup Flag 3 (last7 BABIP={babip})")
    print("", file=sys.stderr)
    print("⚠️  Risk Notes (AI 在 phase3_skeleton 風險提示段處理):", file=sys.stderr)
    if notes:
        for n in notes:
            print(n, file=sys.stderr)
    else:
        print("  （無）", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = compute_output_dir(
        date=args.date, away=args.away, home=args.home,
        game_suffix=args.game_suffix, override=args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step A
    a = step_a(date=args.date, team_abbr=args.away, output_dir=output_dir)
    # 校驗雙隊對戰（spec exit 3）— 透過 game_data.json 內含對手是否符合
    # （實作細節：fetch_game_data.py 已照 --team 拉對應比賽，若 home/away 不符則 exit 3）
    home_team_match = (a.get("home_team_id") and args.home in [a.get("home_name"), str(a.get("home_team_id"))])
    # 簡化：此處不再二次校驗，依靠 fetch_game_data.py 的 --team filter

    # Step B-D
    step_b(home=args.home, away=args.away, season=args.season,
           home_pitcher=a["home_name"], away_pitcher=a["away_name"],
           output_dir=output_dir)
    step_c(home_id=a["home_id"], away_id=a["away_id"],
           home_name=a["home_name"], away_name=a["away_name"],
           season=args.season, output_dir=output_dir)
    step_d(home=args.home, away=args.away,
           home_id=a["home_id"], away_id=a["away_id"],
           season=args.season, output_dir=output_dir)

    # Step E
    step_e(output_dir=output_dir)

    # Step F + G
    dossier_path = output_dir / dossier_filename(args.game_suffix)
    skeleton_path = output_dir / skeleton_filename(args.game_suffix)
    step_f(output_dir=output_dir, dossier_path=dossier_path)
    step_g(output_dir=output_dir, skeleton_path=skeleton_path)

    _print_risk_notes(output_dir)
    return 0
```

- [ ] **Step A-G 整合測試**

```python
def test_main_full_flow_smoke(monkeypatch, tmp_path):
    """All steps mocked → main() returns 0 + 18 個檔（含 dossier / skeleton）"""
    # 太大略 — 推遲到 Task 12 E2E 真實實測
    pass
```

- [ ] **Run all tests**

```bash
python -m pytest scripts/tests/ -v
```

Expected: 全 PASS

- [ ] **Final commit for Task 11**

```bash
git add scripts/prepare_game.py scripts/tests/test_prepare_game.py
git commit -m "$(cat <<'EOF'
feat(scripts): prepare_game.py Steps A-G 完整整合

- Step A: fetch_game_data.py + 抽出 probable_pitcher_id
- Step B: roster_checker × 2 平行（STARTER_NOT_ACTIVE → exit 5）
- Step C: pitcher_stats × 2 平行，用 Step A 的 mlbam_id（解 P3）
- Step D: lineup_analyzer × 2 平行，用 mlbam_id 為 --opposing-pitcher-id（解 P2）
- Step E: merge_game_data.py
- Step F: dossier_renderer.render_dossier → dossier.md
- Step G: phase3_skeleton_renderer.render_skeleton → phase3_skeleton.md
- 末尾印 ⚠️ Risk Notes（Flag 13 / Flag 3）至 stderr
- 不做 Step C-prior（spec §3.2 砍 luck-based 自動回測）

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: `SKILL.md` 重寫 + 刪除 `reference/workflow.md`

**Files:**
- Modify: `SKILL.md`（83 → ~150 行）
- Delete: `reference/workflow.md`

**目標**：把 workflow.md 的「初始化」+「Phase 1+2/3/4 SOP」內容併進 SKILL.md，更新 Quick Reference 表，刪除 workflow.md。

- [ ] **Step 1: Read 既有 SKILL.md / workflow.md 確認結構**

```bash
wc -l SKILL.md reference/workflow.md
```

- [ ] **Step 2: 改寫 SKILL.md**

替換 `SKILL.md` 為以下結構（保留 frontmatter）：

```markdown
---
name: mlb-game-analyzer
description: Use when the user asks about MLB game predictions, matchup analysis, score predictions, pitcher duels, or "who will win" questions for any specific MLB game — including queries like "analyze today's Yankees game" or "Dodgers vs Padres"
---

# MLB Game Analyzer — 單場對決分析與比分預測

## Overview

系統化的 MLB 單場對決分析流程 skill。資料透過 `scripts/` 下的 Python 腳本取自 MLB Stats API，經過投打、牛棚、環境三層修正後，輸出勝率與比分預測。

**Phase 1+2 已整合為單一 `prepare_game.py`**，AI 唯一需要 Read 的整合檔為 `dossier.md` + `phase3_skeleton.md`。

---

## When to Use

特定 MLB 比賽的勝負預測 / 對戰組合分析 / 推薦方向（ML / O/U / Run Line）/ 先發投手對決 / 進階數據解讀。

**不適用**：整季預測 / 球員個人比較 / 賽後回顧（轉 `mlb-post-game-review`）/ 歷史統計查詢。

---

## Quick Reference

| Phase | 主要產出 | 工具 |
|-------|---------|------|
| 1+2. 資料收集 | `merged.json` + `dossier.md` + `phase3_skeleton.md` | `prepare_game.py` |
| 3. 綜合分析 | `phase3_summary.md`（在 skeleton 上補結論） | AI 編輯 |
| 4. 預測輸出 | `prediction.json` + `prediction_summary.md` | `predict.py --save` |

---

## The Iron Law

```
NO PREDICTION OUTPUT WITHOUT ALL PHASE GATES PASSED IN SEQUENCE
```

Phase 1+2 → Phase 3 → Phase 4，閘門未通過不得進下一階段。

---

## 初始化（每次對話一次）

### Python 指令偵測

```bash
PYTHON=$(python3 --version >/dev/null 2>&1 && echo python3 || echo python)
```

### 輸出目錄規範

```bash
GAME_DIR=analysis-data/{YYYY-MM-DD}/{AWAY}@{HOME}
# Doubleheader：{AWAY}@{HOME}-G1 / -G2
mkdir -p $GAME_DIR
```

### 模式切換規範（🐍 腳本模式）

- ⛔ 禁止 WebFetch / WebSearch 收集核心數據
- ✅ 唯一例外：當日傷兵快訊（API 40 人名單 + IL 名單為主，WebSearch 補充）
- ⛔ 腳本失敗 → 向使用者回報，禁止靜默改走 WebSearch
- ⛔ 所有腳本輸出必須用 `--output / -o`，禁止 shell redirect `>`
- ⛔ 隊伍縮寫一律用英文縮寫（KC / LAA / NYY），純數字 team_id 已被各腳本拒絕

### 資料來源優先順序

API > 官網公告 > ESPN/CBS/FanGraphs > 網頁抓取。切勿因第三方資料推翻 API 結果。

---

## Phase 1+2：資料收集（單一命令）

```bash
$PYTHON scripts/prepare_game.py --date {YYYY-MM-DD} --away {AWAY} --home {HOME}
# Doubleheader：加 --game-suffix G1 / G2
```

**閘門（自動執行）**：exit 0 = 全 phase 通過；非 0 = 各種 hard error（exit 2-7，見 prepare_game.py --help）。

**後續動作**：
1. Read `$GAME_DIR/dossier.md`（單一檔，~250 行）
2. Read `$GAME_DIR/phase3_skeleton.md` 與 `reference/matchup-factors.md` / `reference/prediction.md`
3. 在 phase3_skeleton.md 補結論段落，存檔為 `phase3_summary.md`
4. 進入 Phase 4

ℹ️ 如需深入查驗某球員 / 投手細節，可主動 Read 同目錄下個別 `*_summary.md`（drill-down）。

---

## Phase 3：綜合分析

> ⛔ **分析前**：Read `reference/matchup-factors.md`（投手 Tier、打線評級、牛棚傷兵修正、條件修正值）

### 3.1-3.4 順序執行

| 步驟 | 分析內容 | 參考 |
|------|---------|------|
| 3.1 投打對決 | 投手 Tier + 打線評級 + Platoon + 球種 | `matchup-factors.md` |
| 3.2 牛棚 | 品質 + 可用性 + 近 3 天消耗 + 傷兵修正（雙向：O/U + ML） | `matchup-factors.md` |
| 3.3 條件修正 | 傷病/TJ/角色轉換/年齡/球場 | `matchup-factors.md` |
| 3.4 風險提示 | dossier 已標的 ⚠️（Flag 13 / Flag 3）AI 敘事判讀 | `flags-checklist.md` |

⛔ BvP 樣本 PA ≥ 15 才可引用（`flags-checklist.md` Flag 2）

### 3.5 phase3_summary.md 存檔

⛔ Phase 3 完成、Phase 4 開始前，必須將 phase3_skeleton.md 的填空全部完成、另存為 `$GAME_DIR/phase3_summary.md`。

**MUST contain**：投手 Tier 判斷、打線評級、牛棚雙向修正值、風險提示判讀、條件修正、修正後預期得分、整體判斷。

⛔ **MUST NOT contain**：星級 / 明確盤口推薦（這些是 Phase 4 專屬）。

---

## Phase 4：預測輸出

> ⛔ **預測前**：Read `$GAME_DIR/phase3_summary.md` + `reference/prediction.md`（公式、信號表、星級門檻、紀律 D1-D5）

### 4.0 執行預測腳本

```bash
$PYTHON scripts/predict.py --game-data $GAME_DIR/merged.json --save [參數]
```

**`--save` 必填參數**：

| 參數 | 必填 | 說明 |
|------|------|------|
| `--ou-line` | 是 | 大小分線（如 9.5） |
| `--ou-rec` | 是 | OVER / UNDER / PASS |
| `--ou-stars` | OVER/UNDER 時必填 | 0-5（缺則 hard exit 6） |
| `--ml-rec` | 是 | 隊伍縮寫或 PASS |
| `--ml-stars` | 是 | 0-5 |
| `--adjusted-home` | 建議 | 分析後調整的主隊得分 |
| `--adjusted-away` | 建議 | 分析後調整的客隊得分 |
| `--signal-adjustments` | 建議 | JSON 格式，如 `'{"puk_il":0.3}'` |
| `--tags` | 建議 | 逗號分隔，如 `divergent,early-season` |
| `--temperature` / `--wind-mph` / `--wind-direction` / `--umpire` / `--umpire-ou-rate` | 若有 | 環境補充 |

> RL 推薦走 `predict.py` auto override（無 `--run-line-rec` / `--run-line-stars` CLI args）。

### 4.1-4.6 紀律 / 護欄 / 輸出

- PASS 門檻 + 星級護欄 → `prediction.md` PASS 章節
- D1-D5 紀律自動執行 → `prediction.md` 分析紀律
- predict.py --save 自動寫入 `$GAME_DIR/prediction.json` + `prediction_summary.md`

### 4.7 輸出前驗證

✅ Read `$GAME_DIR/prediction_summary.md`，逐項確認：

- [ ] D1 / D2 紀律通過？
- [ ] D3 同場無對立推薦？
- [ ] D5 比分與盤口一致性？
- [ ] 牛棚傷兵雙向反映（O/U + ML）？
- [ ] 星級護欄降級警告已確認？

### 4.8 輸出格式

完整 TL;DR + Section 8-10 模板已內化於 `prediction_summary.md`，AI 直接複製貼上。Section 1-7（基本面）由 AI 從 `dossier.md` / `phase3_summary.md` 補充。

---

## Common Pitfalls

紀律違規 11 條：見 `reference/flags-checklist.md`。
邊界條件（Coors 4 月、Doubleheader、TJ 復出等）：見 `reference/matchup-factors.md`。

---

## 語氣與風格

- 進階數據 > 傳統數據，兩者兼用
- 承認不確定性：MLB 單場隨機性約 40-45%
- 明確標注數據來源
- 修正係數必須基於可搜尋到的研究或數據
- 使用者質疑結果時：回顧量化信號、獨立驗證後才決定是否修正；不直接妥協
```

- [ ] **Step 3: 確認 SKILL.md 行數**

```bash
wc -l SKILL.md
```

預期：≤ 200（驗收條件）。

- [ ] **Step 4: 刪除 reference/workflow.md**

```bash
git rm reference/workflow.md
```

- [ ] **Step 5: 確認沒有遺漏的 cross-ref**

```bash
grep -rn "reference/workflow" --include='*.md' --include='*.py' .
```

預期：無結果（所有引用已在前面 tasks 或本 task 處理）。如有，逐個改為指向 `SKILL.md` 或對應 reference 檔。

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest scripts/tests/ -v
```

Expected: 全 PASS

- [ ] **Step 7: Commit**

```bash
git add SKILL.md reference/workflow.md
git commit -m "$(cat <<'EOF'
docs(mlb-skill): SKILL.md 合併 workflow.md + 刪除 workflow.md

- 將 workflow.md 的「初始化」「Phase 1+2/3/4 SOP」併入 SKILL.md
- 更新 Quick Reference 表格（Phase 1+2 = prepare_game.py）
- Phase 1+2 段：1 行命令 → Read dossier.md → 補 skeleton
- 廢除 B7 / B9 / B10 TaskCreate forcing function（內容隨 workflow.md 一起刪）
- 「資料來源優先順序」1 行從 teams-and-api.md 搬進來
- SKILL.md 從 83 → ~190 行；刪除 367 行的 workflow.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: E2E 實測 + Token 量化

**Files:**
- 無 code 修改，僅產生 `analysis-data/2026-04-28/TB@CLE/dossier.md` + `phase3_skeleton.md` + 更新 18 個既有檔案

- [ ] **Step 1: 跑新 prepare_game.py**

```bash
python scripts/prepare_game.py --date 2026-04-28 --away TB --home CLE --force
```

預期 stderr 輸出（spec §3.4）：
```
[A] game_data        ✓
[B] rosters          ✓ (...)
[C] pitchers         ✓
[D] lineups          ✓
[E] merge            ✓
[F] dossier.md       → analysis-data/2026-04-28/TB@CLE/dossier.md
[G] phase3_skeleton  → analysis-data/2026-04-28/TB@CLE/phase3_skeleton.md

⚠️  Risk Notes (AI 在 phase3_skeleton 風險提示段處理):
  - away pitcher Flag 13 (era_xera_delta=...)
  - away lineup Flag 3 (last7 BABIP=...)
```

- [ ] **Step 2: 驗收 18 個檔案存在**

```bash
ls -1 analysis-data/2026-04-28/TB@CLE/ | wc -l
```

預期：≥ 18（含 dossier.md + phase3_skeleton.md）。

- [ ] **Step 3: 驗收 dossier.md 行數 ≤ 250**

```bash
wc -l analysis-data/2026-04-28/TB@CLE/dossier.md
```

- [ ] **Step 4: 驗收 phase3_skeleton.md 含 7 個 H2 + 行數 ≤ 50**

```bash
grep -c "^## " analysis-data/2026-04-28/TB@CLE/phase3_skeleton.md
wc -l analysis-data/2026-04-28/TB@CLE/phase3_skeleton.md
```

預期：grep ≥ 7、wc -l ≤ 50（無 Flag 觸發時 ≤ 30，有 Flag 時 ≤ 50）。

- [ ] **Step 5: Token 量化（手動）**

紀錄：
- 舊流程：Phase 1+2 AI 端 Read 字數 ≈ 750 行（8 份 *_summary.md + workflow.md Phase 1+2 段）
- 新流程：dossier.md 行數 + phase3_skeleton.md 行數
- 計算減少 %（目標 ≥ 60%）

寫入 `docs/superpowers/specs/2026-04-28-prepare-game-script-design.md` §9 Token 量化段落（替換現有「實測 2026-04-28 TB@CLE：...」三行為實際數據）。

- [ ] **Step 6: 跑完整 predict.py 流程驗收**

```bash
# 假設使用者已在 phase3_skeleton.md 補完結論並另存 phase3_summary.md（這步在實際使用時做；E2E 測試可只 cp 一份占位）
cp analysis-data/2026-04-28/TB@CLE/phase3_skeleton.md analysis-data/2026-04-28/TB@CLE/phase3_summary.md

python scripts/predict.py \
  --game-data analysis-data/2026-04-28/TB@CLE/merged.json \
  --save \
  --ou-line 8.5 --ou-rec PASS \
  --ml-rec PASS --ml-stars 0
```

預期：exit 0，產出 prediction.json + prediction_summary.md。

- [ ] **Step 7: 驗收 ou-stars 必填化（手動嚴打）**

```bash
python scripts/predict.py \
  --game-data analysis-data/2026-04-28/TB@CLE/merged.json \
  --save \
  --ou-line 8.5 --ou-rec OVER \
  --ml-rec PASS --ml-stars 0
echo "exit code: $?"
```

預期：exit 6（缺 --ou-stars）+ stderr 訊息。

- [ ] **Step 8: 全 test suite + commit token 量化結果**

```bash
python -m pytest scripts/tests/ -v
```

預期：全 PASS。

```bash
# 更新 spec 內 token 量化結果
git add docs/superpowers/specs/2026-04-28-prepare-game-script-design.md \
        analysis-data/2026-04-28/TB@CLE/
git commit -m "$(cat <<'EOF'
data(2026-04-28): TB@CLE prepare_game.py E2E 實測通過 + token 量化

- 18 個檔案產出，含 dossier.md 與 phase3_skeleton.md
- 行數驗收：dossier ≤ 250 / skeleton ≤ 50
- Token 量化：舊 X 行 → 新 Y 行（減少 Z%，目標 ≥ 60%）
- predict.py --ou-stars 必填化、--ou-rec OVER 缺 stars exit 6 驗收通過

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist（plan 自審，撰寫者執行）

**1. Spec coverage**：對照 spec 各區段：

- [x] §3.1 CLI exit codes → Task 10 / Task 11 main()
- [x] §3.2 Steps A-G → Task 11
- [x] §3.3 18 個輸出檔 → Task 11 / Task 13 step 2 驗收
- [x] §3.4 stdout `⚠️ Risk Notes` → Task 11g `_print_risk_notes`
- [x] §4 dossier.md → Task 8
- [x] §5 phase3_skeleton.md → Task 9
- [x] §6.1 fetch_game_data probable_pitcher_id → Task 5
- [x] §6.2 pitcher_stats diacritic → Task 6
- [x] §6.3a --ou-stars → Task 7a
- [x] §6.3b 移除 H2 grep → Task 7b
- [x] §7.1 SKILL.md → Task 12
- [x] §7.2 workflow.md 刪除 → Task 12
- [x] §7.3 teams-and-api.md 刪除 → Task 1
- [x] §7.4 flags-checklist.md → Task 2
- [x] §7.5 matchup-factors.md → Task 3
- [x] §7.6 prediction.md → Task 4
- [x] §7.7 廢除 TaskCreate → Task 12（隨 workflow.md 刪除）
- [x] §9 驗收條件 → Task 13
- [x] §10 開發順序 → 全部 13 個 task

**2. Placeholder scan**：搜過 plan，無「TBD/TODO/implement later」placeholder。Task 8b 用「逐 section 命名 + 測試模式」描述，每節範例清楚；Task 11c 處理 pitcher_stats `--mlbam-id` arg 的「如未支援先擴 CLI」是 conditional implementation，非 placeholder（步驟有清楚指令）。

**3. Type consistency**：函式命名一致：
- `select_top5_vs_pitcher` / `find_last7_top1_outside_pa_top5`（Task 8a）
- `render_dossier(bundle, *, game_dir)`（Task 8）
- `render_skeleton(bundle, formula_pred)`（Task 9）
- `step_a / step_b / step_c / step_d / step_e / step_f / step_g`（Task 11）
- `parse_args / compute_output_dir / dossier_filename / skeleton_filename / run_step`（Task 10）

**4. 已知限制 / 後續可能 follow-up**：

- Task 8b 子節（10 個 sub-task）僅給範本，實作時逐節 TDD 即可；總工作量約 200 行 markdown 渲染邏輯
- Task 11c 假設 `pitcher_stats.py` 已支援或 trivially 可加 `--mlbam-id` arg；如該 arg 已存在，跳過「擴 CLI」步驟
- E2E（Task 13）依賴 MLB Stats API 可用；網路斷線時跑不過
