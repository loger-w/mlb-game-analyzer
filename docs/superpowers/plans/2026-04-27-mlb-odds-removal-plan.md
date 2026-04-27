# MLB Game Analyzer — 盤口與 Kelly 系統移除 執行計畫

對應 Design Spec：`docs/superpowers/specs/2026-04-27-mlb-odds-removal-design.md`

---

## C1：predict.py 內部清理

### Step 1.1 — predict.py 編輯

**檔案**：`scripts/predict.py`

| 範圍 | 動作 |
|---|---|
| L5（原） | 刪 `import glob` |
| L42-46 SIGNAL_KEYS_PREFIXES | 刪 `"kelly_"` |
| L219 | 刪 `_SNAPSHOT_FILENAME_RE` |
| L222-264 | 刪 `load_closest_snapshot()` 整個函式 |
| L267-344 | 刪 `_NAME_TO_ABBREV` + `resolve_pinnacle_odds()` 整個函式 |
| L400 / L486 | 刪 `kelly_available` 欄位（rl_override dict） |
| L414, L441 | 刪 apply_rl_guardrail signature 中 `kelly_rl_available` 參數 + docstring 提及 |
| L610-611 | 改註釋從「對齊 fetch_odds.py:21」為「MLB 球季 EDT = UTC-4」（fetch_odds.py 將被刪） |
| L634-830 | 刪 `compute_kelly_block()` 整個函式 |
| L860-882 | 刪 CLI args：`--kelly-divisor` / `--kelly-cap` / `--unit-size` / `--no-auto-odds` / `--ml-odds-*-dec` / `--ou-odds-*-dec` / `--rl-odds-*-dec` / `--game-index` |
| L1129-1157 | 刪 main() 內 Kelly 計算流程 + apply_rl_guardrail 呼叫的 `kelly_rl_available=False` kwarg |
| L1203 | 刪 record dict 內 `"kelly": kelly_block,` |

### Step 1.2 — test 檔刪 14 測試

**檔案**：`scripts/tests/test_predict_snapshot.py`（rename 留到 C2）

刪除：
- L1-15 imports + FIXTURES 路徑（保留必要 imports：sys, os, json, pytest）
- L17-26 `_make_snapshot_dir()` helper
- L29-75（4 個 load_closest 測試）
- L78 `from predict import resolve_pinnacle_odds`
- L81-179（5 個 resolve_odds 測試）
- L182-198 `_make_args()` helper
- L201-425（5 個 end-to-end Kelly 測試）

保留 L429+ 所有測試，但需移除：
- `_rl()` helper 中 `kelly_rl_available=False` 一行（在 L498）

### Step 1.3 — 驗證 + commit

```bash
pytest scripts/tests/ -q
python -c "import sys; sys.path.insert(0, 'scripts'); import predict; print('OK')"
python scripts/predict.py \
  --game-data analysis-data/2026-04-26/LAA@KC/merged.json \
  --save \
  --ou-line 9 --ou-rec UNDER --ou-stars 2 \
  --ml-rec KC --ml-stars 2 \
  --skip-phase3-check
python -c "import json; p=json.load(open('analysis-data/2026-04-26/LAA@KC/prediction.json', encoding='utf-8')); assert 'kelly' not in p; print('OK: no kelly')"

git add scripts/predict.py scripts/tests/test_predict_snapshot.py
git commit -m "refactor(mlb-skill): remove Kelly + snapshot logic from predict.py ..."
```

---

## C2：孤立 scripts、資料、fixtures 刪除 + test rename

### Step 2.1 — git rm

```bash
git rm scripts/odds_analyzer.py scripts/fetch_odds.py
git rm scripts/tests/test_kelly.py scripts/tests/test_odds_analyzer_extended.py
git rm scripts/tests/fixtures/sample_snapshot.json
git rm scripts/tests/fixtures/sample_snapshot_earlier.json
git rm scripts/tests/fixtures/sample_snapshot_open.json
git rm scripts/tests/fixtures/sample_snapshot_close.json
git rm scripts/tests/fixtures/sample_merged.json
git add odds_snapshots/  # 確認 staged D 的 34 檔
```

### Step 2.2 — rename test

```bash
git mv scripts/tests/test_predict_snapshot.py scripts/tests/test_predict.py
```

### Step 2.3 — 驗證 + commit

```bash
pytest scripts/tests/ -q
python -c "import sys; sys.path.insert(0, 'scripts'); import predict; print('OK')"

git commit -m "refactor(mlb-skill): delete odds_analyzer, fetch_odds, snapshots & related tests ..."
```

---

## C3：文件 + spec 更新

### Step 3.1 — 刪 odds-format.md

```bash
git rm reference/odds-format.md
```

### Step 3.2 — workflow.md

刪：
- Phase 2 Step 3b「盤口賠率」表格列 + 「盤口分析（使用者提供盤口數據後執行）」段 + odds_analyzer.py bash 例
- Phase 4.0「自動 Odds 查詢」整段註

改：
- Phase 4.3 標題「### 4.3 盤口推薦」→「### 4.3 推薦結果」
- 內文「O/U、ML、Run Line、讓分方向交叉驗證」→「O/U、ML、Run Line 方向交叉驗證」
- 「⚠️ 勝率與比分皆用 predict.py 的 formula_prediction」括號內文加「Kelly / 盤口 snapshot 系統於 2026-04-27 重構移除」

### Step 3.3 — prediction.md

**先**搬 P(margin ≥ 2 | win) 表至 Run Line 章節（L80-83 區）：
- 表內容（含 Source 註） + 「重要」段
- 改 L83 cross-ref 為直接 inline 表

**後**刪整章 Kelly Sizing & Unit Output（含子章 公式 / Odds 來源 / 機率來源 / P(margin) 查表 / Side 標籤來源 / kelly schema / 紀律）

### Step 3.4 — output-format.md

刪 L18-23「💰 建議注碼（Quarter-Kelly, cap 3% of bankroll）」整個表
改 L11「💰 盤口速查：」→「📊 推薦速查：」（與表頭「盤口」改成「市場」）
改 L37「10. 盤口建議（含讓分方向確認 + 一致性檢查）」→「10. 推薦結果（含方向確認 + 一致性檢查）」

### Step 3.5 — SKILL.md

- L3 description：刪「betting lines, 」
- L10：「投打、牛棚、環境、盤口四層修正」→「投打、牛棚、環境三層修正」
- L16：「盤口推薦（ML / O/U / Run Line）」→「推薦方向（ML / O/U / Run Line）」
- L58：保留（紀律性提示，referring to phase 4 outputs）

### Step 3.6 — slimming-design.md superseded 註記

- L44 Non-goals「❌ 重建 fetch_odds + 盤口追蹤系統（決議 D9...）」後加 superseded 註
- L63 D9 表格行：刪除線 + 改決議內容
- L535 後續工作清單第 1 項：刪除線 + superseded 註

### Step 3.7 — 新建文件

- `docs/superpowers/specs/2026-04-27-mlb-odds-removal-design.md`
- `docs/superpowers/plans/2026-04-27-mlb-odds-removal-plan.md`（本檔）

### Step 3.8 — 驗證 + commit

```bash
pytest scripts/tests/ -q
python scripts/predict.py --game-data ... --save --skip-phase3-check  # smoke test

# Grep 死碼殘留
grep -rn "kelly\|odds_analyzer\|fetch_odds\|odds_snapshots\|--ml-odds\|--ou-odds\|--rl-odds\|--no-auto-odds\|--kelly-" \
  SKILL.md reference/ scripts/ \
  --exclude-dir=__pycache__ \
  --exclude-dir=analysis-data
# 應 0 hits（除本次 design / plan 文件）

# 確認盤口字眼僅剩紀律提示
grep -rn "盤口" SKILL.md reference/

git add reference/ SKILL.md docs/
git commit -m "docs(mlb-skill): remove odds/Kelly references, mark slimming D9 superseded ..."
```

---

## 完成驗收

- [ ] 3 commits 全部完成且 pytest 全綠
- [ ] `git log --oneline` 顯示 C1 → C2 → C3 順序
- [ ] LAA@KC smoke test prediction.json 不含 kelly 欄位
- [ ] `grep "kelly\|odds_analyzer" scripts/` 0 hits
- [ ] `grep "盤口" reference/ SKILL.md` 剩餘僅紀律性提示
