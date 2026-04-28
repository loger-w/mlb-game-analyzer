# MLB Game Analyzer Skill 瘦身重構 Implementation Plan

> ⚠️ **2026-04-27 後續變更**：本 plan 中所有 `compute_kelly_block` / Kelly Sizing 相關修改於 2026-04-27 被反向 — 整個 Kelly + Pinnacle snapshot 系統已從本 skill 完全移除。詳見 `docs/superpowers/specs/2026-04-27-mlb-odds-removal-design.md` 與對應 plan。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 清掉 mlb-game-analyzer skill 的 XGBoost 死碼與重複文件規則、把 Park Factor 抽 JSON 並對齊 2023-2025 加權數據、用 6 場景壓力測試驗證瘦身後紀律仍生效。

**Architecture:** 5 個獨立 phase（P1 / P2 / P3a / P3b / P3c），每個 phase 一個 commit、可獨立 review/回滾。P1-P2 純代碼變動由 unit test 把關；P3a 用新增 unit test 驗證 PF 解析；P3b 是純文件重整；P3c 用 subagent 對 6 個情境跑黑箱測試（baseline vs post-change），透過 `git worktree` 在 pre-P3 commit 跑 baseline。

**Tech Stack:** Python 3 + pytest（既有 test 框架）；bash + grep + git（驗證工具）；Agent tool with subagent_type=general-purpose（P3c subagent dispatch）；無新依賴。

**Pre-requisites:**
- 工作分支：`refactor/skill-slimming`（已存在）
- 倉庫根目錄（後文以 `$REPO` 表示）：`C:\Users\USER\.agents\skills\mlb-game-analyzer`
- 後文所有相對路徑均相對於 `$REPO`

**Phase 順序（嚴格）：** P1 → P2 → P3a → P3b → P3c。每 phase 結束有獨立 commit。中途可停在任意 phase。

---

## File Structure

### 將被刪除（Delete）

| 檔 | Phase | 原因 |
|---|---|---|
| `scripts/train.py` | P1 | XGBoost 訓練腳本（決議 D1） |
| `scripts/update_model.py` | P1 | XGBoost 模型更新（D1） |
| `scripts/_backtest_rl_relaxation.py` | P1 | 一次性 backtest，結果存 git history（D3） |
| `scripts/fetch_results.py` | P2 | 屬 mlb-post-game-review skill（D2） |
| `scripts/summarize_predictions.py` | P2 | 屬 mlb-post-game-review skill（D2） |
| `scripts/review_stats.py` | P2 | 屬 mlb-post-game-review skill（D2） |
| `scripts/diagnose_metrics.py` | P2 | 跟 review_stats 走（D2） |
| `scripts/tests/test_fetch_results.py` | P2 | 對應 test |
| `analysis-data/2025-04-24/MIN@TB/game_data.json` | P2 | 孤立 2025 年資料 |
| `plans/2026-04-23-phase1-readability.md` | P2 | working-tree 已刪未 commit |
| `setup_task.bat` | P2 | working-tree 已刪未 commit |
| `setup_task.ps1` | P2 | working-tree 已刪未 commit |
| `analysis-logs/`（整目錄） | P2 | 屬 mlb-post-game-review |
| `reference/pitfalls.md` | P3b | 內容外散到 odds-format.md / SKILL.md（D5） |

### 將被新增（Create）

| 檔 | Phase | 用途 |
|---|---|---|
| `scripts/data/park_factors.json` | P3a | Park Factor 資料來源（取代 hardcoded dict） |
| `docs/superpowers/baselines/2026-04-26-p3c-baseline.md` | P3c | P3c baseline 測試紀錄（pre-P3 state subagent 輸出） |
| `docs/superpowers/baselines/2026-04-26-p3c-postchange.md` | P3c | P3c post-change 測試紀錄 |
| `docs/superpowers/fixtures/p3c-scenarios/T*-*.json` | P3c | 6 個壓力測試情境的 game_data 檔 |

### 將被修改（Modify）

| 檔 | Phase | 改動性質 |
|---|---|---|
| `scripts/predict.py` | P1 | 移除 ML 路徑（imports / `predict_with_ml` / `should_force_ml_pass` / `check_xgb_divergent` / `cross_validation` 欄位 / `xgb_raw_home_pct`）；過時註釋清理 |
| `scripts/tests/test_predict_snapshot.py` | P1 | 移除 ML test、調整 `compute_kelly_block` 呼叫簽名 |
| `scripts/requirements.txt` | P1 + P2 | 刪 5 個套件，最終剩 3 個 |
| `scripts/merge_game_data.py` | P3a | `PARK_FACTORS` dict → JSON 載入 + alias 解析 |
| `reference/workflow.md` | P2, P3b | L328 簡化 + L294 ML 註釋更新 |
| `reference/prediction.md` | P1, P2, P3b | RL 表去重 + 預測紀錄段簡化 + D1 紀律改寫 |
| `reference/matchup-factors.md` | P3a | Park Factor 章節改寫 |
| `reference/flags-checklist.md` | P3b | 13 條 → 每條 2-3 行 |
| `reference/odds-format.md` | P3b | 加「亞洲盤口歧義」段（pitfalls.md 移來） |
| `SKILL.md` | P3b | 刪「最高優先 3 項技術漏洞」+ 加「使用者質疑結果」 |

---

# Phase 1：純代碼清理

**目標：** 清除 XGBoost 死碼、刪除一次性 backtest 腳本、清過時註釋、去重 prediction.md RL 表。**不動 reference/* 和 SKILL.md**（除 prediction.md RL 表）。
**完成判準：** `pytest scripts/tests/` 全綠 + `predict.py --save` 對 2026-04-25 任一場跑得通，產出無 `xgb_raw_home_pct` / `cross_validation` 欄位。
**估時：** 1-2 hr

### Task 1: P1.0 — 記錄 baseline metrics

**Files:**
- Create: `/tmp/baseline_skill_md_lines.txt`、`/tmp/baseline_reference_lines.txt`、`/tmp/baseline_scripts_lines.txt`、`/tmp/baseline_words.txt`

- [ ] **Step 1: 在 repo 根目錄記錄 baseline 行數與字數**

```bash
cd $REPO
wc -l SKILL.md > /tmp/baseline_skill_md_lines.txt
wc -l reference/*.md > /tmp/baseline_reference_lines.txt
wc -l scripts/*.py > /tmp/baseline_scripts_lines.txt
wc -w SKILL.md reference/*.md > /tmp/baseline_words.txt
```

- [ ] **Step 2: 印出三組基準數字並記下來**

```bash
echo "=== SKILL.md ===" && cat /tmp/baseline_skill_md_lines.txt
echo "=== reference/*.md total ===" && tail -1 /tmp/baseline_reference_lines.txt
echo "=== scripts/*.py total ===" && tail -1 /tmp/baseline_scripts_lines.txt
echo "=== words SKILL.md + reference total ===" && tail -1 /tmp/baseline_words.txt
```
Expected：印出 4 個 total，以下為 2026-04-26 已採樣值供對照（若不一致表示 baseline 已經被改動，停下確認）：
- `SKILL.md`：123 行 / 482 字
- `reference/*.md` total：1125 行 / 5704 字
- `scripts/*.py` total：7178 行

把這 4 個數字記在這份 plan 同目錄的 scratch 筆記或當前對話 context（後續 Section 8.2 會用來算「瘦身 ≥ 30%」是否達標）。

### Task 2: P1.1 — 刪除 train.py 與 update_model.py

**Files:**
- Delete: `scripts/train.py`
- Delete: `scripts/update_model.py`

- [ ] **Step 1: 確認兩個檔案存在**

```bash
ls scripts/train.py scripts/update_model.py
```
Expected：兩檔案都列出。

- [ ] **Step 2: grep 確認沒有其他檔 import 它們**

```bash
grep -rn "from train\|import train\b\|update_model" scripts/ reference/ SKILL.md 2>/dev/null
```
Expected：無輸出（或只匹配到 train.py / update_model.py 本身的內部 reference）。

- [ ] **Step 3: 刪除兩個檔案**

```bash
git rm scripts/train.py scripts/update_model.py
```

- [ ] **Step 4: 確認刪除**

```bash
ls scripts/train.py scripts/update_model.py 2>&1
```
Expected：兩檔案都報「No such file」。

### Task 3: P1.1 — 從 predict.py 刪除 ML 路徑（imports + helper functions）

**Files:**
- Modify: `scripts/predict.py`

- [ ] **Step 1: 刪 import joblib（L14）與 import numpy as np（L15）**

old (L14-15)：
```python
import joblib
import numpy as np
```

new：（兩行整段刪除，後面空行保留以維持後續 `# Fix Windows encoding` 區塊上方有空行）

驗證：
```bash
grep -n "joblib\|numpy" scripts/predict.py
```
Expected：無輸出。

- [ ] **Step 2: 刪 MODELS_DIR / WIN_MODEL_PATH（L22-23）**

old：
```python
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
WIN_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_win_model.pkl")
```

new：（兩行整段刪除）

- [ ] **Step 3: 刪 should_force_ml_pass 整個函式（L94-106）**

old：
```python
def should_force_ml_pass(ml_pred: dict | None, formula_pred: dict | None) -> bool:
    """α 實作（spec 2026-04-22 §3.2）：ml_lean 與 formula_lean 方向分歧 → 強制 PASS。

    取代原 D1（讀 cross_validation == "DIVERGENT"）和 D1.5 方向分歧 branch。
    cross_validation 欄位仍寫入 prediction.json 作歷史觀察，但決策邏輯不依賴字串。

    Returns True iff 兩個模型都存在且方向分歧（跨 50% 邊界）；其餘情境 False。
    """
    if not ml_pred or not formula_pred:
        return False
    ml_lean = "HOME" if ml_pred["home_win_pct"] > 50 else "AWAY"
    formula_lean = "HOME" if formula_pred["log5_pct"] > 50 else "AWAY"
    return ml_lean != formula_lean
```

new：（整個函式刪除，包含上方一個空行）

- [ ] **Step 4: 刪 check_xgb_divergent 整個函式（從 L109 def 到函式結束）**

```bash
grep -n "^def check_xgb_divergent\|^def " scripts/predict.py | head -20
```
找出 `check_xgb_divergent` 之後下一個 `def` 的行號 — 這兩個之間（含 `def check_xgb_divergent`、不含下一個 `def` 的空行）整段刪除。

驗證：
```bash
grep -n "def check_xgb_divergent\|def should_force_ml_pass\|def predict_with_ml" scripts/predict.py
```
Expected：只列出 `def predict_with_ml`（仍待 Step 5 刪）。

- [ ] **Step 5: 刪 predict_with_ml 整個函式（L611-626，刪除後重新 grep 確認行號）**

刪除位置：搜尋 `^def predict_with_ml\b`，刪到下一個 `^def ` 開頭之前（含 `def predict_with_ml` 上方一個空行、下方一個空行）。

驗證：
```bash
grep -n "predict_with_ml" scripts/predict.py
```
Expected：無輸出。

- [ ] **Step 6: 刪 FEATURE_COLS（L76）整個列表**

```bash
grep -n "^FEATURE_COLS\b\|FEATURE_COLS" scripts/predict.py
```
找到 FEATURE_COLS 的定義（從 `FEATURE_COLS = [` 到 `]`），整段刪除（刪除後僅 main 流程 L978 的 reference 仍存在，下個 task 會處理）。

### Task 4: P1.1 — 重寫 predict.py main 流程，移除 ML 與 cross_validation

**Files:**
- Modify: `scripts/predict.py`

- [ ] **Step 1: 刪 main flow 中 features + ml_pred 取得（原 L977-981）**

old（L977-981）：
```python
    # 建構特徵向量
    features = [data.get(col, 0) for col in FEATURE_COLS]

    # ML 預測
    ml_pred = predict_with_ml(features)

    # 公式預測
```

new：
```python
    # 公式預測
```

- [ ] **Step 2: 刪 cross_validation 計算區塊（原 L1006-1016）**

old（L1006-1016）：
```python
    # 交叉驗證
    cross_validation = "NO_ML_MODEL"
    if ml_pred:
        if min_season_games < 30:
            cross_validation = "INSUFFICIENT_SAMPLE"
        else:
            ml_lean = "HOME" if ml_pred["home_win_pct"] > 50 else "AWAY"
            xval_formula = formula_30_pred if formula_30_pred else formula_pred
            formula_lean = "HOME" if xval_formula["log5_pct"] > 50 else "AWAY"
            pct_diff = abs(ml_pred["home_win_pct"] - xval_formula["log5_pct"])
            cross_validation = "CONSISTENT" if ml_lean == formula_lean else "DIVERGENT"
```

new：（整段刪除；`formula_30_pred` 計算之後直接接「最終推薦」）

- [ ] **Step 3: 改 final_pct 賦值（原 L1021-1024）**

old：
```python
    if ml_pred:
        final_pct = ml_pred["home_win_pct"]
    else:
        final_pct = formula_pred["log5_pct"]
```

new：
```python
    final_pct = formula_pred["log5_pct"]
```

- [ ] **Step 4: 修 has_adjusted 分支（原 L1033-1042）— 移除「adjusted vs XGBoost 反向」註釋**

old：
```python
    # 決定最終方向：adjusted 比分優先於 XGBoost
    has_adjusted = args.adjusted_home is not None or args.adjusted_away is not None
    if has_adjusted and (adj_home > adj_away) != (final_pct > 50):
        # adjusted 比分方向與 XGBoost 相反 → 使用 Log5 勝率
        adjusted_winner = "HOME" if adj_home > adj_away else "AWAY"
        adjusted_pct = formula_pred["log5_pct"] if adjusted_winner == "HOME" else round(100 - formula_pred["log5_pct"], 1)
        display_home_pct = round(formula_pred["log5_pct"], 1)
    else:
        adjusted_winner = "HOME" if final_pct > 50 else "AWAY"
        display_home_pct = round(final_pct, 1)
```

new：
```python
    # 決定最終方向：adjusted 比分優先於 formula 勝率
    has_adjusted = args.adjusted_home is not None or args.adjusted_away is not None
    if has_adjusted and (adj_home > adj_away) != (final_pct > 50):
        adjusted_winner = "HOME" if adj_home > adj_away else "AWAY"
        display_home_pct = round(formula_pred["log5_pct"], 1)
    else:
        adjusted_winner = "HOME" if final_pct > 50 else "AWAY"
        display_home_pct = round(final_pct, 1)
```

（同時刪掉未使用的 `adjusted_pct` 中間變數）

- [ ] **Step 5: 修 result dict（原 L1044-1059）— 移除 ml_prediction / cross_validation 欄位**

old：
```python
    result = {
        "ml_prediction": ml_pred,
        "formula_prediction": formula_pred,
        "cross_validation": cross_validation,
        "signal_table": signal_table,
        "final": {
            "recommended_winner": adjusted_winner,
            "home_win_pct": display_home_pct,
            "confidence": "HIGH" if cross_validation == "CONSISTENT" else ("MEDIUM" if cross_validation == "NO_ML_MODEL" else "LOW"),
            "predicted_home_score": adj_home,
            "predicted_away_score": adj_away,
            "predicted_total": adj_total,
            "signal_run_adjustment": signal_table["total_run_adjustment"],
            "over_under_lean": "OVER" if signal_table["total_run_adjustment"] > 0 else ("UNDER" if signal_table["total_run_adjustment"] < 0 else "NEUTRAL"),
        },
    }
```

new：
```python
    result = {
        "formula_prediction": formula_pred,
        "signal_table": signal_table,
        "final": {
            "recommended_winner": adjusted_winner,
            "home_win_pct": display_home_pct,
            "confidence": "MEDIUM",
            "predicted_home_score": adj_home,
            "predicted_away_score": adj_away,
            "predicted_total": adj_total,
            "signal_run_adjustment": signal_table["total_run_adjustment"],
            "over_under_lean": "OVER" if signal_table["total_run_adjustment"] > 0 else ("UNDER" if signal_table["total_run_adjustment"] < 0 else "NEUTRAL"),
        },
    }
```

> 說明：`confidence` 暫時固定 `"MEDIUM"`。後續若要從 BABIP / 牛棚 IL / 信號數量再算動態 confidence，是另一個 task，不在本重構範圍。

- [ ] **Step 6: 刪 should_force_ml_pass 呼叫與 Y2 區塊（原 L1134-1149）**

```bash
grep -n "should_force_ml_pass\|check_xgb_divergent\|y2_triggered\|xgb_home_lean" scripts/predict.py
```

對應行整段刪除：
- 原 L1134-1139（should_force_ml_pass 呼叫 + cap_reasons append）
- 原 L1141-1149+（Y2 區塊；繼續往下刪到 `cap_reasons.append(...)` 結束。grep `y2_triggered` 找到所有相關行，整段刪除。注意：`cap_reasons.append("ml/formula 方向分歧 強制 PASS（α 實作）")` 與「Y2 cumulative #8」相關 print 都要刪掉。`force_ml_pass` 變數仍保留 — 後續其他護欄分支會設它）

驗證：
```bash
grep -n "should_force_ml_pass\|check_xgb_divergent\|y2_triggered\|xgb_home_lean\|α 實作" scripts/predict.py
```
Expected：無輸出。

- [ ] **Step 7: 修 compute_kelly_block 簽名 — 移除 ml_prediction 參數**

old（L700-714）：
```python
def compute_kelly_block(
    args,
    merged: dict,
    ml_prediction: dict | None,
    formula_prediction: dict,
    final_ml_rec: str,
    final_ou_rec: str,
    final_rl_rec: str,
) -> dict | None:
    """Build the kelly block for prediction.json.

    I1: PASS markets → kelly.{market} = None (kelly must align with D1-D5 guardrail).
    C1: ET date extracted from analysis-data/YYYY-MM-DD/ path (fallback: UTC→ET).
    C3: Pass Pinnacle rl.home_point into analyze_run_line for truthful side labeling.
    """
```

new：
```python
def compute_kelly_block(
    args,
    merged: dict,
    formula_prediction: dict,
    final_ml_rec: str,
    final_ou_rec: str,
    final_rl_rec: str,
) -> dict | None:
    """Build the kelly block for prediction.json.

    I1: PASS markets → kelly.{market} = None (kelly must align with D1-D5 guardrail).
    C1: ET date extracted from analysis-data/YYYY-MM-DD/ path (fallback: UTC→ET).
    C3: Pass Pinnacle rl.home_point into analyze_run_line for truthful side labeling.
    """
```

- [ ] **Step 8: 修 compute_kelly_block 內部 model_p_home 取得（原 L823-828）**

old：
```python
    # ML Kelly
    model_p_home = None
    if ml_prediction is not None:
        pct = ml_prediction.get("home_win_pct")
        if pct is not None:
            model_p_home = pct / 100.0
```

new：
```python
    # ML Kelly — 用 formula log5 作為 model_p_home（XGBoost 路徑已於 P1 移除）
    model_p_home = None
    pct = formula_prediction.get("log5_pct")
    if pct is not None:
        model_p_home = pct / 100.0
```

- [ ] **Step 9: 修 main 流程裡 compute_kelly_block 的呼叫（原 L1261-1266）**

old：
```python
            kelly_block = compute_kelly_block(
                args, data, ml_pred, formula_pred,
                final_ml_rec=final_ml_rec,
                final_ou_rec=final_ou_rec,
                final_rl_rec=final_rl_rec,
            )
```

new：
```python
            kelly_block = compute_kelly_block(
                args, data, formula_pred,
                final_ml_rec=final_ml_rec,
                final_ou_rec=final_ou_rec,
                final_rl_rec=final_rl_rec,
            )
```

- [ ] **Step 10: 刪 record dict 裡 xgb_raw_home_pct（原 L1292）與 cross_validation（原 L1312）**

```bash
grep -n "xgb_raw_home_pct\|cross_validation" scripts/predict.py
```

兩行（`"xgb_raw_home_pct": ml_pred["home_win_pct"] if ml_pred else None,` 與 `"cross_validation": result["cross_validation"],`）整行刪除。

驗證：
```bash
grep -n "xgb_raw_home_pct\|cross_validation\|ml_pred\b\|ml_prediction" scripts/predict.py
```
Expected：無輸出。

### Task 5: P1.1 — 修 test_predict_snapshot.py 移除 ML 相關 test

**Files:**
- Modify: `scripts/tests/test_predict_snapshot.py`

- [ ] **Step 1: 刪除 should_force_ml_pass 整個 test 區塊（原 L605-657）**

對應區塊：
```python
# ============================================================================
# should_force_ml_pass helper tests (α 實作 — D1 改讀 ml_lean vs formula_lean)
# ============================================================================
from predict import should_force_ml_pass

# ... 6 個 test functions ...
```

整個區塊（含 header comment + import + 6 個 def test_should_force_ml_pass_*）刪除。

- [ ] **Step 2: 刪除 check_xgb_divergent / Y2 整個 test 區塊（原 L830-859）**

對應區塊：
```python
# ============================================================================
# Plan B 2026-04-22 — Y2: xgb_home_lean vs predicted_winner divergent force PASS
# ============================================================================

def test_y2_xgb_diverges_returns_true():
# ... 5 個 test functions（test_y2_*）
```

整個區塊刪除。

- [ ] **Step 3: 修 compute_kelly_block 呼叫 — 移除 ml_prediction kwarg（原 L223-227 / L272 / L333 / L374 / L418）**

```bash
grep -n "compute_kelly_block\|ml_prediction\|ml_pred =" scripts/tests/test_predict_snapshot.py
```

每個 `compute_kelly_block(args, merged, ml_pred, formula_pred, ...)` 改為 `compute_kelly_block(args, merged, formula_pred, ...)`；每個 `ml_prediction={"home_win_pct": ...}` kwarg 整行刪除。

範例（L223-229 區塊）：

old：
```python
        ml_pred = {"home_win_pct": 60.0}
        formula_pred = {"total": 9.5, "margin": 0.8}
        args = _make_args(merged_path)
        kelly_block = compute_kelly_block(
            args, merged, ml_pred, formula_pred,
            final_ml_rec="CHC", final_ou_rec="OVER", final_rl_rec="PASS",
        )
```

new：
```python
        formula_pred = {"total": 9.5, "margin": 0.8, "log5_pct": 60.0}
        args = _make_args(merged_path)
        kelly_block = compute_kelly_block(
            args, merged, formula_pred,
            final_ml_rec="CHC", final_ou_rec="OVER", final_rl_rec="PASS",
        )
```

> 注意：`formula_pred` 必須加上 `log5_pct` key（取代原 ml_pred 的 `home_win_pct` 來源），否則 ML Kelly 計算會因 model_p_home is None 而被跳過、test 失敗。

針對其他 4 處 kwarg 形式：

old：
```python
            ml_prediction={"home_win_pct": 60.0},
            formula_prediction={"total": 9.5, "margin": 0.8},
```

new：
```python
            formula_prediction={"total": 9.5, "margin": 0.8, "log5_pct": 60.0},
```

5 處每一處都按上述 pattern 修改（保留原 home_win_pct 數值，搬到 formula_prediction.log5_pct）。

- [ ] **Step 4: 跑 test 確認全綠**

```bash
cd $REPO && pytest scripts/tests/ -v 2>&1 | tail -50
```
Expected：全部 PASS（test_fetch_results 仍存在但 P2 才刪）。如果有 fail，根據 fail message 修正。常見 fail 模式：
- 「`should_force_ml_pass` not found」→ 檢查是否還有遺留 import
- 「`compute_kelly_block() got unexpected keyword argument 'ml_prediction'`」→ 還有未改的呼叫
- 「`model_p_home is None`」→ 該 test 的 formula_pred 缺 log5_pct key

### Task 6: P1.1 — 清 requirements.txt 的 ML 套件

**Files:**
- Modify: `scripts/requirements.txt`

- [ ] **Step 1: 改寫 requirements.txt（保留 P2 仍需要的 pybaseball / pytest）**

old (8 行)：
```
requests>=2.31.0
pandas>=2.1.0
numpy>=1.24.0
pybaseball>=2.2.0
xgboost>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
pytest>=7.0.0
```

new (3 行 — 注：pybaseball 在 P2 仍會留著，因 pitcher_stats.py 使用)：
```
requests>=2.31.0
pybaseball>=2.2.0
pytest>=7.0.0
```

> 解釋：`pandas` 是 pybaseball 隱性依賴會被自動裝；不在 mlb-game-analyzer 自己代碼直接用。`numpy` 已從 predict.py 移除；其他腳本（lineup_analyzer, pitcher_stats）若有用到，會透過 pybaseball 帶入。

驗證：
```bash
grep -n "import joblib\|import xgboost\|from xgboost\|from sklearn\|import sklearn" scripts/*.py
```
Expected：無輸出。

```bash
grep -n "import numpy\|import pandas" scripts/*.py
```
Expected：可能仍有（如 pitcher_stats.py），但這些靠 pybaseball 帶入，不需在 requirements.txt 單獨列。

### Task 7: P1.2 — 刪 _backtest_rl_relaxation.py

**Files:**
- Delete: `scripts/_backtest_rl_relaxation.py`

- [ ] **Step 1: 確認檔案存在且無被 import**

```bash
ls scripts/_backtest_rl_relaxation.py
grep -rn "_backtest_rl_relaxation\|backtest_rl_relaxation" scripts/ reference/ SKILL.md 2>/dev/null
```
Expected：檔案存在；grep 無其他 reference。

- [ ] **Step 2: 刪除**

```bash
git rm scripts/_backtest_rl_relaxation.py
```

### Task 8: P1.3 — 清 predict.py 過時註釋

**Files:**
- Modify: `scripts/predict.py`

- [ ] **Step 1: 改 OU-3 註釋（原 L1239 — 重新 grep 取最新行號）**

```bash
grep -n "防止 upload 套 default 3 星" scripts/predict.py
```

old：
```python
        # OU-3: 非 PASS 但 stars 未指定 → PASS（防止 upload 套 default 3 星）
```

new：
```python
        # OU-3: 非 PASS 但 stars 未指定 → PASS
```

> 說明：spec 5.3 的另一條（fetch_results.py L183 docstring 改寫）省略 — 該檔在 P2 整個刪除。

### Task 9: P1.4 — 去重 prediction.md RL 表

**Files:**
- Modify: `reference/prediction.md`

- [ ] **Step 1: 找到原 L82-90 的 P(margin ≥ 2 | win) 參考值表**

```bash
grep -n "P(margin ≥ 2" reference/prediction.md
grep -n "P(margin ≥ 2 \\\\| win)" reference/prediction.md
```
Expected：兩個 hit — L83（ML 星級章節的表）+ L240/L244（Kelly 章節 canonical 版）。

- [ ] **Step 2: 替換 L82-90 整個表**

old（L82-90，行數可能微移；以「**P(margin ≥ 2 \| win) 參考值**：」起始行為錨點）：
```markdown
**P(margin ≥ 2 \| win) 參考值**：

| 熱門方 ML | P(margin ≥ 2 \| win) |
|-----------|---------------------|
| -110~-130 | ~58-60% |
| -130~-170 | ~60-63% |
| -170~-220 | ~63-67% |
| -220+ | ~67-72% |

**Run Line -1.5 星級（區分主/客場）**：
```

new：
```markdown
**P(margin ≥ 2 \| win) 查表** → 見「Kelly Sizing & Unit Output」章節 §「P(margin ≥ 2 \| win) 查表」。

**Run Line -1.5 星級（區分主/客場）**：
```

驗證：
```bash
grep -c "熱門方 ML" reference/prediction.md
```
Expected：1（只剩 Kelly 章節 L244 那份 canonical）。

### Task 10: P1 — 完成驗證 + commit

- [ ] **Step 1: 跑完整 test suite**

```bash
cd $REPO && pytest scripts/tests/ -v 2>&1 | tail -30
```
Expected：全部 PASS。

- [ ] **Step 2: 跑 predict.py 對 2026-04-25 任一場（先檢查 merged.json 是否存在）**

```bash
ls analysis-data/2026-04-25/*/merged.json 2>/dev/null | head -5
```

如有 merged.json 存在，挑一場跑：
```bash
TARGET=$(ls analysis-data/2026-04-25/*/merged.json | head -1)
python scripts/predict.py --game-data "$TARGET" --test 2>&1 | tail -20
```
Expected：能跑完不報錯。如 2026-04-25 還沒有 merged.json，回 fallback：用 fixtures/sample_merged.json（test 中用的）做 smoke test：
```bash
python -c "import scripts.predict" 2>&1
```
Expected：無 import error。

- [ ] **Step 3: grep 確認 dead code 全清**

```bash
grep -rn "predict_with_ml\|should_force_ml_pass\|check_xgb_divergent\|xgb_raw_home_pct\|FEATURE_COLS\|MODELS_DIR\|WIN_MODEL_PATH" scripts/ 2>/dev/null
```
Expected：無輸出。

- [ ] **Step 4: 確認 reference/* 與 SKILL.md 未動（除 prediction.md RL 表）**

```bash
git diff --name-only HEAD reference/ SKILL.md
```
Expected：只列出 `reference/prediction.md`。

- [ ] **Step 5: Commit P1**

```bash
git add scripts/ reference/prediction.md
git status
```

```bash
git commit -m "$(cat <<'EOF'
refactor(mlb-skill): P1 純代碼清理 — 刪 XGBoost 路徑與一次性 backtest

- 刪 scripts/train.py / update_model.py / _backtest_rl_relaxation.py
- 從 predict.py 移除 import joblib/numpy、predict_with_ml/should_force_ml_pass/
  check_xgb_divergent 整 3 個函式、FEATURE_COLS、MODELS_DIR/WIN_MODEL_PATH
- 移除 cross_validation 欄位與 xgb_raw_home_pct（無 ML 後字串無語意）
- compute_kelly_block 改為從 formula_prediction.log5_pct 取 model_p_home
- 同步刪 test_predict_snapshot.py 中 should_force_ml_pass / Y2 test 區塊；
  其他 test 改用 formula_prediction.log5_pct
- requirements.txt 從 8 套件瘦身至 3 個
- prediction.md L82-90 RL 表去重，保留 Kelly 章節 canonical 版
- 清理 predict.py L1239 註釋（移除「防止 upload 套 default 3 星」）

對應 spec：docs/superpowers/specs/2026-04-26-mlb-skill-slimming-design.md §5

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: 驗證 commit 已建立**

```bash
git log -1 --stat
```
Expected：HEAD 是 P1 commit，列出修改檔案。

---

# Phase 2：post-game scripts 移除 + working-tree 已刪檔 commit

**目標：** 移除屬於 mlb-post-game-review skill 的 scripts、清理 working-tree 已刪未 commit 的雜檔、刪 analysis-logs/ 整目錄。
**完成判準：** `pytest scripts/tests/` 全綠 + `predict.py` 仍能跑 + `requirements.txt` 最終 3 個套件。
**估時：** 1-1.5 hr

### Task 11: P2.1 — 刪 4 個 post-game scripts + test

**Files:**
- Delete: `scripts/fetch_results.py`、`scripts/summarize_predictions.py`、`scripts/review_stats.py`、`scripts/diagnose_metrics.py`、`scripts/tests/test_fetch_results.py`

- [ ] **Step 1: grep 確認沒有其他檔（核心流程）import 它們**

```bash
grep -rn "from fetch_results\|import fetch_results\|from summarize_predictions\|import summarize_predictions\|from review_stats\|import review_stats\|from diagnose_metrics\|import diagnose_metrics" scripts/ 2>/dev/null
```
Expected：無輸出（或只匹配到 4 個被刪檔自身的 reference）。

- [ ] **Step 2: 確認 4 個 script + 1 個 test 存在**

```bash
ls scripts/fetch_results.py scripts/summarize_predictions.py scripts/review_stats.py scripts/diagnose_metrics.py scripts/tests/test_fetch_results.py
```
Expected：5 個檔案都列出。

- [ ] **Step 3: 刪除 5 個檔**

```bash
git rm scripts/fetch_results.py scripts/summarize_predictions.py scripts/review_stats.py scripts/diagnose_metrics.py scripts/tests/test_fetch_results.py
```

- [ ] **Step 4: 跑 test 確認沒有殘留 import**

```bash
cd $REPO && pytest scripts/tests/ -v 2>&1 | tail -30
```
Expected：全部 PASS（test_fetch_results 不在）。

### Task 12: P2.2 — 更新 workflow.md L328

**Files:**
- Modify: `reference/workflow.md`

- [ ] **Step 1: 找到目前 L328 對應行（行號可能因 P1 未動 workflow.md 而保持）**

```bash
grep -n "summarize_predictions.py / fetch_results.py / review_stats.py" reference/workflow.md
```

- [ ] **Step 2: 替換**

old：
```markdown
> **當日彙總與賽後回填**（`summarize_predictions.py` / `fetch_results.py` / `review_stats.py`）請交由 `mlb-post-game-review` skill 處理，不屬於本 skill 範圍。
```

new：
```markdown
> **當日彙總與賽後回填**請交由 `mlb-post-game-review` skill 處理，不屬於本 skill 範圍。
```

驗證：
```bash
grep -n "summarize_predictions\|fetch_results\|review_stats\|diagnose_metrics" reference/workflow.md
```
Expected：無輸出。

### Task 13: P2.3 — 更新 prediction.md「預測紀錄存放位置」段

**Files:**
- Modify: `reference/prediction.md`

- [ ] **Step 1: 找到 L296-302 的「預測紀錄存放位置」段**

```bash
grep -n "預測紀錄存放位置" reference/prediction.md
```

- [ ] **Step 2: 替換 L300-302 兩行**

old (L300-302)：
```markdown
- **Per-date summary（快取）**：`analysis-data/{YYYY-MM-DD}/predictions.jsonl`
  當日所有場次的 JSONL。由 `summarize_predictions.py --date {date}` 全量重建。**屬於 mlb-post-game-review skill**。
- **賽後回填**：`fetch_results.py --date {date}` 從 MLB Stats API 抓 Final 比分，寫 `actual_*` + `verified=true`，同時更新 per-date jsonl 與 per-game prediction.json。**屬於 mlb-post-game-review skill**。
```

new：
```markdown
- **Per-date summary（快取）**：`analysis-data/{YYYY-MM-DD}/predictions.jsonl`
  當日所有場次 JSONL，由 `mlb-post-game-review` skill 重建。
- **賽後回填**：`actual_*` / `verified=true` 由 `mlb-post-game-review` skill 回填。
```

驗證：
```bash
grep -n "summarize_predictions\|fetch_results" reference/prediction.md
```
Expected：無輸出。

### Task 14: P2.4 — 確認 requirements.txt 已是最終 3 個

**Files:**
- Read only: `scripts/requirements.txt`

- [ ] **Step 1: 確認檔案內容**

```bash
cat scripts/requirements.txt
```
Expected：
```
requests>=2.31.0
pybaseball>=2.2.0
pytest>=7.0.0
```

> 若已是 3 行，跳過 — P1 已處理。

### Task 15: P2.5 — Commit working-tree 已刪未 commit 的雜檔

**Files:**
- Stage already-deleted: `analysis-data/2025-04-24/MIN@TB/game_data.json`、`plans/2026-04-23-phase1-readability.md`、`setup_task.bat`、`setup_task.ps1`

- [ ] **Step 1: 確認這些檔在 git status 是 D（已刪未 commit）**

```bash
git status --short | grep "^ D\|^D " | grep -E "MIN@TB|2026-04-23-phase1-readability|setup_task"
```
Expected：4 行 `D` status。

- [ ] **Step 2: 用 git rm 把這 4 個 stage 起來**

```bash
git rm analysis-data/2025-04-24/MIN@TB/game_data.json
git rm plans/2026-04-23-phase1-readability.md
git rm setup_task.bat
git rm setup_task.ps1
```

> 注意：上一條 P2.1 task 已經 git rm 了 4 個 post-game scripts；這個 task 只 stage 雜檔。

### Task 16: P2.6 — 刪除 analysis-logs/ 整目錄

**Files:**
- Delete: `analysis-logs/`（整目錄，含所有檔）

- [ ] **Step 1: 列出將被刪除的內容**

```bash
ls analysis-logs/
```
Expected：列出 `cumulative.md` + 多個 `2026-04-XX.md`。

- [ ] **Step 2: 確認 git status — 有些檔是 untracked（從未 commit），有些是 tracked**

```bash
git ls-files analysis-logs/ | head -20
git status --short analysis-logs/ | head -20
```

- [ ] **Step 3: 刪除整個目錄（同時處理 tracked + untracked）**

```bash
# Tracked 檔用 git rm
git ls-files analysis-logs/ | xargs -r git rm
# Untracked 檔用 rm -r
rm -rf analysis-logs/
```

驗證：
```bash
ls analysis-logs/ 2>&1
```
Expected：報「No such file」。

```bash
git ls-files analysis-logs/ | wc -l
```
Expected：0。

### Task 17: P2 — 完成驗證 + commit

- [ ] **Step 1: 跑 test 確認還是綠**

```bash
cd $REPO && pytest scripts/tests/ -v 2>&1 | tail -30
```
Expected：全部 PASS（test_fetch_results 不在）。

- [ ] **Step 2: grep 全 skill 確認 4 個 script 名 + 已刪 reference 全清**

```bash
grep -rn "fetch_results\|summarize_predictions\|review_stats\|diagnose_metrics" scripts/ reference/ SKILL.md 2>/dev/null
```
Expected：無輸出。

- [ ] **Step 3: predict.py smoke test**

```bash
python -c "import sys; sys.path.insert(0, 'scripts'); import predict" 2>&1
```
Expected：無 import error。

- [ ] **Step 4: pip install 驗證**（可選 — 若有乾淨 venv 環境）

```bash
pip install --dry-run -r scripts/requirements.txt 2>&1 | tail -10
```
Expected：能 resolve（不一定要真的裝起來，dry-run 只看 dependency tree）。

- [ ] **Step 5: 看 commit 範圍預覽**

```bash
git status --short
git diff --stat --cached | tail -10
```
Expected：列出已 stage 的：4 post-game scripts + test + 4 雜檔 + analysis-logs/* + reference/workflow.md + reference/prediction.md。

- [ ] **Step 6: Commit P2**

```bash
git commit -m "$(cat <<'EOF'
refactor(mlb-skill): P2 移除 post-game scripts + 清雜檔

- 刪 scripts/fetch_results.py / summarize_predictions.py / review_stats.py /
  diagnose_metrics.py 與對應 test_fetch_results.py（屬 mlb-post-game-review skill）
- 刪 analysis-logs/ 整目錄（屬 mlb-post-game-review）
- 清 working-tree 已刪未 commit：MIN@TB game_data / plans/2026-04-23-phase1 /
  setup_task.bat / setup_task.ps1
- workflow.md L328 簡化「當日彙總與賽後回填」轉介語句
- prediction.md L300-302「預測紀錄存放位置」簡化

對應 spec：docs/superpowers/specs/2026-04-26-mlb-skill-slimming-design.md §6

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: 驗證 commit**

```bash
git log -2 --oneline
```
Expected：兩個最近的 commit 是 P1 + P2。

---

# Phase 3a：Park Factor 更新

**目標：** PF 數值對齊 2023-2025 3 年加權；資料結構從 hardcoded dict 抽到 JSON；加 HR PF 欄位但暫不啟用；舊球場名提供 alias 向後相容。
**完成判準：** `pytest scripts/tests/` 全綠 + `merge_game_data.py` 對 2026-04-25 任一場跑得出 PF + alias 解析正確。
**估時：** 1.5-2 hr

### Task 18: P3a.1 — 建立 scripts/data/park_factors.json

**Files:**
- Create: `scripts/data/park_factors.json`
- Create: `scripts/data/`（目錄）

- [ ] **Step 1: 確認目錄不存在**

```bash
ls scripts/data 2>&1
```
Expected：「No such file」（建立 JSON 時 mkdir 會 fail，所以先確認）。

- [ ] **Step 2: 建立 scripts/data/park_factors.json（完整內容見 spec 附錄 A）**

```bash
mkdir -p scripts/data
```

把以下內容完整寫入 `scripts/data/park_factors.json`（不省略，spec §附錄 A 全文）：

```json
{
  "_meta": {
    "baseline_period": "2023-2025 (3-year weighted)",
    "source": "Baseball Savant",
    "format_note": "runs_pf 100 = 聯盟平均；hr_pf 100 = 聯盟平均",
    "updated": "2026-04-26"
  },
  "park_factors": {
    "Coors Field":                       { "runs_pf": 131, "hr_pf": 111 },
    "Sutter Health Park":                { "runs_pf": 109, "hr_pf": 106 },
    "Target Field":                      { "runs_pf": 106, "hr_pf":  98 },
    "Kauffman Stadium":                  { "runs_pf": 106, "hr_pf":  91 },
    "Comerica Park":                     { "runs_pf": 106, "hr_pf": 105 },
    "loanDepot park":                    { "runs_pf": 106, "hr_pf":  94 },
    "Fenway Park":                       { "runs_pf": 104, "hr_pf":  85 },
    "Citizens Bank Park":                { "runs_pf": 104, "hr_pf": 116 },
    "Great American Ball Park":          { "runs_pf": 104, "hr_pf": 129 },
    "PNC Park":                          { "runs_pf": 102, "hr_pf":  83 },
    "Chase Field":                       { "runs_pf": 101, "hr_pf":  82 },
    "Angel Stadium":                     { "runs_pf": 101, "hr_pf": 105 },
    "Progressive Field":                 { "runs_pf": 101, "hr_pf":  91 },
    "Nationals Park":                    { "runs_pf": 100, "hr_pf":  97 },
    "George M. Steinbrenner Field":      { "runs_pf": 100, "hr_pf": 109 },
    "Rogers Centre":                     { "runs_pf":  99, "hr_pf": 104 },
    "Truist Park":                       { "runs_pf":  98, "hr_pf":  95 },
    "Busch Stadium":                     { "runs_pf":  98, "hr_pf":  87 },
    "Daikin Park":                       { "runs_pf":  98, "hr_pf": 102 },
    "UNIQLO Field at Dodger Stadium":    { "runs_pf":  98, "hr_pf": 121 },
    "American Family Field":             { "runs_pf":  97, "hr_pf": 111 },
    "Rate Field":                        { "runs_pf":  97, "hr_pf":  99 },
    "Citi Field":                        { "runs_pf":  96, "hr_pf": 107 },
    "Oriole Park at Camden Yards":       { "runs_pf":  96, "hr_pf": 107 },
    "Yankee Stadium":                    { "runs_pf":  96, "hr_pf": 112 },
    "Globe Life Field":                  { "runs_pf":  96, "hr_pf": 106 },
    "Petco Park":                        { "runs_pf":  95, "hr_pf": 107 },
    "Wrigley Field":                     { "runs_pf":  92, "hr_pf":  92 },
    "Oracle Park":                       { "runs_pf":  91, "hr_pf":  83 },
    "T-Mobile Park":                     { "runs_pf":  82, "hr_pf":  82 }
  },
  "_aliases": {
    "Tropicana Field": "George M. Steinbrenner Field",
    "Oakland Coliseum": "Sutter Health Park",
    "Minute Maid Park": "Daikin Park",
    "Dodger Stadium": "UNIQLO Field at Dodger Stadium",
    "Guaranteed Rate Field": "Rate Field",
    "Camden Yards": "Oriole Park at Camden Yards"
  }
}
```

- [ ] **Step 3: 驗證 JSON 合法**

```bash
python -c "import json; data = json.load(open('scripts/data/park_factors.json', encoding='utf-8')); print(len(data['park_factors']), 'parks'); print(len(data['_aliases']), 'aliases')"
```
Expected：`30 parks` + `6 aliases`。

### Task 19: P3a.2 — TDD: 寫 resolve_park_factor 失敗測試（RED）

**Files:**
- Modify: `scripts/tests/test_merge_game_data.py`（在檔尾加入新 test 區塊）

- [ ] **Step 1: 在 test_merge_game_data.py 檔尾加入 5 個 test**

加在檔案最後（第 120 行之後）：

```python


# ============================================================================
# 2026-04-26 — Park Factor JSON 化 + alias 解析（spec §7.1.2）
# ============================================================================

def test_resolve_park_factor_canonical_name():
    """正式球場名直接命中 JSON 表。"""
    from merge_game_data import resolve_park_factor
    assert resolve_park_factor("Coors Field") == 131.0
    assert resolve_park_factor("T-Mobile Park") == 82.0


def test_resolve_park_factor_alias_old_name():
    """舊球場名透過 alias 解析到新名 — 向後相容。"""
    from merge_game_data import resolve_park_factor
    # Tropicana → Steinbrenner（Rays 臨時主場）
    assert resolve_park_factor("Tropicana Field") == 100.0
    # Oakland Coliseum → Sutter Health Park
    assert resolve_park_factor("Oakland Coliseum") == 109.0
    # Minute Maid → Daikin
    assert resolve_park_factor("Minute Maid Park") == 98.0
    # Dodger Stadium → UNIQLO Field at Dodger Stadium
    assert resolve_park_factor("Dodger Stadium") == 98.0
    # Guaranteed Rate → Rate Field
    assert resolve_park_factor("Guaranteed Rate Field") == 97.0
    # Camden Yards → Oriole Park at Camden Yards
    assert resolve_park_factor("Camden Yards") == 96.0


def test_resolve_park_factor_unknown_returns_default():
    """未知球場名回傳 100.0（聯盟平均，安全 fallback）。"""
    from merge_game_data import resolve_park_factor
    assert resolve_park_factor("Nonexistent Stadium") == 100.0


def test_resolve_park_factor_none_returns_default():
    """None venue 回傳 100.0。"""
    from merge_game_data import resolve_park_factor
    assert resolve_park_factor(None) == 100.0


def test_resolve_park_factor_returns_float():
    """回傳型別必為 float（predict.py 後續做 PF / 100 浮點除法）。"""
    from merge_game_data import resolve_park_factor
    result = resolve_park_factor("Coors Field")
    assert isinstance(result, float)
```

- [ ] **Step 2: 跑新 test 確認 fail（resolve_park_factor 還沒改造）**

```bash
cd $REPO && pytest scripts/tests/test_merge_game_data.py -v -k "resolve_park_factor" 2>&1 | tail -30
```
Expected：

實際結果有兩種可能（依現況）：
- (a) 5 個 test 全 PASS（如果現有 `resolve_park_factor` L209-213 已能處理 hardcoded dict 的命中與 default，但 `Tropicana Field` 在舊 dict 是 96，而新 JSON 是 100 — 所以 `test_resolve_park_factor_alias_old_name` 會 FAIL）。
- (b) `test_resolve_park_factor_alias_old_name` FAIL（值不對：舊 96 vs 預期 100）+ `test_resolve_park_factor_returns_float` 可能 FAIL（current 是 float 但需確認）。

至少 1 個 test 必須 fail（otherwise refactor 沒測試意義）。如果全部 PASS，stop and check — 表示既有實作已正確，看是否需要這個 refactor。

### Task 20: P3a.2 — 改造 merge_game_data.py（GREEN）

**Files:**
- Modify: `scripts/merge_game_data.py`

- [ ] **Step 1: 把 hardcoded PARK_FACTORS dict（L19-50）改成 JSON 載入**

old (L14-50, 含 import 區與 PARK_FACTORS 定義)：
```python
import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# E2: 30 座球場 Park Factor 對照表（5 年回歸值，2024-2025 基準）
# 來源：FanGraphs Park Factors / ESPN Park Factors
# 100 = 聯盟平均，>100 = 打者友善，<100 = 投手友善
PARK_FACTORS = {
    "Coors Field": 115,
    # ... 30 entries ...
    "Petco Park": 95,
}
```

new：
```python
from pathlib import Path

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Park Factor 資料（2023-2025 3 年加權，Baseball Savant；HR PF 暫不啟用）
_PF_DATA_PATH = Path(__file__).parent / "data" / "park_factors.json"
_PF_DATA = json.loads(_PF_DATA_PATH.read_text(encoding="utf-8"))
PARK_FACTORS = _PF_DATA["park_factors"]
PARK_ALIASES = _PF_DATA["_aliases"]
```

> 注意：`from pathlib import Path` 加在 `import requests` 之前（保持 import 排序：標準庫 → 第三方）。

- [ ] **Step 2: 改寫 resolve_park_factor 函式（L209-213）**

old：
```python
def resolve_park_factor(venue_name: str | None) -> float:
    """以 venue_name 解析 PF；未知 venue 回傳 100.0（聯盟平均）。"""
    if venue_name and venue_name in PARK_FACTORS:
        return float(PARK_FACTORS[venue_name])
    return 100.0
```

new：
```python
def resolve_park_factor(venue_name: str | None) -> float:
    """以 venue_name 解析 runs PF（HR PF 暫不啟用）。

    舊球場名透過 _aliases 表解析到 canonical 新名（如 Tropicana → Steinbrenner）。
    未知 venue 回傳 100.0（聯盟平均，安全 fallback）。
    """
    if not venue_name:
        return 100.0
    canonical = PARK_ALIASES.get(venue_name, venue_name)
    entry = PARK_FACTORS.get(canonical)
    if entry:
        return float(entry["runs_pf"])
    return 100.0
```

- [ ] **Step 3: 跑 test 確認全綠**

```bash
cd $REPO && pytest scripts/tests/test_merge_game_data.py -v 2>&1 | tail -40
```
Expected：所有 test PASS（含原 14 個 + 新 5 個 = 19 個）。

- [ ] **Step 4: 跑全 test suite 確認沒 regression**

```bash
cd $REPO && pytest scripts/tests/ -v 2>&1 | tail -20
```
Expected：全部 PASS。

### Task 21: P3a.3 — 更新 matchup-factors.md Park Factor 章節

**Files:**
- Modify: `reference/matchup-factors.md`

- [ ] **Step 1: 找到原 L161-168 的 Park Factor 章節**

```bash
grep -n "## 球場 & 天氣\|### Park Factor\|^Coors Field 特殊" reference/matchup-factors.md
```

- [ ] **Step 2: 替換 L161-168 區塊（從「## 球場 & 天氣」之後第一個 `### Park Factor` 起到下一個 `### `）**

old (L161-168)：
```markdown
## 球場 & 天氣

### Park Factor
以 100 為聯盟平均。**修正公式**：預期得分 × (PF / 100)。使用 **5 年回歸 PF**（單季不可靠）。

**Coors Field 特殊**：4 月 PF = 112（非全年 128），5 月後恢復 128。
> 物理依據：4 月丹佛 ~50-60°F，空氣密度比夏季高 ~8-10%。

### 影響分析的賽制規則
```

new：
```markdown
## 球場 & 天氣

### Park Factor
資料源：`scripts/data/park_factors.json`（2023-2025 3 年加權，Baseball Savant）
- 修正公式：`E[R] × (PF / 100)`
- 解析：100 = 聯盟平均；> 100 打者友善；< 100 投手友善

**分裂型球場**（Runs PF 與 HR PF 反向，特別處理）：
- Kauffman Stadium：Runs 106 / HR 91 — 利安打與三壘打，壓制 HR
- PNC Park：Runs 102 / HR 83 — 利二三壘打，HR 嚴重壓制
- UNIQLO Field at Dodger Stadium：Runs 98 / HR 121 — 抑制總得分但加成 HR

**近期重大改造**（影響 PF 解讀）：
- Camden Yards 2025 季前左外野牆移近、降低 → 從投手友善 0.96 → 打者友善
- Progressive Field 2024 移除外野貨櫃 → 風洞效應，LHB HR +16%
- 臨時主場：Athletics（Sutter Health）/ Rays（Steinbrenner）— 樣本期短

> ⛔ Coors Field 4 月：物理上空氣密度比夏季高 ~8-10%，4 月 PF ≈ 112，5 月後恢復 131。

### 影響分析的賽制規則
```

驗證：
```bash
grep -n "Park Factor\|分裂型球場\|2023-2025\|Coors Field 4 月" reference/matchup-factors.md | head -10
```
Expected：能看到新章節錨點。

### Task 22: P3a — 完成驗證 + commit

- [ ] **Step 1: pytest 全綠**

```bash
cd $REPO && pytest scripts/tests/ -v 2>&1 | tail -20
```
Expected：所有 PASS。

- [ ] **Step 2: 對 2026-04-25 任一場跑 merge_game_data.py + 比對 PF**

```bash
ls analysis-data/2026-04-25/ | head -3
```

選一場（例：`BOS@BAL`）：
```bash
GAME_DIR=analysis-data/2026-04-25/BOS@BAL
ls $GAME_DIR/
```

如該場已有所需的中間 json（home_pitcher / away_pitcher / home_lineup / away_lineup / game_data），跑 merge：
```bash
python scripts/merge_game_data.py \
  --game $GAME_DIR/game_data.json \
  --home-pitcher $GAME_DIR/home_pitcher.json \
  --away-pitcher $GAME_DIR/away_pitcher.json \
  --home-lineup $GAME_DIR/home_lineup.json \
  --away-lineup $GAME_DIR/away_lineup.json \
  -o /tmp/merged_test.json 2>&1 | tail -5
python -c "import json; print('PF =', json.load(open('/tmp/merged_test.json'))['park_factor'])"
```
Expected：印出新 JSON 對應的 PF（例 BAL = Oriole Park = 96）。

如果該場缺中間 json，改用既有 merged.json 比對 PF 沒變動的部份：
```bash
python -c "
import json
old = json.load(open('$GAME_DIR/merged.json'))
print('venue=', old.get('_meta', {}).get('venue'))
print('old park_factor=', old['park_factor'])
"
```
比對該 venue 在新 JSON 中的值。如果該 venue 是 Camden Yards 之類舊值 96 → 新值 96（變動），記下作為文檔說明。

- [ ] **Step 3: 手動驗證 alias 解析**

```bash
python -c "
import sys; sys.path.insert(0, 'scripts')
from merge_game_data import resolve_park_factor
for old in ['Tropicana Field', 'Oakland Coliseum', 'Minute Maid Park', 'Dodger Stadium', 'Guaranteed Rate Field']:
    print(f'{old:30s} -> {resolve_park_factor(old)}')
"
```
Expected：每個舊名都解析到合理值（100 / 109 / 98 / 98 / 97）。

- [ ] **Step 4: Commit P3a**

```bash
git add scripts/data/park_factors.json scripts/merge_game_data.py scripts/tests/test_merge_game_data.py reference/matchup-factors.md
git status
```

```bash
git commit -m "$(cat <<'EOF'
refactor(mlb-skill): P3a Park Factor 對齊 2023-2025 3 年加權 + JSON 化

- 新增 scripts/data/park_factors.json：30 球場 runs_pf + hr_pf（HR 暫不啟用）+
  6 個舊球場名 alias（Tropicana/Oakland/Minute Maid/Dodger/Guaranteed Rate/Camden）
- merge_game_data.py 從 JSON 載入 PARK_FACTORS + PARK_ALIASES；
  resolve_park_factor 加 alias 解析；保留 100.0 fallback
- 加 5 個 test：canonical / alias / unknown / None / float type
- matchup-factors.md Park Factor 章節改寫：強調 JSON 路徑、分裂型球場、
  近期改造、Coors Field 4 月特殊值

數值大改：Coors 115 → 131；Camden 101 → 96；Yankee 104 → 96 等 — 預期行為

對應 spec：docs/superpowers/specs/2026-04-26-mlb-skill-slimming-design.md §7.1

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: 驗證 commit**

```bash
git log -3 --oneline
```
Expected：HEAD 是 P3a，往下 P2 / P1。

---

# Phase 3b：文件重整（just-in-time + cross-ref 去重）

**目標：** 執行 just-in-time 文件原則，刪 pitfalls.md 並把內容外散、精煉 flags-checklist.md、改寫 D1 紀律（XGBoost 已清）、最後 cross-ref grep 確認規則只在 canonical 處有完整版。
**完成判準：** `pitfalls.md` 不存在、`flags-checklist.md < 80 行`、`SKILL.md < 150 行 且 < 500 字`、grep threshold 出現次數 ≤ 2。
**估時：** 2-3 hr

### Task 23: P3b.2 — pitfalls.md 內容外散（odds-format.md + SKILL.md）

**Files:**
- Modify: `reference/odds-format.md`（加段）
- Modify: `SKILL.md`（加段）
- Delete: `reference/pitfalls.md`

- [ ] **Step 1: 在 odds-format.md 末尾加入「亞洲盤口格式歧義」段**

把以下內容加在 `odds-format.md` 最末（L52「兩者為獨立市場…」之後 `\n` 接續）：

```markdown

---

## 亞洲盤口格式歧義（必檢查）

亞洲格式的讓分符號（特別是 quarter handicap 如 `(1+50)`、`(1-50)`）與西方 American odds 直接對譯時容易誤判：

- 「客隊讓 1.5」≠「客隊 -1.5」— 亞洲 `(1+50)` = -0.5/-1.0 拆分；American `-1.5` = 整數 1.5
- 必須交叉驗證：用獨贏盤 ML（American）+ 投手分析推回讓分方向，再對譯回亞洲格式確認
- 若發生衝突（亞洲讓分方向 vs ML 熱門方不一致）→ 暫停推薦，向使用者確認盤口來源
```

驗證：
```bash
grep -n "亞洲盤口格式歧義" reference/odds-format.md
wc -l reference/odds-format.md
```
Expected：grep 命中 1 行；行數從 52 變約 65。

- [ ] **Step 2: 在 SKILL.md「語氣與風格」章節加「使用者質疑結果」段**

```bash
grep -n "## 語氣與風格\|承認不確定性" SKILL.md
```

old (L115-122 區塊「語氣與風格」整段)：
```markdown
## 語氣與風格

- 進階數據 > 傳統數據，兩者兼用
- 承認不確定性：MLB 單場隨機性約 40-45%
- 明確標注數據來源
- 修正係數必須基於可搜尋到的研究或數據
```

new（在末尾追加 1 個 bullet）：
```markdown
## 語氣與風格

- 進階數據 > 傳統數據，兩者兼用
- 承認不確定性：MLB 單場隨機性約 40-45%
- 明確標注數據來源
- 修正係數必須基於可搜尋到的研究或數據
- 使用者質疑結果時：回顧量化信號、獨立驗證後才決定是否修正；不直接妥協
```

驗證：
```bash
grep -n "使用者質疑結果" SKILL.md
```
Expected：1 行 hit。

- [ ] **Step 3: 刪 pitfalls.md**

```bash
git rm reference/pitfalls.md
```

驗證：
```bash
ls reference/pitfalls.md 2>&1
```
Expected：「No such file」。

- [ ] **Step 4: grep 全 skill 確認沒人 link 到 pitfalls.md**

```bash
grep -rn "pitfalls\.md\|reference/pitfalls" SKILL.md reference/ scripts/ 2>/dev/null
```

如有 hit（例：SKILL.md L111 `Edge Cases + 修正係數：reference/pitfalls.md` 或類似），刪除該行（這條 link 在「Common Pitfalls & Edge Cases」段，整個段在 P3b.1 SKILL.md 改寫時會處理 — 此處先刪 link）。

```bash
grep -n "Edge Cases + 修正係數\|Edge Cases.*pitfalls" SKILL.md
```

對應行整行刪除。

驗證：
```bash
grep -rn "pitfalls" SKILL.md reference/ scripts/ 2>/dev/null
```
Expected：無輸出。

### Task 24: P3b.1 — SKILL.md 刪「最高優先 3 項技術漏洞」段（與 flags-checklist 不重疊）

**Files:**
- Modify: `SKILL.md`

- [ ] **Step 1: 找到並刪除 L97-112 的「Common Pitfalls & Edge Cases」整段**

```bash
grep -n "## Common Pitfalls\|^→ 完整紀律 flag" SKILL.md
```

old (L97-112)：
```markdown
## Common Pitfalls & Edge Cases

最高優先 3 項技術漏洞（與 flags-checklist 不重疊；完整清單見下方連結）：

1. **Hot/Cold 判定未查 BABIP**
   近 7 天 BABIP 極端值（≤ .260 或 ≥ .370）預期回歸 ~.300，未檢查 = Hot/Cold 判定無效。

2. **ERA vs xERA 落差 ≥ 1.5 僅寫成「風險提示」**
   可驗證的現象不得掛成條件性風險。必須補跑 `pitcher_stats.py --year {YYYY-1}` + YoY Statcast 對比。

3. **Phase 3 summary 寫入「初步盤口推薦」或星級**
   盤口推薦 single source = Phase 4 `prediction.json`。Summary 只放基本面，避免 stale。

→ 完整紀律 flag（13 條）：`reference/flags-checklist.md`
```

new（簡化為 1 行 cross-ref，不重複規則內容）：
```markdown
## Common Pitfalls

紀律違規 13 條 + 觸發處理：見 `reference/flags-checklist.md`。
邊界條件（Coors 4 月、Doubleheader、TJ 復出等）：見 `reference/matchup-factors.md` 與 `prediction.md`。
```

> 說明：3 個技術漏洞的內容已分散於 matchup-factors.md（BABIP 回歸） / workflow.md（ERA-xERA 閘門） / workflow.md Phase 3.5（phase3_summary 規則），SKILL.md 不重複。

驗證：
```bash
grep -n "Hot/Cold 判定未查 BABIP\|ERA vs xERA 落差\|Phase 3 summary 寫入" SKILL.md
```
Expected：無輸出（規則散到 reference/* 而非 SKILL.md）。

### Task 25: P3b.3 — 精煉 flags-checklist.md（13 條 → 每條 2-3 行）

**Files:**
- Modify: `reference/flags-checklist.md`

- [ ] **Step 1: 整檔重寫（從 L1 到 EOF），用統一 2-3 行/條格式**

完整新內容（替換整個檔）：

```markdown
# 旗標清單（Flags Checklist）

> 13 條分析紀律硬規則。任一條觸發 = 停下來，回到對應 Phase 閘門。
> 每條僅列觸發條件 + cross-ref；規則完整內容在 canonical 檔。

---

### 1. 用訓練資料/記憶代替腳本 API 輸出
- 觸發：核心數據（ERA/xERA/IP、xwOBA/BABIP、牛棚 ERA）來源不是 `pitcher_stats.py` / `lineup_analyzer.py` / `fetch_game_data.py`
- 處理：腳本失敗 → 向使用者回報，禁止改走 WebSearch / 記憶。詳見 `workflow.md` 初始化「模式切換規範」

### 2. BvP 樣本 < 15 PA 硬推結論
- 觸發：BvP `PA < 15` 但仍寫成趨勢
- 處理：標註「樣本不足」，不引用。詳見 `matchup-factors.md` §BvP

### 3. Hot/Cold 判定未檢查 BABIP
- 觸發：近 7 天 BABIP `≤ .260` 或 `≥ .370`，未做回歸判定
- 處理：跳到 `matchup-factors.md` §BABIP 回歸檢查

### 4. 牛棚傷兵只修 O/U 未修 ML
- 觸發：核心（Closer / Primary Setup / High-leverage）IL 但 phase3_summary 缺 ML 修正 (-%) 或 OU 修正 (+run)
- 處理：B9 雙向閘門。詳見 `workflow.md` §Phase 3 §B9

### 5. 同場推對立方向
- 觸發：ML 推 A 隊 + A 隊受讓
- 處理：D3 硬規則。詳見 `prediction.md` §D3

### 6. 不寫 phase3_summary.md 就進 Phase 4
- 觸發：缺 `$GAME_DIR/phase3_summary.md` 但呼叫 `predict.py --save`
- 處理：predict.py 會 reject。詳見 `workflow.md` §Phase 3.5

### 7. 跳過 Roster 檢查
- 觸發：Phase 2 Step 1 未通過就進 Step 2
- 處理：阻塞閘門。詳見 `workflow.md` §Phase 2 Step 1

### 8. Agent 子代理跑 WebSearch / WebFetch
- 觸發：dispatch subagent 帶 WebSearch task
- 處理：必須在主對話跑。子代理只能跑純計算腳本

### 9. 省 --game-data 或腦補路徑
- 觸發：`predict.py` 缺 `--game-data` 或路徑不符 `analysis-data/<date>/<AWAY>@<HOME>/merged.json`
- 處理：predict.py 會 reject。詳見 `workflow.md` §Phase 4

### 10. shell redirect `>` 取代 --output / -o
- 觸發：腳本呼叫用 `>` 寫檔
- 處理：所有腳本必須用 `--output` / `-o`。詳見 `workflow.md` §模式切換規範

### 11. WebSearch 失敗繼續分析
- 觸發：WebSearch error 但仍輸出推薦
- 處理：回報錯誤等使用者指示，禁止「差不多就好」

### 12. 中文對話用英文輸出
- 觸發：使用者中文 → 報告卻是英文
- 處理：報告語言對齊使用者；搜尋可用英文

### 13. ERA-xERA 落差 ≥ 1.5 僅寫「風險提示」
- 觸發：`|ERA − xERA| ≥ 1.5` 或 `IP < 30 且 ERA 比 prior_year 低 ≥ 1.0`，但僅寫提示未補跑 YoY
- 處理：必須補跑 `pitcher_stats.py --year {YYYY-1}`。詳見 `workflow.md` §Phase 2 Step 2

---

## 使用方式

每條規則均可透過 Phase 閘門自檢。完整 Phase 順序見 `workflow.md`；觸發時的補救動作見對應 Phase section 與 cross-ref 檔。
```

- [ ] **Step 2: 確認行數 < 80**

```bash
wc -l reference/flags-checklist.md
```
Expected：~70 行（< 80 的 spec 完成條件）。

### Task 26: P3b.4 — 改寫 prediction.md D1 紀律 + 開頭 ML 註釋 + 內部 ml_prediction reference

**Files:**
- Modify: `reference/prediction.md`
- Modify: `reference/workflow.md`

- [ ] **Step 1: 改寫 prediction.md L5-7 開頭 ML/XGBoost 註釋**

```bash
grep -n "total_model（xgb_total_model.pkl）\|勝率使用 XGBoost\|ml_prediction.* 用於勝率" reference/prediction.md
```

old (L5-7)：
```markdown
> ⚠️ **total_model（xgb_total_model.pkl）訓練資料有結構性缺陷，比分預測不可靠。**
> 勝率使用 XGBoost win_model，比分使用 formula 公式計算。
> predict.py 已實作此邏輯：`ml_prediction` 用於勝率，`formula_prediction` 用於比分。
```

new：
```markdown
> 勝率與比分皆來自 `formula_prediction`（Log5 + 期望得分公式）。
> XGBoost 路徑於 2026-04 重構移除（spec 2026-04-26-mlb-skill-slimming-design）；
> 舊 `cross_validation` / `ml_prediction` / `xgb_raw_home_pct` 欄位不再產出。
```

- [ ] **Step 2: 改寫 D1 紀律（L129-137 區塊）**

```bash
grep -n "### D1：模型覆蓋紀律\|^### D2" reference/prediction.md
```

old (L129-137)：
```markdown
### D1：模型覆蓋紀律

ML (XGBoost) 與 Log5 (Formula) 方向一致時（即 `ml_lean == formula_lean`），**不得因軟性因素翻轉勝方**（Platoon 劣勢、連勝動能、H2H 等）。

- 可調整：勝率幅度 ±5%、信心降級、星級降級
- 可覆蓋：模型未計入的重大因素（先發臨時更換等）、用戶明確要求
- **不可覆蓋**：方向分歧（`ml_lean != formula_lean`）→ ML 強制 PASS
- **原則**：模型方向 > 直覺。軟性因素影響幅度，不影響方向。
- **實作**：`predict.py` 當場比對 `ml_lean` / `formula_lean`，不讀 `cross_validation` 字串（α 實作，見 spec 2026-04-22-mlb-skill-slimming-design.md §3.2）。`cross_validation` 欄位仍寫入（含 `INSUFFICIENT_SAMPLE` / `DIVERGENT` / `CONSISTENT` / `NO_ML_MODEL`）但僅供觀察。
```

new：
```markdown
### D1：模型輸出紀律

`formula_prediction.lean`（HOME 或 AWAY）為唯一決定方向的依據。

- 可調整：勝率幅度 ±5%、信心降級、星級降級
- 可覆蓋：模型未計入的重大因素（先發臨時更換等）、用戶明確要求
- 不可覆蓋：軟性因素（Platoon / 連勝動能 / H2H 等）影響強度，不影響方向
- ML 路徑（XGBoost）於 2026-04 重構移除，`cross_validation` 欄位不再產出

> 預測紀錄歷史檔仍含 `cross_validation` 欄位（pre-2026-04），僅供觀察，新預測不寫入。
```

- [ ] **Step 3: 修 D3 表頭欄位名（XGBoost → formula）**

```bash
grep -n "XGBoost home_win_pct" reference/prediction.md
```

old (L152)：
```markdown
| XGBoost home_win_pct | ML 推薦 | 受讓推薦 |
```

new：
```markdown
| formula home_win_pct | ML 推薦 | 受讓推薦 |
```

- [ ] **Step 4: 修 Kelly schema 表中的 ml_prediction.home_win_pct（L236）**

```bash
grep -n "ml_prediction.home_win_pct" reference/prediction.md
```

old (L236)：
```markdown
| ML | `ml_prediction.home_win_pct / 100`（XGBoost） | 不用 Log5，避免和 cross_validation 紀律打架 |
```

new：
```markdown
| ML | `formula_prediction.log5_pct / 100`（Log5） | 由 P1 重構統一 — XGBoost 已移除 |
```

- [ ] **Step 5: 移除 prediction.json schema 中的 cross_validation 欄位（L329 附近）**

```bash
grep -n "cross_validation" reference/prediction.md
```

對 L329 對應行（在 prediction.json schema 中）：

old：
```jsonc
  "cross_validation": "CONSISTENT/DIVERGENT/INSUFFICIENT_SAMPLE/NO_ML_MODEL",
```

new：（整行刪除）

驗證：
```bash
grep -n "ml_prediction\|cross_validation\|XGBoost\|xgb_" reference/prediction.md
```
Expected：無輸出（除可能存在的「2026-04 移除」歷史說明）。

- [ ] **Step 6: 修 workflow.md L294-295 ML 註釋**

```bash
grep -n "ml_prediction.home_win_pct\|XGBoost 模型" reference/workflow.md
```

old (L294-295)：
```markdown
> ⚠️ **勝率必須用 predict.py 的 `ml_prediction.home_win_pct`（XGBoost 模型）。**
> **比分使用 `formula_prediction`**。手動估算只能作為輔助驗算。
```

new：
```markdown
> ⚠️ **勝率與比分皆用 predict.py 的 `formula_prediction`**（XGBoost 路徑於 2026-04 重構移除）。手動估算只能作為輔助驗算。
```

驗證：
```bash
grep -rn "ml_prediction\|XGBoost\|xgb_\|cross_validation" SKILL.md reference/
```
Expected：無輸出（除歷史性「2026-04 移除」說明）。

### Task 27: P3b.5 — Cross-ref grep 重整最後掃描

**Files:**
- Read only: SKILL.md + reference/*.md

- [ ] **Step 1: 跑 spec §7.2.5 的 5 條 grep 命令**

```bash
echo "=== BABIP 閾值 ==="
grep -rn "\.260\|\.370" SKILL.md reference/ 2>/dev/null

echo "=== ERA-xERA 落差 ==="
grep -rn "ERA.*xERA\|≥ *1\.5\|>= *1\.5" SKILL.md reference/ 2>/dev/null

echo "=== IP 與 prior_year ERA delta 閾值 ==="
grep -rn "IP *< *30\|≥ *1\.0\|>= *1\.0" SKILL.md reference/ 2>/dev/null

echo "=== BvP PA 閾值 ==="
grep -rn "PA *≥ *15\|PA *>= *15\|< *15" SKILL.md reference/ 2>/dev/null

echo "=== O/U 噪音閾值 ==="
grep -rn "1\.5 *run\|< *1\.5" SKILL.md reference/ 2>/dev/null
```

- [ ] **Step 2: 對每條 threshold 確認在 SKILL.md + reference/* 中出現次數 ≤ 2**

針對每條 threshold（`.260` / `.370` / `1.5`（ERA-xERA / O/U noise）/ `30`（IP 閾值）/ `1.0`（prior_year ERA delta）/ `15`（BvP PA）），人工確認：
- 1 次 canonical 在 `matchup-factors.md`（規則完整版）
- 1 次 inline 在 觸發點（`workflow.md` Phase 閘門 / `flags-checklist.md` 對應條目）

> 若某條 threshold 出現 ≥ 3 次：找出第 3+ 次出現處，把該處改成 1 行 cross-ref（「詳見 `matchup-factors.md` §X」）。

範例修法：若 `.260` 在 SKILL.md L102 / workflow.md L217 / matchup-factors.md L74 / flags-checklist.md L20 都有完整描述（4 次 = 超過 2 次），保留：
- matchup-factors.md L74（canonical）
- flags-checklist.md L20（觸發點 inline，已是 cross-ref 格式 — 注意 P3b.3 重寫後新格式應該已經精簡）
- 把 SKILL.md L102 那段刪掉（已在 P3b.1 task 24 處理）
- 把 workflow.md L217 改為短 inline 提醒 + cross-ref

> 注意：P3b.1 SKILL.md 改寫已刪「最高優先 3 項」段，所以 .260/.370 在 SKILL.md 應該已經剩 0 次（或只有 quick-reference 表的縮寫）。確認此狀態。

- [ ] **Step 3: 確認 cross-ref anchors 都對得上**

```bash
echo "=== 引用 matchup-factors.md 的 anchor ==="
grep -rn "matchup-factors\.md#\|matchup-factors\.md §" SKILL.md reference/ 2>/dev/null | head -20

echo "=== matchup-factors.md 自己的章節 ==="
grep -n "^### \|^## " reference/matchup-factors.md
```

對每個 `matchup-factors.md#xxx` 或 `matchup-factors.md §xxx` 的 anchor，確認對應章節存在於 matchup-factors.md。常見錨點：
- `#babip-回歸檢查` ↔ `### BABIP 回歸檢查（必須執行）`
- `#yoy-statcast-驗證` ↔ `### YoY Statcast 驗證`

如有引用不存在的 anchor，找最接近的章節改成正確 anchor。

### Task 28: P3b — 完成驗證 + commit

- [ ] **Step 1: 行數驗證**

```bash
echo "SKILL.md 行數："
wc -l SKILL.md
echo "SKILL.md 字數："
wc -w SKILL.md

echo "flags-checklist.md 行數："
wc -l reference/flags-checklist.md
```
Expected：
- `SKILL.md` < 150 行 且 < 500 字
- `flags-checklist.md` < 80 行

如 SKILL.md 字數仍 ≥ 500，找冗長處再精煉（特別是 Phase 2/3/4 描述）。

- [ ] **Step 2: pitfalls.md 不存在**

```bash
ls reference/pitfalls.md 2>&1
```
Expected：「No such file」。

- [ ] **Step 3: pytest 全綠（純文件改動不影響 test，但跑一遍保險）**

```bash
cd $REPO && pytest scripts/tests/ -v 2>&1 | tail -10
```
Expected：全 PASS。

- [ ] **Step 4: dead reference grep**

```bash
grep -rn "predict_with_ml\|xgb_\|closing_line\|^clv\b\|upload\|cross_validation" SKILL.md reference/ scripts/ 2>/dev/null | grep -v "analysis-data/.*prediction.json" | grep -v ".pyc"
```
Expected：無輸出（除歷史性「2026-04 移除」說明）。

- [ ] **Step 5: Commit P3b**

```bash
git add SKILL.md reference/
git status
```

```bash
git commit -m "$(cat <<'EOF'
refactor(mlb-skill): P3b 文件重整 — pitfalls 刪除、flags 精煉、D1 改寫

- 刪 reference/pitfalls.md（內容外散）
  - 「亞洲盤口歧義」→ odds-format.md 末尾
  - 「使用者質疑結果」→ SKILL.md 語氣與風格章節
  - Edge Cases 表 / 修正係數已分散於 matchup-factors.md / prediction.md
- flags-checklist.md 13 條精煉為「2-3 行/條」（觸發 + 處理 cross-ref）；行數 56 → ~70
- SKILL.md 刪「Common Pitfalls 最高優先 3 項」段（與 flags-checklist 重疊）；
  改為 1 行 cross-ref；加「使用者質疑結果」bullet
- prediction.md D1 紀律改寫：移除 ml_lean vs formula_lean 比對，
  改為 formula 單模型；schema 移除 cross_validation 欄位；
  Kelly 表 ml_prediction.home_win_pct → formula_prediction.log5_pct
- workflow.md L294 同步：ml_prediction → formula_prediction
- cross-ref grep 重整：每條 threshold (.260/.370/1.5/30/1.0/15) 在
  SKILL.md+reference/* 出現次數 ≤ 2

對應 spec：docs/superpowers/specs/2026-04-26-mlb-skill-slimming-design.md §7.2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: 驗證 commit 與 baseline 對比**

```bash
git log -4 --oneline
echo "=== Skill 級瘦身驗證 ==="
echo "新 SKILL.md 行數 / 字數："
wc -l SKILL.md
wc -w SKILL.md
echo "新 reference/*.md total 行數："
wc -l reference/*.md | tail -1

echo "=== 對比 baseline（P1.0 Step 2 的記錄）==="
echo "baseline SKILL.md：123 行 / 482 字"
echo "baseline reference/*.md total：1125 行 / 5704 字"
```
Expected：
- SKILL.md ≥ 30% 字數縮減（482 字 × 0.7 = ~337 字以下）
- 全 skill 行數比改前少 ≥ 30%

如未達 30%，追加精煉（典型增益點：workflow.md 大段重複 Phase 描述）。

---

# Phase 3c：6 場景 TDD 壓力測試

**目標：** 驗證 P3a + P3b 改動後，紀律規則仍正確觸發。透過 git worktree 在 pre-P3 commit（即 P2 末尾的 commit SHA）跑 baseline，再在當前 HEAD 跑 post-change，比對 6 場景 AI 行為。
**完成判準：** 6 場景全部 baseline PASS + 全部 post-change PASS（spec 附錄 C 的 PASS 標準）。
**估時：** 1-2 hr

> **重要 TDD 原則：** baseline 必須對「改動前」狀態跑（即 P3a + P3b 還沒做的 git commit）。本 phase 用 `git worktree add` 在 P2 末尾 commit 額外開一個 working tree，dispatch subagent 在那個路徑分析；然後在當前 HEAD（P3b 末尾）跑 post-change。

### Task 29: P3c.1 — 建立 6 場景 fixture

**Files:**
- Create: `docs/superpowers/fixtures/p3c-scenarios/T1-babip-high.json`
- Create: `docs/superpowers/fixtures/p3c-scenarios/T2-babip-low.json`
- Create: `docs/superpowers/fixtures/p3c-scenarios/T3-era-xera-gap.json`
- Create: `docs/superpowers/fixtures/p3c-scenarios/T4-bullpen-il.json`
- Create: `docs/superpowers/fixtures/p3c-scenarios/T5-d3-opposite.json`
- Create: `docs/superpowers/fixtures/p3c-scenarios/T6-d5-noise.json`

- [ ] **Step 1: 建立 fixture 目錄**

```bash
mkdir -p docs/superpowers/fixtures/p3c-scenarios
```

- [ ] **Step 2: 為 6 場景建立模擬 game_data.json（每個是模擬完整 phase 1 輸出 + 部份 phase 2 輸出，讓 subagent 從 Phase 3 開始分析）**

> 設計理念：6 個 fixture 只需各自帶 spec §附錄 C 描述的關鍵欄位；其他欄位用合理 default。因為主要驗證的是「subagent 是否觸發紀律規則」，不是預測準確度。

寫入 `docs/superpowers/fixtures/p3c-scenarios/T1-babip-high.json`：

```json
{
  "_scenario": "T1 — BABIP 高極端",
  "_setup": "主隊 PHI 近 7 天 BABIP = .395，連勝 5 場（home_recent_ws = 5）",
  "_expected_behavior": [
    "Phase 3.4 偵測 BABIP 高極端 → 觸發 B10 TaskCreate",
    "phase3_summary.md 含 §BABIP 回歸判定",
    "不將 PHI 標為 Hot",
    "預測 Run Value 不加 +0.5 Hot 修正"
  ],
  "_pass_criteria": "subagent 輸出含「BABIP 回歸」+「不 Hot」字樣，phase3_summary.md 含 §BABIP 回歸判定",
  "game_data": {
    "_meta": {
      "game_pk": 999001,
      "game_date": "2026-04-30T23:05:00Z",
      "home_team": "Philadelphia Phillies",
      "away_team": "New York Mets",
      "venue": "Citizens Bank Park",
      "home_sp": "Aaron Nola",
      "away_sp": "Sean Manaea"
    },
    "park_factor": 104,
    "home_recent_rs": 5.4, "home_recent_ra": 4.1,
    "away_recent_rs": 4.2, "away_recent_ra": 4.5,
    "home_recent_ws": 5,
    "home_lineup": {"recent_babip": 0.395, "avg_babip": 0.305},
    "away_lineup": {"recent_babip": 0.298, "avg_babip": 0.295},
    "home_pitcher": {"era": 3.40, "xera": 3.50, "ip": 35.0, "fip": 3.30, "prior_year": {"era": 3.55}},
    "away_pitcher": {"era": 3.85, "xera": 3.80, "ip": 32.0, "fip": 3.70, "prior_year": {"era": 3.90}},
    "home_bullpen": {"era": 3.50, "core_il_count": 0},
    "away_bullpen": {"era": 3.65, "core_il_count": 0}
  }
}
```

寫入 `docs/superpowers/fixtures/p3c-scenarios/T2-babip-low.json`：

```json
{
  "_scenario": "T2 — BABIP 低極端",
  "_setup": "客隊 NYM lineup BABIP = .245，連敗 4 場",
  "_expected_behavior": [
    "Phase 3.4 偵測 BABIP 低極端 → 觸發 B10 TaskCreate",
    "phase3_summary.md 含 §BABIP 回歸判定",
    "不將 NYM 標為 Cold",
    "預測 Run Value 不扣 -0.5 Cold 修正"
  ],
  "_pass_criteria": "subagent 輸出含「BABIP 回歸」+「不 Cold」字樣，phase3_summary.md 含 §BABIP 回歸判定",
  "game_data": {
    "_meta": {
      "game_pk": 999002,
      "game_date": "2026-04-30T23:05:00Z",
      "home_team": "Philadelphia Phillies",
      "away_team": "New York Mets",
      "venue": "Citizens Bank Park",
      "home_sp": "Aaron Nola",
      "away_sp": "Sean Manaea"
    },
    "park_factor": 104,
    "home_recent_rs": 4.5, "home_recent_ra": 4.5,
    "away_recent_rs": 3.2, "away_recent_ra": 5.1,
    "away_recent_ls": 4,
    "home_lineup": {"recent_babip": 0.302, "avg_babip": 0.305},
    "away_lineup": {"recent_babip": 0.245, "avg_babip": 0.295},
    "home_pitcher": {"era": 3.40, "xera": 3.50, "ip": 35.0, "fip": 3.30, "prior_year": {"era": 3.55}},
    "away_pitcher": {"era": 3.85, "xera": 3.80, "ip": 32.0, "fip": 3.70, "prior_year": {"era": 3.90}},
    "home_bullpen": {"era": 3.50, "core_il_count": 0},
    "away_bullpen": {"era": 3.65, "core_il_count": 0}
  }
}
```

寫入 `docs/superpowers/fixtures/p3c-scenarios/T3-era-xera-gap.json`：

```json
{
  "_scenario": "T3 — ERA-xERA 落差",
  "_setup": "主隊投手 ERA 2.80 / xERA 4.50 (差 1.70)，IP 已過 30，prior_year ERA 3.50",
  "_expected_behavior": [
    "Phase 2 Step 2 閘門：偵測 |ERA-xERA| ≥ 1.5 → 必須補跑 pitcher_stats.py --year 2025",
    "TaskCreate B7（補跑 YoY 對比）",
    "phase3_summary.md §YoY 對比結論",
    "不通過閘門前不得進 Phase 3.5"
  ],
  "_pass_criteria": "subagent 必須 invoke pitcher_stats.py 帶 --year 2025，產出 home_pitcher_2025.json",
  "game_data": {
    "_meta": {
      "game_pk": 999003,
      "game_date": "2026-04-30T23:05:00Z",
      "home_team": "Philadelphia Phillies",
      "away_team": "New York Mets",
      "venue": "Citizens Bank Park",
      "home_sp": "Aaron Nola",
      "away_sp": "Sean Manaea"
    },
    "park_factor": 104,
    "home_recent_rs": 4.5, "home_recent_ra": 4.0,
    "away_recent_rs": 4.2, "away_recent_ra": 4.5,
    "home_lineup": {"recent_babip": 0.302, "avg_babip": 0.305},
    "away_lineup": {"recent_babip": 0.298, "avg_babip": 0.295},
    "home_pitcher": {"era": 2.80, "xera": 4.50, "ip": 38.0, "fip": 4.20, "era_xera_delta": -1.70, "prior_year": {"era": 3.50}},
    "away_pitcher": {"era": 3.85, "xera": 3.80, "ip": 32.0, "fip": 3.70, "prior_year": {"era": 3.90}},
    "home_bullpen": {"era": 3.50, "core_il_count": 0},
    "away_bullpen": {"era": 3.65, "core_il_count": 0}
  }
}
```

寫入 `docs/superpowers/fixtures/p3c-scenarios/T4-bullpen-il.json`：

```json
{
  "_scenario": "T4 — 牛棚雙向閘門",
  "_setup": "客隊 Closer + Setup IL（B9 觸發）",
  "_expected_behavior": [
    "同時計算 OU 修正 +0.5 run + ML 修正 -3%（該隊勝率下修）",
    "TaskCreate B9（牛棚雙向修正值）",
    "phase3_summary.md §牛棚雙向修正值"
  ],
  "_pass_criteria": "phase3_summary.md 出現「OU +」與「ML -%」雙方向修正值，缺一即 FAIL",
  "game_data": {
    "_meta": {
      "game_pk": 999004,
      "game_date": "2026-04-30T23:05:00Z",
      "home_team": "Philadelphia Phillies",
      "away_team": "New York Mets",
      "venue": "Citizens Bank Park",
      "home_sp": "Aaron Nola",
      "away_sp": "Sean Manaea"
    },
    "park_factor": 104,
    "home_recent_rs": 4.5, "home_recent_ra": 4.0,
    "away_recent_rs": 4.2, "away_recent_ra": 4.5,
    "home_lineup": {"recent_babip": 0.302, "avg_babip": 0.305},
    "away_lineup": {"recent_babip": 0.298, "avg_babip": 0.295},
    "home_pitcher": {"era": 3.40, "xera": 3.50, "ip": 35.0, "fip": 3.30, "prior_year": {"era": 3.55}},
    "away_pitcher": {"era": 3.85, "xera": 3.80, "ip": 32.0, "fip": 3.70, "prior_year": {"era": 3.90}},
    "home_bullpen": {"era": 3.50, "core_il_count": 0, "core_il_roles": []},
    "away_bullpen": {"era": 4.85, "core_il_count": 2, "core_il_roles": ["Closer", "Primary Setup"]}
  }
}
```

寫入 `docs/superpowers/fixtures/p3c-scenarios/T5-d3-opposite.json`：

```json
{
  "_scenario": "T5 — D3 對立方向",
  "_setup": "formula home_win_pct = 65%；Game = NYM @ PHI（主隊 PHI）",
  "_expected_behavior": [
    "ml_rec 為主隊縮寫 (PHI)，不得是字面值 'HOME'（predict.py 會 reject HOME 字面值）",
    "run_line_rec 為 PHI / PHI -1.5 / PASS 任一，不得為 NYM / NYM +1.5"
  ],
  "_pass_criteria": "subagent 產出的 prediction.json 中 ml_rec == 'PHI'（主隊 abbrev），且 run_line_rec ∉ {'NYM', 'NYM +1.5', 'AWAY +1.5'}",
  "game_data": {
    "_meta": {
      "game_pk": 999005,
      "game_date": "2026-04-30T23:05:00Z",
      "home_team": "Philadelphia Phillies",
      "away_team": "New York Mets",
      "venue": "Citizens Bank Park",
      "home_sp": "Aaron Nola",
      "away_sp": "Sean Manaea"
    },
    "park_factor": 104,
    "home_recent_rs": 5.2, "home_recent_ra": 3.8,
    "away_recent_rs": 3.9, "away_recent_ra": 4.6,
    "_expected_log5_pct": 65.0,
    "home_lineup": {"recent_babip": 0.302, "avg_babip": 0.305},
    "away_lineup": {"recent_babip": 0.298, "avg_babip": 0.295},
    "home_pitcher": {"era": 2.95, "xera": 3.10, "ip": 35.0, "fip": 3.05, "prior_year": {"era": 3.55}},
    "away_pitcher": {"era": 4.20, "xera": 4.10, "ip": 32.0, "fip": 4.05, "prior_year": {"era": 3.90}},
    "home_bullpen": {"era": 3.20, "core_il_count": 0},
    "away_bullpen": {"era": 4.10, "core_il_count": 0}
  }
}
```

寫入 `docs/superpowers/fixtures/p3c-scenarios/T6-d5-noise.json`：

```json
{
  "_scenario": "T6 — D5 比分一致性",
  "_setup": "formula adjusted_total = 8.2，OU line = 9.5（差距 1.3，< 1.5 噪音閾值）",
  "_expected_behavior": [
    "推 ou_rec: PASS（差距 < 1.5）",
    "不推 OVER（adjusted < line）"
  ],
  "_pass_criteria": "subagent 輸出 ou_rec = PASS",
  "game_data": {
    "_meta": {
      "game_pk": 999006,
      "game_date": "2026-04-30T23:05:00Z",
      "home_team": "Philadelphia Phillies",
      "away_team": "New York Mets",
      "venue": "Citizens Bank Park",
      "home_sp": "Aaron Nola",
      "away_sp": "Sean Manaea"
    },
    "park_factor": 104,
    "home_recent_rs": 4.0, "home_recent_ra": 4.2,
    "away_recent_rs": 4.0, "away_recent_ra": 4.2,
    "_expected_adjusted_total": 8.2,
    "_expected_ou_line": 9.5,
    "home_lineup": {"recent_babip": 0.302, "avg_babip": 0.305},
    "away_lineup": {"recent_babip": 0.298, "avg_babip": 0.295},
    "home_pitcher": {"era": 3.40, "xera": 3.50, "ip": 35.0, "fip": 3.30, "prior_year": {"era": 3.55}},
    "away_pitcher": {"era": 3.85, "xera": 3.80, "ip": 32.0, "fip": 3.70, "prior_year": {"era": 3.90}},
    "home_bullpen": {"era": 3.50, "core_il_count": 0},
    "away_bullpen": {"era": 3.65, "core_il_count": 0}
  }
}
```

- [ ] **Step 3: 驗證 6 個 fixture 都是合法 JSON**

```bash
for f in docs/superpowers/fixtures/p3c-scenarios/*.json; do
  python -c "import json; json.load(open('$f', encoding='utf-8'))" && echo "OK: $f" || echo "FAIL: $f"
done
```
Expected：6 個都 `OK`。

### Task 30: P3c.2 — RED：dispatch baseline subagent（pre-P3 state）

**Files:**
- Create: `docs/superpowers/baselines/2026-04-26-p3c-baseline.md`

- [ ] **Step 1: 找到 pre-P3 commit SHA（即 P2 末尾的 commit，也就是 P3a 之前的 HEAD）**

```bash
git log --oneline -7
```

從 `git log --oneline` 找出 P2 phase 的 commit（commit message 含 "P2 移除 post-game scripts"），複製 7-char SHA。本 task 在 P3b commit 之後執行，所以 HEAD = P3b、HEAD~1 = P3a、**HEAD~2 = P2（pre-P3 state）**：

```bash
PRE_P3_SHA=$(git rev-parse HEAD~2)
git show --no-patch --format="%h %s" $PRE_P3_SHA
```
Expected：印出的 commit subject 含 "P2 移除 post-game scripts"。如不匹配（例如中間有額外 commit），改用 `git log --oneline --grep="P2 移除 post-game"` 找出 SHA 並執行 `PRE_P3_SHA=<該 7-char SHA>`。

- [ ] **Step 2: 建立 git worktree 在 pre-P3 commit**

```bash
WORKTREE=/tmp/mlb-skill-baseline
git worktree add $WORKTREE $PRE_P3_SHA
ls $WORKTREE/SKILL.md $WORKTREE/reference/pitfalls.md
```
Expected：worktree 建立、`pitfalls.md` 仍存在（pre-P3b state）。

- [ ] **Step 3: Dispatch subagent 跑 6 場景在 baseline 路徑**

使用 Agent tool 呼叫，參數：
- `subagent_type`: `general-purpose`
- `description`: `P3c baseline run (pre-P3 worktree)`
- `prompt`: 以下完整文字（包含「---」分隔線之間所有內容）

```
你是 mlb-game-analyzer skill 的測試副代理。我會給你 6 個情境 fixture（pre-P3 重構版本的 SKILL.md + reference）。請你「假裝」針對每個情境跑 Phase 1-4 分析，依照 SKILL.md 與 reference/* 的規則，輸出你會做的判斷。

工作目錄：/tmp/mlb-skill-baseline （這是 pre-P3 commit 的 worktree）
Fixture 路徑（在當前 repo，不是 worktree）：C:\Users\USER\.agents\skills\mlb-game-analyzer\docs\superpowers\fixtures\p3c-scenarios\T*.json

對每個 T1 ~ T6 fixture：
1. Read /tmp/mlb-skill-baseline/SKILL.md 與 /tmp/mlb-skill-baseline/reference/*.md（特別是 workflow.md / matchup-factors.md / prediction.md / flags-checklist.md / pitfalls.md）
2. Read fixture 的 game_data，判斷 Phase 3 / Phase 4 應該採取什麼動作
3. 對每個 _expected_behavior 條目，判斷你「依規則會不會做」（YES / NO / UNCLEAR）
4. 對 _pass_criteria，判斷會 PASS / FAIL

輸出格式（per fixture）：
## T<N> — <情境名>
- [ ] _expected_behavior 條目 1：YES/NO/UNCLEAR — 簡短理由
- [ ] _expected_behavior 條目 2：YES/NO/UNCLEAR — 簡短理由
- ...
- **PASS/FAIL** for _pass_criteria

最後加一段「整體 baseline 判定」：6 場景中幾個 PASS / FAIL。

不要實際呼叫 Python 腳本，純粹依規則文件做判斷。報告聚焦「規則會不會被觸發」而非「預測值是多少」。
```

把 subagent 完整輸出存到 `docs/superpowers/baselines/2026-04-26-p3c-baseline.md`（先 mkdir）：

```bash
mkdir -p docs/superpowers/baselines
# subagent 輸出貼進去（透過 Write tool）
```

- [ ] **Step 4: 確認 baseline 6 個全 PASS**

讀 baseline 報告。如有任何 FAIL，stop and investigate：
- 若 FAIL 是 fixture 設計問題（例如數值未觸發閾值），改 fixture
- 若 FAIL 是 pre-P3 規則本身有問題（規則散得太亂導致 subagent 漏看）→ 這是 P3 重構的動機，但 baseline 不通過時無法當 regression 基準。記下問題，調整 fixture 讓 pre-P3 能 PASS

baseline 必須 6 個全 PASS 才繼續到 Step 5。

- [ ] **Step 5: 移除 worktree（baseline 已記錄到檔案，不再需要 worktree）**

```bash
git worktree remove $WORKTREE
git worktree list
```
Expected：worktree list 不再有 /tmp/mlb-skill-baseline。

### Task 31: P3c.3 — GREEN：dispatch post-change subagent（current HEAD）

**Files:**
- Create: `docs/superpowers/baselines/2026-04-26-p3c-postchange.md`

- [ ] **Step 1: Dispatch 同樣 prompt 到 subagent，但工作目錄改成當前 repo（post-P3 state）**

使用 Agent tool 呼叫，參數：
- `subagent_type`: `general-purpose`
- `description`: `P3c post-change run (current HEAD)`
- `prompt`: 以下完整文字

```
你是 mlb-game-analyzer skill 的測試副代理。我會給你 6 個情境 fixture（post-P3 重構版本的 SKILL.md + reference）。請你「假裝」針對每個情境跑 Phase 1-4 分析，依照 SKILL.md 與 reference/* 的規則，輸出你會做的判斷。

工作目錄：C:\Users\USER\.agents\skills\mlb-game-analyzer （當前 HEAD，已包含 P3a + P3b 改動）
Fixture 路徑：C:\Users\USER\.agents\skills\mlb-game-analyzer\docs\superpowers\fixtures\p3c-scenarios\T*.json

對每個 T1 ~ T6 fixture：
1. Read SKILL.md 與 reference/*.md（特別是 workflow.md / matchup-factors.md / prediction.md / flags-checklist.md；注意 pitfalls.md 已在 P3b 刪除）
2. Read fixture 的 game_data，判斷 Phase 3 / Phase 4 應該採取什麼動作
3. 對每個 _expected_behavior 條目，判斷你「依規則會不會做」（YES / NO / UNCLEAR）
4. 對 _pass_criteria，判斷會 PASS / FAIL

輸出格式（per fixture）：
## T<N> — <情境名>
- [ ] _expected_behavior 條目 1：YES/NO/UNCLEAR — 簡短理由
- [ ] _expected_behavior 條目 2：YES/NO/UNCLEAR — 簡短理由
- ...
- **PASS/FAIL** for _pass_criteria

最後加一段「整體 post-change 判定」：6 場景中幾個 PASS / FAIL。

不要實際呼叫 Python 腳本，純粹依規則文件做判斷。
```

把輸出存到 `docs/superpowers/baselines/2026-04-26-p3c-postchange.md`（透過 Write tool）。

- [ ] **Step 2: 比對 baseline vs post-change**

```bash
diff docs/superpowers/baselines/2026-04-26-p3c-baseline.md docs/superpowers/baselines/2026-04-26-p3c-postchange.md
```

理想：兩份報告對 6 場景的 PASS/FAIL 判定一致（都 6 個 PASS）。

如有差異：
- baseline PASS 但 post-change FAIL：表示 P3 重構漏掉某條規則 → REFACTOR step（Task 32）
- baseline FAIL 但 post-change PASS：表示 fixture 設計問題或重構修復了 baseline 的漏洞，記錄在 commit message
- 兩邊都 FAIL：fixture 本身有問題

### Task 32: P3c.4 — REFACTOR（如有 FAIL 場景）

**Files:**
- Conditionally modify: `reference/workflow.md` 或其他被 grep 出的「漏點」檔

- [ ] **Step 1: 對每個 post-change FAIL 的場景，找出哪個規則沒被 subagent 看到**

讀 subagent 在該場景的解釋，判斷：
- 規則完全沒提到 → 把規則 inline 提醒（1-2 行）加在最相關的觸發點檔（通常是 workflow.md 對應 Phase）
- 規則有提到但太分散 → 在 inline 處加更明確的「⛔」記號或 cross-ref

- [ ] **Step 2: 修改對應 reference 檔**

範例：若 T1 BABIP 高極端 FAIL（subagent 沒提到 B10 BABIP 回歸閘門），檢查 workflow.md L215-224 是否還在（B10 段落）。如已被 P3b 精簡掉，restore 並加更明顯的 inline 提醒：

```markdown
**B10 BABIP 回歸閘門（必檢查）：**

⛔ Phase 3.4 進行 Hot/Cold 判定前，**必須**檢查雙方 lineup 近 7 天 BABIP：
- BABIP ≤ .260 或 ≥ .370 → 立即 TaskCreate B10：「BABIP 回歸判定（{team} 近 7 天 {value}）」
- 結論寫入 `phase3_summary.md` §BABIP 回歸判定
- 詳見 `matchup-factors.md` §BABIP 回歸檢查
```

- [ ] **Step 3: 重新 dispatch post-change subagent，confirm 該場景 PASS**

重複 Task 31 Step 1（dispatch + 比對），直到全 6 場景 PASS。

> 如果 REFACTOR 反覆 3 次仍 FAIL：暫停，回頭看 fixture 設計是否合理 / 該規則是否本就難以觸發。可能要把該場景標 KNOWN_ISSUE 並另開 spec 解決。

### Task 33: P3c — 完成驗證 + commit

- [ ] **Step 1: 確認 6 場景全 PASS**

```bash
grep -c "PASS" docs/superpowers/baselines/2026-04-26-p3c-postchange.md
grep -c "FAIL" docs/superpowers/baselines/2026-04-26-p3c-postchange.md
```
Expected：6 個 `PASS`、0 個 `FAIL`（或符合 _pass_criteria 標準的數量）。

- [ ] **Step 2: pytest 全綠（保險）**

```bash
cd $REPO && pytest scripts/tests/ -v 2>&1 | tail -10
```

- [ ] **Step 3: Commit P3c**

```bash
git add docs/superpowers/fixtures/ docs/superpowers/baselines/
# 如果有 REFACTOR 修了 reference/*：
git add reference/
git status
```

```bash
git commit -m "$(cat <<'EOF'
test(mlb-skill): P3c 6 場景壓力測試 — baseline + post-change 驗證

新增測試資產：
- docs/superpowers/fixtures/p3c-scenarios/T1-T6.json（6 場景模擬 game_data）
- docs/superpowers/baselines/2026-04-26-p3c-baseline.md（pre-P3 worktree 跑）
- docs/superpowers/baselines/2026-04-26-p3c-postchange.md（current HEAD 跑）

6 場景：T1 BABIP 高 / T2 BABIP 低 / T3 ERA-xERA 落差 /
T4 牛棚雙向 / T5 D3 對立方向 / T6 D5 比分一致性

baseline (pre-P3 commit via git worktree)：6/6 PASS
post-change (current HEAD)：6/6 PASS — 紀律規則精煉後仍正確觸發

[若有 REFACTOR：補上「補強 reference/<file>: <說明>」]

對應 spec：docs/superpowers/specs/2026-04-26-mlb-skill-slimming-design.md §7.3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: 驗證 5 個 phase commits 全到位**

```bash
git log --oneline -6
```
Expected：HEAD 是 P3c，往下 P3b / P3a / P2 / P1 / 之前的 spec commit。

---

# Final 驗證（spec §8 整體）

> 5 phase commit 完成後跑這份最後驗證。如有失敗，回到對應 phase 修補（不要 squash 既有 commit）。

### Task 34: Skill 級驗證

- [ ] **Step 1: pytest 全綠**

```bash
cd $REPO && pytest scripts/tests/ -v 2>&1 | tail -20
```
Expected：全 PASS。

- [ ] **Step 2: predict.py 對 2026-04-25 任一場跑得通**

```bash
TARGET=$(ls analysis-data/2026-04-25/*/merged.json 2>/dev/null | head -1)
if [ -n "$TARGET" ]; then
  python scripts/predict.py --game-data "$TARGET" --test 2>&1 | tail -10
else
  echo "2026-04-25 無 merged.json，改用 fixture smoke test"
  python -c "import sys; sys.path.insert(0, 'scripts'); import predict" 2>&1
fi
```
Expected：能跑完不報錯。

- [ ] **Step 3: 行數 / 字數比對 baseline**

```bash
echo "=== 對比 baseline (P1.0 Step 2) ==="
echo "baseline SKILL.md：123 行 / 482 字"
echo "  current：$(wc -l SKILL.md | awk '{print $1}') 行 / $(wc -w SKILL.md | awk '{print $1}') 字"
echo "baseline reference/*.md total：1125 行 / 5704 字"
echo "  current：$(wc -l reference/*.md | tail -1 | awk '{print $1}') 行 / $(wc -w reference/*.md | tail -1 | awk '{print $1}') 字"
echo "baseline scripts/*.py total：7178 行"
echo "  current：$(wc -l scripts/*.py | tail -1 | awk '{print $1}') 行"
```
Expected：
- SKILL.md 字數比改前少 ≥ 30%（≤ ~337 字）
- 全 skill 行數（SKILL.md + reference/*.md + scripts/*.py）比改前少 ≥ 30%
- SKILL.md 字數絕對值 < 500

- [ ] **Step 4: dead code 殘留 grep**

```bash
grep -rn "predict_with_ml\|xgb_\|closing_line\|^clv\b\|cross_validation" scripts/ reference/ SKILL.md 2>/dev/null | grep -v "analysis-data/.*prediction.json" | grep -v ".pyc"
```
Expected：無輸出（或只留歷史性「2026-04 移除」說明）。

- [ ] **Step 5: 文件級驗證**

```bash
echo "pitfalls.md 不存在：" && ls reference/pitfalls.md 2>&1
echo "刪除腳本不存在：" && ls scripts/train.py scripts/update_model.py scripts/_backtest_rl_relaxation.py scripts/fetch_results.py scripts/summarize_predictions.py scripts/review_stats.py scripts/diagnose_metrics.py 2>&1
echo "analysis-logs/ 不存在：" && ls analysis-logs/ 2>&1
echo "park_factors.json 存在：" && ls scripts/data/park_factors.json
echo "flags-checklist.md 行數：" && wc -l reference/flags-checklist.md
echo "SKILL.md 行數：" && wc -l SKILL.md
echo "requirements.txt 內容：" && cat scripts/requirements.txt
```
Expected：
- `pitfalls.md` / 7 個刪除腳本 / `analysis-logs/`：全部「No such file」
- `scripts/data/park_factors.json` 存在
- `flags-checklist.md` < 80 行
- `SKILL.md` < 150 行
- `requirements.txt` 只剩 3 個套件

### Task 35: 完成 — 推送或停在本地

> 5 phase 全部 commit 已建立。下一步由用戶決定：

選項 A：本地停留（不推送），用戶 review 後再決定 PR
```bash
git log --oneline -7
git status
```

選項 B：推上 remote 開 PR
```bash
git push -u origin refactor/skill-slimming
gh pr create --title "refactor(mlb-skill): 瘦身重構 — 5 phase（XGBoost 清除 / scripts 搬家 / PF 對齊 / 文件去重 / TDD 壓測）" --body "$(cat <<'EOF'
## Summary

按 docs/superpowers/specs/2026-04-26-mlb-skill-slimming-design.md 執行的 5 phase 重構：

- **P1**：刪 XGBoost 路徑（train.py / update_model.py / predict_with_ml / cross_validation 欄位 / 5 個 ML 套件）
- **P2**：移除 4 個 post-game scripts（屬 mlb-post-game-review skill）+ analysis-logs/ + 雜檔清理
- **P3a**：Park Factor 從 hardcoded dict 抽到 JSON、對齊 2023-2025 3 年加權、加 6 個球場 alias
- **P3b**：刪 pitfalls.md（內容外散）、flags-checklist 13 條精煉、改寫 D1 紀律、cross-ref 去重
- **P3c**：6 場景壓力測試（baseline via git worktree → post-change），驗證紀律規則精煉後仍生效

## Test plan

- [ ] `pytest scripts/tests/` 全綠
- [ ] SKILL.md < 500 字（writing-skills「Other skills」規範）
- [ ] grep 找不到 predict_with_ml / xgb_ / cross_validation dead reference
- [ ] P3c 6 場景全 PASS（baseline + post-change）

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

> 推送與開 PR 是 user-facing 動作，建議由用戶手動執行（plan 不主動推）。

---

## 後續工作（非本 plan 範圍）

完成本重構後，可獨立規劃：

1. fetch_odds + 盤口追蹤系統（D9）
2. 大數據 Park Factor（LHB/RHB 拆分、單月 PF、HR PF 啟用）
3. mlb-post-game-review skill 整理（在另一台電腦）

詳見 spec §11。
