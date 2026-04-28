# Phase 4 `prediction_summary.md` 設計

**日期**: 2026-04-27
**Skill**: mlb-game-analyzer
**範圍**: Phase 4 預測輸出流程的 context 精簡（B2 — 取代 JSON + output-format.md）

---

## 1. 背景與目標

### 問題

Phase 4 執行 `predict.py --save` 後，Claude 為了組最終報告需讀 **5 個檔案**：

- 上游 3 個 summary：`game_data_summary.md`（Phase 1）/ `merged_summary.md`（Phase 2）/ `phase3_summary.md`（Phase 3）— 已 ship
- `prediction.json`（59-104 行 JSON，視 signal/RL 狀態而定）
- `reference/output-format.md`（30 行模板）

實證觀察（2026-04-26 LAA@KC 跑場）：

- prediction.json 是 schema 化資料，AI 要做欄位 → 散文轉換才能寫 TL;DR + Section 8-10 的「推薦速查表 / 比分 / 勝率 / 推薦結果」
- output-format.md 模板與 JSON 資料源分裂：predict.py 改 schema 兩邊要同步
- AI 同時讀兩份檔，浪費 token 又增加忘填欄位、漂移風險

### 目標

Phase 4 階段，Claude 讀**自己 phase 的單一 summary md** + 上游 3 個 summary（共 3 樣，比目前 5 樣少 2 樣），不再讀 JSON、不再讀 output-format.md。Summary 內含 ready-to-paste TL;DR + Section 8-10；Section 1-7（基本面）由 AI 從上游 summary 補。

### 範圍邊界

| 範圍內 | 範圍外 |
|--------|--------|
| 改 `scripts/predict.py` 額外輸出 summary | 改 prediction.json schema |
| 改 `SKILL.md` / `reference/workflow.md` Phase 4 SOP | 改 `phase3_summary.md` 硬擋 grep 邏輯（spec 2 範圍） |
| **刪除 `reference/output-format.md`** | 改 `merge_game_data.py` / 上游 summary |
| 擴充 `scripts/tests/test_predict.py` | 改 predict.py 翻轉時重算 `predicted_home_pct`（顯示兩個欄位 + 註明翻轉即可） |

---

## 2. Summary 檔案規格

### 路徑與命名

- 與 `prediction.json` 同目錄：`analysis-data/{date}/{AWAY}@{HOME}/prediction_summary.md`
- 固定檔名 `prediction_summary.md`
- **僅在 `--save` 時輸出**（與 Phase 1 「`--output` 未指定不產 summary」對稱；dry-run 模式仍純 stdout）

### 內容結構（範例：LAA @ KC, 2026-04-26）

```markdown
# Prediction Summary — LAA @ KC (2026-04-26)

## TL;DR
- 預測比分: **KC 3.1 − 2.6 LAA**（HOME 勝，勝率 51.9%）
- 比賽走勢: <!-- narrative: AI 依 reference/prediction.md「比賽敘事觸發條件」選 1-2 句填入 -->

📊 推薦速查:

| 市場 | 方向 | 推薦指數 | 一句話理由 |
|------|------|----------|-----------|
| ML | KC | ⭐⭐ | Log5 51.9%（HOME），audit `home-2star-risk` |
| O/U | UNDER | ⭐⭐ | adj_total 5.7 vs line 9.0，差距 3.3 run |
| Run Line | PASS | — | \|diff\|=0.5 < 1.5（RL_DIFF_MIN） |

---

## 比分預測
- Formula 比分: KC 3.1 / LAA 2.6（總分 5.7）
- O/U gap: |adj_total 5.7 − line 9.0| = 3.3

## 勝率預測
- Formula log5: **51.9% (HOME)**

## 信號修正表

### Auto signals
| 信號 | ±run |
|------|------|
| Park Factor 106.0（修正 +0.30） | +0.30 |
| **總和** | **+0.30** |

### User-supplied signals
（無）

## 推薦結果
- **ML**: **KC ⭐⭐** — Log5 51.9%，audit tag `home-2star-risk`
- **O/U**: **UNDER ⭐⭐** — adj_total 5.7 vs line 9.0，差距 3.3 run
- **Run Line**: **PASS** — RL override 未啟動（|diff|=0.5 < RL_DIFF_MIN=1.5）

## 紀律檢查 (D1-D5)
- ✅ D1 模型方向：predicted_winner=HOME 與 ml_rec=KC（HOME）一致
- ✅ D2 信號量化：所有信號已轉為 run value
- ✅ D3 同場無對立推薦
- ✅ D5 比分盤口一致：adj_total 5.7 ≤ ou_line 9.0 vs ou_rec=UNDER

## 趨勢標記
- `home-hot-offense`、`home-pitching-slump`、`home-bullpen-slump`、`away-bullpen-slump`
```

### Section 對應的資料來源

| Section | 來源欄位 | 計算 |
|---------|---------|------|
| TL;DR 比分 / 勝方 / 勝率 | `predicted_home_score` / `predicted_away_score` / `predicted_winner` / `predicted_home_pct` | 直接帶 |
| TL;DR 推薦速查表 | `ml_rec` / `ml_stars` / `ou_rec` / `ou_stars` / `run_line_rec` / `run_line_stars` / `tags` / `rl_override` | `format_recommendation_rows` |
| 比分預測 | `formula_home_score` / `formula_away_score` / `predicted_home_score`（=adjusted）/ `ou_line` / `adjusted_total` | 純渲染 + `\|adj_total − line\|` 計算 gap |
| 勝率預測 | `predicted_home_pct` / `predicted_winner` / `formula_home_score` vs `formula_away_score` / `predicted_home_score` vs `predicted_away_score` | `_format_pct_with_flip` 偵測 flip |
| 信號修正表 — Auto | `compute_signal_table(data)` 結果（不在 record 中，main 算完傳入） | `format_signal_table_md` |
| 信號修正表 — User | `signal_adjustments`（dict） | `format_signal_table_md` |
| 推薦結果 | 同 TL;DR 推薦速查表 + cap_reasons | `format_recommendation_rows` 完整版 |
| 紀律檢查 | 多欄位推導（見下方規則） | `format_discipline_check` |
| Run Line override 細節（soft） | `rl_override` | `format_rl_override_block`（active=false → None） |
| 環境補充（soft） | `temperature_f` / `wind_mph` / `wind_direction` / `umpire_name` / `umpire_ou_rate` | `format_env_block`（全 null → None） |
| 趨勢標記（soft） | `tags` − 已折進推薦行的 tags | `format_trend_tags_block`（剩下空 → None） |

### 球隊縮寫規則

沿用 `predict.py` 既有 `TEAM_ABBREV` 字典（30 隊全名 → 縮寫）。`format_prediction_summary_md` 內部從 `record["home_team"]` / `record["away_team"]` 反查。Fallback：未知隊名 → 取前 3 字大寫。

### 紀律檢查規則（D1/D2/D3/D5）

| 規則 | 條件（✅） | 觸發（⚠️） |
|------|-----------|------------|
| D1 模型方向 | `predicted_winner` 對應的縮寫 == `ml_rec` | direction-override（`tags` 含 `direction-override`）或 `ml_rec=PASS` |
| D2 信號量化 | 永遠 ✅（predict.py 只接受 run value 形式信號） | — |
| D3 同場無對立 | `ml_rec` 與 `predicted_winner` 一致；或 `ml_rec=PASS`；或 `run_line_rec=PASS` | `ml_rec` 推某隊 + `run_line_rec` 推「對方」受讓 |
| D5 比分盤口一致 | `adj_total > ou_line` ↔ `ou_rec=OVER`；`adj_total < ou_line` ↔ `ou_rec=UNDER`；`ou_rec=PASS` 永遠 ✅ | 信號方向與比分矛盾（predict.py guardrail 已自動降為 PASS，正常不會出現） |

D4 已棄用，不顯示。

### 推薦行「一句話理由」 + tag 折進規則

| 市場 | 一句話理由內容 |
|------|--------------|
| ML | `Log5 X%（HOME/AWAY）` + 影響 cap 的 tag 折入（`divergent` / `direction-override` / `home-2star-risk` 出現時，註明於同行） |
| O/U | `adj_total X vs line Y，差距 Z run` |
| Run Line | active：`override_path（big-diff / mid-diff+strong-tag），\|diff\|=X，stars=Y`；inactive：`PASS — RL override 未啟動（\|diff\|=X < RL_DIFF_MIN=1.5）` |

折進推薦的 tag 集合（從 `format_trend_tags_block` 排除）：
- `divergent`、`direction-override`、`home-2star-risk`（這 3 個固定折進 ML 行）
- `rl_override.tags`（active 時折進 RL 行）

剩餘 tag（如 `*-hot-offense`、`*-pitching-slump` / `-hot`、`*-bullpen-slump` / `-strong`）→ 趨勢標記 section。

---

## 3. 邊界條件處理（混合模式）

### Hard sections（必須出現，缺值降級顯示）

- TL;DR
- 比分預測
- 勝率預測
- 信號修正表（兩段空 → 顯示「（無）」）
- 推薦結果
- 紀律檢查（D1-D5）

缺值時使用 `—` 取代數值；不寫「資料不足」字樣。

### Soft sections（不適用就整段省略）

- Run Line override 細節
- 環境補充
- 趨勢標記

### 逐個邊界處理

| 情境 | 處理 |
|------|------|
| `signal_adjustments == {}` | 「User-supplied signals」mini-table 顯示「（無）」 |
| `compute_signal_table` 無 signal 命中 | 「Auto signals」mini-table 顯示「（無）」 |
| `adjusted_home / away` 皆未傳 | 比分預測只列「Formula 比分」一行；不列「Adjusted 比分」 |
| `ml_stars == original_ml_stars` | 推薦結果 ML 行不顯示「原始 stars / 降級原因」 |
| Adjusted 比分翻轉方向（formula 預測 HOME 但 adj_home < adj_away） | 勝率預測加 ⚠️ 註明「Formula X% (HOME) → adjusted 比分 N1 < N2 判 AWAY 勝（pct 未隨翻轉重算）」 |
| `rl_override.active == false` | Run Line override 細節 section 整段省略；TL;DR 與推薦結果 RL 行仍正常列 PASS + 理由 |
| 全 PASS（ml/ou/rl 皆 PASS）| 推薦速查表 / 推薦結果照常列出，每行方向欄顯示 `PASS` |
| 環境欄位皆 null | 環境補充 section 整段省略 |
| 趨勢標記折進後為空（所有 tag 都已折入推薦行） | 趨勢標記 section 整段省略 |

### Fail-fast 條件

直接 raise（不寫 summary、不靜默降級）：

- `record` 缺 `home_team` 或 `away_team` 或 `predicted_winner`

寫檔 IOError → 沿用 Phase 1 模式：stderr warning + 不阻斷 prediction.json 寫入。

---

## 4. 實作項目

### 4.1 `scripts/predict.py` 變更

新增純函式（內聯，對齊 Phase 1 `fetch_game_data.py` 範式）：

| 函式 | 簽章 | 職責 |
|------|------|------|
| `_format_pct_with_flip` | `(formula_pct: float, predicted_winner: str, adj_home: float, adj_away: float, has_adjusted: bool) → str` | 渲染勝率行；翻轉時加 ⚠️ 註明 |
| `format_signal_table_md` | `(auto_signals: list[dict], user_signals: dict) → str` | Auto + user 兩個 mini-table，各自空 → 「（無）」 |
| `format_recommendation_rows` | `(record: dict, cap_reasons: list[str]) → tuple[str, str]` | 回傳 `(tldr_table_md, full_rows_md)`；同來源避免漂移 |
| `format_discipline_check` | `(record: dict) → str` | D1/D2/D3/D5 4 行（D4 已棄用） |
| `format_rl_override_block` | `(rl_override: dict) → str \| None` | active=false → None |
| `format_env_block` | `(record: dict) → str \| None` | 全 null → None |
| `format_trend_tags_block` | `(tags: list, recommendation_tags: set) → str \| None` | 扣除已折進推薦的，剩下空 → None |
| `format_prediction_summary_md` | `(record: dict, signal_table: dict, cap_reasons: list[str]) → str` | Assembler |

修改 `main()`：

- 寫完 `prediction.json` 後呼叫 `format_prediction_summary_md(record, signal_table, cap_reasons)`
- 寫至 `Path(prediction_path).parent / "prediction_summary.md"`
- stderr 輸出 `Saved summary to <path>`
- ValueError → stderr warning 不阻斷 JSON 寫入（沿用 Phase 1）

### 4.2 `scripts/tests/test_predict.py`（擴充既有檔）

新增 ~25-30 個 tests：

| Test 類別 | 案例 |
|----------|------|
| `test_format_pct_with_flip_*` | 不翻轉 / formula HOME→adj AWAY flip / formula AWAY→adj HOME flip / `has_adjusted=False` |
| `test_format_signal_table_md_*` | 兩段都有值 / auto 空 / user 空 / 兩段都空（皆「（無）」） |
| `test_format_recommendation_rows_*` | 全 PASS / ML cap 觸發 + cap_reasons 折進 / RL override active 折進 / `direction-override` tag |
| `test_format_discipline_check_*` | 全 ✅ / D1 direction-override 觸發 / D5 矛盾（理論上 predict.py guardrail 後不會出現，但測 fail-safe） |
| `test_format_rl_override_block_*` | active=true（big-diff）/ active=true（mid-diff+strong-tag）/ active=false → None |
| `test_format_env_block_*` | 全有值 / 部分 null / 全 null → None |
| `test_format_trend_tags_block_*` | 純 trend tags / 全部折進 → None / 混合（過濾後剩 1-2 個） |
| `test_format_prediction_summary_md_*` | smoke full / 全 PASS / RL override active / adjusted-flip / 全 soft section 省略 |
| `test_format_prediction_summary_md_raises_on_missing_*` | 缺 home_team / away_team / predicted_winner |

### 4.3 `SKILL.md` / `reference/workflow.md`

`SKILL.md` 變更：

- Quick Reference 表 Phase 4 行的「主要產出」由 `prediction.json + 報告` 改為 `prediction.json + prediction_summary.md`

`workflow.md` 變更：

- Phase 4.0 後加：「腳本同時輸出 `prediction_summary.md` 至同目錄（含 ready-to-paste TL;DR + Section 8-10）」
- Phase 4.7 開頭加：「✅ Read `$GAME_DIR/prediction_summary.md`；確認 `## 紀律檢查` section 全 ✅；ℹ️ 一般情況下無需 Read `prediction.json`」
- Phase 4.7 閘門加：`[ ] prediction_summary.md 已輸出`
- Phase 4.8 重寫：「完整 TL;DR + Section 8-10 模板已內化於 `prediction_summary.md`，AI 直接複製貼上；Section 1-7（基本面）由 AI 從 `game_data_summary.md` / `merged_summary.md` / `phase3_summary.md` 補。」（**移除 `output-format.md` cross-ref**）

### 4.4 刪除 `reference/output-format.md`

整個檔案 `git rm`。

`grep -rn "output-format.md" .` 應只剩歷史 spec / 舊 plan 中的提及（不主動回填）。

### 4.5 不動的部分

- `prediction.json` schema（下游 `mlb-post-game-review` 依賴）
- `phase3_summary.md` 硬擋 grep 邏輯（spec 2 範圍）
- 歷史 `analysis-data/` 目錄（不回填）
- `merge_game_data.py` / 其他 reference 檔

---

## 5. 預期收益

| 指標 | 改動前 | 改動後 |
|------|-------|-------|
| Phase 4 AI 讀取數 | 5 樣（含 JSON + output-format.md） | 3 樣（含上游 3 個 summary） |
| Phase 4 額外 token 占用 | JSON 600-1200 + output-format 250 ≈ 850-1450 | summary 700-1000 |
| 淨省 | — | ~150-450 tokens |
| TL;DR 渲染漂移風險 | AI 讀模板填欄位（易漂） | predict.py 渲染（一致） |
| 模板與資料源 | 分裂（output-format.md vs JSON） | 同源（predict.py 同一處渲染） |
| 與 Phase 1 範式對齊 | — | ✅ |

---

## 6. 後續

實作計畫由 `superpowers:writing-plans` skill 產出於 `docs/superpowers/plans/2026-04-27-prediction-summary-md-implementation.md`。

後續另有 spec 2 (`predict_phase4_efficiency`) 處理 `phase3_summary.md` grep 放寬 + dry-run 流程簡化（獨立 spec，本 spec 不涵蓋）。
