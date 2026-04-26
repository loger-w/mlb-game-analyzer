# MLB Game Analyzer Skill 瘦身重構 — Design Spec

| 欄位 | 內容 |
|---|---|
| Date | 2026-04-26 |
| Branch | `refactor/skill-slimming` |
| Status | Approved（待實作） |
| 預估時程 | 7-11 hr，分 3 phase |
| 上游 brainstorm | 本檔由 superpowers:brainstorming 流程產出 |

---

## 1. Background & Motivation

mlb-game-analyzer skill 累積至今包含：

- 14 個 .py script（其中 4 個確認無使用、3 個屬於另一 skill）
- 8 個 reference markdown 檔（多處紀律規則重複出現於 4-5 個檔案）
- 已刪除的後端 API 路徑（commit `520e281`）尚有過時註釋殘留
- XGBoost 勝率模型路徑：`predict_with_ml()` 仍在 import 與呼叫，但 `xgb_win_model.pkl` 不存在 → 全程 fallback formula，cross_validation 永遠輸出 `NO_ML_MODEL`
- Park Factor 數值已過時（基準為 5 年回歸 / FanGraphs 舊版），且 3 個球場改名（Tropicana → Steinbrenner、Oakland → Sutter、Minute Maid → Daikin）

**重構動機**：
1. 降低 context 壓力（瘦身 ~30-40% reference 行數）
2. 清除死碼，避免讀者誤以為 ML 路徑生效
3. 文件遵循 just-in-time 原則，符合 `superpowers:writing-skills` 的 token-efficiency 規範
4. Park Factor 對齊 2023-2025 3 年加權數據

---

## 2. 目標 / 非目標

### 目標

- 刪除確認無使用 / 屬另一 skill 的 script
- 文件去重，每條紀律規則只在 canonical 位置詳述
- 透過 6 場景壓力測試驗證紀律精煉後仍生效
- Park Factor 資料抽出至 JSON 並對齊 3 年加權新數值
- 後續可獨立執行 P1 / P2 / P3 三階段，每階段完成可獨立 commit

### 非目標

- ❌ 復活 XGBoost ML 路徑（決議 D1 = 全清光）
- ❌ 重建 fetch_odds + 盤口追蹤系統（決議 D9，留待後續另開設計）
- ❌ 修改 formula / signals / guardrails / Kelly 邏輯
- ❌ 新增任何功能（HR PF 加進 JSON 但暫不啟用）
- ❌ 重構 mlb-post-game-review skill（它在另一台電腦，本 phase 不負責）

---

## 3. 決議摘要

| # | 議題 | 決議 |
|---|---|---|
| **D1** | 勝率 XGBoost 路徑 | 全清光（train.py / update_model.py / `predict_with_ml()` / `joblib`/`numpy` ML 用法 / `xgb_raw_home_pct` / `cross_validation` ML 邏輯） |
| **D2** | 屬於 mlb-post-game-review 的 scripts | 從本 skill 刪除（mlb-post-game-review skill 在另一台電腦，由用戶自行同步） |
| **D3** | `_backtest_rl_relaxation.py` | 刪除（一次性 backtest，結果記在 analysis-logs） |
| **D4** | `predict.py:1239` / `fetch_results.py:183` 過時註釋 | 清理 |
| **D5** | `pitfalls.md` | 刪除，內容外散到 odds-format / SKILL.md |
| **D6** | `flags-checklist.md` | 精煉為「2-3 行/條」索引格式 |
| **D7** | `prediction.md` 內部 RL 表（出現 2 次） | 去重，保留 Kelly 章節版本 |
| **D8** | 多處規則重複（BABIP / ERA-xERA 等） | Just-in-time 重整：matchup-factors.md 升為 canonical 來源；其他檔規則處改 1-2 行 inline + cross-ref |
| **D9** | `fetch_odds.py` & 盤口追蹤系統 | 保持現狀，留待後續另開設計（用戶未來會做 smart money 流向追蹤） |
| **D10** | 執行順序 | B 三階段（P1 代碼清理 → P2 scripts 搬家 → P3 文件重整 + PF 更新 + 壓力測試） |

---

## 4. 架構決策

### 4.1 Just-in-Time 文件原則

依 `superpowers:writing-skills` 的 token-efficiency 規範：

| 類別 | 字數上限 |
|---|---|
| getting-started workflows | < 150 字 |
| Frequently-loaded skills（每對話自動載入） | < 200 字 |
| **Other skills（user-invoked，本 skill 屬此類）** | **< 500 字** |

**mlb-game-analyzer 屬「Other skills」**：使用者明確要求分析比賽時才載入，不是常駐。所以 `SKILL.md` 字數目標 < 500（當前 ~482 字，臨界）。

實作策略：**規則本身（含具體 threshold）只在 canonical 檔詳述，觸發點（workflow.md / SKILL.md / flags-checklist.md）只放 1-2 行 inline 提醒 + 跨檔 ref**。

### 4.2 三階段執行（P1 → P2 → P3）

| Phase | 性質 | 風險 | 測試 |
|---|---|---|---|
| **P1** | 純代碼清理（無紀律改動） | 低 | unit test 全綠 |
| **P2** | post-game scripts 移除 + 已刪檔 commit | 中 | unit test 全綠 + 手動跑 predict.py |
| **P3** | PF 更新 + 文件重整 + 壓力測試 | 中-高 | TDD 6 場景 |

每階段獨立 commit，可獨立 review / 回滾。

### 4.3 TDD 壓力測試（P3 限定）

依 `superpowers:writing-skills` Iron Law：

> NO SKILL WITHOUT A FAILING TEST FIRST. This applies to NEW skills AND EDITS to existing skills.

P3 改文件影響紀律規則，必須走 RED / GREEN / REFACTOR：

1. **RED**：改前用 subagent 跑壓力場景，紀錄 baseline 表現
2. **GREEN**：改後用同樣場景跑，verify AI 仍正確抓到規則
3. **REFACTOR**：失敗的場景補強提醒位置

P1 / P2 是純代碼變動，由 unit test 把關，不需走 TDD。

---

## 5. Phase 1：純代碼清理

**Phase 目標**：清掉死碼，不動 reference 文件，不動紀律規則。

### 5.0 P1.0 — 記錄 baseline metrics（先做）

P1 動工前記錄三個基準值，供 Section 8.2 的「瘦身 30%」驗收使用：

```bash
wc -l SKILL.md > /tmp/baseline_skill_md_lines.txt
wc -l reference/*.md > /tmp/baseline_reference_lines.txt
wc -l scripts/*.py > /tmp/baseline_scripts_lines.txt
wc -w SKILL.md reference/*.md > /tmp/baseline_words.txt
```

把 baseline 數字（總行數 / 總字數）寫進本 spec 的 「11.5 baseline」段（用 commit 記錄），重構結束後對比。

### 5.1 P1.1 — 刪除 XGBoost 路徑

#### 刪除檔案
- `scripts/train.py`
- `scripts/update_model.py`

#### 改動 `scripts/predict.py`
- L14：刪 `import joblib`
- L15：刪 `import numpy as np`（驗證僅 L618 使用）
- L22-23：刪 `MODELS_DIR` / `WIN_MODEL_PATH`
- L611-626：刪 `predict_with_ml()` 整個函式
- 主流程呼叫 `predict_with_ml()` 處：移除呼叫，改僅走 formula
- 移除產出 `xgb_raw_home_pct` 欄位的程式碼
- 移除 `cross_validation` 欄位本身（CONSISTENT / DIVERGENT / INSUFFICIENT_SAMPLE / NO_ML_MODEL 全 4 種狀態都是 ML 比對語意，無 ML 後整個欄位失去意義 — 連同 `predict.py` 中產出此欄位的邏輯一起刪除；BvP < 15 PA 之類的 sample 警示由 workflow / matchup-factors 既有規則承擔）
- D1 紀律 α 實作（`ml_lean vs formula_lean` 比對）：移除比對，改僅依 formula_lean 決定方向（reference/prediction.md 文件改寫延到 P3b）

#### 改動 `scripts/requirements.txt`
- 刪：`numpy`、`pandas`、`joblib`、`xgboost`、`scikit-learn`
- 保留：`requests`、`pybaseball`、`pytest`

### 5.2 P1.2 — 刪除 `_backtest_rl_relaxation.py`

直接刪檔。歷史結果保留在 `analysis-logs/2026-04-20.md`（雖然 P2 會刪 analysis-logs，但 backtest 結果在 git history 仍可查）。

### 5.3 P1.3 — 清過時註釋

| 檔 | 行 | 動作 |
|---|---|---|
| `scripts/predict.py` | L1239 | 改成 `# OU-3: 非 PASS 但 stars 未指定 → PASS`，刪掉 `防止 upload 套 default 3 星` |
| `scripts/fetch_results.py` | L183 | docstring 改成 `為已驗證紀錄補上 ml_result / ou_result / run_line_result`，刪掉 `不含 CLV`（注：fetch_results 在 P2 會被刪除，本項可省略） |

### 5.4 P1.4 — 去重 `prediction.md` RL 表

`reference/prediction.md` L82-90（ML 星級章節的 P(margin≥2|win) 表）：
- 改成 1 行：`P(margin ≥ 2 | win) 查表 → 見「Kelly Sizing & Unit Output」章節`
- 保留 L244-249 那份（含 Source 註解）為 canonical 版本

### 5.5 P1 完成條件

- [ ] `pytest scripts/tests/` 全綠（特別是 `test_predict_snapshot.py`）
- [ ] `python scripts/predict.py --game-data <merged.json> --save` 能跑完並產出 prediction.json
- [ ] 新產出 prediction.json 不含 `xgb_raw_home_pct` / `cross_validation: NO_ML_MODEL` 欄位
- [ ] `python -c "import scripts.predict"` 無 import error
- [ ] git diff 不動 reference/*.md（除了 prediction.md 的 RL 表去重）+ 不動 SKILL.md

### 5.6 P1 估時：1-2 hr

---

## 6. Phase 2：post-game scripts 移除

**Phase 目標**：把屬於 `mlb-post-game-review` skill 的 scripts 從本 skill 刪除（mlb-post-game-review 在另一台電腦，本 phase 不負責建立或同步）。

### 6.1 P2.1 — 刪除 5 個檔

```
scripts/fetch_results.py             ← 屬 mlb-post-game-review
scripts/summarize_predictions.py     ← 屬 mlb-post-game-review
scripts/review_stats.py              ← 屬 mlb-post-game-review
scripts/diagnose_metrics.py          ← 分析診斷工具，跟著 review_stats 走
scripts/tests/test_fetch_results.py  ← 對應 test
```

### 6.2 P2.2 — 更新 `reference/workflow.md`

| 行 | 動作 |
|---|---|
| L96 `Final → 詢問使用者是否要改用 mlb-post-game-review` | 不動 |
| L328 `當日彙總與賽後回填（summarize_predictions.py / fetch_results.py / review_stats.py）請交由 mlb-post-game-review skill 處理...` | 改成 `當日彙總與賽後回填請交由 \`mlb-post-game-review\` skill 處理，不屬於本 skill 範圍。` |

### 6.3 P2.3 — 更新 `reference/prediction.md`

「預測紀錄存放位置」段（L296-302）：

| 項目 | 動作 |
|---|---|
| Per-game | 不動 |
| Per-date summary L300-301 | 簡化為 `當日所有場次 JSONL 由 \`mlb-post-game-review\` skill 重建。` |
| 賽後回填 L302 | 簡化為 `賽後 actual_* / verified=true 由 \`mlb-post-game-review\` skill 回填。` |

### 6.4 P2.4 — 清理 `requirements.txt`

P1+P2 完成後，最終 requirements.txt 應為：
```
requests>=2.31.0
pybaseball>=2.2.0
pytest>=7.0.0
```

### 6.5 P2.5 — Commit working-tree 已刪檔

git status 顯示這 4 檔已在 working tree 刪除但未 commit：

- `analysis-data/2025-04-24/MIN@TB/game_data.json`（孤立 2025 年資料）
- `plans/2026-04-23-phase1-readability.md`
- `setup_task.bat`
- `setup_task.ps1`

P2 commit 一次清掉。

### 6.6 P2.6 — 刪除 `analysis-logs/` 整個目錄

```
analysis-logs/cumulative.md
analysis-logs/2026-04-19.md ~ 2026-04-24.md
```

> 此目錄是 mlb-post-game-review 寫入的回測日誌與結構問題追蹤，跟 4 個 script 一起搬家到另一個 skill。本 skill 不再保留。

### 6.7 P2.7 — `SKILL.md` 不動

`SKILL.md:25` / `:89` 的 `mlb-post-game-review` cross-ref 指向另一個 skill，本身正確，不需改。

### 6.8 P2 完成條件

- [ ] `pytest scripts/tests/` 全綠（test_fetch_results 已不存在）
- [ ] grep 全 skill：`fetch_results / summarize_predictions / review_stats / diagnose_metrics` 無 hits（除 git history）
- [ ] `git ls-files` 無 4 個刪除 script + analysis-logs/ 目錄
- [ ] `pip install -r scripts/requirements.txt` 在乾淨環境裝得起來
- [ ] `python scripts/predict.py --game-data <merged.json> --save` 仍能跑

### 6.9 P2 估時：1-1.5 hr

---

## 7. Phase 3：文件重整 + PF 更新 + 壓力測試

**Phase 目標**：用 just-in-time 重整文件、更新 Park Factor 至 2023-2025 加權、TDD 驗證紀律規則精煉後仍生效。

P3 拆 3 個 sub-phase：P3a / P3b / P3c。

---

### 7.1 P3a：Park Factor 更新

**目標**：PF 數值對齊 2023-2025 3 年加權；資料結構從 hardcoded dict 抽到 JSON；加 HR PF 欄位但暫不啟用；舊球場名提供 alias 向後相容。

#### 7.1.1 P3a.1 — 建立 `scripts/data/park_factors.json`

完整內容見**附錄 A**。要點：

- decimal × 100 + round（1.310 → 131；0.816 → 82）保 100-base 不改 predict.py formula
- 新增 `hr_pf` 欄位（FanGraphs HR PF），暫不被 predict.py 讀取
- 球場改名 alias mapping（Tropicana → Steinbrenner / Oakland → Sutter / Minute Maid → Daikin / Camden → Oriole Park / Guaranteed Rate → Rate / Dodger → UNIQLO at Dodger）

#### 7.1.2 P3a.2 — 改造 `merge_game_data.py`

刪除 hardcoded `PARK_FACTORS` dict（L19-50），改成：

```python
import json
from pathlib import Path

_PF_DATA_PATH = Path(__file__).parent / "data" / "park_factors.json"
_PF_DATA = json.loads(_PF_DATA_PATH.read_text(encoding="utf-8"))
PARK_FACTORS = _PF_DATA["park_factors"]
PARK_ALIASES = _PF_DATA["_aliases"]


def resolve_park_factor(venue_name: str | None) -> float:
    """以 venue_name 解析 runs PF（HR PF 暫不啟用）"""
    if not venue_name:
        return 100.0
    canonical = PARK_ALIASES.get(venue_name, venue_name)
    entry = PARK_FACTORS.get(canonical)
    if entry:
        return float(entry["runs_pf"])
    return 100.0
```

`predict.py` 不需改（仍讀 `data["park_factor"]` 單一數值）。

#### 7.1.3 P3a.3 — 更新 `reference/matchup-factors.md` Park Factor 章節

替換 L161-168 的整段為：

```markdown
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
```

#### 7.1.4 P3a 完成條件

- [ ] `scripts/data/park_factors.json` 存在，schema 符合附錄 A
- [ ] `merge_game_data.py` 從 JSON 讀 PF
- [ ] 跑 `merge_game_data.py` 對 2026-04-25 任一場比賽，確認 `merged.json` 的 `park_factor` 與新 JSON 一致
- [ ] `pytest scripts/tests/` 全綠
- [ ] 舊球場名測試：手動構造 `venue: "Tropicana Field"` → 解析為 96（透過 alias）
- [ ] `matchup-factors.md` Park Factor 章節已更新

#### 7.1.5 P3a 估時：1.5-2 hr

---

### 7.2 P3b：文件重整

**目標**：執行 just-in-time 文件原則，精煉至 token-efficiency 規範。

#### 7.2.1 P3b.1 — 各 reference 檔角色重新定義

| 檔 | 角色 | 預期變化 |
|---|---|---|
| `SKILL.md` | 入口 + 流程骨架（< 150 行） | 刪「最高優先 3 項技術漏洞」段落（散到各檔正確位置）；加「亞洲盤口歧義」與「使用者質疑結果」一行 |
| `workflow.md` | Phase 執行 SOP（**所有閘門 inline 1-2 行 + cross-ref**） | 不動結構，但確認每處規則 inline 提醒 ≤ 2 行 |
| `matchup-factors.md` | 方法論 canonical 單一來源 | 完整保留 BABIP / ERA-xERA / TJ / 年齡 / Park Factor / 牛棚累計效應；P3a 已改 Park Factor 章節 |
| `prediction.md` | 公式 + 紀律 D1-D5 + Kelly | 內部去重（P1.4 已做）；改寫 D1（XGBoost 已清） |
| `output-format.md` | 報告模板 | 不動 |
| `teams-and-api.md` | 隊名/API 對照 | 不動 |
| `odds-format.md` | 盤口輸入 | 加段「亞洲盤口歧義」（從 pitfalls.md 移來） |
| ❌ `pitfalls.md` | — | 刪除 |
| ⚠️ `flags-checklist.md` | 違規旗標索引 | 13 條 → 每條 2-3 行（觸發條件 + cross-ref） |

#### 7.2.2 P3b.2 — 刪除 `pitfalls.md`

| pitfalls 內容 | 去處 |
|---|---|
| Edge Cases 表 11 行 | 已分散於 matchup-factors.md / prediction.md，不需移 |
| 修正係數速查表 | 同上，已分散 |
| 「亞洲盤口格式歧義」 | 移至 `odds-format.md` 末段 |
| 「使用者質疑結果」 | 移至 `SKILL.md`「語氣與風格」章節 |

#### 7.2.3 P3b.3 — 精煉 `flags-checklist.md`

每條 13 旗標用同一格式：

```markdown
### 3. Hot/Cold 判定未檢查 BABIP
- 觸發：近 7 天 BABIP ≤ .260 或 ≥ .370，未做回歸判定
- 處理：跳到 `matchup-factors.md#babip-回歸檢查`
```

預估行數：~150 行 → ~50 行。

#### 7.2.4 P3b.4 — 改寫 `prediction.md` D1 紀律

XGBoost 已清，D1 原本「`ml_lean vs formula_lean` 比對」變單模型。改寫為：

```markdown
### D1：模型輸出紀律

`formula_prediction.lean`（HOME 或 AWAY）為唯一決定方向的依據。
- 可調整：勝率幅度 ±5%、信心降級
- 不可覆蓋：軟性因素（Platoon / 連勝動能 / H2H 等）影響強度，不影響方向
- ML 路徑（XGBoost）於 2026-04 重構移除，舊 `cross_validation` 欄位不再產出

> 預測紀錄歷史檔仍含 `cross_validation` 欄位，僅供觀察，新預測不再寫入。
```

#### 7.2.5 P3b.5 — Cross-ref 重整（最後掃描）

完成 P3b.2-P3b.4 後，全 skill grep 確認：

- 每條規則只在 canonical 檔（matchup-factors.md）有完整版
- 觸發點（workflow / SKILL / flags）有 1-2 行 inline 提醒 + cross-ref
- 沒有相同 threshold 出現超過 2 次（一次 canonical + 一次觸發點 inline）

具體 grep 命令（明確列舉每條 threshold）：

```bash
# BABIP 閾值
grep -rn "\.260\|\.370" SKILL.md reference/
# ERA-xERA 落差
grep -rn "ERA.*xERA\|≥ *1\.5\|>= *1\.5" SKILL.md reference/
# IP 與 prior_year ERA delta 閾值
grep -rn "IP *< *30\|≥ *1\.0\|>= *1\.0" SKILL.md reference/
# BvP PA 閾值
grep -rn "PA *≥ *15\|PA *>= *15\|< *15" SKILL.md reference/
# O/U 噪音閾值
grep -rn "1\.5 *run\|< *1\.5" SKILL.md reference/
```

合格條件：每條 threshold 在「SKILL.md + reference/*.md」總出現次數 ≤ 2（一次 canonical 在 matchup-factors.md，一次觸發點 inline）。

#### 7.2.6 P3b 完成條件

- [ ] `pitfalls.md` 不存在
- [ ] `flags-checklist.md` 行數 < 80
- [ ] `SKILL.md` 行數 < 150 **且** `wc -w SKILL.md` < 500（writing-skills「Other skills」規範）
- [ ] grep 確認每條 threshold 在「SKILL.md + reference/*.md」出現次數 ≤ 2，threshold 列舉：`.260` / `.370`（BABIP）/ `1.5`（ERA-xERA / O/U noise）/ `30`（IP 閾值）/ `1.0`（prior_year ERA delta）/ `15`（BvP PA）
- [ ] D1 紀律改寫完成，不再提 ML / XGBoost / cross_validation 邏輯
- [ ] cross-ref 全部指向有效 anchor（matchup-factors.md 章節 ID 對齊）

#### 7.2.7 P3b 估時：2-3 hr

---

### 7.3 P3c：壓力測試（TDD）

**目標**：驗證 P3a + P3b 改動後，紀律規則仍正確觸發。

#### 7.3.1 P3c.1 — 6 個測試場景

詳細場景見**附錄 C**。摘要：

| # | 場景 | 設定 | 期望 AI 行為 |
|---|---|---|---|
| T1 | BABIP 高極端 | 主隊近 7 天 BABIP = .395，連勝 5 場 | 觸發 BABIP 回歸判定，不標 Hot |
| T2 | BABIP 低極端 | 客隊近 7 天 BABIP = .245，連敗 4 場 | 觸發 BABIP 回歸判定，不標 Cold |
| T3 | ERA-xERA 落差 | 主隊投手 ERA 2.80 / xERA 4.50 (差 1.7) | 觸發 YoY Statcast 補跑 + TaskCreate B7 |
| T4 | 牛棚雙向閘門 | 客隊 Closer + Setup IL | 同時計算 ML -% + OU +run |
| T5 | D3 對立方向 | formula home_win_pct = 65% | 推 ML(HOME)，不推「AWAY 受讓」 |
| T6 | D5 比分一致性 | adjusted_total = 8.2，OU line = 9.5 | 推 UNDER 或 PASS，不推 OVER |

#### 7.3.2 P3c.2 — RED：baseline 跑

對「**改動前**的 SKILL.md + reference + 模擬 game data」dispatch subagent，請它做 Phase 1-4 分析。

- 觀察輸出，確認當前散落多處規則下 AI 能正確抓到
- baseline 必須全 6 個場景 PASS，否則 P3 改動無 regression 基準
- 若 baseline 有場景 FAIL → 回頭修目前 reference 結構，再重跑

#### 7.3.3 P3c.3 — GREEN：post-change 跑

對「**改動後**版本（P3a + P3b 完成）+ 同樣 game data」dispatch subagent。

- 比較 AI 輸出 vs baseline 是否相符
- 全 6 場景 PASS → 紀律規則精煉成功，可 deploy
- 任一場景 FAIL → 進入 REFACTOR

#### 7.3.4 P3c.4 — REFACTOR：補強漏洞

漏掉的場景 → 補強提醒位置（通常是 workflow.md 的 inline 1-2 行）。重跑 GREEN 直到全 PASS。

#### 7.3.5 P3c 完成條件

- [ ] 6 場景全部 baseline PASS
- [ ] 6 場景全部 post-change PASS
- [ ] REFACTOR 過程的補強記錄在 git commit message

#### 7.3.6 P3c 估時：1-2 hr

---

## 8. 整體驗證清單

### 8.1 Phase 級驗證

- [ ] P1 完成條件全綠
- [ ] P2 完成條件全綠
- [ ] P3a 完成條件全綠
- [ ] P3b 完成條件全綠
- [ ] P3c 完成條件全綠

### 8.2 Skill 級驗證

- [ ] `pytest scripts/tests/` 全綠
- [ ] `python scripts/predict.py --game-data <merged.json> --save` 對 2026-04-25 任一場跑得通且輸出合理（與既存 prediction 比較 PF 影響的差異是預期的）
- [ ] 全 skill 行數比改前少 30% 以上（excluding analysis-data/odds_snapshots，對比 P1.0 記錄的 baseline）
- [ ] `SKILL.md` 字數比改前少 ≥ 30%（對比 P1.0 baseline）；絕對值 < 500 字
- [ ] grep 找不到任何「dead code」殘留：`predict_with_ml`、`xgb_`、`closing_line`、`clv`、`upload`、`cross_validation`（除歷史 prediction.json）

### 8.3 文件級驗證

- [ ] `pitfalls.md` 不存在
- [ ] `train.py` / `update_model.py` / `_backtest_rl_relaxation.py` / `fetch_results.py` / `summarize_predictions.py` / `review_stats.py` / `diagnose_metrics.py` 不存在
- [ ] `analysis-logs/` 目錄不存在
- [ ] `scripts/data/park_factors.json` 存在
- [ ] `flags-checklist.md` < 80 行
- [ ] `SKILL.md` < 150 行
- [ ] `requirements.txt` 只剩 3 個套件

---

## 9. 風險與緩解

| 風險 | 緩解 |
|---|---|
| P1 改 predict.py 後 unit test 失敗 | 先列出 `predict_with_ml` 所有呼叫點，改前先 grep；test_predict_snapshot.py 涉及 ML 的部分需同步修 |
| P2 刪 4 script 後另一台電腦的 mlb-post-game-review 無法同步 | 用戶自行確認 mlb-post-game-review 已有副本；本 phase 不負責同步 |
| P3a JSON 路徑相依於 cwd 解析錯誤 | 用 `Path(__file__).parent / "data"` 不依賴 cwd |
| P3a Coors Field 從 115 → 131，預測結果大改 | 預期行為（PF 對齊新數據）。validation 改用「formula 一致性」而非「數值不變」 |
| P3b 文件改動後 AI 漏掉某條規則 | TDD P3c 的 6 場景測試正是為此設立；fail 的場景就是漏點 |
| P3c subagent 跑場景成本高 | 每場景 5-10 min × 6 = 30-60 min；接受此成本 |
| 多階段中途用戶想中斷 | P1 / P2 / P3a / P3b / P3c 各自獨立 commit，可隨時停在某階段 |

---

## 10. 估時總覽

| Phase | 工作 | 估時 |
|---|---|---|
| P1 | 純代碼清理 | 1-2 hr |
| P2 | post-game scripts 移除 + 已刪檔 commit + analysis-logs 刪除 | 1-1.5 hr |
| P3a | Park Factor JSON 化 + 數值更新 + matchup-factors 改寫 | 1.5-2 hr |
| P3b | pitfalls 刪 / flags 精煉 / D1 改寫 / cross-ref 重整 | 2-3 hr |
| P3c | 6 場景壓力測試（baseline + post-change） | 1-2 hr |
| **總計** | | **6.5-10.5 hr** |

---

## 11. 後續工作（非本 spec 範圍）

完成本重構後，獨立規劃的工作：

1. **fetch_odds + 盤口追蹤系統重做**（D9）：smart money 流向追蹤、開盤到收盤的盤口移動、Pinnacle 多時間點 snapshot
2. **大數據 Park Factor 整合**：LHB / RHB 拆分、單月 PF（Coors 4 月）、HR PF 在 predict.py 啟用
3. **mlb-post-game-review skill 整理**（在另一台電腦）：補上 fetch_results / summarize / review_stats / diagnose_metrics + analysis-logs 結構

---

## 附錄 A — `scripts/data/park_factors.json` 完整 schema

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

---

## 附錄 B — 檔案變動總表

### B.1 刪除

| 檔 | Phase |
|---|---|
| `scripts/train.py` | P1 |
| `scripts/update_model.py` | P1 |
| `scripts/_backtest_rl_relaxation.py` | P1 |
| `scripts/fetch_results.py` | P2 |
| `scripts/summarize_predictions.py` | P2 |
| `scripts/review_stats.py` | P2 |
| `scripts/diagnose_metrics.py` | P2 |
| `scripts/tests/test_fetch_results.py` | P2 |
| `analysis-data/2025-04-24/MIN@TB/game_data.json` | P2 |
| `plans/2026-04-23-phase1-readability.md` | P2 |
| `setup_task.bat` | P2 |
| `setup_task.ps1` | P2 |
| `analysis-logs/`（整目錄） | P2 |
| `reference/pitfalls.md` | P3b |

### B.2 新增

| 檔 | Phase |
|---|---|
| `scripts/data/park_factors.json` | P3a |
| `docs/superpowers/specs/2026-04-26-mlb-skill-slimming-design.md` | （本檔，brainstorming phase） |

### B.3 修改

| 檔 | Phase | 改動性質 |
|---|---|---|
| `scripts/predict.py` | P1 | 移除 ML 路徑 + 過時註釋 |
| `scripts/merge_game_data.py` | P3a | PARK_FACTORS dict → JSON load + alias 解析 |
| `scripts/requirements.txt` | P1, P2 | 刪 5 個套件 |
| `reference/workflow.md` | P2 | L328 簡化 |
| `reference/prediction.md` | P1, P2, P3b | RL 表去重 + 預測紀錄段簡化 + D1 改寫 |
| `reference/matchup-factors.md` | P3a | Park Factor 章節改寫 |
| `reference/flags-checklist.md` | P3b | 13 條精煉為「2-3 行/條」 |
| `reference/odds-format.md` | P3b | 加「亞洲盤口歧義」段 |
| `SKILL.md` | P3b | 刪「最高優先 3 項」段 + 加「使用者質疑結果」 |

---

## 附錄 C — 6 個壓力測試場景詳細

### C.1 T1 — BABIP 高極端

**Setup**：
- Game: NYM @ PHI 2026-04-30
- 主隊（PHI）lineup analyzer 輸出近 7 天 BABIP = .395
- 連勝 5 場（home_recent_ws = 5）
- merged.json 其他欄位正常

**期望 AI 行為**：
- Phase 3.4 近期狀態分析：偵測 BABIP 高極端 → 觸發 B10 TaskCreate
- 寫入 `phase3_summary.md` §BABIP 回歸判定
- **不**將 PHI 標為 Hot
- 預測 Run Value 不加 +0.5 Hot 修正

**判定 PASS**：subagent 輸出含「BABIP 回歸」+「不 Hot」字樣，phase3_summary.md 含 §BABIP 回歸判定。

### C.2 T2 — BABIP 低極端

**Setup**：客隊（NYM）lineup BABIP = .245，連敗 4 場。

**期望 AI 行為**：偵測 BABIP 低極端 → 觸發回歸判定 → **不**標 Cold → 不扣 -0.5 Run Value。

### C.3 T3 — ERA-xERA 落差

**Setup**：
- 主隊投手 ERA 2.80 / xERA 4.50（落差 1.70）
- IP 已過 30，prior_year ERA = 3.50

**期望 AI 行為**：
- Phase 2 Step 2 閘門：偵測 |ERA-xERA| ≥ 1.5 → 必須補跑 `pitcher_stats.py --year 2025`
- TaskCreate B7（補跑 YoY 對比）
- phase3_summary.md §YoY 對比結論
- 不通過閘門前不得進 Phase 3.5

**判定 PASS**：subagent 必須 invoke `pitcher_stats.py` 帶 `--year 2025`，產出 `home_pitcher_2025.json`。

### C.4 T4 — 牛棚雙向閘門

**Setup**：客隊 Closer + Setup IL（B9 觸發）。

**期望 AI 行為**：
- 同時計算 OU 修正 +0.5 run + ML 修正 -3% （該隊勝率下修）
- TaskCreate B9（牛棚雙向修正值）
- phase3_summary.md §牛棚雙向修正值

**判定 PASS**：phase3_summary.md 出現「OU +」與「ML -%」**雙方向**修正值，缺一即 FAIL。

### C.5 T5 — D3 對立方向

**Setup**：formula 算出 home_win_pct = 65%。Game = NYM @ PHI（主隊 PHI）。

**期望 AI 行為**：
- ml_rec 為**主隊縮寫**（此例：`PHI`），不得是字面值 `HOME`（workflow.md L280：predict.py 會 reject `HOME` 字面值）
- run_line_rec 為 `PHI` / `PHI -1.5` / `PASS` 任一，**不得**為 `NYM` / `NYM +1.5`

**判定 PASS**：subagent 產出的 prediction.json 中 `ml_rec == "PHI"`（主隊 abbrev），且 `run_line_rec` ∉ {`NYM`, `NYM +1.5`, `AWAY +1.5`}。

### C.6 T6 — D5 比分一致性

**Setup**：formula 算出 adjusted_total = 8.2，OU line = 9.5（差距 1.3，< 1.5 噪音閾值）。

**期望 AI 行為**：推 `ou_rec: PASS`（差距 < 1.5），不推 OVER（adjusted < line）。

**判定 PASS**：subagent 輸出 ou_rec = PASS。

---

## 附錄 D — Brainstorm 紀錄

本 spec 由 superpowers:brainstorming 流程產出，clarifying questions 順序：

1. **Q1**：勝率 XGBoost 路徑處理 → 用戶選 A（全清光）
2. **Q2**：多處規則重複處理 → 用戶選 D（just-in-time + 壓力測試），對齊 superpowers:writing-skills 規範
3. **Q3**：執行順序 → 用戶選 B（三階段 P1 → P2 → P3）
4. **Q4**：design 文件結構 → 用戶選單檔
5. **Q5**：Park Factor 處理（從原本「移除」改成「對齊新數據」）→ JSON 化 + 加 HR PF dormant + alias 向後相容

中途用戶澄清：mlb-post-game-review skill 在另一台電腦，本 phase 不負責建立或同步，僅從本 skill 刪除。

---

**End of spec.**
