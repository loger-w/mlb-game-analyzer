# 2026-04-20 — Skill 路徑路由修正

## Context

MLB skill 執行預測時，資料應儲存於 `analysis-data/<YYYY-MM-DD>/<AWAY>@<HOME>/`，但觀察到模型有時會把檔案（尤其 `prediction.json`）寫到 skill 根目錄下的 `games/` 資料夾。這會讓下游 `upload_prediction.py`、`review_stats.py`、`summarize_predictions.py` 找不到資料，整條後處理鏈都會斷。

## 現況觀察

| 項目 | 狀態 |
|---|---|
| `SKILL.md` 行數 | 173 行 / 757 字 |
| `SKILL.md` 中 `analysis-data` 出現次數 | **0 次** |
| `SKILL.md` 中 `games/` 出現次數 | **0 次** |
| `games/` 資料夾是否存在 | **存在但為空**（最後修改 2026-04-19 23:02） |
| `analysis-data/` 資料夾 | 存在，含 `2026-04-18/`、`2026-04-19/` 子資料夾 |
| 路徑定義所在檔 | `reference/workflow.md:20,24` — `GAME_DIR=analysis-data/{YYYY-MM-DD}/{AWAY}@{HOME}` |
| `predict.py` 如何存檔 | `predict.py:1042-1043` — 從 `--game-data` 參數推得父層，再寫 `prediction.json`（邏輯本身正確） |
| 其他硬編碼路徑的腳本 | `upload_prediction.py:17`、`upload_results.py:17,30,79`、`review_stats.py:16`、`summarize_predictions.py:24,56`、`backfill_clv.py:27` |

## 根因

1. **`SKILL.md` 完全沒提到 `analysis-data/`** — 路徑約定被下放到 `reference/workflow.md`，但模型不保證每次都會載入 reference。
2. **空的 `games/` 資料夾是誘餌** — 語意上 "games" 聽起來像「比賽資料夾」，當模型沒讀到 workflow.md 又看到根目錄有這個資料夾，就會自行腦補「應該寫到 games/」。
3. **沒有中央 guard** — `predict.py` 只在收到正確 `--game-data` 參數時才會寫對地方；只要模型把 `--game-data` 填錯（或自己 `mkdir games/` 再 cp），就繞過所有保護。

結論：這不是程式碼問題，是**指令文件不夠明確 + 誘餌目錄**的組合。

## 解決方案

### Step 1 — 在 `SKILL.md` 頂端新增一段「路徑契約」

在 `SKILL.md` 的第一個 phase 前（大約 `## 執行流程` 之類的 section 上方）插入：

```markdown
## 📁 資料儲存契約（強制遵守）

所有預測相關資料**一律**儲存於：

    analysis-data/<YYYY-MM-DD>/<AWAY>@<HOME>/

禁止寫入以下位置（常見錯誤）：
- ❌ skill 根目錄
- ❌ `games/`（此資料夾不存在；若你看到它，視為垃圾）
- ❌ `scripts/` 內
- ❌ 當日之外的任何其他日期資料夾

執行 `predict.py` 時**必須**帶：
    --game-data analysis-data/<date>/<game>/merged.json --save

`prediction.json` 會自動落在 `merged.json` 同層，這是唯一正確路徑。
```

### Step 2 — 刪除誘餌目錄 `games/`

先確認沒有任何活躍引用：

```bash
# 驗證步驟，都要返回空
grep -r "games/" --include="*.py" scripts/
grep -r "games/" --include="*.md" .
```

確認無誤後刪除：

```bash
rmdir "C:\Users\Loger\.claude\skills\mlb-game-analyzer\games"
```

（若 `rmdir` 拒絕，代表還有隱藏檔；檢查後再 `rm -rf`，但先 commit 一次避免誤刪。）

### Step 3 — 在 `SKILL.md` 的 Phase 4 輸出段補 guard 句

找到 Phase 4（預測/輸出那段），結尾加：

```markdown
> ⚠️ 寫入 `prediction.json` 前，務必確認落點是 `analysis-data/<date>/<game>/` — 若不是，停下來重新定位 `--game-data`。
```

## 風險

| 風險 | 嚴重度 | 緩解 |
|---|---|---|
| `games/` 內可能有漏看的舊檔 | 低 | 先 `ls -la games/` 檢查，若有檔先移到 `archive/` |
| 其他 skill 或外部腳本硬寫 `games/` | 極低 | Step 2 之前的 grep 已涵蓋 |
| SKILL.md 增字會觸發 token 壓力 | 低 | 本次只加 ~15 行，173 → ~188，仍在可控範圍 |
| 模型仍忽視指令 | 中 | 若這次改完還會寫錯，表示得改架構（例如 `predict.py` 拒絕非 `analysis-data/` 的 `--game-data`） |

## 驗證

### 前置檢查
- `grep -rn "games/" .` 在非忽略檔中返回 0 筆
- `ls games/` 回 "No such file or directory"

### 功能驗證
對 `2026-04-20` 跑一場實際預測，過程中：
1. 觀察模型下的 `predict.py --game-data ...` 指令是否指向 `analysis-data/2026-04-20/<AWAY>@<HOME>/merged.json`
2. 執行完後 `ls analysis-data/2026-04-20/<AWAY>@<HOME>/` 須包含 `prediction.json`
3. 確認 skill 根目錄**沒有**出現 `games/` 資料夾（model 不會自己建）
4. 跑 `python scripts/review_stats.py --date 2026-04-20`，能讀到該筆預測 → 證明下游鏈接通

### 長期觀察
- 連續 5 個交易日每天檢查 `ls analysis-data/<date>/` 有沒有遺漏
- 若再發生寫錯路徑，視為 guard 不夠硬，升級為 `predict.py` 內加 `assert "analysis-data" in args.game_data`

## 關鍵檔案

- `C:\Users\Loger\.claude\skills\mlb-game-analyzer\SKILL.md` — 主要修改目標（新增路徑契約）
- `C:\Users\Loger\.claude\skills\mlb-game-analyzer\games\` — 刪除目標
- `C:\Users\Loger\.claude\skills\mlb-game-analyzer\reference\workflow.md:20,24` — 既有路徑定義（不動，但可 cross-reference）
- `C:\Users\Loger\.claude\skills\mlb-game-analyzer\scripts\predict.py:1042-1043` — 儲存邏輯（不動）
