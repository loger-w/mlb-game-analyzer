# 比賽預測統一使用台灣時間（folder + summary）

**日期**: 2026-04-29
**Skill**: mlb-game-analyzer
**範圍**: `prepare_game.py --date` 語意 / `analysis-data/` folder 命名 / `prediction_summary.md` 開打時間 + 同步刪除 post-game-review 殘留

---

## 1. 背景與目標

### 問題

Odds 模組於 2026-04-29 commit `eb1fd58` 把 `--date` 與報告全面切到 TW 時區，但 mlb-game-analyzer 的預測流程仍是 ET-centric：
- `prepare_game.py --date 2026-04-29` 是 ET 開賽日
- folder = `analysis-data/2026-04-29/{AWAY}@{HOME}/` 是 ET 日期
- `prediction_summary.md` 標題 `# Prediction Summary — TB @ CLE (2026-04-29)` 是 ET 日期
- prediction.json record `"date"` 欄位是 ET，`"game_time"` 是 UTC ISO

使用者在 TW 操作時心智模型錯位：
- TW 4/29 21:00 想預測「明天清晨開打的 MLB 場次」，但要打 `--date 2026-04-29`（ET 日），folder 也跑去 `2026-04-29/`，跟「比賽 TW 4/30 開打」對不上

### 目標

將 mlb-game-analyzer 預測流程的「user-facing 介面 + 最終報告」切到 TW 時區，與 odds 模組對齊；同時刪除已棄置的 mlb-post-game-review skill 相關殘留。

### 範圍邊界

| 範圍內 | 範圍外 |
|--------|--------|
| `prepare_game.py --date` 改 TW 語意 + 內部 TW→ET 換算 | `dossier.md` / `merged_summary.md` / 各 `*_summary.md` ET 標記（中間層維持） |
| `analysis-data/{TW-date}/...` folder 結構 | `prediction.json` 內 `game_time`（UTC ISO 維持） |
| `prediction_summary.md` 標題 + 開打時間改 TW | `fetch_game_data.py` / `merge_game_data.py` / `dossier_renderer.py` 內部 ET 邏輯 |
| `predict.py` record `date` 欄值改 TW | `validate_game_data_path` regex（ET 還 TW 它不在意） |
| 刪除 `diagnose_ou_total_error.py` + `predict.py` 內 `actual_*` / `verified` 欄位 | mlb-post-game-review skill folder 本體（已不存在） |
| 文件清理 SKILL.md / prediction.md 內 post-game-review cross-ref | |

---

## 2. TW ↔ ET 換算規則

**核心：`et_date = tw_date − 1 day`（給 MLB schedule API 用）**

依據用戶 2026-04-29 對話：folder 是「US gameday slate 用 TW notation 表示」= `ET_date + 1`。

### 規則表

| 場景 | 用戶輸入 | MLB API ET date | folder |
|------|---------|-----------|--------|
| TW 4/29 21:00 預測明早 TB@CLE（ET 4/29 21:11 = TW 4/30 09:11） | `--date 2026-04-30` | `2026-04-29` | `analysis-data/2026-04-30/TB@CLE/` |
| TW 4/29 14:00 預測 ET 4/29 11:00 day game（TW 4/29 23:00 開賽） | `--date 2026-04-30` | `2026-04-29` | `analysis-data/2026-04-30/.../`（同一資料夾） |
| Doubleheader G1/G2（同 ET 4/29 兩場 TB@CLE） | `--date 2026-04-30 --game-suffix G1` | `2026-04-29` | `analysis-data/2026-04-30/TB@CLE-G1/` |

### 邊界與不變式

- **DST 不影響**：MLB 球季 EDT 固定 UTC-4，TW 固定 UTC+8，差距永遠 12 小時
- **沒有跨多日歧義**：MLB 一場比賽通常 3-4 小時；commence_utc 對應一個明確 ET 日 + 一個明確 TW 日，差頂多 1 天
- **early ET day game 例外**：ET 4/29 11:00 game（commence_utc → astimezone(TW) = TW 4/29 23:00），folder 仍歸 `2026-04-30`（= ET_date + 1），**不是直接 UTC→TW**
- **fallback 路徑算法**：當無法從 path 取 TW date，須用 `et_date + 1 day` 而非 `astimezone(TW)`，兩者只差在這個 early-day-game edge case

---

## 3. 變更點

### `scripts/prepare_game.py`

- `--date` arg help text 改：「YYYY-MM-DD（TW 開打日；內部換算 ET = TW − 1 day）」
- 新增 module-level helper：

  ```python
  def _tw_to_et(tw_date: str) -> str:
      from datetime import datetime, timedelta
      d = datetime.strptime(tw_date, "%Y-%m-%d").date()
      return (d - timedelta(days=1)).strftime("%Y-%m-%d")
  ```

- `step_a` / `step_b` / `step_c` / 任何 sub-script 呼叫處（`fetch_game_data.py` / `lineup_analyzer.py` / `roster_checker.py` / `pitcher_stats.py`）：原本傳 `args.date` 改傳 `_tw_to_et(args.date)`
- `compute_output_dir`：照舊用 `args.date`（現在是 TW），不需修改

### `scripts/predict.py`

- `_extract_game_date_et` rename 為 `_extract_game_date_tw`
- fallback 邏輯改為「先算 ET，再加 1 天」：

  ```python
  utc_dt = datetime.fromisoformat(game_date_iso.replace("Z", "+00:00"))
  et_date = utc_dt.astimezone(_ET_TZ).date()
  game_date_tw = (et_date + timedelta(days=1)).strftime("%Y-%m-%d")
  ```

- `_ET_TZ` 維持（fallback 仍需要）；新增 module-level `_TW_TZ = timezone(timedelta(hours=+8))`（summary 與 fallback 共用）
- predict.py:1015 的最終 fallback `_dt.now().strftime("%Y-%m-%d")` 改成 `_dt.now(_TW_TZ).strftime("%Y-%m-%d")` — 明示 TW 語意，避免非 TW 機器 fallback 漂移
- record dict：`"date": record_date`（現值為 TW）
- record dict 移除五個欄位：`"actual_winner"`, `"actual_home_score"`, `"actual_away_score"`, `"actual_total"`, `"verified"`
- `format_prediction_summary_md`：
  - header 維持 `# Prediction Summary — {away_abbr} @ {home_abbr} ({date})`，date 現在是 TW
  - header 下方加 meta 行：

    ```python
    game_time_iso = record.get("game_time")
    if game_time_iso:
        try:
            utc_dt = datetime.fromisoformat(game_time_iso.replace("Z", "+00:00"))
            tw_label = utc_dt.astimezone(_TW_TZ).strftime("%Y-%m-%d %H:%M TW")
        except ValueError:
            tw_label = "未知"
    else:
        tw_label = "未知"
    lines.append(f"**開打時間**: {tw_label}")
    lines.append("")
    ```

### `scripts/diagnose_ou_total_error.py`

- **整檔刪除**（依賴 `predictions.jsonl`，post-game-review 範疇）

### 中間層腳本（不動）

- `scripts/fetch_game_data.py` — 繼續吃 ET date 給 MLB schedule API；輸出仍寫 `日期 (ET):` / `開賽 (UTC ISO):`
- `scripts/merge_game_data.py` — 不動
- `scripts/dossier_renderer.py` — `日期 (ET):` / `開賽 (UTC ISO):` 維持
- `scripts/lineup_analyzer.py` / `pitcher_stats.py` / `roster_checker.py` — 不動（純查詢腳本）

### 文件更新

- `SKILL.md` line 20：刪「賽後回顧（轉 `mlb-post-game-review`）」這個「不適用」項目
- `reference/prediction.md` line 188-192：`## 預測紀錄存放位置` 區塊改寫為只剩 per-game prediction.json，移除 `predictions.jsonl` per-date summary 行與 mlb-post-game-review 提及；移除「賽後回填 `actual_*` / `verified=true`」這行
- `reference/flags-checklist.md` / `reference/matchup-factors.md`：grep 一遍有沒有 cross-ref，有則清

---

## 4. `prediction_summary.md` 格式

**前後對照：**

| 段落 | 改前 | 改後 |
|------|------|------|
| H1 標題 date | `(2026-04-29)` ET | `(2026-04-30)` TW |
| 開打時間 meta | （無） | `**開打時間**: 2026-04-30 01:11 TW`（純 TW，無 ET 副欄） |

**完整範例（前 5 行）：**

```markdown
# Prediction Summary — TB @ CLE (2026-04-30)

**開打時間**: 2026-04-30 01:11 TW

## TL;DR
```

**設計決定：**
- 純 TW 不掛 ET 副欄（user 確認；保持簡潔）
- meta 行緊接 H1，與 TL;DR 之間隔空行（與 odds reports cover line 風格一致）
- `record["game_time"]` 缺失：fallback `**開打時間**: 未知`（顯示但明示降級）
- 不存新欄位（`game_time_tw` 等）—— 即時從 UTC ISO 算

---

## 5. 測試計畫

### 既有測試需更新

- `scripts/tests/test_predict.py`
  - `_extract_game_date_et` 相關測試 rename + 改斷言（fallback 路徑現在算 `ET + 1 day`）
  - record dict 斷言移除 `actual_*` / `verified` 五欄位
  - `prediction_summary.md` 斷言加「開打時間」meta 行檢查
- `scripts/tests/test_prepare_game.py` / `test_prepare_game_steps.py`
  - `--date` arg 斷言改 TW 語意
  - 驗證內部呼叫 sub-script 時用 `_tw_to_et(args.date)` 而非 `args.date`
  - folder name 斷言用 TW 日期

### 新增測試

- `test_tw_to_et_conversion`：`_tw_to_et("2026-04-30")` → `"2026-04-29"`
- `test_extract_game_date_tw_path_based`：path 含 `analysis-data/2026-04-30/...` → 回 `"2026-04-30"`
- `test_extract_game_date_tw_fallback_early_day_game`：UTC ISO `2026-04-29T15:00:00Z`（ET 4/29 11:00 = TW 4/29 23:00），預期 fallback 回 `"2026-04-30"`（ET_date + 1，**非** astimezone(TW)）
- `test_extract_game_date_tw_fallback_night_game`：UTC ISO `2026-04-30T01:11:00Z`（ET 4/29 21:11 = TW 4/30 09:11），預期 fallback 回 `"2026-04-30"`
- `test_prediction_summary_tw_open_time`：record 含 `game_time = "2026-04-30T01:11:00Z"`，預期 summary md 含 `**開打時間**: 2026-04-30 09:11 TW`
- `test_prediction_summary_missing_game_time_fallback`：record 缺 `game_time` → 預期 summary md 含 `**開打時間**: 未知`

### 測試命令

```
pytest scripts/tests/ -v
```

預期：所有既有 + 新增測試通過。

---

## 6. 不做的事

- **不改** `dossier.md` / `merged_summary.md` / `*_summary.md` 等中間層的 ET 顯示
- **不改** `prediction.json` 的 `game_time`（UTC ISO 維持）
- **不新增** `game_time_tw` / `game_date_tw` / `date_et` 等冗欄位
- **不改** `validate_game_data_path` regex
- **不改** `fetch_game_data.py` / `merge_game_data.py` 的 MLB API 整合邏輯
- **不重啟** post-game-review skill 或保留其 `actual_*` / `verified` 欄位
- **不改** odds 模組（其 TW 處理已於 `eb1fd58` 完成）

---

## 7. 風險與緩解

- **既有 `analysis-data/` 歷史資料夾**：目前 `analysis-data/2026-04-28/TB@CLE/` 顯示為 deleted 狀態（git status），其餘歷史資料夾用 ET 日命名。**不做** retroactive rename；新流程從本 commit 起 TW 命名，舊歷史資料夾保留 ET 命名作為「pre-spec era」標記，不衝突
- **post-game-review 棄置欄位殘留**：刪除 `actual_*` / `verified` 五欄位是 schema 變更；目前無生產讀取端（review skill 已不存在），無 backwards-compat 包袱
- **TW 4/29 23:00 早場 vs ET fallback 路徑混淆**：fallback 算法使用 `ET + 1 day` 而非 `astimezone(TW)`，正好處理 early ET day game 落在 TW 23:00 的 edge case，已在測試覆蓋
- **使用者既有 muscle memory**：使用者過去打 `--date 2026-04-29` = ET，現在打 `--date 2026-04-30` = TW；help text 與 SKILL.md 須明示新語意，避免 silent semantic drift

---

## 8. 預期 commits 切片

實作建議切成 4 個 atomic commits（給 writing-plans 階段參考）：

1. `feat(predict): record date 欄位 + summary 開打時間改 TW` — `predict.py` + 對應 test
2. `feat(prepare_game): --date 改 TW 語意 + 內部換算 ET` — `prepare_game.py` + 對應 test
3. `chore: 移除 post-game-review 殘留` — 刪 `diagnose_ou_total_error.py` + record `actual_*` 欄位 + 文件 cross-ref
4. `docs: SKILL.md / prediction.md 更新 TW 語意說明`

順序建議 1 → 2（先測試 record/summary 層 OK 再改 user-facing entry）→ 3 → 4。
