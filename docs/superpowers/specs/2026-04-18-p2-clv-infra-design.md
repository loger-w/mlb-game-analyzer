# P2: CLV Infrastructure + M5 Line Movement — Design Spec

**Parent roadmap:** MLB Betting Skill — Root-Cause Fixes (P3 Kelly ✅ → **P2 CLV** → P1 Retrain)
**Date:** 2026-04-18
**Author (brainstorming session):** loger.hsu@gobuid.com + Claude (superpowers:brainstorming)
**Status:** Draft, awaiting user review before writing-plans

---

## 1. Goal

在 P3 已上線的 4h Pinnacle snapshot infra 上建立 CLV（Closing Line Value）追蹤：

- 推薦時將三市場完整 line 釘入 `prediction.json`
- 比賽結束後回填 closing line 並計算 CLV（American cents 主展示 + no-vig pct 副欄位）
- 附 line movement advisory block（open→rec、rec→close、steam / RLM flags）

P2 為 **P1 retrain 驗證的資料基礎** — 沒有 CLV baseline 就無法統計證明新模型確實提升 edge（W/L 樣本 <500 bets 時雜訊過大）。

## 2. Non-Goals（明確排除）

- 多 book line shopping（續 Pinnacle-only 策略）
- CLV 進模型 signal_table / 影響 `final.*`（M5 advisory-only）
- 即時 line movement 告警 / push notification
- CLV 異常值 alerting（屬 downstream 分析）
- 次小時級 RLM 偵測（4h 粒度做不到，文檔化此 limit）
- 標籤重訓 / Kelly 參數調整（屬 P1 / P3 scope）
- 新 API credit 消耗（closing line 復用現有 4h cron）

## 3. 既有基建盤點（spec 前提）

| 項目 | 位置 | 狀態 |
|---|---|---|
| Pinnacle 4h cron | `scripts/fetch_odds.py` + Windows Task Scheduler | ✅ 運作中（credits 455/500）|
| Snapshot 儲存 | `odds_snapshots/{YYYY-MM-DD}_{HH}-00-ET.json` | ✅ |
| Snapshot loader | `predict.load_closest_snapshot()` | ✅ P3 Task 9 |
| Pinnacle odds extract | `predict.resolve_pinnacle_odds()` | ✅ P3 Task 10 |
| Kelly block（含 snapshot_source 釘入） | `predict.compute_kelly_block()` | ✅ P3 Task 11-16 |
| `decimal_to_american` | `odds_analyzer.py` | ✅ P3 Task 3 |
| `TEAM_ABBREV` lookup | `predict.py:25-36` | ✅ |
| 每日 roll-up | `summarize_predictions.py` | ✅ |
| Post-game enrichment | `upload_results.py`（寫 `verified` + `result_*`）| ✅ |
| Test infra | `scripts/tests/` + pytest（39 passing） | ✅ |

**Gap（P2 要補）：**
- 無「rec-time 三市場完整 line」持久化（Kelly block 只記推薦方 decimal）
- 無 closing line 概念 / 計算
- 無 CLV 計算 / 儲存
- 無 line movement 偵測
- `_find_latest_snapshot_before` 邏輯只在 `predict.py` 有，需 refactor 共用

## 4. Architecture Decision

**新增 pure-function 模組 `scripts/clv.py`**（zero I/O 依賴，類似 P3 `odds_analyzer.py`），包含 CLV 計算與 line movement 偵測所有純函式。

**三個整合點：**
- `predict.py`（MOD）：rec-time 寫 `recommendation_snapshot` + `line_movement` 到 prediction.json
- `upload_results.py`（MOD）：post-game 寫 `closing_line` + `clv` + `rec_to_close` 到 predictions.jsonl
- `backfill_clv.py`（NEW）：一次性 CLI，對歷史 verified records 補 CLV 欄位

**Two-stage write 理由：** prediction.json 為「rec 時不可變 artifact」；post-game CLV 屬動態資料，寫 predictions.jsonl 附加欄位與既有 `verified`/`result_*` 對齊。避免檔案 race。

**CLV 範圍：三市場全算（非 Kelly-bet only）**，每市場附 `bet_placed` 旗標 — 保留 PASS 市場的模型方向 CLV 作為 P1 驗證資料源。

## 5. API 變更

### 5.1 `scripts/clv.py` — 新增

```python
from typing import TypedDict, Literal, Optional

MarketSide = Literal["HOME", "AWAY", "OVER", "UNDER"]

class MarketLine(TypedDict):
    decimal: float
    american: int
    implied_pct: float

def _find_latest_snapshot_before(
    snapshot_dir: str,
    game_date_et: str,
    cutoff_utc: str,
) -> Optional[dict]:
    """
    掃 snapshot_dir，回傳 game_date_et 相符且 snapshot_time_utc < cutoff_utc
    中最晚的一筆 snapshot dict（完整 JSON，不只 metadata）。
    此函式 refactor 自 predict.load_closest_snapshot 內部邏輯。
    """

def _find_earliest_snapshot_of_date(
    snapshot_dir: str,
    game_date_et: str,
) -> Optional[dict]:
    """回傳 game_date_et 當日最早一筆 snapshot（作為 open）。"""

def pin_rec_snapshot(
    snapshot_game: dict,        # snapshot["games"][i] 形式
    commence_utc: str,          # 從 merged.json _meta.commence_utc
    source_filename: str,
    snapshot_time_et: str,
    snapshot_time_utc: str,
) -> dict:
    """
    從 fetch_odds.py bookmakers.pinnacle 抽三市場完整 line。
    回傳 shape 見 §5.3 recommendation_snapshot。
    缺市場 → 該市場 = null。
    """

def find_closing_snapshot(
    commence_utc: str,
    game_date_et: str,
    snapshot_dir: str = "odds_snapshots",
) -> Optional[dict]:
    """_find_latest_snapshot_before 的語意包裝（cutoff = commence_utc）。"""

def find_opening_snapshot(
    game_date_et: str,
    snapshot_dir: str = "odds_snapshots",
) -> Optional[dict]:
    """_find_earliest_snapshot_of_date 的語意包裝。"""

def compute_clv_cents(
    rec_decimal: float,
    close_decimal: float,
) -> int:
    """
    American cents 差額，正值 = 推薦時 line 比 closing 優（beat closing）。
    實作：american(rec) - american(close)；對負賠率與正賠率皆正確。
    回傳 int（round）。
    """

def compute_clv_pct_no_vig(
    rec_side_dec: float,
    rec_other_dec: float,
    close_side_dec: float,
    close_other_dec: float,
) -> float:
    """
    先除 vig：p_side = (1/side_dec) / ((1/side_dec) + (1/other_dec))；
    CLV_pct = (p_rec_side - p_close_side) × 100；正值 = beat。
    round(_, 2)。
    """

def detect_line_movement(
    open_snap: Optional[dict],   # pin_rec_snapshot 輸出形式
    rec_snap: dict,
    close_snap: Optional[dict],
    recommended_direction: dict, # {"ml":"HOME","ou":"OVER"|None,"rl":"HOME"}
    steam_threshold_cents: int = 5,
) -> dict:
    """
    回傳：
    {
      "open_to_rec": {"ml_home_cents": int, "ou_cents": int,
                       "ou_point_delta": float, "rl_home_cents": int} | null,
      "rec_to_close": <same shape> | null,
      "flags": {
        "steam_toward_rec": bool,  # open_to_rec[推薦方] ≥ +threshold
        "rlm_suspected": bool      # open_to_rec[推薦方] ≤ -threshold
      },
      "warnings": [...]
    }
    cents 方向約定：正值 = 推薦方在 open→rec 區間 line 收縮到我方（利好）。
    """

def compute_bet_placed(kelly_market_block: Optional[dict]) -> bool:
    """kelly.ml/ou/rl 任一 units > 0 → True；null 或 units == 0 → False。"""
```

### 5.2 `scripts/predict.py` — 修改

1. **Refactor**：將 `load_closest_snapshot` 內部 snapshot 掃描邏輯搬到 `clv._find_latest_snapshot_before`；predict 端改為薄包裝（保持既有 call sites 不變）。P3 既有 14 snapshot tests 必須 pass。
2. **新增 rec-time blocks** 寫入 `compute_kelly_block` 之後：
   ```python
   from clv import (pin_rec_snapshot, find_opening_snapshot,
                    detect_line_movement)

   rec_snap_pinned = pin_rec_snapshot(...)  # 或 None
   open_raw = find_opening_snapshot(game_date_et)
   open_pinned = pin_rec_snapshot(open_raw, ...) if open_raw else None

   recommended_direction = {
       "ml": final["recommended_winner"],
       "ou": final["over_under_lean"] if final["over_under_lean"] != "NEUTRAL" else None,
       "rl": rec_snap_pinned["rl"]["favorite_side"] if rec_snap_pinned and rec_snap_pinned.get("rl") else None,
   }
   line_movement = detect_line_movement(
       open_pinned, rec_snap_pinned, None,
       recommended_direction,
   )

   prediction["recommendation_snapshot"] = rec_snap_pinned   # 可為 null
   prediction["line_movement"] = line_movement
   ```
3. 無新 CLI args（rec-time 不需 user 介入）。

### 5.3 `prediction.json` schema 擴充

**新增 `recommendation_snapshot` block：**

```jsonc
"recommendation_snapshot": {
  "source": "2026-04-18_16-00-ET.json",
  "snapshot_time_et": "2026-04-18 16:00 ET",
  "snapshot_time_utc": "2026-04-18T20:00:00+00:00",
  "commence_utc": "2026-04-18T23:10:00+00:00",
  "minutes_before_first_pitch": 190,
  "ml": {
    "home": {"decimal": 1.74, "american": -135, "implied_pct": 57.5},
    "away": {"decimal": 2.28, "american": 128, "implied_pct": 43.9}
  },
  "ou": {
    "point": 8.0,
    "over":  {"decimal": 1.93, "american": -108, "implied_pct": 51.8},
    "under": {"decimal": 1.93, "american": -108, "implied_pct": 51.8}
  },
  "rl": {
    "favorite_side": "HOME",
    "home": {"decimal": 1.56, "american": -179, "implied_pct": 64.1, "point": -1.5},
    "away": {"decimal": 2.52, "american": 152,  "implied_pct": 39.7, "point": 1.5}
  }
}
```
無 snapshot 可用 → 整個 block 為 `null`。某市場缺 → 該子 key 為 `null`。

**新增 `line_movement` block：**

```jsonc
"line_movement": {
  "open_snapshot": "2026-04-18_00-00-ET.json",   // 可為 null
  "rec_snapshot":  "2026-04-18_16-00-ET.json",
  "close_snapshot": null,                          // post-game 於 jsonl 回填，此檔不動
  "open_to_rec": {"ml_home_cents": -4, "ou_cents": 0, "ou_point_delta": 0.0, "rl_home_cents": 3},
  "rec_to_close": null,
  "flags": {"steam_toward_rec": false, "rlm_suspected": false},
  "granularity_note": "4h snapshot cadence; sub-hour steam not detectable",
  "warnings": []
}
```

### 5.4 `scripts/upload_results.py` — 修改

在既有 `verified` / `result_*` 寫入邏輯之後新增 CLV 區塊：

```python
from clv import (find_closing_snapshot, pin_rec_snapshot,
                 compute_clv_cents, compute_clv_pct_no_vig,
                 detect_line_movement, compute_bet_placed)

# 若 record 已有 clv 欄位且非 --force，skip
if "clv" in record and not args.force:
    continue

rec_snap = record.get("prediction", {}).get("recommendation_snapshot")
if rec_snap is None:
    record["clv"] = None
    record.setdefault("clv_warnings", []).append("no_rec_snapshot")
    continue

close_raw = find_closing_snapshot(commence_utc, game_date_et)
if close_raw is None:
    record["clv"] = None
    record.setdefault("clv_warnings", []).append("no_closing_snapshot")
    continue

close_pinned = pin_rec_snapshot(...)
record["closing_line_source"] = close_pinned["source"]
record["closing_line_minutes_before_first_pitch"] = close_pinned["minutes_before_first_pitch"]
record["closing_line"] = {k: close_pinned[k] for k in ("ml", "ou", "rl")}

# CLV per market（三市場全算；direction 取 final.*）
record["clv"] = {
    "ml": _compute_market_clv(rec_snap["ml"], close_pinned["ml"],
                               direction=final["recommended_winner"],
                               bet_placed=compute_bet_placed(kelly.get("ml"))),
    "ou": _compute_market_clv(...) if final["over_under_lean"] != "NEUTRAL" else None,
    "rl": _compute_market_clv(...),
}

# line_movement rec_to_close 補段（寫 jsonl，不動 prediction.json）
lm_update = detect_line_movement(open_pinned, rec_snap, close_pinned,
                                   recommended_direction)
record["rec_to_close"] = lm_update["rec_to_close"]

if close_pinned["minutes_before_first_pitch"] > 240:
    record.setdefault("clv_warnings", []).append(
        f"closing_stale:{close_pinned['minutes_before_first_pitch']}min")
```

### 5.5 `predictions.jsonl` 欄位擴充

```jsonc
{
  // 既有：prediction, verified, result_home_score, result_away_score, ...
  "closing_line_source": "2026-04-18_20-00-ET.json",
  "closing_line_minutes_before_first_pitch": 50,
  "closing_line": { /* ml, ou, rl */ },
  "clv": {
    "ml": {"cents": 4, "pct_no_vig": 1.8, "direction": "HOME", "bet_placed": true},
    "ou": {"cents": 2, "pct_no_vig": 0.9, "direction": "OVER", "point_delta": 0.0, "bet_placed": false},
    "rl": {"cents": -1, "pct_no_vig": -0.4, "direction": "HOME", "bet_placed": true}
  },
  "rec_to_close": {"ml_home_cents": 3, "ou_cents": 2, "rl_home_cents": 1},
  "clv_warnings": []
}
```

### 5.6 `scripts/backfill_clv.py` — 新增

```
usage: backfill_clv.py [--date YYYY-MM-DD] [--force] [--dry-run] [--all]

  --date    單日處理（預設）。省略需配合 --all。
  --all     掃 analysis-data/ 全部日期。
  --force   覆寫已存在 clv 欄位。
  --dry-run 列印欲變更 records，不寫檔（預設 ON，必須明 --no-dry-run 或明 --force 才落地）。

行為：
  - 只處理 verified == true 的 records
  - 對 legacy（無 kelly block）records → skip + log
  - 呼叫與 upload_results.py 相同的 clv 寫入 helper（不複製邏輯）
  - 完成列印：processed=N, updated=M, skipped=K, errors=E
```

## 6. Snapshot-to-Game 對照規則

### 6.1 隊名映射
沿用 P3 `predict.TEAM_ABBREV`（full name ↔ abbrev）。失敗 → `recommendation_snapshot = null` + warnings += `team_resolve_failed`。

### 6.2 Doubleheader
沿用 P3 `--game-index`。line_movement 按 index 選 open/rec/close 的該場比賽。

### 6.3 Closing snapshot 定義
`game_date_et` 相符且 `snapshot_time_utc < commence_utc` 的最晚 snapshot。
- `minutes_before_first_pitch > 240` → warnings += `closing_stale:Nmin`，資料照用
- 完全找不到 → `clv = null` + `no_closing_snapshot`

### 6.4 Opening snapshot 定義
`game_date_et` 當日最早一筆 snapshot。
- 當日只有 1 筆（與 rec 相同）→ open = null，open_to_rec = null，flags 皆 false

## 7. Fallback / Error 矩陣

| 情境 | prediction.json | predictions.jsonl（post-game） |
|---|---|---|
| 無當日 snapshot | rec_snap=null, lm=advisory-null | clv=null + `no_rec_snapshot` |
| 僅 rec 無 open | rec_snap 正常，lm.open_to_rec=null | 同上游決定 |
| 某市場缺 | rec_snap.{市場}=null | clv.{市場}=null |
| 隊名對不上 | rec_snap=null + warning | clv=null |
| 無 closing snapshot | — | clv=null + `no_closing_snapshot` |
| closing > 4h 舊 | — | clv 照算 + `closing_stale:Nmin` |
| Legacy 無 kelly | — | skip + log |
| 已有 clv 欄位 | — | skip（除非 --force）|
| Doubleheader | 按 --game-index | 按 game match 邏輯 |

**Idempotency：** prediction.json blocks 僅 rec 時寫一次；jsonl 欄位只寫一次除非 --force。所有 cents `round()`、pct `round(_, 2)`。

## 8. Reference 文件更新

### 8.1 `reference/prediction.md`
- 新增「CLV 追蹤」段：recommendation_snapshot + line_movement schema 說明
- 新增「Post-game 欄位」段：predictions.jsonl closing_line / clv / rec_to_close schema
- 明標 4h 粒度 caveat 與 advisory-only 性質

### 8.2 `reference/odds-format.md`
- 新增「CLV cents 約定」段：正值 = beat closing，American cents 計算方式
- 新增「no-vig pct」計算公式範例

### 8.3 `SKILL.md`
- Phase 4 後新增「CLV 追蹤」小段：
  - advisory-only（不動 final.*）
  - 4h snapshot 粒度文檔化
  - post-game 自動由 upload_results.py 補欄位
- 明標與 Kelly block 互補（Kelly = 下注決策；CLV = 事後驗證）

## 9. Testing Strategy

### 9.1 單元測試（`scripts/tests/test_clv.py`）

**目標：≥ 18 cases**

- `compute_clv_cents`：beat +5c / tied 0c / lose -5c × (ML neg / ML pos / OU juice / RL plus)
- `compute_clv_pct_no_vig`：已知 Pinnacle vig 場景（~2%）驗算、零 vig 極端、雙向皆負
- `detect_line_movement`：
  - open 缺 → open_to_rec=null
  - close 缺（rec-time 呼叫）→ rec_to_close=null
  - steam 達閾值 → flag true
  - RLM 達閾值 → flag true
  - 兩 flag 互斥
- `_find_latest_snapshot_before`：多日混合、doubleheader 同日兩時間、空目錄
- `_find_earliest_snapshot_of_date`：同上
- `pin_rec_snapshot`：三市場齊 / 缺 OU / 缺 RL / 缺全 → 對應 null
- `compute_bet_placed`：null / units=0 / units=1.5

### 9.2 整合測試（`scripts/tests/test_clv_integration.py`）

**目標：≥ 5 cases**

- E2E：`predict.py --save` 後 prediction.json 含完整 blocks（close=null）
- E2E：upload_results.py 跑過後 jsonl 出現 clv 欄位
- Idempotency：同一 record 第二次呼叫 → 無變更
- `--force` 覆寫：第二次呼叫 → cents 更新
- Legacy record（無 kelly）→ skip 且 log 正確

### 9.3 Backfill 測試（`scripts/tests/test_backfill_clv.py`）

**目標：≥ 4 cases**

- `--dry-run` 預設：僅印、檔不變
- `--no-dry-run`：verified=true 的補入
- `verified=false`：skip
- Summary 輸出欄位數正確

### 9.4 回歸鎖

P3 既有 39 tests（14 kelly + 11 odds_analyzer_extended + 14 snapshot/integration）在 snapshot loader refactor 後必須全 pass，若破則 refactor 回滾至 clv.py 內複製版本。

### 9.5 邊界手測

- 抽一場比賽手算 American cents，對照 `clv.ml.cents` ±1 容差
- Doubleheader 兩場分別 backfill，確認 game match 正確
- 跨日比賽（跨 UTC 午夜）閉線 lookup 正確

### 9.6 Runner

```bash
python -m pytest scripts/tests/ -v
```

目標：新 + 舊 ≥ 66 tests 全 pass。

## 10. Rollout

1. 新測試 + 新模組（Task checklist 將由 writing-plans 展開）
2. predict.py refactor + 新 block 寫入（P3 tests 必 pass）
3. upload_results.py CLV 寫入
4. backfill_clv.py + 實跑 `analysis-data/2026-04-18/` + commit 更新後 jsonl
5. Reference docs + SKILL.md
6. 最終 commit + DoD check

每段獨立 commit；避免 `git add -A`（memory `feedback_never_commit_sensitive_files.md`）。

## 11. Risk / Known Limits

- **4h 粒度**：Closing line 可能距 first pitch 0-4h，sub-hour steam 偵測不到。文檔化；P2 驗證 500+ bets 後如 ROI 顯示 CLV noise 過大，再評估 T-15min 專屬 fetch。
- **Snapshot refactor 風險**：`_find_latest_snapshot_before` 搬家可能 break P3 14 個 snapshot tests；回滾策略 = 在 clv.py 內複製邏輯、predict.py 不動。
- **Pinnacle-only**：與 P3 一致，不做 cross-book line shopping。
- **PASS 市場 CLV 語意**：`bet_placed=false` 時 CLV 為「模型方向 CLV」，aggregate 報表不含 `bet_placed` filter 會混雜；`reference/prediction.md` 明說明此 convention。
- **Direction NULL (OU NEUTRAL)**：此情境 clv.ou 全 block 為 null，line_movement.ou 相關 flag 不納入 steam/rlm 判定。
- **Legacy records**：P3 上線前的 predictions.jsonl 無 kelly block，backfill 跳過；若將來要補需另寫 heuristic 推出 rec-time snapshot（scope out-of-P2）。

## 12. Definition of Done

- [x] `scripts/clv.py` 所有函式 type-hinted + docstring
- [x] `test_clv.py` ≥ 18 cases 全 pass（實際 34）
- [x] `test_clv_integration.py` ≥ 5 cases 全 pass（實際 7）
- [x] `test_backfill_clv.py` ≥ 4 cases 全 pass（實際 4）
- [x] P3 既有 39 tests 全 pass（regression lock）— 完整套件 84 pass
- [x] `reference/prediction.md` CLV / line_movement / post-game 欄位說明
- [x] `reference/odds-format.md` CLV cents + no-vig pct 定義
- [x] `SKILL.md` 新「CLV 追蹤」段（advisory, 4h caveat）
- [x] Smoke test：pytest 84 pass + rec-path `--save` 對 2026-04-17/KC@NYY/ 驗證寫出 `recommendation_snapshot` + `line_movement`；backfill steps 3-6 無 `predictions.jsonl` 可測（已由 `test_backfill_clv.py` 覆蓋）
- [~] Backfill 對 `analysis-data/2026-04-18/` 實跑 — 2026-04-18 尚無 `predictions.jsonl`（當日比賽尚未完成），推遲至首個 post-game cycle
- [x] 此 spec §12 全部 checkbox 勾選
- [x] Memory `project_p3_p2_p1_roadmap.md` 更新 P2 狀態為 ✅
