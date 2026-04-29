# Snapshot loader 改為按內容過濾 ET 日期

日期:2026-04-29
分支:`fix/snapshot-loader-cross-day`

## 問題

`odds/lib/snapshot_loader.py:46-71` 的 `load_snapshots_for_et_date(et_date, snapshot_dir)` 用 `glob(f"{et_date}_*-ET.json")` 過濾日期,**只看檔名不看內容**。

當一份 snapshot 跨日(例如 ET 4/28 21:00 fetch 抓到 4/29 開盤的 14 場),這份檔在 `--date 2026-04-29` 分析時會被忽略,造成 silent data loss——4/29 各場的 line movement timeline 會丟掉「ET 跨日前最後一份開盤錨點」,reverse line movement / steam move 的判讀少一個參考點,且使用者不會察覺。

實際發生情境(2026-04-29 觀察到):
- snapshot 檔 `2026-04-28_21-00-ET.json` 含 26 場(12 場 4/28 + 14 場 4/29)
- 跑 `analyze_smart_money.py --date 2026-04-29` 輸出「INFO 2026-04-29 無 snapshot」
- 14 場 4/29 的 Pinnacle 開盤資料完全被略過

## 修法

**改 loader 不再依賴檔名過濾,改讀全部 snapshot 檔,把日期過濾完全交給下游。**

下游 `collect_game_timeline`(`odds/lib/snapshot_loader.py:90`)已經有場次層的 `g["game_date_et"] != game_date_et` 過濾,所以 loader 不再需要做日期判斷,只要把目錄內所有合法 snapshot 都產出即可。

根因是「用錯訊號」(filename 而非內容),修法是「拿掉錯訊號」而非「換另一個寬鬆的檔名訊號」。不加檔名日期窗口優化(YAGNI;一季 1000 份小 JSON 解析仍 < 1 秒)。

## 變更點

### `odds/lib/snapshot_loader.py:46-71` — `load_snapshots_for_et_date`

- glob 從 `f"{et_date}_*-ET.json"` 改成 `"*-ET.json"`(維持 ET 後綴慣例,排除非 snapshot 檔)
- 函式名稱與簽章**不改**(保留 `et_date` 參數),避免擴大呼叫端 diff;參數語意改為「下游過濾用的目標日期」,當前實作不消費它
- docstring 更新:不再提「按 et_date 篩檔」,改為「讀目錄下所有 snapshot,日期過濾交給 `collect_game_timeline`」

### `odds/tests/test_snapshot_loader.py` — 更新斷言

舊測試假設「filename 前綴=過濾條件」,新邏輯下行為改變:

| 舊斷言 | 新斷言 |
|---|---|
| `load_snapshots_for_et_date("2026-04-27", FIXTURES)` 回 2 份 | 回 3 份(全部 fixture) |
| `load_snapshots_for_et_date("2026-04-26", FIXTURES)` 回 1 份 | 回 3 份 |
| `load_snapshots_for_et_date("2099-01-01", FIXTURES)` 回 0 份 | 回 3 份 |

「不存在日期的場次過濾」邏輯改在 `collect_game_timeline` 層級驗證:對 `"2099-01-01"` 呼叫 `collect_game_timeline` 應回空 dict。

### 新增 fixture

新增 1 份 `odds/tests/fixtures/2026-04-28_21-00-ET.json`,內含:
- 1 場 `game_date_et = 2026-04-28`(完整 Pinnacle 盤口)
- 1 場 `game_date_et = 2026-04-29`(完整 Pinnacle 盤口)

格式對齊現有 fixture(snapshot wrapper + games array + bookmakers.pinnacle.ml/ou/rl)。

### 新增測試

`test_cross_day_snapshot_loaded_for_later_date`:

```
1. snapshots = load_snapshots_for_et_date("2026-04-29", FIXTURES)
2. timelines = collect_game_timeline(snapshots, "2026-04-29")
3. assert 該 4/29 的場次出現在 timelines
4. assert 4/28 的場次不在 timelines(被 collect_game_timeline 場次層過濾掉)
```

## 不做的事

- **不加**檔名日期窗口優化
- **不改** `collect_game_timeline` 的場次過濾邏輯
- **不改** fetcher 的命名與寫檔行為
- **不重命名**函式或參數

## 風險與緩解

- **既有 fixtures 跨日污染**:已驗證 `2026-04-26_20-00-ET.json` / `2026-04-27_*-ET.json` 三份檔內 `game_date_et` 各只含對應日期,新邏輯下不會出現意外讀入。
- **歷史資料**:目前 `odds_snapshots/` 只有 1 份檔(2026-04-29 觀察時),沒有跨季累積疑慮。

## 測試命令

```
pytest odds/tests/test_snapshot_loader.py -v
```

預期:既有測試斷言修正後通過 + 新測試通過。
