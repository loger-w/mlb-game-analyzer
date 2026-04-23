# MLB 預測累計回顧

> 最後更新：2026-04-22

## 累計戰績

| 盤口 | 戰績 | 勝率 | 備註 |
|------|------|------|------|
| ML | 12W-15L | 44.4% | 27 推薦，38 PASS，65 場驗證 |
| O/U | 5W-8L | 38.5% | 13 推薦，52 PASS |
| RL（腳本 `run_line_rec`） | 5W-7L | 41.7% | 4/21 首日整合後 1W-4L，4/22 加 4W-3L 回到 41.7% |
| RL-1b 放寬 off-book（4/20 手動） | 4W-1L | 80.0% | 僅含 4/20 手動回測；4/21 起改走主線 `run_line_rec` |

## 星級拆分（ML）

| 星級 | 戰績 | 勝率 |
|------|------|------|
| 1★ | 1W-3L | 25.0% |
| 2★ | 7W-11L | 38.9% |
| 3★ | 2W-1L | 66.7% |
| 4★ | 2W-0L | 100.0% |

> Plan B 下沉前（4/18–4/21）ml_stars 分佈集中在 1–2★；4/22 首次出現 3★ 3 筆（STL MIA、HOU CLE、PHI CHC）與 4★ 2 筆（ATL WSH、NYY BOS），為星級階梯驗證首日樣本。

## Tag 觀察

> 不拆表格，用文字記錄 pattern。樣本足夠時再考慮量化。

- `early-season`（19 場，9W-10L，47.4%）：4/22 新增 5 場 3W-2L，勝率從 42.9% 回升至 47.4%；樣本累積但變異仍大。
- `insufficient-sample`（8 場，5W-3L，62.5%）：4/22 未新增（NYY@BOS 今日仍為 4★ 但 tag 轉為 `nyy_babip_regression_up`）；保持 62.5%。
- `divergent`（6 場，2W-4L）：4/22 新增 2 場（推薦場 2W-0L）——首次有 `divergent` 推薦獲勝；cumulative 從 0W-4L 改善至 2W-4L（33.3%），Plan B 3.3 的 `divergent` tag stars cap=2 可能讓殘留 2★ 意外擊出。
- `divergent-xera`（2 場，2W-0L）：4/22 未新增，維持 100%。
- `away-pitching-slump`（7 場，4W-3L，57.1%）：4/22 新增 4 場 3W-1L（STL@MIA、PIT@TEX、PHI@CHC 皆 WIN，LAD@SF LOSS），從 50% 跳到 57.1%；是今日最強 tag。
- `away-hot-offense`（4 場，0W-4L）：4/22 新增 1 場 LOSS（LAD@SF），cumulative 連 4 場全敗，惡化趨勢持續；警訊 tag。
- `home-bullpen-slump`（6 場，2W-4L，33.3%）：4/22 新增 4 場 2W-2L，從 0W-2L 回到 33.3%，但仍勝率偏低。
- `home-pitching-slump`（6 場，3W-3L，50.0%）：4/22 新增 3 場 1W-2L（含 HOU@CLE CLE 3★ 主場先發弱化反被零封的反例）。
- `direction-override`（2 場，0W-2L；4/22 新增 LAD@SF 1 場 LOSS）：**觸發反覆門檻（3 天 / 3 場全敗）**，列入反覆性問題。
- `nym-losing-streak`（1 場，0W-1L）：4/22 新增（MIN@NYM MIN 1★ 客場 LOSS）。
- `bullpen-gap`（3 場，0W-3L）：4/22 新增 2 場全敗。
- `bullpen-il`（2 場，2W-0L）：4/22 新增 2 場全勝（STL@MIA、PIT@TEX），100%。
- `coors`（1 場，0W-1L）：4/22 新增 SD@COL 1 場 LOSS（SD 2★ + RL + PASS OU 方向錯），cumulative 連 2 天 Coors 場推薦全敗。
- `platoon-edge` / `pitcher-gap` / `rookie-start` / `lhp_vs_lhp`（均 1 場 1W-0L）：4/22 新增，各勝 1 場。
- `fenway_pf105`（1 場，1W-0L）：4/22 NYY@BOS 新增。

## 問題追蹤

### 狀態定義
- **假設**：觀察到現象，樣本不足，不做結論
- **待確認**：連續 3 天出現 或 累積 15 筆相關推薦（觸發任一）
- **已確認**：累積 30 筆相關推薦，統計上顯著

### 結構性問題追蹤

| # | 問題 | 狀態 | 首次出現 | 出現次數 | 相關推薦數 | 備註 |
|---|------|------|---------|---------|-----------|------|
| 1 | 主場 2★/3★ 推薦持續失靈 | **觀察中（Plan B 3.4 Y-new-1 tag）**，4/22 延伸至 3★ | 2026-04-18 | **5** | 10 | 4/18–4/21 主場 2★ 2W-7L；4/22 HOU@CLE CLE 3★ LOSS 為 Plan B 下沉後首例 3★ 慘敗。`home-2star-risk` audit 範圍可能需擴至 `ml_stars>=2`。 |
| 2 | OU 方向不一致（PASS 場常嚴重 OVER/UNDER） | **待確認**，連 5 天觸發 | 2026-04-18 | **5** | 52 | PASS 52 場中今日 5 場誤差 ≥3，方向從單向 UNDER 轉為雙向（3 UNDER / 2 OVER）；Plan B 未觸及 OU 層，需專案化處理。 |
| 3 | 近身戰（差距 < 0.5）仍出 2★ | **已修復（Plan B 3.2 Y-new-2）** | 2026-04-18 | 4 | 4 | 4/22 無新觸發，Plan B code 下沉生效。 |
| 4 | `divergent` 標籤推薦場全輸 | **已修復（Plan B 3.3 Y-new-3）** | 2026-04-18 | 3 | 6 | 4/22 新增 `divergent` 推薦 2 場 2W-0L（首次改善），cap=2 後殘留 2★ 命中。 |
| 5 | ML 推薦與 predicted_winner 不一致未攔截 | 假設 | 2026-04-19 | 1 | 1 | 4/22 無新案例，Plan B 攔截可能生效。 |
| 6 | 冷天氣訊號量級不足 | 假設 | 2026-04-19 | 2 | 2 | 4/22 無冷天氣場。 |
| 7 | 小樣本 xERA 訊號權重過高 | 假設 | 2026-04-19 | 1 | 1 | 4/22 無新案例。 |
| 8 | xgb_raw 與 predicted_winner 內部矛盾 | **已修復（Plan B 3.1 Y2）** | 2026-04-20 | 3 | 3 | 4/22 無新觸發；需查 prediction.json 的 `xgb-predicted-divergent` audit tag 計數確認實際攔截次數。 |
| 9 | `ml_rec` 存字面值 `HOME`/`AWAY` 導致 `judge_ml` 誤判 | **已修復（Plan B 2.2 W2）** | 2026-04-21 | 1 | 1 | 4/22 15 場 ml_rec 全部合法縮寫或 PASS，schema 驗證生效。 |
| 10 | 人工 `--run-line-rec` 完全繞過 RL-1b gate 門檻 | **已修復（Plan B 2.1 W1）** | 2026-04-21 | 1 | 3 | 4/22 無人工 rec，全走 auto override；CLI args 已廢除。 |
| 11 | `signal_adjustments` 使 `predicted_*` 與 `formula_*` 方向反向 | 假設 | 2026-04-21 | 1 | 1 | 4/22 無新案例（今日 SD@COL 方向錯但為 COL 公式直接看 COL 勝，非 signal 反轉）。 |
| 12 | **新：Coors 場 park_factor 雙向變異** | 假設 | 2026-04-21 | 2 | 2 | 4/21 SD@COL 嚴重高估 10 分、4/22 SD@COL 方向反且 totals 差 1.3；4 月 Coors 已見 4 場（4/19–4/22），建議加 `coors_total_band_widen` 或 OU 強制 PASS 除非差距 ≥3。**4/22 SD@COL phase3 MEDIUM 信心（SD 投手差 2 檔 + Sugano xERA 7.86 回歸 + SD 牛棚 IL）預測 SD 大勝 3.3 分仍方向反向，顯示 Coors 短期變異會壓過基本面強訊號，phase3 信心程度與 Coors 場實際結果脫鉤。** |
| 13 | **新：CIN@TB 對戰組合連 2 天方向錯** | 假設（極端異常） | 2026-04-21 | 2 | 2 | 4/21 預 TB 0.7 險勝實 CIN 6 分勝、4/22 預 CIN 0.3 險勝實 TB 6 分零封；TB 主場先發小樣本 + CIN 打線變異同源但方向完全反轉，列入獨立觀察。 |
| 14 | **新：`fetch_results.py` jsonl 未同步 actuals 導致覆寫 per-game** | **已修復（2026-04-22）** | 2026-04-22 | 1 | — | main() 順序 bug：Step 2 寫 actuals 到 per-game，但 Step 3–5 從舊 jsonl 讀記錄、更新 result codes 後寫回 per-game 覆蓋。修正：新增 `merge_scores_into_records()` 在 update_records 前把比分也 merge 進 records。 |
| 15 | **新：Plan B 下沉後星級階梯分離（4★/3★ 高、1-2★ 低）** | 假設 | 2026-04-22 | 1 | 15 | 4/22 首日 ML 4★ 2W-0L（100%）、3★ 2W-1L（66.7%）、2★ 1W-2L（33.3%）、1★ 1W-2L（33.3%）出現明顯分離。Plan B 前 cumulative 2★ 38.9% 與 3★ 66.7% 差距較小。假設 Plan B 降級機制把「弱 2★」往下壓至 1★ 或 PASS，殘留 2★ 反而是 Plan B 沒擋下的類型（可能訊號更薄）。樣本 1 天 15 筆推薦，需續觀察 4/23–4/25。 |

### 反覆性問題（滑動窗口：5 天內 >= 3 次）

| 問題 | 最近 5 天出現日期 | 次數 | 狀態 |
|------|-----------------|------|------|
| 主場 2★/3★ 失靈 | 2026-04-18、04-19、04-21、04-22 | 4 | 反覆中（4/22 延伸至 3★） |
| `divergent` 標籤推薦場全輸 | 2026-04-18、04-19、04-20 | 3 | **已緩解**（4/22 推薦場 2W-0L） |
| PASS 場次 OU 嚴重低估（誤差 ≥3） | 2026-04-18、04-19、04-20、04-21、04-22 | 5 | 反覆中（方向從單向轉雙向） |
| Coors 場預測高變異 | 2026-04-19、04-20、04-21、04-22 | 4 | 反覆中 |
| `direction-override` tag 連敗 | 2026-04-20、04-21、04-22 | 3 | **新增：反覆中**（cumulative 0W-3L） |
| 近身戰 2★（差距 <0.5 仍推 2★） | 2026-04-18、04-19、04-20、04-21 | 4 | 已緩解（4/22 無新案例，Plan B 生效） |

---

## 備註

- 2026-04-18 為 schema 分界日（pre-4/18 `signal_adjustments` 為空），累計追蹤自此日開始為可橫向比較的乾淨樣本。
- 4/14–4/17 四天無驗證資料（gap）；更早於 4/14 的紀錄在舊 schema 下，未納入此 cumulative。
- 2026-04-22 為 Plan B 三層防線下沉首日：問題 #1/#3/#4/#8/#9/#10 均在此日下沉 code；首日 ML overall 54.5%（vs Plan B 前 cumulative 37.5%），4★ 2W-0L 首次出現星級階梯分離。樣本 1 天不足以下強結論，需看 4/23–4/25 續表現。
- 2026-04-22 發現並修復 `fetch_results.py` 雙寫順序 bug（問題 #14）：Step 2 寫 actual_* 到 per-game 後，Step 5 把未同步的 jsonl records 寫回 per-game 覆蓋；修正後 15 筆 result 正常生成。
- 4/22 RL 從累計 1W-4L（20%）跳到 5W-7L（41.7%），大分差路徑表現穩定；RL-1b 自動 override 在 xgb/formula 一致方向時可靠，mid-diff + tag 路徑樣本仍不足。
- 4/22 首次出現 3★/4★ 多筆樣本：3★ 2W-1L（HOU@CLE LOSS 為主場 3★ 首例擊穿）、4★ 2W-0L（ATL@WSH、NYY@BOS），星級階梯首次驗證。
