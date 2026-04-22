# 常見錯誤與邊界條件

> 合併自舊 SKILL.md 的 Common Mistakes + Edge Cases。SKILL.md 只留一行摘要 + 指到這裡。

---

## Common Mistakes（常見錯誤）

| 錯誤 | 正確做法 |
|------|---------|
| 用訓練資料回答投手數據 | 所有數據必須來自腳本 API 輸出，禁止臆測 |
| WebSearch 失敗後靜默改用記憶中的資料 | 向使用者回報錯誤，等待指示 |
| 跳過 Roster 檢查直接分析投手 | Step 1 閘門必須通過才能進 Step 2 |
| 牛棚傷兵只修正 O/U 未修正 ML | 牛棚傷兵雙向閘門：O/U 和 ML 修正值皆填後才繼續 |
| BvP 樣本 < 15 PA 卻引用結論 | PA < 15 必須丟棄，標註「BvP 樣本不足」 |
| Hot/Cold 判定未檢查 BABIP | BABIP 回歸閘門：極端值預期回歸 ~.300 |
| Phase 3 summary 寫「初步盤口推薦」/ 星級 | 盤口推薦僅在 Phase 4 `prediction.json` 產生；summary 只放基本面，避免 stale |
| Phase 3 結論未存檔就進 Phase 4 | 必須寫入 `phase3_summary.md`，防止對話壓縮遺失 |
| 同場推薦對立方向（如 ML 推 A 隊又推 A 隊受讓） | D3 硬性規則禁止同場對立方向 |
| 用 shell redirect `>` 存腳本輸出 | 必須使用 `--output / -o` 參數 |
| 用 Agent 子代理去跑 WebSearch | 子代理無法存取 WebSearch / WebFetch，必須在主對話中執行 |
| ERA vs xERA 落差僅寫成「風險提示」代替驗證 | 可驗證的現象不得掛成條件性風險。觸發 `|ERA−xERA| ≥ 1.5` 或 `IP<30 且 ERA 低於 prior year ≥1.0` → 必須補跑 `pitcher_stats.py --year {YYYY-1}` 並執行 YoY Statcast 對比（見 `matchup-factors.md#yoy-statcast-驗證`，閘門在 `workflow.md` Phase 2 Step 2） |

---

## Edge Cases（邊界條件）

| 情境 | 處理 |
|------|------|
| 先發臨時更換 | 產生備案分析，重跑 Phase 2 Step 2 |
| Doubleheader | 牛棚消耗累積，G2 對手牛棚可用性必須重估；輸出目錄加 `-G1` / `-G2` 後綴 |
| Opener 策略 | 調整分析框架；Opener 只投 1-2 局，主要戰力由後續 bulk pitcher 提供 |
| Coors Field | **4 月 PF = 112**（非全年 128），5 月後恢復 128。物理依據：4 月丹佛 ~50-60°F，空氣密度比夏季高 ~8-10% |
| 跨聯盟比賽 | BvP 樣本較少，增加不確定性；降低 confidence 一檔 |
| 季後賽 | 得分壓縮 **×0.84-0.86**，先發局數延長 |
| 二次 TJ | **65% RTP**（HSS 研究），42% 能投 10+ 場；復出首年大幅降級 |
| 亞洲盤口格式歧義 | 必須用 ML + 投手分析驗證；見 `odds-format.md` |
| 使用者質疑結果 | 回顧量化信號，獨立驗證後才決定是否修正；不得直接妥協 |
| BABIP 回歸 | 極端值（≤ .260 或 ≥ .370）預期回歸聯盟平均 **~.300**；近 7 天 BABIP 必檢查 |
| 信號修正 vs O/U 差距 < 1.5 | 不推薦（SD ≈ 4.5 run，在噪音範圍內） |

---

## 具體修正係數備忘

| 場景 | 係數 / 數值 | 出處 |
|------|------------|------|
| Coors Field 4 月 PF | 112（非 128） | `matchup-factors.md`「球場 & 天氣」 |
| 季後賽得分壓縮 | ×0.84-0.86 | `prediction.md`「信號 → Run Value 修正表」 |
| 二次 TJ 回歸率 | 65% RTP / 42% 投 10+ 場 | HSS 研究，`matchup-factors.md` |
| 聯盟平均 BABIP | ~.300（需 ~800 AB 穩定） | `matchup-factors.md` |
| MLB 單場隨機性 | ~40-45% | 業界共識 |
| 總分預測 SD | ~4.5 run | `prediction.md` D2 |

> 以上係數已分散在 `matchup-factors.md` / `prediction.md` 的相關章節，此表僅為速查。
