# 旗標清單（Flags Checklist）

> 13 條分析紀律硬規則。任一條觸發 = 停下來，回到對應 Phase 閘門。

## 紀律來源

合併自舊 `pitfalls.md` Common Mistakes（12 條）+ SKILL.md Rationalizations / Red Flags（去重後加 1 條「語言」）。每條對應一個可立即停手的違規情境。

---

## 旗標

### 1. 用訓練資料/記憶代替腳本 API 輸出
所有核心數據（投手 ERA/xERA/IP、打者 wOBA/xwOBA、牛棚 ERA、BABIP）必須來自 `pitcher_stats.py` / `lineup_analyzer.py` / `fetch_game_data.py` 輸出。記憶與訓練資料不是來源。腳本失敗 → 回報錯誤等使用者指示，禁止靜默改走 WebSearch 或記憶。

### 2. BvP 樣本 <15 PA 硬推結論
PA < 15 是雜訊。標註「BvP 樣本不足」，不得引用趨勢。

### 3. Hot/Cold 判定未檢查 BABIP
近 7 天 BABIP 極端值（≤ .260 或 ≥ .370）預期回歸聯盟平均 ~.300。未檢查 = Hot/Cold 判定無效。

### 4. 牛棚傷兵只修 O/U 未修 ML
牛棚雙向閘門：偵測核心（Closer / Primary Setup / High-leverage）IL → 同時寫入 O/U 修正（+run）**和** ML 修正（-%）。只修一側 = Phase 3 未完成。

### 5. 同場推對立方向
D3 硬性規則：ML 推 A 隊 + A 隊受讓被禁。訊號方向 = 勝率強度，不是「兩邊下注對沖」。

### 6. 不寫 `phase3_summary.md` 就進 Phase 4
Phase 3 結論必須寫入 `$GAME_DIR/phase3_summary.md`。對話壓縮後結論會遺失 → Phase 4 就用記憶預測。

### 7. 跳過 Roster 檢查
Phase 2 Step 1 是阻塞閘門。IL 遺漏 → Phase 3 牛棚傷兵基礎就錯。

### 8. Agent 子代理跑 WebSearch / WebFetch
子代理無 WebSearch / WebFetch 權限，輸出是幻想。必須在主對話執行。平行可跑純計算腳本。

### 9. 省 `--game-data` / 腦補路徑
唯一合法命令：`predict.py --game-data analysis-data/<date>/<AWAY>@<HOME>/merged.json --save`。`prediction.json` 自動落在同層。省略或腦補 = predict.py 報錯或寫錯位置。

### 10. shell redirect `>` 取代 `--output` / `-o`
所有腳本必須用 `--output` / `-o`。`>` 會吃掉腳本 stderr + 破壞 pybaseball 的互動訊息輸出。

### 11. WebSearch 失敗繼續分析
WebSearch 失敗 → 向使用者回報錯誤，等待指示。不得「差不多就好」續推。

### 12. 中文對話用英文輸出
使用者中文 → 輸出必須繁體中文。搜尋可先用英文；**報告輸出照用戶語言**。

### 13. ERA-xERA 落差 ≥ 1.5 僅寫「風險提示」
可驗證的現象不得掛成條件性風險。觸發 `|ERA − xERA| ≥ 1.5` 或 `IP < 30 且 ERA 低於 prior year ≥ 1.0` → 必須補跑 `pitcher_stats.py --year {YYYY-1} -o $GAME_DIR/{side}_pitcher_{YYYY-1}.json` 並執行 YoY Statcast 對比。閘門在 `workflow.md` Phase 2 Step 2；方法見 `matchup-factors.md#yoy-statcast-驗證`。

---

## 使用方式

每條規則均可透過 Phase 閘門自檢。完整 Phase 順序見 `workflow.md`；觸發時的補救動作見對應 Phase section。
