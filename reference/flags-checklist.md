# 旗標清單（Flags Checklist）

> 12 條分析紀律硬規則。任一條觸發 = 停下來，回到對應 Phase 閘門。
> 每條僅列觸發條件 + cross-ref；規則完整內容在 canonical 檔。

---

### 1. 用訓練資料/記憶代替腳本 API 輸出
- 觸發：核心數據（ERA/xERA/IP、xwOBA/BABIP、牛棚 ERA）來源不是 `pitcher_stats.py` / `lineup_analyzer.py` / `fetch_game_data.py`
- 處理：腳本失敗 → 向使用者回報，禁止改走 WebSearch / 記憶。詳見 `workflow.md` 初始化「模式切換規範」

### 2. BvP 樣本 < 15 PA 硬推結論
- 觸發：BvP `PA < 15` 但仍寫成趨勢
- 處理：標註「樣本不足」，不引用。詳見 `matchup-factors.md` §打線分析

### 3. Hot/Cold 判定未檢查 BABIP
- 觸發：近 7 天 BABIP `≤ .260` 或 `≥ .370`
- 處理：腳本（`prepare_game.py`）自動標 ⚠️ 風險提示在 dossier 與 phase3_skeleton 的「## 風險提示」段。AI 在敘事中判讀「可能回歸 / 可能持續」**不自動 ±run value**。詳見 `matchup-factors.md` §BABIP 回歸檢查

### 4. 牛棚傷兵只修 O/U 未修 ML
- 觸發：核心（Closer / Primary Setup / High-leverage）IL 但 phase3_summary 缺 ML 修正 (-%) 或 OU 修正 (+run)
- 處理：B9 雙向閘門。詳見 `workflow.md` §Phase 3 §B9

### 5. 同場推對立方向
- 觸發：ML 推 A 隊 + A 隊受讓
- 處理：D3 硬規則。詳見 `prediction.md` §D3

### 6. 不寫 phase3_summary.md 就進 Phase 4
- 觸發：缺 `$GAME_DIR/phase3_summary.md` 但呼叫 `predict.py --save`
- 處理：predict.py 會 reject。詳見 `workflow.md` §Phase 3.5

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

### 13. ERA-xERA 落差 / 小樣本回歸風險
- 觸發：`|ERA − xERA| ≥ 1.5` 或 `IP < 30 且 ERA 比 prior_year 低 ≥ 1.0`
- 處理：腳本（`prepare_game.py`）自動標 ⚠️ 風險提示在 dossier 與 phase3_skeleton 的「## 風險提示」段。AI 在敘事中判讀「運氣 / 結構性退化 / 樣本噪音」**不自動補跑 YoY、不自動下修預測**

---

## 使用方式

每條規則均可透過 Phase 閘門自檢。完整 Phase 順序見 `workflow.md`；觸發時的補救動作見對應 Phase section 與 cross-ref 檔。
