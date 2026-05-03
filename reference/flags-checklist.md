# 旗標清單（Flags Checklist）

8 條分析紀律硬規則：任一條觸發 = 停下來，回到對應步驟。

---

### 1. 用訓練資料/記憶代替腳本 API 輸出
- 觸發：核心數據（ERA/xERA/IP、xwOBA/BABIP、牛棚 ERA）來源不是 `pitcher_stats.py` / `lineup_analyzer.py` / `fetch_game_data.py`
- 處理：腳本失敗 → 向使用者回報，禁止改走 WebSearch / 記憶。詳見 `SKILL.md` §初始化「工具使用規範」

### 2. BvP 樣本 < 15 PA 硬推結論
- 觸發：BvP `PA < 15` 但仍寫成趨勢
- 處理：標註「樣本不足」，不引用。詳見 `matchup-factors.md` §打線分析

### 3. Hot/Cold 判定未檢查 BABIP
- 觸發：近 7 天 BABIP `≤ .260` 或 `≥ .370`
- 處理：腳本（`prepare_game.py`）自動標 ⚠️ 風險提示在 dossier 與 summary 的「## 風險提示」段。AI 在敘事中判讀「可能回歸 / 可能持續」**不自動 ±run value**。詳見 `matchup-factors.md` §BABIP 回歸風險標註

### 4. Agent 子代理跑 WebSearch / WebFetch
- 觸發：dispatch subagent 帶 WebSearch task
- 處理：必須在主對話跑。子代理只能跑純計算腳本

### 5. shell redirect `>` 取代 --output / -o
- 觸發：腳本呼叫用 `>` 寫檔
- 處理：所有腳本必須用 `--output` / `-o`。詳見 `SKILL.md` §初始化「工具使用規範」

### 6. WebSearch 失敗繼續分析
- 觸發：WebSearch error 但仍輸出推薦
- 處理：回報錯誤等使用者指示，禁止「差不多就好」

### 7. 中文對話用英文輸出
- 觸發：使用者中文 → 報告卻是英文
- 處理：報告語言對齊使用者；搜尋可用英文

### 8. ERA-xERA 落差 / 小樣本回歸風險
- 觸發：`|ERA − xERA| ≥ 1.5` 或 `IP < 30 且 ERA 比 prior_year 低 ≥ 1.0`
- 處理：腳本（`prepare_game.py`）自動標 ⚠️ 風險提示在 dossier 與 summary 的「## 風險提示」段。AI 在敘事中判讀「運氣 / 結構性退化 / 樣本噪音」**不自動補跑 YoY、不自動下修預測**

---

## Signals（輔助信號 — 非紀律 Flag）

PR-3（2026-05-03）後新增 `signals_lib`，由 `dossier_renderer` 在 dossier 頂部渲染 `## 🎯 訊號摘要`，並由 `summary_renderer` 在 `## 風險提示` 段尾追加 `### 額外信號`。

**信號與 Flag 的層級差異**：
- **Flag**：硬性紀律。觸發後限制腳本/AI 動作（不自動下修、不自動 ±run value、回報停步等）
- **Signals**：輔助觀察。AI 在 summary 判讀，**不入 scoring formula、不自動 ±run value**

**8 個 signals**（詳細觸發條件與 AI 判讀指引見 `matchup-factors.md` §Signals）：

| Signal | 觸發 | 對應紀律 |
|--------|------|---------|
| tier_mismatch | tier_v2 vs ERA-only gap |≥ 15| 與 Flag 8 同源；不重複進額外信號區 |
| heat_vs_babip | Hot+BABIP≥.350 / Cold+BABIP≤.270 | 與 Flag 3 同源；不重複進額外信號區 |
| platoon_advantage | top 5 中 ≥ 4 人對某手別 OPS 上升 ≥ 0.050 | 純輔助 |
| strong_park | PF ≥ 110 或 ≤ 90 | 純輔助；條件修正一致對待 |
| reverse_platoon | 投手 vs LHB/RHB OPS 反向 \|Δ\| ≥ 0.080 | 純輔助 |
| chain_break | 1-9 棒相鄰 OPS 落差 ≥ 0.150 | 純輔助 |
| pitch_mix_concentration | 主球種使用率 ≥ 45% 或 < 25% | 純輔助 |
| core_il_count | Closer / Setup / High-leverage / Co-Closer IL 計數 | 與 §牛棚傷兵累計效應 對應 |

**邊界**：signals 不違反任何既有 Flag 紀律。`prepare_game._print_risk_notes` stderr 維持只列 Flag 3/8。
