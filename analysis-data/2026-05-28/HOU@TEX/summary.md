## 投手對決

### Nathan Eovaldi (HOME, RHP, 36 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p89），gap vs ERA-only = +21.6
  - <!-- AI 補：是否同意 score-derived tier？若 |gap| ≥ 15 → 簡述運氣 vs 結構性，不自動下修預測 -->
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - <!-- AI 補：若 fired，本場對手核心打者手別組成是否放大此風險？ -->
- **對手打線威脅**：<!-- AI 補：基於 dossier 投手對決表 + 上述兩信號 -->

### Spencer Arrighetti (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟢 Back-end Starter（xFIP p29, K-BB% p40），gap vs ERA-only = -46.6
  - <!-- AI 補：是否同意 score-derived tier？若 |gap| ≥ 15 → 簡述運氣 vs 結構性，不自動下修預測 -->
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - <!-- AI 補：若 fired，本場對手核心打者手別組成是否放大此風險？ -->
- **對手打線威脅**：<!-- AI 補：基於 dossier 投手對決表 + 上述兩信號 -->

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - <!-- AI 補：matchup tier 與 season tier 落差 → 本場對打線評估方向（同意/上修/下修） -->
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - <!-- AI 補：若 fired，影響本場攻擊 chain 哪一段 → 簡述 -->

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - <!-- AI 補：matchup tier 與 season tier 落差 → 本場對打線評估方向（同意/上修/下修） -->
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - <!-- AI 補：若 fired，影響本場攻擊 chain 哪一段 → 簡述 -->

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.14 / 5 / <!-- AI --> | 5.46 / 8 / <!-- AI --> |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：<!-- AI 補：可用性 / 近 3 天消耗 / 對對手末段威脅 -->
- AWAY 牛棚：<!-- AI 補：同上 -->

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-3.46):
  - <!-- AI 補：是運氣還結構性？是否影響本場判斷？不自動下修預測 -->
- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.253):
  - <!-- AI 補：可能回歸或可能持續？是否影響本場判斷？不自動 ±run value -->
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.211):
  - <!-- AI 補：可能回歸或可能持續？是否影響本場判斷？不自動 ±run value -->

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ +0.003（TTO1 0.781 → TTO3 0.784），第三輪明顯衰退；K% 從 27.7% 掉到 22.2%（Δ -5.5pp）
- 🟠 AWAY TTO3 penalty：OPS Δ +0.134（TTO1 0.533 → TTO3 0.667），第三輪明顯衰退
- 🟠 HOME chain breaks at #6-7：OPS 落差 0.204
- 🟠 AWAY chain breaks at #1-2：OPS 落差 0.236
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
- 🟠 ⏳ AWAY 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - <!-- AI 補：本場是否受此信號影響？是否與 Flag 3/8 雙重壓力 → 1-2 句敘事 -->

## 條件修正

- Park Factor: 96.0 → -0.20 run
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：<!-- AI 補 -->

## 修正後預期得分

> v1：信號只進敘事、不進數字（+信號 欄一律 0、adjusted = base）。
> 哪個信號該進數字由未來 ablation 決定（見 spec §10）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.3 | 0 | 3.3 |
| AWAY | 3.9 | 0 | 3.9 |
| Total | 7.2 | 0 | 7.2 |

## 整體判斷

- **方向（基本面）**：AWAY
- **總分（基本面）**：7.2
- **方向信心**：56%（LOW）
- **風險**：<!-- AI 補 1-4 點 -->

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
ℹ️ 方向/總分/信心由 scripts/predict.py 確定性計算；AI 僅補風險敘事，不得改數字。