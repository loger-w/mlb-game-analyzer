## 投手對決

### Dylan Cease (HOME, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +13.8
  - <!-- AI 補：是否同意 score-derived tier？若 |gap| ≥ 15 → 簡述運氣 vs 結構性，不自動下修預測 -->
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - <!-- AI 補：若 fired，本場對手核心打者手別組成是否放大此風險？ -->
- **對手打線威脅**：<!-- AI 補：基於 dossier 投手對決表 + 上述兩信號 -->

### Mitch Keller (AWAY, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p56, K-BB% p54），gap vs ERA-only = +0.9
  - <!-- AI 補：是否同意 score-derived tier？若 |gap| ≥ 15 → 簡述運氣 vs 結構性，不自動下修預測 -->
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - <!-- AI 補：若 fired，本場對手核心打者手別組成是否放大此風險？ -->
- **對手打線威脅**：<!-- AI 補：基於 dossier 投手對決表 + 上述兩信號 -->

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - <!-- AI 補：matchup tier 與 season tier 落差 → 本場對打線評估方向（同意/上修/下修） -->
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - <!-- AI 補：若 fired，影響本場攻擊 chain 哪一段 → 簡述 -->

### AWAY — season tier 🟢 Weak / heat 🥶 Cold
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - <!-- AI 補：matchup tier 與 season tier 落差 → 本場對打線評估方向（同意/上修/下修） -->
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - <!-- AI 補：若 fired，影響本場攻擊 chain 哪一段 → 簡述 -->

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.85 / 9 / <!-- AI --> | 4.19 / 2 / <!-- AI --> |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：<!-- AI 補：可用性 / 近 3 天消耗 / 對對手末段威脅 -->
- AWAY 牛棚：<!-- AI 補：同上 -->

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🔴 HOME TTO3 penalty：OPS Δ +0.459（TTO1 0.396 → TTO3 0.855），第三輪明顯衰退；K% 從 40.4% 掉到 23.0%（Δ -17.4pp）
- 🔴 AWAY TTO3 penalty：OPS Δ +0.373（TTO1 0.403 → TTO3 0.776），第三輪明顯衰退；K% 從 20.2% 掉到 13.5%（Δ -6.7pp）
- 🔴 AWAY chain breaks at #6-7：OPS 落差 0.721
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - <!-- AI 補：本場是否受此信號影響？是否與 Flag 3/8 雙重壓力 → 1-2 句敘事 -->

## 條件修正

- Park Factor: 99.0 → -0.05 run
- 天氣：室內（Roof Closed，不適用）
- 先發 tier / doubleheader：<!-- AI 補 -->

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.6 | <!-- AI 補 --> | <!-- AI 補 --> |
| AWAY | 2.2 | <!-- AI 補 --> | <!-- AI 補 --> |
| Total | 5.800000000000001 | <!-- AI 補 --> | <!-- AI 補 --> |

## 整體判斷

- **方向（基本面）**：<!-- AI 補 HOME / AWAY / 持平 -->
- **總分（基本面）**：<!-- AI 補 數值（formula base ± 信號修正後） -->
- **方向信心**：<!-- AI 補 50-75% 機率（≤ 50% 寫「持平」；> 75% 需在風險段說明依據） -->
- **風險**：<!-- AI 補 1-4 點 -->

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組