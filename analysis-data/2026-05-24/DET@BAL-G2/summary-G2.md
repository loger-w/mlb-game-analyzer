## 投手對決

### Brandon Young (HOME, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = —（—），gap vs ERA-only = —
  - <!-- AI 補：是否同意 score-derived tier？若 |gap| ≥ 15 → 簡述運氣 vs 結構性，不自動下修預測 -->
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - <!-- AI 補：若 fired，本場對手核心打者手別組成是否放大此風險？ -->
- **對手打線威脅**：<!-- AI 補：基於 dossier 投手對決表 + 上述兩信號 -->

### Framber Valdez (AWAY, LHP, 32 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p72, K-BB% p48），gap vs ERA-only = +20.2
  - <!-- AI 補：是否同意 score-derived tier？若 |gap| ≥ 15 → 簡述運氣 vs 結構性，不自動下修預測 -->
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - <!-- AI 補：若 fired，本場對手核心打者手別組成是否放大此風險？ -->
- **對手打線威脅**：<!-- AI 補：基於 dossier 投手對決表 + 上述兩信號 -->

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟢 Weak
  - <!-- AI 補：matchup tier 與 season tier 落差 → 本場對打線評估方向（同意/上修/下修） -->
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - <!-- AI 補：若 fired，影響本場攻擊 chain 哪一段 → 簡述 -->

### AWAY — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak
  - <!-- AI 補：matchup tier 與 season tier 落差 → 本場對打線評估方向（同意/上修/下修） -->
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - <!-- AI 補：若 fired，影響本場攻擊 chain 哪一段 → 簡述 -->

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.77 / 7 / <!-- AI --> | 3.86 / 10 / <!-- AI --> |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：<!-- AI 補：可用性 / 近 3 天消耗 / 對對手末段威脅 -->
- AWAY 牛棚：<!-- AI 補：同上 -->

## 風險提示

- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.233):
  - <!-- AI 補：可能回歸或可能持續？是否影響本場判斷？不自動 ±run value -->

### 額外信號
- 🟠 AWAY single-pitch dependent：主球種使用率 46.1%（≥45.0%）
- 🟠 HOME chain breaks at #4-5：OPS 落差 0.246
- 🟠 AWAY chain breaks at #4-5：OPS 落差 0.246
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
- 🔴 ⏳ AWAY 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - <!-- AI 補：本場是否受此信號影響？是否與 Flag 3/8 雙重壓力 → 1-2 句敘事 -->

## 條件修正

- Park Factor: 96.0 → -0.20 run
- 天氣：Overcast, 58°F, wind 3 mph, In From CF
  - 影響判讀：<!-- AI 補：對得分 / HR 影響判讀 -->
- 先發 tier / doubleheader：<!-- AI 補 -->

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.0 | <!-- AI 補 --> | <!-- AI 補 --> |
| AWAY | 5.3 | <!-- AI 補 --> | <!-- AI 補 --> |
| Total | 9.3 | <!-- AI 補 --> | <!-- AI 補 --> |

## 整體判斷

- **方向（基本面）**：<!-- AI 補 HOME / AWAY / 持平 -->
- **總分（基本面）**：<!-- AI 補 數值（formula base ± 信號修正後） -->
- **方向信心**：<!-- AI 補 50-75% 機率（≤ 50% 寫「持平」；> 75% 需在風險段說明依據） -->
- **風險**：<!-- AI 補 1-4 點 -->

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組