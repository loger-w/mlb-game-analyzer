## 投手對決

### Matthew Boyd (HOME, LHP, 35 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 = —（—），gap vs ERA-only = —
  - <!-- AI 補：是否同意 score-derived tier？若 |gap| ≥ 15 → 簡述運氣 vs 結構性，不自動下修預測 -->
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - <!-- AI 補：若 fired，本場對手核心打者手別組成是否放大此風險？ -->
- **對手打線威脅**：<!-- AI 補：基於 dossier 投手對決表 + 上述兩信號 -->

### Merrill Kelly (AWAY, RHP, 37 📉📉📉 快速退化)
- **Tier 驗證**：腳本 tier_v2 = —（—），gap vs ERA-only = —
  - <!-- AI 補：是否同意 score-derived tier？若 |gap| ≥ 15 → 簡述運氣 vs 結構性，不自動下修預測 -->
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - <!-- AI 補：若 fired，本場對手核心打者手別組成是否放大此風險？ -->
- **對手打線威脅**：<!-- AI 補：基於 dossier 投手對決表 + 上述兩信號 -->

## 打線評級

### HOME — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - <!-- AI 補：matchup tier 與 season tier 落差 → 本場對打線評估方向（同意/上修/下修） -->
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - <!-- AI 補：若 fired，影響本場攻擊 chain 哪一段 → 簡述 -->

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟠 Strong
  - <!-- AI 補：matchup tier 與 season tier 落差 → 本場對打線評估方向（同意/上修/下修） -->
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - <!-- AI 補：若 fired，影響本場攻擊 chain 哪一段 → 簡述 -->

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.81 / 9 / <!-- AI --> | 4.83 / 6 / <!-- AI --> |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：<!-- AI 補：可用性 / 近 3 天消耗 / 對對手末段威脅 -->
- AWAY 牛棚：<!-- AI 補：同上 -->

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=+2.84):
  - <!-- AI 補：是運氣還結構性？是否影響本場判斷？不自動下修預測 -->
- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-5.67):
  - <!-- AI 補：是運氣還結構性？是否影響本場判斷？不自動下修預測 -->

### 額外信號
- 🟠 HOME single-pitch dependent：主球種使用率 52.7%（≥45.0%）
- 🔴 HOME chain breaks at #5-6：OPS 落差 0.310
- 🔴 AWAY chain breaks at #4-5：OPS 落差 0.334
- 🟠 HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - <!-- AI 補：本場是否受此信號影響？是否與 Flag 3/8 雙重壓力 → 1-2 句敘事 -->

## 條件修正

- Park Factor: 92.0 → -0.40 run
- 天氣：Partly Cloudy, 56°F, wind 21 mph, Out To LF
  - 影響判讀：<!-- AI 補：對得分 / HR 影響判讀 -->
- 先發 tier / doubleheader：<!-- AI 補 -->

## 修正後預期得分

> 「+ 信號」欄僅納入規範允許的條件修正：Park Factor、牛棚累計效應（核心 IL ≥ 2 名）、主力打者傷兵。
> ⛔ BABIP 極端值 / ERA-xERA gap **不入此欄**（規範禁止 auto ±run value，見 reference/flags-checklist.md §3, §8）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 9.3 | <!-- AI 補 --> | <!-- AI 補 --> |
| AWAY | 1.8 | <!-- AI 補 --> | <!-- AI 補 --> |
| Total | 11.100000000000001 | <!-- AI 補 --> | <!-- AI 補 --> |

## 整體判斷

- **方向（基本面）**：<!-- AI 補 -->
- **總分（基本面）**：<!-- AI 補 -->
- **信心**：<!-- AI 補 LOW/MEDIUM/HIGH -->
- **風險**：<!-- AI 補 1-4 點 -->

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組