## 投手對決

### Davis Martin (HOME, RHP, 29 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = -0.2
  - <!-- AI 補：是否同意 score-derived tier？若 |gap| ≥ 15 → 簡述運氣 vs 結構性，不自動下修預測 -->
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - <!-- AI 補：若 fired，本場對手核心打者手別組成是否放大此風險？ -->
- **對手打線威脅**：<!-- AI 補：基於 dossier 投手對決表 + 上述兩信號 -->

### Kendry Rojas (AWAY, LHP, 23 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = —（—），gap vs ERA-only = —
  - <!-- AI 補：是否同意 score-derived tier？若 |gap| ≥ 15 → 簡述運氣 vs 結構性，不自動下修預測 -->
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - <!-- AI 補：若 fired，本場對手核心打者手別組成是否放大此風險？ -->
- **對手打線威脅**：<!-- AI 補：基於 dossier 投手對決表 + 上述兩信號 -->

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟡 Average
  - <!-- AI 補：matchup tier 與 season tier 落差 → 本場對打線評估方向（同意/上修/下修） -->
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - <!-- AI 補：若 fired，影響本場攻擊 chain 哪一段 → 簡述 -->

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - <!-- AI 補：matchup tier 與 season tier 落差 → 本場對打線評估方向（同意/上修/下修） -->
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - <!-- AI 補：若 fired，影響本場攻擊 chain 哪一段 → 簡述 -->

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.73 / 6 / <!-- AI --> | 4.94 / 6 / <!-- AI --> |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：<!-- AI 補：可用性 / 近 3 天消耗 / 對對手末段威脅 -->
- AWAY 牛棚：<!-- AI 補：同上 -->

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-1.51):
  - <!-- AI 補：是運氣還結構性？是否影響本場判斷？不自動下修預測 -->
- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-2.40):
  - <!-- AI 補：是運氣還結構性？是否影響本場判斷？不自動下修預測 -->

### 額外信號
- 🟠 HOME reverse platoon Δ +0.136（vs RHB OPS 0.688 > vs LHB OPS 0.552）— RHP 對非預期手別反而吃虧
- 🔴 HOME TTO3 penalty：OPS Δ +0.282（TTO1 0.443 → TTO3 0.725），第三輪明顯衰退；K% 從 31.1% 掉到 18.0%（Δ -13.1pp）
- 🔴 HOME chain breaks at #8-9：OPS 落差 0.366
- 🔴 AWAY chain breaks at #6-7：OPS 落差 0.545
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
- 🟠 ⏳ AWAY 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - <!-- AI 補：本場是否受此信號影響？是否與 Flag 3/8 雙重壓力 → 1-2 句敘事 -->

## 條件修正

- Park Factor: 97.0 → -0.15 run
- 天氣：Sunny, 62°F, wind 16 mph, Out To RF
  - 影響判讀：<!-- AI 補：對得分 / HR 影響判讀 -->
- 先發 tier / doubleheader：<!-- AI 補 -->

## 修正後預期得分

> v1：信號只進敘事、不進數字（+信號 欄一律 0、adjusted = base）。
> 哪個信號該進數字由未來 ablation 決定（見 spec §10）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.4 | 0 | 3.4 |
| AWAY | 1.7 | 0 | 1.7 |
| Total | 5.1 | 0 | 5.1 |

## 整體判斷

- **方向（基本面）**：HOME
- **總分（基本面）**：5.1
- **方向信心**：66%（MEDIUM）
- **風險**：<!-- AI 補 1-4 點 -->

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
ℹ️ 方向/總分/信心由 scripts/predict.py 確定性計算；AI 僅補風險敘事，不得改數字。