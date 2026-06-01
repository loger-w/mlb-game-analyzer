## 投手對決

### Payton Tolle (HOME, LHP, 23 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +5.6
  - <!-- AI 補：是否同意 score-derived tier？若 |gap| ≥ 15 → 簡述運氣 vs 結構性，不自動下修預測 -->
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - <!-- AI 補：若 fired，本場對手核心打者手別組成是否放大此風險？ -->
- **對手打線威脅**：<!-- AI 補：基於 dossier 投手對決表 + 上述兩信號 -->

### Chris Sale (AWAY, LHP, 37 📉📉📉 快速退化)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = -5.4
  - <!-- AI 補：是否同意 score-derived tier？若 |gap| ≥ 15 → 簡述運氣 vs 結構性，不自動下修預測 -->
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - <!-- AI 補：若 fired，本場對手核心打者手別組成是否放大此風險？ -->
- **對手打線威脅**：<!-- AI 補：基於 dossier 投手對決表 + 上述兩信號 -->

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟠 Strong
  - <!-- AI 補：matchup tier 與 season tier 落差 → 本場對打線評估方向（同意/上修/下修） -->
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - <!-- AI 補：若 fired，影響本場攻擊 chain 哪一段 → 簡述 -->

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - <!-- AI 補：matchup tier 與 season tier 落差 → 本場對打線評估方向（同意/上修/下修） -->
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - <!-- AI 補：若 fired，影響本場攻擊 chain 哪一段 → 簡述 -->

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.06 / 5 / <!-- AI --> | 3.16 / 6 / <!-- AI --> |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：<!-- AI 補：可用性 / 近 3 天消耗 / 對對手末段威脅 -->
- AWAY 牛棚：<!-- AI 補：同上 -->

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🔴 HOME reverse platoon Δ +0.227（vs LHB OPS 0.662 > vs RHB OPS 0.435）— LHP 對非預期手別反而吃虧
- 🟠 HOME TTO3 penalty：OPS Δ +-0.401（TTO1 0.566 → TTO3 0.165），第三輪明顯衰退；K% 從 33.3% 掉到 25.8%（Δ -7.5pp）
- 🟠 HOME chain breaks at #4-5：OPS 落差 0.214
- 🟠 AWAY chain breaks at #8-9：OPS 落差 0.233
- 🟠 ⏳ AWAY 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - <!-- AI 補：本場是否受此信號影響？是否與 Flag 3/8 雙重壓力 → 1-2 句敘事 -->

## 條件修正

- Park Factor: 104.0 → +0.20 run
- 天氣：Partly Cloudy, 67°F, wind 9 mph, In From CF
  - 影響判讀：<!-- AI 補：對得分 / HR 影響判讀 -->
- 先發 tier / doubleheader：<!-- AI 補 -->

## 修正後預期得分

> v1：信號只進敘事、不進數字（+信號 欄一律 0、adjusted = base）。
> 哪個信號該進數字由未來 ablation 決定（見 spec §10）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 2.8 | 0 | 2.8 |
| AWAY | 2.7 | 0 | 2.7 |
| Total | 5.5 | 0 | 5.5 |

## 整體判斷

- **方向（基本面）**：持平
- **總分（基本面）**：5.5
- **方向信心**：51%（持平）
- **風險**：<!-- AI 補 1-4 點 -->

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
ℹ️ 方向/總分/信心由 scripts/predict.py 確定性計算；AI 僅補風險敘事，不得改數字。