## 投手對決

### Nathan Eovaldi (HOME, RHP, 36歲 📉📉)
- **Tier 覆寫**：⚪ Below Average → 沿用腳本判定。本季 ERA 5.79 / FIP 5.55 / WHIP 1.47，K-BB% 15.7 中庸；近 3 場 13 ER / 14.7 IP（ERA 7.97）狀態惡劣。
- 真實水平判斷：xERA 4.68、xFIP 3.24 暗示部分 BABIP 運氣與 HR 高峰可能回歸，「真實」水準比 5.79 ERA 略好；但 hard hit% 32.2 / barrel% 9.5 顯示被擊球品質確實差。年齡 36 + 球速僅 88.0 mph（FS 主球種降速）為結構性退化。**綜合評估：4.50-5.00 ERA 區間更接近真實水平**。
- 對手打線威脅：⚠️ vs LHB **.329/.380/.647**（92 BF）災難級裂痕。NYY 主力打線多數 LHB（Bellinger / Rice / Chisholm / Grisham 皆左打），是本場最強信號。

### Elmer Rodríguez (AWAY, RHP, 22歲 📈)
- **Tier 覆寫**：Unknown → ⚠️ **無 2026 MLB 投球紀錄**（本季首發或大聯盟首登）。
- 真實水平判斷：完全沒有 MLB 樣本，腳本以聯盟均值（FIP 4.5）替代為公式輸入；真實水平可能落在 3.50-6.00 ERA 之間，**變異極大**。22 歲 RHP 成長期。**這是本場最大的不確定性**。
- 對手打線威脅：TEX 打線 🟡 Average / 季節 OPS .704、近 7 天 BABIP .281、streak −4。對未知投手「初見效應」通常先吃虧 1-2 局，但若 Rodríguez 球種有限會被 TEX 修正。

## 打線評級

### HOME — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用。xwOBA .312 略低於聯盟均值；K% 23.6、BB% 10.2。Top 5 中 Josh Jung 近 7 天 1.243 OPS（last7 BABIP .421 — ⚠️ 不可持續），其餘多數 OPS .640-.820 中庸。
- 近 10 場 RS 3.20 + streak −4，**進攻明顯降溫**（攻↓信號）。

### AWAY — 🟠 Strong / 🔥 Hot
- **Tier 覆寫**：沿用。xwOBA .351 / OPS .772 / BB% 13.0（高選球）。Judge OPS 1.037、Rice OPS 1.145、Bellinger .732 但 vs RHP .790；近 10 場 RS 6.30、近 30 場 RS 5.17。
- last7 BABIP .262（偏冷）且仍 Hot → 進攻是真實實力，非 BABIP 噪音。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / IL 數 / 核心 IL 估計 | 2.87 / 6 / 0-1 名核心（多為 Carter Baumler、Chris Martin 等） | 3.86 / 4 / 1 名核心（Carlos Rodón 屬輪值傷兵，非牛棚；Angel Chivilli 中段） |

### 牛棚雙向修正值
- HOME 牛棚：對手 +0.0 run | HOME ML +0%（牛棚是 TEX 強項，ERA 2.87）
- AWAY 牛棚：對手 +0.0 run | AWAY ML 0%（NYY 牛棚 ERA 3.86 中庸但無 2+ 核心傷退）

## 風險提示

⚠️ **Elmer Rodríguez 無 2026 MLB 數據**：腳本以聯盟均值 FIP 4.5 / ERA ~4.50 替代計算 TEX 期望得分（4.6）。實際表現可能落在 3-7 ERA 區間，**模型對 HOME 得分估計不確定性 ±1.5 run**。整體預測信心降至 MEDIUM。

⚠️ **Eovaldi vs LHB .329/.380/.647 災難級**：NYY 4 名主力左打對 RHP，這是本場最可量化的信號（Run Value +0.4-0.6）。

⚠️ **Eovaldi xERA/xFIP 與 ERA 大幅分歧**（5.79 vs 4.68 / 3.24）：暗示部分回歸空間 → AWAY 期望得分微下修 ~0.3-0.5。

## 條件修正

- Park Factor 96.0 → -0.20 run（總分下修）
- 先發 tier：Eovaldi ⚪ Below Average + Rodríguez Unknown → 不適用「雙方先發 Solid+ -0.5」
- 天氣：Globe Life Field 為室內封頂球場，無風雨影響
- 牛棚調整：無 2+ 核心 IL 信號

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (TEX) | 4.6 | -0.3（streak / 進攻冷）-0.2（PF）+0.0（牛棚）-0.5（Rodríguez 不確定下調保守估） | 3.6 |
| AWAY (NYY) | 6.4 | +0.4（vs Eovaldi LHB platoon）-0.4（Eovaldi xFIP 回歸）-0.0（PF 已折入 base） | 6.4 |
| Total | 11.0 | 淨修正 -1.0 | **10.0** |

> **保守估計範圍**：考量 Rodríguez 未知變異，總分區間 8.5-11.5；中位估計 10.0。

## 整體判斷

- **方向（基本面）**：**AWAY (NYY)** 明顯優勢 — Eovaldi vs LHB 災難 + NYY 進攻熱潮 + 連勝動能。Pythagorean (6.4² / (6.4²+3.6²)) ≈ **76% NYY 勝率**，但因 Rodríguez 未知下修至 65-70%。
- **總分（基本面）**：**OVER 8.5** — 修正後總分 10.0，差距 +1.5 run，剛好觸及最低門檻；Rodríguez 變異使信心降低。
- **信心**：**MEDIUM**（受 Rodríguez 未知變異拖累）
- **風險**：
  1. Rodríguez 若大聯盟首登發揮超預期（SS-tier prospect call-up），總分與 NYY 勝率均下修
  2. Eovaldi xFIP 3.24 暗示可能爆發回歸性好投，壓 NYY 火力到 5 分內
  3. TEX 主場 + Globe Life Field 4 月偏壓制（PF 96）
  4. NYY 牛棚 ERA 3.86 中段，後段守備能否守住領先存疑
