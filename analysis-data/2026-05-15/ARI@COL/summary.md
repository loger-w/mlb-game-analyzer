## 投手對決

### Kyle Freeland (HOME, LHP, 33 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p95, K-BB% p76），gap vs ERA-only = **+69.4**（極大）
  - **嚴重不同意**：ERA **6.00** / xERA 5.19 / FIP 5.07 / xFIP 3.29 / K-BB% 14.7% — tier_v2 看 xFIP 3.29 抬升到 Strong Ace 是極端誤判（xFIP 過度信任）。實際 vs LHB **.292/.357/.542** + vs RHB **.293/.352/.505** 雙邊都被打 OPS .857-.899，**真實水平 ⚪ Below Average**（ERA 5.5+ 區間）。
  - **本場按 ⚪ Below Average 對待**，特別 Coors 主場放大失分。
- **對手打線威脅**：🟠 高。ARI matchup tier 🟢 Weak (vs LHP) 但 Carroll vs LHP **1.107** last7 1.074 + Vargas vs LHP **1.020** — 個別 anchor 對 Freeland 是真威脅。

### Merrill Kelly (AWAY, RHP, 37 📉📉📉 快速退化)
- **Tier 驗證**：腳本 tier_v2 沒給定（GS 5 / 14.7 IP）。原始 tier ⚪ Below Average — ERA **7.62** / xERA **9.93** / FIP **6.64** / K-BB% **1.6%**（極端低）/ vs LHB **.345/.457/.759 SLG**（70 BF）— 全方位崩盤，37 歲快速退化中。**近 3 場 ER 15 / IP 14.7** 區間 ERA 9.20。
  - **本場按 ⚪ Below Average** 對待，特別 Coors 主場是 Kelly 噩夢配置。
- **TTO 反向**：OPS Δ **-0.508**（TTO1 **1.142** → TTO3 .634）— 首輪嚴重被打但 TTO3 K% 反而提升，但這是小樣本（34 BF）失真。
- **對手打線威脅**：🟠 高。COL matchup tier 🟡 Average (vs RHP) — Rumfield vs RHP **.915** last7 1.138 / Goodman .806 / Johnston **.960** last7 .900 — 中段火力齊備，配合 Coors 場效應放大。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致；對 Kelly ⚪ Below Average + Coors 主場 → 進攻面強放大。

### AWAY — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak — 比 season tier 下修一檔；但 Carroll/Vargas vs LHP individual anchor 真實。
- **Flag 3 last7 BABIP .195** — 極冷期（見風險段）。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.37 / 5 / **1 名（🟠 中高，Herget IL15d）** | 4.38 / 6 / **3 名（🔴🔴 極高，Puk + Saalfrank + 1）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（COL）：ERA 4.37 中段偏弱，1 核心 IL → 🟠 中高。配合 Freeland ⚪ Below Average 預期 4-5 IP 早下，COL 中繼整場吃滿後 5-6 IP — 對 ARI Carroll 真實壓制力低。
- AWAY 牛棚（ARI）：ERA 4.38 中段偏弱 + **3 名核心 IL** → 🔴🔴 崩盤級。Kelly ⚪ Below Average 預期 3-4 IP 早下，ARI 整場後 6+ IP 牛棚崩盤級 + Coors 主場 — 對 COL 中心打序是 nightmare 配置。

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-2.31):
  - **異常 — xERA 更差**：Kelly ERA 7.62 vs xERA **9.93** gap **負向 -2.31** 表示 xERA 更差（接觸品質崩盤）— 本場 Coors 放大可能 ERA 達到 10+。**不自動下修預測**，base AWAY 7.0 可能還偏低。
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.195):
  - **可能反彈**：ARI 7 場樣本 BABIP .195 + heat Cold + 攻↓ 嚴重（近 10 RS 3.30 vs 季 4.47）— 但 Coors 主場 + Freeland ⚪ Below Average → 反彈空間極大。**不自動 ±run value**，但 base AWAY 7.0 可能反映反彈空間不足。

### 額外信號
- 🔴 HOME chain breaks at #6-7：OPS 落差 0.410 — 嚴重後段，但 COL 前 5 棒 vs Kelly 火力齊備。
- 🟠 AWAY chain breaks at #5-6：OPS 落差 0.228 — 中度。
- 🟠 HOME 牛棚 core IL ×1：🟠 中高。
- 🔴 AWAY 牛棚 core IL ×3：🔴🔴 崩盤級 — Kelly 早下後 ARI 中繼 + Coors 主場是極端噩夢。
- 🔴 打者友善球場 PF **131**（極端）— Coors 5 月全功率（夏季模式），雙弱 starter + 雙崩盤牛棚配置下，Total 上行壓力極大。

## 條件修正

- Park Factor: **131.0 → +1.55 run（已內建於 base）**
- 天氣：未公布（跳過天氣分析）— 5 月中 Denver 春末，空氣稀薄全功率
- 先發 tier：HOME Freeland ⚪ Below Average vs AWAY Kelly ⚪ Below Average → 雙崩盤投手戰
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 9.2 | +0.6（ARI 牛棚 ×3 核心 IL 崩盤 + Kelly xERA 9.93 全方位崩盤） | 9.8 |
| AWAY | 7.0 | +0.4（COL 牛棚崩盤後段 + Freeland 真實 ⚪ Below Avg + Carroll/Vargas anchor） | 7.4 |
| Total | 16.2 | +1.0（雙崩盤 + Coors 全功率，但 cap 內） | 17.2 |

## 整體判斷

- **方向（基本面）**：**HOME (COL)**。Kelly 比 Freeland 崩盤程度更嚴重（xERA 9.93 vs 5.19 / K-BB% 1.6% vs 14.7%）+ Coors 主場放大 + ARI 牛棚 ×3 核心 IL 崩盤級。base 已偏 COL (9.2 vs 7.0)，實際差距合理。
- **總分（基本面）**：**17.2 偏高，落點 15.0-19.0**。雙弱 starter + 雙崩盤牛棚（ARI 嚴重）+ Coors PF 131 全功率 → Total 極高，OVER 風險。
- **方向信心**：**60-65%**（HOME 有利）— Kelly 全方位崩盤 + ARI 牛棚崩盤是硬數據，但 ARI Carroll/Vargas vs LHP 真實 anchor + Coors 雙邊放大 — 方向確定但幅度不確定。
- **風險**：
  1. ARI last7 BABIP .195 嚴重冷期 — 若繼續冷，base 7.0 偏高；但 Coors 主場 + Freeland ⚪ Below Average，反彈空間大
  2. Kelly 37 歲快速退化 — 本場 Coors 可能極端崩盤（ER 8+），Total 上行
  3. Coors 5 月全功率 PF 131 — 任何擊球都被放大，HR 風險高
  4. ARI 連敗 2 + 攻↓ 連 10 心理面崩 — 可能比 base 預期更弱

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
