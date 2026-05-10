## 投手對決

### Walker Buehler (HOME, RHP, 31 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p82, K-BB% p64），gap vs ERA-only = +54.0
  - **不完全同意 + 強調混合信號**：ERA **5.64** 看起來像 ⚪ Below Average 但 xERA 4.72 / FIP **3.23** / xFIP 3.62 / K-BB% 12.4% / barrel% 5.3% — FIP-base 顯示真實水平 🟡 Solid（4.0 ERA 區間）。本場按 🟢 Back-end ~ 🟡 Solid 對待。
- **vs LHB 弱點（.309/.397/.368 SLG，78 BF）**：對 LHB 控球差（OBP .397），STL 多 RHB 中段（Wetherholt R / Burleson L / Walker R / Gorman L #5）— LHB 點是 Burleson + Gorman。
- **TTO3 +0.022 career fallback**：career 第三輪輕微衰退，影響輕。
- **對手打線威脅**：🟡 中等。STL matchup tier 🟡 Average (vs RHP) — Wetherholt vs RHP .799 / Herrera **.865** last7 .952 / Burleson **.926** / Walker **.945** — 中心 2-4 棒對 Buehler 是真實爆分群。

### Kyle Leahy (AWAY, RHP, 28 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p81, K-BB% p41），gap vs ERA-only = +33.4
  - **不同意 + 強調 ERA 偏高估**：ERA 4.93 / xERA 5.96 / FIP 4.92 / xFIP 3.64 / K-BB% **8.8%** / barrel% **11.3%** / vs LHB **.358/.422/.580**（90 BF）— 真實水平 ⚪ Below Average ~ 🟢 Back-end。+33.4 gap 是 xFIP 樣本失真，xFIP 3.64 不可信（hard_hit% 33.0% 太高）。本場按 ⚪ Below Average 對待。
- **vs LHB 嚴重弱點（.358/.422/.580）**：是真實結構問題（90 BF 不算小），SD 多 LHB 中心（Tatis R, Merrill L #2, Machado R, Bogaerts R, Laureano R）— Merrill 是攻擊點。
- **TTO3 penalty（🔴 +0.727 → OPS 1.336）**：第三輪極度爆掉，K% 從 26.6% 掉到 5.9%（-20.7pp 暴跌）— Leahy 撐不過 5 IP，SD 中段第三輪可能爆分。
- **對手打線威脅**：🟠 高。SD matchup tier 🟢 Weak (vs RHP) — Tatis vs RHP .582 / Merrill .640 last7 .797 / Machado .585 / Bogaerts .764 last7 .814 / Laureano .757 — 中段火力分散但 Bogaerts 是真實爆分點，Merrill last7 反彈。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak — 比 season tier 下修；中心 Tatis/Machado last7 冷期。
- **chain_break 信號（🟠 #7-8）**：影響輕。
- **last7 BABIP .207 Flag 3 警訊**：見風險段。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致；Wetherholt/Herrera/Burleson/Walker 中心強。
- **chain_break 信號（🟠 #4-5）**：Walker .946 → Gorman .667 — 中度，但 Walker 中段強。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.59 / 5 / **0 名核心** | **4.67** / 1 / **0 名核心** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（SD）：ERA 3.59 中段穩定，無核心 IL。Estrada/Suárez 等 setup 健康。對 STL 中心壓制力中等。
- AWAY 牛棚（STL）：ERA **4.67** 偏弱，無核心 IL 但整體深度有限。若 Leahy 早下（K-BB% 8.8% + TTO3 1.336 撐不住），SD 中心可能爆分末段。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.207):
  - **可能反彈**：SD last7 BABIP .207 嚴重偏低，Tatis EV/Barrel 強（58.5 / 11.3）但 last7 BABIP .222 冷期。本場對 Leahy 弱投手 → 反彈 + 數據反彈雙重，SD base 5.0 偏低。

### 額外信號
- ℹ️ HOME balanced 4+ pitches — Buehler 球種多元，影響輕。
- 🔴 AWAY TTO3 +0.727 K% -20.7pp — Leahy 第三輪極度崩盤，SD 中段第三輪是反彈關鍵。
- 🟠 雙方 chain breaks — 影響輕。

## 條件修正

- Park Factor: 95.0 → -0.25 run（Petco Park 偏輕度投手友善但 HR +7%）
- 天氣：未公布（跳過天氣分析）— SD 海風常吹進場，壓 HR
- 先發 tier：HOME Buehler 🟡 Solid vs AWAY Leahy ⚪ Below Average → AWAY 失分基準應偏高
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.0 | 0（核心 IL 0 名） | 5.0 |
| AWAY | 3.4 | 0（核心 IL 0 名） | 3.4 |
| Total | 8.4 | 0 | 8.4 |

## 整體判斷

- **方向（基本面）**：**HOME (SD) 中度有利**。Leahy ⚪ Below Average + TTO3 1.336 + vs LHB .580 SLG → SD 進攻面有空間（Bogaerts/Merrill 強）；Buehler 真實 🟢 Back-end ~ 🟡 Solid 對 STL 中心 (Burleson/Walker) 仍可吃但 K-BB% 12.4% 撐住。SD last7 反彈機會 + Leahy 第三輪崩 → SD 進攻面優勢。
- **總分（基本面）**：**8.4 接近實際**，落點 7.5-10.0。雙弱 starter（特別 Leahy 真實水平差）+ 雙方中段火力 → Total 中性偏上行；Petco PF 95 + 海風壓 HR 部分抵消。
- **方向信心**：~62%（HOME），結構性支撐（Leahy 真實水平差更多 + SD last7 BABIP 反彈機會 + 主場優勢）。
- **風險**：
  1. SD last7 BABIP .207 — 本場大幅反彈機率高，特別 Leahy 弱投手
  2. Leahy ERA 4.93 vs xFIP 3.64 — 本場可能反常打出 5 IP 2R，但 TTO3 1.336 顯示撐不住
  3. STL Walker .946 vs RHP / Herrera .865 — 真實爆分點，可能單棒打破投手戰
  4. Petco 海風（未公布天氣）— 壓 HR 但雙方都受影響

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
