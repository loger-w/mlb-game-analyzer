## 投手對決

### Justin Wrobleski (HOME, LHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = ⚪ Below Average（xFIP p5, K-BB% p12），gap vs ERA-only = -67.1
  - **同意 + 強調 ERA 嚴重高估真實水平**：ERA **1.25** 看起來像 🔴 Elite Ace 但 xERA **4.23** / FIP 3.10 / xFIP **5.01** / K-BB% **3.5%**（極低！）/ whiff% 6.7% / barrel% 6.0% — 真實水平 ⚪ Below Average（5.0 ERA 區間）。-67.1 gap 是極端嚴重運氣加持。本場按 ⚪ Below Average 對待。
- **Flag 8 era_xera_delta=-2.98**：嚴重運氣加持，BABIP 偏低 + LOB% 偏高，K-BB% 3.5% 是真實偏弱。
- **Reverse platoon 信號（🔴 +0.224）**：vs LHB OPS .666（33 BF）vs vs RHB OPS .442（110 BF）— LHP 對 LHB 反而吃虧，但 33 BF 樣本偏小。
  - ATL 多 LHB 中段（Olson L #2、Baldwin L #1、Albies S #3）— Olson vs LHP **.861** 是真實爆分點。
- **單一球種 FF 49.8%**：FF-heavy LHP，球種組合不健全（CU 7.8%）。
- **TTO3 penalty -0.281 + K% -11.5pp**：第三輪 K% 暴降但 OPS 反向下降，混合信號（樣本不足？）。
- **對手打線威脅**：🟠 高。ATL matchup tier 🟢 Weak (vs LHP) — 但 Olson vs LHP **.861** last7 1.091 / Albies vs LHP **1.029** / Baldwin .943 last7 .980 — 中心 1-3 棒對 Wrobleski 是 dream matchup。

### Bryce Elder (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p82, K-BB% p77），gap vs ERA-only = -10.5
  - **同意**：ERA 2.02 / xERA 2.83 / FIP 3.04 / xFIP 3.62 / K-BB% 14.9% / WHIP 1.02 / barrel% 3.8% — 各項一致 🟠 Strong Ace。本場按 🟠 Strong Ace 對待。
- **Reverse platoon 信號（🟠 +0.081）**：vs RHB OPS .603 vs vs LHB OPS .522 — 微反向，影響輕。
- **TTO3 penalty -0.046**：第三輪反向，影響輕。
- **對手打線威脅**：🟡 中等。LAD matchup tier 🔴 Elite (vs RHP) — Pages vs RHP **.961** last7 1.271（火燙！）/ Muncy .884 / Rushing **1.150**（小樣本但火燙）/ Freeman .799 / Tucker .732 — 中心 3-7 棒對 Elder 是真實威脅，特別 Pages last7 1.271。

## 打線評級

### HOME — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🔴 Elite — 比 season tier 上修一檔；Pages/Muncy/Rushing/Tucker vs RHP 全火力齊備。
- **chain_break 信號（🔴 #7-8）**：Rushing 1.106 → Kim .777 — 雖然落差 0.329 但 Kim 仍 .777 不算嚴重黑洞。

### AWAY — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak — 嚴重下修；但 Olson/Albies/Baldwin vs LHP 火燙。實際應該回到 Strong（個別打者強）。
- **chain_break 信號（🔴 #6-7）**：影響輕（dossier 沒有 #6-7 詳細數據）。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.36 / 10 / **2 名（Stewart + Díaz）** | 3.26 / 7 / **1 名（Young）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（LAD）：ERA 3.36 中段穩定 + **2 核心 IL 🔴 高**（Stewart + Díaz 雙缺陣，Díaz 是 closer 級）→ ATL 末段攻擊放大關鍵。
- AWAY 牛棚（ATL）：ERA 3.26 中段穩定 + 1 核心 IL → 🟠 中高。Iglesias 等 setup 群可用，後段對 LAD 中心壓制力中等。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-2.98):
  - **嚴重運氣加持 + 結構性弱點**：Wrobleski K-BB% 3.5% / xFIP 5.01 都是真實偏弱，ERA 1.25 完全不可持續。本場按 ⚪ Below Average 對待，**不自動下修**，但敘事上 ATL 進攻面 base 3.5 嚴重偏低，實際應該 5.0+ runs。

### 額外信號
- 🔴 HOME reverse platoon Δ +0.224 — Olson/Albies/Baldwin vs LHP 火燙是真實爆分點。
- 🟠 HOME single-pitch dependent FF 49.8% — 球種組合不健全。
- 🟠 HOME TTO3 K% -11.5pp — Wrobleski 第三輪壓制力崩潰，ATL 第三輪反彈關鍵。
- 🟠 AWAY reverse platoon Δ +0.081 — 影響輕。
- 🟠 AWAY TTO3 -0.046 — 影響輕。
- 🔴 雙方 chain break — 影響中等。
- 🔴 HOME 牛棚 2 核心 IL — ATL 末段攻擊極大化關鍵。
- 🟠 AWAY 牛棚 1 核心 IL — 影響輕。

## 條件修正

- Park Factor: 98.0 → -0.10 run（UNIQLO Field at Dodger Stadium runs 98 但 HR **+21%**）
- 天氣：Sunny 79°F, wind 6 mph **Out To CF** — 順風中外野推 HR；對 RHB pull 也順風
- 先發 tier：HOME Wrobleski ⚪ Below Average vs AWAY Elder 🟠 Strong Ace → 嚴重不對稱，AWAY 失分基準偏低
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.6 | 0（核心 IL 1 名，未達 ≥2 門檻） | 3.6 |
| AWAY | 3.5 | +0.6（HOME 牛棚 2 核心 IL → ATL 末段攻擊極大化） | 4.1 |
| Total | 7.1 | +0.6 | 7.7 |

## 整體判斷

- **方向（基本面）**：**AWAY (ATL) 中強有利**。Wrobleski 真實 ⚪ Below Average + ATL 多 LHB 中段（Olson/Albies/Baldwin vs LHP 火燙）+ LAD 牛棚 2 核心 IL → ATL 進攻面三重利好。Elder 🟠 Strong Ace 對 LAD 強打中心仍是壓制配對。
- **總分（基本面）**：**7.7（base 7.1 + +0.6 信號）**，落點 7.0-9.5。Wrobleski 真實水平差 + LAD 牛棚崩 + Out To CF 風 + Dodger Stadium HR +21% → Total 上行；但 Elder 強壓制限制 LAD 上限。
- **方向信心**：~68%（AWAY），結構性強支撐（Wrobleski K-BB% 3.5% + LAD 牛棚 2 核心 IL + ATL 強打配對）。
- **風險**：
  1. **Wrobleski ERA 1.25 vs xERA 4.23** — 本場可能繼續運氣加持（5 IP 1R），ATL 進攻面可能投手戰
  2. LAD Pages last7 1.271 + Rushing 1.150 vs RHP — 真實爆分點，可能單棒打破 Elder 壓制
  3. ATL G1 連勝 1（7-2 大勝）信心面強 — 延續優勢機率高
  4. Out To CF 風 + Dodger Stadium HR +21% — 雙方都可能單棒爆分，極端高 Total（10+）也可能

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
