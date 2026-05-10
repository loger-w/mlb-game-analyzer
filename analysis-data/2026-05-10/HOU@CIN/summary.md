## 投手對決

### Andrew Abbott (HOME, LHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟢 Back-end Starter（xFIP p23, K-BB% p18），gap vs ERA-only = +5.9
  - **同意**：ERA 5.13 / xERA 5.35 / FIP 4.74 / xFIP 4.58 / K-BB% **5.0%**（極低）/ whiff% 9.0% — 各項一致 ⚪ Below Average ~ 🟢 Back-end。本場按 ⚪ Below Average 對待。
- **Reverse platoon 信號（🔴 +0.309）**：vs LHB OPS **1.030**（38 BF）vs vs RHB OPS .721（144 BF）— LHP 對 LHB 反而吃虧。
  - HOU 多 LHB（Alvarez/Smith LHB；Walker R）— Alvarez vs LHP **1.093**！剛好踩中 Abbott 弱點 → 是真實爆分點。
- **TTO3 penalty（-0.270）**：第三輪反向（OPS 反而下降），但 K-BB% 5.0% 撐不住 6 IP。
- **對手打線威脅**：🔴 極高。HOU matchup tier 🔴 Elite (vs LHP) — Alvarez vs LHP **1.093** + Paredes vs LHP **.827** + Walker .812 + Matthews .681 — 中心 2-4 棒對 Abbott 是 dream matchup。

### Kai-Wei Teng (AWAY, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 沒給定（樣本 GS 1 / 5.7 IP 太薄）。原始 tier 🟠 Strong Ace — ERA 2.35 / xERA 3.28 / FIP 3.80 / xFIP 3.40 / K-BB% 16.8% — 樣本太薄不可信。
  - **謹慎按 🟢 Back-end ~ 🟡 Solid 對待**：5.7 IP 樣本，所有 ERA/FIP 都不可信；但 ST 球種 RV +3.9 是真實亮點。
- **vs LHB 弱點（.205/.239/.455 SLG，46 BF）**：對 LHB 控球 OK 但 SLG .455 顯示 power 漏；CIN 中段 LHB 多（Benson L #1、Bleday L #5）。
- **對手打線威脅**：🟠 高。CIN matchup tier 🟢 Weak (vs RHP) — Benson vs RHP .743 last7 1.039 / De La Cruz .813 / Bleday vs RHP **1.114** — 中心 1-5 棒對 Teng 仍可吃。

## 打線評級

### HOME — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🔴 Elite — 比 season tier 上修一檔！script 認為 CIN 對 RHP 是聯盟頂級。Bleday 1.114 / De La Cruz .813 vs RHP 全火力齊備。
- **chain_break 信號（🔴 #7-8）**：Dunn 1.850 → McLain .625 — Dunn 是新人小樣本（OPS 1.850 不可信），實際 chain break 有限。
- **last7 BABIP .369 偏高警訊（未觸 Flag 3 .370 門檻但接近）**：CIN last7 火燙含運氣成分，本場可能部分回歸。

### AWAY — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🔴 Elite（與 dossier 標 🟢 Weak 衝突 — 但 Alvarez vs LHP 1.093 / Paredes .827 真實火燙，腳本可能誤判）
  - 實際應上修：HOU 對 LHP 火力是聯盟頂級。
- **chain_break 信號（🔴 #7-8）**：Cole 1.154 → Allen .566 — Cole 小樣本不可信。
- **last7 BABIP .248 Flag 3 警訊**：見風險段。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.55 / 4 / **2 名（Ferguson + Pagán）** | **6.12** / 9 / **1 名（Hader closer IL60d）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（CIN）：ERA 4.55 中段稍弱 + **2 核心 IL 🔴 高**（Ferguson + Pagán 雙缺陣）→ HOU Alvarez 等末段攻擊放大。
- AWAY 牛棚（HOU）：ERA **6.12** 嚴重崩盤 + Hader closer 60-day IL（1 核心 IL → 🟠 中高）。CIN 中段火力末段反咬機率極高。

## 風險提示

- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.248):
  - **可能部分回歸**：HOU last7 BABIP .248 偏低，但 EV/Barrel 數據仍強（Alvarez/Walker/Smith 都 EV95% 45+）。本場對 Abbott 弱投手，回歸壓力 + 數據反彈雙重推升 → HOU base 6.0 偏低。

### 額外信號
- 🔴 HOME reverse platoon Δ +0.309 — Alvarez vs LHP 1.093 是真實 nightmare for Abbott。
- 🟠 HOME single-pitch dependent FF 47.3% — Abbott FF-heavy 對 HOU 強打 sit fastball 是噩夢。
- 🔴 HOME chain break #7-8（Dunn 1.850 vs RHP 5.000 是樣本失真）— 影響輕。
- 🔴 AWAY chain break #7-8（Cole 1.154 樣本失真）— 影響輕。
- 🔴 HOME 牛棚 2 核心 IL — HOU 末段攻擊極大化。
- 🟠 AWAY 牛棚 1 核心 IL（Hader）— 影響中等。

## 條件修正

- Park Factor: 104.0 → +0.20 run（Great American Ball Park runs 104 + HR **+29%** — 嚴重 HR friendly）
- 天氣：Partly Cloudy 66°F, wind 3 mph **Out To RF** — 微順風推 LHB pull HR（Alvarez/Smith/Benson/Bleday LHB 多）→ 推升 HR
- 先發 tier：HOME Abbott ⚪ Below Average vs AWAY Teng 🟢 Back-end ~ 🟡 Solid（小樣本）→ 雙弱配對
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.8 | +0.3（AWAY 牛棚 1 核心 IL → 中度推升 CIN 末段攻擊） | 5.1 |
| AWAY | 6.0 | +0.6（HOME 牛棚 2 核心 IL → 嚴重推升 HOU 末段攻擊） | 6.6 |
| Total | 10.8 | +0.9 | 11.7 |

## 整體判斷

- **方向（基本面）**：**AWAY (HOU) 強勢有利**。Abbott ⚪ Below Average + reverse platoon 對 LHB（Alvarez 1.093）+ HOU 中心強打 + CIN 牛棚 2 核心 IL → HOU 進攻面三重利好。Teng 樣本不確定但 CIN 中心 (Bleday 1.114) 也可能爆分。
- **總分（基本面）**：**11.7（base 10.8 + +0.9 信號）**，落點 10.5-13.0。雙弱 starter + 雙方強打 + Great American HR +29% + Out To RF → Total 強上行。
- **方向信心**：~68%（AWAY），結構性支撐（Abbott vs LHB + HOU 火力 + CIN 牛棚崩）。
- **風險**：
  1. **Teng 樣本零（5.7 IP）**：可能反常打出 4 IP 5R，CIN 進攻面爆分風險上升
  2. HOU last7 BABIP .248 — 部分回歸但 Abbott 太弱，回歸機率高
  3. CIN G1 連勝 1 信心面 + Bleday 1.114 vs RHP — 真實火燙，可能爆分
  4. Great American HR +29% + Out To RF + 雙方多 LHB — 極端高 Total（13+）也可能

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
