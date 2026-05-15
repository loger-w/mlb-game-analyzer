## 投手對決

### Aaron Civale (HOME, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p45, K-BB% p56），gap vs ERA-only = -29.8
  - **同意 + ERA 高估**：ERA **2.59** 看起來像 🟠 Strong Ace 但 xERA **4.16** / FIP 3.70 / xFIP 4.20 / K-BB% 11.0% / velo 85.9（極低）/ vs LHB **.291/.360/.427**（OPS .787）— 真實水平 🟢 Back-end ~ 🟡 Solid（ERA 4.0+ 區間）。
  - **本場按 🟡 Solid** 對待，承認 ERA 2.59 不可持續。
- **TTO3 penalty**：OPS Δ +0.117（K% Δ -19.3pp） — 中度第三輪衰退。
- **對手打線威脅**：🟢 低。SF matchup tier 🟢 Weak (vs RHP) — Adames .662 / Devers .704 last7 **1.179**（BABIP .571 火燙）/ Lee .742 — Devers last7 火燙是唯一真威脅，整體 SF 弱。

### Tyler Mahle (AWAY, RHP, 31 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p84, K-BB% p62），gap vs ERA-only = **+45.0**（極大）
  - **不完全同意**：xFIP 3.58 看起來 Strong 但 ERA **5.18** / xERA 4.43 / FIP 5.02 / vs RHB **.316/.409/.566**（OPS .975 嚴重弱點）— 真實水平 🟢 Back-end（ERA 4.5-5.0 區間）。tier_v2 過度看重 xFIP。
  - **本場按 🟢 Back-end** 對待。
- **Reverse platoon 嚴重（🔴 Δ +0.350）**：vs RHB OPS **.975** > vs LHB .625 — RHB 嚴重弱點。
  - ATH 中段 RHB 多 — Kurtz (LHB) / Langeliers (RHB) / Soderstrom (LHB) / McNeil (LHB) / Butler (LHB) — 實際 LHB 多反而踩不到 reverse platoon 弱點，但 Langeliers RHB 季度 OPS **1.020** vs RHP 1.013 / Kurtz LHB vs RHP **1.007** — 兩大 anchor 都對 RHP 強。
- **Single-pitch dependent（🟠）**：FF 47.6% — 邊緣，FS 25.0% 補強。
- **TTO3 penalty**：OPS Δ +0.033（K% Δ -8.9pp） — 輕度衰退。
- **對手打線威脅**：🔴 高。ATH matchup tier 🟢 Weak (vs RHP) 但 Kurtz/Langeliers vs RHP 全 1.000+ + last7 1.308/1.031 火燙 — 對 Mahle 真實 🟢 Back-end + vs RHB 弱點是 dream matchup。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong — 比 season tier 上修一檔；Kurtz/Langeliers anchor 真實 + last7 火燙。
- **chain_break 信號（🔴）**：#2-3 OPS 落差 **0.360**（Langeliers 1.020 → Soderstrom .660）— 嚴重，但前 2 棒 vs Mahle 是極端優勢配對。

### AWAY — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak — 與 season tier 一致；Devers last7 1.179 是 anchor 但 SF 整體 vs RHP 弱。
- **chain_break 信號（🟠）**：#7-8 OPS 落差 0.204 — 中度。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.63 / 2 / **0 名核心** | 3.47 / 8 / **4 名（🔴🔴 極高）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（ATH）：ERA 4.63 偏弱，無核心 IL。後段對 SF 弱進攻仍可壓制。
- AWAY 牛棚（SF）：ERA 3.47 中段穩定但 **4 名核心 IL**（Miller/Birdsong + 2）→ 🔴🔴 崩盤級。配合 Mahle 真實 🟢 Back-end + vs RHB 弱點，5 IP 內離場機率高 → SF 整場後 5-6 IP 牛棚崩盤級 + ATH 中心 Kurtz/Langeliers 火燙是 nightmare。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-1.57):
  - **強運氣加持**：Civale ERA 2.59 vs xERA 4.16 gap -1.57。FIP 3.70 + vs LHB OPS .787 顯示真實水平 🟡 Solid 邊緣 🟢 Back-end。**不自動下修**，base HOME 6.3 反映 SF 弱進攻 vs Civale 真實水平。

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ +0.117 — Civale 5 IP 後 SF 攻勢爆，但 SF 弱進攻影響輕。
- 🔴 AWAY reverse platoon Δ **+0.350**（Mahle vs RHB 嚴重弱點）— ATH Langeliers RHB anchor 踩中。
- 🟠 AWAY single-pitch dependent：FF 47.6% — 邊緣。
- 🟠 AWAY TTO3 penalty：+0.033 — 輕度。
- 🔴 HOME chain breaks at #2-3：OPS 落差 0.360 — 嚴重但前 2 棒齊備。
- 🟠 AWAY chain breaks at #7-8：OPS 落差 0.204 — 中度。
- 🔴 AWAY 牛棚 core IL ×4：🔴🔴 崩盤級 — Mahle 早下後 SF 中繼對 Kurtz/Langeliers 是噩夢。

## 條件修正

- Park Factor: 109.0 → +0.45 run（Sutter Health Park 偏打者友善，HR +6%）
- 天氣：未公布（跳過天氣分析）
- 先發 tier：HOME Civale 真實 🟡 Solid vs AWAY Mahle 真實 🟢 Back-end → HOME 投手戰略優
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 6.3 | +0.5（Mahle vs RHB 弱點 + SF 牛棚 ×4 核心 IL 崩盤 + Kurtz/Langeliers 火燙）| 6.8 |
| AWAY | 4.0 | 0（Civale 真實 🟡 Solid 壓制 SF 弱進攻，base 已準確）| 4.0 |
| Total | 10.3 | +0.5 | 10.8 |

## 整體判斷

- **方向（基本面）**：**HOME (ATH)**。Mahle 真實 🟢 Back-end + vs RHB 嚴重弱點（OPS .975）+ SF 牛棚 ×4 核心 IL 崩盤 vs Civale 真實 🟡 Solid + ATH 牛棚穩定。ATH 中心 Kurtz/Langeliers 火燙 + 對 Mahle 弱點配對良好。base 已偏 HOME (6.3 vs 4.0)，實際差距類似。
- **總分（基本面）**：**10.8 偏高，落點 9.5-12.0**。ATH 進攻面強放大 + SF 牛棚崩盤後段 + Sutter Health PF 109 打者友善 → Total 上行壓力。
- **方向信心**：**65-70%**（HOME 有利）— Mahle vs RHB 弱點 + SF 牛棚崩盤是硬數據；Kurtz/Langeliers 火燙真實 anchor。
- **風險**：
  1. Devers last7 BABIP **.571** + OPS 1.179 — 火燙不可持續但 vs RHP 季度 .704 中等水平，本場仍可能 HR
  2. Civale ERA 2.59 雖然 Flag 8 但本場可能繼續壓制（FIP 3.70 + SF 弱進攻）
  3. Kurtz/Langeliers last7 BABIP .444/.400 火燙 — 部分回歸但 vs RHP 季度 1.007/1.013 真實
  4. Sutter Health 5 月夜場（未公布天氣）— 風向可能放大或壓制長球

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
