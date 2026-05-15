## 投手對決

### Sean Burke (HOME, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p82, K-BB% p76），gap vs ERA-only = +18.4
  - **同意 🟠 Strong**：ERA 3.68 / xERA 3.78 / FIP 3.62 / xFIP 3.62 / K-BB% 14.5% / WHIP 1.09 / barrel% 5.4 — 數據面真實 Solid+ ~ Strong。本場按 🟠 Strong Ace 對待（ERA 3.5 區間）。
- **TTO 反向**：OPS Δ **-0.166**（TTO1 .694 → TTO3 **.528**）+ K% Δ -14.9pp 掉但 OPS 也跌 — Burke 越投越穩，可撐 6+ IP。
- **對手打線威脅**：🟡 中等。CHC matchup tier 🟡 Average (vs RHP) — Happ vs RHP **.928** 是 anchor 但 last7 .643 + 整體 last7 BABIP **.192** 極冷期 — Bregman/Busch/PCA 全冷期。對 Burke 真實壓制 → CHC 進攻面被三重壓力。

### Edward Cabrera (AWAY, RHP, 28 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p76），gap vs ERA-only = +27.2
  - **謹慎同意**：ERA 3.88 / xERA **4.31** / FIP 4.18 / xFIP 3.34 / K-BB% 14.6% / barrel% **12.8%**（偏高）— xFIP 真實壓制基礎 OK 但接觸品質有結構問題（hard_hit% 30.0 / barrel% 12.8）。本場按 🟠 Strong Ace（ERA 3.5-4.0 區間）對待較合理，不到 Elite。
- **TTO3 penalty**：OPS Δ +0.124（TTO1 .671 → TTO3 .795） — 中度第三輪衰退，5-6 IP 後危險。
- **對手打線威脅**：🟡 中等。CWS matchup tier 🟡 Average (vs RHP) — Murakami vs RHP **.907** + Vargas .694 last7 1.058 / Montgomery vs RHP .827 — 中段 3 棒（含 Vargas last7 火燙 BABIP .333）是真威脅。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致；Murakami / Vargas / Montgomery 三人 vs Cabrera 中段配對良好。

### AWAY — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致；但 heat Cold + last7 BABIP **.192** 極端冷期。Happ vs RHP .928 但 last7 .643。
- **chain_break 信號（🟠）**：#3-4 OPS 落差 0.156 — 輕度。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | **4.46** / 4 / **1 名（🟠 中高）** | 3.74 / 9 / **4 名（🔴🔴 極高）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（CWS）：ERA 4.46 偏弱，Vasil IL60d 是 1 核心 IL → 🟠 中高。Civale (closer) 健康但 setup 群弱。Burke 預期 6+ IP（TTO 反向），後段對 CHC 中心仍可壓制（CHC 冷期）。
- AWAY 牛棚（CHC）：ERA 3.74 中段，**4 名核心 IL**（Thielbar/Harvey + 2）→ 🔴🔴 極端崩盤級。配合 Cabrera TTO3 +0.124，5-6 IP 後離場機率高 → CHC 中繼對 CWS 中心 (Murakami/Vargas/Montgomery) 是 nightmare。

## 風險提示

- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.192):
  - **可能反彈 + 部分持續**：CHC 7 場樣本 BABIP .192 + heat Cold + 全打線 last7 OPS 散亂（.455-.666）→ 嚴重冷期。對 Burke 真實 🟠 Strong → 部分反彈但本場仍難爆。**不自動 ±run value**，AWAY base 4.0 可能準確。

### 額外信號
- 🟠 HOME TTO3 penalty 反向（-0.166）— Burke 越投越穩，CWS 後段壓制力強。
- 🟠 AWAY TTO3 penalty：+0.124 — Cabrera 5-6 IP 後 CWS 攻勢爆。
- 🔴 HOME chain breaks at #8-9：OPS 落差 0.392 — 嚴重後段，但前 5 棒齊備。
- 🟠 AWAY chain breaks at #3-4：OPS 落差 0.156 — 輕度。
- 🟠 HOME 牛棚 core IL ×1：🟠 中高 — Burke 撐 6+ 影響輕。
- 🔴 AWAY 牛棚 core IL ×4：🔴🔴 崩盤級 — Cabrera 早下後 CHC 中繼對 CWS 中心是噩夢。

## 條件修正

- Park Factor: 97.0 → -0.15 run（Rate Field 中性偏輕度投手友善，HR -1%）
- 天氣：未公布（跳過天氣分析）
- 先發 tier：HOME Burke 🟠 Strong vs AWAY Cabrera 真實 🟠 Strong（被 Elite 抬升一檔）→ 對等
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.5 | +0.5（AWAY 牛棚 ×4 核心 IL 崩盤 + Cabrera TTO3 + CWS 中段 vs Cabrera barrel% 12.8 接觸品質弱點）| 5.0 |
| AWAY | 4.0 | 0（CHC cold + Flag 3 + Burke TTO 反向強壓制，base 4.0 已準確）| 4.0 |
| Total | 8.5 | +0.5 | 9.0 |

## 整體判斷

- **方向（基本面）**：**HOME (CWS)**。Burke TTO 反向 + 真實 🟠 Strong + CHC 三重壓力（Cold + Flag 3 + chain break）vs Cabrera 真實 🟠 Strong（不到 Elite）+ TTO3 + barrel% 12.8 接觸弱點 + CHC 牛棚 ×4 崩盤；雖然 base AWAY (CHC) 4.0 偏低，但 HOME 牛棚崩盤 + 投手戰中等對等 → CWS 略有利。
- **總分（基本面）**：**9.0 接近實際，落點 8.0-10.0**。雙方先發都 Strong 對等但 CHC 牛棚崩盤 + 兩端 TTO 都會早下 → Total 中等偏高。
- **方向信心**：**55-60%**（HOME 微利）— CWS 連勝 5 主場狀況強 + CHC 冷期 + CHC 牛棚崩盤是硬數據，但 Cabrera 統治力仍可能壓 CWS。
- **風險**：
  1. CHC 連勝 1 但近 30 RS 5.37 → 攻↓ 連 10 嚴重（RS 3.50），是真實冷期還是噪音？
  2. CWS 連勝 5（含 KC/SEA）+ 主場優勢 — 狀況強但對手較弱，本場面 Cabrera 是升級
  3. Cabrera barrel% 12.8 對 Murakami barrel% 22.7 / Montgomery 14.9 — HR 風險高，CWS 一棒可能改變總分
  4. Cabrera 23 ERA 假象（被 ERA-xERA 抬升）— 本場若回歸真實水平，CHC 可吃，方向變數

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
