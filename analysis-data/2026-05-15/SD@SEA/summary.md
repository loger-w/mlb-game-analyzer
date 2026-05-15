## 投手對決

### Emerson Hancock (HOME, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +21.3
  - **同意**：ERA 3.21 / xERA 3.91 / FIP 3.75 / xFIP **2.74** / K-BB% **22.1%** / WHIP 1.01 / barrel% 11.3（偏高）— xFIP 2.74 是真實 elite 基礎，K-BB% 22.1 強。但 barrel% 11.3 是接觸品質警訊。本場按 🟠 Strong Ace ~ 🔴 Elite Ace（ERA 3.0-3.5 區間）對待。
- **TTO3 penalty 嚴重**：OPS Δ **+0.240**（TTO1 .603 → TTO3 **.843**）+ K% Δ -3.4pp — 5 IP 後危險。
- **對手打線威脅**：🟢 低。SD matchup tier 🟢 Weak (vs RHP) — Tatis Jr. vs RHP .593 / Merrill .627 / Machado .544 / Bogaerts .758 — 全弱 + last7 BABIP .231 冷期 + heat Cold + 攻↓ 連 10 RS 3.30。對 Hancock Elite Ace + T-Mobile PF 82 → 接近 shut out。

### Randy Vásquez (AWAY, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p81），gap vs ERA-only = +13.0
  - **謹慎同意**：ERA 3.05 / xERA **4.64** / FIP 3.26 / xFIP 3.43 / K-BB% 16.0% / barrel% 12.1（偏高）— xFIP 3.43 + K-BB% 16% 是真實 Strong 基礎，但 xERA 4.64 / barrel% 12.1 顯示接觸品質有警訊。本場按 🟠 Strong Ace（ERA 3.0-3.5 區間）對待，不到 Elite。
- **TTO 反向**：OPS Δ **-0.098** — Vásquez 越投越穩，可撐 6+ IP。
- **對手打線威脅**：🟠 中高。SEA matchup tier 🟠 Strong (vs RHP) — Arozarena vs RHP **.861** last7 **1.230**（BABIP .667 火燙）+ Rodríguez .671 last7 .846 + Naylor .762 last7 .858 — Arozarena 是真威脅，Rodríguez/Naylor anchor。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong — 比 season tier 上修一檔；Arozarena/Rodríguez/Naylor 三人對 Vásquez 中等水平 RHP 是真實 anchor。

### AWAY — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak — 比 season tier 下修一檔；對 Hancock Elite Ace + T-Mobile PF 82 → 接近 shut out。
- **Flag 3 last7 BABIP .231** — 嚴重冷期。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.08 / 4 / **3 名（🔴🔴 極高）** | 3.53 / 5 / **0 名核心** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（SEA）：ERA 3.08 elite + **3 名核心 IL**（Vargas + Speier + 1）→ 🔴🔴 崩盤級。配合 Hancock TTO3 +0.240 5 IP 後離場機率高 → SEA 整場後 5-6 IP 牛棚是潛在崩盤點，但 SD 弱進攻可能無法利用。
- AWAY 牛棚（SD）：ERA 3.53 中段穩定 + 無核心 IL → 完整可用。Suarez (closer) 健康。後段對 SEA 中心仍有壓制。

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-1.59):
  - **接觸品質警訊**：Vásquez ERA 3.05 vs xERA 4.64 gap -1.59，barrel% 12.1 + hard_hit% 29.1 顯示接觸品質弱。但 K-BB% 16.0% + xFIP 3.43 是真實基礎，本場按 🟠 Strong Ace 對待，**不自動下修**。
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.231):
  - **嚴重持續**：SD 7 場樣本 BABIP .231 + heat Cold + matchup 🟢 Weak vs Hancock Elite + T-Mobile PF 82 → 四重壓力，本場仍難反彈。**不自動 ±run value**，base 3.2 可能還偏高。

### 額外信號
- 🔴 HOME TTO3 penalty：OPS Δ +0.240 — Hancock 5 IP 後 SD 攻勢有空間但 SD 弱進攻 + T-Mobile 壓制 → 影響輕。
- 🟠 AWAY TTO3 反向（-0.098） — Vásquez 越投越穩。
- 🔴 HOME chain breaks at #6-7：OPS 落差 **0.478** — 嚴重後段，但前 5 棒 vs Vásquez 火力齊備。
- 🟠 AWAY chain breaks at #7-8：OPS 落差 0.234 — 中度，但 SD 整體攻擊弱，影響輕。
- 🔴 HOME 牛棚 core IL ×3：🔴🔴 崩盤級 — 配合 Hancock TTO3 早下 + SEA 後段對 SD 弱進攻是反向有利。
- 🔴 投手友善球場 PF **82**（極端）— T-Mobile HR -18%，雙弱進攻 + 雙 Strong Ace + 極端壓制球場 → Total 嚴重壓低。

## 條件修正

- Park Factor: **82.0 → -0.90 run（已內建於 base）** — T-Mobile 極端投手友善
- 天氣：未公布（跳過天氣分析）— 5 月中 Seattle 春末，溫度低 + 海風進一步壓制 HR
- 先發 tier：HOME Hancock 真實 🟠 Strong ~ 🔴 Elite vs AWAY Vásquez 真實 🟠 Strong → 對等
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.1 | +0.3（SEA 中段 Arozarena 火燙 + Rodríguez/Naylor 對 Vásquez 真實威脅）| 3.4 |
| AWAY | 3.2 | -0.4（SD 四重壓力：Cold + Flag 3 + matchup Weak + T-Mobile 壓制 + Hancock Elite）| 2.8 |
| Total | 6.3 | -0.1 | 6.2 |

## 整體判斷

- **方向（基本面）**：**HOME (SEA)**。Hancock vs Vásquez 都 🟠 Strong Ace tier 對等 + SEA 中段 (Arozarena/Rodríguez/Naylor) 對 Vásquez 真實威脅 vs SD 四重壓力（Cold + Flag 3 + matchup Weak + T-Mobile + Hancock Elite）。base 接近持平但 SEA 進攻面略優。
- **總分（基本面）**：**6.2 偏低，落點 5.0-7.0**。雙強 starter + 雙方都有牛棚壓制 + T-Mobile PF 82 極端壓制 + SD 弱進攻 → Total 嚴重壓低，UNDER 風險高。
- **方向信心**：**60-65%**（HOME 微利）— SD 四重壓力 + Arozarena last7 1.230 火燙是 Tilt 因子；Hancock 真實接近 Elite。
- **風險**：
  1. Arozarena last7 BABIP **.667** + OPS 1.230 — 火燙不可持續但 vs RHP 季度 .861 真實，本場可能 HR（即使 T-Mobile）
  2. Hancock barrel% 11.3 + Vásquez barrel% 12.1 — 雙方都有接觸品質警訊，可能 HR
  3. SEA 牛棚 ×3 核心 IL — 若 Hancock 早下 + SD 反彈，SEA 末段崩盤
  4. T-Mobile 5 月中夜場 — 海風 + 低溫壓制長球，但 doubles 仍可拿分

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
