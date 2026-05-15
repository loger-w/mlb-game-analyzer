## 投手對決

### Zack Littell (HOME, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = ⚪ Below Average（xFIP p5, K-BB% p5），gap vs ERA-only = +7.8
  - **完全同意**：ERA 6.94 / xERA 7.64 / FIP **8.25** / K-BB% **2.4%** / whiff% 5.1（極低）/ velo 87.3（低）/ vs LHB **.305/.363/.732 SLG**（92 BF）— 數據面崩盤。本場按 ⚪ Below Average（ERA 6.5+ 區間）對待。
- **vs LHB 嚴重弱點**：SLG .732 是真實結構問題；BAL 有 Henderson (L) / Basallo (L/switch) 是攻擊點。
- **對手打線威脅**：🟠 高。BAL 中段 Basallo last7 **1.142**（BABIP .524 火燙）+ Alonso vs RHP .765 / Ward .787 — 但前 5 棒 last7 OPS 全 .269-1.142 散亂，整體不穩。

### Shane Baz (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p57, K-BB% p47），gap vs ERA-only = **+37.4**（極大）
  - **不完全同意**：xFIP 4.02 / FIP 4.14 看起來 Solid 但 ERA 5.48 / xERA 4.64 / vs LHB **.323/.400/.500**（115 BF）真實 LHB 弱點。本場按 🟢 Back-end（ERA 5.0 區間）對待。
- **TTO3 penalty 嚴重**：OPS Δ **+0.262**（TTO1 .668 → TTO3 **.930**）+ K% 從 24.7% 掉到 13.5% — 第三輪極端衰退，5 IP 後危險。
- **vs LHB 弱點**：WSH 多 LHB 中心 — Wood (L) / Lile (L) / Abrams (L) — 精準踩 Baz vs LHB OPS .900。
- **對手打線威脅**：🔴 高。WSH Wood vs RHP **.961** / Lile last7 **1.145** / Abrams vs RHP **1.022** — 前 3 棒全是 Baz 弱點 LHB 集中。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 但 vs Baz vs LHB 弱點 + LHB-heavy → 上修一檔有空間。
- **chain_break 信號（🟠）**：#3-4 OPS 落差 0.264（Abrams .922 → House .658）— 中度，但前 3 棒 vs Baz 集中火力，影響輕。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 對 Littell ⚪ Below Average → 應上修一檔。
- **chain_break 信號（🔴）**：#8-9 OPS 落差 0.304 — 嚴重後段斷層，但 Littell 投不到那麼久，影響極輕。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.81 / 7 / **2 名（🔴 高）** | 4.09 / 8 / **3 名（🔴🔴 極高）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（WSH）：ERA **4.81** 偏弱，Beeter + Kranick 雙核心 IL → 🔴 高。配合 Littell ⚪ Below Average 預期早下（3-4 IP），WSH 整場後 5-6 IP 全靠中繼 — 對 BAL 火力是 nightmare 配置。
- AWAY 牛棚（BAL）：ERA 4.09 中段稍弱但 Bautista (closer) + Wolfram + 還有 1 名核心 IL → 🔴🔴 極高（崩盤級）。Baz TTO3 5 IP 後離場機率高，BAL 後段同樣崩盤。

**雙方牛棚崩盤對峙** — 本場 Total 上行壓力極大。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ -0.057 — 反向（career fallback），影響輕。
- 🔴 AWAY TTO3 penalty：OPS Δ +0.262 — 嚴重，Baz 5 IP 後 WSH 攻勢爆。
- 🟠 HOME chain breaks at #3-4：OPS 落差 0.264 — 中度。
- 🔴 AWAY chain breaks at #8-9：OPS 落差 0.304 — 嚴重但對弱投手影響輕。
- 🔴 雙方牛棚 core IL ×2/×3 — 雙崩盤，雙方末段攻擊全面放大。

## 條件修正

- Park Factor: 100.0 → 0.00 run（Nationals Park 中性）
- 天氣：未公布（跳過天氣分析）
- 先發 tier：HOME Littell ⚪ Below Average vs AWAY Baz 真實 🟢 Back-end → 雙弱 starter
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.7 | +0.5（AWAY 牛棚 ×3 核心 IL 崩盤 + Baz TTO3 嚴重）| 5.2 |
| AWAY | 9.0 | +0.5（HOME 牛棚 ×2 核心 IL + Littell 真實 ⚪ Below Avg + LHB-heavy vs Baz） | 9.5 |
| Total | 13.7 | +1.0 | 14.7 |

## 整體判斷

- **方向（基本面）**：**AWAY (BAL)**。Littell vs Baz 雖然都弱，但 Littell xFIP 5.21 + K-BB% 2.4% 顯著比 Baz xFIP 4.02 + K-BB% 9.6% 更弱；WSH 雖然 base 9.0 已偏高（反映 Baz 真實 vs Littell 表面 ERA），實際差距更大；BAL 中心 vs Littell vs LHB 弱點不算完全踩中（BAL 多 RHB），但 Basallo / Henderson 可吃。
- **總分（基本面）**：**14.7 偏高，落點 12.5-16.0**。雙弱 starter + 雙崩盤牛棚 + 雙方都有 vs 對方 RHP 強配對 → Total 極高。Nationals Park 中性無壓制。
- **方向信心**：**60%**（AWAY 有利）— BAL 進攻基準明確較高，但 BAL last7 BABIP .261 偏冷期 + WSH 進攻火力 last7 散亂；方向確定但幅度不確定。
- **風險**：
  1. Basallo last7 BABIP **.524** + OPS 1.142 — 火燙不可持續，部分回歸壓力
  2. WSH Wood / Abrams vs RHP OPS .961 / 1.022 真實，配合 Baz vs LHB 弱點 → WSH 進攻仍可能爆但 base 9.0 已含
  3. 雙方牛棚崩盤可能讓比分超過 Total 預期更多（13.7+ 區間），紀律上不超 cap
  4. WSH 牛棚 ×3 核心 IL 是極端情況，若 Baz 撐 6+ IP（與 TTO3 預期反向），BAL 末段攻擊放大失敗

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
