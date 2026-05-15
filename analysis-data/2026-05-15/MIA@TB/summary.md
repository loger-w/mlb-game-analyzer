## 投手對決

### Jesse Scholtens (HOME, RHP, 32 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 沒給定（GS 2 / 15.3 IP 樣本不足）。原始 tier 🟡 Solid Starter — ERA 3.29 / xERA **4.22** / FIP **4.56** / xFIP 4.14 / K-BB% 9.6% / whiff% 9.0 — 數據面是 🟢 Back-end ~ 🟡 Solid，ERA 偏低主因小樣本 + 運氣。
  - **本場按 🟢 Back-end**（ERA 4.0-4.5 區間）對待。
- **TTO3 penalty 嚴重（career fallback）**：OPS Δ **+0.367**（TTO1 .626 → TTO3 **.993**）+ K% 從 22.2% 掉到 9.4% — 極端第三輪衰退，5 IP 後 disaster。
- **對手打線威脅**：🟡 中等。MIA matchup tier 🟢 Weak (vs RHP) 但 Edwards vs RHP **.856** last7 .958 / Lopez .775 last7 .808 / Hicks vs RHP **.935** — 中段 3 棒可吃 Scholtens 中後段。

### Janson Junk (AWAY, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p80, K-BB% p66），gap vs ERA-only = +2.7
  - **謹慎同意**：ERA 3.25 / xERA 3.83 / FIP 3.46 / xFIP 3.67 / K-BB% 12.7% / WHIP 1.11 / vs RHB **.149/.186/.224**（極端壓制 RHB）— 數據面真實 🟠 Strong。但 vs LHB .287/.336/.465（OPS .801）是相對弱點。
  - **本場按 🟠 Strong Ace**（ERA 3.0-3.5 區間）對待。
- **TTO 反向**：OPS Δ **-0.306**（TTO1 .772 → TTO3 .466）— 第三輪反而 K% 提升，Junk 越投越穩，可撐 6+ IP。
- **對手打線威脅**：🟡 中等。TB matchup tier 🟢 Weak (vs RHP) — Caminero vs RHP .809 / Aranda vs RHP **.928** last7 .960 / Díaz .806 — 中段三棒對 Junk vs LHB 弱點配對良好（Aranda LHB）。

## 打線評級

### HOME — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 比 season tier 上修一檔（vs RHP 表現超過整體）；Aranda LHB 可吃 Junk vs LHB OPS .801。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak — 比 season tier 下修一檔；但對 Scholtens TTO3 嚴重衰退 + 樣本薄，MIA 中段 (Edwards/Lopez/Hicks) 仍可吃。
- **Flag 3 last7 BABIP .228** — 偏低（見風險段）。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.18 / 7 / **3 名（🔴🔴 極高）** | 3.34 / 3 / **1 名（🟠 中高）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（TB）：ERA 4.18 中段，**3 名核心 IL** → 🔴🔴 崩盤級。配合 Scholtens TTO3 +0.367 5 IP 內離場機率極高 → TB 中繼後段對 MIA 中段攻擊容錯極低。
- AWAY 牛棚（MIA）：ERA 3.34 中段穩定，Henriquez IL60d 是 1 核心 IL → 🟠 中高。Junk 預期 6+ IP（TTO 反向），後段壓力小。

## 風險提示

- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.228):
  - **可能反彈 + 部分持續**：MIA 7 場樣本 BABIP .228 偏低（冷期），但對 Scholtens 🟢 Back-end + TB 牛棚崩盤後段 → 部分反彈合理。**不自動 ±run value**，敘事上 MIA base 4.9 可能往 5.5+ 走。

### 額外信號
- 🔴 HOME TTO3 penalty：OPS Δ +0.367（career） — 極端，Scholtens 5 IP 後 MIA 攻勢爆。
- 🟠 雙方 chain breaks at #4-5：中度，影響輕。
- 🔴 HOME 牛棚 core IL ×3：🔴🔴 極高 — 配合 Scholtens 早下 → TB 整場後 5-6 IP 全靠中繼，崩盤級。
- 🟠 AWAY 牛棚 core IL ×1：🟠 中高 — Junk 6+ IP 預期下影響輕。

## 條件修正

- Park Factor: 100.0 → 0.00 run（Tropicana Field 中性，HR +9%）
- 天氣：室內球場（無天氣修正）
- 先發 tier：HOME Scholtens 🟢 Back-end vs AWAY Junk 🟠 Strong Ace → AWAY 投手戰嚴重優勢
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.6 | -0.2（Junk Strong Ace 預期壓制 TB 弱攻擊 + TTO 反向撐 6+ IP） | 3.4 |
| AWAY | 4.9 | +0.6（HOME 牛棚 ×3 核心 IL 崩盤 + Scholtens TTO3 +0.367 嚴重 + MIA 中段攻擊配對） | 5.5 |
| Total | 8.5 | +0.4 | 8.9 |

## 整體判斷

- **方向（基本面）**：**AWAY (MIA)**。Junk 真實 🟠 Strong Ace + TTO 反向（越投越穩）vs Scholtens 真實 🟢 Back-end + TTO3 +0.367 極端衰退；TB 牛棚 ×3 核心 IL 崩盤是壓垮駱駝的最後稻草。MIA 雖 Flag 3 冷期但對 Scholtens + TB 牛棚崩盤組合，攻擊面有真實放大空間。
- **總分（基本面）**：**8.9 接近實際，落點 8.0-10.0**。Junk 壓制 TB 弱進攻 + MIA 中段攻擊吃 Scholtens + TB 牛棚崩盤 — Total 中等偏高。
- **方向信心**：**60-65%**（AWAY 有利）— Junk vs Scholtens tier 落差 + TB 牛棚崩盤是硬數據；但 TB 連勝 7-3 + 主場仍有狀況面壓力。
- **風險**：
  1. MIA last7 BABIP .228 冷期 — 若繼續冷，base 4.9 偏高；但對 Scholtens + TB 牛棚崩盤組合，反彈機率高
  2. TB 近 30 RS 4.30 / RA 3.17 雙優 + 主場 28-15 — 整體狀況強，可能任一場爆冷
  3. Scholtens 2 GS 小樣本 — 可能本場壓制 5 IP 1R 反常，TTO3 career fallback 不是本季數據
  4. Tropicana HR +9% — TB 端 Caminero/Aranda barrel% 9.9-10.6 可能 HR

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
