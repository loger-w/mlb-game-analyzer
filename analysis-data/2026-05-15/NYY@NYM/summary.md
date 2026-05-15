## 投手對決

### Clay Holmes (HOME, RHP, 33 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p80, K-BB% p56），gap vs ERA-only = -24.3
  - **同意 + ERA 高估**：ERA 1.86 看起來像 Elite 但 xERA **3.75** / FIP 3.37 / K-BB% 11.0% — 真實水平 🟡 Solid ~ 🟠 Strong（3.5 ERA 區間）。-24.3 gap 主因運氣（BABIP 偏低 + LOB% 偏高）。本場按 🟠 Strong Ace 對待但保留向上回歸風險。
- **Single-pitch dependent（🟠）**：SI 50.6% — 但 ST 17.7% + CH 14.7% 三球種組合多樣，影響輕。
- **vs RHB 極端壓制**：vs RHB **.171/.250/.211**（OPS .461）+ vs LHB .596 — 雙邊都壓制。
- **對手打線威脅**：🔴 高。NYY matchup tier 🟠 Strong (vs RHP) — Judge vs RHP **1.007** + Bellinger .832 + Rice vs RHP **1.073** — 中心 3 棒全火力齊備，Judge/Rice 是 Holmes 真威脅（Judge vs RHB 是 Holmes 弱點檢驗）。

### Cam Schlittler (AWAY, RHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +1.1
  - **完全同意**：ERA 1.35 / xERA 2.49 / FIP **1.64** / xFIP 2.56 / K-BB% **24.7%** / WHIP **0.81** / velo 95.5（max 101.3）/ vs LHB OPS .492 / vs RHB **.147/.179/.200**（OPS .379 極端壓制）— 全項頂級。本場按 🔴 Elite Ace 對待。
- **TTO3 penalty**：OPS Δ +0.040（K% Δ -17.2pp） — K 率掉但未轉成 OPS 爆發，影響輕。
- **對手打線威脅**：🟢 極低。NYM matchup tier 🟢 Weak (vs RHP) — Bichette .542 / Semien .594 / Benge .640 / Baty .630 / Vientos .668 — 全 vs RHP 平庸 + NYM last7 BABIP **.223** 冷期 + heat 🥶 Cold → 對 Schlittler Elite Ace 接近 shut out。

## 打線評級

### HOME — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak — 比 season tier 下修一檔；對 Schlittler 強壓制 → 進攻接近 shut out。
- **chain_break 信號（🔴）**：#6-7 OPS 落差 0.340 — 嚴重後段，但 NYM 中心都弱，影響輕。
- **Flag 3 last7 BABIP .223** — 冷期（見風險段）。

### AWAY — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong — 與 season tier 一致；Judge/Rice anchor 真實。
- **chain_break 信號（🔴）**：#5-6 OPS 落差 **0.502** — 極端後段，但前 5 棒 (Judge/Bellinger/Chisholm/Grisham/Rice) 火力齊備，影響輕。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.54 / 6 / **3 名（🔴🔴 極高）** | 3.34 / 3 / **0 名核心** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（NYM）：ERA 3.54 中段，**3 名核心 IL**（Minter + Núñez + 1）→ 🔴🔴 崩盤級。Díaz (closer) 健康但 setup 群崩盤。配合 Holmes 真實 🟠 Strong（可能 5-6 IP 後離場），NYM 中繼對 Judge/Bellinger/Rice 是 nightmare。
- AWAY 牛棚（NYY）：ERA 3.34 elite + 無核心 IL → 完整可用。Williams (closer) 健康。後段對 NYM 弱進攻完全壓制。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-1.89):
  - **強運氣加持**：Holmes ERA 1.86 vs xERA 3.75 gap -1.89，主因 BABIP 偏低。但 FIP 3.37 + vs RHB .211 SLG 真實壓制力存在，本場按 🟠 Strong 區間 (3.0-3.5 ERA) 對待。**不自動下修**。
- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.223):
  - **嚴重持續**：NYM 7 場樣本 BABIP .223 + heat Cold + 全打線 vs RHP 平庸 + Schlittler Elite Ace → 三重壓力，本場仍難反彈。**不自動 ±run value**，base 1.7 可能準確（接近 shut out）。

### 額外信號
- 🟠 HOME single-pitch dependent：Holmes SI 50.6% — 三球種補強，影響輕。
- 🟠 HOME TTO3 penalty：OPS Δ -0.036 反向 — Holmes 越投越穩。
- 🟠 AWAY TTO3 penalty：OPS Δ +0.040 — Schlittler 第三輪輕度衰退但 OPS 仍 .500。
- 🔴 HOME chain breaks at #6-7：OPS 落差 0.340 — 嚴重但 NYM 中心都弱，影響輕。
- 🔴 AWAY chain breaks at #5-6：OPS 落差 0.502 — 極端但 NYY 前 5 棒齊備。
- 🔴 HOME 牛棚 core IL ×3：🔴🔴 崩盤級 — Holmes 早下後 NYM 中繼對 NYY 中心是噩夢。

## 條件修正

- Park Factor: 96.0 → -0.20 run（Citi Field 中性偏輕度投手友善，HR +7%）
- 天氣：未公布（跳過天氣分析）
- 先發 tier：HOME Holmes 真實 🟠 Strong vs AWAY Schlittler 🔴 Elite Ace → AWAY 投手戰嚴重優勢
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 1.7 | 0（Schlittler Elite Ace 強壓制 + NYM 三重壓力 = base 1.7 已準確） | 1.7 |
| AWAY | 3.9 | +0.5（HOME 牛棚 ×3 核心 IL 崩盤 + Judge/Rice vs Holmes 弱點 RHB 壓制有限） | 4.4 |
| Total | 5.6 | +0.5 | 6.1 |

## 整體判斷

- **方向（基本面）**：**AWAY (NYY)**。Schlittler Elite Ace vs Holmes 真實 🟠 Strong + ERA 假象 + 33 歲退化；NYY 中心 Judge/Rice vs RHP 全 1.000+ OPS vs NYM 全打線 vs RHP 平庸 + cold + Flag 3 — 投手戰 + 進攻雙優。
- **總分（基本面）**：**6.1 偏低，落點 5.0-7.0**。Schlittler 強壓制 NYM 弱進攻接近 shut out + Holmes 真實 🟠 Strong 但 NYM 牛棚崩盤後段 → Total 由 NYY 單邊得分驅動。
- **方向信心**：**70-75%**（AWAY 有利）— Schlittler vs Holmes tier 落差 + NYM 進攻三重壓力 + NYM 牛棚崩盤是硬數據；NYM 連勝 3 主場仍有狀況面壓力但難擋投手戰。
- **風險**：
  1. Holmes ERA 1.86 雖然 Flag 8 但 FIP 3.37 真實壓制 — NYY 中心可吃但不會爆分
  2. NYM 連勝 3 + 主場 — 狀況面對抗，但 Schlittler 統治力太強難擋
  3. Schlittler GS 9 樣本仍中等 — 本場可能任一方向波動，但 K-BB% 24.7% 是高度可信基礎
  4. Citi Field HR +7% — Judge/Rice barrel% 24.5/20.4 可能 HR

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
