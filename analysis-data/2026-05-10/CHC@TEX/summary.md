## 投手對決

### Jacob deGrom (HOME, RHP, 37 📉📉📉 快速退化)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +12.7
  - **同意**：ERA 3.11 / xERA 3.45 / FIP 3.31 / xFIP 2.59 / K-BB% **26.0%** / WHIP 1.01 / velo 93.2 — 各項一致 elite。37 歲快速退化但本季數據撐住。本場按 🔴 Elite Ace 對待。
- **Reverse platoon 信號（🔴 +0.217）**：vs RHB OPS .775（47 BF）vs vs LHB OPS .558（103 BF）— RHP 對 RHB 反而被打。
  - CHC 多 RHB 中段（Suzuki R / Happ S / Kelly R / Crow-Armstrong R），但 deGrom vs RHB 樣本僅 47 BF — 可能樣本噪音。
- **單一球種依賴（🟠 FF 46.1%）**：FF 球質強，影響輕。
- **TTO3 penalty（🟠 +0.132，career fallback）**：第三輪 OPS 上升，CHC 中段第三輪可能反彈。
- **對手打線威脅**：🟠 高。CHC matchup tier 🟡 Average (vs RHP) — Suzuki vs RHP **.984** last7 1.006 / Happ .933 / Busch last7 **1.227** / Crow-Armstrong last7 .993 — 中段 last7 火燙，是真實爆分群。

### Jameson Taillon (AWAY, RHP, 34 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p58, K-BB% p70），gap vs ERA-only = +11.7
  - **同意**：ERA 4.24 / xERA 3.95 / FIP **5.90**（差）/ xFIP 4.00 / K-BB% 13.5% / barrel% 12.4% / velo 86.0（極低 p5）— 中段水平 🟡 Solid 但 FIP 5.90 警訊（HR/9 偏高）。本場按 🟢 Back-end ~ 🟡 Solid 對待。
- **vs LHB 弱點（.217/.308/.507 SLG，78 BF）**：對 LHB 控球 OK 但 SLG .507 power 漏；TEX 多 LHB（Carter L #5、Foscue L #9、Pederson L #6）。
- **TTO3 penalty（-0.181）**：第三輪反向，影響輕。
- **對手打線威脅**：🟠 高。TEX matchup tier 🟠 Strong (vs RHP) — Duran vs RHP **.890** last7 1.019 / Seager .784 / Jung .957 — 中心強，加上 Foscue vs RHP .834 last7 1.000。

## 打線評級

### HOME — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong — 與 season tier 一致；前 4 棒 Nimmo/Duran/Seager/Jung 火力齊備。
- **chain_break 信號（🟠 #4-5）**：Jung .891 → Carter .631 — 中度，但 Carter vs RHP .729 仍 OK。
- **platoon advantage（🟠）**：top 5 中 4 人對 RHP OPS 強配對。
- **last7 BABIP .243 Flag 3 警訊**：見風險段。

### AWAY — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average — 比 season tier 下修；但 Suzuki/Happ vs deGrom 仍是真實威脅。
- **last7 BABIP .259 Flag 3 警訊**：見風險段。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | **2.84** / 6 / **0 名核心** | 3.81 / 9 / **4 名（Thielbar + Harvey + 2 others）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（TEX）：ERA **2.84** elite，無核心 IL！整體後段壓制力極強，對 CHC 中心末段是真實壓制。
- AWAY 牛棚（CHC）：ERA 3.81 中段稍弱 + **4 核心 IL → 🔴🔴 極高**（牛棚崩盤級）！TEX Duran/Seager/Jung 等中心末段攻擊極大化。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.243):
  - **可能部分回歸**：TEX last7 BABIP 偏低，但 EV/Barrel 數據（Nimmo 51.3 / Seager 48.0）強，回歸 + 數據反彈雙重，本場對 Taillon FIP 5.90 → TEX 進攻面有反彈空間。
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.259):
  - **可能部分回歸 + 部分持續**：CHC last7 偏低，但對 deGrom Elite 強壓制 → 回歸不易發生在本場。CHC base 3.7 偏高，實際可能 2.5-3.5。

### 額外信號
- 🔴 HOME reverse platoon Δ +0.217 — deGrom vs RHB 47 BF 可能噪音，但 CHC 多 RHB 中段值得留意。
- 🟠 HOME single-pitch FF 46.1% — 球質強，影響輕。
- 🟠 HOME TTO3 penalty +0.132 — CHC 第三輪反彈關鍵。
- 🟠 HOME platoon advantage — TEX 對 Taillon 強配對。
- 🟠 HOME chain break #4-5 — 影響輕。
- 🔴 AWAY 牛棚 4 核心 IL — TEX 末段攻擊極大化關鍵。

## 條件修正

- Park Factor: 96.0 → -0.20 run（Globe Life Field 偏輕度投手友善但 HR +6%）
- 天氣：室內（Roof Closed，不適用）
- 先發 tier：HOME deGrom 🔴 Elite Ace vs AWAY Taillon 🟢 Back-end ~ 🟡 Solid → 嚴重不對稱
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 6.6 | +0.8（AWAY 牛棚 4 核心 IL → TEX 末段攻擊極大化，cap 上限） | 7.4 |
| AWAY | 3.7 | 0（核心 IL 0 名） | 3.7 |
| Total | 10.3 | +0.8 | 11.1 |

## 整體判斷

- **方向（基本面）**：**HOME (TEX) 強勢有利**。deGrom Elite Ace vs Taillon FIP 5.90 + CHC 牛棚 4 核心 IL → TEX 投手戰 + 進攻雙優。雖然 deGrom reverse platoon vs RHB 是潛在弱點但 47 BF 樣本可能噪音。CHC 中段 Suzuki/Busch last7 火燙是 deGrom 唯一真實威脅。
- **總分（基本面）**：**11.1（base 10.3 + +0.8 信號）**，落點 9.5-12.5。Taillon FIP 5.90 + CHC 牛棚崩 + TEX 強打 → Total 上行；但 deGrom 強壓制限制 CHC 上限。
- **方向信心**：~70%（HOME），結構性強支撐（CHC 牛棚 4 核心 IL + Taillon FIP 5.90）。
- **風險**：
  1. **CHC 連勝近 10 場（9-1）+ G1 0:6 失利可能反彈** — 但客場 + Taillon 實際 ⚪ Below Average，反彈力有限
  2. deGrom 37 歲快速退化 — 可能本場早下，CHC Suzuki/Busch 對 TEX 牛棚（ERA 2.84 elite）末段不易爆分
  3. CHC last7 BABIP .259 偏低 — 部分回歸但對 deGrom 不易發生
  4. TEX last7 BABIP .243 + Foscue/Duran last7 火燙 — 真實爆分點

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
