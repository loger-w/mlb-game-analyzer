## 投手對決

### Davis Martin (HOME, RHP, 29 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = -0.6
  - **同意但謹慎**：ERA 1.64 / xERA **4.20** / FIP 2.28 / xFIP 2.93 / K-BB% 20.7% — FIP-base elite 但 xERA 4.20 顯示嚴重運氣加持。本場按 🟠 Strong Ace 對待較合理。
- **Flag 8 era_xera_delta=-2.56**：嚴重運氣加持，BABIP 偏低 + LOB% 偏高。但 K-BB% 20.7% 是真實 elite。
- **TTO3 penalty（🔴 +0.326，K% -16.2pp）**：第三輪嚴重崩盤，SEA 第三輪反彈關鍵。
- **對手打線威脅**：🟡 中等。SEA matchup tier 🟠 Strong (vs RHP) — Donovan vs RHP **1.150**（小樣本但火燙）/ Rodríguez .673 / Naylor .772 last7 .917 / Raley .898 — Donovan + Raley 是真實爆分點。

### Logan Gilbert (AWAY, RHP, 29 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p87），gap vs ERA-only = +39.2
  - **同意但謹慎**：ERA 4.30 / xERA 4.78 / FIP 4.19 / xFIP **3.16** / K-BB% 17.8% — xFIP-base 真的 elite 但 FIP 4.19 + barrel% **10.1%** 接觸品質有問題。本場按 🟠 Strong Ace 對待。
- **TTO3 penalty（🔴 +0.354）**：第三輪 OPS 暴升 1.138，極度嚴重；CWS 中段 Murakami/Vargas/Montgomery 第三輪可能爆分。
- **vs LHB OPS .848（.343 OBP, .505 SLG, 105 BF）**：對 LHB 控球差，CWS 多 LHB（Vargas L #3、Montgomery S、Kelenic L、Romo S）。
- **對手打線威脅**：🟠 高。CWS 🔴 Elite season tier — 但 matchup tier vs RHP 下修為 🟡 Average（script）；實際 Murakami vs RHP **.945** / Vargas .677 last7 1.081 / Montgomery .804 / Romo **1.111** — 中段火力強，加上 Gilbert TTO3 weakness。

## 打線評級

### HOME — season tier 🔴 Elite / heat 🔥 Hot
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average — 比 season tier 嚴重下修兩檔，但個別打者強（Murakami/Vargas/Romo）。
- **chain_break 信號**：未 fired。

### AWAY — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong — 與 season tier 一致；Donovan/Raley 是真實火力。
- **chain_break 信號（🟠 #6-7）**：Raley .881 → Crawford .704 — 中度，影響輕。
- **last7 BABIP .249 Flag 3 警訊**：見風險段。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.58 / 4 / **1 名（Vasil setup）** | 3.32 / 5 / **3 名（Vargas + Speier + 其他）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（CWS）：ERA 4.58 偏弱，1 核心 IL → 🟠 中高。Civale (closer) 健康。對 SEA 中心壓制力中等。
- AWAY 牛棚（SEA）：ERA 3.32 中段強但 **3 核心 IL → 🔴🔴 極高**（牛棚崩盤級），多名 high-leverage RP 缺陣。若 Gilbert 早下，CWS 中心可能爆分末段。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-2.56):
  - **嚴重運氣加持但 K-BB% 真實**：Davis Martin ERA 1.64 不可持續，xERA 4.20 顯示真實水平 ~3.5 ERA。本場按 Strong Ace 對待，**不自動下修**。
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.249):
  - **可能部分回歸**：SEA last7 BABIP 偏低，本場對 Davis Martin 強壓制 + Davis Martin xERA 4.20 不可信 → 回歸 + 數據反彈雙重，SEA base 2.6 偏低。

### 額外信號
- 🔴 HOME TTO3 K% -16.2pp — Davis Martin 第三輪極度崩盤，SEA 第三輪是反彈關鍵。
- 🔴 AWAY TTO3 OPS +0.354 → 1.138 — Gilbert 第三輪極度爆掉，CWS 中段第三輪可能爆分。
- 🟠 AWAY chain break #6-7 — 影響輕。
- 🟠 HOME 牛棚 1 核心 IL — 影響中等。
- 🔴 AWAY 牛棚 3 核心 IL — 🔴🔴 極高，CWS 末段攻擊極大化。

## 條件修正

- Park Factor: 97.0 → -0.15 run（Rate Field 中性偏輕度投手友善，HR -1%）
- 天氣：Partly Cloudy **57°F**, wind 11 mph **In From LF** — 涼風 + 逆風左外野 → 顯著壓 RHB pull HR；對 RHB 多的兩隊都壓 HR
- 先發 tier：HOME Davis Martin 🟠 Strong Ace（運氣加持）vs AWAY Gilbert 🟠 Strong Ace（FIP-base 強）→ 對等
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.2 | +0.6（AWAY 牛棚 3 核心 IL → CWS 末段攻擊極大化） | 5.8 |
| AWAY | 2.6 | 0（核心 IL 1 名，未達 ≥2 門檻） | 2.6 |
| Total | 7.8 | +0.6 | 8.4 |

## 整體判斷

- **方向（基本面）**：**HOME (CWS) 中強有利**。Davis Martin ERA 1.64 + 運氣加持但 K-BB% 真實強 vs Gilbert 實際 FIP 4.19 / barrel% 10.1 + TTO3 1.138 → CWS 投手戰雙端優勢；CWS 中段 (Murakami/Vargas/Romo) 對 Gilbert vs LHB 弱點 + TTO3 weakness 是真實爆分點。SEA 牛棚 3 核心 IL 是壓死駱駝的稻草。
- **總分（基本面）**：**8.4（base 7.8 + +0.6 信號）**，落點 7.5-9.5。雙方先發都看似 elite 但實際都有運氣 + 結構問題；SEA 牛棚崩 + 涼風逆風 + Davis Martin TTO3 → Total 中性偏上。
- **方向信心**：~62%（HOME），結構性支撐（SEA 牛棚崩 + Gilbert TTO3 + Davis Martin 主場優勢）。
- **風險**：
  1. Davis Martin xERA 4.20 — 本場可能繼續運氣加持，CWS 投手戰可能延續
  2. Gilbert TTO3 +0.354 是 42 BF 樣本，可能回歸 — 但 K-BB% 17.8% 真實
  3. SEA Donovan vs RHP 1.150 / Raley last7 .934 — 可能單棒爆分打破投手戰
  4. 涼風 57°F + In From LF — Total 下行明顯，HR 機率被壓

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
