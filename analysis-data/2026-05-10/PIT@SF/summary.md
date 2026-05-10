## 投手對決

### Tyler Mahle (HOME, RHP, 31 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p79, K-BB% p50），gap vs ERA-only = +35.0
  - **不完全同意 + 強調混合信號**：ERA 5.00 看起來像 ⚪ Below Average 但 xFIP **3.69** / K-BB% 10.1% / FIP 4.88 / xERA 4.68 — 真實水平 🟢 Back-end ~ 🟡 Solid（ERA 4.0-4.5 區間）。tier_v2 認為腳本 ERA 嚴重低估真實水平 +35.0；本場按 🟢 Back-end ~ 🟡 Solid 對待。
- **Reverse platoon 信號（🔴 +0.446）**：vs RHB OPS **1.019**（76 BF）vs vs LHB OPS .573（82 BF）— 嚴重反向。
  - PIT 多 RHB 中段（O'Hearn L #3、Lowe L #4、Cruz L #2 — 實際 PIT 多 LHB？等等：Reynolds S, Cruz L, O'Hearn L, Lowe L, Gonzales R）— 實際 PIT 多 LHB（Cruz/O'Hearn/Lowe LHB），Mahle 對 LHB 反而強壓制（OPS .573）→ Mahle 配對 PIT LHB 多陣容是優勢。Reverse platoon vs RHB 風險集中在 Reynolds 切換 & Gonzales R。
- **單一球種 FF 48.1%**：影響輕。
- **TTO3 penalty -0.109 + K% -7.8pp**：第三輪 K% 崩跌但 OPS 反向，混合信號。
- **對手打線威脅**：🟠 高。PIT matchup tier 🟠 Strong (vs RHP) — Reynolds vs RHP .790 / O'Hearn .878 / Lowe vs RHP **1.117** last7 1.223（火燙）/ Gonzales .831 — 中心強。

### Bubba Chandler (AWAY, RHP, 23 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = ⚪ Below Average（xFIP p5, K-BB% p11），gap vs ERA-only = -13.4
  - **同意**：ERA 4.76 / xERA 5.11 / FIP 5.48 / xFIP 5.14 / K-BB% **3.3%**（極低）/ velo 94.8 max 101.3（球速強但球質不轉化結果）— 真實水平 ⚪ Below Average。本場按 ⚪ Below Average 對待。
- **單一球種 FF 54.3%**：FF-heavy 但球質強（max 101.3），SF 中心可能 sit fastball 但 Devers 等都偏冷期。
- **TTO3 penalty（🔴 +0.407 → OPS 1.104）**：第三輪極度爆掉，SF 中段第三輪可能反彈（雖然 last7 BABIP .238 偏低）。
- **vs LHB OPS .810（.485 SLG）**：對 LHB 嚴重弱點，SF 多 LHB（Devers L #3、Lee L #4、Arraez L #5）。
- **對手打線威脅**：🟡 中等。SF matchup tier 🟡 Average (vs RHP) — Chapman vs RHP .575 last7 .112 / Adames .626 / Devers vs RHP .590 last7 .998 / Lee .717 / Arraez .787 — Devers last7 0.998 是真實爆分點，其他人冷期。

## 打線評級

### HOME — season tier 🟢 Weak / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 比 season tier 上修一檔；Devers last7 .998 補強。
- **chain_break 信號（🟠 #7-8）**：影響輕。
- **last7 BABIP .238 Flag 3 unlucky-cold**：見風險段。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong — 比 season tier 上修一檔；Lowe vs RHP 1.117 火燙。
- **chain_break 信號（🟠 #8-9）**：影響輕。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.85 / 9 / **4 名（Miller + Birdsong + 2 others）** | 4.13 / 2 / **0 名核心** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（SF）：ERA 3.85 中段稍弱 + **4 核心 IL → 🔴🔴 極高**（牛棚崩盤級）→ PIT Lowe/Reynolds 等中心末段攻擊極大化。
- AWAY 牛棚（PIT）：ERA 4.13 中段稍弱，無核心 IL。整體後段對 SF 弱進攻仍 OK。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.238) **unlucky-cold**:
  - **可能反彈**：SF last7 BABIP .238 偏低，Devers last7 OPS .998 已顯示反彈跡象。本場對 Chandler ⚪ Below Average → 反彈 + 數據反彈雙重，SF base 5.0 可能略低。

### 額外信號
- 🔴 HOME reverse platoon Δ +0.446 — Mahle vs RHB 1.019 是真實弱點；但 PIT 多 LHB，影響反向（Mahle 對 PIT LHB 反而壓制）。
- 🟠 HOME single-pitch dependent FF 48.1% / 🟠 AWAY 54.3% — 雙方都 FF-heavy；對對方打線 sit fastball 是雙向風險。
- 🟠 HOME TTO3 K% -7.8pp — Mahle 第三輪壓制力下降。
- 🔴 AWAY TTO3 OPS 1.104 — Chandler 第三輪極度爆掉，SF 中段第三輪可能爆分。
- 🟠 雙方 chain break — 影響輕。
- 🔴 HOME 牛棚 4 核心 IL — PIT 末段攻擊極大化關鍵。

## 條件修正

- Park Factor: 91.0 → -0.45 run（Oracle Park 嚴重投手友善 + HR -17%）
- 天氣：未公布（跳過天氣分析）— SF 海風常吹進場，壓 HR
- 先發 tier：HOME Mahle 🟢 Back-end vs AWAY Chandler ⚪ Below Average → 雙弱，HOME 微優
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.0 | 0（核心 IL 0 名） | 5.0 |
| AWAY | 5.1 | +0.8（HOME 牛棚 4 核心 IL → PIT 末段攻擊極大化，cap 上限） | 5.9 |
| Total | 10.1 | +0.8 | 10.9 |

## 整體判斷

- **方向（基本面）**：**AWAY (PIT) 中度有利**。Chandler ⚪ Below Average + SF 進攻冷期但 Devers 反彈 → SF 進攻面有空間；Mahle 真實 🟢 Back-end + SF 牛棚 4 核心 IL → PIT Lowe/Reynolds 等中心爆分機率極高。PIT 連勝 + 主場 G1 13:3 大勝後狀態強。
- **總分（基本面）**：**10.9（base 10.1 + +0.8 信號）**，落點 9.5-12.5。雙弱 starter + SF 牛棚崩 + Oracle PF 91 壓 Total → 中性偏上行；Oracle 海風壓 HR 部分抵消。
- **方向信心**：~62%（AWAY），結構性支撐（Chandler 弱 + SF 牛棚崩 + Mahle reverse vs RHB 但 PIT 多 LHB 反而對 PIT 不利反向）。
- **風險**：
  1. **Mahle reverse platoon vs RHB 1.019 但 PIT 多 LHB** — Mahle 配對 PIT LHB 是優勢，PIT 進攻面 base 5.1 可能偏高
  2. SF Devers last7 .998 + last7 BABIP 反彈 — 真實爆分點，可能單棒打破
  3. Chandler K-BB% 3.3% + xFIP 5.14 + TTO3 1.104 — 本場可能 4 IP 5R 早下
  4. Oracle Park 海風（未公布天氣）— 壓 HR 但 SF/PIT 雙方都受影響

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
