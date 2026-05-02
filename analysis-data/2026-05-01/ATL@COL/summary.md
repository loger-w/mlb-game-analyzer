## 投手對決

### Jose Quintana (HOME, LHP, 37 📉📉📉 快速退化)
- **Tier 覆寫**：⚪ Below Average（從 🟢 Back-end 降級）— K-BB% **-2.4%**（負值，極端警訊）+ FIP **6.75** vs ERA 4.91（gap +1.84，被四壞 + 球種價值雙崩）+ xERA 5.35（ERA 沒運氣加持）+ velo 86.2 mph（LHP 平均水平偏低，37 歲退化進行中）+ whiff% 8.5%（聯盟均 ~25%）。
- 真實水平判斷：本季 4 GS 數據與 Statcast 一致指向中後段 → 邊緣 starter。被 LHB 慘打（slash .250/.368/.625，雖 19 BF 樣本小但 SLG .625 警訊明確），對 RHB 也僅 .255/.339/.431。近 3 場 ER 9 / IP 13 → 區間 ERA 6.23，無止血跡象。
- 對手打線威脅：ATL 🟠 Strong（xwOBA .352 / OPS .801）+ vs LHP 火力集中（Albies vs LHP OPS **1.024** / Baldwin .918 / Olson .853 / Riley .848）→ 對 Quintana 是 nightmare matchup，Coors 球場再放大長球破壞力。

### Grant Holmes (AWAY, RHP, 30 📉 初期退化)
- **Tier 覆寫**：沿用 🟡 Solid Starter — ERA 3.62 體面，但 FIP 4.86 / xFIP 4.38 顯示有運氣成分（gap 1.24，6 GS 小樣本警訊）。velo 89.4 / max 96.9 中等偏低，但 SL 38.1% slider-heavy + vs LHB .197/.286/.344 與 vs RHB .220/.292/.356 雙邊壓制都不錯。
- 真實水平判斷：近 3 場 ER 5 / IP 17.7（區間 ERA 2.55）持續進步，但 hard_hit% 31.8% / FIP-ERA gap 暗示往回拉力存在。本場真實水平估 4.0-4.5 ERA 區間。
- 對手打線威脅：COL 主場 🟡 Average（xwOBA .319 / OPS .744）+ Coors 球場效應放大，主要威脅 Hunter Goodman vs RHP OPS .938 + last7 1.131 / TJ Rumfield vs RHP .820 — 但 Tovar/Karros 雙小將 vs RHP OPS .507/.563 分散風險。

## 打線評級

### HOME — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用 🟡 Average — 本季基本面數據（K% 27.1 偏高 / OPS .744）一致指向中後段打線；但 last7 BABIP .404 是 Flag 3 警訊（見風險段判讀），不足以升 Tier。

### AWAY — 🟠 Strong / ⚖️ Normal
- **Tier 覆寫**：沿用 🟠 Strong — xwOBA .352 / OPS .801 / chain OBP .369 / chain SLG .427 全聯盟前段；vs LHP 重點打者 Albies 1.024 / Baldwin .918 / Olson .853 / Riley .848 強化今晚對 Quintana 的優勢。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.81 / 4 / **0 名核心** | 3.32 / 9 / **2 名核心**（🔴 高） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（COL）：4 名 IL 主要為先發/depth（Feltner/Criswell/Brown/Ohl），Active 末段 Vodnik (closer) + Lorenzen + Bernardino 完整。整體 ERA 3.81 偏中段，對對手末段威脅普通。
- AWAY 牛棚（ATL）：**Raisel Iglesias (closer, 15-day IL)** + **Joe Jiménez (high-leverage setup, 60-day IL)** 雙核心缺陣 → 🔴 高 影響度。雖 Robert Suarez 來填補 closer 角色但壓力陡增；後段 7-9 局 leverage 走 Bummer/Kinley/Lee → 對 Coors 主場 COL 末段攻勢防守強度明顯下降。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.404):
  - **可能部分回歸 + 部分持續**：聯盟均 BABIP ~.300 需 ~800 AB 穩定，7 天樣本噪音極大 → 必有回歸壓力。但 Coors 球場特性（高反彈、外野廣袤、空氣稀薄使球速衰減小）天然支撐 COL 主場 BABIP 較高，**完全回歸到 .300 不合理**。預期區間 .310-.340，仍偏高但低於 .404。
  - **不自動 ±run value**，但敘事上對 COL 預期得分中性偏負（短期過熱）。

## 條件修正

- Park Factor: 131.0 → +1.55 run（5 月已恢復 Coors 全功率夏季模式，4 月修正不適用）
- 先發 tier：HOME Quintana ⚪ Below Average vs AWAY Holmes 🟡 Solid → 不對稱，AWAY 進攻面顯著優勢（已反映於 base formula 用 ERA 計算）
- doubleheader：無
- 天氣：5 月初 Denver 春末，溫度可能偏低（夜場 50-60°F），稍微抑制 HR 距離但相對 Coors 自身仍極端打者友善

## 修正後預期得分

> 「+ 信號」欄僅納入規範允許的條件修正：Park Factor、牛棚累計效應（核心 IL ≥ 2 名）、主力打者傷兵。
> ⛔ BABIP 極端值 / ERA-xERA gap **不入此欄**（規範禁止 auto ±run value，見 reference/flags-checklist.md §3, §8）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 6.9 | +1.0（ATL 牛棚 2 核心 IL → COL 末段得分機會 ↑） | 7.9 |
| AWAY | 10.6 | 0 | 10.6 |
| Total | 17.5 | +1.0 | 18.5 |

## 整體判斷

- **方向（基本面）**：**AWAY (ATL) 顯著有利**。Quintana 數據全面崩盤（K-BB% -2.4 / FIP 6.75 / xERA 5.35）對上 ATL 強打 + vs LHP 集中火力（Albies/Baldwin/Olson）+ Coors PF 131 → 三重利空疊加。Holmes 雖有 ERA-FIP gap 隱憂但 vs LHB/RHB 雙邊壓制 + 近 3 場狀況上揚。
- **總分（基本面）**：**偏 HIGH，落點 17.5–19.5**。Coors 全功率 + Quintana 投不出 quality start + ATL 牛棚弱（後段易失分）→ Total 不易壓低。
- **信心**：**MEDIUM-HIGH** — 基本面方向極強（Quintana 退化是硬數據），但變數：Holmes 樣本 6 GS、ATL 牛棚 IL 影響真實大小、COL last7 BABIP 回歸幅度。
- **風險**：
  1. Holmes ERA 3.62 vs FIP 4.86（gap 1.24）+ hard_hit% 31.8 — 6 GS 樣本，本場可能向 FIP 4.5+ 區間回歸，特別在 Coors 空氣稀薄環境
  2. COL last7 BABIP .404 屬極端值，部分回歸（預期 .310-.340 區間）→ 短期過熱風險，COL 進攻基準偏低
  3. ATL Albies last7 OPS **1.426** / Olson 1.193 / Acuña .856 — 同樣火燙手感極端，平均回歸壓力對等存在
  4. ATL 牛棚 Iglesias (closer 15d) + Jiménez (60d) 雙核心 IL — 若 Holmes 5 局內離場交給 Bummer/Kinley，COL 主場攻勢可能放大；反向若 Holmes 撐 6+ 局則信號弱化

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
