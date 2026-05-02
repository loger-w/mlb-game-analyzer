## 投手對決

### Max Meyer (HOME, RHP, 27 ⚡ 巔峰期)
- **Tier 覆寫**：沿用 🟡 Solid Starter — FIP 2.97 / xFIP 3.40 / K-BB% 15.9 / WHIP 1.20 全部紮實，但 xERA 4.46 比 ERA 3.30 高 1.16，暗示 Statcast 反指 ERA 有些幸運（Hard Hit% 24.9 偏低，但 barrel% 9.5 偏高）。
- 真實水平判斷：本季 6 GS 數據以 FIP 為主訊號 → 真實 ERA 區間 **3.50-4.00**。Slider/Sweeper heavy（SL 27.4 + ST 26.0 = 53%）對 RHB 是強武器。velo 90.9（27 歲 RHP 偏低但維持），whiff 13.8 中等。
- 對手打線威脅：PHI 🟢 Weak（OPS .667）但中段 Schwarber/Harper/Turner 個別 vs RHP 都頂級（OPS 1.105 / .880 / .815）— Schwarber EV95 50.7 + barrel 21.1 是 Meyer 的最大威脅。Bohm（vs RHP .420）+ Stott 中段拖低期望值。

### Andrew Painter (AWAY, RHP, 23 📈 成長期)
- **Tier 覆寫**：升級 🟢 Back-end Starter（從 ⚪ Below Average 上調）— ERA 5.25 看似爛，但 **FIP 3.18 / xFIP 3.73 / xERA 3.97** 三個指標一致指向 Painter 真實水平在 Solid Starter 邊緣。ERA-FIP gap **-2.07 極端**，4 GS 小樣本被 sequencing/BABIP 反咬。velo max 98.7（最快球速比 Meyer 高），FF 38.2 / SL 16.8 為主。
- 真實水平判斷：頂級新秀（Phillies 第一順位 prospect），Statcast hard_hit% 20.4 + barrel% 6.3 都極優 → 並非「被打爆」型。但需注意 vs RHB 慘（.343/.415/.514，41 BF 樣本）— MIA 有 Norby/Lopez/Edwards 多名 RHB，是隱憂。
- 對手打線威脅：MIA 🟢 Weak（OPS .722，xwOBA .298）但中段 Otto Lopez vs RHP .869 + Edwards .903 + Norby .786 對 Painter 上風一致；Marsee/Ramirez 偏弱拖低期望。整體威脅中等，主要風險在 Painter vs RHB 弱點被放大。

## 打線評級

### HOME — 🟢 Weak / ⚖️ Normal
- **Tier 覆寫**：沿用 🟢 Weak — xwOBA .298 / OPS .722 / chain SLG mid .374，但 chain OBP top3 .364 還算可接受，主要靠製造機會而非長球；近 7 BABIP .303 健康無極端。

### AWAY — 🟢 Weak / ⚖️ Normal
- **Tier 覆寫**：沿用 🟢 Weak（雖團隊 OPS .667）— PHI 是「重砲頭重腳輕」型，Schwarber/Harper/Turner 三人 vs RHP 都 .815+ OPS，中後段（Bohm .446 / Stott / Marsh）極弱。chain SLG mid **.298** 是聯盟尾段，得分仰賴前 3 棒爆發。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.78 / 3 / **1 名核心**（Pete Fairbanks closer 15d）| 4.40 / 4 / **1 名核心**（Jhoan Duran closer 15d） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（MIA）：ERA 3.78 中段，**Fairbanks (closer) 15-day IL** → Anthony Bender / Calvin Faucher 補位，後段 leverage 中等。Active 名單還有 Sandy Alcantara（轉牛棚？或長 relief），整體深度尚可。
- AWAY 牛棚（PHI）：ERA **4.40** 偏高 + **Duran (closer) 15-day IL** → Alvarado / Kerkering 補位 closer。但 ERA 4.40 是隱憂（PHI 牛棚整體不穩）。對方末段 leverage 下降明顯。
- **雙邊都損失 closer**：相互抵消 ML 影響，但對 Total 是 +run 信號（兩隊末段都易失分）。

## 風險提示

無 ⚠️ Flag 觸發（dossier 無 BABIP / pitcher health 風險標註）。

## 條件修正

- Park Factor: 106.0 → +0.30 run（loanDepot park 微利打者，HR -6% 抑制長球但利安打/三壘打）
- 先發 tier：HOME Meyer 🟡 Solid vs AWAY Painter 🟢 Back-end（覆寫後）→ 主隊先發優勢，對 ML 利 MIA
- 天氣：5 月初 Miami 室內球場（loanDepot 有屋頂），無天氣變數
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄僅納入規範允許的條件修正：Park Factor、牛棚累計效應（核心 IL ≥ 2 名）、主力打者傷兵。
> ⛔ BABIP 極端值 / ERA-xERA gap **不入此欄**（規範禁止 auto ±run value，見 reference/flags-checklist.md §3, §8）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.4 | 0（核心 IL 僅 1 名，未達 ≥ 2 門檻）| 3.4 |
| AWAY | 3.3 | 0（同上） | 3.3 |
| Total | 6.7 | 0 | **6.7** |

## 整體判斷

- **方向（基本面）**：**略偏 HOME (MIA)** — Meyer 🟡 Solid 顯著優於 Painter 🟢 Back-end（雖 Painter Statcast 反指可能比 ERA 顯示的好）；MIA 主場 + chain 串聯能力略佳。但差距不大（ML 預期約 51-53%）。
- **總分（基本面）**：**強烈偏 LOW，base 6.7**。formula 顯著低於市場線（Pinnacle 8.5），**1.8 runs gap**。兩個 RHP FIP 都在 2.97-3.18 區間，是相當好的雙投對決；兩隊打線中後段都疲軟。**真實區間估 7.0-8.0**（formula 可能略保守，但仍明顯 Under 8.5）。
- **信心**：**LOW** — Painter 4 GS 樣本 + ERA-FIP gap -2.07 極端，真實水平判讀有顯著不確定性；Sharp 信號方向（Over +3.9pp）與 base formula（強 Under）divergent，建議格外謹慎。
- **風險**：
  1. **Painter 樣本僅 4 GS**，ERA 5.25 vs FIP 3.18 gap **-2.07** 是極端值，formula 可能高估或低估真實水平
  2. Meyer xERA 4.46 vs ERA 3.30 gap +1.16 → ERA 有運氣加持，可能朝 4.0 區間回歸
  3. **兩隊 closer 雙缺**（Fairbanks + Duran 同 15-day IL）— 末段失分機會放大，對 Total 偏 +
  4. **Sharp 信號 Over +3.9pp 但 base Total 6.7 — divergent**。Sharp 窗口僅 2.5h（弱信號），juice 動但線沒動（8.5 沒抬），訊號強度有限

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
