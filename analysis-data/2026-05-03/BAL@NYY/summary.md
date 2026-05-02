## 投手對決

### Max Fried (HOME, LHP, 32 📉 初期退化)
- **Tier 覆寫**：沿用腳本 🟠 Strong Ace。ERA 2.09 / xERA 2.24 一致（差 0.15，遠低於 1.5 trigger 閾值），無運氣偏差。
- 真實水平判斷：本季 47.1 IP / 7 GS，K-BB% 14.1（聯盟平均偏上但非頂尖），但 contact suppression 一流：whiff% 9.9、hard_hit% 20.5、barrel% 2.4 都是 elite contact 抑制水準。FIP 2.57 vs xFIP 3.75 落差 1.18，HR/9 0.19 不可持續，未來會回升 — 但 Yankee Stadium 短右外野對 LHP 反而有風險。球速 87.8 mph 偏低，靠 6 球種（SI 21.7 / FC 20.8 / FF 18.5 / CU 15.3 / CH 13.1）混搭吃結果。32 歲在 📉 初期退化窗，但本季 Statcast 維持頂尖，不額外退化扣分。近 3 場 ER/IP = 3/20.0（ERA 1.35）熱手中。
- 對手打線威脅：BAL vs LHP 整體比 vs RHP 好（Henderson .869 / Ward 1.035 / Jackson .901），但 Fried 對 LHB 只放 .156/.188 = .388 OPS，所以 BAL 左打沒優勢；威脅集中在 RHB 主力（Henderson、Ward、Basallo）。Alonso vs LHP 全季只 .367 OPS（左投殺手對象），但 BvP 48 PA 對 Fried .238/.333/.405 = .738 OPS（樣本足，採用）— Alonso 是 Fried 唯一較難壓的右打點。整體威脅中等偏低。

### Trey Gibson (AWAY, RHP, 23 📈 成長期)
- **Tier 覆寫**：⚪ Unknown → 暫定 🟢 Back-end（首戰調整檔位）。**🚨 STARTER_NOT_ACTIVE 觸發**：MLB Stats API 列為今日 BAL 先發，但不在 active 26 人也不在 IL，2026 季完全無 MLB 投球紀錄 — 實質為 MLB 首登（可能 5/3 賽前才從 AAA 召上）。
- 真實水平判斷：**完全無 MLB sample，所有評估都建立在「prospect 預期」之上，不確定性極大**。Tier 為慣例 placeholder，並非數據基礎。23 歲 📈 成長期窗代表上限可期、下限風險也大。首戰客場、面對 NYY hot 打線、Yankee Stadium 短右外野 — 三項環境因子都不利首登調整。
- 對手打線威脅：NYY top3（Judge 1.019 / Bellinger .755 / Rice 1.169 OPS）對 RHP 全部優異（Judge .948 / Bellinger .773 / Rice 1.112），近 7 天 Judge 1.342、Rice 1.165 都熱火中；chain OBP top3 = .397 是聯盟頂級的「跑者堆積」能力。對首登投手是噩夢級對手。**威脅 🔴 極高**。

## 打線評級

### HOME — 🟠 Strong / 🔥 Hot
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Tier 覆寫**：沿用腳本 🟠 Strong / 🔥 Hot。xwOBA .349、chain OBP top3 .397 都是聯盟頂端區間；近 7 天 Judge .516 OBP、Rice 1.165 OPS、Caballero 1.050 OPS（後者 BABIP .263 不極端，可信）多人同步熱手。中段 chain SLG mid .325 是唯一弱點 — 4-5 棒 Chisholm（.618）/ Grisham（.617）拖低長打串聯。Volpe 與 Stanton 在 IL 但 Caballero / Rosario 補上後表現甚至更好（替補品質反向檢查通過）。對 Gibson 首登威脅 🔴 極高。

### AWAY — 🟡 Average / ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Tier 覆寫**：沿用腳本 🟡 Average / ⚖️ Normal。chain SLG mid .455 是中段火力亮點（Jackson / Basallo）；但 chain OBP top3 .340 略低於聯盟，1-3 棒（Henderson / Ward / Alonso）上壘能力中等。RHB 主力 vs LHP 平台優勢明顯，但本場對 Fried 此優勢被 Fried 對 LHB / RHB 雙向壓制（.388 / .466 OPS）抵消。Holliday、Westburg、Mountcastle、Kjerstad 4 名打線主力 IL — 替補頂上後實質戰力下滑（特別是 Mayo .555、Beavers .622 OPS 是明顯弱點）。對 Strong Ace LHP 威脅有限。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.69 / 4 / 0-1 名（Chivilli 中段，其餘 3 名是 SP） | 3.97 / 7 / **2 名（Bautista + Helsley，🔴 高）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 3.69 健康水準。4 名 IL 中 Cole / Rodón / Schmidt 都是先發輪值傷兵，**對牛棚實際影響極小**；Chivilli 為中段 reliever（非核心）。後段核心 Bednar / Doval / Cruz / Hill 全員可用，深度充足。對 BAL 末段（替補多、Mayo / Beavers / Alexander 弱點集中在中後棒）威脅明顯。
- AWAY 牛棚：ERA 3.97 中性，但 IL 名單看深度問題嚴重 — **Bautista（前 All-Star Closer，60-day）+ Helsley（前 STL Closer，15-day）兩個明確核心同時缺陣，達 §牛棚傷兵累計效應「2 名核心 → 🔴 高影響」分級**。剩下 Kittredge / Cano 兩名前段穩定，但 high-leverage 第三人深度被吃掉。對 NYY hot 打線末段威脅顯著 — 7-8 局若戰況咬住，BAL 後段反而是 NYY 加分窗口。

## 風險提示

雖然腳本未自動標 ⚠️（BABIP / ERA-xERA gap 都未觸發），但 AI 補列以下兩項結構性風險：

1. **🚨 STARTER_NOT_ACTIVE — Trey Gibson 不在 BAL active roster 也不在 IL**
   - 觸發條件：`roster_checker.py` 標出此 trigger；`pitcher_stats.py` 回傳「No 2026 pitching stats found」
   - 解讀：Gibson 為 23 歲新人，2026 季 0 IP MLB sample，實質為 MLB 首登（推測 5/3 賽前才從 AAA 召上）
   - 對分析的影響：投手對決、base formula 預測得分、tier 判讀都是猜測 — 不確定性遠高於一般場次
   - skill 規範要求暫停回報，此處顯式列出供使用者決策；若 BAL 公布實際先發換人（active roster 內任一人），請重跑 `prepare_game.py --force`

2. **BAL 牛棚核心 2 名 IL（Bautista + Helsley）**
   - 觸發條件：`matchup-factors.md` §牛棚傷兵累計效應「2 名核心 → 🔴 高影響」
   - 影響：BAL 7-9 局對 NYY hot 打線是明顯破口，本場 NYY 末段加分機率上升

## 條件修正

- Park Factor: 96.0 → -0.20 run（投手友善，但 HR +12% 對 LHB 利多 — Bellinger / Rice 是 LHB 主要受益者）
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：非 doubleheader。先發 tier 雙方落差極大（Fried 🟠 Strong Ace vs Gibson ⚪ Unknown / 首登），對投手對決判讀是核心變數，但因 Gibson 無 sample，不入 base formula 修正。

## 修正後預期得分

> 「+ 信號」欄僅納入規範允許的條件修正：Park Factor、牛棚累計效應（核心 IL ≥ 2 名）、主力打者傷兵。
> ⛔ BABIP 極端值 / ERA-xERA gap **不入此欄**（規範禁止 auto ±run value，見 reference/flags-checklist.md §3, §8）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.1 | +0.4（BAL 牛棚核心 2 IL 對末段威脅） | 5.5 |
| AWAY | 2.7 | 0（NYY IL 全為 SP，牛棚核心完整） | 2.7 |
| Total | 7.8 | +0.4 − 0.2（Park） | **8.0** |

> ⚠️ Gibson 首登造成的不確定性 **不入此欄**（規範禁止對缺 sample 自動 ±run value）；實際 NYY 得分有 upside（若 Gibson 崩盤可達 7-8 runs），也有 downside（若 prospect 表現出色可能壓在 3-4 runs）。adjusted 8.0 是基本面中位估計。

## 整體判斷

- **方向（基本面）**：明顯偏向 **NYY**。主場 + 🟠 Strong Ace（Fried 熱手）+ 🟠 Strong / 🔥 Hot 打線 + 對手先發為 MLB 首登 + 對手牛棚核心 2 名 IL — 五項基本面因子全部一邊倒。系列賽 G1 NYY 已 7-2 取勝，動能延續。
- **總分（基本面）**：**~8.0 runs（中位估計）**。NYY 5.5 / BAL 2.7。Gibson 因子使區間極寬：上行（崩盤情境）9-10 runs、下行（prospect 表現出色 + Fried 維持）5-6 runs。Yankee Stadium PF 96 略壓總分但 HR 因子（+12%）對 NYY LHB（Bellinger / Rice）有利，HR 一發定江山。
- **信心**：**MEDIUM-LOW**。方向（NYY 勝）信心 MEDIUM-HIGH，但總分與分差信心 LOW — 主因是 Gibson 完全無 MLB sample，無法可靠估算被打程度。
- **風險**：
  1. **Gibson 首登變數最大**（單場 prospect 首戰標準差遠高於有 sample 投手；可能驚喜也可能崩盤 — STARTER_NOT_ACTIVE 已列風險段第 1 項）
  2. **BAL 牛棚核心 2 名 IL**（Bautista + Helsley），末段防守破口 — NYY 7-9 局加分機率上升
  3. **Fried HR/9 0.19 不可持續**（FIP 2.57 vs xFIP 3.75 落差 1.18），Yankee Stadium HR +12% 是潛在反向變數，BAL Henderson / Ward / Basallo 任一發 HR 可改變局勢
  4. **NYY last7 BABIP 0.266 略低**（未到 ⚠️ 0.260 閾值，但接近回歸警戒）；單場 BABIP 噪音大，不直接修正

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組