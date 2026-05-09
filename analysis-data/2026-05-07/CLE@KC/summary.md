## 投手對決

### Seth Lugo (HOME, RHP, 36 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p82, K-BB% p72），gap vs ERA-only = -9.7
  - |gap| < 15 → 認可 score-derived tier（ERA 2.68 與 xERA 4.08 / xFIP 3.61 的差距屬輕度運氣紅利範圍，可接受 Strong Ace 判定，但本季 ERA 不該被視為「真實天花板」）。年齡 36 與球速均速 85.8 mph 反映明顯退化，敘事採「品質仍在但 margin 變薄」基調，不自動下修。
- **Reverse platoon 信號**：未 fired（vs LHB OPS .663 / vs RHB OPS .670，落差 < 0.080，且 SLG 反向程度有限）
- **對手打線威脅**：CLE 打線 vs RHP 為 🟡 Average matchup tier，xwOBA .319 略優於 KC，但 last7 BABIP .249（Flag 3）顯示近期擊球落點走背；DeLauter 是唯一 last7 OPS > 1（1.369 / BABIP .591）的爆發點，其餘前 5 棒 last7 OPS 均 < 0.65。Lugo SI/CU/FC mix（balanced 4+ 球種）對 CLE 中後段陳舊打順具壓制力，難讓 José Ramírez（last7 OPS .521）回到 vs RHP .617 的水準以上 — 對 CLE 是壓制型對位。

### Slade Cecconi (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p58, K-BB% p44），gap vs ERA-only = +47.1
  - |gap| ≥ 20（high）→ ERA 6.56 顯著低估真實水平。xFIP 4.01 / K-BB% 9.2 / barrel% 11.7 顯示真實天花板約落在中後段先發；近 3 場 ER 10/IP 15.7 = 5.73 ERA，仍偏結果導向不利。AI 判讀偏「BABIP/HR-FB 帶來的運氣偏差」（hard_hit% 25.2 並不極端、velo 87.2 average 在巔峰年齡可接受），formula 仍以 base 6.3 為 KC 預期得分上限的 sanity rail，**不主動 override 下修**，但本場結果區間其實比 ERA 6.56 暗示的窄。
- **Reverse platoon 信號**：fired（Δ +0.224，vs RHB OPS .978 > vs LHB .754）— RHP 對右打反而吃虧。KC 1-9 棒中右打主力為 Maikel Garcia / Bobby Witt Jr. / Salvador Perez / Isaac Collins（switch，多場以右側面對 RHP）四人，含 1-2-4-7 棒；其中 Witt（season .771 / vs RHP .701 / EV95% 51.7%）與 Collins（.731 / vs RHP .838 / last7 OPS 1.016）此 reverse 紅利會明顯放大。
- **對手打線威脅**：KC 打線 vs RHP 為 🟢 Weak matchup tier，但本場是「弱打線打弱投」的合流。Cecconi FF 35.5% / FC 24.4% 是 platoon-vulnerable 高三振低 GB% 的型態，對 KC 中段熱手 Caglianone（last7 OPS .912 / Barrel 15.9）與 Collins（last7 OPS 1.016）幾乎沒有抑制力；TTO3 1.459 的崩盤型曲線疊加 KC 牛棚壓力 → KC 預期能在第 5-6 局把 Cecconi 趕下場。

## 打線評級

### HOME (KC) — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak
  - 落差 = season Average → matchup Weak。基本盤 KC vs RHP 數據不亮眼（多數人對 RHP OPS 低於 .720），但本場面對 ERA 6.56（即使真實水平 Solid Starter）的 RHP，又疊 reverse_platoon 紅利 → 本場應上修一檔，採「Average +」處理。
- **chain_break / heat_vs_babip 信號**：未 fired（KC 1-9 OPS 連續性無 ≥ 0.150 落差；last7 BABIP .281 中性）

### AWAY (CLE) — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - 落差 = 0。但 Average tier 面對 Strong Ace（Lugo）必然下修一檔，本場以「Average -」處理；Ramírez（last7 OPS .521）/ Kwan（last7 OPS .572）/ Rocchio（last7 OPS .499）整條前段同步降溫是更大壓力。
- **chain_break / heat_vs_babip 信號**：chain_break #8-9 fired（OPS 落差 0.341，high）— DeLauter 之後一路掉到 #6 之後實質銜接斷裂，CLE 從第 6 棒以下無有效 chain；但因 lineup 為 projected，#8-9 是 PA 近似排序，實際落點可能稍緩；此 signal 仍對 CLE 總分有壓制效果。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.8 / 4 / 1 (Estévez = Closer) | 4.03 / 2 / 1 (Armstrong = HL RP) |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（KC）：ERA 4.80 本就略劣於聯盟平均，Closer Estévez IL15d 缺陣使第 9 局封鎖力下降 — 1 名 core IL = 🟠 中高，如果 Lugo 能投到 7 局以上影響可被吸收，但 Lugo 36 歲球種 mix 多 / 球速低意味 PC 上限偏低，可能 6 局後就需要進入薄弱牛棚。CLE 第 7-9 局得分機會邊際上升。
- AWAY 牛棚（CLE）：ERA 4.03 略優，Armstrong（HL RP）IL15d → 1 名 core IL = 🟠 中高。但 Cecconi TTO3 penalty Δ +0.780 的崩盤型曲線意味 Cecconi 大概率 5 局內就要交給牛棚，加長牛棚負擔；KC 中段熱手 Caglianone / Collins 在後段仍能維持壓力，CLE 牛棚消耗 risk 比 KC 牛棚明顯。

## 風險提示

- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.249):
  - CLE 整條前段 Ramírez / Kwan / Rocchio 三人同步進入 BABIP < .210 區間，屬「集中性低迷」而非單人運氣偏差；不自動 ±run，但敘事傾向「短期可能反彈但本場 vs Strong Ace Lugo 不是反彈場景」 — 反彈期需要遇 mistake-pitch 多的投手，Lugo balanced 4 球種、season 表現結構性穩，難成為 BABIP 觸底點。本場 CLE 得分 floor 偏低判讀不變。

### 額外信號
- ℹ️ HOME balanced 4+ pitches：Lugo 最高球種僅 21.0% — 對 platoon-advantaged 對手難對位 = 對 CLE 打線壓制（pressure on 對手）
- 🔴 AWAY reverse platoon Δ +0.224：見上 §投手對決 Cecconi 段；對 KC 右打主力（Witt / Collins / Garcia / Perez）放大效益
- 🔴 AWAY TTO3 penalty OPS Δ +0.780（K% drop -8.4pp）：第三輪近乎全失能；KC 預期在第 5-6 局把 Cecconi 趕下場
- 🔴 AWAY chain breaks at #8-9（OPS Δ 0.341）：CLE 後段串聯實質斷裂；但因 projected lineup 排序近似，影響強度略低於 official 場景
- 🟠 ⏳ HOME 牛棚 core IL ×1（Estévez）：Closer 缺陣，CLE 第 7-9 局得分機會邊際上升；⏳ short half-life signal，需留意 Estévez 是否本日激活
- 🟠 ⏳ AWAY 牛棚 core IL ×1（Armstrong）：CLE 牛棚負擔在 Cecconi 早退場景下被放大；⏳ 同上
  - 雙隊核心 IL 各 1 → 對總分判讀偏「+0.1 ~ +0.2 / 隊」邊際上修，與 TTO3 + reverse_platoon 同側對 KC 累積 → 取單側 max + 0.1 不直接相加（見下表）

## 條件修正

- Park Factor: 106.0 → +0.30 run（Kauffman Runs 106 / HR 91 — 利安打三壘打、壓 HR）
- 天氣：Sunny, 65°F, wind 15 mph, Out To CF
  - 影響判讀：65°F 中性溫度（60-85°F 範圍）；風 15 mph Out To CF 屬「中度順風」門檻邊緣，對中外野方向飛球加成。Kauffman HR PF 91 本來壓制 HR，但中度順風會抵銷部分壓制 → HR 機率回到接近中性；對總得分影響 = 輕度利攻 +0.1 ~ +0.2，但本來 PF 已含 → 不重複加值，敘事提示為主。
- 先發 tier / doubleheader：非 doubleheader；Lugo 🟠 Strong Ace vs Cecconi 🟡 Solid Starter（tier_v2）= 一檔 tier 落差，KC 投手側顯著佔優；formula base 已反映此差距。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 6.3 | +0.6 | 6.9 |
| AWAY | 3.0 | -0.2 | 2.8 |
| Total | 9.3 | +0.4 | 9.7 |

**+ 信號計算**：
- HOME 同側上修（reverse_platoon Cecconi → KC 右打 +0.3 / tto3_penalty → KC 中段 +0.4 / 同側互動 max + 0.1 = +0.5；CLE 牛棚薄 +0.1）→ +0.6（單側 cap 0.8 內）
- AWAY 同側淨值（chain_break -0.2 / HOME balanced 4+ pitches 壓 CLE -0.1 / KC 牛棚薄 +0.1）→ -0.2

## 整體判斷

- **方向（基本面）**：HOME (KC) 偏向勝
- **總分（基本面）**：9.7（adjusted total）
- **方向信心**：60-65%
  - KC 投手側 tier 落差顯著佔優，且 Cecconi 同時觸發 reverse_platoon + TTO3 + 近 3 場 5.73 ERA 三重壓力；但 Cecconi tier_v2 真實水平為 Solid Starter（非 Below Average），且 CLE 打線 last7 BABIP 偏低有正向回歸概率，避免將信心拉到 70%+
- **風險**：
  1. Lugo 36 歲 / 球速 85.8 mph，球種 mix 多本是壓力分散的優勢，但同時意味 PC 上限低 → 6 局後可能進入 Estévez-缺陣的薄弱牛棚，CLE 後段反撲視窗存在
  2. AWAY last7 BABIP .249 集中性低迷，存在均值回歸壓力；DeLauter 一人扛 last7 OPS 1.369 是潛在引爆點
  3. Cecconi 雖 tier_v2 升級，但 ERA 6.56 / 近 3 場 5.73 反映實際結果導向仍極差 — 一場「真實水平」場景出現的機率仍存在，KC 得分上限可能不到 base 6.3
  4. 風 15 mph Out To CF 為中度順風邊緣，可能讓 Kauffman HR-壓制效應減弱，總分略微 +0.1 ~ +0.2 的不確定性

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
