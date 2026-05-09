## 投手對決

### Michael King (HOME, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p82, K-BB% p69），gap vs ERA-only = -1.4
  - 同意 tier_v2。ERA 2.95 vs xERA 4.00 gap 1.05（Flag 8 量級），FIP 3.48 / xFIP 3.61 認可 Strong Ace 結構，但 ERA 是被 BABIP 壓低後的數字 — 真實期望介於 ERA 與 xFIP 之間，formula 用 ERA 略低估其失分。
- **Reverse platoon 信號**：fired，Δ +0.344（vs RHB OPS .833 > vs LHB OPS .489）
  - **嚴重放大**：STL 預期打序 1-5 中 Herrera / Wetherholt / Burleson / Walker / Gorman 至少 4 人為右打或左打但 vs RHP 仍強（Burleson .921, Walker .961 vs RHP），King 主球種 SI（27.6%, RV/100 -0.5）對右打殺傷力下降 — STL 核心打者吃滿 reverse platoon 紅利。
- **對手打線威脅**：偏高。STL 打線 vs RHP 為 🟠 Strong，top 5 last7 OPS 高達 1.159 (Walker) / .891 (Burleson) — King 的 ERA 帳面雖好，但 xERA 與 reverse platoon 雙紅旗指向今晚 STL 對 King 不會像對手平均（vs LHB .247 SLG）那麼乖。

### Matthew Liberatore (AWAY, LHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟢 Back-end Starter（xFIP p20, K-BB% p24），gap vs ERA-only = -8.7
  - 同意 tier_v2 + 結構性偏弱。ERA 4.50 反而**比** xERA 5.80 / FIP 5.79 好，意味目前 ERA 是 BABIP / 殘壘運氣護住的（Flag 8 反向），formula 用 ERA 甚至**高估** Liberatore — SD 真實預期得分可能再上修。
- **Reverse platoon 信號**：fired，Δ +0.315（vs LHB OPS 1.071 > vs RHB OPS .756），LHB BF 樣本只 44（small）
  - SD 預期打序左/雙打有 Cronenworth / Merrill 等，但核心 RHB Tatis / Machado / Bogaerts / Laureano 占多數。RHB 樣本 BF 114（穩），slash .270/.336/.420 — Liberatore 對 RHB 已不算強。Machado vs LHP OPS 1.098 是**最大爆點**。
- **對手打線威脅**：中高。SD 打線整體 vs LHP 評為 🟢 Weak（last7 BABIP .212 拖累），但 Machado / Bogaerts / Tatis 三位核心仍可單點殺穿。

## 打線評級

### HOME (SD) — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected
- **Matchup tier (vs LHP)**：🟢 Weak
  - 與 season Average 對比下修一檔。但這是「整隊平均」結論 — 真正威脅集中在 Machado（vs LHP OPS 1.098）與 Bogaerts（.753）兩位深層威脅，預期得分非「均勻 weak」而是「點狀爆破」。
- **chain_break #7-8（OPS Δ 0.305）**：壓制末段串聯，攻擊集中在 1-6 棒，第 7 棒以後 Liberatore 可較輕鬆續航 — 對 SD 不利信號，但同時也意味 Liberatore TTO3 災難（OPS 1.235）若到第三輪面對前段，會被點爆。

### AWAY (STL) — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected
- **Matchup tier (vs RHP)**：🟠 Strong
  - 比 season Average 上修一檔。top 5 vs RHP slash 平均 .800+ OPS，Walker last7 OPS 1.159 / Burleson .891 都在熱期（且 BABIP 不極端，非單純運氣 — Walker EV95% 57.8 / Barrel% 20.0 為結構性 hot）。
- **chain_break #4-5（OPS Δ 0.278）**：Walker / Gorman 之間有落差（.961 → .730 vs RHP），但 Walker 自己就是清壘級威脅，影響 < 一般 chain_break。

## 牛棚

| | HOME (SD) | AWAY (STL) |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.75 / 5 / 0 | 4.87 / 1 / 0 |

### 牛棚影響判讀
- **HOME (SD)**：3.75 ERA 屬中上水準，core IL 0 名 → 可用性完整，後段（7-9 局）可派完整 high-leverage 鏈。SD 牛棚是本場明顯優勢。
- **AWAY (STL)**：4.87 ERA 偏弱，core IL 0 名但整體深度不如 SD。Liberatore TTO3 1.235 → STL 教練極可能 5-6 局換投，**牛棚負擔被迫拉長**，4.87 ERA 牛棚要扛 3-4 局 → 失分風險顯著。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.212):
  - 屬「unlucky-cold」端：xwOBA .316 / OPS .676 不算極弱，BABIP .212 比預期低 ~50pt → 部分回歸合理（last7 樣本 ~10 場，雜訊空間大）。**敘事不自動上修預期分**，但若今晚 BABIP 回歸到 .280 區間，SD 打線輸出可能比 formula base 多 0.3-0.5 分。

### 額外信號
- 🔴 HOME reverse_platoon (King)：見上文，**強放大**對 STL 右打核心威脅
- 🔴 HOME tto3_penalty (King)：Δ +0.685（high），STL 第三輪 OPS 1.086 → King 多半 5 局退場，後段牛棚得加班
- 🔴 AWAY reverse_platoon (Liberatore)：對 RHB 樣本穩 (.756 OPS)，意味 SD 右打陣對 Liberatore 不會被壓制
- 🔴 AWAY tto3_penalty (Liberatore)：Δ +0.539（high），SD 第三輪 OPS 1.235 → Liberatore 5 局以內，STL 牛棚要扛 4 局以上（核心風險）
- 🔴 HOME chain_break #7-8 → 末段串聯弱，SD 攻擊偏 1-6 棒
- 🟠 AWAY chain_break #4-5 → 影響有限（Walker 自帶清壘）

## 條件修正

- Park Factor: 95.0 → -0.25 run / 側（已含於 base，Petco 微利投但 HR +7%）
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：King > Liberatore 一個 tier 半（🟠 Strong Ace vs 🟢 Back-end），且 Liberatore xERA / FIP 都已超過 5.50 — Liberatore 結構性偏弱是本場最大一致信號

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (SD) | 5.9 | +0.3 | 6.2 |
| AWAY (STL) | 3.7 | +0.5 | 4.2 |
| Total | 9.6 | +0.8 | 10.4 |

**+信號 推導**：
- HOME +0.3：tto3_penalty (Liberatore high) +0.4 + reverse_platoon (LHB 樣本小) +0.1 + chain_break #7-8 -0.2 = +0.3
- AWAY +0.5：tto3_penalty (King high) +0.4 + reverse_platoon (King vs RHB 嚴重) +0.4，interaction 取 max+0.1 = +0.5；chain_break #4-5 影響有限不扣
- Total +0.8 已逼近 cap 1.6（合計兩側）；考量兩側 tto3 雙 fire + Liberatore Flag 8 反向（formula 高估 Liberatore），real over-bias 仍存在但敘事處理

## 整體判斷

- **方向（基本面）**：SD（HOME）勝面
- **總分（基本面）**：10-11 分區間（Over 偏向）
- **方向信心**：62%（SD 領先 tier 半級且牛棚明顯較好；下修因素：Liberatore 對 RHB 不算強 + STL 打線 vs RHP 是 Strong tier + 兩位先發 reverse platoon 都 fire 拉近差距）
- **風險**：
  1. 兩位先發 TTO3 雙崩 → 比賽很可能進入 5-6 局後就是牛棚對轟，雜訊放大
  2. King ERA 2.95 vs xERA 4.00 → 若今晚回歸，SD 領先幅度收窄
  3. Liberatore vs RHB 樣本（.756 OPS, 114 BF）穩定，SD 右打深度（Tatis / Machado / Bogaerts / Laureano）足以單點打爆
  4. SD 打線 last7 BABIP .212 若繼續低迷，得分可能仍卡在 base 5.9 以下，反而不過 Total

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
