## 投手對決

### Bubba Chandler (HOME, RHP, 23 歲)
- **Tier 覆寫**：沿用 🟢 Back-end Starter（ERA 4.88 / xERA 5.27 同樣偏差，無運氣偏離）
- 真實水平判斷：球速本錢出色（avg 95.1 / max 101.3），但 **K-BB% 4.6% 極低**（聯盟均值 ~14%），WHIP 1.50、whiff% 9.9%。FF 使用率 56.3% 過度依賴速球。對左打 .262/.367/.500（49 BF）顯著吃虧，但對右打 .188/.322/.292 穩定。23 歲成長期，控球與第二球種仍是建構中。
- 對手打線威脅：STL 對 RHP 中性 — Iván Herrera（.856 vs RHP, last7 OPS 1.136）熱手、JJ Wetherholt（.830, last7 OPS .999）進入手感區、Jordan Walker（.827）power 來源。Chandler 對 LHB 弱，Burleson、Gorman、Wetherholt（L）都有放大空間。

### Andre Pallante (AWAY, RHP, 27 歲)
- **Tier 覆寫**：沿用 🟢 Back-end Starter，但實際偏弱 — ERA 4.26 比 xERA 5.17 / FIP 4.72 都好，**運氣加持明顯**，回歸風險高。
- 真實水平判斷：velo 89.0 偏低、K-BB% 5.2% 同樣極低、Hard Hit 26.5% / Barrel 10.3% 被打偏硬。球種較分散（FF 29.7 / SL 26.8 / SI 20.5）但無壓制性球種。近 3 場 ER/IP 8/15.0（ERA 4.80）有惡化。
- 對手打線威脅：PIT 對 RHP 同樣中性，但 **Ryan O'Hearn vs RHP .965、Brandon Lowe vs RHP 1.077** 是嚴重個別威脅；Oneil Cruz 雖 last7 .640 偏冷，但 EV95 62.5 / Barrel 23.6 power 隨時爆發。

## 打線評級

### HOME (PIT) — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用。xwOBA .335 / OPS .731；近 10 戰 RS 4.40 略低，但對 RHP 有 O'Hearn / Lowe 兩隻 vs RHP OPS > .96 的明確火力點，vs Pallante 期望值高。

### AWAY (STL) — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用。xwOBA .329 / OPS .737；近 10 戰 RS 5.30 表現更熱，Herrera + Wetherholt last7 OPS 雙破 1.000。對 Chandler 速球壓力可消化（多名打者 EV95 > 44）。

## 牛棚

| | HOME (PIT) | AWAY (STL) |
|---|---|---|
| ERA / IL 數 / 核心 IL 估計 | 3.49 / 1 / 0（Jared Jones IL60d 屬先發名單，**不計入**牛棚核心） | 5.18 / 2 / 0~1（Hunter Dobbins / Matt Pushard 皆 IL15d，多為邊緣角色） |

### 牛棚雙向修正值
- HOME 牛棚（3.49）顯著優於聯盟均值（~4.0）：對 STL 得分 −0.2 run | PIT ML +1~2%
- AWAY 牛棚（5.18）顯著弱於聯盟均值：對 PIT 得分 +0.5 run | STL ML −2~3%
- **雙向淨值**：總分 +0.3~0.7 run（已由 predict.py 信號表計入 +0.5），ML 偏向 PIT 約 +3~5%

## 風險提示

無 prepare_game.py 自動標記的風險（無 BABIP 異常、無 Doubleheader、無 Coors 4 月、無重大傷病警訊）。

額外 AI 觀察：
- 兩位先發 K-BB% 同樣 ~5%（極低）→ 打者擊球機會多，BIP 噪音放大但偏向 OVER 走向
- Pallante ERA−xERA 顯著負偏離 → 回歸風險偏向 STL 失分上修
- PNC Park HR −17% → 壓制 STL（Walker、Burleson）長打型威脅，小幅抵銷得分上修

## 條件修正

- Park Factor: 102.0 → +0.10 run（已由腳本計入）
- 先發 tier：兩 🟢 Back-end，**未觸發 -0.5/-1.0 雙投手降分**（無）
- Doubleheader / 天氣 / 多/少休息日：無顯著條件修正

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (PIT) | 5.5 | +0.5（STL 弱牛棚）+0.05（PF) | 6.05 |
| AWAY (STL) | 6.3 | +0.05（PF) | 6.35 |
| Total | 11.8 | +0.6 | **12.40** |

## 整體判斷

- **方向（基本面）**：formula log5 = HOME 50.6%（接近 50/50）。PIT 主場 + 牛棚優勢 + PNC HR 壓制略偏向 PIT；但 STL 打線近 10 場 RS 5.30 較熱、Pallante 雖屬 Back-end 但 STL 整體得分能力略高。**極為勢均力敵**，方向信心 LOW。
- **總分（基本面）**：base 11.8 + 信號 +0.6 = **12.4**，相對 O/U 8.5 line **差距 +3.9 run**，遠超 SD 噪音範圍。雙方先發 K-BB% 極低、STL 弱牛棚、PNC PF 偏打者，OVER 信號明確。**信心 HIGH**。
- **信心**：方向 LOW / 總分 HIGH
- **風險**：
  1. PNC Park HR -17% 抑制長打，需靠安打串聯堆疊得分（高分但不依賴 HR）
  2. Pallante ERA−xERA 已負偏離，但若反向回歸（運氣持續），STL 失分可能不如預期高
  3. 兩位先發投球局數有限（GS 5），會更早進牛棚 → 放大 STL 弱牛棚效應，亦支持 OVER
  4. 早季樣本（28-30 場）整體不確定性偏高，星級護欄需保守

⛔ MUST NOT contain：星級、明確盤口推薦
