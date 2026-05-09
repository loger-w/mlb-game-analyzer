## 投手對決

### Michael Wacha (HOME, RHP, 34 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p45, K-BB% p62），gap vs ERA-only = -21.0
  - 同意 tier_v2。ERA 3.05 vs xERA 4.30（Δ-1.25）+ K-BB% 12.1%（聯盟均值附近）+ whiff 9.9%（偏低）顯示其 ERA 表面好看但 underlying 偏中段。屬「結構性 Solid」非運氣偏低；不再下修，但不要把它當 ace 看。
- **Reverse platoon 信號**：fired，Δ +0.139（vs RHB OPS .723 > vs LHB OPS .584；雙側 BF 64/110 充足）
  - 放大風險。DET 核心打線 LHB 偏多（McGonigle、Greene、Carpenter 三人都吃 RHP 卻會被 Wacha 反咬），主要 RHB 威脅集中在 Torkelson / Dingler 兩人；Wacha 結構性對 DET LHB 群有利，DET 攻擊壓力主要落在 R-bat 兩人 + 連帶吃 chain_break 風險。
- **對手打線威脅**：中下。DET 整體 vs RHP 雖屬 🟠 Strong（last7 OPS 0.720 / xwOBA 0.341），但 reverse_platoon 直接削去其 LHB 那半邊輸出；Wacha 球種以 FF/CH/FC 三球為核心，其中 CH 對 LHB 殺傷力大（whiff 31%, xwOBA .238），更壓 DET 的 L-bat 上限。

### Burch Smith (AWAY, RHP, 36 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 = —（GS 0、IP 11.3，沒投過先發足夠局數計分），gap vs ERA-only = —
  - **不同意 tier_v2 留白即等於 Elite**。Smith 屬牛棚出身的臨時先發（Skubal / Verlander / Mize 都在 IL，Detroit 在拼補 12 號投手），ERA 1.59 / FIP 1.34 是 11.3 IP + 全部站短局牛棚的 sample 噪音（Flag 8：IP < 30 → 不自動下修預測，但 narrative 必須回歸）。年齡 36、近 3 場 ER/IP 1/5.3 仍是短局形式。實質期望值更接近 🟢 Back-end / Spot Starter（xFIP 2.14 也是 11 IP 的小樣本，不可當基準）。預期局數 4-5 IP 後就要交棒，DET 牛棚 3 名核心 IL 會被推到前線。
- **Reverse platoon 信號**：未獨立 fire，但雙側手別樣本嚴重不對稱（vs LHB 21 BF .389/.450/.556；vs RHB 25 BF .087/.160/.174）— **任一側都不到穩定樣本**，AI 不引用此分裂作預測。提醒：KC 中段 Pasquantino / Jensen 屬 LHB，若 Smith 對 LHB 真有問題，這裡會被點到。
- **對手打線威脅**：中等偏上。KC 整體 vs RHP 🟡 Average / .709 OPS，但 Witt（last7 OPS 1.008、BABIP .400 → Flag 3 lucky-hot）+ Pasquantino / Jensen 兩個 LHB 對位 + 主場優勢綜合，Smith 第二輪一過很可能就被點。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - matchup tier = season tier，方向同意 🟡 Average。但 vs Smith（小樣本 RHP）的 top 5 中 **Pasquantino vs RHP .737 / Jensen vs RHP .772 / Dingler vs RHP .838** 都比各自 season OPS 高（≥ 0.050 上修），有 platoon 偏多 signal 跡象（未官方 fire 因 top5 只有 3 人達標）。本場小幅上修。
- **chain_break / heat_vs_babip 信號**：fired
  - chain_break #2-3（落差 0.180）：Witt → Garcia → Perez。Witt 引擎熱（last7 1.008 / BABIP .400 lucky-hot，Flag 3 narrative），但 Garcia .611 + Perez .327 兩人冷組合形成上下接斷層 → KC 想連線就要靠 Witt 自己上壘 + 4-5 棒 (Pasquantino .646 / Jensen .259) 收尾，依賴度高、上限被限制。
  - Perez last7 BABIP .176（Flag 3 unlucky-cold）+ Jensen last7 .167 → 兩人有反彈空間但本場不一定兌現。

### AWAY — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - matchup tier 比 season tier 下修一檔（🟠 → 🟡）。原因：Wacha reverse_platoon 直接吃掉 DET LHB 三人；同意下修判斷。McGonigle vs RHP .915 / Greene vs RHP .829 / Carpenter vs RHP .773 雖然數據漂亮，但 Wacha 是反 platoon 樣本（vs LHB OPS .584），這三位實際對位產出會比 vs RHP 平均收斂。
- **chain_break / heat_vs_babip 信號**：fired
  - chain_break #7-8（落差 0.160）：尾段斷鏈 → 第二輪過後 1-5 棒攻完，6-9 棒接不上。
  - heat 個別風險：McGonigle last7 OPS .396 / BABIP .136（unlucky-cold，Flag 3 narrative，可能反彈）；Torkelson last7 .542 / BABIP .167（unlucky-cold）；Carpenter .200 BABIP；Greene last7 1.002 / BABIP .688（lucky-hot，Flag 3，本場有回收風險）。整體 last7 數字噪音大，Wacha matchup 又限制 LHB 端，AI 對 DET 的「強打線」預期持保留。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.77 / 5 / 1（Estévez closer 缺陣） | 3.88 / 10 / 3（Vest closer + 至少 Brieske/Melton 兩名核心 RP） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（KC）：ERA 4.77 季帳偏差，但 Estévez 一名 closer 缺陣屬 🟠 中高（單一核心缺陣），仍有可用結構末段。可用性可承受 4-5 IP 接班。對 DET 末段壓力為「壓上限不致失控」。
- AWAY 牛棚（DET）：ERA 3.88 表面好看，但 Vest（IL15d）+ Brieske（IL60d）+ Melton（IL60d）三名核心同步缺陣 = 🔴🔴 極高 / 崩盤級。剩 Jansen / Finnegan / Holton 三人撐 high-leverage，後段一旦被點開就沒得換；Burch Smith 預期 4-5 IP 後立刻吃進這個薄牛棚。對 KC 末段（6-9 局）形成顯著得分機會。

## 風險提示

Flag 3/8 無觸發（自動）；額外信號如下：

### 額外信號
- 🟠 HOME reverse platoon Δ +0.139（vs RHB OPS 0.723 > vs LHB OPS 0.584）— RHP 對非預期手別反而吃虧
- 🟠 HOME TTO3 penalty：OPS Δ +0.096（TTO1 0.506 → TTO3 0.602），第三輪明顯衰退；K% 從 24.7% 掉到 14.0%（Δ -10.7pp）
- 🟠 HOME chain breaks at #2-3：OPS 落差 0.180
- 🟠 AWAY chain breaks at #7-8：OPS 落差 0.160
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
- 🔴 ⏳ AWAY 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
  - 與 Burch Smith 預期 4-5 IP 短先發雙重作用：DET 從第 5 局起就要薄牛棚撐 4 局以上對 KC 主場主線（Witt + 4-5 棒 LHB），失分曲線偏陡。本場 KC 後段攻擊機會是直接打開的 — 屬本場最關鍵單點信號。

#### 額外手動風險（Flag 8 / Flag 3 narrative）
- **Burch Smith 小樣本（Flag 8）**：IP 11.3 / GS 0 / 全部牛棚短局拼湊；ERA 1.59 / FIP 1.34 不可當先發 baseline。formula HOME 1.5 run 受此 sample 拖累，**實際對 Smith 的得分期望值應顯著上修**（narrative 不入 ±run 信號欄，但在「修正後預期得分」段的整體判讀中明說）。
- **Wacha tier_mismatch（Flag 8）**：score gap -21、ERA-xERA Δ -1.25。屬「ERA 美化真實水準」narrative，AI 不下修 Wacha 預測，仍以 tier_v2 🟡 Solid Starter 為基準。

## 條件修正

- Park Factor: 106.0 → +0.30 run（Kauffman 利安打 / 三壘打，但 HR PF 91 壓制全壘打型攻勢；DET 多 LHB 拉打受 Kauffman 右外野空間影響中性偏中）
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：HOME tier 🟡 Solid（Wacha） vs AWAY tier 🟢 Back-end（Smith 修正後實際分級）— Tier gap 本場明顯偏 KC。非 doubleheader。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 1.5 | +0.7（platoon_advantage KC vs Smith ~+0.2 / core_il_count AWAY 3 → +0.5；chain_break #2-3 −0.1；total ≤ +0.8 cap） | 2.2 |
| AWAY | 4.9 | -0.3（reverse_platoon HOME（被 Wacha 壓 LHB） −0.2；chain_break #7-8 −0.1） | 4.6 |
| Total | 6.4 | +0.4 | 6.8 |

> ⚠️ formula HOME 1.5 受 Burch Smith 11.3 IP 樣本拖累嚴重低估。Flag 8 narrative：若把 Smith 視為 🟢 Back-end（合理 ERA 4.50-5.00），KC 真實期望接近 4.0-4.5。**信號欄保守 cap +0.7，但整體判斷段以 narrative 形式說明**：HOME 的 adjusted 2.2 屬「公式 floor」，narrative ceiling 約 4.0-4.5。

## 整體判斷

- **方向（基本面）**：HOME（KC）— 信號、牛棚、tier gap 三軸都偏 KC，DET 唯一賭注是 Burch Smith 小樣本續炸（低機率）。
- **總分（基本面）**：formula adjusted 6.8；narrative 上修 Smith 樣本後實際區間 7.5-9.0。
- **方向信心**：~65%。KC 勝面明確但不到 75% 的原因：(1) DET 打線 vs RHP season-level 還是 🟠 Strong，Greene / McGonigle / Carpenter 任一個在 LHB 對位下打開就足以追分；(2) Wacha tier_mismatch 提示其 underlying 比 ERA 差，DET 第二輪可能上量。
- **風險**：
  1. **Burch Smith 第 1-3 IP 神奇延續**：sample 雖小但球路 movement 真好就能壓 KC 到 5 局。若如此 KC 牛棚（Estévez 缺陣）能否撐 4 局是另一變數。
  2. **DET LHB 三人對 Wacha 突破**：Wacha vs LHB 110 BF 的 .584 OPS 是有意義樣本，但 McGonigle / Greene / Carpenter 任一人單場熱手都能破壞 Wacha 的反 platoon 信號。
  3. **DET 牛棚先頂 4-5 IP 的不確定性**：Jansen / Finnegan / Holton 三老將短期高負荷可能單場撐住，但連續對 KC 主線一次也許就崩 — 屬尾段風險。
  4. **Flag 3 lucky-hot 收斂風險（雙向）**：Witt（KC）last7 BABIP .400 + Greene（DET）last7 BABIP .688 都在過熱區，本場任一個冷下來會壓抑該隊上限；signal 中性，但對 narrative 信心有減項。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
