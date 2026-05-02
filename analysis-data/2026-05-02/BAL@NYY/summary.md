## 投手對決

### Ryan Weathers (HOME, LHP, 26 ⚡ 巔峰期)
- **Tier 覆寫**：沿用腳本 🟡 Solid Starter，但**附 ERA-xERA 警示**（gap −1.43）
- 真實水平判斷：ERA 3.21 偏幸運（xERA 4.64），但 xFIP 2.56 / FIP 3.37 / K-BB% 23.4 都顯示球種品質與壓制力是真材實料；max velo 101 mph + whiff% 12.1 表示有 swing-and-miss 工具。落差來自 BABIP / HR-FB 的回歸空間，真實水平在 🟡 Solid Starter 中段（ERA 預期靠近 3.7–4.0）。年齡 26 巔峰，趨勢正向。**最大弱點**：vs RHB SLG .465（109 BF 樣本足），右打能打長打。
- 對手打線威脅：BAL 1-3 棒（Ward 右、Henderson 左、Rutschman switch）對 LHP OPS 都 .869+，是真威脅；但 4-5 棒 Alonso .367 / O'Neill .286 對左投是大破口。Mayo (.845) / Jackson (.901) / Wilson (.817) 三位右打對 LHP 數字漂亮但樣本偏小，需注意 Weathers 對右打 SLG .465 的破口。**整體威脅：中等偏高**。

### Kyle Bradish (AWAY, RHP, 29 ⚡ 巔峰期)
- **Tier 覆寫**：沿用腳本 🟢 Back-end Starter，但**真實水平偏向 ⚪ Below Average**
- 真實水平判斷：ERA 4.20 / xERA 4.46 數字接近，沒有運氣修正空間；K-BB% 10.3（聯盟平均 ~13）、WHIP 1.73 都偏弱；velo avg 89.6 對 RHP 偏低。近 3 場 ER 8 / 13.7 IP → ERA 5.27 顯示狀況下滑。**致命弱點**：vs RHB OPS 1.102（50 BF 樣本偏小，但 .422/.480/.622 三圍極端，跟 vs LHB .219/.326/.342 形成反向 platoon — sweeper-heavy 投手對右打反而 platoon 不利的典型模式）。
- 對手打線威脅：NYY 右打代表 Aaron Judge（season 1.019 / vs RHP .948 / last7 1.342）+ José Caballero（last7 1.050）+ switch-hit 但對 RHP 打左的 Ben Rice（vs RHP 1.112 / last7 1.165）— 雖然 Rice 是左打但 Bradish 對 LHB SLG .342 也只是普通壓制。**整體威脅：高**。Judge/Rice 兩人就足以單場決定走向。



## 打線評級

### HOME — 🟡 Average / ⚖️ Normal
- 打線來源：🟢 official
- **Tier 覆寫**：上修為 🟠 Strong（vs Bradish 特定情境）
  - 整體 xwOBA .333 / OPS .698 看似 Average，但這是 last7 BABIP .260 拉低後的數字；Aaron Judge（season OPS 1.019 / last7 1.342 / Barrel 26.3%）+ Ben Rice（season 1.169 / vs RHP 1.112）兩位巔峰級打者足以撐起 chain。
  - chain.obp_top3 .378 是聯盟前段；Bradish vs RHB OPS 1.102 + 右打 Judge/Caballero 兩位 → 局部上修。
  - 4-6 棒 Bellinger/Chisholm/Domínguez 是 LHB 為主，遇 RHP Bradish 較中性，**non-platoon advantage 但仍是平均水準**。

### AWAY — 🟡 Average / ⚖️ Normal
- 打線來源：🟢 official
- **Tier 覆寫**：沿用 🟡 Average（無覆寫）
  - xwOBA .332 / OPS .733 / chain top3 .361 / slg_mid .345 都在聯盟平均區間。
  - 1-3 棒 Ward/Henderson/Rutschman 對 LHP OPS .869~1.035 是強段，但 4-5 棒 Alonso (.367) + O'Neill (.286) 對 LHP 是平均以下的低點 — chain 容易斷在中間。
  - 缺 Holliday (10d) / Westburg (60d) / Mountcastle (60d) 三位內野主力，替補 Jackson/Wilson/Alexander 對 LHP 數字看起來還可以但樣本量都不大，可信度有限。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.69 / 4 / **1**（Chivilli） | 3.97 / 7 / **2**（Bautista closer + Helsley closer/setup） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 3.69 較佳，4 位 IL 中只有 Chivilli 是純牛棚（其餘 Rodón/Schmidt/Cole 都是先發），核心 IL ≈ 1 → 🟠 中高但可吸收。可用 Bednar / Doval 雙 closer / setup 配置 + Cruz / Hill / Bird，後段防守仍紮實。前一場 G1（5/1）勝出 7-2，牛棚使用相對保留，今天可用性高。
- AWAY 牛棚：ERA 3.97 中等，但核心 IL **2 名**（Bautista 是原 closer / Helsley 是補進的另一位 closer 級），加上 Selby（牛棚投手）長傷 → 🔴 高影響，等同 closer 直接缺失 + setup 變薄。剩下 Kittredge / Cano / Akin 撐後段，遇 Yankees 中後段打線顯著吃緊。本場若 Bradish 早退，牛棚消耗放大為串聯風險。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.26):
  - **判讀：可能回歸（不自動 ±run value）**。Yankees 近 10 場 8-2 / RS 5.00，攻擊產出沒下滑；low BABIP 與 last7 OPS（Judge 1.342 / Rice 1.165 仍極熱）並存，代表 BABIP 偏低集中在 4-9 棒（Bellinger .586 / Domínguez .311 / Escarra .669 / McMahon .565 / Caballero 1.050 — 跟 BABIP 0.26 不一致 → 應屬其他棒次拉低）。對本場判讀影響輕微：基本面仍偏向 Yankees 攻擊，但若 Judge/Rice 任一啞火，整體 chain 容易斷。
- ⚠️ Bradish vs RHB 50 BF 樣本警示：OPS 1.102 是極端值但樣本只有 50 BF（Flag 1 邊緣）— 仍應視為訊號（platoon-reverse 是 sweeper-heavy RHP 的已知模式），但**不自動 ±run value**，留意實戰結果可能比預期溫和。

## 條件修正

- Park Factor: 96.0 → -0.20 run（Yankee Stadium runs PF 96 投手友善，但 HR PF +12% 對 LHB 短右外野有利 — Ben Rice / Bellinger / Chisholm / Domínguez 是受惠群）
- 天氣：Cloudy, 63°F, wind 8 mph, Out To LF
  - 影響判讀：63°F 在中性下緣（50-60°F 才微利投，剛過下界）；wind 8 mph 是輕度區間下緣，Out To LF 對右打 pull-side 有利 — 直接受惠者是 Aaron Judge（RHB pull-LF）。整體**輕度利攻 + 輕度利右打 HR**，不入 formula 但敘事方向一致。
- 先發 tier / doubleheader：本場非 doubleheader（13:35 ET 日場），無先發降級議題；Bradish 季初 GS 6 表示已穩定先發中，非角色轉換。NYY 系列 G2、本季已 6 GS，例行日場常規。

## 修正後預期得分

> 「+ 信號」欄僅納入規範允許的條件修正：Park Factor、牛棚累計效應（核心 IL ≥ 2 名）、主力打者傷兵。
> ⛔ BABIP 極端值 / ERA-xERA gap **不入此欄**（規範禁止 auto ±run value，見 reference/flags-checklist.md §3, §8）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.4 | Park −0.10（runs PF 96 平分） / BAL 牛棚核心 IL ≥ 2 名 +0.50 | **4.8** |
| AWAY | 3.7 | Park −0.10 / NYY 牛棚核心 IL 1 名 +0.10 | **3.7** |
| Total | 8.1 | 淨 +0.40 | **8.5** |

## 整體判斷

- **方向（基本面）**：**Yankees 中度偏優**。三大核心訊號疊加：(1) Bradish vs RHP 1.102 OPS 對上 NYY 右打 Judge / Caballero + LHB 但對 RHP 強勢的 Rice；(2) BAL 牛棚核心 IL 2 名（Bautista + Helsley 兩位 closer 級），中後段防守顯著吃緊；(3) Yankee Stadium HR PF +12% + wind out to LF 利 Judge pull-side。Weathers 雖 ERA-xERA 有回歸風險，但 K-BB% 23.4 / xFIP 2.56 仍在 BAL 中段斷鏈處有壓制力。
- **總分（基本面）**：**adjusted 8.5（base 8.1 + 0.4）**。中性偏高，主要由 BAL 牛棚 IL 推升 NYY 端攻擊空間。
- **信心**：**MEDIUM**。投打與牛棚訊號一致指向 NYY，但 (a) Bradish vs RHB 50 BF 樣本偏小、(b) Weathers ERA-xERA gap −1.43 有回歸空間、(c) BAL 1-3 棒對 LHP 強段（Ward/Henderson/Rutschman OPS .869~1.035）可能撐住攻擊 — 三點限制信心上修。
- **風險**：
  1. Bradish 對右打 50 BF 是邊緣樣本，platoon-reverse 訊號可能比實戰溫和（不自動修正但提醒）。
  2. Weathers ERA 3.21 / xERA 4.64 顯著落差，本場若 BABIP / HR-FB 回歸，HOME 失分上修空間大。
  3. BAL 1-3 棒對 LHP 都是強段，Ward 對 LHP 1.035 OPS 可能單場炸開，限縮 NYY 領先幅度。
  4. NYY 打線 last7 BABIP .260 雖判讀偏向其他棒次回歸，若 Judge / Rice 突然降溫，整體 chain 斷裂風險高（核心 over-reliance）。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組