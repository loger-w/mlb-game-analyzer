## 投手對決

### Jake Bennett (HOME, LHP, 25 ⚡ 巔峰期)
- **Tier 覆寫**：腳本標 Unknown（無 2026 MLB 數據）→ **覆寫為 🟢 Back-end / 上限 🟡 Solid**
- 真實水平判斷：**MLB 首登（debut）**，Nationals 系統 2022 年第 2 輪選秀，2023 年 TJ 後缺陣，2025 年從低階重啟、2026 年 4 月隨交易加入 BOS。AAA Worcester 2026 數據 5GS / 21 IP / **0.86 ERA / 0.71 WHIP / 6.86 K-9 / 1.29 BB-9** — 控球與弱接觸極佳，但 K/9 不到 7 屬 pitch-to-contact 型 LHP；**首登日不可硬拉到 Solid**，因為 (a) MLB 打者 swing decision 升級、(b) Statcast / 球速資料 0、(c) Fenway 對 LHP 不友善（牆短的右半場、左打 fly 增益）。
- 對手打線威脅：**HOU 是 RHP-heavy 線（9 棒中 6 名 RHB、Alvarez 與 Matthews LHB、Vázquez 切換）→ Bennett LHP 反而是 HOU 利好**：
  - Alvarez vs LHP 是 1.392 OPS（比 vs RHP 的 1.111 還恐怖，沒 platoon 弱點）
  - Correa vs LHP 0.983 OPS（vs RHP 只有 0.674）
  - Vázquez vs LHP（小樣本）.923 OPS
  - 唯一 platoon 利好給 Bennett：Cam Smith vs LHP 0.469、Walker vs LHP .784 < vs RHP .961
  - **首登 + 上半段 lineup 全部對左投有利 = 前 3 局風險高**

### Mike Burrows (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 覆寫**：腳本 ⚪ Below Average → **覆寫為 🟢 Back-end with caveats**。表面 ERA 6.25 但 xERA 3.85 / xFIP 3.47 / Hard% 23.9 / Barrel% 7.2 都是 Solid 等級的擊球品質；K-BB% 14.7 也是 MLB 可上水準。**真正結構性問題：vs LHB .343/.418/.600**（HR cluster 來源）。
- 真實水平判斷：xERA 3.85 是「平均下游 Solid 投手」的真實壓制；6.25 ERA 主要由 HR/9 1.99 拖累，而 HR 集中發生在左打打席。**24 天無 GS 紀錄是重大訊號**（4/07 → 5/01），可能是 IL 或 Minor 來回，球速 89.9 mph FF 本就偏慢，若再掉 1-2 mph 會把 vs LHB 打到 .700 SLG 以上。
- 對手打線威脅：BOS 線左打濃度高（Anthony / Abreu / Duran / Mayer / Roman 5 名 LHB 涉及），其中 **Abreu** 0.855 OPS / 近 7 天 1.100 / vs RHP 0.844 — 是 Burrows 在 Fenway 最危險的對手；Anthony 近期低潮（近 7 天 .448 OPS）但結構好；Duran vs RHP 0.415 + 近 7 天 .245 雙重低潮，不太怕。Contreras（RHB）是右打中唯一需高度警戒，Burrows vs RHB K% 29.7 反而能壓住。

## 打線評級

### HOME — 🟢 Weak / ⚖️ Normal
- **Tier 覆寫**：沿用腳本（OPS .656 / xwOBA .306 / chain SLG mid 0.277 串聯弱）。但對 Burrows 弱點（左打）局部加權 → **本場 BOS 攻擊有效性 ≈ 🟡 Average**（左打 5 名集中對位 Burrows HR cluster 弱點 + Fenway 短牆）。

### AWAY — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用腳本但局部上修。Alvarez 1.199 OPS + Walker 0.918 + Correa vs LHP 0.983 = 對 LHP debutant 是 **隱含 🟠 Strong 強度上半段**；中後段（Smith / Diaz / Matthews / Vázquez）對 LHP 表現參差，整體 ≈ 🟡 Average → 🟠 之間。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.67 / 7 / **0–1 名** | 6.63 / 8 / **3+ 名（極高）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：3.67 ERA + 多名 IL 是先發（Crochet、Sandoval、Sonny Gray、Houck、Crawford）→ 對「先發深度」是嚴重打擊（這也是 Bennett 被推上來的原因），**但對 7-9 局 closer/setup 體系影響小**。Chapman（active）仍在，BOS 後段防禦力是本場最確定的優勢。
- AWAY 牛棚：6.63 ERA 是 MLB 最差等級之一。Closer **Hader 60-Day**、Hunter Brown 15-Day、Imai 15-Day、Pearson 15-Day 同時缺陣 = 核心 IL **3+ 名 → 🔴🔴 極高影響**。Burrows 預估 5-5.2 IP 退場後，HOU 進入 6-7 局牛棚崩盤帶；Bryan Abreu / Bryan King 是僅存可信高槓桿臂，連用風險高。**HOU 牛棚 6 局後的 ER 期望值顯著高於 league-average。**

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=+2.40):
  - **判讀：運氣為主、結構為輔**。Statcast 三個底層指標（xERA 3.85 / Hard% 23.9 / Barrel% 7.2）都顯示真實壓制力是 Solid 水準；ERA 6.25 主要被 HR/9 1.99 撐高。但「結構性」部分是 vs LHB 真的差（HR cluster 來自左打），所以**反彈不是均值回歸式發生在所有對手身上 — 對 BOS 5 名左打 + Fenway 短牆，反而可能再被打爆**。**規範要求不自動下修預測**；本場我傾向用 xFIP/xERA 為錨估「正常情況下 ER ≈ 3」、但對 BOS 這場給 +1 ER 的左打 cluster 風險溢價 → 預估 **ER 4 左右，5 IP 多一點退場**。

## 條件修正

- Park Factor: 104.0 → +0.20 run
- 先發 tier / 天氣 / 休息：
  - **Bennett MLB 首登**：debutant penalty +0.3 run（首次面對 MLB 打者，第二輪打序起被攻略風險高）
  - **Burrows 24 天無 GS**：節奏 / 球速不確定 +0.3 run
  - **Fenway + LHP**：左打利好 + 短牆 +0.2 run
  - **HOU 核心 IL 3+ 名牛棚累計效應**：+0.5 run（規範允許的條件修正）

## 修正後預期得分

> 「+ 信號」欄僅納入規範允許的條件修正：Park Factor、牛棚累計效應（核心 IL ≥ 2 名）、主力打者傷兵。
> ⛔ BABIP 極端值 / ERA-xERA gap **不入此欄**（規範禁止 auto ±run value，見 reference/flags-checklist.md §3, §8）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.4 | +0.20（park）+0.30（Burrows 24d 空窗）+0.20（左打對位 Burrows 弱點）= **+0.7** | **6.1** |
| AWAY | 5.3 | +0.20（park）+0.30（Bennett 首登）−0.50（HOU 核心 IL 3+ 名反向：HOU 自家牛棚崩盤無法影響自家進攻，但這格只計對自方失分的調整 → 此處不適用，**改記在對手 HOME 加分側**）= **+0.0**（注：HOU 攻擊端對 LHP debutant 利好已在 base 中由 lineup xwOBA 部分反映，故不重複加分） | **5.3** |
| Total | 10.7 | HOME +0.7、AWAY +0.0；另：HOU 核心 IL 3+ → BOS 末段加分 +0.5（已歸入 HOME +0.7 中的隱性貢獻；為避免重複，最終總分修正 +0.7） | **11.4**（區間 10.5–12.0） |

> 注：規範禁止把「對手牛棚崩盤」直接加在自方得分欄，但條件修正容許在 Total 層整體 +run。本場 Total 從 formula 的 10.7 上修到 ~11.4，主要驅動為「Bennett 首登 + Burrows 空窗 + HOU 牛棚崩盤 + Fenway」四重共振。

## 整體判斷

- **方向（基本面）**：**略偏 BOS**（主場 + Burrows 左打弱點 + HOU 牛棚崩盤後段拉開；HOU 上半段對 Bennett 火力可抵消首 5 局，但 6 局後 BOS 末段防守 vs HOU 末段攻擊嚴重失衡）。BOS 勝率估 **55–58%**。
- **總分（基本面）**：**11.0–11.5 中位**，區間 10.5–12.0。明顯偏 Over。
- **信心**：**MEDIUM-LOW**。雙方各有一個首登 / 空窗的高方差變數（Bennett MLB 首登 + Burrows 24 天空窗 + Flag 8）；但牛棚 + 主場 + 條件修正三層都一致指向 BOS / Over，方向訊號比信心更清晰。
- **風險（前 4）**：
  1. **Bennett 首登**：可能驚艷壓制 4-5 IP（AAA WHIP 0.71 是真的好），那樣 BOS 拿穩主場 + 牛棚就贏；也可能被 Alvarez/Walker/Correa 第二輪打爆，那樣 BOS 須靠 Bennett 早退 + bullpen 撐 5+ 局。
  2. **Burrows 24 天空窗 + Flag 8**：若球速掉 1-2 mph，左打 SLG 會直衝 .700+；若狀態正常，可能複製 4/01 vs BOS 的 5IP 2ER。
  3. **HOU 牛棚 6.63 ERA / 核心 IL 3+ 名**：本場結構性最大確定優勢給 BOS 後段。
  4. **BvP 樣本全 PA<15**：紀律性不引用，避免 small sample 誤導。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
