## 投手對決

### Connelly Early (HOME, LHP, 24 📈 成長期)
- **Tier 覆寫**：降級 🟢 Back-end Starter（從 🟠 Strong Ace 大幅下調）— ERA 2.84 嚴重高估真實水平，xERA **5.09** + FIP 4.30 + xFIP 3.93 三項指標一致指向 Solid/Back-end 區間。Flag 8 -2.25 = ERA 受 sequencing/防守加持極多，6 GS 樣本不可信。
- 真實水平判斷：velo 88.1 mph（LHP 中等偏低），whiff 8.5% 偏低，主球種 FF/CH/SI 缺 swing-and-miss 武器。vs LHB .207 SLG .414（被 LHB 打長球）/ vs RHB .214 但 K-BB% 10.7 中等。真實 ERA 區間 **4.20-4.80**。
- 對手打線威脅：HOU 🟡 Average + Yordan Alvarez vs LHP **1.235** + Walker .778 / Correa vs LHP **1.133** → 對左投無解三人組，是 Early 最大噩夢。Altuve / Cam Smith vs LHP 弱（.703 / .439）拖低期望。

### Spencer Arrighetti (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 覆寫**：降級 🟡 Solid Starter（從 🟠 Strong Ace 下調）— ERA 2.00 / FIP 2.99 都優秀，但 xERA **5.03** + Flag 8 **-3.03** 是極端警訊。FIP-xERA 並存 2 vs 5 矛盾：FIP 2.99 反映 K/BB/HR 平衡好（K-BB% 15.4 + barrel 6.8 都優），但 xERA 5.03 暗示 contact quality 反指（possibly 防守 + sequencing 加成）。
- 真實水平判斷：velo 83.6 mph（**極低！** finesse pitcher type），主球種 CU 29.8% / FF 24.0% — 偏軟丟。vs LHB .178/.339/.244（56 BF 顯示控球差但壓 SLG）/ vs RHB .150/.227/.350（22 BF 太小）。真實 ERA 估 **3.50-4.20** — 落點在 Solid，不到 Ace。
- 對手打線威脅：BOS 🟢 Weak（OPS .661 / chain SLG mid .292 中後段極弱）。Contreras 0.832 + Abreu .815 是核心威脅；Story / Anthony / Durbin 都偏弱 → BOS 全隊對 Arrighetti 上風有限。

## 打線評級

### HOME — 🟢 Weak / ⚖️ Normal
- **Tier 覆寫**：沿用 🟢 Weak — chain SLG mid .292 是聯盟尾段，得分仰賴前 3 棒（Contreras/Abreu）。

### AWAY — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用 🟡 Average — chain OBP top3 .387 + chain SLG mid .404 都健康，Yordan vs LHP 是今晚最大武器。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.54 / 7 / **1-2 名核心**（Crochet 15d high-leverage swing）| **6.54** / 8 / **2 名核心**（Walter 60d / Javier 60d 多名重要）|

### 牛棚影響判讀
- HOME 牛棚（BOS）：ERA **3.54**（不錯）。Crochet 雖列為 starter，但 leverage 高，缺陣對後段有影響。Whitlock / Slaten / Jansen（assumed closer）等仍在。
- AWAY 牛棚（HOU）：ERA **6.54**（極差，全聯盟尾段）。Walter + Javier 雙 60-day → starter depth 崩盤，但對牛棚直接影響中等；牛棚整體 6.54 ERA 才是核心問題 — 後段任何時候上來都易失分。
- **HOU 牛棚 6.54 是 Total 偏 + 的最大引信**：若 Arrighetti 5-6 局 100 球後下，Total 容易爆。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-2.25):
  - **判讀為運氣為主**，ERA 2.84 vs xERA 5.09 差 2.25 — 6 GS 樣本被 BABIP / LOB% 加持極多。本場估真實水平在 Solid/Back-end 區間，**不下修預測**但敘事上對 BOS 失分容易性偏正面（HOU 攻勢預期會放大）。
- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-3.03):
  - **判讀為混合**，FIP 2.99 是真實良好（K-BB% 15.4 / barrel 6.8 / hard_hit 20.2 都優），但 xERA 5.03 反映 contact quality 異常。**真實在 Solid Starter** 區間，比 ERA 顯示的 Strong Ace 弱一檔；本場若 BOS 有零星硬擊，xERA 提示可能逆襲。

## 條件修正

- Park Factor: 104.0 → +0.20 run（Fenway 微利打者；HR -15% 抑制長球但利二壘安打 / Wall ball）
- 先發 tier：雙 🟡 Solid（覆寫後）/ 🟢 Back-end → 中段對決，無顯著一面倒
- 天氣 / DH：5/02 Fenway 春末晚場，無變數

## 修正後預期得分

> 「+ 信號」欄僅納入規範允許的條件修正：Park Factor、牛棚累計效應（核心 IL ≥ 2 名）、主力打者傷兵。
> ⛔ BABIP 極端值 / ERA-xERA gap **不入此欄**（規範禁止 auto ±run value，見 reference/flags-checklist.md §3, §8）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.2 | 0（HOU 牛棚 ERA 6.54 是品質問題非 IL ≥ 2 規範範圍）| 3.2 |
| AWAY | 5.1 | +0.5（HOU 牛棚高 ERA 已內含於 base，但 BOS 後段失分風險仍偏高 — 邊界保守）| 5.6 |
| Total | 8.3 | +0.5 | **8.8** |

## 整體判斷

- **方向（基本面）**：**偏 AWAY (HOU)** — Yordan vs LHP 1.235 + Correa 1.133 對 Early 是 lethal matchup；Arrighetti FIP 2.99 vs BOS 弱打線優勢明確。但 ML 差距不大（HOU 預期 53-56%）。
- **總分（基本面）**：**接近持平，base 8.3 + HOU 牛棚 ERA 6.54 隱憂 → 真實 8.5-9.5 區間**，與市場 9.0 接近。HOU 攻勢被 Yordan/Correa 集中爆發風險高（Over 偏向）；BOS 中後段疲軟拖低 Total 風險（Under 偏向）。**判斷：略偏 Over**。
- **信心**：**LOW** — **雙 Flag 8** 警訊（Early 運氣為主、Arrighetti 矛盾混合），加上 BOS 牛棚 ERA 3.54 vs HOU 牛棚 6.54 的不對稱 → 變數極大。
- **風險**：
  1. Early ERA 2.84 vs xERA 5.09 — 若回歸，BOS 攻勢面被放大
  2. Arrighetti FIP 2.99 vs xERA 5.03 矛盾 — 若 BOS 中段（Contreras/Abreu）突破，模型崩
  3. **HOU 牛棚 ERA 6.54 是定時炸彈** — 若 Arrighetti 早下 Total 易暴衝
  4. Yordan vs Early（LHP）OPS 1.235 — 單人即可決定本場走勢，方差極高

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
