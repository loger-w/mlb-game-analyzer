## 投手對決

### Chris Bassitt (HOME, RHP, 37歲)
- **Tier 覆寫**：沿用腳本 ⚪ Below Average — 數據與年齡曲線一致
- 真實水平判斷：ERA 6.75 / xERA 6.23 / FIP 5.82 / K-BB% **-2.7%**（K 比 BB 還少，極端壞），WHIP 2.06，平均球速僅 84.4 mph。37 歲滑坡期 + 5 GS 樣本顯示**結構性退化**，並非 BABIP 噪音。vs LHB **.377/.478/.623（67 BF）** 是一片血洗區。近 3 場 11 ER/11 IP，無止血跡象。
- 對手打線威脅：Astros 雖整體 🟡 Average，但 vs RHP 火力強。**Yordan Alvarez (LHB) vs RHP OPS 1.113 + Bassitt vs LHB 1.101 OPS 被打**，是直接互相點燃的火藥庫。Christian Walker vs RHP 1.017、Carlos Correa last7 0.971 — 這場 Bassitt 撐 4 IP 都樂觀。

### Peter Lambert (AWAY, RHP, 29歲)
- **Tier 覆寫**：⚠️ **🟡 Solid Starter → 🟢 Back-end** — script 顯示 FIP **1.28** / xERA 2.94 過於亮眼，但**僅 GS=2 樣本** 完全不可採信。Lambert 生涯多年在 4.50-5.50 ERA 區間（Rockies 時期），近 3 場 4 ER/11 IP 已暴露真實水平。
- 真實水平判斷：球速 90.4 mph 一般，K-BB% 25% 是樣本噪音。**vs RHB 11 BF .400/.455/.500** 雖樣本小但呼應其右打苦手史。BAL 打線 1-3 棒以右打為主（Henderson 是 LHB 但 .760 vs RHP）。預期 5 IP 4-5 ER。
- 對手打線威脅：BAL last7 BABIP 0.248 偏低（後述），整體 OPS .710，對 Lambert 不算炸線級對手；但 Lambert 真實能力被 GS=2 過度美化。

## 打線評級

### HOME (BAL) — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用 🟡 Average。xwOBA 0.334 中性，OPS .710 偏低，K% 24% 偏高。1-3 棒 OBP .337 一般。對 Lambert（GS=2）反而可能有水準演出。

### AWAY (HOU) — 🟡 Average / 🔥 Hot
- **Tier 覆寫**：上修為 🟡↑ — script 雖標 Average，但 OPS .800 / xwOBA 0.335 / 中段 SLG .381 + 1-3 棒 OBP **.395** 是接近 🟠 Strong 邊界，加上 last7 Hot。對 Bassitt 這種被 LHB 屠殺的 RHP，威脅是**結構性放大**（Alvarez/Walker/Correa 對 RHP 全部 .950+ OPS）。

## 牛棚

| | HOME (BAL) | AWAY (HOU) |
|---|---|---|
| ERA / IL 數 / 核心 IL 估計 | 4.14 / 6 / **1-2 名核心**（Selby 60d、Kremer 屬先發暫不計入） | 6.27 / 9 / **3+ 名核心**（Brandon Walter 60d 主力後援、Cody Bolton 15d） |

### 牛棚雙向修正值
- HOME 牛棚（BAL，1-2 核心 IL）：對手 **+0.3** run | HOME ML **-2%**
- AWAY 牛棚（HOU，6.27 ERA + 3+ 核心 IL = 災難級）：對手 **+1.0** run | AWAY ML **-5%**

> AWAY 牛棚是本場最大的單一變量：Bassitt 早退 → HOU 牛棚要吃 4-5 IP，但 ERA 6.27 + 核心傷兵堆疊，BAL 中後段火力會被引爆。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.248):
  - 偏低值反映近 7 天運氣稍差，**可能小幅回歸至中性**（聯盟均值 ~.295-.300）。BAL xwOBA 0.334 / 整體 BABIP 沒嚴重偏離，本場修正空間有限。**不自動 ±run value**，敘事承認 BAL 有微幅向上空間，但因對手 Lambert 真實水平也不到 ace 級，對沖後不大幅調分。

- ⚠️ Lambert GS=2 樣本噪音：FIP 1.28 / xERA 2.94 為樣本極端值，AI 已在 Tier 覆寫降級處理。
- ⚠️ 4 月底全聯盟樣本 ~28-30 場，早季噪音偏大，信心降一檔。

## 條件修正

- Park Factor: Camden 96 → **-0.15 run**（注意 2025 季前左外野牆改造，3 年加權尚未完整反映打者友善方向，原 -0.20 略下修）
- 先發 tier：雙方皆非 🟡 Solid 以上（Bassitt ⚪ Below / Lambert 真實 🟢 Back-end）→ 不觸發雙 ace 下修信號
- 天氣：未提供，視為中性
- HOU 連敗動能：軟性因素，不調整方向

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (BAL) | 1.4 | + Lambert 樣本降級 +0.7 / HOU 牛棚災難 +1.0 / PF -0.15 / BABIP 微幅 +0.15 | **≈ 3.10** |
| AWAY (HOU) | 6.4 | + LHB platoon 火力 +0.3 / BAL 牛棚 +0.3 / PF -0.15 / 早季噪音壓回 -0.5（base 已過熱） | **≈ 6.35** |
| Total | 7.80 | net +1.65 | **≈ 9.45** |

> 註：base AWAY 6.4 來自 Bassitt 6.75 ERA 直推，但 starter ERA 過早季容易過熱。我把 -0.5 列為「噪音回歸」，避免 base 把 Bassitt 5 GS 失血視為穩態。

## 整體判斷

- **方向（基本面）**：HOU 大幅占優。投手對決、打線匹配、牛棚對比三項都站 HOU 一側。BAL 唯一籌碼是主場 + Lambert GS=2 不可信任。
- **總分（基本面）**：修正後總分 ≈ 9.45 vs O/U 9.0 → 差距僅 **+0.45 run**，**遠低於 1.5 PASS 門檻**。比分傾向 Over，但信號強度不足以下注。
- **信心**：MEDIUM-LOW — 樣本期早（4 月底，雙方 GS 樣本僅 2-5 場），雙方打線皆 🟡 Average，Bassitt 失控雖明顯但 5 GS 仍有逆轉空間，Lambert 真實能力高度依賴推估。
- **風險**：
  1. Bassitt 突然找回節奏（37 歲老投手偶有單場控球回歸）
  2. Lambert 真實能力若接近 FIP 1.28 而非 Back-end，HOU 大勝（這對 ML 推薦反而有利，但對 O/U 是雙邊風險）
  3. BAL last7 BABIP 0.248 反向回歸幅度比預期大
  4. HOU 連敗心理 + 客場 + 老牌打線爆發力滑坡（Altuve 近期 last7 OPS 0.394）
