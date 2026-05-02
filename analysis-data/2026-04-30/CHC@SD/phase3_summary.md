## 投手對決

### Matt Waldron (HOME, RHP, 29歲, ⚪ Below Average)
- **Tier 覆寫**：⚪ Below Average（沿用腳本，但向上微修至「⚪ 但有結構不確定性」）
- 真實水平判斷：ERA 12.46 / xERA 4.68 / FIP 5.87 / xFIP 4.37 — Flag 13 觸發。Waldron 是蝴蝶球先發，蝴蝶球本身就高方差；本季 GS=2、BF=47 屬極小樣本，12.46 ERA 含明顯運氣與序列噪音，但 xFIP 4.37 / K-BB% 6.4 / WHIP 2.31 / whiff% 6.8 顯示底層水平也只是 Below Average（不是被低估的 Solid）。真實天花板大概落在 ERA 5.00-5.80 區間，遠不到 12.46，但也別把他當 4.68 解讀。
- 對手打線威脅：Cubs 打線左打濃度高（Happ S/L、Busch L、PCA L），Waldron vs LHB .560/.607/1.080（28 BF）是結構性弱點 — 蝴蝶球對左打的旋轉軸經常吃虧（已知 platoon 缺陷），這個訊號在小樣本下仍然可信。Bregman/Busch/Happ/PCA 近 7 天 OPS 皆 .800+ 處於熱手期。

### Jameson Taillon (AWAY, RHP, 34歲, 🟢 Back-end Starter)
- **Tier 覆寫**：🟢 Back-end Starter（沿用腳本）
- 真實水平判斷：ERA 4.55 / xERA 4.59 / FIP 5.88 / xFIP 4.24 — ERA 與 xERA 收斂良好，K-BB% 11.3 與 WHIP 1.30 都是稱職 Back-end 水準。FIP 5.88 偏高來自 HR/9 較高（球速 86.0 mph 平均、94.7 max 屬中下），34 歲球速持續退化但控球維持。近 3 場 9 ER / 16.7 IP（ERA ~4.86）大致符合本季水平，無趨勢性惡化。
- 對手打線威脅：Padres top hitters 大多是 RHB（Tatis / Bogaerts / Machado / Laureano），無 platoon 加成；近 7 天 Bogaerts 1.010 / Machado .919 處於熱手期，Tatis / Merrill / Laureano 反而冰冷。Taillon 給左打 OPS .833、給右打 OPS .775 — 兩邊均勻偏高，HR 風險存在但不致命。

## 打線評級

### HOME (San Diego Padres) — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：🟡 Average（沿用腳本）。xwOBA .321 / OPS .676 實際偏低，1-3 棒 OBP .305 啟動力一般，4-5 棒 SLG .424 中段清壘力中等。本季 RS 4.66 但近 10 天 RS 跌到 4.50。最大威脅集中在 Bogaerts / Machado 的中段熱手。

### AWAY (Chicago Cubs) — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：🟡 Average + half tier（接近 🟡→🟠 邊緣）。xwOBA .334 / OPS .774 都優於 SD，1-3 棒 OBP .353 啟動力佳；近 10 天 RS 5.40 / 30 天 RS 5.33 持續火熱。對 Waldron 的左打優勢與整體熱度疊加，本場打線端有微幅優勢。

## 牛棚

| | HOME (SD) | AWAY (CHC) |
|---|---|---|
| ERA / IL 數 / 核心 IL 估計 | 4.14 / 7 / 1-2 名核心估計（Hoeing IL60、Canning IL15 偏中段角色） | 3.75 / 10 / 1-2 名核心估計（Horton IL60、Thielbar IL15 — Thielbar 是 high-leverage LH）|

### 牛棚雙向修正值
- HOME 牛棚（Padres）：對手 +0.3 run | HOME ML -2%（IL 7 名但多為中後段先發/中繼，Closer Suarez 假設可用，Padres 實際牛棚深度尚可）
- AWAY 牛棚（Cubs）：對手 +0.3 run | AWAY ML -2%（10 名 IL 看似多，但 Horton 是先發、Thielbar 是 high-lev LH 所以扣 1 個核心）

兩邊牛棚扣分相當；Cubs ERA 3.75 vs Padres 4.14，Cubs 實質仍略優。

## 風險提示

- ⚠️ HOME 投手 Flag 13 (era_xera_delta=7.78):
  - 判讀：**運氣 + 結構** 混合 — 12.46 ERA 含 GS=2 序列噪音與蝴蝶球高方差；xERA 4.68 / xFIP 4.37 是合理的「重力中心」。本場分析使用 ERA ~5.50 的 blended 水平（不直接使用 12.46，也不直接信任 4.68），不自動下修 Cubs 預期得分到底層 xERA 對應水平。Waldron vs LHB 的 1.080 OPS 是可信的結構性 platoon 訊號，繼續引用。

## 條件修正

- Park Factor: 95.0 (Petco) → -0.25 run（已在 base 公式中反映）
- Petco HR factor +7%（slight HR boost，但 Runs 95 主導）
- Taillon vs lg avg 休息日：標準 5 天（無修正）
- 雙先發 tier 都不到 Solid+ 等級 → 無 -1.0 ace 折扣信號
- Doubleheader：N
- 天氣：Petco 4/29 通常溫和（無極端風 / 高溫）

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (SD) | 6.1 | -0.5（Taillon Back-end 但本季波動小、SD 打線中段熱、AWAY 牛棚 +0.3 互抵）| 5.6 |
| AWAY (CHC) | 6.3 | -0.6（Waldron Flag 13 部分回歸 -1.0；Cubs vs Waldron LHB 結構優勢 +0.4；HOME 牛棚 +0.3 含於原始）| 5.7 |
| Total | 12.4 | -1.1 | 11.3 |

> 註：base 6.1/6.3 含 Waldron 12.46 ERA 完整效應 → 必須做 Flag 13 部分回歸。回歸後 Total 11.3 仍明顯高於 9.0 line。

## 整體判斷

- **方向（基本面）**：Cubs 微幅優勢 — 投手端 Taillon 4.55 ERA 實質遠優於 Waldron blended ~5.50 ERA；打線端 Cubs xwOBA / OPS / 近期熱度全面領先；牛棚端 Cubs 略優；唯一 SD 優勢是主場 + 蝴蝶球突發性可能擾亂 Cubs。
- **總分（基本面）**：明確 OVER 傾向。修正後總分 11.3 vs O/U 9，差距 +2.3 落在 ⭐⭐⭐⭐ 區間。即使做更激進的 Waldron 回歸（拉到 base -1.5），Total 仍在 10.0+。
- **信心**：MEDIUM（Waldron 高方差 + 蝴蝶球單場可能突然壓制 Cubs，是主要的下行風險；Padres 牛棚若被迫早接也可能進一步推升 Total）
- **風險**：
  1. 蝴蝶球高方差 — Waldron 任一場可能突然 6 IP 1 ER，雖機率低但會直接打掉 OVER
  2. Padres 主場 9 局下半不打 → 影響 RL 與 OVER 的最後 0.5 局得分機會
  3. CHC G1 已經 8-3 大勝，部分主力可能輪休或集中度下降（需確認下午 lineup 出爐後）
  4. Petco 入夜後濕氣抑制飛球 — 雖 4/29 屬於白天場（ET 15:10）影響較小

⛔ MUST NOT contain：星級、明確盤口推薦
