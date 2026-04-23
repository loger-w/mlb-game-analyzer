# Phase 3 綜合分析 — 2026-04-23 MIL @ DET

**比賽**：Milwaukee Brewers @ Detroit Tigers｜Comerica Park｜ET 13:10（美東）
**先發**：Brandon Sproat (R) vs Tarik Skubal (L)
**系列**：DET 主場系列第 3 戰（4/22 DET 5-2 MIL；4/21 MIL 12-4 DET）

---

## 投手對決

### Tarik Skubal（DET, LHP, 29 歲 ⚡ 巔峰期）— 🔴 **Elite Ace**
- **本季**：ERA 2.08 / FIP 1.95 / xFIP 2.59 / xERA 3.35 / K% 28.0 / BB% 5.1 / K-BB% 22.9 / IP 30.1
- **Prior Year (2025)**：ERA 2.21 / FIP 2.34 / K% 32.2 — 連兩年頂尖水準
- **Statcast**：avg velo 92.3 / hard-hit 25.6% / whiff 14.0 / CSW 29.5
- **Platoon**：vs RHB .189/.237/.256（BF 97，elite 等級）；vs LHB .300/.333/.550（BF 21，**小樣本且較差**）
- **ERA 閘門**：|ERA−xERA| = 1.27 < 1.5，**未觸發 YoY** ✓。但 xERA 3.35 暗示 ERA 有輕微 regression 空間
- **近 3 場**：vs SD 6IP 0ER、vs ARI 7IP 1ER、vs MIN 4.2IP 4ER（最近一場被打）

### Brandon Sproat（MIL, RHP, 25 歲 rookie）— ⚪ **Below Average**
- **本季**：ERA 6.88 / FIP 6.22 / xFIP 4.46 / xERA 5.47 / K% 21.6 / **BB% 14.9（嚴重控球問題）** / IP 17 小樣本
- **Prior Year (2025, 4 GS)**：ERA 4.79 / FIP 2.47 / 亦小樣本
- **Statcast**：avg velo 92.4 / hard-hit 23.5% / whiff 9.5（低）/ barrel 10.6%（偏高）
- **Platoon**：vs LHB .357/.500/.643（BF 38，**災難**）；vs RHB .242/.306/.455（BF 36）
- **ERA 閘門**：IP 17 < 30 但本季 ERA **高於** prior year 2.09（非低 ≥1.0）→ 未觸發 YoY ✓
- **近 3 場**：vs CHW 3IP 7ER、vs KC 3.2IP 4ER、vs WAS 3.2IP 1ER（都短局且失分）

### 對決結論
**投手層級差距 ~2.5 tier**（Elite Ace vs Below Avg）。Skubal 對右打無解，MIL 打線左打比例高但 Skubal vs LHB 正好是**小樣本弱點**（21 BF .300/.333/.550），可能部分抵銷。Sproat 對左打是災難（.500 OBP），DET 的 Greene（L）+ Carpenter（L）+ McGonigle（L）將形成威脅。

---

## 打線評級

### DET vs Sproat (RHP) — 🟠 **Strong**
- team OPS .755 / xwOBA .354 / BABIP .317 / K% 21.3 / BB% 10.1
- **近 7 天**：BABIP .341（不觸發閘門）、recent_heat ⚖️ Normal、OU_lean +1（略偏 over）
- **熱棒**：Riley Greene L7 OPS 1.175、Kerry Carpenter L7 .975、Dingler L7 .940、McGonigle L7 .923
- **冷棒**：Torkelson L7 .588、Keith L7 .556、Báez L7 .613
- chain：OBP_top3 .390、SLG_mid .399（結構中上）

### MIL vs Skubal (LHP) — 🟡 **Average**
- team OPS .748 / xwOBA .335 / BABIP .303 / K% 20.5 / **BB% 14.1**（選球好）
- **近 7 天**：BABIP .326、recent_heat ⚖️ Normal、OU_lean 0
- **傷兵失血**：🚨 **Yelich (10d)、Chourio (10d)、Vaughn (10d)** 三大主力 IL — 打線深度明顯下滑，Lockridge / Frelick / Rengifo / Hamilton 均 OPS < .700
- **熱棒**：Rengifo L7 .800、Mitchell L7 .791、Turang L7 .794
- **冷棒**：Frelick L7 .467、Bauers L7 .555、Sánchez L7 .647

### BvP 閘門
- 最大 BvP PA = 13（Sánchez vs Skubal）< 15 → **BvP 不可引用** ✓

---

## 牛棚與傷兵

| 項目 | DET | MIL |
|------|-----|-----|
| 牛棚 ERA | 4.40 | 4.31 |
| 主力 IL | 無核心 closer/setup | **Koenig (LHP setup, 15d)**、Priester、Yoho |

**牛棚雙向修正（MIL 側）**：
- ML 修正：**−2% 勝率**（主要 LHP setup 缺陣，對 Carpenter/Greene 後段 matchup 變差）
- OU 修正：**+0.15 run**（第 7-8 局漏水面向）
- **未觸發 B9 強制 TaskCreate**（僅 1 人核心 IL，非 3 人同時）

**DET 傷兵對得分影響**：位置球員 McKinstry / Sweeney IL 但已替換，影響中性。

---

## 條件修正（信號彙總）

| 信號 | 方向 | Run Value | 備註 |
|------|------|-----------|------|
| Skubal Elite Ace vs MIL average lineup | DET 防守 ↓ | −0.5 run（MIL 得分）| 層級差主導 |
| Sproat Below Avg vs DET strong lineup | MIL 防守 ↑ | +0.4 run（DET 得分）| xFIP 4.46 暗示不會這麼糟 |
| MIL 打線三主力 IL（Yelich/Chourio/Vaughn）| MIL 得分 ↓ | −0.3 run | 深度 & 右打 vs LHP 更弱 |
| MIL 牛棚 LHP setup (Koenig) IL | DET 得分 ↑ | +0.15 run | 後段 |
| Park Factor Comerica 99 | 中性偏投手 | −0.05 run | 微弱 |
| Skubal ERA−xERA 1.27（regression 潛力）| 輕微 MIL 得分 ↑ | +0.1 run | 未觸發硬閘門 |
| Sproat rookie + BB% 14.9 | DET 得分 ↑ | +0.1 run | 控球風險 |
| Early-season 樣本標籤 | 不確定性提高 | N/A | tag: early-season |

---

## 近期狀態

| 球隊 | 近 10 | 近 30 | 本季 | 趨勢 |
|------|-------|--------|------|------|
| DET（主）| 7-3（RS 4.7 / RA 3.9）| 13-12（4.36 / 3.88）| 13-12（25 場）| ↑ 上升 |
| MIL（客）| 5-5（RS 4.7 / RA 4.3）| 13-11（5.0 / 4.0）| 13-11（24 場）| → 持平 |

- DET 近 10 連勝 1，主場勢頭佳
- MIL 近 10 場失利後連敗 1，但近 30 得分力實際高於 DET（5.0 vs 4.36）
- H2H：4/22 DET 5-2、4/21 MIL 12-4 — 各 1 勝

---

## 修正後預期得分

**DET 進攻**：Sproat xFIP 4.46 基準 × ~4.5 IP = 2.2 ER + MIL bullpen 4.31 × 4.5 IP = 2.2 → 約 4.4；加 lineup strong +0.3、BB 機率高（BB% 14.9）+0.2 → **DET ≈ 4.9 runs**

**MIL 進攻**：Skubal ERA 2.08 / xERA 3.35 折中 × 6.5 IP = 1.9 ER + DET bullpen 4.40 × 2.5 IP = 1.2 → 約 3.1；扣 Yelich/Chourio/Vaughn IL −0.4、MIL 打線 average −0.1 → **MIL ≈ 2.6 runs**

**總分預估**：**7.5 runs**（線 7.45 → 接近 push）
**分差預估**：**DET +2.3**（線 −1.85 → DET 讓分面偏過）

---

## 整體判斷

**方向傾向**：**基本面偏 DET**
**信心程度**：中等偏強（投手差距 2.5 tier 主導）
**值得注意的風險**：
1. Skubal ERA 2.08 vs xERA 3.35 有 regression 潛力，真實水準更接近 xFIP 2.59
2. Sproat 小樣本 ERA 6.88，xFIP 4.46 顯示可能不會繼續這麼糟（MIL 先發不必然崩盤）
3. Early-season 樣本（IP 17-30）整體不確定性高
4. MIL 前場爆打 12 分（4/21）證明打線點火時仍有爆發力
5. Skubal vs LHB 小樣本弱點 + MIL 左打比例高的潛在反撲

**預測總得分接近 OU line（7.5 vs 7.45），不具方向優勢**。方向優勢主要在 ML（DET 贏）與讓分（DET 覆蓋 −1.85 的希望）。

> 盤口 & 星級推薦將由 Phase 4 `predict.py` 輸出，本 summary 僅為基本面快照。
