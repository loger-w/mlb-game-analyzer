# Phase 3 分析結論 — STL @ MIA（2026-04-22 ET 12:10）

## 先發投手

### Janson Junk（MIA，右，30 歲，🟢 Back-end Starter）
- 2026：4 GS / 22.0 IP / ERA 4.50 / **xERA 3.57** / FIP 3.74 / xFIP 3.68 / WHIP 1.32
- K% 16.1 / BB% 6.5 / HR/9 0.82 / **GB% 59.6**（高滾地率）
- Statcast：avg velo 87.6、barrel 4.3%、whiff 8.3%、EV95 42.9%、hard-hit 26.5%
- Platoon：vs L `.388/.434/.551`（53 BF）、vs R `.111/.179/.222`（40 BF） — 左打噴
- 2025 PY：110 IP、4.17 ERA / 3.00 FIP（先發為主）
- **評語**：ERA 高於 xERA 0.93（未觸發 YoY gate），高 GB% + 低 barrel，表現接近 back-end tier。

### Kyle Leahy（STL，右，28 歲，⚪ Below Average）
- 2026：4 GS / 19.0 IP / ERA 5.21 / **xERA 5.71** / **FIP 5.73** / xFIP 4.09 / WHIP 1.53
- K% 15.1 / BB% 9.3 / **HR/9 1.89** / GB% 61.9
- Statcast：avg velo 89.4、**barrel 14.3%**、whiff 6.7%、EV95 49.2%、hard-hit 31.0%
- Platoon：vs L `.400/.478/.650`（46 BF）、vs R `.139/.205/.361`（40 BF）
- **2025 PY：62 games / 1 GS → 牛棚**（ERA 3.07 / FIP 2.98 / 88 IP）
- ⛔ **role_change（reliever → starter）**：prior year 數據不能直接用於先發評估；實績期 4 場 GS 樣本極小
- **評語**：xERA、FIP、barrel% 三項全線崩，沒有回歸空間；左打威脅極大。

**投手差距**：Junk（🟢 Back-end）> Leahy（⚪ Below Avg）約 **2 檔**，xERA 差距 2.14。

---

## 打線

### MIA（🟢 Weak） vs Leahy（R）
- avg OPS 0.698 / avg xwOBA 0.303 / avg BABIP 0.299 / last7 BABIP 0.308
- K% 22.2 / BB% 8.7 / heat ⚖️ Normal
- 關鍵左打：Xavier Edwards（.914 OPS）、Otto Lopez（.894）、Liam Hicks（.900）
- Leahy vs L `.400/.478/.650` + 本季 HR/9 1.89 → **MIA 左打有機會單場爆炸**
- BvP：全員 PA < 15，不引用

### STL（🟡 Average） vs Junk（R）
- avg OPS 0.696 / avg xwOBA 0.316 / avg BABIP 0.281 / last7 BABIP 0.314
- K% 22.5 / BB% 9.9 / heat ⚖️ Normal
- 關鍵左打：Burleson（.803）、Gorman（.598）、Scott II、Church
- Junk vs L `.388/.434/.551` 樣本小（53 BF）但數據差 → STL 有切入點
- 近 10 場 7-3、趨勢 ↑；Jordan Walker 右打 .964 OPS
- BvP：全員 PA < 15，不引用

---

## 牛棚雙向修正

| 項目 | MIA | STL |
|------|-----|-----|
| 牛棚 ERA | **3.17** | **5.34** |
| 差距 | — | +2.17（STL 差） |
| 核心 IL | 無（Mazur/Henriquez 60-D 非核心） | Hunter Dobbins 15-D、Matt Pushard 15-D（非核心） |

- **ML 修正**：STL 牛棚弱、落後時難守 → MIA +2% ML 加成
- **O/U 修正**：STL 牛棚 ERA 高（5.34）+ Leahy 提前被打爆 → +0.3 runs 偏大分
- B9 閘門：未觸發（雙方核心 closer/setup 皆健康）

---

## 條件修正（Run Value）

| 信號 | 觸發 | RV | 方向 |
|------|------|-----|------|
| Leahy role_change（reliever→starter） | ✅ | +0.3 runs | 利 MIA 得分 / 偏大分 |
| Leahy FIP 5.73 + HR/9 1.89 + barrel 14.3% | ✅ | +0.25 runs | 偏大分 |
| Junk xERA < ERA (3.57 vs 4.50) | ✅ | −0.2 runs | 偏小分 / STL 被壓制 |
| loanDepot Park Factor 98 | ✅ | −0.1 runs | 微偏小分 |
| MIA 打線 🟢 Weak（OPS 0.698） | ✅ | −0.2 runs | 偏小分 |
| BABIP 回歸（last7 皆 .30x） | 未觸發 | — | — |
| BvP（全員 PA<15） | 未觸發 | — | — |

**淨 O/U 修正**：+0.3 + 0.25 − 0.2 − 0.1 − 0.2 = **+0.05 runs**（近乎中性，偏極微大分）

---

## 修正後預期得分

- **基礎**（季 RS/G 與對手投手 tier 綜合）：
  - STL vs Junk（Back-end）：4.83 × 0.95 ≈ **4.6**
  - MIA vs Leahy（Below Avg）：4.50 × 1.15 ≈ **5.2**
- 牛棚修正：STL 牛棚弱 → MIA 加 +0.3、STL 微扣 −0.1
- Park/環境：loanDepot PF 98 → 雙方微扣 −0.05

**預估終場比分**：MIA **5.4** vs STL **4.5**（總分 **9.9**）

---

## 近期趨勢

- **STL**：近 10 場 7-3、RS/G 5.1 / RA/G 5.0、趨勢 ↑；本季 14-9（強於期待）
- **MIA**：近 10 場 3-7、RS/G 4.1 / RA/G 5.3、趨勢 ↓；本季 11-13
- H2H：昨日 STL 5-3 贏、前日 MIA 5-3 贏（系列分裂）
- BABIP 回歸閘門：**未觸發**（MIA last7 .308 / STL last7 .314）

---

## 整體判斷

1. **投手對決**：Junk 優於 Leahy 約 2 檔（xERA 3.57 vs 5.71）。Leahy 是前牛棚 role_change 案例，第二輪（round）之後表現快速惡化風險高 → **方向偏 HOME（MIA）**
2. **牛棚**：MIA 3.17 vs STL 5.34，落後方難逆轉 → 強化 MIA 方向
3. **打線**：雙方打線同屬 Weak/Average，均 vs LHP 不佳但 Leahy vs L 被噴、Junk vs L 也不好 → 左打端互相剋制
4. **O/U**：總和信號近中性偏極微大分（+0.05）；預估 9.9 vs 盤口 9.75 → 差 0.15（薄）
5. **風險**：
   - Leahy 樣本僅 4 GS，真實能力不明；若 MIA 打線冷打，分數可能低於預期
   - MIA 近期 3-7 冷、整體打線弱 → MIA 雖有投手優勢但得分能否到 5+ 有不確定
   - 系列前兩場比分 3-5 / 5-3 → 比賽節奏偏低

**方向性傾向**：基本面偏 **HOME（MIA）** 在 ML 面；O/U 中性偏極微 Over（差距不足推薦）。

盤口推薦留待 Phase 4 `predict.py` 模型判定。
