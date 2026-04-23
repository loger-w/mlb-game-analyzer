# Phase 3 綜合分析 — CIN @ TB (2026-04-22)

## 賽事基本資訊
- 時間：ET 2026-04-22 13:10 / 台灣 04/23 01:10
- 球場：Tropicana Field（PF 96，pitcher-friendly）
- 客：Cincinnati Reds（近 10：8-2，連 5 勝，run diff +20）
- 主：Tampa Bay Rays（近 10：6-4，連 3 敗，run diff -5；季 -17）
- 前場（4/21）：CIN 12-6 大勝 TB；本系列 CIN 已 2-0

---

## §投手對決

### Nick Martinez（TB, R, 35yo）
- Tier：🟠 Strong Ace（腳本分級偏高，實力未必匹配）
- ERA 2.45 / FIP 4.55 / xFIP 4.38 / **xERA 4.66**（ERA 與 xERA 落差 2.21，幸運值極高）
- K% 15.2（低）/ BB% 7.6 / K-BB% 7.6
- GB% 47.1（善用 Trop 地形）
- 球速 86.6 mph（35 yo 明顯退化）
- 球種：CH 28.6 / SI 27.5 / FC 20.2（軟接觸 profile）
- Platoon：vs L .259/.333/.444（60 PA）— 對左打脆弱；vs R .194/.219/.290（32 PA SSS）

### Brandon Williamson（CIN, L, 28yo）
- Tier：🟢 Back-end Starter
- ERA 4.35 / FIP 5.71 / xFIP 5.96 / **xERA 6.78**（peripherals 災難級）
- K% 13.3（極低）/ BB% 14.4（災難）/ **K-BB% -1.1**（負值）
- GB% 29.2（高飛球型）/ barrel% 11.1（高）
- 球速 87.3 / Whiff 8.9%
- Platoon：vs L .250/.357/.750（14 PA 極 SSS）/ vs R .197/.329/.295（76 PA BB% 15.8 控球失控）

### §YoY 對比結論（ERA vs xERA 閘門觸發）

**Martinez 2026 vs 2025：**
- xERA 4.66 ↑ vs 4.04（+0.62，實力稍退）
- Whiff 7.0% ↓ vs 8.4%
- HardHit 25.2 ↑ vs 23.6
- 球速基本持平（86.6 vs 86.9）
- 球種大幅改變：CH 28.6↑vs 19.8、SI 27.5↑vs 17.2（大量增 CH/SI）
- **結論**：ERA 2.45 源於序列幸運（BABIP/LOB%），xERA 4.66 為真實水準；後續回歸風險極高。

**Williamson 2026 vs 2025：**
- 2025 僅 14.33 IP 4 場（prior_year 子欄位），實質無年度比較基準
- 2025 xFIP 4.10 → 2026 xFIP 5.96（明顯退步）
- xERA 6.78 搭配 K-BB% -1.1 → 系統性崩壞型 peripherals

### 投手差距判定
- xERA 差距 4.66 vs 6.78 = **2.12 差距**（Martinez 優勢）
- 表面 ERA 2.45 vs 4.35（Martinez 更優，但部分為雜訊）
- 綜合評估：Martinez 稍佔 1-1.5 檔投手優勢（實戰），但並非「王牌碾壓底層」級差距

---

## §打線評級

### Tampa Bay Rays（vs Williamson L）
- Tier：🟢 Weak（avg OPS 0.695，xwOBA 0.297）
- 本季 BABIP .294（正常）/ Last 7 BABIP .282（正常，無回歸閘門）
- 熱度：⚖️ Normal
- 關鍵：Caminero (vs L .797 OPS)、Aranda (vs L .850)、Diaz (.891 整體)
- Williamson 對左打 SSS 14 PA 樣本過小，但 BB% 15.8 的控球給予 TB 機會

### Cincinnati Reds（vs Martinez R）
- Tier：🟡 Average（avg OPS 0.649 / xwOBA 0.320）← OPS 偏低但 xwOBA 明顯較高
- 本季 BABIP .248（明顯偏低）/ Last 7 BABIP .262（接近但未觸發 ≤.260 閘門）
- 熱度：⚖️ Normal（近 10 戰 8-2，但腳本未判 Hot）
- 關鍵：Elly De La Cruz (vs R .807)、Sal Stewart (vs R .922)、多位 R 打者對 Martinez 適配
- CIN 打線實質 > 外顯（BABIP 低迷中），向常態回歸的空間

### 打線相對優勢
- CIN 打線 xwOBA 高 2 個百分點，且 BABIP 有回補空間
- TB 打線 OPS 高但 xwOBA 低（結果優於過程）
- **雙方打線接近**，略偏 CIN

---

## §牛棚傷兵雙向修正值（B9 閘門）

### 牛棚對比
- **TB bullpen ERA 5.70**（災難級）
- **CIN bullpen ERA 2.54**（elite）
- 差距 **3.16 ERA** — 本場最大結構性差距

### IL 影響
- TB 核心 IL：Uceta、Cleavinger、Boyle、Englert（共 4 位投手）← bullpen core 重傷
- CIN IL：Ferguson（核心左投）+ Greene/Lodolo 輪值（對本場 SP 決鬥無影響）

### 雙向修正值
- **O/U 修正：+0.5 run**（TB 後段牛棚大量失分機率高）
- **ML 修正：-5% TB**（晚局劣勢明顯，CIN 帶分進後段牛棚 win%↑）

---

## §條件修正摘要

| 信號 | 觸發 | Run Value 方向 |
|------|------|----------------|
| Martinez ERA-xERA 落差 2.21 | ✅ | Martinez 回歸風險 +0.3 run（TB 失分↑）|
| Williamson xERA 6.78 + K-BB% -1.1 | ✅ | CIN 得分上修 +0.2 run |
| Park Factor 96（Trop pitcher-friendly）| ✅ | 總分 -0.3 run |
| Martinez 35yo velo 86.6 明顯退化 | ✅ | 次回合效率 ↓（隱含） |
| CIN BABIP .248 低迷有回補 | ✅ | CIN 得分 +0.1 run |
| TB bullpen 5.70 + core IL | ✅ | CIN 後段 +0.3 run / TB 後段 -0.15 run |
| 系列賽 CIN 2-0 連勝氣勢 | ⚠️ | 不計入量化（僅定性） |

---

## §修正後預期得分

- **基礎（投打 + 打線）**：CIN 4.6 / TB 4.3
- **加牛棚修正**：CIN 4.9 / TB 4.1
- **加條件修正 + Park**：CIN ~4.7-5.0 / TB ~3.8-4.0
- **總分預期**：~8.6-9.0
- **比分差異預期**：CIN 以 0.8-1.2 分取勝

---

## §整體判斷（基本面，不含盤口）

**方向傾向**：偏 CIN（客隊）
**信心程度**：中等
- 核心理由：CIN 牛棚碾壓（3.16 ERA 差距）+ Williamson xERA 警示未達毀滅級（K-BB% 負值但 CIN 野手防守尚可）+ CIN 形勢連勝

**風險**：
1. Martinez 表面 ERA 2.45 可能讓市場定價過於偏 TB；實際實力偏 xERA 4.66
2. Williamson K-BB% -1.1 → CIN 本身也可能被拖累，若 Williamson 第 1-3 局崩盤，CIN 自身失分高
3. Trop PF 96 中性偏投手 → 總分上限壓縮
4. 連勝 reversion 機率（CIN 4/20, 4/21 已連贏 2 場）

**基本面總分方向**：略偏 OVER（expected 8.6-9.0 vs line 8.30），但幅度不大，為低信心訊號。

---

> 本檔為 Phase 3 基本面快照；具體盤口推薦（ML / O/U / Run Line 星級 + Kelly）由 Phase 4 `predict.py --save` 產出至 `prediction.json`。
