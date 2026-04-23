# Phase 3 Summary — STL @ MIA, 2026-04-20

## 先發投手對決

### Max Meyer (MIA) — 🟡 Solid Starter
- Season: 4GS / 19.2 IP / ERA 4.12 / FIP 3.76 / xFIP 3.69 / xERA 5.24
- K% 22.7 / BB% 10.2 (walks high) / K-BB% 12.5
- GB% 54.8 (heavy groundball), Barrel% 10.2, Hard-Hit 29.5
- Avg velo 91.0, Whiff% 13.6, CSW% 29.2
- RHP, age 27, prior year 4.73 ERA / 3.01 xFIP (skilled but results-bumpy)
- |ERA-xERA| 1.12 → not triggering YoY gate
- Profile: above-avg K, heavy GB, walk-prone; solid mid-rotation floor

### Michael McGreevy (STL) — label 🟠 Strong Ace，YoY 驗證後 **實質 🟢 Back-end**
- Season: 4GS / 21.2 IP / ERA **2.49** (surface) / FIP 4.21 / xFIP 4.03
- xERA **7.42** / xwOBA **.415** / xBA **.309** ← 全部警訊
- K% 14.6 (very low), BB% 3.7 (good), K-BB% 10.9 (below avg)
- Barrel% **14.9** (up from 8.8 YoY), Hard-Hit 25.2, Whiff **5.8%** (down from 7.9)
- Avg velo **86.3** (DOWN 2.0 mph from 88.3 YoY)
- RHP, age 25, prior year 4.42 ERA / 4.15 FIP

**YoY Statcast 驗證結論（3/5 指標一致退化）**：
| 指標 | 2025 → 2026 | 方向 |
|------|-------------|------|
| avg_velo | 88.3 → 86.3 | ⬇ 2.0 mph (實質衰退) |
| barrel% | 8.8 → 14.9 | ⬆ +6.1 (接觸品質惡化) |
| whiff% | 7.9 → 5.8 | ⬇ 揮空率下降 |
| xERA | 4.67 → 7.42 | ⬆ +2.75 |
| xwOBA | .333 → .415 | ⬆ +.082 |

2.49 ERA 是 21.2 IP 的運氣 + 小樣本 + BABIP/HR 奇蹟。**真實水平 ≈ 5.00-5.50 ERA**。降級為 🟢 Back-end（xERA 視角甚至 ⚪ Below Avg）。

### 投手對決 edge
**Meyer > McGreevy ~1 檔**。兩人同為 RHP，無 platoon 效應。

---

## 打線

### MIA（🟡 Average） vs RHP McGreevy
- Team: avg_ops .726 / xwoba .312 / BABIP .316 / K% 20.6 / BB% 8.8
- Recent heat ⚖️ Normal
- **Top-half vs RHP 強**: Edwards .914, Lopez .918, Hicks 1.064, Norby .865
- **Bottom-half vs RHP 弱**: Marsee .403, Ramírez .473, Hernández .379
- BABIP flags: Edwards .386 / Lopez .381（熱度但非極端，輕微回歸下修）
- BvP 樣本 PA=3 全部 <15 → 不引用
- 缺陣影響：Conine、Morel、Ruiz 3 OF 傷兵但當日先發預計不含；Kyle Stowers 剛 activate 但未在 projected lineup（潛在 swing bat）

### STL（🟡 Average） vs RHP Meyer
- Team: avg_ops .695 / xwoba **.324** / BABIP .277 / K% 21.8 / BB% 10.2
- xwOBA > OPS 反差 → 輕微運氣差，預期回歸上修
- Recent heat ⚖️ Normal（但球隊 5W streak）
- **Mid-order 火力**: Walker OPS **1.013** (.341 ISO, 23.6 barrel%) + Burleson .819 (vs RHP .959)
- Top-3 OBP .376（含 Wetherholt 14.4 BB%, Herrera 18.4 BB%）
- **Bottom-half vs RHP 弱**: Gorman .667, Winn .570, Scott II .382, Church .518
- Nootbaar 60-day IL（OF 主力缺陣）
- BvP 樣本全 <15 → 不引用
- chain slg_mid **.495** (vs MIA .378) → 中棒長打延續優勢

### 打線 edge
**STL 略優**（xwOBA 更高 + Walker/Burleson vs RHP 雙重威脅 + 回歸紅利），但僅 ~0.1-0.2 xwOBA 差距，非結構性。

---

## 牛棚（雙向閘門）

| 指標 | MIA | STL |
|------|-----|-----|
| Bullpen ERA | **3.39** 🟢 | **5.40** 🔴 |
| 核心 IL | Mazur 60d, Henriquez 60d（長期，已計入季內 ERA） | Dobbins 15d, Pushard 15d（短期） |

**STL 牛棚明顯較弱，差距 ~2.0 ERA**。STL 無新增短期核心 IL 觸發累計條款（長期 IL 已反映），但牛棚本季表現就是 5.40。MIA 牛棚深度優。

### 雙向修正
- O/U：STL 牛棚放火 → 對手（MIA）得分 +0.3~+0.5 run
- ML：STL 隊 ML -2%~-3%（後段局數劣勢）

---

## 條件修正

| 信號 | 修正值 | 適用方 |
|------|-------|-------|
| Park Factor 98 | (98-100)×0.05 = -0.1 | 總分 -0.1 |
| loanDepot park 天頂通常關閉（溫度/風無效） | 0 | - |
| STL 牛棚遠差於 MIA | MIA 得分 +0.4 | MIA 得分端 |
| MIA 牛棚遠優於 STL | STL 得分 -0.3 | STL 得分端 |
| 雙方先發皆 🟡 Solid+？ | No（McGreevy 實質 🟢） | 不適用 -0.5 |
| 雙方打線 Hot 場均 ≥ 5？ | 否（皆 Normal） | 不適用 |
| 主審 Over/Under% 57%+ | 無資料 | skip |
| 雙方開季樣本（MIA 22G / STL 21G）| - | 觸發 INSUFFICIENT_SAMPLE / early-season tag |

---

## 近期狀態

- **STL**: 近 10 場 7-3、W5 streak、近 30 場 13-8 Pyth 偏正。動能強但 RS=RA 中性。
- **MIA**: 近 10 場 3-7 但 W1 streak，近 30 場 10-12 回暖中。主場優勢。
- H2H 當場系列首戰，無前場資料。

---

## 期望得分估算（formula 預估，供 Phase 4 驗算）

基礎期望：
- E[STL runs vs Meyer] = 4.4 × (.324/.315) × (~4.2/4.3) × 0.98 ≈ **4.3**
  （Meyer FIP 3.76 / xFIP 3.69，混合混合早季 regression ~4.2）
- E[MIA runs vs McGreevy] = 4.4 × (.312/.315) × (~5.1/4.3) × 0.98 ≈ **5.1**
  （McGreevy 真實 ERA band 5.00-5.50 取下緣）

信號修正：
- MIA 得分：5.1 + 0.4 (STL 牛棚) ≈ **5.5**
- STL 得分：4.3 - 0.3 (MIA 牛棚) ≈ **4.0**
- Total：**~9.5**

O/U line 8.4-8.5，預期差距 ~+1.0 → **接近 PASS 門檻**（1.5 閾值）。
ML 方向：MIA 5.5 vs STL 4.0 → 預期 MIA 勝率 ~58-60%。

## 整體判斷

- **方向**：基本面偏 **MIA**。投手差約 1 檔（Meyer 實質優於 McGreevy），牛棚差 ~2 ERA 明顯優於 STL。
- **信心**：MEDIUM-LOW。雙方開季樣本都 <30 場，D1.5 INSUFFICIENT_SAMPLE 觸發，XGBoost 可靠度受限；需等 predict.py cross_validation 結果。
- **值得注意的風險**：
  1. McGreevy 4 場 ERA 2.49 看似無懈可擊，但 xERA 7.42 + velo 退化告訴你這是運氣
  2. STL 5W streak + Walker 熱（OPS 1.013）非結構但需警惕
  3. Stowers 若進 MIA 打線可能改變 projected 估算
  4. 早季樣本不足 → 預測可靠度標註 LOW
  5. MIA home-spread 形式 book vs Pinnacle 可能不一致，需 Phase 4 交叉驗證
