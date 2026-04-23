# Phase 3 綜合分析：LAD @ SF (2026-04-23 ET 15:45)

## 先發投手對決

### Tyler Glasnow (LAD, R, 32y) — 🟠 Strong Ace（本季表現趨近 Elite）
- 本季 4GS 25 IP：ERA 3.24 / FIP **2.54** / xERA **2.60** / K-BB% 24.2 / K% 30.5 / BB% 6.3
- Statcast：avg_velo 89.8（偏低但非結構性下滑）、hard hit 23.3%（極低）、barrel 6.7%
- Platoon：vs L .179/.242/.321（BF 62）、vs R .156/.182/.313（BF 33，樣本小）
- 球種：FF 36.9 / KC 24.4 / SL 16.3 / SI 14.6 / CU 7.6（5 球種）
- YoY：去年 ERA 3.19 / FIP 3.62 / K-BB% 17.3 → **本季控球 + 接觸壓制明顯升級**
- 風險：IP 25 偏低，但 peripherals 一致指向真實水平 > ERA
- ⛔ 不觸發 YoY 補跑（ERA−xERA 差 0.64 < 1.5）

### Logan Webb (SF, R, 29y) — 🟡 Solid Starter（本季 ERA 膨脹，真實水平偏 🟠）
- 本季 5GS 30 IP：ERA 5.10 / **FIP 3.27** / xERA 4.40 / xFIP 3.14 / K-BB% 12.2 / GB% 72.1
- Statcast：avg_velo 89.1、hard hit 32.3%、barrel 6.5%、whiff 8.4%（低）
- Platoon：vs L **.299/.365/.448**（BF 74，弱點）、vs R .212/.281/.269（BF 58）
- 球種：SI 39.0 / CH 23.2 / ST 21.1 / FF 8.8 / FC 7.9（重 GB 配球）
- YoY：去年 ERA 3.22 / FIP 2.48 / K-BB% 20.8（王牌）→ 本季 FIP 3.27 顯示接觸運氣為主
- ⛔ 不觸發 YoY 補跑（ERA−xERA 差 0.70 < 1.5，且本季 ERA > 去年非低 ≥1.0）

**SP 差距：Glasnow > Webb（本季 FIP 差 0.73，Statcast 接觸品質 Glasnow 全面壓制）**

---

## 打線評級與熱度

### LAD vs Webb（R）— 🟠 Strong
- Team：OPS .810 / xwOBA .356 / BABIP .334 / K% 22.8 / BB% 9.7
- Last 7 BABIP .325 → **正常區間，不觸發 B10 回歸閘門**
- Recent heat：⚖️ Normal（整體）但個別打者極燙：
  - **Muncy L7 1.423 OPS（BABIP .412）**
  - **Freeman L7 1.100 OPS（BABIP .455，略偏運氣）**
- Chain：obp_top3 .358 / slg_mid **.587**（強力中段）
- **Platoon 利多**：Webb vs LHH .299/.365/.448 → LAD 左打群（Ohtani / Freeman / Muncy / Tucker）可吃
- **BvP sufficient（PA ≥ 15）**：
  - **Muncy vs Webb：42 PA / .257/.381/.571 / 3 HR / 8K / 6BB**（壓制）
  - **T.Hernández vs Webb：25 PA / .318/.400/.545 / 1 HR**（強）
- IL 缺陣：Mookie Betts（10-day）+ Tommy Edman（10-day）+ E.Hernández（60-day）→ 打線略弱，但 Ohtani/Freeman/Tucker/Muncy 核心齊，影響可控

### SF vs Glasnow（R）— 🟢 Weak
- Team：OPS .652 / xwOBA .292 / BABIP .312 / K% 21.0 / BB% 5.0
- Last 7 BABIP .321 → **正常，不觸發 B10**
- Chain：obp_top3 **.300**（偏弱）/ slg_mid .362
- **Platoon 劣勢**：右打為主（Adames / Chapman / Arraez / Ramos / Schmitt / Encarnacion），面對 Glasnow vs R .156/.182/.313 = 極端壓制
- 左打（Devers DH / JH Lee / Bailey）vs Glasnow vs L .179/.242/.321 仍被壓制
- L7 燙手：Ramos 1.253 OPS（BABIP .438，略偏運氣）、Chapman .766 OPS（BABIP .435）、JH Lee .852 OPS
- BvP 全部 PA<15 不可引用

**打線差距：LAD >> SF（OPS 差 .158，xwOBA 差 .064；Webb 弱點左打正是 LAD 強項）**

---

## 牛棚雙向修正值（B9 觸發）

### LAD 牛棚（客）— 🔴🔴 極高傷兵影響
- 整體 ERA **4.27**（季）
- **核心 IL 4+ 人**：Edwin Díaz（Closer, 15-day）+ Evan Phillips（primary setup, 60-day）+ Brusdar Graterol（15-day）+ Brock Stewart（15-day）+ Ben Casparius + Jake Cousins
- 剩餘核心：Tanner Scott / Blake Treinen / Alex Vesia（左打專殺）→ 尚有深度但薄
- 依 matchup-factors.md 表：**3+ 名核心 → 對手 +1.0 run, 信號 +2, ML -5%**

### SF 牛棚（主）— 🟠 中高傷兵影響
- 整體 ERA **3.28**（季，優於聯盟平均）
- IL：Sam Hentges（15-day）+ Randy Rodríguez（60-day）+ Hayden Birdsong（60-day）+ Joel Peguero（15-day）
- 核心仍在：Ryan Walker（closer）+ Tyler Rogers role / Robbie Ray swing / Matt Gage LOOGY
- 保守估計 1-2 名核心 IL → **對手 +0.3~0.5 run, ML -2~3%**

### 淨效應（寫入 predict.py signal-adjustments）
- `bullpen_il_away`（LAD 牛棚 IL）：對手（SF）+1.0 run；LAD ML -5%
- `bullpen_il_home`（SF 牛棚 IL）：對手（LAD）+0.3 run；SF ML -2%
- **淨差 LAD ML -3%；OU 淨 +0.7 run（LAD IL 影響較大）**

---

## 條件修正摘要

| 信號 | 觸發 | Run Value 方向 |
|------|------|---------------|
| Oracle Park Factor 96 | ✅ | 總分 ×0.96 壓低 |
| Webb FIP 強於 ERA（落差 1.83） | ✅ | SF 失分下修（LAD 得分預期 -0.3） |
| LAD 牛棚核心 3+ IL | ✅ | SF +1.0 run；LAD ML -5% |
| SF 牛棚 1-2 核心 IL | ✅ | LAD +0.3 run；SF ML -2% |
| Muncy vs Webb BvP 42 PA .257/.571 | ✅ | LAD +0.2 run |
| T.Hernández vs Webb 25 PA .318/.545 | ✅ | LAD +0.1 run |
| Webb vs LHH .299 vs LAD 多左打 | ✅ | LAD +0.2 run |
| Glasnow vs R .156 vs SF 右打群 | ✅ | SF −0.3 run |
| LAD 主力 IL（Betts + Edman） | ✅（次要） | LAD -0.2 run |
| SF 連勝 LAD 2 場（投手戰 3-1, 3-0） | ✅（敘事） | 警訊但樣本小 |
| Glasnow IP<30 不確定性 | ✅（弱） | 非結構性 |

---

## 修正後預期得分（基本面估算）

- **LAD 基線**：本季 RS/G 5.58 × PF 0.96 = 5.36
  - vs Webb（FIP 3.27，🟡-🟠）：5.36 × (3.27 / 4.30 lg avg) ≈ 4.08
  - BvP + Platoon + Hot L/H 打者：+0.5 → **4.6**
  - 加 SF 牛棚 IL +0.3：→ **4.9**
  - Betts/Edman IL -0.2：→ **4.7**
  
- **SF 基線**：本季 RS/G 3.38 × PF 0.96 = 3.25
  - vs Glasnow（xERA 2.60）：3.25 × (2.60 / 4.30) ≈ 1.96
  - 熱打者微補：+0.3 → **2.3**
  - 加 LAD 牛棚 IL +1.0：→ **3.3**

- **總分預估**：4.7 + 3.3 = **8.0**
- **分差預估**：LAD **+1.4**

---

## 整體判斷

- **方向性**：**基本面偏 LAD**
  - SP 差距 2 檔（Glasnow 🟠 Strong Ace vs Webb 🟡 Solid，xERA 差 1.80）
  - 打線差距：LAD 🟠 Strong（OPS .810）vs SF 🟢 Weak（OPS .652）
  - Webb vs LHH 是 LAD 左打群的直接打擊點
- **反向風險（重要）**：
  1. **LAD 牛棚核心 3+ IL**是最大變數；若第 7-8 局接近，SF 主場有逆轉空間
  2. SF 剛在家橫掃 LAD 兩戰（3-1, 3-0），樣本雖小但 Glasnow 非無敵（Oracle 球場壓球特性+SF GB% 強攻 Glasnow SI 有空間）
  3. Webb FIP 強於 ERA，若今日回歸可能把 LAD 壓制在 3-4 分
  4. Glasnow 本季 IP 僅 25，樣本不確定
- **信心程度**：**中等**（60-65% 區間）— 基本面強傾向 LAD 但牛棚 IL 與近期 H2H 拉回

- **得分預期**：LAD 4.7 / SF 3.3 / 總分 8.0（位於 OU 7.20 線之上，但分差 1.4 < RL 1.5，邊界）
