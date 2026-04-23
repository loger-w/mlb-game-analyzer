# Phase 3 Summary — BAL @ CLE, 2026-04-18 18:10 ET

**Venue**: Progressive Field (Cleveland) | Park factor 98.0（略投手友善）
**Status**: Preview（尚未開打）

---

## 1. 先發投手對決

### BAL — Dean Kremer (RHP, 30y, 🟡 Solid Starter, 📉 初期退化 flag)

| 指標 | 2026 (GS=1, IP=5) | 2025 (GS=29, IP=171.2) | 差異 |
|---|---|---|---|
| ERA | 3.60 | 4.19 | −0.59 |
| **xERA** | **6.31** | 3.82 | **+2.49** ⚠️ |
| xwOBA | .389 | .304 | +.085 ⚠️ |
| FIP | 7.30 | 3.90 | +3.40（1 場雜訊）|
| xFIP | 0.02 | 3.75 | 1 場雜訊 |
| Barrel% | 33.3 | 8.5 | +24.8 ⚠️（1 場雜訊）|
| HardHit% | 23.1 | 21.1 | +2.0 |
| avgVelo | **86.2** | 87.6 | **−1.4 mph** ⚠️ |
| Whiff% | 15.0 | 10.6 | +4.4 |
| Pitch mix | FS 36.2% / FF 21.2% / SI 13.8% / CU 11.2% / FC 10.0% | FF 26.6 / FS 21.1 / FC 20.2 / SI 19.3 / CU 12.8 | 大改，FS 變主武器 |

**YoY Statcast 對比閘門**：|ERA−xERA|=2.71 ≥ 1.5 已觸發，已補跑 2025 對比。
**關鍵讀取**：
- ERA 3.60 僅 1 場樣本，**xERA 6.31 是更可靠的真實表現估計**
- Velo 掉 1.4 mph + Pitch mix 從均衡型改為 splitter 主導 → 可能是**退化後的重新定位**
- 2025 baseline 已是平均水準（xERA 3.82），若退化為真會更糟
- 1 場 9K 0BB 是樣本雜訊，不可外推

### CLE — Gavin Williams (RHP, 26y, 🟠 Strong Ace, ⚡ 巔峰期)

| 指標 | 2026 (GS=4, IP=22.2) | 2025 (GS=31, IP=167.2) | 差異 |
|---|---|---|---|
| ERA | 2.38 | 3.06 | −0.68 |
| **xERA** | 4.07 | 4.30 | −0.23（穩定）|
| xwOBA | .322 | .321 | ≈ |
| FIP | 4.38 | 4.30 | ≈ |
| xFIP | 3.52 | 3.71 | −0.19 |
| **BB%** | **18.0** | 11.8 | **+6.2** ⚠️ |
| K% | 32.6 | 24.6 | +8.0 ✓ |
| avgVelo | 91.3 | 90.6 | +0.7 |
| Whiff% | 12.5 | 12.0 | ≈ |

**YoY Statcast 對比閘門**：|ERA−xERA|=1.69 ≥ 1.5 已觸發，已補跑 2025 對比。
**關鍵讀取**：
- ERA 2.38 與 xERA 4.07 落差 1.69 → **表面 ERA 有運氣成分，真實約 4.0**
- BB% 從 11.8 飆到 18.0 + 近 3 場 BB 數 6/3/5 → **控球警訊**（可能影響 O/U）
- Velo、Whiff% 維持健康
- K% 提升（24.6→32.6）可抵消控球代價

### 先發對決 Net Read
兩位 xERA 都比帳面 ERA 差，但**差距不同**：Williams xERA 4.07（可接受）vs Kremer xERA 6.31（警戒）。Williams 控球問題是 O/U 提升因子，但 xERA 仍優於 Kremer 2 分以上。**Williams 邊際優勢**。

---

## 2. 打線與 BvP

### BAL 打線 — 🟡 Average tier
- OPS .660 / xwOBA .314 / BABIP .278 / K% 25.4 / BB% 10.3
- over_under_lean = −1 / recent_heat = ⚖️ Normal
- **傷兵重創**：Rutschman (C)、J. Holliday (2B)、Westburg (3B)、Mountcastle (1B)、O'Neill (OF)、Kjerstad (OF) 全數 IL — 主力陣容掏空
- 取代打者：Basallo (C) xwOBA .289、Jackson (2B) OPS .923（62PA）、Mayo、Beavers、Alexander、Cowser — 多為新秀/替補深度
- 正負向 BABIP regression 候選：Ward .371↑、Jackson .350↑、Basallo .143↓

### CLE 打線 — 🟠 Strong tier
- OPS .706 / xwOBA .343 / BABIP .269 / K% **19.4**（明顯低）/ BB% 10.4
- over_under_lean = +1 / recent_heat = 🔥 Hot
- **BABIP Hot 確認閘門**：team BABIP .269 低於聯盟 ~.300 → Hot 標籤**非 BABIP 膨脹造成**，是真實火熱 ✓
- 傷兵輕微：Walters (BP)、Arias (utility SS)
- 核心：Ramírez、Kwan、DeLauter、A. Martínez、Schneemann 均健在
- 隱藏正向 regression：**Ramírez BABIP .210 / xwOBA .401**、**DeLauter BABIP .220 / xwOBA .387** → 運氣偏低，實質火力高於表面

### BvP — **INSUFFICIENT_SAMPLE (D1.5 觸發)**
| 對決 | 最高個人 PA | 總 PA | 評估 |
|---|---|---|---|
| BAL vs Williams | Henderson 6PA | 22PA | 所有個人 PA < 15 → 雜訊 |
| CLE vs Kremer | Kwan 11PA | 34PA | 所有個人 PA < 15 → 雜訊 |

**BvP 不進入訊號層**（即使 Henderson .400/6PA、Hoskins .500/6PA 看似極端，樣本不足以外推）。

### 打線 Net Read
CLE 打線顯著優於 BAL：**OPS +.046、xwOBA +.029、K% −6.0**。BAL 失去 6 位打線核心且替補新秀化，對抗 Williams 不利。CLE 對抗退化中的 Kremer（xERA 6.31、velo 下降）優勢極大。

---

## 3. 牛棚（雙向閘門）

| 項目 | BAL (away) | CLE (home) |
|---|---|---|
| 帳面 ERA | 3.45 | 5.55 |
| IL | **Bautista (closer, 60-day)、Kittredge (15d)、Akin (15d)、Eflin (60d)、Enns (15d)、Selby (60d)、Hiraldo (60d)** | Walters (15d) |
| 實質深度 | **嚴重縮水** — 終結者 + 三位主要 setup 全缺 | 完整但 ERA 5.55 |

**雙向閘門評估**：
- BAL 牛棚帳面 3.45 **不可信** — 主要臂群 IL，實際可用牛棚遠差於數字
- Kremer 預期僅 5 IP 左右（2026 樣本 5IP、2025 每場 5.9IP），BAL 需要 4 IP 牛棚
- CLE 牛棚 5.55 帳面差，但 Williams 控球雖抖能投 5-6 IP（近 3 場 5/7/5.2 IP），需要 3-4 IP 牛棚
- **O/U 提升因子**：兩隊牛棚都有漏洞 → 總分 ↑
- **ML 影響**：BAL 牛棚深度問題抵消一部分「牛棚帳面差距」優勢 → CLE 在 6 局後的優勢擴大

---

## 4. 環境與條件修正

- **球場**：Progressive Field park factor 98（微投手友善，接近中性）
- **天氣**：未查（4 月克里夫蘭，預期風 / 溫度中性；若需可追加 WebSearch）
- **主審**：未查
- **近 30 天得分**：
  - CLE: 3.77 RS / 4.05 RA（22 場）— 打線比整體數據看起來冷
  - BAL: 4.25 RS / 4.30 RA（20 場）
  - 近 10 場 RS: CLE 4.9 / BAL 4.9；RA: CLE 4.9 / BAL 4.2
- **連勝走勢**：CLE 剛輸（streak −1），4/17 被 BAL 打敗 6-4；4/16 CLE 贏 4-2 → **系列賽 1-1 平手**（兩隊剛對打 2 場）
- **背靠背**：系列賽第 3 場

---

## 5. 綜合因子整理（供 Phase 4）

| 因子 | 方向 | 強度 | 說明 |
|---|---|---|---|
| 先發對決 | CLE | 中 | Williams xERA 4.07 vs Kremer xERA 6.31（差 2+）；但 Williams 控球警訊 |
| 打線對決 | CLE | 中強 | CLE tier Strong + Hot vs BAL tier Average + 6 位主力 IL |
| 牛棚（兩側修正）| CLE 小優 | 小 | BAL 帳面 3.45 vs CLE 5.55 表面差 2 分，實際 BAL IL 過多差距縮小 |
| 隱藏回歸 | CLE | 小 | Ramírez/DeLauter 低 BABIP 有上修空間 |
| 傷兵差 | CLE | 中 | BAL 打線 +牛棚 IL 極重、CLE 輕微 |
| 球場 | 中性偏 Under | 小 | PF 98 |
| 主場優勢 | CLE | 小 | ~.030 勝率 |
| BvP | — | — | INSUFFICIENT_SAMPLE |
| Hot/Cold BABIP check | CLE Hot 真實 | — | team BABIP .269 非膨脹 |

**淨向量**：**CLE 明顯偏強側**，多因子同向。
**總分 lean**：略偏 Over（兩側牛棚漏洞 + Williams 控球抖 + BAL 打者新秀化容易高 K 低接觸但出手多）；但 Kremer 可能被痛擊的反差存在 → 中性略 Over。

> 此處僅列基本面因子，**不含盤口推薦 / 星級 / 信心值** — 由 Phase 4 `predict.py` 統一輸出。
