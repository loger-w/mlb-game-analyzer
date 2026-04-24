# Phase 3 分析摘要：OAK @ TEX (2026-04-24)

## 基本資料
- 日期：2026-04-24 20:05 ET（台灣時間 4/25 08:05）
- 場地：Globe Life Field (PF = 103.0，輕微打者友善)
- 先發：Luis Severino (OAK, RHP) @ Nathan Eovaldi (TEX, RHP)
- 狀態：Preview

---

## 先發投手分級

### Nathan Eovaldi (TEX, RHP, age 36) — ⚪ Below Average
- 2026: 5 GS, 26.67 IP, ERA 5.06, FIP 4.26, **xERA 4.09**
- K% 23.8, BB% 6.6, HR/9 1.69 (偏高)
- Statcast: avg_velo **88.0 mph**（極低 RHP），whiff% 13.9，hard_hit% 30.9
- Platoon: vs LHB **.319/.372/.528**（被爆打），vs RHB .250/.295/.350
- 2025 prior year：ERA 1.73 / FIP 2.60（精英級）
- Age 36 age_assessment: 📉📉 明顯退化 — 球速下滑已反映在本季數據
- **ERA-xERA gap 閘門**：|5.06−4.09|=0.97 < 1.5 ✓ 無須 YoY 補跑
- 判定：xERA 4.09 代表真實水平 ≈ 🟢 Back-end，ERA 5.06 略虛高但本質仍不強

### Luis Severino (OAK, RHP, age 32) — ⚪ Below Average
- 2026: 5 GS, 24.67 IP, ERA 6.20, FIP 5.45, **xERA 5.88**
- K% 24.3, **BB% 18.0**（極差），HR/9 1.46
- Statcast: avg_velo 92.1，whiff% **9.0**（低），hard_hit% 26.8
- Platoon: vs LHB .212/.388/.365（控球差放保送），vs RHB **.306/.419/.556**（被猛打）
- 2025 prior year: ERA 4.54 / FIP 3.78
- Age 32 age_assessment: 📉 初期退化
- **ERA-xERA gap 閘門**：|6.20−5.88|=0.32 < 1.5 ✓ 無須 YoY 補跑
- 判定：xERA 5.88 驗證本季真實很差（非運氣），⚪ Below Average 名實相符

### 投手對決
- 兩者同為 ⚪ Below Average tier，但 **Eovaldi xERA 4.09 明顯優於 Severino xERA 5.88**（差約 1.8 run）
- Eovaldi 最大風險：vs LHB 被 .528 SLG 打爆，且 OAK 打線有 6+ 名左打
- Severino 最大風險：BB% 18% + vs RHB .556 SLG，TEX 右打群（Jung / Burger / Smith）有利可圖
- 未觸發「雙方皆 🟠 Strong Ace+」或「Solid+」下修信號

---

## 打線評級與熱度

### TEX — 🟡 Average
- avg OPS 0.720, xwOBA **0.313**, BABIP 0.296, K% 24.1, BB% 10.2
- Chain: OBP_top3 0.322 / SLG_mid 0.456
- Recent heat: ⚖️ Normal，last7_BABIP 0.334（在 [.260, .370] 內，BABIP 閘門未觸發）
- 核心：Nimmo (OPS .893 L), Josh Jung (.883 R), Seager (.765 L)
- **缺 Wyatt Langford（OF, 10-Day IL）**— 主力中心打者缺陣影響中等

### OAK — 🟠 Strong
- avg OPS 0.720, xwOBA **0.352**, BABIP 0.301, K% 20.9, BB% 9.8
- Chain: OBP_top3 0.367 / SLG_mid 0.407
- Recent heat: ⚖️ Normal，last7_BABIP 0.292（BABIP 閘門未觸發）
- 核心：Nick Kurtz (OPS .849 + xwOBA .426 elite), Langeliers (.962), Cortes (.916)
- **缺 Brent Rooker（OF, 10-Day IL）**— 主砲缺陣，但打線 tier 仍 Strong
- BvP: **Langeliers 對 Eovaldi 15 PA**（.333/.333/.667, 1 HR, 5K）達閾值 → 中性混合訊號（強力但高 K）

### 打線對比
- **OAK xwOBA 0.352 比 TEX 0.313 高 39 pts，打線明顯偏 OAK**
- Handedness fit 對 OAK 特別有利：OAK 左打群對 Eovaldi vs LHB .528 SLG 形成威脅
- TEX 右打群對 Severino vs RHB .556 SLG 也有威脅，但 Severino BB% 18 同時送 TEX 左打上壘

---

## 牛棚雙向修正值（B9 閘門）

### TEX 牛棚：季 ERA 2.98（⭐ 強）但核心傷兵累計
- **IL 核心 2 名**：
  - Chris Martin（High-leverage RHP，15-Day IL）
  - Robert Garcia（Late-inning LHP，15-Day IL）
- 其他 IL：Carter Baumler (15-Day), Luis Curvelo (15-Day), Cody Bradford & Jordan Montgomery (60-Day, SP)

### OAK 牛棚：季 ERA 4.36（一般）
- IL 主要是 Gunnar Hoglund (SP, 60-Day) — 不影響牛棚
- 無核心後援缺陣

### 雙向修正（matchup-factors.md「2 名核心 IL」條件）
- **O/U 修正**：對手（OAK）+0.6 run（取 +0.5~0.7 中值），信號 +1
- **ML 修正**：TEX -3.5%（取 -3~4% 中值）
- 信號 tag 建議：`bullpen_il_home` +0.6

---

## 條件修正摘要

| 信號 | 觸發 | Run Value |
|------|------|-----------|
| Park Factor (Globe Life 103) | ✓ | (103-100)×0.05 = +0.15/側（公式內已含） |
| Eovaldi 明顯退化 (age 36) | 本季已反映 | 不額外修正 |
| Severino 初期退化 (age 32) | 本季已反映 | 不額外修正 |
| TEX 牛棚 2 核心 IL | ✓ | OAK +0.6（加到 OAK 得分） |
| Platoon 全打線同手 | ✗（混和） | 0 |
| 雙方 Hot/Cold | ✗ | 0 |
| 雙方先發 🟠+ 或 🟡+ | ✗（皆⚪） | 0 |
| Doubleheader 第二場 | ✗ | 0 |
| 投手休息異常 | ✗（預估 5 天） | 0 |

---

## 修正後預期得分（手算預估，最終以 predict.py formula 為準）

基礎（聯盟平均得分 4.50，xwOBA 0.320，ERA 4.30）：
- E[R_TEX] = 4.50 × (0.313/0.320) × (5.88/4.30) × (103/100) ≈ 6.20
- E[R_OAK] = 4.50 × (0.352/0.320) × (4.09/4.30) × (103/100) ≈ 4.85
- 小樣本 xERA 偏噪聲；用 2025 prior 血緣回歸後實際總分可能 ~8.5-9

Signal 加成：
- OAK +0.6（TEX 牛棚 IL）
- 修正後 TEX ≈ 6.20 / OAK ≈ 5.45（小樣本上界）
- 血緣回歸版 TEX ≈ 4.2 / OAK ≈ 4.5（下界）

**推估區間**：總分 8.5–11.5 run，中值約 9.5。實際以 predict.py formula 輸出為準。

---

## H2H 與趨勢
- 最近 H2H (4/14-4/16 在 OAK 主場 4 場) OAK 3-1 微勝，雙方總分 TEX 23 / OAK 15（Eovaldi 並未在那個系列先發）
- TEX 近 10 場 5-5，近 3 場 2-1 包括昨天 6-1 勝 PIT → **趨勢 ↑**
- OAK 近 10 場 5-5，但 RS/RA 4.3/5.6（run_diff -13）→ Pythagorean 不利，**趨勢混亂偏下**
- OAK 休息 1 天（最後一場 4/22 在 SEA）
- TEX 打完 3 場主場連戰對 PIT，今天是第四場

---

## 整體判斷（方向性，不含星級）

### 方向傾向
- **ML**：微幅偏 TEX（主場、較佳 SP xERA、較佳牛棚即便 2 核缺陣）。OAK 靠打線抬升平衡，但 Pythagorean 拖累。預估勝率差約 4-8% 偏 TEX。
- **O/U**：基本面輕偏 **Over**（小樣本 xERA 驅動，兩側均為 ⚪ 投手，PF 103 輕微推高，TEX 牛棚 IL 對 OAK 得分有利）。但若實際 xERA 回歸至 2025 level，可能降至接近 line（8.3–8.7 區間）。**差距可能 < 1.5 run 預估 PASS**。
- **讓分（TEX -0.5/-1.0）**：TEX 為讓分方。預估 margin 0.3–1.0 run。**方向輕偏 LEAN_RECEIVING（OAK +0.5/+1.0）**，但信心不高。

### 值得注意的風險
1. **小樣本陷阱**：兩位投手 IP 均 < 30，xERA 仍含噪聲。若回歸到 2025 水準（Eovaldi 1.73 vs Severino 4.54），差距會拉大，TEX 獲利可能增加。
2. **Eovaldi 左打弱點 + OAK 左打密集**：vs LHB .528 SLG 是結構性隱憂，若 Kurtz / Soderstrom / Muncy 等左打集中發動，可能翻盤。
3. **TEX 牛棚 IL 的實際深度影響**：季 ERA 仍 2.98 代表替補撐住，影響可能低於帳面 -3.5%。
4. **BvP 小樣本**：除 Langeliers 外皆不足 15 PA，H2H 不具統計意義。

### Final source of truth
盤口推薦與星級：以 Phase 4 `predict.py --save` 輸出的 `prediction.json` 為準。
