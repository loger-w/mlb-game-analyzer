# Phase 3 Summary — NYM @ CHC, 2026-04-18 14:20 ET (Wrigley Field)

## 3.1 先發投手對決

### Freddy Peralta (NYM, R, 29 ⚡巔峰期) — 🟡 Solid Starter（偏 🟠）
- 本季 4 GS, 21 IP：ERA 3.86 / xERA 3.69 / FIP 3.72 / **xFIP 2.91** / K-BB% 19.1 / WHIP 1.14
- Statcast：avg velo 88.4 / max 96.2 / Hard-Hit% **21.5**（極低）/ Barrel% **5.5**（優）/ CSW 29.8
- GB% 56.4%（遠高於往年 41.7% — 新配球？FF 49.3% 主球種，CH 23.1% 升級，需觀察）
- Prior year (2025)：ERA 2.70 / K-BB% 19.1 / FIP 3.46 — **skill 指標完全延續**
- Platoon（本季小樣本）：vs_L .212/.268/.404 (57 BF) / vs_R .185/.313/.222 (32 BF)
- **ERA-xERA 閘門**：|ERA-xERA| = 0.17；本季 ERA 高於 prior → 閘門未觸發，無需 YoY 補跑

### Jameson Taillon (CHC, R, 34 📉📉明顯退化) — 🟢 Back-end Starter
- 本季 3 GS, 16.67 IP：ERA 4.86 / xERA 4.29 / FIP **6.04** / xFIP 3.70 / K-BB% 15.4 / WHIP 1.26
- Statcast：avg velo **86.0**（偏低）/ max 93.3 / Hard-Hit 23.8 / Barrel **12.8**（高）/ HR/9 **2.7**（極高）
- 球種：FF 27.9% / FC 25.6% / CH 14.7% / ST 12.4% / CU 11.8%
- Prior year (2025)：ERA 3.68 / FIP 4.62 / HR/9 1.67 — 本季 HR 壓制顯著變差
- Platoon（本季小樣本）：vs_L .250/.314/**.656** (35 BF, 小樣本 SLG 異常) / vs_R .219/.306/.375 (36 BF)
- **ERA-xERA 閘門**：|0.57|<1.5；本季 ERA 高於 prior → 閘門未觸發
- FIP 6.04 vs xFIP 3.70 落差大 → HR/FB 運氣差是主因，但 xERA 4.29 仍顯示真實水平平庸

### 投手對決結論
**Peralta 勝 Taillon 約 2 檔**（Solid Peak vs Back-end Decline）。Peralta xFIP 2.91 vs Taillon xERA 4.29，差距約 1.3-1.4 run/9。NYM 這側投手防守是明顯優勢。

---

## 3.2 打線評級

### CHC 對 R-Peralta — 🟡 Average, 🔥 Hot
- avg OPS **.767** / xwOBA **.328** / BABIP .301 / K% 20.9 / BB% 10.8
- over_under_lean 腳本值 0 / recent_heat Hot
- **BABIP 回歸檢查**：團隊 BABIP .301 合理，無運氣 bias
- 個別警示：Carson Kelly BABIP .375、Ballesteros BABIP .414 → OPS 有運氣灌水 ~0.05-0.08
- 串聯：OBP 前三棒 .337、SLG 中段 .453 — 串聯能力中上
- 對 Peralta 劣勢：Peralta FF 49.3% + CH 23.1% 的組合、GB 傾向高 → Wrigley 打者利基不易發揮

### NYM 對 R-Taillon — 🟡 Average, 🥶 Cold
- avg OPS **.626** / xwOBA .313 / BABIP **.266** / K% 21.5 / BB% 7.7
- over_under_lean 1 / recent_heat Cold
- **BABIP 回歸檢查**：團隊 BABIP .266 < .280 → **運氣成分偏差**，回歸預期略上升；Cold run value 不全額套用
- 個別：Alvarez OPS .959 / xwOBA .446 — 主戰力之一；Lindor OPS .61 低迷
- 串聯：OBP 前三棒 .278、SLG 中段 .317 — 串聯能力極弱
- 對 Taillon 優勢：HR/9 2.7 + Barrel 12.8% → 長打機會，但 NYM 打線「能不能接住」是問題

### 打線結論
CHC 打線真實水平略高於 NYM ~100 點 OPS。BABIP 回歸後差距收斂至 ~70-80 點，但仍 CHC 勝。

---

## 3.3 牛棚分析

### CHC 牛棚（自動取值 ERA 3.96）
- IL 名單（11 位中 10 位投手）：
  - 核心 high-leverage：**Hunter Harvey, Phil Maton, Porter Hodge, Daniel Palencia** → ≥ 3 名核心
  - 先發輪值：Justin Steele (60-Day), Matthew Boyd, Jordan Wicks
- **牛棚傷兵修正**：🔴🔴 極高（3+ 核心 IL）
  - 對手（NYM）+1.0 run in late innings
  - CHC ML -5%

### NYM 牛棚（自動取值 ERA 3.96）
- IL：**A.J. Minter (15-Day 高桿左投), Reed Garrett (60-Day setup)**, Gerber, Hagenman, Núñez, Megill
- 替補深度尚可：Devin Williams, Clay Holmes, Craig Kimbrel 仍在
- **牛棚傷兵修正**：🔴 高（2 名核心 IL）
  - 對手（CHC）+0.5 run
  - NYM ML -3%

### 牛棚雙向閘門
✅ 兩側修正都完成：O/U（兩邊加 total → 淨 +0.3~0.5 run），ML（CHC 雙方折抵後仍略扣更多）。

---

## 3.4 條件修正

| 項目 | 狀態 | 修正 |
|------|------|------|
| Park Factor | Wrigley 102 | 總分 +2% |
| 天氣 | 4 月芝加哥（待 WebSearch 確認） | 預估冷 → -0.2 run |
| **Juan Soto 10-Day IL** | NYM 主砲缺 | NYM 預期得分 -0.3~0.5 run |
| Polanco / Baty 2B IL | NYM 中線薄 | -0.1 run |
| 連敗心理 | NYM streak -9 | 難量化，微偏利空 |
| 早季樣本 | 雙方僅 20 場 | 降低本季數據權重、D4 觸發 |
| 賽前場驗證 | 昨日 CHC 12-4 NYM | 同系列延續 |
| 主審 | 未知 | 跳過 |
| Age/TJ | Taillon 34 📉📉 | 已反映在本季 ERA，不額外扣 |

---

## 3.5 修正後預期得分（粗估，待 predict.py 計算）

- **CHC 預期得分**：基準 ~4.3 → vs Peralta（優投）→ 3.5 → PF +2%、冷天 -0.2、NYM 牛棚傷 +0.3 ≈ **~3.6**
- **NYM 預期得分**：基準 ~4.3 → vs Taillon（差投）→ 4.4 → Soto IL -0.4、PF +2%、冷天 -0.2、CHC 牛棚傷 +0.5 ≈ **~4.3**

手算總分預估 ~7.9，但預測最終採 formula_prediction。

---

## 3.6 整體判斷

- **投手優勢**：NYM（Peralta 明顯勝 Taillon 2 檔）
- **打線優勢**：CHC（OPS 差 ~100 點，Soto IL 放大落差）
- **牛棚**：雙方都缺核心，CHC 更嚴重（淨 +0.5 run 對 NYM）
- **環境**：Wrigley 略打者友善、冷天抵消
- **動能**：CHC 🔥 L10 6-4 / NYM 🥶 L10 1-9（BABIP 部分解釋 NYM Cold）

**基本面傾向**：
- ML：兩方力量接近對沖 — 投手 NYM 優、打線 + 牛棚 + 主場 + 動能 CHC 優；**結果略偏 CHC 直接勝**，但 Peralta 有能力單場壓制。
- Run Line：CHC -1.95 讓分偏重（20 場 run diff 每場才 +1.35），真實方向 CHC ML 強度中等；**NYM +1.95 有價值**。
- O/U：8.15 線；Peralta 壓制 + 冷天偏 UNDER，但雙牛棚傷 + NYM BABIP 回歸偏 OVER；**偏中性偏 UNDER**。

**風險**：
1. Peralta xFIP 2.91 / 新的 GB 傾向若延續 → 真實水平比 🟡 更高，CHC 得分可能壓到 2-3
2. Soto 傷況復出時機若為今日 → NYM 得分預期需立即上修
3. Taillon 小樣本 FIP 6.04 運氣偏差，xFIP 3.70 真實水平接近 Solid；不宜視為 pure 🟢

盤口星級與最終推薦 → Phase 4 predict.py。
