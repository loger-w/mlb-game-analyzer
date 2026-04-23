# Phase 3 Summary — PIT @ TEX (2026-04-23, Globe Life Field)

## 先發投手對決

### Jacob deGrom (TEX, R, 37歲) — 🔴 Elite Ace
- 本季 4 GS / 19.7 IP | ERA 2.29 / FIP 3.46 / **xERA 2.68 / xFIP 2.99 / xwOBA 被 .264**
- K% 32.1 / BB% 7.7 / **K-BB% 24.4（頂級）** / HR/9 1.37
- Statcast: avg velo 93.3 / hard hit 21.2% / barrel 10.6% / whiff 14.1 / csw 28.4
- Prior year (2025): ERA 2.97 / FIP 3.56 / 172.7 IP — 一整季健康的 ace 身手
- 年齡退化（37 歲 📉📉📉）但 velo 與 xFIP 與 prior 相符，不額外扣分
- Platoon: vs_L 小樣本 (54 BF) OPS 允許 .508；vs_R 允許 .897（24 BF 噪音極大，不可獨立引用）
- PIT 打線 top-4（Cruz / Reynolds / Lowe / O'Hearn）打擊強但 K% 皆 >22%，面對 32% K% 的 deGrom 很容易被壓

### Bubba Chandler (PIT, R, 23歲, 成長期) — 名義 🟠 Strong Ace，實效 🟡 Solid- / 🟢
- 本季 4 GS / 20.0 IP | **ERA 3.15 但 FIP 4.65 / xFIP 4.97 / xERA 4.55 / xwOBA 被 .338**
- K% 20.0 / BB% 15.3（偏高）/ **K-BB% 僅 4.7（嚴重）** / HR/9 0.90
- Statcast: avg velo 95.2（強球速）/ hard hit 23.4% / barrel 7.3% / whiff 11.0 / csw 26.4
- Prior year (2025): ERA 4.02 / FIP 2.33（僅 31.3 IP 小樣本）
- **判定**：ERA 3.15 有 BABIP 運氣成分，FIP/xFIP/xERA 一致指向真實水平 ≈ Solid- / Back-end。不觸發閘門 13（|ERA-xERA| = 1.40 < 1.5；IP<30 但本季 ERA 比 prior 低 0.87 < 1.0），但 xFIP vs ERA 差 1.82 足以降檔 effective_tier。
- Platoon: vs_L BB% 11.8 / vs_R BB% 17.6 — 面對右打控球更差

### 對決差距
- 名義 tier：Elite 🔴 vs Strong 🟠 = 1 檔
- 實效 tier：Elite 🔴 vs Solid- 🟡 = **2 檔**（xFIP 為準）
- deGrom 對 PIT 預期壓制明顯強於 Chandler 對 TEX

---

## 打線評級

### TEX (主) — 🟡 Average
- xwOBA .313 / OPS .714 / K% 24.3 / BB% 10.2 / last7 BABIP .317（⚖️ Normal）
- 核心：Nimmo (OPS .859, xwOBA .363) / Josh Jung (OPS .883, xwOBA .349) / Seager (xwOBA .340 但 BABIP .204 表現被拖 — 回歸預期上升)
- ⚠️ **Wyatt Langford (IL-10, OF)** 缺陣 — 中心打者缺席，重大減損
- 面對 Chandler 15.3% BB% → 耐心戰可能榨出更多保送

### PIT (客) — 🟡 Average（但上沿）
- xwOBA .335 / OPS .745 / K% 22.8 / BB% 10.4 / last7 BABIP .309（⚖️ Normal）
- 核心 top-4：Cruz (xwOBA .383) / O'Hearn (xwOBA .402) / Lowe (OPS .941) / Reynolds (xwOBA .357) — 全隊最強 4 棒
- Cruz BABIP .382（閘門 .370 上緣，個人級略需警覺但未達全隊觸發）
- Ozuna（DH）OPS 0.542 — 5 棒弱點
- 面對 deGrom 32% K% + xwOBA 被 .264：上述強打 K% 皆高（Cruz 33.3 / Reynolds 25.5 / Lowe 22.6）→ K% 全壘掃射風險高

---

## 牛棚雙向修正值

### TEX bullpen（ERA 3.09 — 整體好）
- **核心 IL 1 人：Chris Martin（15-day, high-leverage setup）**
- matchup-factors.md 查表：1 核心 IL → ML -2% / OU 對手 +0.3 run
- 對 PIT 得分修正：**+0.3 run**（加到 PIT 得分）
- 對 TEX ML：**-2%**

### PIT bullpen（ERA 3.65 — 平均）
- 核心 IL 0 人（Jared Jones 是先發輪值；Triolo 是內野手）
- 無修正

**（對 O/U）牛棚總修正**：+0.3 run（全歸 PIT 得分）
**（對 ML）TEX -2%**

---

## 條件修正摘要

| 信號 | Run Value | 套用對象 |
|------|----------|---------|
| Park Factor 103 (Globe Life Field) | (103-100) × 0.05 = +0.15 run | 總得分（略偏打） |
| Wyatt Langford IL（打線主力缺陣，🔴 高） | -0.30 run | TEX 得分 |
| TEX 牛棚 Chris Martin IL（1 核心） | +0.30 run | PIT 得分 |
| deGrom 年齡 37 📉📉📉 | 無額外（本季 Statcast 未退步） | — |
| 雙方先發皆 🟠+ 信號 | **不套用**（Chandler 實效 🟡-，deGrom 🔴；非同檔級） | — |
| 雙方先發皆 🟡+ 信號（Chandler 實效 🟡-） | **不套用**（差距過大，壓制不對稱；分別反映在基礎 ERA 計算內） | — |
| Hot/Cold（近 7 天） | last7 BABIP 雙方 Normal → 不觸發 | — |
| H2H / Streak | PIT +1 streak（昨日 8-4 勝）；資訊性，不加 run value | — |

---

## 修正後預期得分（公式）

聯盟平均：得分 4.4 / 隊 / 場；xwOBA ~0.320；ERA ~4.20

### TEX 期望（面對 Chandler）
- 基礎 = 4.4 × (TEX xwOBA 0.313 / 0.320) × (Chandler xERA 4.55 / 4.20) × (103/100)
- = 4.4 × 0.978 × 1.083 × 1.03 ≈ **4.80 run**
- 修正：Langford IL -0.30 → **4.50 run**

### PIT 期望（面對 deGrom）
- 基礎 = 4.4 × (PIT xwOBA 0.335 / 0.320) × (deGrom xERA 2.68 / 4.20) × (103/100)
- = 4.4 × 1.047 × 0.638 × 1.03 ≈ **3.03 run**
- 修正：TEX 牛棚 Chris Martin IL +0.30 → **3.33 run**

**修正後總分：TEX 4.50 + PIT 3.33 = 7.83 run**

**修正後比分差：TEX 領先 1.17 run**

---

## 整體判斷（不含盤口星級）

### 方向
- 基本面偏 **TEX**：投手差距 2 檔（deGrom 🔴 vs Chandler 實效 🟡-），主場優勢，Park 微偏打但對雙方對稱。
- 比分差 1.17 run：TEX 小幅領先，但差距不到 1.5 run 的穩健門檻。

### 風險點
1. **Chandler 隱藏的 BABIP 運氣**：ERA 3.15 vs xERA 4.55，若今日 regression（HR/Hard Hit 打通）→ 比分可能 TEX 爆打 7+，PIT 多 Cruz/O'Hearn 砲 → Over
2. **deGrom 年齡 37**：一場突發 velo 掉 → 破局；但本季 statcast 未顯示退化
3. **Langford 缺陣**：TEX 得分力被削 ~0.3 run（已計入）
4. **PIT 打線強度**：實際上 xwOBA .335 > TEX .313，若 deGrom 不在 Elite 狀態 → PIT 可能反壓
5. **雙方首場 PIT 以 8-4 贏 TEX**（昨夜）→ Chandler 未上，但 TEX 牛棚昨日消耗需檢查（未於本 phase 擴展）
6. **總分 7.83 vs OU line 8.10**：差距 -0.27 < 1.5 run SD → **PASS 區**

### 信心
- ML 方向：中等偏 TEX
- OU：**低信心**（差距 < 1.5 run 在噪音區）
- 讓分：TEX -1 的差距（1.17）略低於讓分（1.0），**刀尖平衡**

> 具體盤口星級 / 推薦於 Phase 4 `predict.py --save` 產生。
