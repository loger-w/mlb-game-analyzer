# Phase 3 分析結論 — 2026-04-24 COL @ NYM

## 先發投手對決

### Freddy Peralta (NYM, R) — 🟡 Solid Starter
- 本季：ERA 4.05 / xERA 3.92 / FIP 4.08 / xFIP 3.44 / WHIP 1.09
- K% 25.0 / BB% 8.9 / K-BB% 16.1 / HR/9 1.35 / Hard Hit% 23.9
- 球種：FF 50.2% / CH 22.3 / CU 16.3 / SL 7.5
- Prior year (2025): ERA 2.70 / xFIP 3.28 / K% 28.2 — 菁英水準
- Platoon: vs LHB .213/.273/.426, vs RHB .158/.289/.184（RHB 壓制更佳，小樣本 45 BF 雜訊）
- 年齡 29 ⚡ 巔峰期；ERA vs xERA 落差僅 0.13（無 YoY 觸發）
- **結論**：巔峰期 Solid Starter，略差於 2025 但仍穩定；26.7 IP 樣本不足但數據協調

### Michael Lorenzen (COL, R) — ⚪ Below Average
- 本季：ERA 7.48 / xERA 5.77 / FIP 4.85 / xFIP 3.77 / WHIP 2.12
- K% 13.9 / BB% 5.2 / K-BB% 8.7 / HR/9 1.66 / Hard Hit% 34.7
- 年齡 34 📉📉 明顯退化；IP 21.67 小樣本
- **Platoon 警訊 vs LHB：.440/.463/.760（54 BF）— 災難性**

#### YoY 對比結論（2025 vs 2026）

| 指標 | 2025 | 2026 | Δ | 判定 |
|------|------|------|---|------|
| avg_velo | 88.2 | 87.7 | -0.5 | 邊際退化 |
| pitch_types | FF 22/SI 18/CH 17/SL 12 | CH 20/FF 18/SI 16/CU 14/FC 14 | 主球路重組（SL 消失，增 FC/CU）| 策略轉換 |
| whiff_pct | 10.1 | 9.7 | -0.4 | 微降 |
| hard_hit_pct | 26.0 | 34.7 | **+8.7** | ⚠️ 顯著惡化 |
| xera | 4.61 | 5.77 | **+1.16** | ⚠️ 真實水平下滑 |
| K% | 21.0 | 13.9 | **-7.1** | ⚠️ 揮空能力崩盤 |

> 🔴 **三項以上一致退化**（K%↓ / hard_hit↑ / xera↑ / velo↓ / 球種重組）→ **非運氣或小樣本**，結構性退化確定。即使以 xERA 5.77 取代 ERA 7.48，仍屬 ⚪ Below Average 等級，不得回歸至 2025 的 4.64 ERA 水平。

---

## 打線評級

### Mets (vs Lorenzen R) — 🟡 Average
- OPS .651 / xwOBA .312 / 本季 RS/G 3.52 / **近 10 場 RS/G 2.6（極寒）**
- last7_BABIP .287（Normal — 未觸發回歸閘門）
- recent_heat ⚖️ Normal，但傳統數據顯示冷
- OU lean 0

**傷兵衝擊**：Francisco Lindor (SS, 10-Day) + Jorge Polanco (2B, 10-Day) — 核心兩棒空缺，大幅削弱串聯。已反映在 2.6 RS/G 寒冰期。

**關鍵對位**：
- Juan Soto (L, vs RHP OPS 1.131) vs Lorenzen vs LHB .760 SLG → 🔥 MAJOR 左打爆擊點
- Brett Baty (L) vs RHP OPS .552 → 小幅受益
- Francisco Alvarez (R, vs RHP OPS .900) → 有火力
- Luis Robert Jr. BvP vs Lorenzen: **18 PA .188/.278/.250, 4 K (PA≥15 有效樣本)** → Lorenzen 壓制 Robert，但 Robert vs RHP 整體 OPS .600 本就疲弱

### Rockies (vs Peralta R) — 🟡 Average
- OPS .749 / xwOBA .321 / 本季 RS/G 4.15 / 近 10 場 RS/G 4.3（相對活躍）
- last7_BABIP .333（Normal — 未觸發回歸閘門）
- OU lean **-2（偏 Under）**

**傷兵衝擊**：Kris Bryant (DH, 60-Day) — 已長期缺陣，現行數據已反映。

**關鍵對位**：
- Mickey Moniak vs RHP OPS 1.218（最炙手可熱）但 BABIP .311 / xwOBA .326（偏運氣）
- Hunter Goodman 長打火力（vs RHP SLG .529, barrel% 13.7）
- Edouard Julien vs RHP OPS .812
- 整體 vs RHP 普遍向上，但 Peralta Hard Hit 23.9% 壓抑強擊球

---

## 牛棚雙向修正值

### Mets (主) 牛棚
- ERA 3.99（API 已排除 IL，反映當前活躍者）
- **核心 IL 4 人**：A.J. Minter (LHP setup) + Reed Garrett (primary setup) + Dedniel Núñez + Joey Gerber
- **替補品質強力**：Devin Williams (菁英 closer, ex-MIL) + Clay Holmes (前 closer) + Craig Kimbrel (veteran) + Brooks Raley (LHP)
- 字面触發「3+ 核心 IL → 對手 +1.0 run / ML -5%」，但 Williams + Kimbrel 取代使替補品質**反超**

**淨修正值**：
- **OU 修正**：+0.2 run（僅反映中段接力偶有空缺；後段 Williams/Holmes/Kimbrel 足以抵消）
- **ML 修正**：-1%（幾乎中性，後段依然強勢）

### Rockies (客) 牛棚
- ERA 3.69（罕見 COL 牛棚表現不差）
- IL：Jeff Criswell (60-Day), McCade Brown, Pierson Ohl, Kyle Freeland(starter 15-Day) — 無核心後段 IL
- 活躍包含 Victor Vodnik + Seth Halvorsen + Brennan Bernardino

**淨修正值**：OU +0.0 / ML +0%

---

## 條件修正摘要

| 信號 | 觸發 | OU Run Value | ML Lean |
|------|------|-------------|---------|
| Peralta 🟡 vs Lorenzen ⚪（2 檔差） | ✓ | UNDER 0.2 run | NYM +3% |
| Lorenzen YoY 結構性退化 | ✓ | OVER 0.3 run（Lorenzen 側） | COL -2% |
| Lorenzen vs LHB 崩盤 + Soto/Baty | ✓ | OVER 0.3 run | COL -2% |
| Mets 打線寒冰 2.6 RS/G 近10 | ✓ | UNDER 0.4 run | NYM -2% |
| Mets 傷兵 Lindor + Polanco | ✓ | UNDER 0.2 run（已部分反映）| NYM -1% |
| Rockies 傷兵 Bryant 長期 | 反映完畢 | 0 | 0 |
| Peralta BvP 壓制 Robert 18 PA | ✓ 有效 | UNDER 0.1 run（Rockies 側）| NYM +0.5% |
| Park Factor 97（Citi Field 壓分）| ✓ | UNDER × 0.97 total | 中性 |
| Mets 牛棚 IL + 高質量替補 | ✓ | OVER 0.2 run | NYM -1% |
| Rockies 牛棚 3.69 ERA，無 IL | ✓ | 0 | 中性 |
| COL 客場歷史弱勢 + Citi Field 大外野 | 隱性 | UNDER 0.1 run | NYM +1% |

---

## 修正後預期得分

**基礎估算**：

- **Mets 攻勢**：本季 RS 3.52 × 對上 Lorenzen xERA 5.77 加分（vs 聯盟平均 ~4.0 ERA 略差 +0.5 run）+ Soto/Baty LHB 爆擊 +0.3 − 打線寒冰 −0.4 − PF 0.97 ×
  → **3.52 + 0.5 + 0.3 − 0.4 = 3.92 × 0.97 ≈ 3.8 runs**

- **Rockies 攻勢**：本季 RS 4.15 × 對上 Peralta xERA 3.92 減分（vs 聯盟平均略佳 −0.3 run）− Robert BvP −0.1 − COL 客場 −0.2 − PF 0.97 ×
  → **4.15 − 0.3 − 0.1 − 0.2 = 3.55 × 0.97 ≈ 3.4 runs**

- **牛棚調整**：雙方各 +0.1 run 平均分攤

**最終**：
- Mets ~3.9 / Rockies ~3.5 / **Total ~7.4**
- 比賽差距：0.4 run（非常接近）

---

## 整體判斷

**方向傾向**：
- ML：基本面略偏 **Mets**（投手 2 檔優勢 + 主場 + 牛棚後段更強），但 Mets 打線極寒為最大不確定性
- 比分差距小，預期賽事懸念到中段
- Total：預期約 7.4，**略低於盤口 7.80**；傾向 UNDER 但差距不大

**信心程度**：
- 中等偏低 — 兩邊皆有結構性弱點（Mets 打線寒冰 / COL 先發崩盤），但互相拉扯使差距縮小
- Mets 投手優勢可信，但能否轉化為勝利取決於能否在 Lorenzen 身上打進 3 分以上

**值得注意的風險**：
1. Mets 打線持續寒冰：2.6 RS/G 是近 10 場，若連 2.5 分都打不進，Peralta 單場完美也只能打到 2-1
2. Lorenzen xERA 5.77 仍代表 Rockies 能給 Mets 機會；若 Rockies 牛棚上場後狀況穩定，Mets 追分不易
3. Soto 單人爆發可能是 Mets 攻勢的唯一亮點（高度集中於一人）
4. Rockies 近期 last7_BABIP .333 偏高，若回歸 Peralta 壓制效果會加乘
5. Citi Field 本身壓 HR，Lorenzen HR/9 1.66 可能不會如預期吞炸裂

**基本面方向總結**：NYM 有投手與主場微優勢；Total 偏 7.0-7.5 區間。盤口推薦交 Phase 4 模型裁定。
