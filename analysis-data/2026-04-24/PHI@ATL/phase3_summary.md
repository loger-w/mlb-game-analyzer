# Phase 3 綜合分析 — PHI @ ATL 2026-04-24

## 先發投手對決

### Grant Holmes (ATL home, RHP, age 30)
- **Tier：🟡 Solid Starter**
- 2026 (5 GS / 26.1 IP)：ERA 3.42 / FIP 4.24 / xFIP 4.29 / xERA 3.53 / WHIP 1.10
- K% 19.6 / BB% 10.3 / K-BB% 9.3 / HR/9 1.03 / GB% 45.6
- Statcast：avg velo 89.6 mph（低）、Whiff 12.6%、CSW 27%、Barrel 5.3%、EV95 40%
- Prior (2025)：ERA 3.99 / FIP 4.18 / K% 25.0 / BB% 11.0
- **近 3 場**：vs KCR 5 IP 3 ER、vs ARI 6 IP 0 ER、vs LAA 6.2 IP 2 ER（近況 2.38 ERA，穩定）
- **Platoon：vs LHB .159/.269/.295 (OPS .564) — 全面壓制左打；vs RHB .216/.273/.314 (OPS .587)**
- 閘門：|ERA − xERA|=0.11、|ERA − prior|=0.57（皆未觸發 YoY）

### Andrew Painter (PHI away, RHP, age 23, rookie)
- **Tier：🟢 Back-end Starter（按 ERA）**，但 FIP 顯示實質遠優於此
- 2026 (4 G / 3 GS / 18.1 IP)：ERA 4.42 / FIP 2.28（elite）/ xFIP 3.13 / xERA 2.94 / WHIP 1.36
- K% 25.0 / BB% 5.0 / K-BB% 20.0（ACE 級紀律）/ HR/9 0.49 / GB% 35.3
- Statcast：avg velo 90.7 mph（偏低，可能初季保守）、Whiff 10.1%、CSW 26%、Barrel 5.5%、EV95 30.9%
- Prior：無（MLB rookie）
- **近 3 場**：vs WSH 5.1 IP 1 ER K8、vs SFG 4 IP 4 ER、vs ARI 5 IP 1 ER K7
- **Platoon：vs LHB .255/.280/.383 (OPS .663)；vs RHB .360/.414/.480 (OPS .894) — 右打被爆**
- 閘門：|ERA − xERA|=1.48（剛好未達 1.5，不觸發 YoY）、rookie 無 prior_year
- **ERA-FIP gap 2.14** → 實質表現被 BABIP/sequencing 拖累，規律上預期正向回歸

### 投手比較結論
- 表面 ERA 是 Holmes 好（3.42 vs 4.42），但進階指標 Painter 真實水平更強（FIP 2.28 vs 4.24）
- **關鍵 Platoon 錯位**：
  - Holmes 吃 LHB → 直接針對 PHI 核心 LHB（Schwarber/Harper/Marsh/Crawford/Stott）
  - Painter 怕 RHB → ATL 右打（Acuña/Riley/Baldwin/Dubón/Albies）有壓制空間
- 淨判：Holmes 的針對性遠比 Painter 嚴重，Painter 的 FIP-ERA 負偏差無法抵消面對 ATL 熱打線 + 右打劣勢

---

## 打線評估

### ATL vs Painter (RHP)
- **Tier：🟠 Strong** | avg OPS 0.807 / xwOBA 0.358 / BABIP 0.313
- **近期熱度：🔥 Hot**（last7 BABIP 0.299 — 屬正常區間）
- 核心對 RHP 火力：
  - Dominic Smith (LHB) 1.081 OPS vs RHP
  - Michael Harris II 0.997 vs RHP
  - Matt Olson (LHB) 0.986 vs RHP
  - Drake Baldwin 0.901、Ozzie Albies 0.844、Austin Riley 弱於 RHP (.484) ×
- Painter 對 RHB 0.894 OPS allowed → ATL 右打（Acuña 小樣本、Riley、Baldwin、Dubón）放大
- **結論：ATL 打線全面施壓 Painter**

### PHI vs Holmes (RHP)
- **Tier：🟡 Average** | avg OPS 0.693 / xwOBA 0.323 / BABIP 0.273
- **近期熱度：⚖️ Normal**（last7 BABIP **0.232** → 觸發 B10 回歸）
- 核心對 RHP 火力：
  - Schwarber (LHB) 1.113 OPS、Marsh (LHB) 0.945、Harper (LHB) 0.868、Crawford (LHB) 0.847
  - 但 Holmes 對 LHB **.159/.269/.295 (OPS .564)** — **完全針對 PHI 強打 LHB 群**
  - RHB：Turner .702、García .644、Bohm .440（極冷）、Sosa 小樣本 1.091
- **結論：Holmes 的 vs LHB 優勢系統性中和 PHI 左打主力**

---

## BABIP 回歸判定

B10 閘門觸發（PHI last7_babip 0.232 ≤ .260）：
- 聯盟平均 BABIP ~.300，PHI 近 7 天低 0.068（運氣偏差）
- 回歸效應 → 預期 PHI 打線本場較 7 天數據稍好（約 +0.1～0.2 run）
- 但 Holmes vs LHB 的 platoon 壓制是 **技術性/投手品質**因子，不是 BABIP 運氣
- **最終判定**：BABIP 回歸 **小幅上調 PHI +0.15 run**，不調整 recent_heat（原本就是 Normal）
- ATL last7_babip 0.299 在正常區間，不觸發 ATL 回歸調整

---

## 牛棚雙向修正值

### ATL 牛棚
- 季初 ERA 3.05（已包含 Iglesias、Jiménez 過往貢獻）
- **核心 IL**：
  - 🔴 Raisel Iglesias (Closer) — 15-day IL
  - 🔴 Joe Jiménez (High-leverage) — 60-day IL
- 缺陣人數：**2 名核心** → 表格對應 🔴 高影響
  - 對手 (PHI) 得分 **+0.6 run**
  - ATL ML **-3%~-4%**（取 -3.5%）
- Strider、Waldrep、Smith-Shawver 為輪值層，不納入牛棚計算

### PHI 牛棚
- 季初 ERA 4.56（弱）
- **核心 IL**：
  - 🔴 Jhoan Duran (Closer) — 15-day IL
- 缺陣人數：**1 名核心** → 🟠 中高影響
  - 對手 (ATL) 得分 **+0.3 run**
  - PHI ML **-2%**

### 雙向淨效果
- **OU 面**：兩隊 bullpen 皆弱化 → 總分 **+0.9 run**（ATL +0.3 from PHI weak bullpen、PHI +0.6 from ATL weak bullpen）
- **ML 面**：ATL -3.5%、PHI -2% → ATL 相對 ML **-1.5%**（ATL 損更大但仍擁牛棚品質優勢）
- **predict.py signal_adjustments**：`bullpen_il_home`、`bullpen_il_away` 皆需設定

---

## 條件修正

- **Park Factor**：Truist Park = 98（略 pitcher-friendly）→ 總分 × 0.98
- **天氣/風向**：未取得，跳過
- **主審**：未指定，跳過
- **角色轉換**：Painter rookie 初年、inning limit 風險 → 後段 PHI 牛棚暴露加倍
- **年齡**：Holmes 30（plateau）、Painter 23（成長期）

---

## 傷兵影響（非牛棚）

- **ATL**：Sean Murphy (C) 10-day IL → Drake Baldwin 接替（打擊數據亮眼 OPS 0.943、vs LHP 1.023、vs RHP 0.901）；C 替補品質佳，影響小
- **PHI**：J.T. Realmuto (C) 10-day IL → 替補 C 影響 pitch framing；Wheeler 非本場先發，不影響

---

## 近期狀態與 H2H

### 趨勢
- **ATL**：📈📈 強勢上升 — L10 8-2、L30 18-8（RS 5.77 / RA 3.38 / diff +62）、W2 streak
- **PHI**：📉📉📉 深度低潮 — L10 1-9、L30 8-17（RS 3.56 / RA 5.60 / diff -51）、**L9 連敗**

### 直接對決（上週）
- 4/17 ATL 9-0 @ PHI（shutout）
- 4/18 ATL 3-1 @ PHI
- 4/19 ATL 4-2 @ PHI
- ATL 3-0 系列賽，平均 5.33 - 1.00 大勝
- 且系列在 PHI 主場都能橫掃，現在 Braves 回主場優勢更大

### L9 連敗反彈風險
- PHI 歷史 mean reversion 存在，但反彈機會有限：
  - 面對同一強敵 ATL 第 4 次交手
  - 主場優勢給 ATL
  - Holmes 針對 PHI LHB
- 不視為強力反彈信號

---

## 修正後預期得分

### 計算

| 項目 | ATL (home) | PHI (away) |
|-----|-----------|-----------|
| Baseline (offense × opponent RA / League) | ~5.2 | ~3.3 |
| 投手品質修正 | × 0.95 (Painter FIP 2.28 negative, but platoon gift) = 4.94 | × 0.75 (Holmes vs LHB destroyer) = 2.48 |
| BABIP 回歸 (PHI) | — | +0.15 |
| 牛棚傷兵（對手 +run）| +0.3 | +0.6 |
| 熱冷狀態 (ATL W2 / PHI L9) | +0.2 | -0.2 |
| 小計 | 5.44 | 3.03 |
| Park Factor × 0.98 | 5.33 | 2.97 |
| **Adjusted** | **5.3** | **3.0** |

### 基本面整體判斷

- 方向：**HOME (ATL) 明顯優勢**（投手差 1 檔 platoon 化、打線火熱差 1 檔、主場、反向連敗）
- 信心：中高（多重因子同向）
- 總分：預期 **~8.3**，明顯**低於**盤口線 9.15（-30 quarter）
- 比分差：預期 **ATL 贏 ~2 分**，靠近 -1/-1.5 quarter 的 push/win 邊界
- 值得注意的風險：
  1. Painter FIP 真材實料，6 局內可能壓制 ATL 到 2-3 run
  2. PHI L9 極端連敗，mean reversion 機率略存
  3. Painter rookie variance 極大（可能 3 IP 5 ER 或 6 IP 1 ER）
  4. 若 Painter 崩場，ATL 得分可能衝到 7+，OU 翻轉
