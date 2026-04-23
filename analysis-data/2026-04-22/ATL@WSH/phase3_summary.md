# Phase 3 綜合分析 — ATL @ WSH (2026-04-22)

## 先發投手

### Zack Littell (WSH, R, 30yo) — ⚪ Below Average（退化確認）
- 2026: 4G/3GS/19 IP, **ERA 7.11 / xERA 7.49 / FIP 7.36** / xFIP 3.87 / WHIP 1.74
- K% 15.7 / BB% 6.7 / K-BB% 9.0 / HR/9 3.32 / GB% 54.8
- Statcast: avg velo **87.1**（警訊）, whiff 7.5, barrel 14.5, hard-hit 34.7
- Platoon: vs L .293/.356/.683 SLG, vs R .357/.386/.548 SLG（雙向慘烈）
- 2025 prior: 3.81 ERA / 4.73 FIP / 186 IP（穩定 4 號）
- **判定**：ERA≈xERA 確認非運氣，球速明顯下降 + HR/9 爆炸 = 結構性退化；無 YoY 硬閘門觸發（|ERA−xERA|=0.38<1.5；ERA 高於 prior 非低於）但 Statcast 支持真實水平 ≈ 6+ ERA

### Didier Fuentes (ATL, R, 20yo) — 實際 🟡→🟢 變異極大（role_change 下修）
- 2026: 1G/**0 GS**/4 IP 救援, ERA 2.25 / FIP 1.85 / xERA 3.29
- 2025 prior: 4 GS / 13 IP, ERA 13.85 / FIP 8.64 / xERA 7.72
- **YoY Statcast 對比結論**（補跑 2025 完成）：
  - avg velo: 90.3 → 92.4（+2.1 mph，✅ 結構性升級）
  - hard-hit: 33.0 → 22.7（−10.3 pp，✅ 接觸品質改善）
  - barrel: 14.3 → 10.0（✅ 改善）
  - ev95%: 57.1 → 30.0（✅ 大幅改善）
  - whiff: 10.2 → 5.8（⚠️ 下降）
  - pitch mix: 丟 ST/CU → FF/SL/FS 精簡（策略轉換）
  - xERA: 7.72 → 3.29（樣本太小，4 IP 救援無法外推）
- **判定**：三項接觸品質 + velo 一致改善 = 實質成長，但 whiff 下降 + 僅 4 IP 救援樣本 + **role change（牛棚→先發）第一場**，按規則降級一檔。今日先發能力預期 🟢 Back-end（xERA ~4.0-4.5），且體力限制預計 4-5 IP。`pitcher_stats.py` 的 🟠 tier 標記**因樣本過小不採信**。

## 打線

### WSH vs Fuentes (R) — 🟠 Strong
- OPS .739 / xwOBA .345 / BABIP .285 / K% 18.2 / BB% 9.3
- **last7 BABIP .210 → B10 觸發，見下方 §BABIP 回歸判定**

## BABIP 回歸判定

WSH 近 7 天 BABIP = .210（≤.260 極低運氣帶）→ 回歸至聯盟平均 ~.300 預期產量上升。判定：**近期 Cold 觀感不得扣 Run Value**，反向微加 +0.2 run 至 WSH 得分。ATL .301 屬 Normal，無回歸修正。
- 核心棒次健康：James Wood (.948 OPS, .442 xwOBA), CJ Abrams (.976 OPS), Dominic Smith (.962 OPS 客串), Curtis Mead 熱打
- BvP 無樣本（Fuentes 數據空）

### ATL vs Littell (R) — 🟠 Strong
- OPS .797 / xwOBA .356 / BABIP .313 / K% 19.2 / BB% 8.3, last7 BABIP .301（Normal）
- 打線主力健康：Matt Olson (.916 OPS), Acuña Jr. (.398 xwOBA 雖 OPS .704 偏冷), Harris II (.797 OPS, .415 xwOBA), Dubón .835 OPS, Drake Baldwin .908 (Murphy 替補)
- **缺陣影響**：Sean Murphy (C, 10D), Ha-Seong Kim (IF, 10D) — Baldwin 接手表現 .908 OPS 緩衝 Murphy 流失；Kim 位置已由 Albies/Dubón 補上
- BvP 無樣本

## 牛棚

- **WSH 牛棚 ERA 5.51（底段）** — IL 全為先發深度（Henry/Herz/Gray/Waldichuk/Williams）；高槓桿可用但整體薄弱
- **ATL 牛棚 ERA 3.13（帳面精英）** — 但核心 IL：
  - **Raisel Iglesias (Closer, 15-Day IL)** 🔴
  - **Joe Jiménez (Setup/High-leverage, 60-Day IL)** 🔴
  - Danny Young (LR, 60D)
  - 剩餘高槓桿：Robert Suarez, Aaron Bummer, Tyler Kinley, Joel Payamps

## 牛棚雙向修正值

核心 2 人 IL → 對手得分 +0.5~0.7 run / ATL ML -3~4% / 信號 +1

- O/U：WSH 得分 +0.6 run
- ML：ATL 勝率 -3.5%
- 仍優於 WSH 牛棚：替換後有效 ERA 估 ~3.8-4.0，相對 WSH 5.51 仍領先 ~1.5 run/9

## 條件修正

| 信號 | 方向 | Run Value |
|------|------|-----------|
| Fuentes role-change（bullpen→start）第一場 | WSH 得分 +0.3 | 先發體力 + 初始不適應 |
| WSH BABIP .210 回歸（近 7 天） | WSH 得分 +0.2 | B10 閘門，Cold 觀感修正 |
| ATL 牛棚核心 2 人 IL | WSH 得分 +0.6 | B9 雙向 |
| Park Factor 101（Nationals Park） | 總分 +0.05 | (101−100)×0.05 |
| Sean Murphy（C）缺陣 | ATL 得分 −0.1 | Baldwin 補 |
| Littell 結構性退化 + 被左右打線炸 | ATL 得分 +0.5 | xERA 7.49 支持 |
| ATL 昨夜 11:4 慘敗（rest/rebound） | 無確定方向 | MLB 反彈效應噪音 |

## 近期狀態 + H2H

- ATL L10 7-3（RS 5.8/RA 4.3，戰力曲線強，但昨夜 4-11 被 WSH 血洗 → streak −1）
- WSH L10 5-5（RS 5.5/RA 6.5），主場 Nationals Park 系列賽正 1-1
- 本系列 4/20 ATL 9-4 勝，4/21 WSH 11-4 勝 → 打線兩邊都能爆分，drop-off 前本季的整體 run environment 偏高
- Pythagorean（依 L30）：ATL Pyth 約 68%（16-8 + RS/RA 5.62/3.33 大差），WSH Pyth 約 46%

## 整體判斷

- **投手差距**：Fuentes（🟢 Back-end, 高變異）vs Littell（⚪ Below Avg 且持續退化）— Fuentes 名目上兩檔優勢，但**樣本太小**且 role change，實戰差距收窄到 **1-1.5 檔**
- **打線對等**：雙方 🟠 Strong，ATL 產出略高；WSH BABIP 回歸補回一些
- **牛棚**：WSH 5.51 vs ATL 3.8(IL 後) = ATL 仍領先 ~1.5 run/9，但縮水
- **環境**：Nationals Park 輕度偏打（PF 101）；本系列兩天共 28 分顯示高 run environment
- **方向傾向**：基本面偏 **ATL 勝**（投手 + 牛棚雙側優勢未完全失血），但 Littell 雙向爆打 + ATL 牛棚 IL 雙向 + PF 輕偏打 + 兩隊打線健康 → **總分偏高**
- **風險**：Fuentes 首次先發史（2026 0 GS）變異極大，若提早退場（<4 IP）WSH 牛棚 5.51 會被 ATL 打線打爆 → 可能變成互爆局；若 Fuentes 撐住 5-6 IP 低失分則 ATL 控盤。ATL -1.25 受 Fuentes 變異牽制。
