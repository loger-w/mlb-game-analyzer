# PHI @ CHC — 2026-04-22 Phase 3 基本面分析結論

## 先發投手

### Matt Boyd (CHC, home) — LHP
- 2026：2 GS / 9.33 IP，ERA **6.75** / FIP 1.81 / xFIP 1.40，K% 45.9、BB% 8.1、GB% 22.2（極端飛球型）
- 2025：31 GS / 179.7 IP，ERA 3.21 / FIP 3.46，K% 21.4 — 穩定中段先發
- **狀態：今日從 15-Day IL 啟動回歸**（預期 pitch count 限制 ~75-85，約 4.5-5 IP）
- **YoY Statcast 結論（雙極化）**：
  - 改善：avg velo 85.9→86.7（+0.8 mph）、whiff 10.7→17.3、CSW 26.4→31.9
  - 退化：**hard hit 22.2→34.7、barrel 9.0→17.6、EV95 38.0→58.8**（三項一致惡化）
  - 判定 = K 或挨轟二選一；ERA 偏高是真實 barrel 風險，FIP 1.81 是小樣本 + 少 HR 的幸運值
- **分級：🟢 Back-end**（IL return 首戰降半檔）

### Kyle Backhus (PHI, away) — RHP
- 2026：7 G / 0 GS / 6.67 IP，ERA 5.40 / FIP 4.75 / xFIP 1.83，HR/9 2.70 ⚠️
- 2025：32 G / 0 GS / 25.3 IP，ERA 4.62 / FIP 3.85 — 純後援
- 球種 SI 57% / ST 27% / CH 9%，是軟投 ground-ball 型後援（avg velo 85.6）
- **狀態：MLB 生涯 0 次先發**，今晚實質為 **opener / bulk（2-3 IP）**，PHI pen 要吃 6+ IP（受 Wheeler IL 牽連）
- **分級：⚪ Below-Average opener**

## 打線評級

### CHC（home）🟡 Average
- 全隊：avg_ops 0.775, xwoba 0.333, K% 20.3, BB% 11.4
- Heat 標示 🔥 Hot（avg_babip .302 → L7 **.389**），⛔ **B10 BABIP 回歸閘門觸發**
  - **回歸判定**：L7 BABIP .389 ≥ .370 → 回歸 ~.300 後 Hot 判定虛高，**不加 +0.5 Hot run value**
  - 實際真實水平 = Normal-Warm
- 強點 vs Backhus (RHP)：Ian Happ L vs RHP .943 / PCA L vs RHP .550 / Busch L / Ballesteros L vs RHP 1.101
- 核心打者 vs RHP 整體：OPS .663-.943，均衡
- Bregman vs LHP / Hoerner vs LHP（1.171）— 不適用今晚（Backhus RHP）
- 雙方系列賽前兩場：CHC 7-4, 5-1（連轟 PHI）

### PHI（away）🟡 Average
- 全隊：avg_ops 0.685, xwoba 0.323, K% 20.4, BB% 8.8
- Heat 標示 🥶 Cold（avg_babip .273 → L7 **.236**），⛔ **B10 BABIP 回歸閘門觸發**
  - **回歸判定**：L7 BABIP .236 ≤ .260 → 回歸 ~.300 後 Cold 判定過重，**不扣 -0.5 Cold run value**
  - 實際真實水平 = Normal
- Boyd 是 LHP — **PHI 左打大量 vs LHP 劣勢**：
  - Schwarber vs LHP OPS **.598**（vs RHP 1.209）
  - Marsh vs LHP **.376**（vs RHP .795）
  - Crawford vs LHP **.237**（vs RHP .908）
  - Harper vs LHP .879（少數硬抗 LHP 左打）
  - Stott vs LHP .786（反常左打打 LHP）
- 強點仍在 Harper (xwoba .423) + Schwarber (xwoba .394) — 但 Schwarber 今晚 platoon 劣勢

## BABIP 回歸判定

- **CHC 近 7 天 BABIP = .389（≥ .370 閘門觸發）**
  - 回歸 ~.300 後：Hot 標示虛高，近 7 天表現含 0.9 run/game 的 BABIP 噪音
  - **結論**：不加 +0.5 Hot run value；真實水平 = Normal-Warm

- **PHI 近 7 天 BABIP = .236（≤ .260 閘門觸發）**
  - 回歸 ~.300 後：Cold 標示過重，運氣差壓抑了實際產出
  - **結論**：不扣 -0.5 Cold run value；真實水平 ≈ Normal

- 個別打者：Bohm L7 BABIP .056（極端爛運）、Swanson L7 BABIP .176（極端爛運）、Schwarber L7 .188 — 這些打者不應被視為真 Cold/Slump；預期回歸

## 牛棚雙向修正值

### CHC（home）牛棚：4 名核心 IL
- Hunter Harvey（15-Day）、Porter Hodge（15-Day）、Daniel Palencia（15-Day）、Phil Maton（15-Day）
- 另 Justin Steele / Cade Horton（先發輪值 60-Day，不計入 pen）
- **修正值**：
  - O/U：對手（PHI）**+1.0 run** / 信號 +2（極高）
  - ML：CHC **−5%**
- 替補：Assad/C.Rea/C.Martin/R.Rolison/Milner/Thielbar/Webb/Little/R.Martin — 深度球員 ERA 可能偏高，與 team pen ERA 3.63 不對等（pen ERA 被好 IP 撐高）

### PHI（away）牛棚：1 名核心 IL
- Jhoan Duran（Closer, 15-Day）；其餘 Bowlan/Lazar/Pop 非核心
- **修正值**：
  - O/U：對手（CHC）**+0.3 run**
  - ML：PHI **−2%**
- **額外負擔**：Backhus opener = PHI pen 要扛 6+ IP；系列賽過去三天消耗未知但 PHI 連輸 2 場（7-4、5-1，可能 starter 未吃完 IP）

## 條件修正 + 環境

| 信號 | Run Value |
|------|----------|
| CHC 牛棚核心 4 人 IL → PHI 得分↑ | **+1.0 to PHI** |
| PHI 牛棚 opener 模式 → CHC 得分↑ | **+0.4 to CHC** |
| PHI Closer Duran IL → CHC 得分↑ | **+0.3 to CHC** |
| Boyd IL return pitch limit（早出場 → 深 pen exposure） | **+0.3 to PHI** |
| PHI 左打 vs LHP platoon 劣勢（若 Schwarber/Marsh/Crawford 全出賽） | **-0.4 to PHI** |
| Park Factor 102（Wrigley）| 得分 ×1.02 |
| CHC Hot BABIP 回歸（閘門觸發） | 不加 +0.5 |
| PHI Cold BABIP 回歸（閘門觸發） | 不扣 -0.5 |
| 4 月 Wrigley 風向未知（潛在影響 ±0.5-1 total） | 暫以中性處理 |

## 修正後預期得分（formula baseline）

- League baseline 4.5 RS/G，xwOBA 0.320，ERA 4.30
- **E[R_CHC]** = 4.5 × (.333/.320) × (PHI pen ERA 4.52/4.30) × (PF 1.02) ≈ **5.0**
  - + 0.4 (opener) + 0.3 (Duran IL) = **5.7**
- **E[R_PHI]** = 4.5 × (.323/.320) × (blended CHC ERA ~4.2/4.30) × (PF 1.02) ≈ **4.5**
  - + 1.0 (CHC pen 4-core IL) + 0.3 (Boyd early exit) − 0.4 (L platoon) = **5.4**
- **Total ≈ 11.1**（vs OU line ~9.25-9.5）

## YoY 對比結論（Boyd）

- **三項接觸品質一致退化**（hard-hit +12.5 / barrel +8.6 / EV95 +20.8）→ Boyd 今年 FIP 1.81 明顯**低估真實水平**
- **兩項 K 相關改善**（velo +0.8 / whiff +6.6）→ 保留 K 上行（但小樣本）
- 淨判定：**ERA 6.75 是真實的 barrel 問題，不是純運氣**；formula 用 FIP 1.81 給 PHI 僅 2.0 run 太樂觀；補 +0.5 run 到 PHI

## 整體判斷（基本面方向）

- **Formula 原始比分**：CHC 5.5 / PHI 2.0（formula 用 Boyd FIP 1.81 過度壓低 PHI）
- **修正後 (Signal Adjustments)**：
  - CHC：5.5 + 0.4 (opener) + 0.3 (Duran IL) = **6.2**
  - PHI：2.0 + 0.5 (Boyd YoY barrel) + 1.0 (CHC pen 4-IL) + 0.3 (Boyd 早退深 pen) − 0.4 (L platoon) = **3.4**
  - **Total ≈ 9.6**（vs OU 9.5 差 +0.1，在 SD 4.5 噪音內）
- **勝負方向**：**CHC 勝**（XGBoost 75.5%、Log5 65.7%，兩方向一致）— 主場 + 系列賽連勝 2 場 + Boyd K 上行 + PHI 左打 platoon 被壓
- **讓分方向**：CHC 讓 1.25 run；模型比分差 6.2 − 3.4 = **2.8 run**，超過讓分值 — 方向支持 CHC cover -1.5
- **信心**：MEDIUM（模型一致指向 CHC，但 Boyd 首戰回歸 IL 為主要不確定來源）
- **信心**：LOW-MEDIUM（大量雙向未知：Boyd IL 後實際水準、PHI opener pen 可用深度、Wrigley 風向）
- **主要風險**：
  - Boyd 如果複製 4/1 LAA 那場（10K/5.67IP/1ER）= CHC 輕鬆贏，Over 不成立
  - 風向若吹出（out to CF/RF）→ Wrigley 大爆炸，Over 強化
  - 風向若吹入（in from CF）→ 兩方都壓 fly balls，Under 回歸
