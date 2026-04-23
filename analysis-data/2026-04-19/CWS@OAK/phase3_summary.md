# Phase 3 Summary — CWS @ OAK 2026-04-19

## 先發投手

### Jeffrey Springs (OAK, LHP, age ~33) — 🟡 Solid Starter (trending up)
- 2026: ERA 1.46 / WHIP 0.77 / FIP 2.45 / xFIP 3.87 / xERA 2.45 / K% 22.0 / BB% 8.8 / GS 4 / IP 24.7
- Velo FF 85.5 avg (max 93.8) — pitcher does not overpower; lives on mix (FF 43 / CH 21 / SL 21 / ST 11 / FC 5)
- **YoY Statcast gate triggered** (IP<30 且 ERA ↓2.65 vs 2025 4.11)：
  | 指標 | 2025 | 2026 | 判定 |
  |---|---|---|---|
  | hard_hit_pct | 24.1 | 16.8 | ↓7.3 結構性改善 |
  | barrel_pct | 8.1 | 4.8 | ↓3.3 結構性改善 |
  | xERA | 4.30 | 2.45 | ↓1.85 大幅改善 |
  | csw_pct | 27.8 | 30.5 | ↑2.7 輕微改善 |
  | whiff_pct | 11.1 | 9.8 | ↓1.3 輕微退化 |
  | avg_velo | 85.2 | 85.5 | 持平 |
- 3/5 接觸品質一致改善 → **部分 new-version**；但 xFIP 3.87 暗示 ERA 1.46 仍有回升空間
- **真實水平估計**：xERA 2.45 與 xFIP 3.87 的中位 ~3.2（solid starter tier）
- Platoon 2025（大樣本）：vsL OPS .660 | vsR OPS .732 — 對右打者略差
- role_change = None；age 33（輕度退化區），但 Statcast 未惡化

### Noah Schultz (CWS, LHP, age 22, MLB debut 2026-04-14) — ⚪ Below-Avg (tiny sample)
- 2026（僅 1 場 MLB 先發，4.1 IP）：ERA 6.23 / WHIP 1.62 / FIP 4.05 / xFIP 4.68 / K% 20 / **BB% 20（極差）** / HR/9 0
- Sabermetrics fipMinus 94.9、eraMinus 147（1 場樣本）
- 無 Statcast prior year（美職菜鳥，2025 全部 MiLB）
- pybaseball lookup 缺失 → 手動從 MLB Stats API 建 JSON（`away_pitcher.json`）
- 樣本極小，不可判定 tier；但首次先發 4BB/4.1IP 的 walk 率顯示短期控球問題
- **⛔ 小樣本注意**：預期可信區間寬（真實水平 FIP 範圍 3.8-5.0）
- 左投身高 6'10"、2022 draft、頂級農場投手，天花板存在但今日狀態不明

## 打線

### OAK vs LHP Schultz — 🟡 Average
- avg OPS .706 / xwOBA .330 / BABIP .316 / K% 23.9 / BB% 10.5
- **vLHP 優勢打者**：Muncy .998 OPS (26 PA)、Kurtz .847 (29 PA)、Murakami 不相關 / Langeliers .856 (20 PA)
- Recent heat: Normal；但個體有 BABIP 過熱：
  - Kurtz L7 OPS 1.167 BABIP .417 ⛔ 不沖熱（BABIP 回歸閘門）
  - Langeliers L7 OPS 1.077 BABIP .500 ⛔ 不沖熱（回歸預期明顯下降）
  - Muncy L7 OPS .594 BABIP .364（small-sample 無淨信號）
- **缺陣**：Brent Rooker (10-Day IL, OF) — 主力右打火力，對 LHP 有優勢 → 打線 -0.2 run
- 打線對左投整體有利，但不到 elite

### CWS vs LHP Springs — 🟡 Average
- avg OPS .663 / xwOBA .312 / BABIP .262（略低於聯盟，輕微上修空間）/ K% 26.4
- **vLHP 樣本看好但 PA 小**：Murakami .744 (22 PA)、Montgomery 1.131 (22 PA)、Vargas .865 (25 PA)、Pereira 1.158 (9 PA)
- Recent heat: Normal；Murakami L7 OPS 1.182 BABIP .333 → BABIP 不極端，熱度可信
- **缺陣**：Teel (C 10-Day)、Hays (OF 10-Day)、Baldwin (60-Day) — 打線深度受損
- 整體偏右打為主，對 LHP 有機會但 Murakami 本身 vLHP 較弱

## 牛棚

- **OAK**：bullpen ERA 4.80，IL 1 人（Hoglund, 60-Day）— 核心未波及，正常運作
- **CWS**：bullpen ERA 5.60 🔴，**6 名投手 IL**（Murphy 15、Thorpe 15、Cannon 15、Bush 60、Vasil 60、Berroa 60）
  - Berroa 原為 high-leverage setup → 核心級 IL
  - Murphy、Cannon、Vasil 皆為輪值/牛棚 swing 角色
  - **判定**：≥2 名核心等效缺陣 → **O/U 對手 +0.5~0.7 run、CWS ML -3~4%、信號 +1**
- **雙向反映**：CWS 弱勢 O/U 加 run + CWS ML 扣 win% 均已計入

## 條件修正

| 條件 | 觸發 | Run Value |
|---|---|---|
| CWS 牛棚核心 IL ≥ 2 | ✅ | OAK +0.6 run / CWS ML -4% / 信號 +1 |
| OAK Rooker 缺陣 | ✅ | OAK offense -0.2 run |
| Springs YoY 接觸品質改善（3/5 一致）| ✅ | CWS offense -0.3 run（xERA 基準、非 ERA）|
| Schultz 菜鳥控球問題（1 場 BB% 20）| ✅ | OAK offense +0.2 run（walk upside）|
| OAK Kurtz/Langeliers 熱棒 BABIP 過熱 | ✅ | OAK offense -0.2 run（回歸扣分）|
| Park Factor (Sutter Health) | 100 中性 | 0 |
| 天氣 / 主審 | 未查 | — |

## 近期狀態與趨勢

- **OAK**：L10 7-3、L30 11-10、Seas 11-10；RS/G 4.14 RA/G 4.9；連敗 1
- **CWS**：L10 3-7、L30 7-15、Seas 7-15；RS/G 3.41 RA/G 4.95；連勝 1（昨場 9-2 破雙響砲 G1）
- **H2H（前場 4/18）**：OAK 7-6 險勝 CWS，系列雙方已有火力釋放
- CWS 為聯盟底層戰績，Seas Pyth 約 40%（依 RS/RA 對比）

## 修正後預期得分

基準（season RS/G）+ 對決修正：
- **OAK**：4.14（base）+ 0.6（CWS pen IL）+ 0.2（Schultz wild）- 0.2（Rooker IL）- 0.2（熱棒回歸）= **≈ 4.5 run**
- **CWS**：3.41（base）- 0.3（Springs new-version 折扣）= **≈ 3.1 run**
- **預期總分**：≈ 7.6（vs 盤口 9.5 → 明顯偏 Under）
- **OAK 領先幅度**：≈ 1.4 run（handicap line -1/-1.5 接近 fair）

## 整體判斷

**方向傾向**：基本面偏 OAK，三層優勢並行：
1. 先發投手差距（Springs solid starter vs Schultz rookie 小樣本 high-BB）
2. 牛棚差距大（CWS 6 名投手 IL、ERA 5.60）
3. OAK 主場 + 打線小優勢（但 Rooker 缺陣拉低）

**信心程度**：中等偏高（基本面三點互相一致）；風險為 Schultz 菜鳥天花板未知 + OAK 熱棒若延續

**值得注意的風險**：
- Schultz 僅 1 MLB 先發，真實水平區間寬（可能突然大幅好轉）
- Springs xFIP 3.87 > ERA 1.46 → 若今晚被打爆也不意外
- OAK L7 BABIP 過熱的兩位核心打者若回歸明顯，OAK 5+ 分機率下降
- CWS 昨場 9 分戰績、打線剛升溫（但 H2H 1 場不具統計意義）
- 盤口 9.5 相對基本面偏高 → Under 方向可能有價
- 具體盤口推薦與星級由 Phase 4 `predict.py` 生成
