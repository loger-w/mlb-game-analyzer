# Phase 3 Summary — 2026-04-22 PIT @ TEX

## 先發投手

### Jack Leiter (TEX, RHP, Age 26)
- **分級**：🟢 Back-end Starter
- ERA 4.87 / xERA 5.50 / FIP 3.99 / xFIP 3.35 / K-BB% 16.7%
- HR/9 1.33 (偏高), GB% 42.9, WHIP 1.48
- Statcast: avg velo 92.0, whiff 13.9%, CSW 31.1%, hard hit 35.2%, barrel 16.1%
- **Platoon（樣本 vs_L 44 BF 可信）**：
  - vs LHB：.324 / .432 / .541（重大弱點）
  - vs RHB：.209 / .261 / .279
- Prior year 2025 (151.2 IP, 29 GS): ERA 3.86 / FIP 4.02 / K-BB% 12.5 → 本季 K% 從 22.9% 跳至 26.7%（改善），但 BB% 10.0 略升，整體 xERA 比去年差
- xFIP 3.35 vs FIP 3.99 落差 → HR/9 1.33 有向下回歸空間

### Braxton Ashcraft (PIT, RHP, Age 26)
- **分級**：🔴 Elite Ace（腳本判定）→ 手動調回 🟠 Strong（僅 4 GS 小樣本 + 2025 為 mixed role 8 GS / 26 G）
- ERA 2.38 / xERA 1.99 / FIP 1.64 / xFIP 2.73 / K-BB% 22.0%
- HR/9 0.00, GB% 51.3, WHIP 1.06
- Statcast 接觸壓制全面 Elite：hard hit 22.0%, barrel 3.6%, ev95% 33.9%, whiff 13.5%
- **Platoon**：vs LHB .220/.319/.293 (BF 48)，vs RHB .200/.233/.250 (BF 43) — 雙向壓制
- 球種組合：FF 31.4 / CU 29.6 / SL 18.5 / SI 16.7 — GB 型 + curveball 主導
- 2025 prior year ERA 2.71 / FIP 2.66 — 先發/牛棚混合下仍優秀
- **樣本注意**：4 GS 22.7 IP 樣本不足以確認 Elite，但 xERA 1.99 + statcast 全面支撐 → 至少 Strong Ace 以上

**投手對決結論**：Ashcraft 🟠 vs Leiter 🟢，差距約 1.5-2 檔（xERA 1.99 vs 5.50 差距 3.5 分，即使回歸後仍 ~1.5 run 差距）。

## 打線

### TEX (vs Ashcraft RHP)
- **分級**：🟡 Average（avg OPS .702, xwOBA .315, BABIP .292）
- K% 24.2, BB% 10.0
- Chain 弱：OBP top3 .323, SLG mid .355
- Last7 BABIP .333（略高但未達 ≥.370 閘門）— Heat Normal
- 核心強棒：Nimmo .863 OPS, Jung .836, Seager .753
- 面對 Ashcraft 的挑戰：全員對 Sinker/Curveball/Slider 組合的 Swing decision，Ashcraft 雙向無弱點
- **BvP**：全員 PA < 15，sample_sufficient 全 False → ⛔ 不引用

### PIT (vs Leiter RHP)
- **分級**：🟡 Average（avg OPS .749, xwOBA .336, BABIP .306）— 較 TEX 打線強 ~20 OPS 點
- K% 22.8, BB% 10.8
- Chain 較佳：OBP top3 .373, SLG mid .407
- Last7 BABIP .309 — Heat Normal
- 核心強棒：Cruz .897 OPS（L）, Lowe .975（L）, O'Hearn xwOBA .404（L）, Reynolds .788（S）
- **關鍵優勢**：打線多位左打者（Cruz/Lowe/O'Hearn/Horwitz/Yorke），Leiter vs LHB 毀滅性 .324/.432/.541
- **BvP**：全員 PA < 15 → ⛔ 不引用

**打線對決結論**：PIT 打線對 Leiter 匹配度顯著高於 TEX 對 Ashcraft。

## 牛棚

| 面向 | TEX | PIT |
|------|-----|-----|
| 整體 ERA | 2.81（頂尖） | 3.55（中上） |
| 核心 IL | Chris Martin（setup, 15-day） | 無（Jared Jones 先發，不列入） |

## 牛棚雙向修正值

**B9 觸發**：TEX 有 1 名核心（Chris Martin, setup）在 15-day IL → 雙向修正：

- **O/U 修正**：對手（PIT）得分 **+0.3 run**
- **ML 修正**：TEX **-2%**

扣除後 TEX 牛棚有效 ERA ~3.10，相對 PIT 3.55，TEX 後段仍微優勢但差距縮小。

## 條件修正摘要

| 信號 | 方向 | Run Value |
|------|------|-----------|
| Park Factor 103（Globe Life） | 雙方 +3% | 總分 +0.2 |
| Chris Martin IL（TEX setup） | PIT 得分 +0.3 | TEX ML -2% |
| Leiter vs LHB 毀滅（PIT 左打 stack） | PIT 得分 +0.3 | — |
| Ashcraft 小樣本 (4 GS) + prior role 混合 | 下修 Elite → Strong | TEX 得分 +0.3 |
| Ashcraft GB% 51.3 + HR/9 0.0 | TEX 長打難度高 | OU 壓制 -0.2 |
| TEX 連勝 1（系列賽主場贏） | 略上升 | 微 |
| PIT 連敗 1 | 持平 | 無 |

## 修正後預期得分（基本面估算）

基礎：PIT avg 4.96 RS, TEX avg 4.22 RS；對戰投手 xERA 後調整
- **TEX 預期得分**：~3.3（Ashcraft 壓制 + 打線 Average + 小樣本回補）
- **PIT 預期得分**：~4.6（Leiter xERA 5.50 + LHB 優勢 + 牛棚缺口 +0.3）
- **總分基本面**：~7.9

（正式比分以 Phase 4 `predict.py formula_prediction` 為準）

## BABIP 回歸檢查

- TEX Last7 BABIP .333（未達 ≥.370 極端值）→ 維持 Heat Normal
- PIT Last7 BABIP .309 → 維持 Heat Normal
- 無需觸發 B10 回歸 Task

## 整體判斷

**方向傾向**：基本面明顯偏 **PIT**
- 投手差距 1.5-2 檔（Ashcraft Strong/Elite vs Leiter Back-end）
- Leiter vs LHB 重大弱點 x PIT 左打 stack = 高質量匹配優勢
- 牛棚 TEX 仍微優，但 Chris Martin IL 拉近差距

**信心程度**：中等
- 市場卻開出 PK + 總分 9（取中位）→ 市場充分定價 TEX 主場優勢與近期連勝；基本面分歧點在市場低估 Ashcraft xERA/FIP 支撐
- Ashcraft 僅 4 GS + 去年混合角色 → 回歸風險是主要 downside

**值得注意的風險**：
1. Ashcraft 小樣本（22.7 IP）— Elite 持續性未證
2. Leiter 在主場雖過去波動大，但 HR/9 1.33 有向 xFIP 3.35 回歸空間
3. 系列前場 TEX 剛贏 5-1 可能顯示 PIT 團隊打擊手感偏低
4. 市場 PK + OU 9 意味著基本面與市場觀點有顯著分歧 → 任何方向的信心打折扣
