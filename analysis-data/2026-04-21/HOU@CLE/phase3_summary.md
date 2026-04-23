# Phase 3 Summary — HOU @ CLE 2026-04-21

## 1. 投手對決

### Parker Messick (CLE, LHP, age 25) — 🟠 Strong Ace
- 2026: **ERA 1.05 / xERA 2.67 / FIP 2.48 / xFIP 3.14**, IP 25.67 (4GS), K% 26.9, BB% 7.5
- Statcast: avg_velo 88.2, Hard% 20.9, Barrel% 3.3, GB% 52.1（GB-heavy finesse LHP）
- Δ(ERA−xERA) = **−1.62** ≥ 1.5 → 已補跑 2025 YoY
- **YoY 驗證結論**: underlying Statcast 2025→2026 穩定（velo 87.8→88.2、Hard% 20.2→20.9、xwOBA 0.274→0.264、GB% 50.7→52.1）。**xERA 2.67 才是真實能力基線**，ERA 1.05 含 BABIP/HR luck。按 xERA 視為 Strong tier 有效，但不得假設 1.05 會持續

### Ryan Weiss (HOU, RHP, age 29) — ⚪ Below Average
- 2026: **ERA 6.75 / xERA 5.68 / FIP 6.24 / xFIP 4.20**, IP 14.67, 6G 1GS
- Statcast: avg_velo 90.6, Hard% 30.6, **Barrel% 11.4**, HR/9 2.45（極差）
- G/GS = 6/1 → **swingman/role_change pattern**，今日是他本季第 2 次先發
- Δ(ERA−xERA) = +1.07（未觸發 1.5 閘門）
- 先發期樣本極小（1 GS 2 IP），評估需靠本季整體 + prior-year N/A → 只能用 xERA/xFIP 作能力基線

### 對決明顯失衡
- Messick xERA 2.67 vs Weiss xERA 5.68 → **先發優勢差 3.01 ERA** 明顯倒向 CLE
- Weiss barrel% 11.4 allowed → HOU lineup alvarez/walker/correa 有長打機會，但 HOU 是打線客隊

## 2. 打線

### CLE（🟠 Strong）vs Weiss RHP
- avg OPS 0.706, xwOBA 0.343, BABIP **0.266**（低於 .290 → 小幅 +regression 預期）
- 核心：José Ramírez OPS 0.85 (xwOBA 0.417)、Brayan Rocchio 0.847、Chase DeLauter 0.811
- **個別 BABIP 極端**：
  - Schneemann BABIP .400 → 預期 −regression（降低 top OPS 0.89）
  - Bo Naylor .162 → +regression（OPS 0.432 會反彈）
  - DeLauter .19 → +regression
- BvP PA all 0 → 無 BvP 訊號可用（PA<15 閘門）
- recent_heat: ⚖️ Normal（無 Hot/Cold 訊號）

### HOU（🟡 Average）vs Messick LHP（GB-heavy）
- avg OPS 0.808, xwOBA 0.339, BABIP 0.290（穩定）
- 核心：**Yordan Alvarez OPS 1.215 (xwOBA 0.553)**、Walker 0.903、Altuve 0.834、Correa 0.734
- chain.obp_top3 **0.404**（table-setters 表現好）
- Alvarez LHB vs Messick LHP：歷史上 Alvarez vs LHP 略弱但仍 > 聯盟平均；但 GB-heavy finesse LHP 不易被 pull-HR 壓制
- RHB 群（Altuve/Walker/Correa/Paredes/Vázquez）vs LHP 通常 platoon 優勢，但 Messick GB% 52 傾向壓制 pull-HR 傾向的 pull-heavy RHB（如 Paredes）
- IL 失 Peña, Meyers, Dezenzo, Loperfido, Allen → 打線不完整，**depth 受損**
- **個別 BABIP 極端**：Vázquez OPS 1.132 但 BABIP .435 → 預期 −regression
- Brice Matthews K% 45.7 → 嚴重空棒風險
- BvP PA all 0 → 無訊號
- recent_heat: ⚖️ Normal

## 3. 牛棚（🔒 雙向閘門）

- CLE bullpen ERA 5.18（自 MLB API）— 略弱於平均，IL 1 位投手（Walters）
- HOU bullpen ERA 5.66（自 MLB API）— **嚴重殘缺**，IL 9 位投手：Hader（closer, 60-day）、Hunter Brown、Blanco、Wesneski、Javier、Pearson、Imai、Sousa、Bolton
- **雙向閘門推斷**：
  - HOU 牛棚殘 → HOU 中後期守備弱 → O/U 偏 **Over**（修正 +0.3~0.5）
  - 同時 HOU ML 向下修（不能只修 O/U 一側）
  - CLE 牛棚普通，Weiss 大概率早退 → CLE 中後期進攻對 HOU 牛棚 = CLE 容易再加分
- **今日 Weiss 先發 IP 預期 3-4 局**（xFIP 4.20 且 6.75 ERA + 單季 1 GS），牛棚要擋 5-6 局

## 4. 條件修正

### 環境
- **Progressive Field 主場**（park factor 98.0，略 pitcher-friendly）
- **天氣**：冷（42-50°F）+ 晴 + 風速 3-9 mph（輕微）
- 冷天 + pitcher-friendly park → 整場壓 **-3~5% 進攻**（飛球不飛）
- **主審 Andy Fletcher**（HP）— 無明確極端紀錄，視為 neutral

### 近期趨勢
- CLE 近 10: 5-5，昨日被 HOU 9-2 大敗（-streak 1）
- HOU 近 10: 3-7，昨日大勝 9-2（+streak 1）
- HOU 整季 9-15，run_diff −12；CLE 13-12，run_diff −7 → CLE 整季較穩

### 系列賽前場
- 2026-04-20: HOU 9-2 CLE（HOU 4 連敗後爆炸日，但先發 SP 不同：該日 Weiss 今日主投）

## 5. Phase 3 閘門自檢

- [x] BvP PA<15 → 無任何 BvP 結論被引用
- [x] 牛棚雙向閘門：HOU 殘 → O/U 偏 Over + HOU ML 向下，同步處理
- [x] BABIP 極端值已標註回歸（individual + team）
- [x] ERA vs xERA 閘門：Messick 已跑 YoY 2025 對比，xERA 2.67 作基線
- [x] **未寫入 ML/RL/OU 星級或盤口推薦**（single source of truth = Phase 4 prediction.json）
- [x] recent_heat ⚖️ Normal（無 Hot/Cold 判定需 BABIP 覆核）

## 6. 進 Phase 4 前的結論 framing（僅供 predict.py 輸入參考）

- **先發對決**：Messick（真實能力 Strong, xERA 2.67） >> Weiss（Below Average, xERA 5.68, 2.45 HR/9）
- **打線**：HOU 有 Alvarez 單核爆發力，但深度受 IL 影響；CLE 整體均衡但無超巨星
- **牛棚**：HOU 牛棚嚴重殘，CLE 中後期優勢
- **環境**：冷天壓整場進攻，park 微壓制
- **方向**：整體訊號向 CLE 傾斜；O/U 方向矛盾（HOU 牛棚殘→Over vs 冷天→Under）→ 交給 predict.py guardrail 處理

（此文件不含盤口/星級推薦；盤口 single source = Phase 4 prediction.json）
