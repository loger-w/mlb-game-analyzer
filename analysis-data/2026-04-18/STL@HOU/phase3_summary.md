# Phase 3 Summary — 2026-04-18 STL @ HOU

## 先發投手

### Lance McCullers Jr. (HOU RHP, 32)
- **分級：🟢 Back-end（但 FIP 顯示真實水平接近 🟡 Solid 邊緣）**
- 2026: 3 GS / 15.1 IP / ERA **5.87** / **xERA 4.41** / FIP **2.90** / xFIP 2.90
- K% 27.9 / BB% 9.8 / K-BB% 18.1 / GB% 58.3
- Statcast: avg velo **88.0 mph**（低、警訊）/ max 94.2 / Barrel% 10.5 / Hard Hit% 30.7
- xwOBA .334 / CSW% 26.0
- Platoon: **vs L .250/.351/.438（弱）** / vs R .261/.292/.348（強）
- Prior year (2025): ERA 6.51 / FIP 5.36 / K-BB% 8.1 / 13 GS — TJ 復出次年
- **ERA vs xERA 差距 1.46**（閘門 1.5，未觸發 YoY）；ERA 與 prior year 差 -0.64（<1.0 未觸發）
- 判定：ERA 5.87 是運氣偏差，真實水平 FIP 2.90 / xERA 4.41。但 velo 88 低迷是結構警訊，K% 維持中可否持續存疑。

### Andre Pallante (STL RHP, 27)
- **分級：🟢 Back-end Starter（今年退化趨勢）**
- 2026: 3 GS / 15.0 IP / ERA **4.80** / **xERA 6.00** / FIP 4.63 / xFIP 4.98
- **K% 10.0 / BB% 11.4 / K-BB% -1.4（極差、負值）** / GB% **63.2**（elite）
- Statcast: avg velo 89.2 / max 97.1 / Barrel% 7.5 / Whiff% 8.5（低）/ CSW% 25.8
- xwOBA .381 / xBA .292（被擊球質量差）
- Platoon: vs L .206/.308/.265（可） / **vs R .308/.419/.462（被右打痛擊）**
- Prior year (2025): ERA 5.31 / FIP 4.56 / K-BB% 6.8 / 31 GS — 今年 K-BB% 由 +6.8 → -1.4，**顯著退化**
- 判定：ERA 4.80 運氣好，xERA 6.00 + K-BB% -1.4 顯示真實水平極差。GB% 高 + HOU 打線強 → 被集中打爆風險高。

## 打線

### HOU 🟠 Strong（vs RHP Pallante）
- OPS .833 / xwOBA .342 / BABIP .320（**正常，無回歸修正**）
- K% 19.1 / BB% 10.9 / recent_heat **⚖️ Normal**
- **Yordan Alvarez 🔥**：OPS **1.229** / xwOBA **.544** / vs RHP OPS .993
- Altuve（.897 OPS）、Cam Smith（.807, barrel% 18.5）、Correa 在陣
- HOU 右打為主 + Yordan/Altuve 左打 → 對 Pallante (RHP) 整體右打優勢（Pallante vs RHB slg .462）
- 缺陣主力：Peña（SS, 10-IL）、Jake Meyers（OF）、Dezenzo — 已由 lineup_analyzer 反映
- BvP 所有球員 PA < 5 → **不引用**

### STL 🟡 Average（vs RHP McCullers）
- OPS .691 / xwOBA .323 / BABIP .280（**正常**）
- K% 22.0 / BB% 11.2 / recent_heat **⚖️ Normal**
- **Burleson 🔥 vs RHP**：OPS **1.031** / xwOBA .410
- Herrera .746 OPS、Wetherholt .745 OPS、Walker, Winn 等
- McCullers 略弱 vs LHB（.438 SLG）→ STL 左打 (Burleson/Wetherholt/Gorman) 有利
- BvP 樣本全部 < 15 → **不引用**

## 牛棚

- **HOU bullpen ERA 6.09（災難級）** — 反映 Hader (60-IL, elite closer) + Javier + Blanco + Pearson 多核心缺陣
- **STL bullpen ERA 5.14**（差但不如 HOU 糟）
- 核心傷兵同時反映：
  - **O/U 修正**：雙方牛棚都差 → 偏 OVER 方向（已由 merged.json bullpen ERA 反映，不再 double-count）
  - **ML 修正**：HOU 牛棚比 STL 差約 0.95 ERA → HOU ML 降 ~3-4%
- HOU 牛棚消耗：04/17 輸 4-9，推測牛棚或 SP 都被打很慘，04/18 前有消耗

## 條件修正

- **Park**：Daikin Park PF 100（neutral）— 無修正
- **Weather**：室內可開頂，無風無雨 — 無修正
- **Umpire**：HP Alfonso Marquez — 傾向數據不足，neutral
- **季初樣本不足（HOU 21 場, STL 19 場, min=19 < 30）→ D1.5 INSUFFICIENT_SAMPLE 觸發 + D4 受讓盤偏見防護**
- **TJ 復出次年（McCullers）velo 88 低迷** → 小幅下修 HOU ML（velo 是 true leading indicator）

## 近期狀態

- HOU 近 10：**2-8**（RS 3.4 / RA 6.2）/ streak **-2** / 📉 強烈下滑
- STL 近 10：**6-4**（RS 4.9 / RA 5.5）/ streak **+3**（三連勝）/ 📈 上升
- 系列賽前場（04/17）：STL 9-4 勝 HOU
- BABIP 雙方 normal → 無回歸修正

## 修正後預期得分（估算）

- HOU 預期：Strong lineup vs Pallante xERA 6.00 → **~5.7-6.2 分**
- STL 預期：Average lineup vs McCullers xERA 4.41 / FIP 2.90 → **~4.3-4.8 分**
- 預期總分 **~10.0-11.0** vs OU 9.25 → **偏 OVER 方向**
- 預期差距：HOU - STL ~ +1.0~+1.5（HOU 主場強打）

## 整體判斷

- **方向**：基本面模糊／略偏 HOU 主場（強打 + Yordan 火熱 + Pallante 垃圾 xERA 6.00）
- **但** HOU 投手 velo 低警訊 + 牛棚崩壞 + 連敗 + 系列前場慘敗 → 抵消主場優勢
- **MOST LIKELY 走向**：高分局面。雙方爛投 + 雙方爛牛棚 → OVER 訊號相對清晰
- **ML 難判**：STL 是 road favorite (-1.10) 合理（HOU 投手牛棚傷兵重、狀態差），但 Yordan mash + HOU 主場可能壓榨 Pallante xERA 6.00
- **主要風險**：McCullers FIP 2.90 若當天 velo 回升、K% 維持 → STL 得分受壓縮，UNDER 翻車
- D1.5 / D4 觸發 → ML / RL 星級將自動上限 ⭐⭐
