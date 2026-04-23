# Phase 3 Summary — PHI @ CHC, 2026-04-21 19:40 ET @ Wrigley Field

## 1. Starter Matchup

### Jesús Luzardo (PHI, LHP) — 2026: 22.2 IP, ERA 7.94, FIP 2.84, K% 30.6, BB% 5.1, WHIP 1.46
- **ERA−FIP gap 5.10**：極端壞運，非能力衰退。
- **YoY 對比（2025: 183.2 IP, ERA 3.92, FIP 2.81, K% 28.5, BB% 7.5）**：peripherals 與去年一致（K% 微升、BB% 明顯下降）。真實水準 ≈ FIP 2.8 區間。
- **但：最近 vs 同一批 CHC 打線（4/15）** 5.1 IP / 12 H / 8 ER，被生吃一場。最近資料顯示即便 FIP 好看，這支打線對他有解。
- Step 2 閘門：role_change=None、2026 ERA 高於 2025（未觸發「IP<30 且 ERA 比 prior year 低 ≥1.0」條件）→ 不需 Statcast 下修，但需注意 ERA vs FIP 收斂預期。

### Shota Imanaga (CHC, LHP) — 2026: 22.0 IP, ERA 2.45, FIP 2.15, K% 37.8, BB% 6.1, WHIP 0.77
- **YoY 對比（2025: 144.2 IP, ERA 3.73, FIP 4.81, K% 20.6, BB% 4.6）**：K% 飆升 +17.2pp，2025 FIP 4.81 顯示 HR-prone 體質。2026 FIP 2.15 不具 sustainability。
- **Step 2 閘門觸發**：IP<30 且 2026 ERA 2.45 比 2025 ERA 3.73 低 1.28（> 1.0）→ 需做 YoY 修正。已取 2025 資料確認 2.15 FIP 為 SSS 噪音，真實水準較接近 3.5-4.0 ERA 帶。
- 但：上次先發對 PHI（4/16 系列賽）11 K、26 whiffs（2026 全聯盟單場最多），對這支打線 stuff 有 carry-over 效果。

### 對決相對強弱
- Luzardo 長期能力 ≥ Imanaga（FIP 2.84 vs 真實 3.5-4.0 投手）。
- 但「近期對這支打線」的資料方向相反：Luzardo 被 CHC 打線硬吃、Imanaga 對 PHI 打線有宰制感。

## 2. Lineups

| 指標 | PHI (Away) | CHC (Home) |
|---|---|---|
| Tier | 🟡 Average | 🟡 Average |
| avg_OPS | 0.688 | 0.768 |
| avg_xwOBA | 0.326 | 0.330 |
| avg_BABIP | 0.280 | 0.300 |
| K% | 21.0 | 20.6 |
| BB% | 8.9 | 10.7 |
| recent_heat | ⚖️ Normal | 🔥 Hot |

### Hot/Cold BABIP 檢查（關鍵閘門）
- CHC Hot：avg_babip 0.30 = 中性，**不屬極端回歸區（≤.260 或 ≥.370）**。Hot 部分有支撐，不需強力下修。
- PHI recent cold：最近 8 場 1.25 R/G，但季賽 OPS 0.688 與 xwOBA 0.326 接近聯盟平均。近期冷淡非真實衰退，但動能確實差。

### BvP
- 兩隊全部打者 vs 對方先發樣本 <15 PA（多為 3-8 PA），**insufficient_sample，全數不引用**（D1.5 閘門）。

## 3. Bullpens（雙向閘門）

### CHC Home Bullpen — season ERA 3.51，**受傷嚴重**
- IL：Daniel Palencia（closer, 左腹斜肌）、Phil Maton、Hunter Harvey、Ethan Roberts、Matthew Boyd / Cade Horton / Porter Hodge / Jordan Wicks（部分為 rotation）
- 深度大幅受損，但牌面 ERA 仍優於 PHI。

### PHI Away Bullpen — season ERA 4.06，**同樣受傷**
- IL：Jhoan Duran（closer, 左腹斜肌）、Zach Pop、Jonathan Bowlan
- 深度與賠率同時較差。

**雙向閘門結論**：兩隊都缺 closer，影響對稱。O/U 微幅偏高（長局高張力且雙方 closer 缺席，9+ 局時失血風險上升）；ML 方面 CHC 仍保有較小的牛棚邊際。

## 4. 環境條件

- **球場**：Wrigley Field，park_factor 102.0（微偏打者）
- **氣溫**：4/21 傍晚 upper 60s（~19-21°C，中性）
- **風**：預報 ~11 mph；4/20 夜場為 right-to-left（blowing out to LF），4/21 晚間未查到確切方向 → **保守視為中性**（不做單向加減）
- **主審**：未查證（略）

## 5. 近期動能

| 項 | PHI | CHC |
|---|---|---|
| 近 8 場 W-L | 1-7 | 6-0（連勝） |
| 近 8 場得失分 | 10 / 42 | 44 / 14 |
| 單場均 R/G | 1.25 RS / 5.25 RA | ~7.3 RS / ~2.3 RA |
| 本系列已打 | 連敗中、4/15 及 4/20 皆落敗 | 系列通吃中 |

PHI 動能明顯低於其季賽 baseline；CHC 動能高於季賽 baseline。但 Hot streak BABIP 尚未過熱（0.30），不做強力下修。

## 6. 關鍵面向 snapshot（供 Phase 4 參考，不含盤口判斷）

1. **FIP 模型 vs 實戰落差**：Luzardo FIP 2.84 可信，但近期對 CHC 被打爆；Imanaga FIP 2.15 不可信（YoY 顯示真實 3.5-4.0 水準），但對 PHI 打線 stuff 有 carry-over。
2. **兩隊動能完全相反**，且 PHI 季賽本就 -1.91 run diff。
3. **兩隊牛棚雙 closer 受傷**，O/U 上行風險增加；ML 方面 CHC 邊際仍略佳。
4. **park 微偏打者、風向不確定**；不加單向環境係數。
5. **BvP 全部 insufficient**，不引用。
