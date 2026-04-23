# Phase 3 Summary — DET @ BOS 2026-04-18 16:10 ET (Fenway)

## Matchup Basics
- gamePk 824775 | Fenway Park | Park factor 105 (hitter-friendly)
- Away SP: **Tarik Skubal** (L, 29) | tier 🟠 Strong Ace
- Home SP: **Brayan Bello** (R, 26) | tier ⚪ Below Average

## 先發對決（投打核心）

| 指標 | Skubal (DET) | Bello (BOS) |
|---|---|---|
| 2026 GS / IP | 3 / 17.67 | 3 / 14.67 |
| ERA / WHIP | 2.55 / 1.08 | 6.14 / 1.91 |
| xERA / xwOBA | 3.85 / .314 | 5.63 / .371 |
| FIP / K-BB% | 2.24 / 20.0% | 5.49 / 0.0% |
| Prior year ERA/FIP | 2.21 / 2.34 | 3.35 / 3.92 |
| Velocity (avg/max) | 92.5 / 100.4 | 89.7 / 95.9 |
| CSW% / Whiff% | 28.9 / 12.7 | 29.4 / 16.2 |
| Barrel% / HardHit% | 7.5 / 25.0 | 9.3 / 24.4 |

**Platoon 關鍵**：
- Skubal 對 R 壓制（.203/.250/.270, K% 22.5）— BOS 打線以右打為主（Story, Contreras, Rafaela, Anthony, Mayer, Narváez、Duran L、Abreu L、Durbin R），Skubal vs R 樣本優勢明顯。對 L 小樣本 15 BF slg .600 是雜訊。
- Bello 對 R/L 都崩（.300/.353/.467 vs L；.323/.417/.484 vs R），僅 71 BF 合計但 xwOBA .371 佐證。

**Step 2 閘門**：
- Skubal |ERA−xERA|=1.30<1.5；ERA(2.55) 非比 prior(2.21) 低≥1.0 → 不觸發 YoY 補跑。
- Bello |ERA−xERA|=0.51<1.5；ERA(6.14) 高於 prior(3.35) → 不觸發 YoY 補跑。
- 備註（非阻塞）：Bello xERA 5.63 vs 去年 FIP 3.92 差 1.71，顯示今年 Stuff/指令退化，非僅 BABIP 運氣。

## 打線對決

| 指標 | DET vs Bello | BOS vs Skubal |
|---|---|---|
| Tier | 🟠 Strong | 🟢 Weak |
| OPS / xwOBA | .729 / .348 | .627 / .307 |
| BABIP | .307 | .286 |
| K% / BB% | 22.0 / 10.1 | 24.5 / 7.9 |
| OBP top3 / SLG mid | .373 / .395 | .328 / .398 |

**BABIP 回歸檢查**：兩隊 BABIP 皆在 .286–.307 之間，屬正常區間（.260–.370），無需回歸修正。

**近 7 天 recent_heat**：兩隊皆 ⚖️ Normal，無 Hot/Cold 判定。

**BvP 充足樣本**（PA≥15）：
- Gleyber Torres vs Bello：25 PA, .333/.360/.417（+正向傾向；BOS 打者對 Skubal 全部 <15 PA → 全部忽略）

## 牛棚與 IL

| 指標 | DET | BOS |
|---|---|---|
| Bullpen ERA | 3.41 | 3.84 |

**IL 摘要**：
- DET (9)：先發 SP 大傷 — Verlander、Olson、Jobe、Brieske、Melton；Bailey Horn (RP)；野手 Meadows、Sweeney、McKinstry。Skubal 頂替壓力集中單人。
- BOS (7)：**先發輪值重傷 4 人** — Crawford、Sandoval、Houck、Oviedo；牛棚 Slaten；一壘 Casas 10-day（Contreras 代打一壘）。Bello 本身就是頂不上去的結果。

**牛棚雙向閘門**：BOS 先發深度被嚴重消耗，牛棚 ERA 仍 3.84 中等；DET 牛棚 3.41 稍優。Bello 早退機率高（近 3 場平均僅 4.9 IP），BOS 將提前暴露牛棚長局。DET 方面 Skubal 效率高（場均 5.9 IP）。此項對 BOS 同時壓 ML 與下修 O/U 上限（長局 BOS 牛棚曝光）。

## 條件修正

**近期進攻（last 7 / season）**：
- DET RS 3.9 / 4.2；RA 3.0 / 3.6（進攻偏冷、防守熱）
- BOS RS 4.8 / 4.11；RA 4.1 / 4.53（進攻偏熱、防守差於季均）

**環境**：
- 球場：Fenway PF 105 偏打者
- 天氣：**49.9°F、風 9.2 mph、4-5pm 雨/雪交替、5pm 後降雨**（明顯壓低得分；冷天 + 降水 → ~0.3–0.5 R 下修）
- 主審：Ryan Additon（非極端 ump，暫按中性處理）

## 盤口（僅作基本面參照）
- 莊家（4h 前 snapshot）：BOS ML 1.70 / DET 2.32、O/U 7.0
- 使用者提供（台式）：BOS 讓 1.55（0.95/0.95）、O/U 7.40（0.94/0.94）
- 使用者的 O/U 7.40 比莊 7.0 多 0.4 runs → 使用者這本書偏 Over 開盤

---

**下一步**：Phase 4 交由 `predict.py` 做最終定量輸出（此檔不得寫星級或盤口推薦）。

Sources:
- [MLB Weather / AccuWeather Fenway](https://www.accuweather.com/en/us/fenway-park/02215/hourly-weather-forecast/53586_poi)
- [UmpScorecards](https://umpscorecards.com/data/games)
