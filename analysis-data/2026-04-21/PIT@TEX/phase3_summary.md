# Phase 3 Summary — PIT @ TEX, 2026-04-21

## 先發投手

| | Rocker (TEX, R) | Mlodzinski (PIT, R) |
|---|---|---|
| Tier | 🟢 Back-end Starter | 🟠 Strong Ace（實質偏高估） |
| ERA / xERA | 4.30 / 3.80 | 1.77 / 3.11 |
| FIP / xFIP | 4.40 / 3.86 | 2.31 / 3.40 |
| K% / BB% | 20.9 / 10.4 | 23.0 / 9.2 |
| HR/9 / GB% | 1.23 / 53.3 | 0.00 / 56.4 |
| IP / GS | 14.7 / 3 | 20.3 / 3 (prior 34G/12GS) |
| avg velo (YoY) | 90.7 → 90.2 | 91.4 → 89.4（**-2.0 mph**） |
| hard_hit% (YoY) | 29.4 → 23.4 ✅ | 25.7 → 30.8 ❌ |
| barrel% (YoY) | 11.8 → 6.7 ✅ | 6.0 → 1.7（HR 運氣） |
| whiff% (YoY) | 11.9 → 11.9 | 10.0 → 7.9 ❌ |
| 主要球種變化 | 新版：SL 38% + SI 37%（蟲殺配球）大量換滑球、去掉 CU | 新版：FS 29% + FF 29% + CU 14.8%（以 FS 為核心） |

**解讀**：
- Rocker：新版改造**有實證**（barrel 減半、hard_hit 大降）；代價是 BB% 10.4 偏高、14.7 IP 小樣本
- Mlodzinski：ERA 1.77 是**運氣（20.3 IP 0 HR）+ xFIP 3.40 持平 + velocity −2 mph**；真實水平接近 xERA 3.11 / xFIP 3.40
- 兩人 xERA 差距 0.7（3.11 vs 3.80），**不是表面 ERA 差距 2.53 暗示的天壤之別**
- Platoon 警訊：Rocker vs LHB BB% 16.2（PIT 有 Cruz-L、Reynolds-S、Lowe-L、O'Hearn-L、Horwitz-L 五位左打核心 → 送出保送潮風險）
- BvP 所有打者 PA < 15 → 不引用 ⛔

## 打線

| | TEX（主） | PIT（客） |
|---|---|---|
| Tier | 🟡 Average | 🟠 Strong |
| Team OPS / xwOBA | .701 / .312 | .764 / .340 |
| Team K% / BB% | 23.9 / 9.7 | 22.7 / 11.0 |
| Team BABIP | .288 | .310 |
| recent_heat | ⚖️ Normal | 🔥 Hot |
| chain obp_top3 / slg_mid | .331 / .358 | .377 / .423 |

**BABIP 回歸檢查**：
- PIT 🔥 Hot 但 top bat Cruz (.392 BABIP)、Reynolds (.346)、O'Hearn (.352) 三人高過聯盟均 .300 → 個別回歸壓力，Hot 加成需折半
- TEX 的 last_7 大多 .200-.310 合理範圍，無 Cold 懲罰
- 整體：PIT 攻擊優勢存在但沒表面那麼大

## 牛棚

| | TEX | PIT |
|---|---|---|
| 牛棚 ERA | 2.91（表面優） | 3.68 |
| 核心傷兵 | **Chris Martin 15d IL** 確定缺陣 + **Robert Garcia（LHP setup）** 左肩自 4/16 未投、今日狀態不明（best case Tues/Wed）| 無核心缺陣 |

**牛棚傷兵累計效應**：保守取 1.5 名核心缺陣
- 對手（PIT）+ 0.4 run
- TEX ML −2.5%
- 雙向閘門（O/U ＋ ML 皆反映）✓

## 條件修正（信號 → Run Value）

| 信號 | 方向 | Run Value |
|---|---|---|
| rocker_new_arsenal | PIT 得分 ↓ | −0.3 |
| mlodzinski_hr_luck_regression | TEX 得分 ↑ | +0.4 |
| mlodzinski_velo_drop_role_shift | TEX 得分 ↑ | +0.15 |
| mlodzinski_era_vs_xera_gap | TEX 得分 ↑ | +0.3（部分與 HR luck 重疊，淨 +0.2） |
| tex_bullpen_martin_garcia | PIT 得分 ↑ | +0.4 |
| park_factor_103 | 雙方 | +0.05 / +0.05 |
| pit_lineup_hot_but_babip_regression | PIT 得分 ↑ | +0.1（折半） |
| early_season_22g | D4 防護 | 自動 |

## 預期得分（基於上述修正）

- **TEX baseline**（對 Mlodzinski xERA 3.11）≈ 3.7；+ HR 回歸 (0.4) + velo drop (0.15) + PF (0.05) + era_xera (0.2) = **TEX ≈ 4.5 run**
- **PIT baseline**（對 Rocker xERA 3.80）≈ 4.4；− arsenal (0.3) + TEX 牛棚 (0.4) + PF (0.05) + hot (0.1) = **PIT ≈ 4.65 run**
- **總分 ≈ 9.15**（vs 市場線 ~8.3）
- **差距 ≈ PIT +0.15**（接近 pickem，基本面微偏 PIT）

## 整體判斷

- **方向**：基本面略偏 PIT（xERA 優 0.7 + 打線強度 + 對手牛棚缺人），但**幅度遠小於表面 ERA 差距暗示**
- **信心**：LOW（雙方 SP 皆 IP<30、xERA 與 ERA 差距大、22 場賽季樣本偏早）
- **風險 1**：Rocker 新版本改造剛起步（14.7 IP），若今晚繼續壓制接觸品質，PIT 的強打線會被降檔
- **風險 2**：Mlodzinski 的 1.77 ERA 若繼續（對 TEX 🟡 Average 打線）、TEX 仍可能無功而返
- **風險 3**：Rocker vs LHB BB% 16% → 若送 3-4 個保送給左打核心，差距可能擴大

盤口推薦 single source of truth = Phase 4 `prediction.json`。
