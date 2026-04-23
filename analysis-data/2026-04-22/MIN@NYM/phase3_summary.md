# Phase 3 綜合分析 — MIN @ NYM (2026-04-22, Citi Field)

## 1. 先發投手

### 🏠 NYM — Clay Holmes (RHP, 33)
- 2026：ERA 1.96 / xERA 4.02 / FIP 4.01 / xFIP 4.01 / WHIP 1.09 / K% 17.2 / BB% 9.7 / K-BB% 7.5 / GB% 60.8 / HR9 0.78（23 IP, 4 GS）
- **真實分級 🟡 Solid Starter（非帳面 Ace）**：ERA − xERA = 2.06（閘門觸發），真實水平由 FIP/xFIP/xERA 三指標共識落在 ~4.0 區間
- **Role Change 2025 RP → 2026 SP**：2025 年 67 G / 0 GS 全後援（FIP 3.84, ERA 3.53）→ 2026 全先發；K% 從 25.1 降至 17.2（RP→SP 遞減正常）；降一檔評等謹慎
- Statcast: avg_velo 89.0（LP），whiff% 11.2，hard_hit% 29.7，barrel% 6.0
- Platoon：vs_L OPS .679 / vs_R OPS .457（左打反而較有效對付）
- Age 33 初期退化，今季 Statcast 持平（不額外扣）

### 🛫 MIN — Connor Prielipp (LHP, 25)
- **MLB 首秀 / 0 樣本** ⚪ Rookie Debut
- 2026 AAA：4 GS / 15.2 IP / ERA 2.30 / K9 12.64 / BB9 4.60 / WHIP 1.21 / OppAvg .204（上場 5 IP 1 ER 8K）
- 2025 AA 主樣本：19 GS / 61.2 IP / ERA 3.65 / K9 10.65 / BB9 2.63 / WHIP 1.49
- 估計 MLB 首季基準 FIP ~4.20；左投對左打者優勢，對右打不確定性大

## YoY 對比結論（Holmes 2025→2026）
| 指標 | 2025 | 2026 | 判定 |
|------|------|------|------|
| avg_velo | 89.3 | 89.0 | 持平 |
| pitch_types | SI40/ST19/SL11/CH16/FC9 | SI43/ST19/CH17/FC13/CU7 | 重組（加 FC/新 CU/砍 SL）|
| whiff% | 9.4 | 11.2 | +1.8 微升 |
| hard_hit% | 29.4 | 29.7 | 持平 |
| xERA | 4.33 | 4.02 | -0.31 微降 |

**判定**：技術面持平或微升，但 ERA 1.96 是 BABIP/LOB 運氣產物。真實水平 = 2026 xERA 4.02（接近 Back-end）。

---

## 2. 打線

### 🏠 NYM — 🟢 Weak / 🥶 Cold
- avg_xwoba .306 / avg_ops .601 / K% 21.4 / BB% 7.5
- **近 10 場：0-10 連敗，RS 1.9/game，極端低迷**
- 近 30 場 RS 3.26/game（本季 7-16）
- **🔴 三主力 IL**：Juan Soto (OF), Jorge Polanco (2B), Jared Young (OF) 全部受傷
- 打線 talent 結構性降級（非暫時）

### 🛫 MIN — 🟡 Average / ⚖️ Normal
- avg_xwoba .335 / avg_ops .730 / K% 20.7 / BB% 12.5（高耐心）
- 近 10 場：5-5，RS 5.7/game（火熱）
- 近 30 場 RS 5.09，本季 12-11

## BABIP 回歸判定（B10）
- **NYM last7 BABIP .22**：符合 ≤ .260 閾值。Soto/Polanco/Young 皆 IL → 70% 來自 talent drop / 30% 來自運氣；回歸目標 ~.270（非 .300）。**不扣 Cold run value，但也不補 luck bonus**
- **MIN last7 BABIP .25**：同閾值。MIN 主力健全 + xwoba 0.335 → 多數是短期運氣，預期回歸 ~.290。**不扣 Cold run value**；近 10 場 RS 5.7 是真實水平

---

## 牛棚雙向修正值（B9 閘門）

### 🏠 NYM 牛棚（表面 ERA 3.81 / 實際核心缺席嚴重）
- IL 核心：**A.J. Minter (LHP setup, 15d), Reed Garrett (primary setup, 60d), Dedniel Núñez (setup, 60d), Joey Gerber (15d), Justin Hagenman (60d)**
- **核心 ≥ 3 人 IL → 影響等級 🔴🔴 極高**
- **O/U 修正**：MIN 得分 **+0.7 run**（3 人 IL 檔次取中下）
- **ML 修正**：NYM 勝率 **-3~4%**

### 🛫 MIN 牛棚（ERA 4.82 / 小量傷兵）
- IL：Travis Adams, Mick Abel, Cody Laweryson（後援角色較低 leverage）
- Pablo López 60-day 是先發輪值，**不計入本場牛棚**
- **核心 ≈ 1-2 人 IL → 影響等級 🟠 中高**
- **O/U 修正**：NYM 得分 **+0.3 run**
- **ML 修正**：MIN 勝率 **-2%**

### 淨效果
牛棚缺口 NYM 比 MIN 大：對總分 +0.6~0.8 run（MIN 後段攻擊期優勢），對 MIN ML +1~2%。

---

## 4. 條件修正摘要

| 信號 | Run Value | 影響側 |
|------|-----------|--------|
| Park Factor 97 (Citi Field) | -0.15 | 總分下修 |
| NYM 牛棚核心 ≥3 IL | +0.7 | MIN 得分 |
| MIN 牛棚核心 1-2 IL | +0.3 | NYM 得分 |
| Prielipp rookie debut 不確定性 | +0.2 | NYM 得分 |
| NYM 打線 Cold（BABIP luck-driven 部分）| 0 | B10 擋下 |
| Holmes Role change RP→SP（已 4 GS） | 0 | game_log 為主 |

---

## 5. 修正後預期得分（formula baseline）

Formula baseline（`predict.py` 未 save）：home 4.2 / away 4.4 / total 8.6 (含 PF -0.15)

**信號修正後**：
- NYM (home): 4.2 + 0.3 (MIN 牛棚) + 0.2 (Prielipp rookie) = **~4.7**（但近 10 場 RS 1.9 + talent drop 提示保守下修至 ~4.0）
- MIN (away): 4.4 + 0.7 (NYM 牛棚) = **~5.1**
- 修正後總分 ≈ **9.1~9.3**（vs O/U line 8.5 → 差距 +0.6~0.8，**未達 1.5 閾值**）

---

## 6. 整體判斷

**方向**：基本面明顯偏 **MIN (AWAY)**
- 打線等級 MIN 高一檔（🟡 vs 🟢）+ 主力健全 vs NYM 三主力 IL
- 先發投手：Holmes 真實水平 🟡 vs Prielipp rookie ⚪，投手面 NYM 略優但差距不大
- 牛棚缺口 NYM 遠大於 MIN
- 動能：NYM 10 連敗 / MIN 昨日才贏 NYM 5-3

**信心**：MEDIUM（單場隨機性 ~45% + Prielipp 首秀不確定性）

**值得注意風險**：
1. Prielipp MLB 首秀可能翻車（0 樣本）→ 若早段崩盤 NYM 反而 cover
2. Holmes 真實 FIP 4.0 若對 MIN 耐心型打線（BB% 12.5）容易高 PC / 早退
3. NYM BABIP .22 回歸可能在今晚兌現
4. 10 連敗後心態爆發反彈（季初壓力釋放）

**盤口方向提示（不具體量化）**：基本面支持 MIN ML + 總分走高（牛棚互撞劇本），但總分差距 < 1.5 run → OU 多半 PASS。Run Line -1.5 因 margin 不確定而難支持高星級。
