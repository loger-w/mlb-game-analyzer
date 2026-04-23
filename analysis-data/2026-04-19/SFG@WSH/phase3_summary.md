# Phase 3 Summary — SFG @ WSH 2026-04-19

## 先發投手

### Robbie Ray（SFG, LHP, age 34）
- **Script tier**：🟠 Strong Ace（但 2026 ERA 2.42 明顯運氣成分）
- **實際水平**：降一檔至 🟡 Solid Starter
  - xERA 4.42 / FIP 4.62 / xFIP 4.04 ← 這三項一致指向中段先發
  - HR/9 1.61 偏高、BB% 11.4 偏高
  - IP 22.3（本季 4 場 4 GS）→ 樣本小
- **YoY Statcast（2025 → 2026）**：方向分歧，不是 new-version
  - Velo 89.6 → 88.6（↓1.0 mph，年齡退化）
  - Hard-Hit% 26.5 → 21.1（↓5.4pp，改善）
  - Barrel% 9.6 → 14.8（↑5.2pp，惡化 — 被打到就打很強）
  - Whiff% 12.6 → 13.9（↑1.3pp，微升）
  - CSW 26.9 → 27.2（持平）
  - 2025 xERA 3.71 → 2026 xERA 4.42（去運氣後約 solid 水平）
- **Platoon**：vs LHB 5.6 BB% / 27.8 K% / .513 OPS；vs RHB 12.9 BB%（偏高）/ 27.1 K% / .650 OPS
- **球種**：FF 47% / SL 32% / CH 13% / KC 8%（四縫 + 滑球主幹）

### Miles Mikolas（WSH, RHP, age 37）
- **Script tier**：⚪ Below Average（快速退化）
- **實際水平**：持平 ⚪ Below Average
  - ERA 11.49 / xERA 6.21 / FIP 8.40 / xFIP 4.91 — 全面惡化，即便 xERA 也很差
  - IP 15.7（4 場 3 GS）
  - 2025 ERA 4.84 → 2026 ERA 11.49（絕對退化）
- **YoY Statcast（2025 → 2026）**：一致退化
  - Velo 87.5 → 86.6（↓0.9 mph）
  - Hard-Hit% 27.3 → 33.0（↑5.7pp 明顯惡化）
  - Barrel% 12.7 → 11.5（微降，但接觸品質整體下滑）
  - Whiff% 7.0 → 7.6（仍極低，本來就不靠 K 壓制）
  - 2025 xERA 5.27 → 2026 xERA 6.21（基準線本就下滑）
- **Platoon 災難**：vs LHB .324/.410/.500；vs RHB **.378/.429/.784 SLG**（RHB 徹底打爆）
- **球種**：FF 24% / SI 22% / SL 16% / CU 15% / CH 12%（五球種分散，無穩定武器）

### 投手差距
Strong Ace-降級 Solid Starter（Ray）vs Below Average（Mikolas）= **投手差 ≥ 2 級** → 觸發「單方碾壓」敘事候選

---

## 打線

### WSH vs Ray(LHP)
- **Tier**：🟡 Average（OPS .774 vs LHP）
- **xwOBA** .335 / **BABIP** .327（正常範圍，非極端）
- **Recent heat**：⚖️ Normal
- **Chain**：Top3 OBP .384（上壘佳）/ Mid SLG .279（清壘弱）
- **威脅點 vs LHP**：
  - James Wood（L）OPS .953（34PA） ← 關鍵威脅，左打打左投反而壓得住
  - Joey Wiemer（R）OPS 1.489（22PA，小樣本）
  - Jorbit Vivas 1.267（7PA 極小樣本）
- **弱點 vs LHP**：Abrams .681（24PA）、Lile .638、Nuñez .476
- **Hot BABIP 警告**（回歸預期下降，last_7）：
  - CJ Abrams BABIP .389 / OPS 1.306
  - Daylen Lile BABIP .391 / OPS .860
- **Cold BABIP**（回歸預期上升）：Nuñez .063、García .125

### SFG vs Mikolas(RHP)
- **Tier**：🟢 Weak（OPS .661 vs RHP）
- **xwOBA** .292 / **BABIP** .313（正常）
- **Recent heat**：⚖️ Normal
- **Chain**：Top3 OBP .300（上壘普通）/ Mid SLG .379
- **但 Mikolas vs RHB SLG .784 是災難** — 弱打線遇到 Below-Avg 投手的 platoon 大漏洞
- **威脅 vs RHP**：Adames .909（63PA）、Schmitt .830
- **BvP Mikolas 有效樣本**：
  - Adames 29 PA / .214 avg **✓ 歷史壓制**（唯一足夠樣本）
- **Hot BABIP 警告**：Jung Hoo Lee last7 BABIP .407 / OPS .867

---

## 牛棚

| 球隊 | bullpen ERA | 核心 IL | 影響 |
|------|------|-----|------|
| WSH | **5.80** | Cole Henry（15-day）；其餘 60-day IL 多為輪值（Gray、Herz、Waldichuk、T.Williams） | 牛棚品質整體差（非 IL 造成） |
| SFG | **3.64** | Randy Rodríguez（60），Foley（60），Hentges（15）— **核心 2 人 IL**，Buttó、Wick、Peguero 也缺 | 理論修正 O/U +0.5，但 ERA 3.64 顯示替補撐住 → **降為 +0.3** |

**適用修正**：
- **對 WSH 得分 +0.3**（SFG 牛棚核心 2 人 IL，溫和幅度 — 替補 ERA 已好於被替換者的平均預期）
- **SFG ML -2%**（牛棚影響較小 ML 幅度）
- **對 SFG 得分 +0.0**（WSH 牛棚 ERA 5.80 但無核心 IL，已反映在其賽季 RA/G 中）

---

## 條件修正

| 因子 | 值 | Run Value |
|------|------|-----|
| Park Factor | 101（Nationals Park，中性偏打） | +0.05（total） |
| Temperature | TBD（DC 4/19 日間，估 60-70°F） | 0（未達 ≤55 或 ≥85） |
| Wind | TBD | 0 |
| Umpire | TBD | 0 |
| Doubleheader | 否 | 0 |
| 賽季階段 | 🌱 開季（雙方 21 場，Mikolas 3GS / Ray 4GS）| LOW confidence flag |

---

## 近期狀態

- **SFG 近 10**：6-4（↑ 上升；RS 4.5 / RA 3.6）— 連贏前 2 場（7-6、10-5）
- **WSH 近 10**：5-5（→ 持平；RS 4.8 / RA 6.2）
- **系列賽**：SFG 2-0 領先
- **開季**：雙方 9-12（同記錄）

---

## 基準得分（公式）

以聯盟平均 RS 4.40 為基準，混合 2025/2026 xERA（賽季樣本權重）：
- Ray 混合 xERA ≈ 3.84（0.18×4.42 + 0.82×3.71）
- Mikolas 混合 xERA ≈ 5.46（0.20×6.21 + 0.80×5.27）

| 計算項 | 值 |
|-------|-----|
| E[SFG] 基礎 | 4.40 × (.292/.318) × (5.46/4.40) × 1.01 ≈ **5.06** |
| E[WSH] 基礎 | 4.40 × (.335/.318) × (3.84/4.40) × 1.01 ≈ **4.08** |
| 基礎總分 | **9.14** |
| +SFG 牛棚 IL → WSH +0.3 | 9.44 |
| +Park 修正 +0.05 | 9.49 |
| **修正後總分** | **~9.5** |

vs O/U line 8.5 → 差距 +1.0 run → **未達 1.5 門檻，預估 O/U PASS 但傾向 Over**

（注：predict.py formula 可能給出不同結果，以 prediction.json 為準）

---

## 整體判斷（方向性）

- **基本面強烈傾向 SFG**：
  - 投手差 ≥ 2 級（Solid Starter vs Below Average）
  - SFG 近 10 場明顯優於 WSH（6-4 vs 5-5，RA/G 3.6 vs 6.2）
  - SFG 剛贏前 2 場系列賽，兩場合得 17 分
- **風險點**：
  - Ray 本季 ERA 2.42 有運氣成分，xERA 4.42 真實水平下修；Barrel% 14.8 紅燈
  - Mikolas vs RHB 雖爛，但 SFG 打線整體 🟢 Weak 可能浪費 platoon 漏洞
  - SFG 牛棚核心 2 人 IL（溫和警訊）
  - 開季樣本少 → 預測可靠度 LOW
  - Adames BvP Mikolas 29PA .214（唯一足樣本 → 歷史壓制不利 SFG）
- **基本面方向**：SFG 勝、總分傾向 Over 但差距不足
- **D4 受讓盤偏見警報**：雙方 < 30 場，任何 RL 推薦需審慎，星級上限 ⭐⭐

（ML / O/U / RL 具體推薦由 Phase 4 predict.py 決定）
