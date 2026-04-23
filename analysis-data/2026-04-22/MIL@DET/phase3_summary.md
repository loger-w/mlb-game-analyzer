# Phase 3 Summary — 2026-04-22 MIL @ DET（Comerica Park）

> Phase 1+2 完成，merged.json 已寫入。本檔僅含基本面結論，**不含任何盤口星級或推薦**（single source of truth = Phase 4 prediction.json）。

---

## 3.1 投打對決

### Casey Mize（DET，RHP，28 歲）

| 指標 | 2026 季（22.67 IP / 4 GS） | 2025 季（149 IP / 28 GS） | Δ |
|---|---|---|---|
| ERA | 2.78 | 3.87 | −1.09 🔥（outlier） |
| xERA | **3.62** | **3.66** | ≈（**無實質改善**） |
| FIP | 2.97 | 3.79 | −0.82 |
| K-BB% | 19.2 | 16.5 | +2.7 |
| GB% | 39.0 | 42.9 | −3.9 |
| avg velo | 89.7 | 90.7 | **−1.0 mph ⚠️** |
| whiff% | 12.7 | 11.1 | +1.6 |
| hard% | 25.2 | 26.4 | −1.2 |
| barrel% | 9.8 | 9.1 | +0.7 |

**YoY 對比結論**：5 項 Statcast 指標除揮空率微升外，**xERA/xwOBA/hard-hit 幾乎與 2025 一致**，球速反而下降 1.0 mph（小樣本不排除 ramp-up，但值得注意）。表面 2.78 ERA 高度依賴低 BABIP 與 HR/FB 運氣（season.whip 1.19 但 xwOBA .304 與 2025 .298 相同）。**真實水平 ≈ 2025 的 3.6-3.9 ERA，分級實質為 🟡 Solid Starter 上沿，非帳面 🟠 Strong Ace。**

**Platoon（2026，小樣本）**：
- vs_L OPS **.304**（BF 54）— 霸王
- vs_R OPS **1.263**（BF 40）— 災難
- 配球：FF 35% / FS 30% / SL 20.8% / SI 9.9%，SV 大幅減少（13→4%），FS 使用增加，對左打更有效；但 R 打者等於沒球種壓制

### Chad Patrick（MIL，RHP，27 歲）

| 指標 | 2026 季（19 IP / 3 GS + 1 RP） | 2025 季（119.67 IP / 23 GS / 27 G） | Δ |
|---|---|---|---|
| ERA | **0.95** | 3.53 | **−2.58 🚨（嚴重 outlier）** |
| xERA | **3.06** | **3.88** | −0.82（小改善） |
| FIP | 3.94 | 3.39 | +0.55 |
| K% | **12.0** | 25.2 | **−13.2 🚨🚨** |
| BB% | 9.3 | 8.0 | +1.3 |
| K-BB% | 2.7 | 17.2 | −14.5 🚨 |
| GB% | 38.6 | 37.9 | ≈ |
| avg velo | 90.5 | 90.6 | ≈ |
| whiff% | 10.6 | 10.6 | = |
| hard% | **29.8** | 23.9 | **+5.9 ⚠️** |
| barrel% | 6.8 | 8.1 | −1.3 |

**YoY 對比結論**：球速與揮空率維持，但 **K% 暴跌 13.2 點、hard-hit 上升 5.9 點**，xERA 雖 3.06 比 2025 略好但**FIP 3.94 與 K-BB% 2.7 已亮紅燈**。0.95 ERA 完全是 BABIP/HR/LOB% 聯合運氣（xwoba .281 vs 2025 .306 → 小樣本 xwoba 偏低但 hard% 反向惡化）。**真實水平 ≈ 2025 的 xERA 3.88 ≈ 🟡 Solid Starter 中段，K-BB% 崩跌暗示下一場開始 regression 幅度遠大於 Mize。**

**Platoon（2026，小樣本）**：
- vs_L OPS .783（BF 38）— 受虐
- vs_R OPS .455（BF 37）— 霸王
- 配球：FC 36.8% / FF 23% / SI 21.2% / SV 14.8%，卡特球為主 → 對 L 打者效果打折

### DET 打線 vs Patrick（RHP，弱 vs L）

- Tier: 🟠 Strong | avg OPS .743 | xwOBA .353 | BABIP .316 | last7 BABIP .325 | heat ⚖️ Normal
- O/U lean: **+1**（微偏大分）
- **L 打者威脅叢**（對 Patrick vs_left 弱點）：
  - **Kevin McGonigle L** — OPS .904 / xwOBA **.424** / vs_RHP **.960**（78 PA）🔥 新秀突破
  - **Riley Greene L** — OPS .764 / xwOBA .394 / vs_RHP .694（L7 OPS **1.000**）
  - **Kerry Carpenter L** — OPS .762 / vs_RHP .773 / L7 OPS **1.067** 🔥
  - **Colt Keith L** — vs_RHP .784（但 L7 OPS .400 冷）
  - **Dillon Dingler R** — OPS .874 / xwOBA **.458** / vs_RHP .879 🔥（右打但總合恐怖）
- **冷手**：Torkelson（OPS .566 / L7 .390）、Báez（L7 .554）
- 主力 IL：Parker Meadows（60d OF）、Zach McKinstry（10d 3B）、Trey Sweeney（10d SS）
- **總評**：打線受益於 McGonigle 新秀補位，Strong tier 合理

### MIL 打線 vs Mize（RHP，弱 vs R）

- Tier: 🟠 Strong | avg OPS .762 | xwOBA .340 | BABIP .305 | last7 BABIP .325 | heat ⚖️ Normal
- O/U lean: 0（中性）
- **R 打者威脅叢**（對 Mize vs_right 1.263 OPS 弱點）：
  - **William Contreras R** — OPS .855 / xwOBA .360 / vs_RHP .840
  - **Gary Sánchez R** — OPS **1.027** / xwOBA **.428** / vs_RHP .782
- **L 打者**（Mize vs_left .304 壓制區）：
  - **Brice Turang L** — OPS .990 / xwOBA **.428** / vs_RHP **1.134** 🔥
  - **Jake Bauers L** — OPS .776 / vs_RHP .910
  - **Garrett Mitchell L** — OPS .873 / vs_RHP .826 / L7 OPS .902
  - Frelick L（OPS .573，冷）
- 主力 IL（🚨 打線核心 4 缺）：**Christian Yelich（10d）、Jackson Chourio（10d）、Andrew Vaughn（10d）、Akil Baddoo（60d）** — 打線 Strong tier 是在缺 4 名主力下達成，若 Mize 靠 vs_L 壓制進行，MIL 缺乏傳統 L 火力靠 Turang/Bauers 撐
- **總評**：Turang + Contreras + Sánchez 三人串聯 + Mitchell，對 Mize vs_R 弱點是實質威脅

### BvP 閘門
- 雙方所有打者 BvP < 15 PA → ⛔ 一律不引用 BvP 樣本

---

## 3.2 牛棚

| 側 | 季 ERA | 核心 IL | 雙向修正 |
|---|---|---|---|
| DET（home） | **4.46** | Beau Brieske（60d，高槓桿 setup）— **1 名核心** | **O/U +0.3 對 MIL；ML DET −2%** |
| MIL（away） | 4.10 | Jared Koenig（15d，LHP setup）+ Craig Yoho（15d，RP）— 約 **1 名核心** | **O/U +0.3 對 DET；ML MIL −2%** |

- DET 先發群也大量 IL（Verlander / Olson / Jobe / Melton / Brieske），Mize 後若提前下場，牛棚 bulk 負擔大
- 雙向互抵（both +0.3 opposing team），**但 DET 牛棚基底 ERA 較差 4.46 > 4.10**，MIL 打線延伸場次仍較吃香

## 3.3 條件修正

- **球場**：Comerica Park PF **99**（中性偏投手，尤其壓外野全壘打）
- **天氣**：4/22 晚間底特律 ~45-55°F 春季冷空氣（密度偏高 → 飛球壓制 → 偏 UNDER ~0.1-0.2 run）
- **Umpire / Wind**：未提供資料，不納入修正
- **TJ / 角色轉換 / 年齡**：Mize 28、Patrick 27 均 prime；無 role_change 標記；Mize 曾動 TJ（2023）但已過復出首年 → 不做 TJ 修正

## 3.4 近期狀態

- **MIL**：streak +1，昨日 12-4 大勝 DET；近 30 天 13-10（RS 5.13 / RA 3.96，**run diff +27**），last7 BABIP .325（正常），近 10 場 5-5 但 run diff 已轉正
- **DET**：streak −2（昨被爆 4-12，前二場敗 BOS），近 30 天 12-12（RS 4.33 / RA 3.96，run diff **+9**），last7 BABIP .325（正常）
- **趨勢對比**：MIL 近 30 天 offense 明顯優於 DET（RS 5.13 vs 4.33），雖 MIL 打線缺 4 主力 — 顯示 30 天數據已 baked in 缺陣狀態
- **H2H**：系列前場 4/21 MIL 12-4 DET；但 single-game，不建立 trend

## 修正後預期得分（基本面估算）

| 側 | 真實 RA 基準 | 敵打線修正 | 牛棚 | 球場/天氣 | **預期得分** |
|---|---|---|---|---|---|
| MIL vs Mize（xERA 3.62 真實） | 3.6 runs/9 | +0.3（Turang/Contreras/Sánchez vs_R 威脅） | +0.3（DET bullpen IL） | PF 99 + 冷空氣 −0.1 | **≈ 4.1** |
| DET vs Patrick（xERA 3.06 真實，K% regression 風險） | 3.1 runs/9 | +0.4（多 L 打者 vs Patrick vs_L 弱 + K% 崩跌） | +0.3（MIL bullpen IL） | PF 99 + 冷空氣 −0.1 | **≈ 3.7** |
| **Total** | | | | | **≈ 7.8** |

## 整體判斷

- **方向性**：基本面微偏 **MIL**（打線 30d run diff +27 >> DET +9；Patrick K% 崩跌但 xERA 3.06 仍比 Mize 真實 xERA 3.62 稍佳；MIL 在 Mize 弱 vs_R 側有 Contreras/Sánchez 兩張王牌），但 **MIL 打線缺 4 主力**降低上限、DET 主場且有 McGonigle 熱手與 Dingler + L 打群，優勢不明顯 → **低信心 MIL 偏向**。
- **大小盤**：盤口 8.35，基本面估算 ≈ 7.8，**偏 UNDER 0.5 左右**，但 Patrick 0.95 ERA 的 K% 崩跌若本場爆走（regression 集中） → 反向風險存在
- **讓分**：DET +1.5 主場，MIL −1.5 客場，若 MIL 贏差 ≤1 分則賭客隊讓分輸
- **值得注意的風險**：
  1. Mize vs_R 小樣本 1.263 OPS 極端值，不排除回歸 → 若 Mize 壓制右打，MIL 陣容靠 Turang 單點獨撐
  2. Patrick K% 崩跌（25.2→12.0）regression 若今晚觸發 → DET 5-6 分以上可能
  3. MIL 打線缺 Yelich/Chourio/Vaughn 三主力，長線深度不足
  4. DET 昨天 4-12 爛場後 bounce-back spot；主場 + 新秀 McGonigle（OPS .904）+ Dingler（xwOBA .458）

> 具體 ML / O/U / Run Line 推薦交由 Phase 4 `predict.py` 輸出。
