## 投手對決

### Noah Cameron (HOME, LHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p57, K-BB% p60），gap vs ERA-only = +40.7
  - **不完全同意 + 強調混合信號**：ERA **5.40** 看起來像 ⚪ Below Average 但 xERA 6.32 / FIP 4.43 / xFIP 4.02 / K-BB% 11.8% — 真實水平 🟢 Back-end ~ 🟡 Solid（4.0-4.5 ERA 區間）。+40.7 gap 主因 xFIP 改善但 xERA 6.32 警訊。本場按 🟢 Back-end 對待。
- **Reverse platoon 信號（🔴 +0.209）**：vs LHB OPS **1.010**（33 BF）vs vs RHB OPS .801（111 BF）— LHP 對 LHB 嚴重弱點。
  - DET 多 LHB 中段（Greene L #2、Carpenter L #5、Dingler R #4）— Greene vs LHP **1.053** last7 1.024 是真實爆分點，剛好踩中 Cameron 弱點。
- **TTO3 penalty（🔴 +0.213）**：第三輪 OPS 上升 +0.213，DET 中段第三輪可能爆分。
- **對手打線威脅**：🔴 高。DET matchup tier 🟢 Weak (vs LHP) — 但 Greene vs LHP 1.053 + Torkelson vs LHP .680 / McGonigle .635 / Dingler .594 — Greene 是真實 nightmare for Cameron。

### Brenan Hanifee (AWAY, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 沒給定（樣本 GS 0 / 3.7 IP 太薄 — opener 或緊急角色）。原始 tier 🟠 Strong Ace — ERA 0.00 / xERA 2.60 / FIP 2.35 / xFIP 3.33 / K-BB% 12.0% / WHIP 0.90。
  - **謹慎按 🟢 Back-end ~ 🟡 Solid 對待**：3.7 IP 樣本太薄，所有 ERA/FIP 都不可信；vs LHB 5 BF / vs RHB 20 BF — 樣本太薄無有效對照。本場可能 2-4 IP 後接力。
- **Flag 8 era_xera_delta=-2.60**：嚴重運氣加持（ERA 0.00 但 xERA 2.60），樣本太薄不可信任一方向。
- **單一球種 SI 59.8%**：球種組合不健全，KC RHB 多可能 sit sinker。
- **對手打線威脅**：🟡 中等。KC matchup tier 🟡 Average (vs RHP) — Witt vs RHP **.798** last7 1.150（火燙）/ Garcia .646 / Pasquantino .728 / Jensen .769 — Witt 是真實爆分點。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致；Witt last7 1.150 火燙。
- **chain_break 信號（🟠 #2-3）**：Garcia .756 → Perez .555 — 中度，但 Witt #1 強撐前段。

### AWAY — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak — 比 season tier 下修；但 Greene vs LHP 1.053 火燙。
- **chain_break 信號（🟠 #7-8）**：影響輕。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | **4.77** / 5 / **1 名（Estévez closer）** | 3.87 / 10 / **3 名（Brieske + Melton + 1 other）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（KC）：ERA **4.77** 偏弱 + 1 核心 IL（Estévez closer）→ 🟠 中高。後段對 DET 中段壓制力中等，特別 Greene/Torkelson 末段反咬可能。
- AWAY 牛棚（DET）：ERA 3.87 中段稍弱 + **3 核心 IL → 🔴🔴 極高**（牛棚崩盤級）！KC 中段 (Witt last7 1.150) 末段攻擊極大化關鍵。

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-2.60):
  - **小樣本不可信任一方向**：Hanifee 3.7 IP 太薄，ERA 0.00 是樣本失真；FIP 2.35 看起來精彩但無實際根據。本場按 🟢 Back-end 對待，**不自動下修**。

### 額外信號
- 🔴 HOME reverse platoon Δ +0.209 — Cameron vs LHB 1.010 是真實 nightmare；DET Greene vs LHP 火燙踩中。
- 🔴 HOME TTO3 +0.213 — Cameron 第三輪壓制力下降，DET 中段反彈關鍵。
- 🟠 AWAY single-pitch SI 59.8% — Hanifee 球種組合不健全。
- 🟠 雙方 chain breaks — 影響輕。
- 🟠 HOME 牛棚 1 核心 IL — 影響中等。
- 🔴 AWAY 牛棚 3 核心 IL — KC 末段攻擊極大化關鍵。

## 條件修正

- Park Factor: 106.0 → +0.30 run（Kauffman runs 106 但 HR -9% 利安打/三壘打）
- 天氣：未公布（跳過天氣分析）
- 先發 tier：HOME Cameron 🟢 Back-end vs AWAY Hanifee 樣本零 → 雙弱
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 2.7 | +0.7（AWAY 牛棚 3 核心 IL → KC 末段攻擊極大化） | 3.4 |
| AWAY | 5.4 | 0（核心 IL 1 名，未達 ≥2 門檻） | 5.4 |
| Total | 8.1 | +0.7 | 8.8 |

## 整體判斷

- **方向（基本面）**：**AWAY (DET) 略有利**。Cameron reverse platoon vs LHB + Greene vs LHP 1.053 → DET 進攻面有空間。但 base AWAY 5.4 / HOME 2.7 嚴重不平衡 — DET 連敗 5 場 + 信心面崩 → 實際可能比 base 低。KC last7 4-3 + Witt last7 1.150 強，主場優勢 + DET 牛棚 3 核心 IL 是 KC 反彈助力。整體微偏 DET 投手戰但 KC 攻擊面強。
- **總分（基本面）**：**8.8（base 8.1 + +0.7 信號）**，落點 7.5-10.0。雙弱 starter + DET 牛棚崩 + Cameron vs LHB 弱點 + Witt 火燙 → Total 上行；Kauffman 利安打但壓 HR。
- **方向信心**：~52%（AWAY），微偏；DET 連敗 5 信心面對立 + base 嚴重不平衡。
- **風險**：
  1. **DET 連敗 5 場 + last7 cold heat** — 信心面崩盤，可能延續壓制
  2. Hanifee 3.7 IP 樣本 — 可能任一方向結果，KC 進攻面 base 2.7 偏低
  3. Greene vs LHP 1.053 last7 1.024 — 真實爆分點，可能單棒打破
  4. KC 牛棚 ERA 4.77 + Estévez IL — 若 Cameron 早下，DET 末段可能反咬

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
