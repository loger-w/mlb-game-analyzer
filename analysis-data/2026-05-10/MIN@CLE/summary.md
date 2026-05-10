## 投手對決

### Gavin Williams (HOME, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +22.7
  - **同意 + 強調 ERA 低估真實水平**：ERA 3.28 / xERA 4.26 / FIP **4.05** / xFIP **2.95** / K-BB% **19.7%** / WHIP 1.09 — xFIP/K-BB% 都 elite 但 FIP/xERA 偏高（barrel% **14.7%** 警訊）。本場按 🟠 Strong Ace 對待較合理（混合信號）。
- **vs LHB 強壓制（.173/.248/.382，121 BF）**：對 LHB 嚴格壓制；vs RHB .212/.325/.348 也不錯但 OBP .325 控球差。
- **TTO3 penalty（-0.308）**：第三輪 OPS 反向下降，TTO 影響不大。
- **對手打線威脅**：🟡 中等。MIN matchup tier 🟡 Average (vs RHP) — Buxton vs RHP **1.007** last7 .991 / Jeffers .967 last7 1.059 / Larnach .848 — Buxton + Jeffers 是真實爆分點，barrel% 高擊球品質。

### Andrew Morris (AWAY, RHP, 24 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 沒給定（樣本 GS 0 / 7 IP 太薄 — opener 或緊急角色）。原始 tier 🟢 Back-end — ERA 4.96 / xERA 3.33 / FIP 2.98 / xFIP 3.85 / K-BB% 13.7%。
  - **謹慎按 🟢 Back-end 對待**：xERA 3.33 / FIP 2.98 看起來 Strong 但樣本太薄；vs RHB **.344/.400/.344**（36 BF）顯示控球嚴重崩。本場可能 3-4 IP 後接力。
- **Flag 8 era_xera_delta=+1.63**：ERA 比 xERA 高，可能不幸或結構問題。樣本太薄，按平均對待。
- **對手打線威脅**：🟠 高。CLE matchup tier 🟢 Weak (vs RHP) — DeLauter vs RHP **.841** last7 1.179 / Schneemann .777 / Ramírez .657 — DeLauter 是真實爆分點，其他人 vs RHP 中段。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak — 比 season tier 下修一檔；DeLauter 之外整體弱。
- **chain_break 信號（🟠 #7-8）**：Bazzana .645 → Bailey .396 — 中度後段斷層。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致；Buxton/Jeffers 中心強。
- **chain_break 信號（🔴 #3-4）**：Jeffers .928 → Bell .605 — 嚴重斷層；Bell vs RHP .486 是黑洞，Buxton/Jeffers 之後攻勢被切。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.07 / 2 / **1 名（Armstrong IL15d）** | **5.60** / 7 / **1 名（Sands IL15d）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（CLE）：ERA 4.07 中段稍弱，1 核心 IL（Armstrong setup）→ 🟠 中高。Clase (closer) 健康。對 MIN Buxton 等末段壓制力中等。
- AWAY 牛棚（MIN）：ERA **5.60** 嚴重崩盤 + 1 核心 IL（Sands）→ 🟠 中高。對 CLE 中心（DeLauter last7 1.179）末段反咬機率高。

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=+1.63):
  - **樣本不足**：Morris 7 IP 樣本太薄，xERA/FIP 看起來不差但實際對 LHB/RHB 雙邊都被打。本場按 🟢 Back-end 對待。**不自動下修**。

### 額外信號
- 🟠 HOME chain break #7-8 — CLE 後段熄火。
- 🔴 AWAY chain break #3-4 — MIN Bell #4 拖累中心。
- 🟠 雙方牛棚 1 核心 IL — 影響中等。

## 條件修正

- Park Factor: 101.0 → +0.05 run（Progressive Field 中性，HR -9% 但 2024 後改造可能反向）
- 天氣：Sunny **55°F**, wind 6 mph **In From CF** — 涼風 + 逆風中外野壓 HR → 顯著壓 Total
- 先發 tier：HOME Williams 🟠 Strong Ace ~ Elite Ace vs AWAY Morris 🟢 Back-end 樣本零 → 嚴重不對稱
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.4 | 0（核心 IL 1 名，未達 ≥2 門檻） | 3.4 |
| AWAY | 4.5 | 0（核心 IL 1 名，未達 ≥2 門檻） | 4.5 |
| Total | 7.9 | 0 | 7.9 |

## 整體判斷

- **方向（基本面）**：**AWAY (MIN) 略有利**。Williams Elite Ace 但 barrel% 14.7% + Buxton/Jeffers 強打 → CLE 進攻面優勢被中和。Morris 樣本零 + MIN 牛棚 ERA 5.60 → MIN 投手戰嚴重不利。整體偏 MIN 但雙方都可能爆分或被壓制。
- **總分（基本面）**：**7.9 接近實際但偏低，落點 6.5-9.0**。Williams 強壓制 + 涼風逆風壓 HR → Total 下行；但 Morris 樣本不確定 + MIN 牛棚崩 → 下半場可能爆分。
- **方向信心**：~55%（AWAY），結構性弱（雙方都不確定）。
- **風險**：
  1. Morris 7 IP 樣本太薄 — 本場可能任一方向
  2. MIN 牛棚 ERA 5.60 — 若 Morris 早下，CLE 中心可能爆分末段
  3. Buxton last7 .991 + Jeffers 1.059 火燙 — 真實爆分點，可能單棒打破投手戰
  4. Progressive Field 5 月初 55°F 涼風逆風 — Total 下行明顯，HR 機率被壓

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
