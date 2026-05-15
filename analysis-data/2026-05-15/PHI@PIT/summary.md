## 投手對決

### Braxton Ashcraft (HOME, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +11.9
  - **同意**：ERA 2.77 / xERA 2.57 / FIP 2.94 / xFIP 3.02 / K-BB% 19.4% / WHIP 1.05 — 全項一致 elite，gap +11.9 表示 ERA 微微沒運氣加成（FIP/xFIP 都更好）。本場按 🔴 Elite Ace 對待。
- **TTO3 penalty**：OPS Δ +0.006（K% Δ -12pp）— TTO3 主要是 K 率下滑、未轉成 OPS 爆發；本場 6+ IP 風險可控。
- **對手打線威脅**：🟡 中等。PHI vs RHP — Schwarber vs RHP **1.004** last7 1.409 / Harper .977 / 但 Turner .664 / García .577 / Bohm .480 — Schwarber 一棒獨大 + chain break #1-2 嚴重切斷 → 攻擊密度低。

### Aaron Nola (AWAY, RHP, 32 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p80），gap vs ERA-only = **+54.4**（極大）
  - **不同意 Elite Ace**：xFIP 3.25 表面 elite 但 ERA **5.14** / xERA 4.31 / vs LHB **.322/.404/.511**（104 BF）是真實結構弱點。tier_v2 過度看重 xFIP 而忽略 platoon execution。本場按 🟡 Solid ~ 🟢 Back-end（ERA 4.0-4.5 區間）對待。
- **TTO3 penalty 嚴重**：OPS Δ +0.142（TTO1 .868 → TTO3 **1.010**）— 第三輪極端衰退，5 IP 後危險。
- **vs LHB 弱點**：PIT 中段 Lowe (LHB) vs RHP **1.019** / O'Hearn (LHB) vs RHP .895 / Cruz (LHB) vs RHP .747 — 4 名 LHB 集中可吃 Nola。
- **對手打線威脅**：🔴 高。PIT chain break #8-9 嚴重但前 5 棒（Cruz/Reynolds/O'Hearn/Lowe/Gonzales）全 .744+ OPS vs RHP，前 5 棒爆分機率高。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致；但 vs Nola 弱點 + LHB-heavy → 上修一檔有空間。
- **chain_break 信號（🟠）**：#8-9 OPS 落差 0.282 — 後段斷層，但前 5 vs Nola 火力齊備，影響輕。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致。
- **chain_break 信號（🔴）**：#1-2 OPS 落差 **0.340**（Schwarber .965 → Turner .625）— 嚴重 chain break。
  - PHI 攻勢完全靠 Schwarber + Harper 單打，對 Ashcraft Elite Ace 難以連續施壓。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.05 / 2 / **0 名核心** | 3.94 / 3 / **0 名核心** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（PIT）：ERA 4.05 中段穩定，無核心 IL。對 PHI Schwarber/Harper 中心仍可壓制。
- AWAY 牛棚（PHI）：ERA 3.94 中段穩定，無核心 IL。後段對 PIT 中心仍 OK，但若 Nola 5 IP 內離場（TTO3 penalty 嚴重），中繼會被吃較多。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ +0.006（K% Δ -12pp）— 輕度，影響可控。
- 🟠 AWAY TTO3 penalty：OPS Δ +0.142（TTO3 OPS 1.010）— 嚴重，Nola 5 IP 後 PIT 攻勢放大。
- 🟠 HOME chain breaks at #8-9：OPS 落差 0.282 — 中度，影響輕。
- 🔴 AWAY chain breaks at #1-2：OPS 落差 0.340 — 嚴重，PHI 攻勢被切斷。

## 條件修正

- Park Factor: 102.0 → +0.10 run（PNC Park runs 中性、HR -17% 嚴重壓 HR — 利安打三壘打而非長球）
- 天氣：未公布（跳過天氣分析）
- 先發 tier：HOME Ashcraft 🔴 Elite Ace vs AWAY Nola 真實 🟡 Solid（被 xFIP 假象抬升 tier_v2）→ 嚴重不對稱，PIT 投手戰顯著優勢
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.9 | +0.3（Nola TTO3 penalty 嚴重 + vs LHB 弱點 + LHB-heavy chain） | 5.2 |
| AWAY | 3.3 | -0.2（chain break #1-2 嚴重切斷 Schwarber 後火力） | 3.1 |
| Total | 8.2 | +0.1 | 8.3 |

## 整體判斷

- **方向（基本面）**：**HOME (PIT)**。Ashcraft 真實 Elite Ace vs Nola 真實 🟡 Solid 被 xFIP 假象抬升；PIT LHB-heavy 中段精準踩 Nola vs LHB 弱點 + Nola TTO3 嚴重衰退（5 IP 後 OPS 1.010）。PHI 端 Schwarber 一棒獨大、chain break 後火力斷裂。
- **總分（基本面）**：**8.3 落點 7.5-9.0**。PIT 進攻面對 Nola 有實質優勢但 PNC HR -17% 壓制長球 + Ashcraft Elite 壓制 PHI → Total 不易爆。
- **方向信心**：**60-65%**（HOME 有利）— Nola 數據面（ERA 5.14 + vs LHB 1.044）+ TTO3 嚴重衰退 + Ashcraft Elite 是清楚的硬數據，但 Nola 仍可能在 4-5 IP 內壓制 PHI 連敗壓力下的 PIT 中心
- **風險**：
  1. Nola xFIP 3.25 是真實壓制基礎，本場可能 5 IP 1-2R 暫壓 PIT，Total 端風險
  2. Schwarber vs RHP 1.004 + last7 1.409 一棒可能改變總分結構（HR）
  3. PIT 攻勢前 4 棒 last7 BABIP 散落 .190-.444 — Reynolds 冷期 + Cruz 火燙，平均回歸方向不確定
  4. PNC 5 月夜場（未公布天氣）— 冷風 + 廣闊外野壓制長球，但 doubles 仍可拿分

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
