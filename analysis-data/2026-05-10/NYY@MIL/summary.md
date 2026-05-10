## 投手對決

### Logan Henderson (HOME, RHP, 24 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 沒給定（樣本 GS 2 / 8 IP 太薄）。原始 tier 🟢 Back-end — 但實際 ERA 4.50 / xERA **1.45** / FIP **0.73** / xFIP 2.19 / K-BB% **33.4%** — 數據看起來像 🔴 Elite Ace **但樣本太薄**。
  - **謹慎按 🟠 Strong Ace 對待**：8 IP 樣本，xERA 1.45 是樣本失真不可信；但 K-BB% 33.4% / FIP 0.73 / 0.0 barrel% 都是真實精彩。本場可能 4-5 IP 後接力。
- **Flag 8 era_xera_delta=+3.05**：ERA 比 xERA 高 3.05，這次是 ERA 高估（不幸），xERA 1.45 過於極端不可信，FIP 0.73 也不可信。
- **vs LHB 強壓制（.182/.182/.318，22 BF）**：對 LHB 嚴格壓制；NYY 多 LHB 中段（Bellinger L #4、Chisholm L #5、Rice S #2）— 部分被壓制。
- **對手打線威脅**：🟠 高。NYY matchup tier 🟡 Average (vs RHP) — 但 Judge vs RHP **1.006** last7 1.130 / Rice 1.120 / Bellinger .871 last7 1.053 — 前 4 棒對小樣本投手是真實威脅。

### Carlos Rodón (AWAY, LHP, 33 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 沒給定（GS None — 可能傷後復出或 opener 角色）。NYY 季初 Rodón IL 中，本場可能首場復出。
  - **無數據判讀**：32 歲 LHP，2025 季前完整大谷一樣依賴控球；2026 季復出狀態未知。本場按 🟢 Back-end ~ 🟡 Solid 對待，依賴歷史 4.0+ ERA 區間。
- **對手打線威脅**：🟠 高。MIL matchup tier 🟡 Average (vs LHP) — Sánchez vs LHP **1.305**（小樣本）/ Vaughn 1.000 / Mitchell .845 / Perkins .785 — 對 LHP 火力分散但 Vaughn 是真實火燙。

## 打線評級

### HOME — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟡 Average — 比 season tier 下修；但 Chourio season **1.085** / Turang season .908 / Vaughn 1.000 vs LHP — 中段火力強。
- **chain_break 信號（🔴 #5-6）**：Vaughn .922 → Rengifo .516 — 嚴重，但 Vaughn vs LHP 1.000 補強；Rodón 後段壓制力可能下降。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致；Rice/Judge/Bellinger 中心強。
- **chain_break 信號（🔴 #7-8）**：Caballero .709 → Jones .167 — 嚴重，但 NYY 前 4 棒（Grisham/Rice/Judge/Bellinger）vs Henderson 火力齊備。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.46 / 5 / **2 名（Zerpa + Koenig）** | 3.29 / 3 / **0 名核心** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（MIL）：ERA 3.46 中段穩定 + **2 核心 IL 🔴 高**（Zerpa + Koenig 雙缺陣，都是 high-leverage LHP）→ NYY LHB 末段攻擊放大。
- AWAY 牛棚（NYY）：ERA 3.29 中段強，無核心 IL。Williams (closer) 健康。對 MIL 中心壓制力高。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=+3.05):
  - **小樣本 + 數據極端**：Henderson 8 IP 樣本中，所有 ERA/xERA/FIP 都失真。本場按 🟠 Strong Ace 對待，承認 K-BB% 33.4% 是真實精彩但 ERA 4.50 也是真實爆掉過。**不自動下修**。

### 額外信號
- 🟠 HOME single-pitch dependent FF 47.4% — Henderson FF-heavy；但 K-BB% 33.4% 顯示球質強。
- 🟠 AWAY TTO3 penalty +0.028（career fallback）— 影響輕。
- 🔴 HOME chain break #5-6 — MIL 後段斷層，限制 MIL 上限。
- 🔴 AWAY chain break #7-8 — NYY 後段斷層，影響輕。
- 🔴 HOME 牛棚 2 核心 IL — NYY 末段攻擊放大關鍵。

## 條件修正

- Park Factor: 97.0 → -0.15 run（American Family Field runs 97 但 HR +11%）
- 天氣：Partly Cloudy 60°F, wind 14 mph **Out To LF** — 強順風左外野推 RHB pull HR（Judge/Sánchez RHB）→ 強推升 HR
- 先發 tier：HOME Henderson 樣本零 + 數據極端 vs AWAY Rodón 復出 → 雙弱可信度
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.2 | 0（核心 IL 0 名） | 5.2 |
| AWAY | 0.8 | +0.6（HOME 牛棚 2 核心 IL → NYY 末段攻擊極大化） | 1.4 |
| Total | 6.0 | +0.6 | 6.6 |

## 整體判斷

- **方向（基本面）**：**HOME (MIL) 中度有利 — 雖然 base 數據與基本面衝突**。base AWAY 0.8 是嚴重低估（NYY Judge/Rice 強打 + Rodón 復出 + MIL 牛棚 2 核心 IL）→ 實際 AWAY 應該 3.0+ runs。但 MIL 連勝 3 + 主場優勢 + Henderson K-BB% 真實精彩 → 微偏 HOME。
- **總分（基本面）**：**6.6 嚴重偏低，落點 7.0-9.5**。base 嚴重低估 NYY 進攻能力（Judge 1.006 vs RHP）+ 強順風 HR + Rodón 復出狀態未知 → Total 上行壓力大。
- **方向信心**：~55%（HOME），微偏；雙方先發都極端不確定。
- **風險**：
  1. **Henderson 樣本太薄（8 IP）**：可能本場打出 5 IP 0R 或 4 IP 5R 兩極結果
  2. Rodón 復出 — 復出狀態未知，可能 4 IP 4R 或 5 IP 1R
  3. Out To LF 強順風 + Judge RHB pull power — Judge 單棒可能 2 HR 推升 NYY 得分
  4. base AWAY 0.8 嚴重低估 — formula 可能受 Henderson xERA 1.45 樣本失真誤導，實際 NYY 進攻面遠高於此

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
