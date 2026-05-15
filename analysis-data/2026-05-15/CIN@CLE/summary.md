## 投手對決

### Tanner Bibee (HOME, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p80, K-BB% p73），gap vs ERA-only = +26.7
  - **謹慎同意**：ERA 4.17 / xERA 3.80 / FIP 4.31 / xFIP 3.65 / K-BB% 13.9% — 數據面是 🟡 Solid 邊緣 🟠 Strong，tier_v2 過度抬升一檔。本場按 🟡 Solid Starter（ERA 3.8-4.2 區間）對待較合理。
- **Reverse platoon（🟠）**：vs RHB **.300/.356/.425**（OPS .781）> vs LHB **.200/.290/.389**（OPS .679）— Δ +0.102。CIN 中段 RHB 多（De La Cruz switch / Stewart RHB / Steer RHB / Friedl LHB）— De La Cruz vs RHP **.837** + Stewart .755 / Steer .704 — 是攻擊點。
- **TTO3 penalty 嚴重**：OPS Δ +0.228（TTO1 .637 → TTO3 .865）— 嚴重第三輪衰退，5 IP 後危險。
- **對手打線威脅**：🟡 中等偏高。CIN matchup tier 🟢 Weak (vs RHP) 但 De La Cruz vs RHP .837 last7 **1.033**（BABIP .583 火燙警訊）是真威脅。

### Andrew Abbott (AWAY, LHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟢 Back-end Starter（xFIP p28, K-BB% p24），gap vs ERA-only = -5.9
  - **同意 🟢 Back-end**：ERA 4.47 / xERA 4.73 / FIP 4.37 / xFIP 4.49 / K-BB% **6.4%**（低）— 數據面一致中後段。本場按 🟢 Back-end（ERA 4.5+ 區間）對待。
- **Reverse platoon 嚴重（🔴 Δ +0.223）**：vs LHB **.372/.400/.512**（OPS .912）> vs RHB .245/.329/.360（OPS .689）— LHB 嚴重弱點。
  - CLE 中段 LHB 多 — Kwan (L) / Rocchio (L) — Rocchio vs LHP .804 last7 .846 是攻擊點，但 Kwan vs LHP .511 弱。實際 LHB 攻擊力有限。
- **Single-pitch dependent（🟠）**：FF 46.7% 邊緣。
- **對手打線威脅**：🟡 中等。CLE matchup tier 🟡 Average (vs LHP) — Ramírez vs LHP **.976** + DeLauter vs LHP **1.030** — 兩大威脅是 switch/LHB 都能吃 LHP，剛好踩 Abbott reverse platoon。

## 打線評級

### HOME — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟡 Average — 但 Ramírez vs LHP .976 + DeLauter 1.030 是 anchor，能踩 Abbott reverse platoon。
- **chain_break 信號（🔴）**：#8-9 OPS 落差 0.338 — 嚴重後段斷層，但前 5 棒對 Abbott 弱點配對良好。

### AWAY — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak — 比 season tier 嚴重下修兩檔（season 🟠 Strong → matchup 🟢 Weak），script 不認可 vs RHP；但 De La Cruz vs RHP .837 last7 1.033 是真實 anchor。
- **chain_break 信號（🟠）**：#4-5 OPS 落差 0.244 — 中度。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.95 / 2 / **1 名（🟠 中高）** | 4.61 / 5 / **2 名（🔴 高）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（CLE）：ERA 3.95 中段穩定，Armstrong IL15d 是 1 核心 IL → 🟠 中高。Clase (closer) 健康。後段對 CIN 中心仍有壓制。
- AWAY 牛棚（CIN）：ERA **4.61** 偏弱，Ferguson + Pagán 雙核心 IL → 🔴 高。Diaz (closer) 健康。配合 Abbott TTO3 + LHB 弱點 5 IP 內離場機率高，後段對 CLE 中心攻擊嚴重容錯不足。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.252):
  - **可能反彈 + 部分持續**：CLE 7 場樣本 BABIP .252 偏低（冷期），但對 Abbott 🟢 Back-end + 投手戰相對中性球場 → 部分反彈合理。**不自動 ±run value**，敘事上 CLE base 4.8 可能往 5.0+ 走。

### 額外信號
- 🟠 HOME reverse platoon Δ +0.102（Bibee vs RHB 弱）— CIN 多 RHB 中段，De La Cruz/Stewart/Steer 是攻擊點。
- 🔴 HOME TTO3 penalty：OPS Δ +0.228 — 嚴重，Bibee 5 IP 後 CIN 攻勢爆。
- 🔴 AWAY reverse platoon Δ +0.223（Abbott vs LHB 弱）— CLE Ramírez (switch) + Rocchio (L) 可吃，但 Kwan (L) 弱抵消部分。
- 🟠 AWAY single-pitch dependent：FF 46.7% 邊緣。
- 🔴 HOME chain breaks at #8-9：OPS 落差 0.338 — 嚴重後段。
- 🟠 AWAY chain breaks at #4-5：OPS 落差 0.244 — 中度。
- 🟠 HOME 牛棚 core IL ×1 / 🔴 AWAY 牛棚 core IL ×2 — CIN 後段崩盤級，CLE 末段攻擊放大。

## 條件修正

- Park Factor: 101.0 → +0.05 run（Progressive Field runs 中性、HR -9% 但 2024 移除外野貨櫃後 LHB HR +16%）
- 天氣：未公布（跳過天氣分析）
- 先發 tier：HOME Bibee 真實 🟡 Solid vs AWAY Abbott 🟢 Back-end → CLE 投手戰略優
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.8 | +0.4（AWAY 牛棚 ×2 核心 IL + Abbott TTO 反向 -0.236 不算太利 CLE 但 LHB 弱點 + 後段崩盤） | 5.2 |
| AWAY | 5.1 | +0.3（Bibee TTO3 嚴重 + reverse platoon RHB 弱點 + De La Cruz 火燙） | 5.4 |
| Total | 9.9 | +0.7 | 10.6 |

## 整體判斷

- **方向（基本面）**：**持平**（極微偏 AWAY）。雙弱 starter 對峙（Bibee 🟡 Solid vs Abbott 🟢 Back-end），雙方都有 reverse platoon 弱點被對手踩中（Bibee vs RHB / Abbott vs LHB）；CIN 牛棚 ×2 核心 IL 崩盤對 CLE 略有利，但 Bibee TTO3 penalty 對 CIN 末段攻擊也是好消息。base 已反映持平（4.8 / 5.1 差距小）。
- **總分（基本面）**：**10.6 偏高，落點 9.5-11.5**。雙弱 starter + 雙崩盤 reverse platoon + 雙方 chain break 但都不致命 → Total 中等偏高，Progressive Field 不算極端。
- **方向信心**：**持平 ~ 55% AWAY** — 沒有絕對方向，雙方對等配對。
- **風險**：
  1. De La Cruz last7 BABIP **.583** + OPS 1.033 — 火燙不可持續，但 vs RHP 季度 .837 真實，本場仍真威脅
  2. CLE last7 BABIP .252 冷期 — Ramírez .575 / DeLauter .656 異常冷，若反彈 CLE 進攻爆
  3. Bibee/Abbott 雙 TTO3 衰退方向不同（Bibee +0.228 / Abbott -0.236）— Abbott 第三輪反而 K% 提升可能撐 6+ IP，CIN 牛棚壓力反降
  4. Progressive Field 2024 移牆後 LHB HR +16% — CLE Kwan/Rocchio (L) 可能受益

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
