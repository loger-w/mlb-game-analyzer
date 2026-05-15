## 投手對決

### Spencer Strider (HOME, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 沒給定（GS 2 復出 / 9.3 IP 樣本不足）。原始 tier 🟠 Strong Ace — ERA 2.89 / xERA 3.30 / FIP 3.74 / xFIP 3.46 / K-BB% 17.5% / velo 88.9（max 97.6）/ whiff% 13.5 / hard_hit% **18.4%**（極低）— 數據面真實 elite 接觸壓制。
  - **同意 🟠 Strong Ace**：Strider TJ 復出後第 2 場（GS 2），velo 從巔峰 99 mph 掉到 88.9 平均（max 97.6 復速復出中）— 但 K-BB% 17.5% / hard_hit% 18.4% 真實壓制力 OK。本場按 🟠 Strong Ace 對待但保留樣本不確定性。
- **Single-pitch dependent（🟠）**：FF 48.3% — 但 SL 25.8% 補強雙球種威脅。
- **TTO3 penalty（career fallback）**：OPS Δ +0.084（K% Δ -5.9pp） — 輕度衰退，6 IP 內 OK。
- **對手打線威脅**：🟢 低。BOS matchup tier 🟢 Weak (vs RHP) — Abreu vs RHP .836 / Contreras .747 / Story .540 — Abreu 是唯一 anchor，BOS last7 BABIP .225 冷期 + Strider 強壓制 → 進攻接近 shut out。

### Connelly Early (AWAY, LHP, 24 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p67, K-BB% p60），gap vs ERA-only = -3.8
  - **謹慎同意**：ERA 3.16 / xERA **4.70** / FIP 4.06 / xFIP 3.88 / K-BB% 11.8% / hard_hit% 26.8 / barrel% 12.1（偏高）— xERA 比 ERA 高 1.54 是 Flag 8 警訊。本場按 🟡 Solid（ERA 4.0+ 區間）對待，xFIP 3.88 是真實基礎但接觸品質有結構問題。
- **TTO 反向**：OPS Δ **-0.210**（TTO1 .728 → TTO3 .518）— 第三輪反而 K% 提升，Early 越投越穩，可撐 6+ IP。
- **vs LHB 強壓制**：vs LHB .179/.250/.333 / vs RHB .235/.338/.365 — 雙邊都 OK，無 reverse。
- **對手打線威脅**：🔴 高。ATL matchup tier 🟢 Weak (vs LHP) 但個別 Baldwin vs LHP **.984** / Olson .905 / Albies **.914** — 中心 3 棒全是 LHP killer，雖然 last7 OPS 散亂（Albies .212 冷期）但季度數據真實。

## 打線評級

### HOME — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak — script 認為 ATL 對 LHP 不行，但中心 Baldwin/Olson/Albies 個別都 .900+ vs LHP，矛盾。應上修一檔。
- **chain_break 信號（🟠）**：#6-7 OPS 落差 0.250 — 中度，中心 1-4 棒 vs LHP 火力齊備，影響輕。

### AWAY — season tier 🟢 Weak / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak — 與 season tier 一致。對 Strider 強壓制 → 接近 shut out。
- **chain_break 信號（🔴）**：#1-2 OPS 落差 **0.313**（Abreu .860 → Story .547）— 嚴重，Abreu 之後 Story 黑洞，BOS 攻擊完全靠 Abreu 一棒。
- **Flag 3 last7 BABIP .225** — 冷期（見風險段）。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.10 / 7 / **1 名（🟠 中高）** | 3.25 / 6 / **1 名（🟠 中高）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（ATL）：ERA 3.10 elite，1 核心 IL（Young）→ 🟠 中高。Suarez (closer) 健康。後段對 BOS 弱進攻仍是壓制。
- AWAY 牛棚（BOS）：ERA 3.25 同樣 elite，1 核心 IL（Coulombe）→ 🟠 中高。Chapman (closer) 健康。後段對 ATL 中心仍有壓制力。

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-1.54):
  - **小樣本 + 接觸品質弱**：Early ERA 3.16 vs xERA 4.70 gap +1.54，barrel% 12.1 偏高顯示接觸品質結構問題。但 xFIP 3.88 是真實基礎，本場按 🟡 Solid 對待，**不自動下修**。
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.225):
  - **可能反彈 + 嚴重持續**：BOS 7 場樣本 BABIP .225 + heat Cold + matchup tier 🟢 Weak vs Strider → 三重壓力，本場仍難反彈。**不自動 ±run value**，敘事上 BOS base 3.8 可能往 3.0 走。

### 額外信號
- 🟠 HOME single-pitch dependent：FF 48.3% — SL 補強，影響輕。
- 🟠 HOME TTO3 penalty：OPS Δ +0.084 — 輕度。
- 🟠 HOME chain breaks at #6-7：OPS 落差 0.250 — 中度。
- 🔴 AWAY chain breaks at #1-2：OPS 落差 0.313 — 嚴重，BOS 攻勢只剩 Abreu 一棒。
- 🟠 雙方牛棚 core IL ×1 — 都中高但 starter 預期都撐 6+ IP，影響輕。

## 條件修正

- Park Factor: 98.0 → -0.10 run（Truist Park 中性偏輕度投手友善，HR -5%）
- 天氣：未公布（跳過天氣分析）
- 先發 tier：HOME Strider 🟠 Strong Ace（復出小樣本）vs AWAY Early 🟡 Solid → HOME 投手戰優勢
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.7 | +0.2（ATL 中心 Baldwin/Olson/Albies vs LHP 強，雖 matchup tier Weak 但個別 anchor 真實） | 4.9 |
| AWAY | 3.8 | -0.3（chain break #1-2 嚴重 + Strider 接觸壓制 + BOS cold + Truist 投手友善） | 3.5 |
| Total | 8.5 | -0.1 | 8.4 |

## 整體判斷

- **方向（基本面）**：**HOME (ATL)**。Strider 接觸壓制 (hard_hit% 18.4%) + BOS 三重壓力（chain break #1-2 / heat Cold / matchup Weak vs RHP）vs Early 真實 🟡 Solid + ATL 中心 vs LHP 強配對。base 已偏 ATL（4.7 vs 3.8），實際差距類似。
- **總分（基本面）**：**8.4 接近實際，落點 7.5-9.0**。Strider vs Early 雙偏穩定 starter + Truist 中性偏投手 + 雙方牛棚 ERA 3.10/3.25 elite → Total 中等略低。
- **方向信心**：**65%**（HOME 有利）— BOS 攻擊面三重利空 + ATL 中心 anchor vs LHP 強配對，但 Strider 9.3 IP 復出樣本不足是變數。
- **風險**：
  1. Strider TJ 復出 GS 2 — velo 88.9 平均（巔峰 99+）顯示尚未復速，K-BB% 17.5% 撐住但本場可能波動
  2. ATL 連敗 1 + 攻↓ 連 10 RS 4.10（vs 季 5.33）— 中心打序 last7 散亂（Albies .212）整體進攻可能不如 base
  3. Early GS 2 樣本 — 可能 5 IP 1-2R 反常壓制 ATL，但 ATL 季度 vs LHP 強
  4. BOS Abreu 單打獨鬥 — 若 Strider 處理掉 Abreu，BOS 進攻完全熄火，但若 Abreu 爆分 BOS 仍有空間

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
