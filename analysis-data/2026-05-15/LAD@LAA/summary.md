## 投手對決

### Jack Kochanowicz (HOME, RHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟢 Back-end Starter（xFIP p29, K-BB% p12），gap vs ERA-only = -21.1
  - **同意 🟢 Back-end + ERA 高估**：ERA 3.97 看起來 🟡 Solid 但 xERA **4.99** / FIP 3.87 / xFIP 4.47 / K-BB% **3.5%**（極端低）/ whiff% 10.4 — 真實水平 🟢 Back-end ~ ⚪ Below Average（ERA 4.5-5.0 區間）。-21.1 gap 主因運氣。
  - **本場按 🟢 Back-end** 對待。
- **TTO 反向**：OPS Δ -0.060 — Kochanowicz 越投越穩，但 K-BB% 3.5% 表示靠 GB 球種，TTO3 K% 掉到 14.5% 是 risk。
- **對手打線威脅**：🔴 高。LAD matchup tier 🟠 Strong (vs RHP) — Pages vs RHP **.894** + Muncy vs RHP **.904** + Freeman .788 + Ohtani .733 last7 .670 + Tucker .731 last7 .914 — 前 5 棒全 .700+ OPS，對 Kochanowicz K-BB% 3.5% 是 nightmare。

### Blake Snell (AWAY, LHP, 33 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 沒給定（GS 1 / 3 IP 樣本極小）。原始 tier ⚪ Below Average — ERA **12.00** 是樣本失真，FIP **1.77** / xFIP 2.20 / K-BB% 16.7% / whiff% **20.8%** / hard_hit% **13.6%**（極低）/ barrel% **0.0%** — Statcast 接觸品質頂級。
  - **強烈不同意 ERA-only ⚪**：Snell 是 LAD 季初 IL 復出（傷後）GS 1，ERA 12.00 是極端小樣本（3 IP）失真。Snell career 真實水平 🟠 Strong Ace ~ 🔴 Elite Ace；Flag 8 era_xera_delta **+8.28** 是樣本噪音不是結構性。
  - **本場按 🟠 Strong Ace ~ 🔴 Elite Ace**（ERA 3.0-3.5 區間）對待，預期 4-5 IP 復出限制 IP。
- **vs LHB 樣本 6 BF / vs RHB 12 BF**：完全無 platoon 樣本，沿用 career baseline。
- **對手打線威脅**：🟢 低。LAA matchup tier 🟢 Weak (vs LHP) — Trout vs LHP .867 last7 .489 / Neto vs LHP .918 / Adell vs LHP **.999** last7 1.066（BABIP .357 火燙）— Adell 是 anchor，但 Snell career vs LHB/RHB 都壓制。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak — 比 season tier 下修一檔；但 Adell vs LHP last7 1.066 / Neto .918 個別 anchor 真實。
- **chain_break 信號（🟠）**：#6-7 OPS 落差 0.207 — 中度。

### AWAY — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong — 與 season tier 一致；前 5 棒（Tucker/Ohtani/Freeman/Pages/Muncy）vs RHP 全 .700+。對 Kochanowicz K-BB% 3.5% 是 dream matchup。
- **chain_break 信號（🟠）**：#5-6 OPS 落差 0.171 — 中度。
- **Flag 3 last7 BABIP .256** — 冷期（見風險段）。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | **5.27** / 5 / **1 名（🟠 中高，Pomeranz IL15d）** | 3.52 / 10 / **2 名（🔴 高，Stewart + Díaz IL60d）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（LAA）：ERA **5.27** 嚴重崩盤（聯盟均 ~3.90），Pomeranz IL15d 是 1 核心 IL → 🟠 中高。配合 Kochanowicz 真實 🟢 Back-end + K-BB% 3.5% 預期 5 IP 內離場，LAA 整場後 5-6 IP 牛棚崩盤對 LAD 中心是 nightmare。
- AWAY 牛棚（LAD）：ERA 3.52 中段穩定，**2 名核心 IL**（Stewart + Díaz）→ 🔴 高。Sasaki/Phillips 接 setup/closer 健康。配合 Snell 復出限制 IP（4-5 IP），LAD 後段對 LAA 弱進攻仍可壓制。

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=+8.28):
  - **嚴重小樣本噪音**：Snell GS 1 / 3 IP / ER 4 是極端小樣本失真，FIP 1.77 + Statcast 接觸品質頂級（barrel% 0.0%）是真實水平。**不自動下修預測**，本場按 🟠 Strong Ace ~ 🔴 Elite Ace 對待。
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.256):
  - **可能反彈 + 部分持續**：LAD 7 場樣本 BABIP .256 偏低 + 中心 last7 OPS 散亂（Pages .520 BABIP .105 / Muncy .643 BABIP .154 冷期）— 對 Kochanowicz 弱投手 + LAA 牛棚崩盤，反彈空間大。**不自動 ±run value**，base 4.8 可能往 5.5+ 走。

### 額外信號
- 🟠 HOME TTO3 反向（-0.060）— Kochanowicz 越投越穩，但 K-BB% 3.5% 撐 6 IP 困難。
- 🟠 AWAY TTO3 penalty：OPS Δ +0.070 — Snell career 第三輪輕度衰退，本場復出限制 4-5 IP 應該避免 TTO3。
- 🟠 HOME chain breaks at #6-7：OPS 落差 0.207 — 中度。
- 🟠 AWAY chain breaks at #5-6：OPS 落差 0.171 — 中度。
- 🟠 HOME 牛棚 core IL ×1：🟠 中高 + ERA 5.27 崩盤 — LAA 後段對 LAD 中心是 nightmare。
- 🔴 AWAY 牛棚 core IL ×2：🔴 高 — LAD 後段對 LAA 弱進攻仍 OK。

## 條件修正

- Park Factor: 101.0 → +0.05 run（Angel Stadium 中性，HR +5%）
- 天氣：未公布（跳過天氣分析）
- 先發 tier：HOME Kochanowicz 真實 🟢 Back-end vs AWAY Snell 真實 🟠 Strong Ace（復出 IP 受限）→ AWAY 投手戰嚴重優勢
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 2.0 | 0（Snell 真實 🟠 Strong Ace 壓制 LAA 弱進攻）| 2.0 |
| AWAY | 4.8 | +0.5（LAA 牛棚 ERA 5.27 崩盤 + Kochanowicz K-BB% 3.5% + LAD 前 5 棒 vs RHP 強）| 5.3 |
| Total | 6.8 | +0.5 | 7.3 |

## 整體判斷

- **方向（基本面）**：**AWAY (LAD)**。Snell 真實 🟠 Strong Ace（ERA 12.00 是 3 IP 樣本噪音）vs Kochanowicz 真實 🟢 Back-end + K-BB% 3.5%；LAD 前 5 棒（Pages/Muncy/Tucker/Ohtani/Freeman）對 Kochanowicz 是 dream matchup + LAA 牛棚 ERA 5.27 崩盤 — 投手戰 + 進攻雙優。
- **總分（基本面）**：**7.3 接近實際，落點 6.5-8.5**。Snell 壓制 LAA 弱進攻 + Kochanowicz 雖然弱但 LAA 牛棚崩盤後段 LAD 攻擊放大 → Total 中等。
- **方向信心**：**65-70%**（AWAY 有利）— LAA 三重利空（弱投手 + 牛棚崩盤 + 連敗 3 攻↓）+ LAD 中心對 RHP 強。Snell 復出限制 IP 是唯一風險（4-5 IP 後 LAD 牛棚要扛 4-5 IP）。
- **風險**：
  1. Snell 復出 IP 限制（預期 4-5 IP）— LAD 牛棚要扛較多 IP，雖然 ERA 3.52 但有 2 核心 IL
  2. Adell vs LHP **.999** last7 1.066（BABIP .357）— Adell 是 LAA 真威脅，可能 HR
  3. Snell ERA 12.00 是極端值 — 雖然 FIP 1.77 但本場可能繼續波動
  4. LAA 連敗 3 + 攻↓ — 心理面壓力大，base 2.0 可能準確（接近 shut out）

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
