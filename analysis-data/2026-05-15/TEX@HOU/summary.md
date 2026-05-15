## 投手對決

### Spencer Arrighetti (HOME, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 沒給定（GS 5 / 18 IP 樣本不足）。原始 tier 🟠 Strong Ace 是 ERA 1.88 假象。FIP 3.24 / xFIP 4.19 / xERA **5.40** / K-BB% **10.4%** / velo **84.2**（極低）/ barrel% 6.8 — 數據面真實是 🟡 Solid ~ 🟢 Back-end（ERA 4.0-4.5 區間）。
  - **不同意 Strong Ace**：Flag 8 era_xera_delta **-3.52** 是極端運氣警訊（BABIP 偏低 + LOB% 偏高），xERA 5.40 顯示接觸品質弱。本場按 🟢 Back-end ~ 🟡 Solid 對待。
- **TTO3 penalty 嚴重**：OPS Δ +0.166（TTO1 .582 → TTO3 .748）— 5 IP 後危險。
- **對手打線威脅**：🟠 高。TEX matchup tier 🟡 Average (vs RHP) — Seager .703 / Nimmo .777 / Jung vs RHP **.937** — Jung 是 anchor，TEX 中段對 Arrighetti 接觸品質弱點配對良好。

### Jack Leiter (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p83），gap vs ERA-only = **+53.1**（極大）
  - **不完全同意**：xFIP 3.26 / K-BB% 16.6% 看起來 Elite 但 ERA 4.85 / xERA 4.61 / FIP 4.20 / barrel% **11.9%**（偏高）/ vs LHB OPS **.825** 弱點 — 真實水平 🟡 Solid（ERA 4.0+ 區間）。tier_v2 過度看重 xFIP。
  - **本場按 🟡 Solid Starter**（ERA 3.5-4.5 區間）對待。
- **TTO3 penalty**：OPS Δ +0.129（K% Δ -3.4pp） — 中度第三輪衰退。
- **vs LHB 弱點**：HOU 中段 Walker (R) 為主力，但 Alvarez (L) vs RHP **1.082** + Smith (R)；剛好踩 Leiter vs LHB OPS .825 中的 Alvarez。
- **對手打線威脅**：🟠 高。HOU matchup tier 🟠 Strong (vs RHP) — Alvarez vs RHP **1.082** last7 1.134（BABIP .500 火燙）+ Walker .884（last7 .374 冷期 BABIP .050）— Alvarez 是真威脅，Walker 反彈空間大。

## 打線評級

### HOME — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong — 比 season tier 上修一檔；Alvarez/Walker 是真實 anchor 對 Leiter 中等水平。
- **chain_break 信號（🔴）**：#8-9 OPS 落差 **0.512** — 極端後段斷層，但 HOU 前 5 棒火力齊備，影響輕。
- **Flag 3 last7 BABIP .254** — 冷期。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致；Jung anchor 真實。
- **chain_break 信號（🟠）**：#4-5 OPS 落差 0.248 — 中度。
- **Flag 3 last7 BABIP .241** — 冷期。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | **5.86** / 8 / **1 名（🟠 中高，Hader IL60d）** | 3.07 / 6 / **0 名核心** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（HOU）：ERA **5.86** 嚴重崩盤（聯盟均 ~3.90）— Hader (closer) IL60d 是核心缺陣。配合 Arrighetti TTO3 5 IP 後離場機率高 → HOU 整場後 5-6 IP 牛棚崩盤級風險。
- AWAY 牛棚（TEX）：ERA **3.07** elite + 無核心 IL → 完整可用。後段對 HOU 中心 (Alvarez/Walker) 仍是真實壓制。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-3.52):
  - **極端運氣加持**：Arrighetti ERA 1.88 vs xERA 5.40 gap **-3.52** 是本場最大警訊；xERA 5.40 / barrel% 6.8 / hard_hit% 20.7 顯示接觸壓制有限。本場按 🟢 Back-end（ERA 4.0+）對待，TEX 進攻基準應該偏高 — base AWAY 3.4 可能往 4.5+ 走。
- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.254):
  - **可能部分反彈**：HOU 冷期，Walker last7 .374 + BABIP .050 + 整體攻↓ 連 10 RS 2.80 — 但 Leiter 中等水平 RHP，部分反彈合理。**不自動 ±run value**。
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.241):
  - **可能反彈**：TEX 冷期，Seager last7 .138 / BABIP .000 是極端冷，但 Jung vs RHP .937 anchor 真實。**不自動 ±run value**。

### 額外信號
- 🔴 HOME TTO3 penalty：OPS Δ +0.166 — Arrighetti 5 IP 後 TEX 攻勢爆。
- 🟠 AWAY TTO3 penalty：OPS Δ +0.129 — Leiter 第三輪輕度衰退。
- 🔴 HOME chain breaks at #8-9：OPS 落差 0.512 — 極端但 HOU 前 5 棒齊備。
- 🟠 AWAY chain breaks at #4-5：OPS 落差 0.248 — 中度。
- 🟠 HOME 牛棚 core IL ×1：🟠 中高 — Hader 缺陣 + 整體 ERA 5.86 是雙重崩盤。

## 條件修正

- Park Factor: 98.0 → -0.10 run（Daikin Park 中性，HR +2%）
- 天氣：室內球場（無天氣修正）
- 先發 tier：HOME Arrighetti 真實 🟢 Back-end vs AWAY Leiter 真實 🟡 Solid → AWAY 投手戰略優
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.5 | -0.2（TEX 牛棚 ERA 3.07 elite 壓制 + Leiter vs LHB 弱點但 Alvarez 一人） | 4.3 |
| AWAY | 3.4 | +0.5（HOU 牛棚 ERA 5.86 崩盤 + Hader IL + Arrighetti TTO3 嚴重 + Jung 真實 anchor） | 3.9 |
| Total | 7.9 | +0.3 | 8.2 |

## 整體判斷

- **方向（基本面）**：**AWAY (TEX)**。Leiter 真實 🟡 Solid + TEX 牛棚 ERA 3.07 elite vs Arrighetti 真實 🟢 Back-end + HOU 牛棚 ERA 5.86 崩盤 + Hader IL — 投手戰 + 牛棚雙優。雖然 base 偏 HOU（4.5 vs 3.4）反映 Arrighetti ERA 假象，實際 TEX 略有利。
- **總分（基本面）**：**8.2 接近實際，落點 7.0-9.0**。雙弱 starter（被 ERA 假象掩蓋）+ HOU 牛棚崩盤 + Daikin 室內中性 → Total 中等。Alvarez 一棒可能爆分 HR。
- **方向信心**：**55-60%**（AWAY 微利）— TEX 投手戰 + 牛棚優勢是硬數據，但 HOU 主場 + Alvarez 1.082 vs RHP 真實 anchor 平衡部分。
- **風險**：
  1. Alvarez last7 BABIP **.500** + 1.134 火燙 — 部分回歸但 vs RHP 1.082 真實，本場仍危險
  2. Walker last7 BABIP **.050** 極冷期 — 嚴重反彈空間，但 Walker 9 分 vs RHP .884 真實 anchor
  3. Arrighetti Flag 8 -3.52 極端 — 本場可能繼續吃 ERA 1.88 假象 OR 大幅回歸 xERA 5.40，方向波動大
  4. TEX 連勝 2 vs HOU 連勝 1 — 雙方狀況略對立，主場/客場平均化

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
