## 投手對決

### Tatsuya Imai (HOME, RHP, 28 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = —（樣本 <30 BF，未打分），gap vs ERA-only = —
  - 3 GS 樣本，tier_v2 無法判斷。原始：ERA **7.27** / xERA 4.09 / FIP 3.91 / xFIP 4.51 / K-BB% **4.5**（極低）/ WHIP **2.08** / Single-pitch FF 47.8%. NPB 過渡期控球崩潰，**era_xera_delta +3.18（Flag 8）**。實質 ⚪ Below Average / Back-end 邊緣（按 xERA 4.09 / FIP 3.91 推估，但 K-BB 4.5 + WHIP 2.08 結構警報未解除）。
- **Reverse platoon 信號**：未 fired。
  - n/a
- **對手打線威脅**：高。SEA top 5 vs RHP（Donovan .996 / Rodríguez .721 / Naylor .770 / Arozarena .788 / Raley .868）+ Imai K-BB 4.5 控球差 → 上壘 + 強打結合 = SEA 應強上修至 🟠 Strong matchup。

### Bryan Woo (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p66, K-BB% p81），gap vs ERA-only = +20.2
  - 同意 Strong Ace。ERA 4.02 / xERA 3.55 / FIP 3.65 / xFIP 3.90 / K-BB% 16.0 + 近 3 GS 3 ER/18 IP (1.50 ERA recent) — 結構與 form 都很硬。gap +20.2 主要因 K-BB% p81（v2 抓得到）。Single-pitch FF 50.7%.
- **Reverse platoon 信號**：未 fired。
  - n/a
- **對手打線威脅**：中。HOU top 5 vs RHP 強（Altuve .748 / Alvarez 1.045 / Paredes .725 / Walker .892 / Smith .686）但 last7 BABIP 0.231（Flag 3 Cold）+ 個別 last7 數字弱（Walker .303 / Smith .335）。Alvarez 是主要威脅點（vs RHP 1.045 + EV95 52.3 + Barrel 18.0）。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong
  - 部分同意。Alvarez 1.045 vs RHP 是 elite 個體，整體 vs RHP 接近 Strong；但 Cold BABIP 0.231 + Walker/Smith last7 .303/.335 — 維持 🟠 Strong but with Cold caveat。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - HOME chain_break #7-8 gap 0.372 fired — Shewmake 1.041 (small sample) → Matthews .669，部分樣本污染 → 部分採用，−0.2 run。Flag 3 last7 BABIP 0.231 cold — 短期可能反彈但 Walker/Smith xwOBA 也冷，真實狀態冷，反彈幅度有限。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong
  - 上修同意。Donovan .996 / Naylor .770 / Arozarena .788 / Raley .868 + Imai K-BB 4.5 → 對 Imai 控球差 RHP 結構性 edge 大。實質 🟠 Strong 上緣。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - AWAY chain_break #6-7 gap 0.177 — Raley .854 → Crawford .677，small chain break，−0.1 run。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 5.98 / 8 / 1 | 3.31 / 5 / 3 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：**HOU 5.98 ERA 結構性差** + 1 core IL（Hader closer）。**Hader 是頂尖 closer，IL 嚴重升級此 1 IL 影響至 🔴 高**。後段 8-9 局完全失能，SEA 末段加分機率高。
- AWAY 牛棚：SEA 3.31 ERA + 3 core IL（Vargas + Speier + 1）數量上崩盤級，但 ERA 仍 3.31 顯示深度撐住 — 實質 🟠 中高。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=+3.18):
  - 雙刃。若 Imai 回歸 xERA 4.09 → HOU 失分降，SEA edge 縮小；若延續 ERA 7.27 → SEA edge 放大。但 K-BB% 4.5 + WHIP 2.08 結構性警報未解除 — 預期 4.5-5.0 ERA 區間（介於 xERA 與 ERA）。不自動下修預測。
- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.231):
  - 短期可能反彈，但 Walker/Smith last7 OPS 也低 — 反彈含真實狀態冷，幅度有限；Alvarez (1.045 vs RHP) 個別仍是真實威脅。不自動 ±run value。

### 額外信號
- 🟠 HOME single-pitch dependent：主球種使用率 47.8%（≥45.0%）
- 🟠 AWAY single-pitch dependent：主球種使用率 50.7%（≥45.0%）
- 🔴 AWAY TTO3 penalty：OPS Δ +0.162（TTO1 0.607 → TTO3 0.769），第三輪明顯衰退；K% 從 19.5% 掉到 15.9%（Δ -3.6pp）
- 🔴 HOME chain breaks at #7-8：OPS 落差 0.372
- 🟠 AWAY chain breaks at #6-7：OPS 落差 0.177
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
- 🔴 ⏳ AWAY 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
  - 雙 Flag 8 + Flag 3 + 雙 single-pitch + 雙 core IL 異常複雜；雙方訊號互相抵銷，distribution 極寬。AWAY 數量上 3 IL 崩盤級但 ERA 3.31 證明深度撐住；HOME 1 IL 但 Hader 是 closer 升級。實質後段近於持平。

## 條件修正

- Park Factor: 98.0 → -0.10 run
- 天氣：室內（Roof Closed，不適用）
- 先發 tier / doubleheader：Woo Strong Ace > Imai Back-end / Below Avg 一級以上；HOU 5.98 牛棚 + Hader IL vs SEA 3.31 牛棚 3 IL — SEA 後段優勢實質存在。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.0 | +0.2（AWAY core IL ×3 +0.2 + AWAY single-pitch +0.2 + AWAY TTO3 +0.2 互動 max+0.1 −0.2 chain HOME） | 4.2 |
| AWAY | 4.4 | +0.2（HOME core IL Hader closer +0.2 + HOME single-pitch +0.2 互動 max+0.1 −0.1 chain AWAY） | 4.6 |
| Total | 8.4 | +0.4 | 8.8 |

## 整體判斷

- **方向（基本面）**：AWAY (SEA 微傾)
- **總分（基本面）**：8.8（distribution 極寬，雙 Flag 不確定）
- **方向信心**：58%（卡上信心最低之一） — Woo 等級優於 Imai + HOU 牛棚 5.98 + Hader IL；但雙 Flag 8 + Flag 3 + Imai 1.5 GS 樣本不確定使預測 distribution 寬。
- **風險**：
  1. ⚠️ Imai Flag 8 +3.18 雙刃 — 若回歸 xERA → SEA edge 縮小；若延續 ERA 7.27 → SEA edge 放大；單場結果方差極大
  2. ⚠️ HOU Flag 3 BABIP 0.231 冷期 — 若 Alvarez 等爆發 → HOU 主場突發爆分
  3. HOU 牛棚 5.98 + Hader IL — SEA 若取得早期領先後段難追
  4. Woo TTO3 OPS Δ +0.162（rookie penalty）— SEA 教練若早換投，SEA 自己 3 core IL 牛棚也需撐

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
