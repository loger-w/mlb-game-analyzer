## 投手對決

### Shota Imanaga (HOME, LHP, 32 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +0.8
  - ✓ 同意 Elite Ace。ERA 2.40 / xFIP 3.32 / K-BB% 20.7 / WHIP 0.85，四項全在 elite 區間且互相一致；gap +0.8 微小，無 |≥15| 結構落差，直接採用。
- **Reverse platoon 信號**：vs LHB OPS .601 > vs RHB OPS .488（Δ +0.113，medium）
  - 對 Reds top 5 影響有限——核心多右打（De La Cruz S, Stewart R, McLain R, Steer R）+ Friedl L 一人；Imanaga 對 RHB 壓制反而更好（OPS .488）→ Reds 主要威脅打者落在他壓制強的對位面。Reds 右打對 LHP 有 platoon advantage（season 數據偏高）會局部抵銷，但 reverse 訊號本身不放大本場風險。
- **對手打線威脅**：低。Reds 是 🟡 Average / matchup vs LHP 🟢 Weak（最弱層）；連敗 6 場、last7 集體冷（De La Cruz .498、Stewart .204、McLain .350）；Imanaga 球路（FF 41% / FS 34%）對 chase swing 弱化的冷打者有利。

### Rhett Lowder (AWAY, RHP, 24 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = 🟢 Back-end Starter（xFIP p36, K-BB% p34），gap vs ERA-only = +15.9
  - 同意 Back-end。gap +15.9 主因 ERA 5.09 vs FIP 3.18 / xERA 4.64 偏離，是 BABIP / 序列運氣不利推升 ERA，**不是隱藏 ace**——K-BB% 7.8% 仍在後段（elite 門檻 > 20%），whiff% 7.8% 偏低代表結構並未升檔。判讀「ERA 略低估真實水平」，但維持中後段先發預期，不自動下修對手得分。
- **Reverse platoon 信號**：dossier 訊號摘要未列（兩側 BF 與 Δ 未過門檻），跳過。
- **對手打線威脅**：高。Cubs 🟠 Strong / vs RHP 🟠 Strong；近 30 戰 22-8 攻守皆強；Happ last7 1.250 / Busch 1.208 / Crow-Armstrong .936 三人全熱；Lowder 主球種 SI 31% + SL 24% 對右打 OPS 仍 .557（K-BB% 不足以靠球種壓回去）。

## 打線評級

### HOME — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong
  - 同意，不上下修。season 與 matchup 一致；last7 BABIP .289 中性，無熱手 / 冷手回歸風險。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #3-4 fire（Δ 0.184，medium）：1-3 棒 Bregman .695 / Hoerner .806 / Happ .890 OBP 串聯尚可，但 #4 Busch season .706 形成落差。緩解因素：Busch last7 OPS 1.208 極熱，臨場可能填補；若回歸均值則 4-5 棒清壘能力受限，影響單局多分上限。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - matchup tier 比 season 下修一檔，但 platoon_advantage 同時 fire（top 5 中 4 人 vs LHP OPS 較 season +0.050 以上：De La Cruz +0.239, Stewart +0.242, Steer +0.163, McLain +0.073）→ 兩股相反力量。實際本場進一步下修：Reds last7 普遍冷（De La Cruz .498、Stewart .204、McLain .350、BABIP 0.067-0.278）+ Imanaga 對 RHB 反而更強，platoon advantage 主要受惠者 RHB 在他面前實際 OPS 不會跑出來。整體取「Weak 偏向」。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #6-7 fire（Δ 0.218，high）：Reds 後段（#6 之後）OPS 嚴重斷層，本就弱的攻擊在後半棒次幾乎無威脅 → 即便前段靠 De La Cruz / Stewart 偶發 platoon advantage 上壘，後段難清壘，壓制總分上限。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.86 / 10 / **4（崩盤級 🔴🔴）** | 4.23 / 5 / **2（吃緊 🔴）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：**雙重壓力極大**。core IL ×4（Caleb Thielbar, Hunter Harvey 等）已對應「3+ 名 → 崩盤等級」；同時 Imanaga TTO3 penalty fire（OPS Δ +0.273、K% drop 19.3pp，high tier）→ 教練第三輪一輪到就須換投，**但接班的高槓桿層幾乎沒人**。近 3 天 G1（5/06 7-6 鏖戰）已消耗。對 Reds 末段威脅看似不大（Reds 攻擊弱）但只要 Reds 在第 6-9 局打到 3-4 棒（De La Cruz / Stewart）+ platoon advantage → 失分風險顯著放大。
- AWAY 牛棚：core IL ×2（Caleb Ferguson, Emilio Pagán）= 「2 名 → 高」；Reds 連 6 敗、系列 0-1，過去三天牛棚消耗高（雖然 Lowder TTO 走勢 -0.288 反而越投越好，但 ERA 5.09 在強隊主場不會撐深）。Cubs 攻擊強 + 主場 + 後段對冷投捕互動較佳 → **AWAY 牛棚是 Cubs 多得分的主要來源**。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🟠 HOME reverse platoon Δ +0.113（vs LHB OPS 0.601 > vs RHB OPS 0.488）— 對 Reds 多右打打線**不放大風險**（Imanaga 對 RHB 壓制更好）。
- 🔴 HOME TTO3 penalty：OPS Δ +0.273（TTO1 .594 → TTO3 .867），K% 27.4 → 8.1（Δ -19.3pp）— **本場最關鍵單一風險**。high-tier signal，與 HOME core_il_count ×4 形成 §量級錨點 Table B 雙信號同向交互（Cubs 牛棚崩盤 + 先發第三輪衰退）→ 後段失分機制疊加，summary 量級採「單側 max +0.1」(取 +0.6)，不分項相加。
- 🟠 HOME chain breaks at #3-4：Δ 0.184，壓制 4-5 棒清壘 → 限制單局多分上限。
- 🟠 AWAY platoon advantage：top 5 vs LHP OPS 普遍上修，但 Imanaga 對 RHB 反向強 + Reds last7 冷 → 抵銷後實際取下界 +0.1。
- 🟠 AWAY chain breaks at #6-7：Δ 0.218（high）→ Reds 後段串聯壓制，搭配前段冷打與 weak matchup tier，總分上限受限。
- 🔴 ⏳ HOME 牛棚 core IL ×4 → 崩盤級。短半衰期 (⏳)，對手反應快，但本場已影響——Cubs 換投選項極窄。
- 🔴 ⏳ AWAY 牛棚 core IL ×2 → 高吃緊。Cubs 攻擊強 + 主場 → 預期將擊出本場關鍵分。

## 條件修正

- Park Factor: 92.0 → -0.40 run（Wrigley HR -8%，投手友善；Imanaga FF/FS 飛球路為主，PF 利他）
- 天氣：未公布（跳過天氣分析）。**注意**：Wrigley 5 月午場風向變化大，若臨場吹 Out To CF/LF 將實質改變總分判讀，請進場前覆查。
- 先發 tier 落差：Imanaga 🔴 Elite Ace vs Lowder 🟢 Back-end Starter — Cubs 結構性投手優勢明顯，是本場主要方向力。
- doubleheader：無。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.4 | -0.2 (chain_break 3-4) +0.3 (AWAY core_il ×2) = **+0.1** | **3.5** |
| AWAY | 2.8 | +0.6 (TTO3 + HOME core_il ×4 互動，取單側 max +0.1) +0.1 (platoon_adv 取下界) -0.2 (chain_break 6-7) = **+0.5** | **3.3** |
| Total | 6.2 | +0.6 | **6.8** |

## 整體判斷

- **方向（基本面）**：HOME（Chicago Cubs）
- **總分（基本面）**：~ 6.8 run（HOME 3.5 / AWAY 3.3）
- **方向信心**：65-70% HOME。Cubs 全季 25-13、近 10 戰 8-2、連勝 8 場、主場 Wrigley、先發 tier 全面壓制（Elite vs Back-end）；Reds 連敗 6 場、last7 集體冷、vs LHP matchup 最弱層；唯一壓抑 HOME 單調走勢的是 Cubs 牛棚崩盤級 IL，可能在 Imanaga TTO3 後製造翻盤窗口。
- **風險**：
  1. **Imanaga TTO3 penalty (high) × Cubs 牛棚 core IL ×4 雙重壓力**：若 Imanaga 5-6 局退場後接班的高槓桿層幾乎沒人，Reds 即便弱攻擊也可能在第 6-9 局靠 De La Cruz / Stewart 撕出失分機會 → 翻盤主路徑。
  2. **Reds last7 集體冷 + BABIP 偏低**（De La Cruz .278、Stewart .087、McLain .067）：屬於不可持續的低 BABIP，本場可能就是反彈點，需注意 De La Cruz 單一輪次的 platoon advantage 爆發。
  3. **Lowder ERA 偏低估真實水平 (gap +15.9)**：FIP 3.18 / xERA 4.64 顯示他比 ERA 5.09 帳面好；雖 K-BB% 仍弱不是 hidden ace，但 Cubs 攻擊不一定如帳面強隊那麼爽快出分，total under 風險存在。
  4. **Wrigley 5 月午場風向**：未公布，若 Out To CF/LF ≥ 15 mph 將顯著拉高總分；In From CF 反向則 Imanaga 可能完封 Reds 弱攻 → 上下都有 tail。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
