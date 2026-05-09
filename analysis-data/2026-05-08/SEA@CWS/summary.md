## 投手對決

### Sean Burke (HOME, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p89, K-BB% p79），gap vs ERA-only = +2.6
  - 同意 Elite Ace（gap +2.6 < 15 不觸發 tier_mismatch），但**強度偏 Strong Ace 上緣**：whiff 8.9% / velo 88.8 mph 對 Elite 標籤偏弱，靠 K-BB% 15.4 + WHIP 1.01 撐 percentile；ERA 2.72 vs xERA 3.38 (-0.66) 顯示 ERA 微帶運氣成分（未到 Flag 8 門檻）。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - **未 fire** — vs LHB OPS ≈.701 / vs RHB OPS ≈.502，正常 platoon（同手別更佳），與 RHP 預期一致；不放大本場風險。
- **對手打線威脅**：SEA 打線 Average / Normal，主力 Julio / Arozarena / Young 為 RHB（Burke 強項側 .172/.221/.281）；Cal Raleigh last7 OPS .415 + BABIP .000 + EV95% 29.7（接觸品質弱）顯著冰冷，威脅低。Naylor 是唯一主力 LHB（vs Burke .728，中性偏好）。**Burke 1-2 巡應可壓制至失 1-2 分；TTO3 樣本 32 BF 訊號矛盾（OPS 反降但 K% 從 27.8 → 15.6），不視為加分項，第三巡仍建議換投。**

### Emerson Hancock (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +8.0
  - 同意 Elite Ace（gap +8.0 < 15 不觸發 tier_mismatch）；**結構性支撐強**：xFIP 2.42 + K-BB% 25.1 + hard_hit 21.8% 為三項頂標。ERA 2.59 vs xERA 3.63 (-1.04) 雖未到 Flag 8 門檻 1.5，但接近警戒（運氣成分存在）。**風險側**：barrel% 10.5（偏高）與 velo 89.5 mph 顯示 stuff 非真頂級，靠 command + 三球路平均（FF 36.2 / SI 24.7 / ST 23.3）取勝。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - **未 fire** — vs LHB OPS .675 / vs RHB OPS .661，|Δ| = 0.014 << 0.080 門檻；CWS 左右開弓打線（Murakami / Montgomery / Benintendi 為 LHB；Vargas / Meidroth 為 RHB）對 Hancock 無 platoon 槓桿。
- **對手打線威脅**：CWS Average tier，但 **Murakami（season .934 / vs RHP .934）是真威脅**，單打可主導本場 swing。Meidroth last7 .955 OPS / BABIP .529 為極端不可持續熱度（AI 不上修）。Vargas / Montgomery / Benintendi vs RHP 全數下修 0.05-0.16，5-7 棒為對 RHP 弱項。**Hancock 1-2 巡可壓制；但 TTO3 OPS spike +0.311（.587 → .898, 33 BF）+ Murakami 第三巡 = 全場最高風險點，5-6 局後續局數應提早交班牛棚。**

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - matchup tier (vs RHP) = season tier，無調整方向；但 top 5 中 Vargas (.775→.618) / Montgomery (.823→.770) / Meidroth (.729→.670) 三人 vs RHP 明顯下修（platoon_advantage 不 fire 但有隱性壓力），**實質本場略下修**（Murakami 為例外，vs 兩手別均 .934）。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - **chain_break #5-6 (Δ 0.222) fired** — Meidroth (.955 last7) 與 Benintendi (.642) 之間落差大，意味 #5 之後對 Hancock 屬壓制段，6-9 棒幾乎無反擊力。AI 判：壓制 CWS 攻擊串聯上限約 -0.15 run。
  - heat_vs_babip 不 fire（heat = Normal）。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - matchup tier = season tier，無調整方向。Julio (.734→.646) 主力 RHB 對 Burke 偏好降低，Cal Raleigh (.616→.674) 微上修但 last7 嚴重低迷；**整體與 season tier 一致**，無明顯上下修。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - **chain_break #6-7 (Δ 0.264) fired** — 中段（Naylor #4 / Young #5）尚可，但 #6 之後對 Burke 為壓制段，反擊鏈只在 1-5 棒有效。AI 判：壓制 SEA 攻擊上限約 -0.2 run。
  - heat_vs_babip 不 fire（heat = Normal），但 last7 BABIP .227 觸 Flag 3，見下方風險提示。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.49 / 4 / **1（Vasil IL60d）** | 3.14 / 5 / **3（Vargas IL60d / Speier IL15d / +1）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 4.49 偏弱（聯盟均 ~4.0），1 名 core IL → 🟠 **中高影響**。本來就是球隊弱點，6-9 局對 SEA 攻擊壓制力低；如比賽進入後段（Burke 5-6 局退場），CWS 牛棚會放大失分風險。AI 不額外加分，因為 SEA 中段以下打線（chain_break #6-7）也較難利用此弱點。
- AWAY 牛棚：ERA 3.14 帳面強，但 3 名 core IL → 🔴🔴 **極高（崩盤級）**。可用 high-leverage RP 已大幅縮減，若 Hancock 在 5-6 局被換下且比賽仍膠著，HOME 後段得分機率顯著上升 → 是本場 +run 方向偏 HOME 的主要驅動。

## 風險提示

- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.227):
  - **可能小幅回歸但不會大幅反彈**：Cal Raleigh BABIP .000 + EV95% 29.7（接觸品質弱）顯示部分為運氣（向 ~.260 反彈），但 season OPS .616 反映**結構性弱化**而非單純冷期。AI 判：本場引用 last7 數據時可下調預期 ~0.05 OPS（敘事），**不自動 ±run**（Flag 3 紀律）。

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ -0.311（TTO1 0.698 → TTO3 0.387），第三輪明顯衰退；K% 從 27.8% 掉到 15.6%（Δ -12.2pp）
- 🔴 AWAY TTO3 penalty：OPS Δ +0.311（TTO1 0.587 → TTO3 0.898），第三輪明顯衰退；K% 從 28.6% 掉到 24.2%（Δ -4.4pp）
- 🟠 HOME chain breaks at #5-6：OPS 落差 0.222
- 🟠 AWAY chain breaks at #6-7：OPS 落差 0.264
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
- 🔴 ⏳ AWAY 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
  - **本場主軸：AWAY core IL ×3 + Hancock TTO3 high severity = 同向疊加**（兩者都讓 SEA 後段防守變崩盤級）。Hancock 第三巡（~5-6 局）若被 CWS 攻破，SEA 牛棚無法接手 → HOME 後段得分明顯上修，是本場 HOME 略佔上風的核心理由。同向 signals 取單側 max + 0.1 = +0.6 run 上限給 HOME 攻擊。
  - HOME 側 Burke TTO3（OPS 反降但 K% 大跌）訊號矛盾，加上 CWS 牛棚 core IL ×1 + ERA 4.49 已弱 → SEA 後段攻擊小幅利好，但 SEA 自身 chain_break #6-7 削弱反擊鏈，淨效應接近抵銷。

## 條件修正

- Park Factor: 97.0 → -0.15 run
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：兩位先發同為 RHP / 26 歲 / 巔峰期，腳本 tier 同為 Elite Ace（peripherals）；非 doubleheader（單場）；Rate Field 略利投（PF 97, HR -1%），park 調整已併入 base。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.7 | +0.45（Hancock TTO3 high +0.3 + SEA core IL 3+ +0.5 → 同向取 max+0.1 = +0.6；扣 HOME chain_break -0.15） | 4.15 |
| AWAY | 3.4 | 0.0（Burke TTO3 訊號矛盾僅 +0.05 + CWS core IL ×1 +0.1 → 同向取 max+0.1 = +0.2；扣 AWAY chain_break -0.2） | 3.40 |
| Total | 7.1 | +0.45 | 7.55 |

## 整體判斷

- **方向（基本面）**：HOME（Chicago White Sox）略佔上風
- **總分（基本面）**：~7.5（base 7.1 + 信號 +0.45）
- **方向信心**：55%
- **風險**：
  1. **兩隊打線皆 projected**（賽前 ~2-4h 才會公布實際打序），Murakami 是否排在 #1 / Cal Raleigh 是否在 #3 等位置變動會明顯改變對 chain_break 與 TTO3 風險的判讀
  2. **Hancock 真實實力可能比 ERA 還高**（xFIP 2.42 / K-BB% 25.1），若 1-2 巡指揮力良好可幾乎完全壓制 CWS（Murakami 除外），第三巡才打開 → CWS 攻擊集中時間窗短
  3. **SEA 牛棚 3 名 core IL 是最大不確定性**：若 SEA 提早領先（Hancock 撐住前 6 局），影響有限；若膠著至 7-9 局，HOME 後段 +0.5-0.8 run 是合理上界
  4. **Murakami 單一打者 swing 風險**：vs RHP .934 OPS，單支 HR / 雙安可決定本場勝負；Hancock barrel% 10.5 對其有顯性危險

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
