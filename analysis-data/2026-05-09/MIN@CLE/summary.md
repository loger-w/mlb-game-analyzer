## 投手對決

### Tanner Bibee (HOME, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p70, K-BB% p59），gap vs ERA-only = +27.4
  - 同意 tier_v2 上修方向但保留下修風險：xFIP 3.84 / WHIP 1.47 / K-BB% 11.6 介於 Solid 與 Strong 之間，ERA 4.58 表象偏 Back-end 是被 hard_hit% 30.2 + barrel% 8.3 拖累。近 3 場 5 ER/13.7 IP（ERA 3.29）顯示已部分收斂。**判讀：運氣 vs 結構性混合** — FC（28.3% / RV -1.5）+ FF（24.9% / RV -1.2）雙速球結構性偏弱，CH（17.7% / RV +2.6）是唯一正貢獻；本場以 Strong Ace 下緣評價，不自動下修預測。
- **Reverse platoon 信號**：未 fired（vs LHB OPS .783 vs RHB OPS .750，|Δ| < 0.080 門檻）— 略反方向但量級不足，可忽略。
- **對手打線威脅**：MIN 季 OPS .735 vs RHP Average tier，但核心威脅是 Buxton（vs RHP .977 / EV95% 45.5 / Barrel% 20.5 / last7 OPS 1.013 全熱）— Bibee FC + FF 被打較硬，正中 Buxton 高 EV95% 對位區。次級熱手 Keaschall (last7 1.086)。但 AWAY chain_break #7-8 (Δ 0.333) → 7-9 棒易斷層，把 Buxton 推回打席的頻率受限。**整體威脅：中（Buxton 單棒 HR 風險高，但 chain 斷在後段壓制延伸得分）**。

### Joe Ryan (AWAY, RHP, 29 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +29.8
  - 同意 tier_v2 上修：K-BB% 19.1 / xFIP 3.29 / FIP 2.82 / WHIP 1.06 全都 Elite 級，ERA 3.72 vs xERA 3.13 反映輕度負運（HR 偏多）。**判讀：結構性偏多** — FF 42.2% 主球（RV -0.3 中性）+ KC（12.2% / RV +3.4）為殺手球。對 RHB 75 BF 樣本下 .185/.267/.262 強壓。本場以 Elite Ace 評價，無下修理由。
- **Reverse platoon 信號**：未 fired（vs LHB .729 vs RHB .529 為正常 RHP platoon，符合預期）— Ryan 對 LHB 略受傷（.244/.280/.449）但 CLE 主力多 RHB，影響有限。
- **對手打線威脅**：CLE 季 OPS .696 vs RHP Weak tier，最具威脅 DeLauter（vs RHP .876 / last7 1.421 熱手 / EV95% 39.4）。但 Ramirez vs RHP 僅 .641 / last7 .624（冷），Kwan / Rocchio last7 OPS < .640。Ryan 對 RHB 結構性強壓 + KC 殺手球 → CLE 主力 RHB 群被全面壓制。HOME chain_break #8-9 (Δ 0.311) 後段更薄。**整體威脅：低（DeLauter 是孤點，缺鏈接放大）**。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak
  - 與 season tier 落差一檔，本場下修。Ramirez (主將 vs RHP .641) 結構性弱於 season，DeLauter (last7 1.421) 是唯一放大點但易被孤立。**評估：下修**。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #8-9 fire (Δ 0.311 high)。CLE 1-5 棒 OBP top3 .338 / SLG mid .500 還能成串，但 #8-9 棒崩塌 → 過了 5 棒就難回到 Ramirez/DeLauter 打點區。對 Ryan 這種「TTO 越後越穩」的投手影響被放大（Ryan TTO3 OPS 反降至 .591）。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - 與 season tier 一致，本場以 season tier 評估。Buxton 帶頭 + Keaschall 熱手 → Average 偏上限。**評估：同意（不調整）**。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #7-8 fire (Δ 0.333 high)。MIN 1-5 棒組合稍弱（OBP top3 .305 / SLG mid .375），#7 棒之後直接斷層 → Buxton 若打席無人壘上，HR 也只算 solo。對 Bibee（TTO3 高 penalty）影響相對輕，因第三輪後接手的是 CLE 牛棚 (4.22 ERA + Armstrong IL)。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.22 / 2 / 1（Shawn Armstrong, IL15d）| 5.81 / 7 / 1（Cole Sands, IL15d）|

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 4.22 中等水平，core IL ×1（Armstrong）對應 ⏳ 🟠 中高 — 後段防守變薄。**關鍵交互**：Bibee TTO3 高 penalty (Δ +0.342) → CLE 教練可能 5-6 局就動牛棚 → MIN 後段攻擊面對的是已有缺口的 CLE 牛棚。對 Buxton/Keaschall 熱手是利好。
- AWAY 牛棚：ERA 5.81 顯著偏弱（聯盟倒數段），core IL ×1（Sands）對應 ⏳ 🟠 中高。Ryan 若能撐到 6+ 局（TTO3 OPS 反降至 .591 + K% 仍維持 23.7% → 體質支援撐長），可大幅省牛棚負擔；但若被提早換下，5.81 ERA 對 CLE 主力（Ramirez / DeLauter）是放分風險。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🔴 HOME TTO3 penalty：OPS Δ +0.342（TTO1 0.628 → TTO3 0.970），第三輪明顯衰退；K% 從 23.2% 掉到 12.1%（Δ -11.1pp）
- 🟠 AWAY TTO3 penalty：OPS Δ -0.078（TTO1 0.669 → TTO3 0.591），第三輪 OPS 反降，但 K% 從 31.1% 掉到 23.7%（Δ -7.4pp）（career fallback）— K% 觸發但 OPS 未惡化，意味 Ryan 第三輪「靠 contact 控制」而非三振宰割
- 🔴 HOME chain breaks at #8-9：OPS 落差 0.311
- 🔴 AWAY chain breaks at #7-8：OPS 落差 0.333
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
- 🟠 ⏳ AWAY 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - **雙重壓力判讀**：CLE 端 `tto3_penalty (high)` + `core_il × 1` + `chain_break` 三層同向 → MIN 後段攻擊機會明顯放大（牛棚提早換 + 牛棚薄）。MIN 端僅 `chain_break` + `core_il × 1` 兩層，且 Ryan TTO3 撐得住 → MIN 牛棚使用情境較少觸發。**結論：CLE 側淨負面壓力顯著大於 MIN 側**。

## 條件修正

- Park Factor: 101.0 → +0.05 run（Progressive Field 中性 / HR -9% 是 Buxton 風險的緩衝）
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：投手 tier 落差顯著（Ryan Elite Ace 結構 vs Bibee Strong Ace 下緣 / ERA 表象 Back-end）→ MIN 投手結構性優勢；非 doubleheader（系列 G2，CLE 主場 G1 已 6-4 勝）。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.1 | -0.2（chain_break #8-9 high）| 2.9 |
| AWAY | 4.9 | +0.1（CLE tto3_penalty + core_il 交互 +0.3 / 自家 chain_break -0.2）| 5.0 |
| Total | 8.0 | -0.1 | 7.9 |

## 整體判斷

- **方向（基本面）**：AWAY (MIN) 略佔優
- **總分（基本面）**：7.9 run（略低於 base 8.0；HR-suppress 球場 + Ryan 結構性壓制 CLE 主力 RHB）
- **方向信心**：~58%（投手結構性差距明確且 base 已給 MIN +1.8；信心未拉更高是因 CLE 主場 + DeLauter 熱手 + Bibee 若實戰能撐過第三輪則 CLE 牛棚反而較穩 — Bibee TTO3 樣本只 33 BF，high severity 但 confidence 偏低）
- **風險**：
  1. **Buxton 單棒 HR**：EV95% 45.5 / Barrel% 20.5 / last7 OPS 1.013，Bibee FC + FF 被打較硬（雙負 RV）正中對位區 → 1 球翻盤風險（Progressive HR -9% 略緩衝但壓不住 Barrel% 20.5）
  2. **DeLauter 熱手孤點**：last7 OPS 1.421 是 CLE 唯一火力，若延伸 → CLE 攻擊面可能超過 base 3.1；但 Ryan 對 RHB 強壓 + DeLauter vs RHP .876 (季) 已含此熱度 → 回歸壓力 > 持續壓力
  3. **雙隊 projected lineup 未公布**：9 人順序為 PA 近似，公布後若 CLE 推 Naylor / Manzardo (LHB) 進中段棒次可消化 Ryan 對 LHB 略弱（.244/.280/.449）優勢；MIN 公布後若打順異於預期亦影響 chain 評估
  4. **Bibee TTO3 樣本只 33 BF**：high severity 但 small_sample confidence — 若實戰他能撐到第三輪，CLE 牛棚（4.22 ERA + Armstrong IL）反而成為 buffer，當前 -0.2 / +0.3 修正方向會反向

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
