## 投手對決

### Gage Jump (HOME, LHP, 23 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = —（—），gap vs ERA-only = —
  - dossier script tier = Unknown（GS 紀錄為 None，本季先發樣本不足以撐 tier_v2 計算）；無 ERA/FIP/K-BB% 可參照。年齡 23 仍在成長期，但對 Mariners vs-LHP「🟢 Weak」對位下，雖無數據基準，**對位優勢可一定程度補償新人不確定性**。AI 採保守 Solid Starter 以下估計，不自動上修。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 未 fired（vs LHB / vs RHB 樣本均不足，無從計算）。
- **對手打線威脅**：Mariners vs LHP 整體 🟢 Weak — 僅 Julio Rodríguez (vs LHP 1.060) 與 Randy Arozarena (.830) 構成 top-of-order 威脅，Naylor (.485) / Young (.599) / Crawford (.583) 對 LHP 全部弱，#8-9 chain break 進一步壓制 backend 串聯。本場 Jump 雖無 tier 資料，但對位天平偏 Jump，AI 視為「未知但對位友善」。

### Emerson Hancock (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +18.3
  - tier_mismatch 高 gap，方向為「ERA 低估真實水平」（v2 score 96.1 vs ERA-only 77.8）。ERA 3.07 vs xERA 4.08（gap -1.01，未觸發 Flag 8 之 ≥1.5 門檻），但 xFIP 2.92 / FIP 3.49 / K-BB% 20.5 / WHIP 1.06 全部支持「真實水平接近 Strong Ace 級」。AI 同意 v2 tier，視為偏 Strong Ace（dossier 展開表所示），不自動下修預測；ERA 與 xERA 微幅落差屬樣本噪音。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 未 fired。vs LHB .220/.276/.381 與 vs RHB .240/.284/.385 OPS 差距僅 ~0.004，正常 RHP 對位曲線。
- **對手打線威脅**：Athletics vs RHP 整體 🟠 Strong — Nick Kurtz (vs RHP 1.030, last7 1.063) 與 Shea Langeliers (.930) 構成真實火力，但 Langeliers last7 OPS .528 / BABIP .100 顯示短期冷期，#2-3 chain breaks (Langeliers .930 → Soderstrom .660) 切斷上半段串聯。Hancock 主球種 FF/SI/ST 對混合手別 OAK 打線足以壓制，主要風險來自 TTO3 penalty（見下）。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong
  - matchup tier 比 season tier 高一檔（Average → Strong），AI 對本場 OAK 打線**上修評估方向**；對 RHP 火力來自 Kurtz / Langeliers 兩名 .900+ OPS 打者，但需與 chain_break / Hancock Elite tier 對沖。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break fired at #2-3（OPS 落差 0.270，Langeliers .930 → Soderstrom .660）— 影響打線**上半段串聯**：Kurtz / Langeliers 若上壘，下游 Soderstrom / McNeil / Rooker 清壘能力顯著下滑，壓制 multi-run innings 機率。heat_vs_babip 未 fired。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - matchup tier 比 season tier 低一檔（Average → Weak），AI 對本場 SEA 打線**下修評估方向**；vs LHP 只剩 Julio (1.060) / Arozarena (.830) 兩名打者具威脅，#3-5 棒（Naylor .485 / Young .599 / Crawford .583）對 LHP 全面壓制。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break fired at #8-9（OPS 落差 0.283）— 影響**backend 串聯**，但 #8-9 棒打席機會本就有限，對總分壓制有限。heat_vs_babip 未直接 fire（Flag 3 為 last7 BABIP .215 而非熱度信號）。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.31 / 3 / 0 | 3.17 / 2 / 1 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 4.31 屬聯盟中下水準，無核心 IL（0 名），整體可用性完整但品質普通。對 SEA 後段（vs LHP/RHP 混合）構成中等壓制力；近 3 天消耗資訊未列，AI 採中性判讀。SEA 打線 vs LHP weak 之外亦無熱手段，OAK 牛棚不會被特別放大或抵消。
- AWAY 牛棚：ERA 3.17 屬聯盟前段，但 1 名核心 IL（Carlos Vargas，IL60d 長傷）— 對應 §牛棚傷兵累計效應「1 名核心 → 🟠 中高」分級，後段防守變薄。若 Hancock TTO3 衰退提早換投，SEA 後段牛棚負擔 ↑，OAK Kurtz / Langeliers 在 7-9 局得分機會升高，**對 HOME 末段得分產生 +0.0 ~ +0.2 推力**（Table B core_il ×1）。

## 風險提示

- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.215):
  - SEA 近 7 天 BABIP .215 顯著低於 .260 門檻，純運氣面有回歸空間；但本場 SEA vs LHP matchup tier 已下修為 🟢 Weak，3-5 棒結構性對 LHP 弱，**結構面壓制可能持續**而非純運氣。AI 採「部分回歸、部分持續」敘事 — 不會因 BABIP 自動上修 AWAY 得分，整體仍以 weak matchup 為主導判斷。**不自動 ±run value**。

### 額外信號
- 🔴 AWAY TTO3 penalty：OPS Δ +0.194（TTO1 0.592 → TTO3 0.786），第三輪明顯衰退；K% 從 27.8% 掉到 24.5%（Δ -3.3pp）
- 🟠 HOME chain breaks at #2-3：OPS 落差 0.270
- 🟠 AWAY chain breaks at #8-9：OPS 落差 0.283
- 🟠 ⏳ AWAY 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - 本場受影響：Hancock TTO3 OPS +0.194 弱化，若 5-6 局後換投 → SEA 變薄牛棚（Vargas IL60d 缺陣）面對 Kurtz / Langeliers，後段失分風險上行。⏳ 半衰期短，但 5/26 一日內無法補強。**TTO3 (high) + core IL ×1 同向 fire** → 取單側 max 區間（TTO3 +0.3）+ core IL (+0.1)，總計 HOME 端 +0.4 上推力。

## 條件修正

- Park Factor: 109.0 → +0.45 run
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：非 doubleheader（系列 G2，G1 已於 05-25 結束）。HOME 先發 Gage Jump tier Unknown（樣本不足），AWAY 先發 Hancock 🔴 Elite Ace（v2）。tier 落差顯著，formula base 已反映（HOME 4.3 < AWAY 5.5）。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.3 | +0.2 | 4.5 |
| AWAY | 5.5 | -0.2 | 5.3 |
| Total | 9.8 | 0.0 | 9.8 |

## 整體判斷

- **方向（基本面）**：AWAY（SEA 5.3 vs OAK 4.5，落差 0.8 run ≥ 0.5 門檻）
- **總分（基本面）**：9.8 run（adjusted Total，formula base 與信號修正互相對沖後維持）
- **方向信心**：60% — Hancock Elite Ace 對 OAK Strong-vs-RHP 構成的壓制是主導因子，formula base 已給 SEA +1.2 run 領先，但 OAK 端 TTO3 + core IL 雙信號疊加削弱領先幅度；HOME 球場 PF 109 已含於 base，Jump 真實水平未知是最大下行風險。
- **風險**：
  1. **Jump tier Unknown**：HOME 先發無 ERA/FIP/K-BB% 數據，若實際表現遠超 Below-average baseline（年輕 LHP 對 SEA weak-vs-LHP 對位）→ HOME 防守上修、AWAY 得分下修，可能反轉方向。
  2. **SEA last7 BABIP .215（Flag 3）**：若回歸正常化（向 .290 靠攏）→ AWAY 得分上推，強化 AWAY 方向但同時 Total 偏高。
  3. **Hancock TTO3 penalty 落地時機**：教練決策（5 局 vs 6 局換投）直接影響 OAK 後段攻擊窗，SEA 牛棚 Vargas 缺陣放大此不確定性。
  4. **打線 projected（未公布）**：HOME / AWAY 打序近似 PA 排序，實際打序可能調動 #2-3 / #8-9 chain_break 位置，信號方向可能微調。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組