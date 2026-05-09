## 投手對決

### Robby Snelling (HOME, LHP, 22 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = Unknown（無 2026 球季數據；career TTO 樣本僅 63 BF），gap vs ERA-only = —
  - 無法評等。Career TTO1 OPS .769 / K% 24.5 / BB% 13.2，TTO2 樣本僅 10 BF（OPS 1.125、BB% 20%）— 控球是最大隱憂。本場視為**新人/重回大聯盟首戰級別**的高變異樣本，不要假設他能投到第三輪。
- **Reverse platoon 信號**：未 fired（樣本不足）。
- **對手打線威脅**：WSH vs LHP 整體 🟢 Weak（top 5 vs LHP OPS 平均 .743），但 Brady House .932 (43 PA, small) / Wood .881 / Curtis Mead .725 是擾動點。真正的威脅不是 swing damage，而是 BB% 給 WSH 連鎖上壘 + 短局數逼牛棚提早接手。

### Foster Griffin (AWAY, LHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p68, K-BB% p59），gap vs ERA-only = -22.9
  - **不同意 ERA-only 的 🟠 Strong Ace 評等**，採 score-derived 🟡。理由：barrel% 11.4（偏高）+ FF RV/100 −2.9（速球被打爆）+ 平均球速 85.8（30 歲已進入退化期）+ 樣本僅 39.67 IP — ERA 2.27 多半是 hard_hit% 20.9 撐起的「軟接觸假象 + 正運氣 BABIP」混合。xERA 4.41 是真實水平，**結構性偏離大於運氣偏離**。本場以 Solid Starter 看待、不以 Ace 折讓對手得分。
- **Reverse platoon 信號**：未 fired（vs LHB .627 vs vs RHB .607，差 0.020）。但兩側都壓得不錯（左打.167 AVG），對 MIA 偏左打打線（Lopez 1.027 vs LHP 是 outlier）整體有利。
- **對手打線威脅**：MIA vs LHP top 5 平均 .707 — Otto Lopez 1.027 是必須避開的高威脅點，Marsee/Edwards/Norby/Hicks 對 LHP 都偏弱（≤ .731）。Griffin 平衡 5 球種（FC 28.4 / FF 17.5 / ST 14.1 / SI 11.7 / CH 11.2）對 platoon-advantaged MIA 打線是優勢；但 TTO3 +0.156 警告 → 第三輪打線回算到 Lopez/Norby/Hicks 段時要警覺，預期 5 IP 上下。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak — 等等，dossier 此欄寫「🟢 Weak（vs LHP）」其實是 WSH 對 LHP 的對應；MIA 對 LHP 應對照 home 自身 platoon table。MIA top 5 vs LHP OPS：Marsee .617 / Edwards .731 / Lopez 1.027 / Norby .531 / Hicks .629 — Lopez 是 outlier，整體均值 .707 屬「🟡 Average → 略 weak」。**對 Griffin 看法：略下修**（season .700 OPS → 預期 ~.660-.680 OPS）。
- **chain_break / heat_vs_babip 信號**：🔴 #5-6 OPS 落差 0.372（Hicks .956 → Caissie .584） — Hicks last7 OPS 1.187 是當前隊內最熱（含運氣成分，BABIP .294 還好），但他下一棒直接斷鏈到 .584 OPS 的 Caissie，會壓制 MIA 第 5-6 棒之後的得分串聯。Lopez（#3）+ Hicks（#5）兩個 vs LHP 強點之間隔了 Norby（vs LHP .531），是 chain 的真正缺口。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟡 Average（dossier 此欄為 "AWAY vs LHP" → 對應 WSH 對 Snelling 的 LHP 評等）。WSH top 5 vs LHP OPS：Wood .881 / Lile .678 / Abrams .652 / House .932 / Young .572 — Wood + House 是強點，但 Abrams（強打）對 LHP 明顯掉檔（vs RHP 1.066 → vs LHP .652，−0.414）。**對 Snelling 看法：持平 → 略上修**（Snelling 控球疑慮加成，預期靠 BB 拿基壘的機率高）。
- **chain_break / heat_vs_babip 信號**：🟠 #3-4 OPS 落差 0.237（Abrams .933 → House .696，但 House 對 LHP 反而 .932 → 本場該 chain break **本場顯著淡化**）；Flag 3 last7 BABIP .260 → Lile .176 / García .188 / Vivas .143 / Mead .250 一片冷彈，Wood / Abrams / House 倒是熱 — chain 上半段（1-4 棒）穩，下半段（7-9 棒）幾乎全冷，場面會集中在前 4 棒能否打開。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.42 / 3 / 2（🔴 高） | 4.62 / 7 / 2（🔴 高） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（MIA, ERA 3.42）：核心傷 Pete Fairbanks（Closer, IL15d）+ Ronny Henriquez（IL60d, setup-quality）兩名，第 9 局 closer 角色由 Calvin Faucher / Anthony Bender 接手。整體 ERA 仍屬聯盟前段，但**少了 Fairbanks 的 high-leverage 救援彈性**；對 WSH 末段（7-9 棒一片冷）影響有限，主要風險在 6-7 局過渡段（middle reliever 對 WSH 1-3 棒 Wood / Lile / Abrams）。
- AWAY 牛棚（WSH, ERA 4.62）：核心傷 Clayton Beeter（IL15d）+ Max Kranick（IL15d）兩名，加上 Cole Henry / DJ Herz / Josiah Gray 等多名長期 IL，**牛棚深度結構性偏薄**。Griffin 預期 5 IP 上下退場後，WSH 必須用 Lord / Cavalli / Lovelady / Schultz 等接力面對 MIA 1-3 棒（Marsee / Edwards / Lopez），對 Lopez vs LHP 1.027 / Edwards vs RHP .915 都是高風險點。**末段是本場 MIA 得分的主來源。**

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-2.14):
  - 結構性 > 運氣。30 歲 LHP，平均球速 85.8 mph 已是初期退化期，barrel% 11.4（高）+ FF RV/100 −2.9（速球被打爆）+ K-BB% 11.6（中下）+ FIP 4.13 / xFIP 3.87 都跟 xERA 4.41 對齊。ERA 2.27 是 hard_hit% 20.9 + 樣本 39.67 IP 的雙重正運氣。**本場不把他當 Strong Ace 看，視為 Solid Starter**；不自動下修對手得分（仍走 formula base + 其他信號）。
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.26):
  - 部分回歸、不全部反彈。WSH 季內 BABIP .285（接近聯盟均值），近 7 天的 .260 是冷期但非極端低。組成上：Wood / Abrams / House 已經熱了（last7 OPS .665/.994/.875），冷的是 Lile .176 / García .188 / Vivas .143 / Mead .250 — 都是後段棒次與弱接觸打者，**回歸空間有限**（Vivas / García 的接觸品質本來就差）。本場不額外調整 ±run value。

### 額外信號
- 🔴 AWAY TTO3 penalty：OPS Δ +0.156（TTO1 0.497 → TTO3 0.653），第三輪明顯衰退
- 🔴 HOME chain breaks at #5-6：OPS 落差 0.372
- 🟠 AWAY chain breaks at #3-4：OPS 落差 0.237
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
- 🔴 ⏳ AWAY 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - 雙方牛棚同樣吃緊，但 ERA 落差 1.20 → WSH 牛棚是更大的結構問題。配上 Griffin TTO3 +0.156（5 IP 後就要交給薄牛棚）+ Snelling 大概率短局數 → **本場 6-9 局是兩隊得分分歧的關鍵窗口，預期 high-variance 的 late-game scoring**。但因為 ⏳ short half-life，對手陣容隨日異動，先發退場時間若打亂（Griffin 撐到第 6 / Snelling 提早被換）會反向稀釋此信號；不上修上界（取 +0.2 而非 +0.5）。

## 條件修正

- Park Factor: 106.0 → +0.30 run（loanDepot park 利安打三壘打但 HR -6%，分裂型偏向「多單壘長打、少全壘打」，對 MIA 線速球員 Lopez / Hicks 加分有限）
- 天氣：未公布（loanDepot park 有可開合屋頂，5 月邁阿密通常關屋頂室內條件 — 等同無風 / 70°F 中性）
- 先發 tier / doubleheader：Griffin Solid Starter / Snelling Unknown（first 2026 start，career 樣本 63 BF）→ **對決 tier 落差大但變異也大**；非雙重賽。Snelling 的不確定性使 WSH 得分上限拉開 — 若他 4 IP / 5 BB 提早退場，MIA 牛棚（ERA 3.42）尚可頂住 5 局以上對 WSH 不利。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.7 | +0.2（TTO3 +0.2 + WSH core IL +0.2 − chain break #5-6 −0.2，互動取單側不疊加） | 4.9 |
| AWAY | 5.4 | +0.1（MIA core IL +0.2 − chain break #3-4 −0.1，#3-4 因 House vs LHP .932 部分淡化） | 5.5 |
| Total | 10.1 | +0.3 | 10.4 |

## 整體判斷

- **方向（基本面）**：略偏 HOME（MIA）— 但信號薄，趨近持平
- **總分（基本面）**：~10.4（base 10.1 + 信號 +0.3）
- **方向信心**：~54%（即「微 HOME lean」）。理由：MIA 牛棚 ERA 3.42 vs WSH 4.62（−1.20）+ 主場 PF 106 + Snelling 雖然新人但 WSH 對 LHP 整體 🟢 Weak，這三項是 home edge；對沖項是 Griffin xERA 4.41 強過 Snelling 季外推估、WSH 近 10 場 RS 5.10（攻擊穩過 MIA 3.50）。淨 edge 不大，盤面接近 coin flip。
- **風險**：
  1. **Snelling 樣本太小無法可靠評估**（career 63 BF，本季 0 BF）— 真實表現可能是 4 IP / 5 BB blowup，也可能是 prospect stuff 直接壓制 → 本場最大 single source of variance。
  2. **Griffin Flag 8 結構性偏離**：xERA 4.41 顯示 ERA 2.27 不可持續；若本場 barrel% 重現，5 IP / 4 ER 級別的崩盤情境存在。
  3. **雙方牛棚 core IL ×2**：6-9 局 high-variance scoring，總分上下震盪 ±1.5 run 的可能性偏高。
  4. **打線來源仍為 projected**：兩隊打序未公布，Mead / García / Vivas 等冷棒次的順序若被換動會改變 chain break 結構（特別是若 Mead 提到第 6 棒，HOME chain 會獲緩解）。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組