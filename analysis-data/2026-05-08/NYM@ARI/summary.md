## 投手對決

### Ryne Nelson (HOME, RHP, 28 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟢 Back-end Starter（xFIP p22, K-BB% p53），gap vs ERA-only = +33.9
  - 同意 tier_v2。gap +33.9（high）= ERA 6.61 嚴重低估底層，但 xFIP 4.63 / K-BB% 10.5% 顯示真實水平在 Back-end 中段、不是 Below Average。原因：FF 61.4% 單球種依賴（FF RV/100 -0.90、SL RV/100 -2.60 兩主球種皆負分）+ hard_hit% 29.3% / barrel% 12.9% / GB% 27.5%（飛球率高、品質差），是「結構性中後段+運氣偏差混合」綜合：本場以 xFIP ~4.6 為錨，**不額外下修預測**。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 未 fire。vs LHB .792 OPS / vs RHB .835 OPS（Δ 0.043，未達 0.080 門檻），雙側均偏高、無實質手別優勢。
- **對手打線威脅**：🟡 中。NYM 雖 season tier Average，但 vs RHP 落到 🟢 Weak — Bichette .585 / Semien .603 / Baty .590 / Benge .606 全 ≤ .610；唯 Alvarez（.741 vs RHP / last7 .721、BABIP .438）有 lucky-hot 嫌疑。Nelson 的 FF 單球種 + barrel% 12.9% 給長打留漏洞，但 NYM 缺後段串聯（chain break #7-8 落差 0.376 high），即使咬到也須集中在 1-5 棒。

### Nolan McLean (AWAY, RHP, 24 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +16.2
  - 同意 tier_v2，但帶 small-sample 警告（7 GS / 39.3 IP / 24 yo）。gap +16.2（medium）= ERA 2.97 已強，xERA 2.32 / xFIP 2.17 / xwOBA 0.244 / xBA 0.195 / hard_hit% 17.8% / barrel% 5.6% / GB% 60.9% 全部頂級 — **結構性 Elite，非運氣堆砌**。SI 36.6%（RV/100 +3.10）為主球，5-pitch 配置（balanced）抗 platoon。**不下修預測**，但單場仍受 7 GS 樣本量自然回歸風險約束。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 未 fire。vs LHB .508 OPS / vs RHB .529 OPS（Δ 0.021），雙側均壓制、無 reverse 風險。
- **對手打線威脅**：🟢 低。ARI vs RHP grade 🟡 Average（season tier 🟢 Weak 上修），但 last7 全隊 OPS 普遍降溫（Marte last7 .265 / Carroll .587 / Arenado .581）。對 McLean 此種 SI+GB 型投手（GB% 60.9%、xBA 0.195）軟擊偏多、滾地球壓制，命中率本就低。chain break #5-6 落差 0.245 medium 進一步拉低中段串聯。唯一 sticky：第 5 棒 Vargas vs RHP .939 OPS 個別高威脅；但 4/9 上次對戰 McLean 投了 6.3 IP / 2 ER / 8K 已壓制 ARI 一次（小樣本參考）。

## 打線評級

### HOME — season tier 🟢 Weak / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - 上修一檔處理：season Weak 但 vs RHP 為 Average — Carroll / Perdomo / Vargas 三人 vs RHP 表現比 season 更穩，整體對 RHP 抵抗略強。本場對 McLean 可從 Average 起步；惟 last7 BABIP 0.201 大幅低於 league .290（>1σ）→ 屬「冷期+運氣偏差」並存，實際手感介於 Weak~Average 之間（Flag 3 紀律不自動 ±run）。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #5-6 medium fired（Δ 0.245）：4 棒 Arenado .753 → 5 棒 Vargas .995 後 6 棒以下急速衰退，ARI 中段攻勢只能靠 1-5 棒一波到底；若 1-3 棒上壘失靈（Marte last7 .265 / Carroll last7 .587），單局攻擊串聯易斷。heat_vs_babip 點 unlucky-cold ⏳：last7 BABIP 0.201 偏低為敘事 flag，**不自動 ±run**（Table A 紀律）；但對手 McLean 是 GB 型高壓制投手，本場反彈空間有限。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak
  - 下修一檔處理：season Average 但 vs RHP 落到 Weak — Bichette / Semien / Baty / Benge season OPS 全 ≤ .606 且 vs RHP 同樣低迷，並非偶發冷期，本季結構性問題。本場對 Nelson 從 Average 下修為 Weak；唯一不下修的是 Alvarez（vs RHP .741、last7 .721 但 BABIP .438 含運氣成分）。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #7-8 high fired（Δ 0.376）：6 棒之後攻擊近乎斷層，NYM 只能靠 1-5 棒（Bichette/Semien/Alvarez/Baty/Benge）製造分數，6-9 棒實質「白洞」。對 Nelson 這種會給長打的投手是反向制衡 — 即便 Nelson 失球，也須集中於前段才有效。heat_vs_babip 未 fire（last7 BABIP 0.289 中性）。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.56 / 6 / 3+ 名（崩盤級） | 4.04 / 7 / 3+ 名（崩盤級） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：🔴🔴 極高風險（崩盤級）。core IL ×3（A.J. Puk + Andrew Saalfrank + 1 名）— closer / setup 雙線缺。Nelson 近 3 場 IP 4.7/4.7/5.7 → 6th 起牛棚必接，前 6 局後最多 4 局牛棚要扛。對 NYM 1-5 棒（Alvarez 為主）末段有實質得分機會，但 NYM 後段亦弱（chain break #7-8）— 對應有限。
- AWAY 牛棚：🔴🔴 極高風險（崩盤級）。core IL ×3（A.J. Minter + Dedniel Núñez + 1 名）— 左投高槓桿 + 右投 setup 同時缺。McLean 近 3 場 IP 5.0/5.3/6.3 → 同樣 6th 起換投，整晚 NYM 後段牛棚要對 ARI 1-5 棒（Marte/Carroll/Perdomo/Arenado/Vargas）封鎖 12+ 個打席，是 NYM 勝負的最大隱患。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.201):
  - 偏「可能回歸」但本場反彈空間有限：BABIP 0.201 比 league .290 低 0.089（>1σ），純粹打不到的低極值不該持續。但本場面對 GB% 60.9% / xBA 0.195 / hard_hit% 17.8% / barrel% 5.6% 的 McLean 是 worst case — 軟擊+滾地球本就少安打、硬擊接觸又壓得低 → 短期 BABIP 自然回升至均值的窗口在本場被結構性壓縮。Flag 3 紀律不自動 ±run，**敘事保留 ARI 上限略高於 base 的可能性**（特別是 Vargas/Arenado 個別硬擊）。

### 額外信號
- 🟠 HOME single-pitch dependent：主球種使用率 61.4%（≥45.0%）
- 🟠 HOME TTO3 penalty：OPS Δ +0.107（TTO1 0.745 → TTO3 0.852），第三輪明顯衰退；K% 從 22.1% 掉到 15.8%（Δ -6.3pp）（career fallback）
- 🔴 AWAY TTO3 penalty：OPS Δ +0.355（TTO1 0.442 → TTO3 0.797），第三輪明顯衰退；K% 從 34.0% 掉到 29.6%（Δ -4.4pp）（career fallback）
- 🟠 HOME chain breaks at #5-6：OPS 落差 0.245
- 🔴 AWAY chain breaks at #7-8：OPS 落差 0.376
- 🔴 ⏳ HOME 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
- 🔴 ⏳ AWAY 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
  - 雙隊牛棚 IL ×3 同時 fire — directional 偏移互相抵銷（雙方都得早換投、雙方都有崩盤風險），但**單側 variance 顯著放大**：任何一隊 7-9 局崩 → 總分跳 1-3 runs。AWAY TTO3 penalty 高度但走 career fallback 71 BF，confidence: heuristic（McLean 樣本太少），實際本場 NYM 教練多半 5-6 IP 就把 McLean 換掉 → TTO3 penalty 不易兌現，但下游 NYM 牛棚崩盤風險取代之。Flag 3 + Flag 8 + 雙牛棚崩盤三層疊加 → 本場 total 信心應降一檔。

## 條件修正

- Park Factor: 101.0 → +0.05 run（中性偏微，HR -18% 微利投手抑長球）
- 天氣：未公布（Chase Field 有伸縮屋頂，5 月 ARI 通常關閉，視為室內中性，跳過天氣分析）
- 先發 tier 落差：McLean 🔴 Elite Ace vs Nelson 🟢 Back-end Starter — 跨 2 個 tier，本場最大單因子。doubleheader：無。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 2.1 | +0.4（gain max(core_IL_AWAY +0.5, AWAY_tto3_career +0.2) +0.1 = +0.6；suppress own chain_break −0.2） | 2.5 |
| AWAY | 5.9 | +0.3（gain max(core_IL_HOME +0.5, HOME_tto3 +0.3, HOME_single_pitch +0.2) +0.1 = +0.6；suppress own chain_break −0.3） | 6.2 |
| Total | 8.0 | +0.7 | 8.7 |

## 整體判斷

- **方向（基本面）**：**AWAY（NYM 勝面）**。投手 tier 跨 2 階（🔴 Elite Ace vs 🟢 Back-end Starter）為本場最大單因子；雖 NYM vs RHP 落為 Weak 但仍能利用 Nelson 的 FF 61.4% 單球種依賴 + TTO3 衰退。ARI 攻擊面在 last7 冷期 + 面對 GB 型 McLean 處於最差時機（BABIP 反彈空間結構性受限）。
- **總分（基本面）**：~8.7 runs（HOME 2.5 + AWAY 6.2）。雙牛棚崩盤級疊加 → variance 顯著放大；7-9 局任何一隊崩可能再推高 1-3 runs，因此實際分布為「8.7 中位數，右尾偏厚」。
- **方向信心**：62%（NYM 勝面）。投手 tier 落差 = 主軸。下修因素：(1) McLean 7-GS 樣本回歸風險、(2) ARI BABIP 自然反彈、(3) Chase Field PF 101 中性、(4) Nelson 4/8 vs NYM 才剛投出 5.7 IP / 1 ER，可能 carry over。
- **風險**：
  1. **McLean small-sample 回歸**：7 GS / 39.3 IP / 24 yo，xFIP 2.17 雖 elite 但單場仍受變異約束；若被 Vargas/Arenado 中段串聯打 1-2 大局即可改寫劇本，個別硬擊（EV ≥ 95% 32.2% 偏高）是漏洞。
  2. **ARI BABIP 0.201 自然反彈**：last7 極端值不該持續，但對 GB+barrel 雙低的 McLean 反彈空間結構性受限 — 屬「該標、未必 fire」的風險，仍保留個別硬擊變現可能。
  3. **雙牛棚崩盤級**（兩隊都 core IL ×3）：6th 起雙方都靠次級牛棚扛 4 局，O/U 上行 variance 高；任何一邊 7-9 局崩 → total 可能跳 1-3 runs，O/U 邊不應重壓。
  4. **Nelson vs NYM 4/8 carry over**：上次 5.7 IP / 1 ER 表現遠勝 ERA 6.61，若 NYM 整體冷期持續 + Nelson FF 命中率好 → 可能上修為 Solid 級單場表現，吃掉 NYM 勝面 10-15pp。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
