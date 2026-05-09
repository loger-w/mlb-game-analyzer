## 投手對決

### Max Meyer (HOME, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p86），gap vs ERA-only = +7.1
  - tier_v2 與 ERA-only 同向（gap +7.1 < 15，未觸發 tier_mismatch），同意 Elite Ace 評級。但 ERA 2.68 / xERA 3.89 落差 1.21（Flag 8 區），表示目前數據含 BABIP / HR 運氣成分，xFIP 3.19、FIP 2.69、K-BB% 17.6 仍支持頂尖體質但實際水準更接近「強 Solid Starter」。
- **Reverse platoon 信號**：未 fire（vs LHB .181/.263/.306 80 BF / vs RHB .185/.270/.262 74 BF，平台幾乎中性）
  - 平台中性意味著左打打線無平台優勢可依賴；BAL 中心打者（Henderson L、Alonso R、Basallo L、Ward R）混和組成不能集中對 Meyer 製造左打火力。
- **對手打線威脅**：BAL 整體 vs RHP tier 🟢 Weak（projected），Alonso vs RHP .864 + last7 OPS 1.328（💪 火燙）為唯一明確威脅；Henderson 季 .691 / last7 .344（⚖️ 冷）、Ward last7 .578 — 中段棒次熱度不足，難對 Meyer 形成連續壓力。

### Cade Povich (AWAY, LHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = —（樣本/血統不足以混合 xFIP-blend），script tier 🟢 Back-end Starter
  - 同意 Back-end 評級。ERA 4.41 / FIP 5.06 / xFIP 4.10 → 真實水準介於 Back-end~Solid 邊界，xFIP 表示 HR 運氣偏負（FIP 比 xFIP 高 0.96），但 K-BB% 8.8 + WHIP 1.29 仍是底層配置。
- **Reverse platoon 信號**：未 fire（vs LHB BF 28 < 30 門檻），但 vs LHB **.333/.357/.667** vs vs RHB .194/.275/.306 — 數值上嚴重 reverse（左打反而打爆他）
  - 28 BF 小樣本要打折，但 MIA 打線左右組成關鍵：若 X. Edwards（S）、Marsee 等左打居前段，BAL 投手對 LHB 表現弱化的隱憂存在；不過 MIA 主力 Otto Lopez vs LHP 1.005、Hicks vs LHP .511 — vs LHP 整體只有 Lopez 強 → reverse signal 對本場威脅有限。
- **對手打線威脅**：MIA 整體 vs LHP tier 🟢 Weak、xwOBA .302（projected），核心威脅集中在 Otto Lopez（vs LHP 1.005、last7 .931、BABIP .417 含運氣）；其他主力（Marsee .553、Edwards vs LHP .694、Hicks vs LHP .511）對 LHP 都未必占優 → 整體威脅可控。

## 打線評級

### HOME — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - matchup tier 與 season tier 同檔（Weak），方向**同意**，本場 MIA 打線不會因 vs LHP 而升級；唯一 outlier 是 Otto Lopez vs LHP 1.005，需重點對位處理。
- **chain_break / heat_vs_babip 信號**：🔴 chain breaks at #5-6（OPS 落差 0.361，high 級）
  - 影響本場攻擊 chain：1-3 棒（Marsee/Edwards/Lopez）OBP top3 .359 還算過得去、4 棒（Norby）.719 仍可，但 5-6 棒銜接斷裂壓制大局得分上限；Lopez（#3）若被 IBB 或謹慎對付，後續清壘乏力。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak
  - matchup tier 比 season tier **下修一檔**（Average → Weak），方向**下修**：BAL 整體對 RHP 不如對 LHP，加上面對 Elite Ace 體質的 Meyer，本場攻擊上限再壓一檔。
- **chain_break / heat_vs_babip 信號**：🟠 chain breaks at #8-9（OPS 落差 0.258，medium 級）
  - 影響 chain 末段：8-9 棒接 1 棒回轉時容易斷掉，但對核心 1-5 棒影響小；對單場得分壓制 effect 有限。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.45 / 3 / 2（Fairbanks Closer + Henriquez HL RP）| 4.82 / 7 / 2（Bautista Closer + Helsley Setup）|

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（MIA）：底層 ERA 3.45（聯盟前段），但核心 IL ×2（Fairbanks Closer + Henriquez HL RP）→ 🔴 高影響度。9 局 closer 角色實質空缺，後段對 Alonso / Henderson 這類能 1 球決勝的右打者風險明顯放大。
- AWAY 牛棚（BAL）：底層 ERA 4.82（聯盟後段），核心 IL ×2（Bautista Closer + Helsley Setup）→ 🔴 高 + 雙重壓力（基線本就薄）。後段對 Otto Lopez / Hicks 這類 vs LHP 火燙打者極具失分風險，是本場賽果最大不確定因素之一。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🔴 HOME TTO3 penalty：OPS Δ +0.275（TTO1 0.644 → TTO3 0.919），第三輪明顯衰退；K% 從 27.7% 掉到 15.0%（Δ -12.7pp）（career fallback）
- 🔴 AWAY TTO3 penalty：OPS Δ +0.253（TTO1 0.686 → TTO3 0.939），第三輪明顯衰退；K% 從 25.0% 掉到 15.4%（Δ -9.6pp）（career fallback）
- 🔴 HOME chain breaks at #5-6：OPS 落差 0.361
- 🟠 AWAY chain breaks at #8-9：OPS 落差 0.258
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
- 🔴 ⏳ AWAY 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - 雙方都吃 Flag 3/8 之外的雙重壓力：（1）兩位先發 TTO3 penalty fire（career fallback heuristic 級），第三輪 OPS 都漲到 0.919/0.939 等級 → 教練很可能在 5-6 局提前換投；（2）兩隊核心牛棚都 IL ×2，BAL 一側基線 ERA 4.82 更危險。預期賽事走向：兩位先發都不會超過 18-20 BF，比賽中後段（6 局後）成為牛棚對決，本場「後段失分爆量」風險顯著高於一般場次，總分判讀偏多。

### 額外風險（敘事，不入錨點）
- Meyer ERA 2.68 / xERA 3.89（Flag 8 區）：實際水準接近「強 Solid Starter」非絕對 Elite，BAL 雖打線一般但仍可能比 ERA 預期得分更多；不自動下修預測，但壓「Meyer 完全鎖死 BAL 打線」的劇本要打折。
- Povich vs LHB .667 SLG 28 BF：小樣本 reverse split 隱憂，若 MIA 主力左打（Edwards / Hicks）找到節奏可能爆量，但 28 BF 不足以當穩定 signal。
- 動能：BAL 連勝 +2（連兩場勝 MIA），MIA 連敗 -4；G1 7-4 BAL 客勝 → 本場心理動能偏 BAL，但勿過度加權。

## 條件修正

- Park Factor: 106.0 → +0.30 run（loanDepot park HR -6%；偏壓 HR、利安打/三壘打型得分）
- 天氣：未公布（跳過天氣分析；loanDepot park 為可開閉室內球場，預設受限）
- 先發 tier / doubleheader：先發 tier 落差 Elite Ace（Meyer）vs Back-end Starter（Povich）已大量反映於 base formula（HOME 5.5 / AWAY 3.1，差 2.4 run）；非 doubleheader。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.5 | +0.2（AWAY core_il +0.4 取上界 + AWAY TTO3 +0.1 → 同向 cap +0.5；HOME chain_break #5-6 high −0.3）| 5.7 |
| AWAY | 3.1 | +0.3（HOME core_il +0.3 + HOME TTO3 +0.1 同向 cap +0.4；AWAY chain_break #8-9 medium −0.1）| 3.4 |
| Total | 8.6 | +0.5 | 9.1 |

## 整體判斷

- **方向（基本面）**：HOME（Miami Marlins）
- **總分（基本面）**：9.1（base 8.6 + 信號淨 +0.5；雙方牛棚薄 + TTO3 雙 fire 推升後段失分）
- **方向信心**：60%
  - 支持 HOME：Meyer xFIP-blend Elite vs Povich Back-end，先發品質落差 2.4 run；BAL 打線 vs RHP 平庸；MIA 主場 PF 106 略加成。
  - 不利 HOME（壓信心）：Meyer ERA 2.68 / xERA 3.89 含運氣回歸風險（Flag 8）；MIA 自身 chain breaks #5-6 high 壓低得分上限；BAL 連勝動能 +2 + G1 已 7-4 客勝。
- **風險**：
  1. 雙方先發都帶 TTO3 penalty + 雙方核心牛棚 IL ×2 → 6 局後牛棚對決變數放大，總分易爆量（押 Total 偏多較有依據）。
  2. Meyer Flag 8 ERA-xERA gap：若 BABIP 回歸，BAL 中段打線可能比預期多敲 1-2 分。
  3. Povich vs LHB 28 BF .667 SLG 小樣本隱憂：若 MIA 左打（Edwards / Hicks）找到對位 → MIA 爆量勝。
  4. 打線皆未公布（projected）：Otto Lopez 是否 #3 / BAL 中心棒次手別組成將顯著影響 chain 與平台對位實際走向，賽前 2-4h 公布後可重新校準。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
