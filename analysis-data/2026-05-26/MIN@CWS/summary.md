## 投手對決

### Sean Burke (HOME, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p77, K-BB% p75），gap vs ERA-only = +24.1
  - tier_v2 偏高合理但需保留：FIP 3.38 / xFIP 3.73 / K-BB% 14.3 確實比 ERA 4.08 強，球種 RV 三球種全正（FF +1.1 / SL +0.6 / KC 還沒亮但作 22.5% 使用率），peripherals 結構性偏 Strong；不過 ERA 與 xERA 同為 4.08 完全一致 → 沒有運氣偏差證據，實際 run prevention 仍落在 Solid 區間，採「Strong-ace 球質但 Solid-starter 結果」中間判讀，**不自動下修預測**（Flag 8）
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 未 fired。vs LHB .742 / vs RHB .649（差 0.093，符合 RHP 對 LHB 略弱常態），AWAY 打線 9 人全右打 → 對位上反而是 Burke 較舒服的方向
- **對手打線威脅**：AWAY 打線 vs RHP 整體僅 🟢 Weak，但 1 棒 Buxton 對 RHP OPS .998 / last7 OPS 1.089 / Barrel% 19.7 處於極熱狀態為單點威脅；2-3 棒 Lee / Larnach（.751 / .837）形成連續上壘段；4 棒之後跌至 .673 以下 + #5-6 chain break 0.256，威脅集中在 1-3 棒；Burke TTO3 OPS 反升 + K% 暴跌 18.4pp，若撐到第三輪正好遇 Buxton 第三次打席，failure mode 明確

### Joe Ryan (AWAY, RHP, 29 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +15.5
  - 同意 Elite Ace 判讀且為結構性：ERA 3.02 vs xERA 2.87（Δ-0.15，無 Flag 8 觸發），FIP 2.38 / xFIP 3.13 / K-BB% 20.9% / WHIP 0.97 全在 Elite 區段，三球種 RV 全強（FF +0.7 / SI +2.2 / CU +2.6），TTO3 OPS 反而暴跌至 .269（Δ -0.315）顯示 sequencing 能撐第三輪。ERA-only 給 Strong 是 surface-level under-counting，tier_v2 校正合理
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 未 fired。vs LHB .612 / vs RHB .505（差 0.107，RHP 對 LHB 弱屬常態方向），HOME 打線左打 Antonacci / Benintendi / Romo 計 3 人但 Antonacci .831 vs RHP、Benintendi .662、Romo .861，無集體 platoon 優勢觸發
- **對手打線威脅**：HOME 打線 vs RHP 升為 🟡 Average（season .697 OPS 但 vs RHP 提升），核心威脅集中 2-3 棒 Murakami（.944 vs RHP / Barrel 20.5%）+ Vargas（.691 vs RHP）+ 4 棒 Montgomery（.805），但 5-9 棒全在 .723 以下且 last7 OPS 多人低迷（Montgomery .313 / Benintendi .546）。Ryan 的 K-BB% 20.9% + TTO3 強壓 .269 → 對這種頭重腳輕陣容應能逐輪壓制，威脅集中在 Murakami 單次 solo 風險

## 打線評級

### HOME — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - 由 season 🟢 Weak 上修一檔至 🟡 Average，主因 1-4 棒 vs RHP 升級（Murakami .944 / Antonacci .831 / Montgomery .805），但 chain top3 OBP .372 / SLG mid .422 偏中性，採「上修一檔」評估，本場面對 Elite Ryan 仍偏壓制
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break fire 在 #8-9（OPS Δ 0.191），位置在 lineup 尾段、低槓桿區，對 chain 連續性影響有限（吃到一次第三輪即可繞回 1 棒）。heat_vs_babip 未 fire（last7 BABIP 0.289 在中性區間）

### AWAY — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak
  - matchup tier 與 season tier 一致為 🟢 Weak，不上修也不下修；2-3 棒 Lee / Larnach OPS 達 .751-.837 為中段亮點，但 4-9 棒整體 vs RHP OPS 多人低於 .700 且 last7 多人偏冷（Clemens .372 / Martin .555 / Gray .437），chain top3 OBP 僅 .342（偏低），整體攻擊仍偏 Weak
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break fire 在 #5-6（OPS Δ 0.256），位置在中段打序，影響「3 棒打出機會後 4-6 棒接續清壘」的能力 → 對 RBI conversion 直接壓制，比 HOME 的尾段 chain break 更傷。heat_vs_babip 未 fire（last7 BABIP 0.261 接近 .260 下緣但未觸發 ≤ .260 門檻）

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.75 / 6 / 2（Hicks IL15d + Vasil IL60d → 🔴 高） | 4.7 / 6 / 1（Sands IL15d → 🟠 中高） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 4.75 已偏高，加上 Hicks（Closer）+ Vasil（HL RP）雙缺 → 對應 `matchup-factors.md §牛棚累計效應` 2 名核心 IL = 🔴 高（明顯吃緊）；末段失分機率上升，若 Burke 在第三輪 TTO penalty 提前下場，後段 6-7 局接的 leverage spot 等於用替補頂 → AWAY 得分 +0.2~+0.5 區間（取上緣 +0.3，反映 Buxton 末段威脅）
- AWAY 牛棚：ERA 4.70 與對手相近，Sands（HL RP）缺陣 → 1 名核心 IL = 🟠 中高（後段防守變薄）；但 AWAY 投手是 Elite Ace Ryan + TTO3 強壓（.269），預期能撐 6+ IP 大幅降低牛棚使用量 → 對 HOME 得分 +0.0~+0.2 區間（取下緣 +0.1）

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ +0.045（TTO1 0.664 → TTO3 0.709），第三輪明顯衰退；K% 從 29.3% 掉到 10.9%（Δ -18.4pp）
- 🟠 HOME chain breaks at #8-9：OPS 落差 0.191
- 🟠 AWAY chain breaks at #5-6：OPS 落差 0.256
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
- 🟠 ⏳ AWAY 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - 本場以 HOME 牛棚 IL ×2（⏳ short half-life）+ HOME TTO3 penalty 雙重 fire 為主軸，兩信號同向（皆放大 AWAY 末段得分），依量級錨點交互規則「不直接相加，取單側 max 區間 + 0.1」→ AWAY 採 +0.3（core IL 上緣）+ +0.2（TTO3 medium 中位）拆計使用合理。AWAY core IL ×1 由 Ryan 預期長 IP 抵消，HOME 僅 +0.1。Flag 3 未觸發、Flag 8 為敘事不入錨點，無雙重壓力衝突

## 條件修正

- Park Factor: 97.0 → -0.15 run
- 天氣：Partly Cloudy, 73°F, wind 7 mph, In From LF
  - 影響判讀：73°F 在 60-85°F 中性區間，球的飛行距離無顯著偏移；風 7 mph 在 < 8 mph 噪音門檻以下，雖然方向是逆風進場 LF，但風速太弱實質影響可忽略；Rate Field PF 97 + HR -1% 本身輕度壓 HR，整體天氣無額外修正（不再 ±run）
- 先發 tier / doubleheader：本場為兩隊正常先發（Burke vs Ryan 皆 11 / 8 GS 賽季常規），dossier 系列脈絡顯示這是 MIN@CWS 系列第 2 場（非 G1/G2 doubleheader 模式，05-25 已先打 G1 Sox 勝 3-1），不需 doubleheader split 調整

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 2.4 | +0.0（AWAY core IL +0.1 − HOME #8-9 chain_break −0.1） | 2.4 |
| AWAY | 3.4 | +0.3（HOME core IL +0.3 + HOME TTO3 +0.2 − AWAY #5-6 chain_break −0.2） | 3.7 |
| Total | 5.8 | +0.3 | 6.1 |

## 整體判斷

- **方向（基本面）**：AWAY（adjusted 3.7 vs HOME 2.4，gap 1.3 run > 0.5 門檻）
- **總分（基本面）**：6.1
- **方向信心**：68%（落在 50-75% 區間內，無需額外辯護）
- **風險**：
  1. Burke tier_v2 +24.1 偏差為本場最大不確定性：peripherals 顯示 Strong-ace 球質、實際 ERA 卻是 Solid，若 Burke 本場展現 FIP-side 真實水平（K-BB% 14.3% 收緊），HOME 失分可能比 base 3.7 顯著少（壓低總分至 5.5 以下、削弱 AWAY 方向）
  2. HOME core IL ×2 為 ⏳ short half-life signal，CWS 可能在 G1 後重新洗牌牛棚 roles，運作模式存在不確定性
  3. Buxton last7 OPS 1.089 為 lucky-hot 風險（雖然 last7 BABIP 0.261 未觸發 Flag 3），若 Buxton 對 Burke 單場 multi-extra-base，AWAY 直接拉高超過 base + 信號區間 → 倒推風險 push 方向過頭
  4. Ryan TTO3 OPS .269 樣本僅 42 BF（small sample），若實際遇到 Murakami 第三輪掉 sequence，Elite 評估有 fragility

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組