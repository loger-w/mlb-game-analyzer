## 投手對決

### Jack Leiter (HOME, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p90, K-BB% p79），gap vs ERA-only = +43.7
  - tier_v2 與 ERA-only 落差 +43.7 屬「結構性訊號偏多」：xFIP 3.45 / FIP 4.07 / K-BB% 15.5 都比 ERA 4.61 中後段更樂觀，球速 92.1 mph、whiff% 12.2 屬中段水準，難稱 Elite。AI 採折衷判讀為 🟡 Solid Starter 偏上限，不自動下修預測，但也不全盤接受 Elite 標籤。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 未 fire（vs LHB OPS .760 vs vs RHB OPS .709，Δ +0.051 未達 0.080 門檻；屬正常 RHP platoon），本場可忽略。
- **對手打線威脅**：AWAY 打線 vs RHP 評級 🟢 Weak，威脅集中在 #3-5 三人連續區（Alvarez 1.018 / Walker .905 / Trammell .835）。Leiter FF 36.7% + CH 19.4% + SL 17.7% mix 分散，對下半棒 #7-9（vs RHP OPS 全低於 .580）壓制力佳，主要風險是 Alvarez-Walker 同局上壘串聯。

### Jason Alexander (AWAY, RHP, 33 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = —（—），gap vs ERA-only = —
  - ERA 7.30 vs xERA 3.89 落差 +3.41 觸發 Flag 8，但 FIP 5.05 / xFIP 4.31 / K-BB% 7.0% / 球速 85.8 mph / 近 3 場 ER/IP 10/12.3 仍指向 ⚪ Below Average ~ 🟢 Back-end。xERA 樂觀來自小樣本（vs RHB 僅 18 BF），AI 維持 Below Average，**不因 xERA 自動上修**。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - dossier signal 摘要未列入；raw split vs LHB .343/.410/.486 vs vs RHB .188/.278/.625（樣本小，方向尚屬正常 RHP platoon）。HOME 1-3-5-6 棒有 Pederson/Nimmo/Carter 等多名左打，吃到正向 platoon 而非 reverse。
- **對手打線威脅**：HOME 打線 vs RHP 評級 🟡 Average，對低速 CH-heavy（35.7%）的老將威脅 ↑。Nimmo（.796 vs RHP, last7 .964）、Jung（.840 vs RHP）、Burger（.719, last7 1.018）、Foscue（last7 1.107）中段四人形成集中火力；Alexander vs LHB .896 OPS 弱點將被 HOME 左打陣型放大。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - matchup tier 與 season tier 一致為 🟡 Average，但對 Alexander 這檔等級偏弱的低速 CH-heavy RHP 應微幅上修：last7 OPS 中段四人（Jung .582 例外）多數 > .800，xwOBA 0.337 / OPS .733 屬中段，但 chain OBP top3 .377 / SLG mid .450 顯示前中段串聯尚可。本場評估同意 Average 偏中上限。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - 🟠 chain breaks at #8-9（OPS 落差 0.229，medium）：影響打序回到頂時的 #1 Pederson 是否帶人在壘，但 #1-3 OPS top3 .377 自上壘力尚足，影響有限。heat_vs_babip 未 fire（last7 BABIP 0.325 屬正常區間）。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak
  - matchup tier 🟢 Weak 比 season 🟡 Average 下修一檔，反映對手前 9 棒 vs RHP 嚴重兩極化：#3-5（Alvarez/Walker/Trammell）三人形成 elite 核心區，但 #1-2、#6-9 六人 vs RHP OPS 都 ≤ .664，下半棒幾乎沒上壘能力。本場評估同意 Weak，得分將高度倚賴 #3-5 連續上壘成局。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - 🔴 chain breaks at #8-9（OPS 落差 0.336，high）：壓制下半棒回到 #1 Peña（.574 vs RHP）時帶人在壘的能力。heat_vs_babip 未 fire 為 signal（last7 BABIP 0.258 在 dossier 列為 Flag 3 unlucky-cold 風險），詳見「## 風險提示」段判讀。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.04 / 5 / 1 (Cole Winn) | 5.48 / 8 / 1 (Josh Hader) |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：整體 ERA 3.04 屬聯盟前段（強質），核心 IL ×1（Cole Winn）對應 §牛棚累計效應 🟠 中高一檔，但替補深度足以吸收；面對 AWAY 下半棒 #6-9 都 < .640 OPS，末段 6-9 局封鎖力佳，對總得分壓制方向明確。
- AWAY 牛棚：整體 ERA 5.48 屬聯盟後段（弱質），核心 IL ×1（Josh Hader，60 天）後段防守變薄，對應 §牛棚累計效應 🟠 中高一檔。面對 HOME 中段火力（Nimmo/Jung/Burger/Foscue），先發 Alexander 若 5 局以內退場，6-9 局牛棚失分風險顯著 → 對手末段得分機會 ↑（依 Table B `core_il_count` 1 名 +0.0~0.2 區間）。

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=+3.41):
  - 偏向「樣本噪音 + 部分結構問題並存」：xERA 3.89 來自 vs RHB 僅 18 BF + vs LHB 39 BF 共 57 BF 的微樣本（極不穩），但 FIP 5.05 / xFIP 4.31 / 球速 85.8 / whiff% 9.0 / 近 3 場 ER/IP 10/12.3 都同向指出真實品質仍偏弱。AI 不自動下修對 HOME 得分的預期（base 5.6 保留），但本場仍視 Alexander 為弱質先發。
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.258):
  - 偏「短期低運氣 + 部分中段冷凍並存」：last7 BABIP 0.258 略低於 .260 門檻，但 #3 Alvarez（last7 .583）、#1 Peña（.531）、#8 Matthews（.467）的冷期屬實質、非純運氣。AI 不自動 ±run value，但本場對 AWAY 攻擊面採中性偏保守判讀，不期待整列同步反彈。

### 額外信號
- 🔴 HOME TTO3 penalty：OPS Δ +0.176（TTO1 0.539 → TTO3 0.715），第三輪明顯衰退
- 🔴 AWAY TTO3 penalty：OPS Δ +0.168（TTO1 0.797 → TTO3 0.965），第三輪明顯衰退；K% 從 20.4% 掉到 13.9%（Δ -6.5pp）（career fallback）
- 🟠 HOME chain breaks at #8-9：OPS 落差 0.229
- 🔴 AWAY chain breaks at #8-9：OPS 落差 0.336
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
- 🟠 ⏳ AWAY 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - 雙側 TTO3 penalty 都 fire（HOME Δ+0.176 / AWAY Δ+0.168）+ 雙側牛棚都缺 1 名核心 → 兩位先發都可能 5-6 局退場，後段失分機會雙向放大。但 AWAY 牛棚 ERA 5.48 vs HOME 3.04 質地差距明顯，後段交鋒 HOME 取得結構優勢，採 `tto3_penalty` × `core_il_count` interaction 取單側 max + 0.1 → HOME 得分 +0.3、AWAY 得分 +0.1（受 HOME 牛棚壓制無法完全兌現 TTO3 紅利）。

## 條件修正

- Park Factor: 96.0 → -0.20 run
- 天氣：室內（Roof Closed，不適用）
- 先發 tier / doubleheader：HOME Leiter 🟡 Solid Starter（折衷 tier_v2 Elite 與 ERA-only Back-end）vs AWAY Alexander ⚪ Below Average；單場非 doubleheader；G1（05-25）AWAY 已勝 9-0，本場 G2 系列脈絡 HOME 連敗 ×4 vs AWAY 連勝 ×4 → 動能朝 AWAY 但不入 ±run 紀律。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.6 | +0.4 | 6.0 |
| AWAY | 4.3 | +0.2 | 4.5 |
| Total | 9.899999999999999 | +0.6 | 10.5 |

## 整體判斷

- **方向（基本面）**：HOME（adjusted 6.0 vs 4.5，落差 1.5 run > 0.5）
- **總分（基本面）**：10.5
- **方向信心**：68%
- **風險**：
  1. AWAY 投手 Alexander 觸發 Flag 8（ERA-xERA 落差 +3.41），若實質接近 xERA 3.89 而非 ERA 7.30 → HOME 5.6 base 將被高估，HOME 得分可能下修至 4.5-5.0 區間，方向信心降至 55-60%。
  2. AWAY 打線 last7 BABIP 0.258（Flag 3 unlucky-cold）若反彈 + Alvarez/Walker/Trammell 中段三人連發 → AWAY 得分可能上修至 5.0-5.5，總分維持 10-11 但方向落差縮小。
  3. AWAY 牛棚雖 ERA 5.48 + 缺 Hader，但若 Alexander 撐 6 局以上 → HOME 接觸 AWAY 牛棚局數少，TTO3 + 牛棚 IL interaction 紅利無法兌現，HOME 得分可能停在 5.0 區間。
  4. 雙側 chain_break #8-9 + 室內球場 PF 96 偏投手友善 → 實際總分有低於 9.5 的尾部風險（投手戰可能性 20-25%）。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組