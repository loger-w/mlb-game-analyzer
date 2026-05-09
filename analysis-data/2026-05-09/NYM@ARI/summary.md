## 投手對決

### Merrill Kelly (HOME, RHP, 37 📉📉📉 快速退化)
- **Tier 驗證**：腳本 tier_v2 = —（small sample 未計算），gap vs ERA-only = —
  - 同意 ⚪ Below Average，且實際情況更糟：xERA 11.75 > ERA 9.95，Flag 8 era_xera_delta=-1.80 顯示 ERA 還在「美化」真實表現。對應 37 歲 + avg velo 89.1 + barrel% 18.8 + hard_hit% 33.6 → 結構性退化主導，運氣偏差不大。本季 4 GS 樣本雖小，但 trajectory 與年齡退化方向一致，不下修預測。
- **Reverse platoon 信號**：未 fired，但 dossier 顯示 vs LHB .396/.500/.875（58 BF）對 LHB 是災難
  - NYM 本場 top 5 三 RHB（Bichette、Semien、Alvarez）+ 兩 LHB（Baty、Benge），LHB 比例不算高；但 vs RHB 的 .303/.385/.424（40 BF）也已遠高於聯盟平均，無論手別 Kelly 都被擊穿。
- **對手打線威脅**：🔴 高。Kelly 主球種 CH/FF RV/100 雙負（-3.8 / -5.9），近 3 場 ER/IP = 15/14.7（≈9.20 ERA）顯示熱度持續惡化，平均一場很可能 4-5 IP 就被換下。NYM top 6 對 RHP OPS 多在 .550-.730 但 last7 含 Alvarez .802 / Benge 1.067 / Semien .744 多人熱手，對 Kelly 等級的打擊區應該全面開火。

### Clay Holmes (AWAY, RHP, 33 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p77, K-BB% p51），gap vs ERA-only = -27.3
  - 同意 tier_v2，**ERA 1.69 高估真實水平**：xERA 3.80 / FIP 3.55 / xFIP 3.73 都收斂在 ~3.7，Flag 8 era_xera_delta=-2.11 屬運氣偏差為主（whiff% 僅 10.0% 但 hard_hit 29.6% / barrel 4.1% 屬 ground-ball 抗強擊型 — SI 49.7% 高 GB rate 把 BABIP 壓在偏低）。結構面是 Solid Starter 沒錯，不下修預測，但也別把 1.69 當實力指標。
- **Reverse platoon 信號**：未 fired，vs LHB .190/.258/.321 與 vs RHB .176/.253/.221 兩側都壓制
  - ARI top 5 三 RHB（Marte、Perdomo、Vargas）兩 LHB（Carroll、Arenado）。Holmes 兩側都好 → 無論對位都吃虧，無 platoon edge 可借。
- **對手打線威脅**：🟢 低-中。SI 49.7% single-pitch dependent + 對手 ARI matchup tier 🟢 Weak + last7 BABIP 0.203 極冷，Holmes 風格（pitch-to-contact + GB heavy）正好壓制目前無爆擊力 ARI。但 TTO3 K% drop -7.1pp（42 BF career heuristic）暗示第三輪 bat-missing 銳度下降，若 Holmes 撐到 6+ IP 第三輪可能被 ARI 借機串小球。

## 打線評級

### HOME — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak（dossier 表記 Holmes 端 🟢 Weak，與 season Average 比略下修一檔）
  - 對 Holmes 等級 RHP 評估同意 🟢 Weak — top 5 vs RHP OPS 多在 .612-.916，但 last7 全員大幅縮水（Marte .265 / Carroll .541 / Perdomo .660 / Arenado .798 / Vargas .610），近期攻擊熱度與 SI-heavy GB pitcher 對位先天吃虧。本場 ARI 打線實際開火能力偏 🟢 Weak 端。
- **chain_break / heat_vs_babip 信號**：⏳ HOME unlucky-cold（last7 BABIP 0.203）+ chain breaks at #5-6（OPS Δ 0.231）
  - BABIP 0.203 偏低（Flag 3）→ 可能反彈但不自動 ±run，今晚有單局 BABIP 修正空間（比如二壘穿越 / 場地反彈球運氣回正）。chain_break 5-6 表示 4-5 棒（Arenado/Vargas）後到 6-7 棒掉得很重 → 即使 Holmes 在 1-3 棒被破壞，下半段 chain 中斷不易連續得分，整體 cap 在 2-3 run 以內。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average（dossier 表記 Kelly 端 🟡 Average）
  - 同意 🟡 Average — 但 Kelly 此刻是 ⚪ Below Average 的「等級下打靶」對位，NYM Average tier 對 ⚪ tier 投手按錨點屬「嚴重落差」；上修為對位有利，預期得分大幅高於 vs 平均 RHP 的基線。Alvarez（last7 .802 / 17.3% barrel）+ Benge（last7 1.067）+ Semien（last7 .744）三熱手對 Kelly 主球種 CH/FF（RV/100 雙負）有結構性命中率優勢。
- **chain_break / heat_vs_babip 信號**：🔴 AWAY chain_break at #7-8（OPS Δ 0.336，high severity）
  - Bottom 7-9 NYM 板凳/底序明顯弱（season 整隊 RS 2.93 / 30 場印證），即便 top 6 把 Kelly 打爆，輪到下半棒次接續無威脅 → 一輪內 5-6 分得分上限，但兩三輪累積仍可進帳 7-8 分。對 cap 總分有壓制作用（heuristic：-0.2 ~ -0.3）。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.51 / 6 / 3 (🔴🔴 崩盤級) | 3.93 / 7 / 3 (🔴🔴 崩盤級) |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（ARI）：A.J. Puk + Andrew Saalfrank + 1 名 core IL，剩餘可用組合品質明顯下降，加上整體 ERA 4.51 → 後段易失分。Kelly 預期早下（4-5 IP），中後段 5+ IP 需牛棚 carry → NYM top 6 第二輪 / 第三輪可能再對 ARI 牛棚開火，總分上修壓力主要來自此處（Table B core_il_count 3+ → +0.4~0.8 對 NYM 末段得分）。
- AWAY 牛棚（NYM）：A.J. Minter + Dedniel Núñez + 1 名 core IL，整體 ERA 3.93 仍優於 ARI，但末段 high-leverage 角色吃緊。Holmes 較可能撐 5-6 IP，NYM 牛棚負擔輕；ARI 攻擊本身極冷（last7 BABIP .203）即便對牛棚弱點也未必能爆 → +0.2~0.3 對 ARI 末段，但實質拉抬有限。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-1.80):
  - **結構性退化主導**：Kelly 37 歲 + avg velo 89.1 mph + barrel% 18.8 + hard_hit% 33.6 + 主球種 RV/100 雙負 → ERA 9.95 是真實水平（甚至 xERA 11.75 顯示更糟）。本場判斷站在「Kelly 大概率被打爆」這一側，不上修也不下修預測。
- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-2.11):
  - **運氣偏差主導**：Holmes ERA 1.69 但 xERA 3.80 / FIP 3.55 / xFIP 3.73 都收斂在 3.5-3.8。SI 49.7% high GB rate + barrel% 4.1% 把 BABIP 壓低了，但 whiff% 僅 10.0% 不是真 ace。本場判斷視 Holmes 為 🟡 Solid Starter（不是 🟠 Strong Ace），預期失分 2-3.5 區間而非 ERA 1.69 的延伸。
- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.203):
  - **可能反彈但仍在弱攻擊期**：BABIP .203 偏低（聯盟均 ~.290），統計上有正向回歸空間；但 ARI top 5 last7 OPS 全員低（Marte .265 / Carroll .541 → 屬整隊冷期非單純運氣壓低）。本場可能單局 BABIP 修正、但整體仍預期 2-3 run 區間，不自動加減 run value。

### 額外信號
- 🟠 AWAY single-pitch dependent：主球種使用率 49.7%（≥45.0%）
- 🟠 AWAY TTO3 penalty：OPS Δ +-0.018（TTO1 0.566 → TTO3 0.548），第三輪明顯衰退；K% 從 19.0% 掉到 11.9%（Δ -7.1pp）
- 🟠 HOME chain breaks at #5-6：OPS 落差 0.231
- 🔴 AWAY chain breaks at #7-8：OPS 落差 0.336
- 🔴 ⏳ HOME 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
- 🔴 ⏳ AWAY 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
  - 雙方牛棚都 core IL ×3 → 後段都吃緊。但 Holmes 比 Kelly 撐得久，NYM 牛棚實際使用 IP 預期 < ARI 牛棚 → 影響不對稱：ARI 受牛棚崩盤級壓力遠大於 NYM。配合 Flag 3 ARI 打線冷期，NYM 後段牛棚即便弱也未必被引爆；反向 ARI 牛棚需頂 5+ IP 對 NYM 熱手 top 6 機率高出局，是本場高 variance 的最大來源。

## 條件修正

- Park Factor: 101.0 → +0.05 run
- 天氣：未公布（跳過天氣分析；Chase Field 5 月通常 retractable roof closed → 室內中性環境機率高）
- 先發 tier / doubleheader：先發 tier mismatch 顯著（Kelly ⚪ Below Average vs Holmes 🟡 Solid Starter），是本場最大單一變因；非 doubleheader。系列脈絡：ARI 0-1 NYM（G1 在主場輸 1-3），系列已落後且面對遠優於己方的先發。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.8 | +0.2（NYM bullpen core IL ×3 +0.3 / Holmes pitch_mix +0.1 / chain_break 5-6 -0.2，cap interaction） | 4.0 |
| AWAY | 8.7 | +0.2（ARI bullpen core IL ×3 +0.5 / chain_break 7-8 high -0.3） | 8.9 |
| Total | 12.5 | +0.4 | 12.9 |

## 整體判斷

- **方向（基本面）**：AWAY (NYM) 勝
- **總分（基本面）**：12.9 run（4.0–8.9 拆分；formula base 12.5 微幅上修）
- **方向信心**：62%（投手對位 mismatch 大，但 Kelly 4 GS 小樣本 + Holmes 1.69 有運氣成份 + 雙隊 bullpen 高 variance → 信心未到 70%）
- **風險**：
  1. **Kelly 突然反彈** — 4 GS 樣本太小，xERA 11.75 雖支持結構性退化結論，但 single-game 偶有 5 IP 2 ER 的均值回歸出現；若發生 NYM 預期失分大幅縮水。
  2. **Holmes 真實水平 ~3.7 ERA** — 1.69 是 SI/GB 運氣的延伸，今晚 ARI 即便冷期，1-2 個 BABIP 修正穿越 + 牛棚崩盤級壓力 → ARI 有條件衝 4-5 分。
  3. **雙方 bullpen core IL ×3** — 高 leverage 後段都吃緊，blowout 與 late-inning 翻盤都有空間，total 區間實際可能落在 9-14 而非單點 12.9。
  4. **打線未公布（projected）** — top 5 PA 排序近似，若 Lindor / Soto 之類熱手有實際排在 1-2 棒（dossier projected 未顯示 Lindor/Soto，需查實際公布）會放大 NYM 得分上限；亦可能 Mendoza 因連勝期雪藏 Alvarez 等核心。建議賽前 ~1 小時再 refresh 確認 official 打序。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組