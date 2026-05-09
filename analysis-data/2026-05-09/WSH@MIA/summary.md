## 投手對決

### Janson Junk (HOME, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p82, K-BB% p63），gap vs ERA-only = -6.3
  - 同意 Strong Ace tier，但屬於「稍微高估」一側。ERA 2.82 vs FIP 3.33 / xERA 3.32 ~0.5 ER 差距 → 有 BABIP / 殘壘運氣支撐；近 3 場 ER/IP = 8/16.7（ERA 4.32）也顯示往 peripherals 收斂的跡象。peripherals 實際接近 Solid Starter 上沿。gap 落在 |6.3| < 15 → tier_mismatch 未觸發，formula 不下修，但敘事側留意「ERA 過低估強度」風險。
- **Reverse platoon 信號**：未 fire（vs LHB .739 OPS / vs RHB .401 OPS 屬正常 RHP platoon 走勢，差距 .338 是放大版而非反向）
  - 雖未 fire，本場 WSH 上半段 Wood / Lile / Abrams 三連左打（前 3 棒 OBP top3 = .366）正好對位 Junk 弱側 vs LHB SLG .417 → 「正常 platoon 放大」對 Junk 不利
- **對手打線威脅**：威脅集中前 3 棒。Wood (vs RHP .914, **EV95% 60.7% / Barrel% 25.8%** — Statcast 怪物等級) + Abrams (vs RHP 1.054, top of league) 是核心爆破點；前 3 棒 OBP top3 .366 + 多左打對位 → Junk 第一輪即可能失分。中後段 (House .583 / Young .595 vs RHP) 對 RHB 嚴重弱化，Junk vs RHB .401 OPS 能高效壓制 → 第 4-9 棒對位形成「製造機會難、清壘更難」的雙重瓶頸。Junk TTO2 OPS .823 (career 弱點) → 第二輪上半段 (Wood/Abrams) 是真正危險區。

### Zack Littell (AWAY, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = ⚪ Below Average（xFIP p5, K-BB% p11），gap vs ERA-only = +10.0
  - 同意 Below Average tier，但屬於「過度低估」一側。ERA 7.24 vs xFIP 5.09 = 2.15 ER 差距 → BABIP / HR 不運氣放大；近 3 場 ER/IP = 7/15.0（ERA 4.20）顯示朝 xFIP 收斂。peripherals 實際是 Back-end 後段先發等級。gap |10.0| < 15 → tier_mismatch 未觸發，但敘事側留意「base formula 用 ERA 7.24 推 MIA 得分 9.4 偏高」的可能。
- **Reverse platoon 信號**：未 fire，但對 LHB 慘況極端（vs LHB .319/.380/**.750**, OPS 1.130；vs RHB .766 OPS）
  - 屬「正常 platoon 放大版」非 reverse。MIA 中段 Edwards / Lopez / Hicks 等若有左打或開放打席對 RHP 強 → 災難級放大；Hicks vs RHP OPS 1.034 + Lopez vs RHP .835 → Littell 對位 MIA 中段是賠率最差的對位
- **對手打線威脅**：MIA 中段（2-5 棒）對 Littell 為 RHP-killer 集合：Edwards (.907) / Lopez (.835, last7 BABIP **.440**) / Norby (.798, last7 BABIP **.500**) / Hicks (1.034, EV95% 40.6%) 全在 vs RHP 強度。Lopez / Norby 的 last7 BABIP 含明顯運氣成分（heat_vs_babip 未自動 fire 但接近邊緣）→ 可能延續或回歸；Hicks 與 Edwards 是 EV95% / Barrel 結構支撐的真實強度。Littell whiff% 5.6 / hard_hit% 31.0 / barrel% 14.0 三項都偏弱 → 沒有壓制工具。預期 4-5 IP 前即可被擊出 ER。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - season 與 matchup tier 一致 (Average)。但對 Littell 這種「弱化版 RHP」(xFIP 5.09 / 對 LHB SLG .750) → 場景特定上修約 0.5-1.0 run；中段 (Edwards / Lopez / Hicks) 對位優勢明顯。tier 不動，但本場攻擊期望值高於 season 平均。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - 🔴 HOME chain breaks at #5-6 (OPS 落差 0.364, high)：Hicks (#5 .948 OPS) → #6 落差顯著。1-5 棒能對 Littell 製造威脅（特別是 2-5 棒 OPS 都 ≥ .798 vs RHP），但若得分串聯仰賴 5-6 棒延續就斷裂。對「滿壘 / 兩出局後 2-3 棒延續」場景影響有限，但壓制連續長打鏈，影響大局得分上限約 -0.2 ~ -0.3 run。Lopez / Norby last7 BABIP 0.440 / 0.500 雖未自動觸發 heat_vs_babip，仍提示 last7 OPS 含運氣，可能 1-2 局內回歸。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - season 與 matchup tier 一致 (Average)。對 Junk 此種 RHP（vs LHB OPS .739 / vs RHB OPS .401, 差距 .338）→ 若投射打線中 LHB 占多（Wood / Lile / Abrams 前 3 棒推測左打）→ 場景特定上修約 0.2-0.4 run；但 4-5 棒 House / Young 對 RHP 嚴重弱化（.583 / .595）→ 拉平整體預期。tier 不動，期望值與 season 接近。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - 🟠 AWAY chain breaks at #3-4 (OPS 落差 0.241, medium)：Abrams (#3 .931) → House (#4 .690)。1-3 棒（Wood + Abrams）能上壘但 4 棒 House 清壘能力弱化 → 「壘上有人但回不來」風險。實際得分仍依賴 Wood / Abrams 自摸長打或 1-3 棒連環安打；4-5 棒主要做為延續、保送性質。影響大局得分上限約 -0.1 ~ -0.2 run。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.32 / 3 / 2 (Closer Fairbanks + setup Henriquez) | 4.57 / 7 / 2 (Beeter + Kranick, 皆 high-leverage RP) |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（MIA）：結構 ERA 3.32（聯盟前段）。可用性中等：失去 closer Fairbanks 後 9 局收尾與 7-8 局 high-leverage 槓桿吃緊。近 3 天消耗應不大（G1 兩隊 3-2 收，低分賽）。Junk 預期能撐 5-6 IP（FIP 3.33 結構穩） → 牛棚進場時間 6-7 局，仍能用結構性深度（ERA 3.32）對付 WSH 中後段 (House / Young 對 RHP 弱)。但 Wood / Abrams 第三、四個打席若回到 high-leverage 對位空缺 → 單發長打風險上升。整體仍屬可控但 closer 缺席是邊際失分點。
- AWAY 牛棚（WSH）：結構 ERA 4.57（後段）+ 投手 IL 7（含 2 名 core）→ **雙重壓力**。加上 Littell 7.24 ERA / 場均 ~3 IP/start → 預計 4-5 局即下場 → 牛棚負擔極重，可能用到 long relief / 第 4-5 棒層級 RP。MIA 中段（Edwards / Lopez / Hicks）若進入後段對位 → 持續加分機會大增。後段 6-9 局期望失分明顯放大，是本場 Total 偏多的主要結構推力。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ +0.005（TTO1 0.667 → TTO3 0.672），第三輪明顯衰退；K% 從 21.2% 掉到 14.9%（Δ -6.3pp）（career fallback）
- 🟠 AWAY TTO3 penalty：OPS Δ +-0.052（TTO1 0.776 → TTO3 0.724），第三輪明顯衰退；K% 從 21.8% 掉到 17.0%（Δ -4.8pp）（career fallback）
- 🔴 HOME chain breaks at #5-6：OPS 落差 0.364
- 🟠 AWAY chain breaks at #3-4：OPS 落差 0.241
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
- 🔴 ⏳ AWAY 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - 兩隊都觸發 core_il_count 🔴 高，但結構落差不對稱：MIA 牛棚結構 ERA 3.32 + Junk 預計撐長 → 即便缺 closer 仍有結構支撐；WSH 牛棚 ERA 4.57 + Littell 預計早退（5 GS 平均 3 IP）→ 後段對位明顯不利。雙重壓力疊加 → WSH 後段失分風險顯著放大，是 Total 偏多的主要推力之一。Flag 8 (Junk gap -6.3 / Littell gap +10.0) 雖均未自動 fire，但接近邊緣 → 兩端 ERA 都在 sanity rail 範圍內，formula base 9.4/4.0 可能向 peripherals 收斂（HOME 攻擊下修、AWAY 攻擊微上修）。

### TTO3 signals 補充判讀（雙方均屬弱信號）
- Junk TTO3：OPS Δ +0.005 等於持平，僅 K% 掉 6.3pp 觸發；TTO2 OPS 0.823 才是真正弱點（career, 114 BF）→ Marlins 教練在 TTO2 即可能換投，但 closer 缺席 → 換投時機受限
- Littell TTO3：OPS Δ -0.052（反向「改善」），僅 K% drop 觸發 → 「越投越爛但都很爛」型；無實質第三輪懲罰加成

## 條件修正

- Park Factor: 106.0 → +0.30 run
- 天氣：未公布（loanDepot park 室內 retractable roof，跳過天氣分析）
- 先發 tier / doubleheader：兩名先發同手別 (RHP vs RHP) 無 platoon 上修；同年齡 30 同 📉 初期退化 trend；無 doubleheader 影響；Junk GS 7 / Littell GS 5（兩人均已過「角色轉換前 3 場降級」門檻，沿用 season 數據）

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 9.4 | 0.0（chain_break -0.3 + core_il_count(WSH) +0.3 = 互抵） | 9.4 |
| AWAY | 4.0 | +0.2（chain_break -0.2 + core_il_count(MIA) +0.3 + tto3 弱信號 +0.1） | 4.2 |
| Total | 13.4 | +0.2 | 13.6 |

> ⚠️ Sanity rail 提醒（不入 + 信號欄，敘事補充）：
> - Junk gap -6.3 / Littell gap +10.0 雙端均未 fire Flag 8，但兩名先發 ERA 都偏離 peripherals → formula base 9.4 / 4.0 含可能高估 / 低估
> - HOME 9.4：MIA 攻擊近 30 天 RS 3.80（季均偏低），對應 Littell xFIP 5.09 而非 ERA 7.24 → **實際 MIA 得分期望可能落 6.5–8.0 區間**（敘事下修，不改 formula 數字）
> - AWAY 4.0：Junk FIP 3.33 / xERA 3.32 / 近 3 場 ERA 4.32 → ERA 2.82 含好運成分 → **WSH 攻擊期望可能落 4.0–5.0 區間**

## 整體判斷

- **方向（基本面）**：HOME (Marlins) 占優
- **總分（基本面）**：formula adjusted 13.6；sanity rail 後實際期望區間 **10.5–13.0**（兩端 ERA 都向 peripherals 收斂）
- **方向信心**：65%（Junk Strong Ace vs Littell Below Average tier 落差結構性明確；但 Junk ERA 含好運 + Littell xFIP 沒爛到 ERA 那麼誇張 → 信心未到 75%）
- **風險**：
  1. Junk peripherals 收斂風險（gap -6.3 接近 Flag 8 邊緣）：FIP 3.33 / 近 3 場 ERA 4.32 顯示朝 Solid Starter 收斂；遇 WSH 上半段 Wood (EV95% 60.7%) / Abrams (vs RHP 1.054) → 單局 3-4 分爆發場景仍存在
  2. Littell 結構不如 ERA 那麼爛（xFIP 5.09 / 近 3 場 4.20）：MIA 攻擊近 30 天 RS 3.80 偏低 → base 9.4 結構性高估；Lopez (.440) / Norby (.500) last7 BABIP 含運氣 → 若回歸 → MIA 得分進一步下修至 6 區間
  3. 兩隊牛棚 core IL ×2 但結構不對稱：WSH 牛棚 ERA 4.57 + Littell 預計早退（場均 3 IP）→ 後段對位明顯不利；MIA 牛棚 ERA 3.32 即便缺 closer 仍有支撐 → MIA 後段壓制力 > WSH
  4. 打線 chain breaks 雙向：MIA #5-6 (0.364 high) 壓制中後段串聯，WSH #3-4 (0.241 medium) 壓制清壘 → 雙方 4 分以上單局都不易打出，比賽走「點狀得分」而非「大局壓制」型態

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
