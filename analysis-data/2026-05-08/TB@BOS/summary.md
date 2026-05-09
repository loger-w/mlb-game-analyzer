## 投手對決

### Connelly Early (HOME, LHP, 24 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p51, K-BB% p44），gap vs ERA-only = -3.9
  - 同意 tier_v2，但偏「樂觀邊界」。|gap| 3.9 未觸 mismatch (≥ 15)，但 Flag 8 era_xera_delta = -1.56 暗示 ERA 3.79 比 xERA 5.35 低 1.56 — 結合 barrel% 14.0（聯盟 ~7.5%）+ hard_hit% 27.0（偏低）→ 球被打中時打很扎實，但接住的 LOB%/BABIP 偏向 Early。**結構偏向 🟢 Back-end**，目前 ERA 受運氣加持。本場若回歸均值，BOS 投手戰假設可能失靈。
- **Reverse platoon 信號**：dossier 訊號摘要無 fire（vs LHB .176/.237/.353 / vs RHB .250/.363/.406 → 正向 platoon）
  - 不適用。
- **對手打線威脅**：對 LHP 名單 Caminero (.815) / Díaz (.897) 構成主要威脅；Aranda last7 OPS 1.019 但 BABIP .550 — 高運氣熱手，回歸風險。Mullins (.426 vs LHP)、Simpson (.582)、Aranda 季均 .664 — Rays 對左投核心 5 人實際只 2 人有威脅。Early 主球種 SI（17.4%）RV/100 = +2.3（被打很慘）→ 配球若多丟 SI 對 Caminero / Díaz 風險高。整體：對位中性偏 Early。

### Jesse Scholtens (AWAY, RHP, 32 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = —（資料不足，本季 GS 1），ERA-only 顯示 🟠 Strong Ace 但 ERA 3.18 來自單場樣本，**信賴度極低**
  - 不同意「Strong Ace」標籤。Scholtens 32 歲 + 📉 初期退化 + 速球均速 89.3 mph（聯盟 RHP 平均 ~93）+ 主球種滑球 35.5% → 屬「軟球依賴變化球」型，career TTO splits 已暴露第三輪崩盤（OPS 1.020 / K% drop -13pp）。**真實 tier 應落在 🟢 Back-end ~ ⚪ Below Average 之間**。本季 1 GS 的 ERA 3.18 視為雜訊，不採信為基準。
- **Reverse platoon 信號**：dossier 訊號摘要無 fire（vs LHB .209/.277/.419 / vs RHB .256/.319/.395 — 兩側接近，無顯著 reverse）
  - 不適用，但兩側 BF 都僅 47 → career sample，不放大解讀。
- **對手打線威脅**：**TTO3 penalty 是本場最重信號（Δ +0.388, K% -13pp, severity high）**。對 BOS vs RHP 名單 — Contreras (.758) / Abreu (.806) 構成 1-2 棒威脅，但 Story (.492) / Duran (.553) / Rafaela (.563) 中段熄火 → 短期 Scholtens 可吃下 1-2 輪，但第三輪起對手調整 + Sox 主場 timing 看慣 → 後段失分機率顯著提升，**配合 TB 牛棚崩盤級**（見下）→ Scholtens 5 IP 後是雙刃。

## 打線評級

### HOME — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak
  - season vs matchup 一致 🟢 Weak — top5 vs RHP OPS 只 Contreras/Abreu (>.750) 兩人有威脅，Story/Duran/Rafaela 全在 .492 ~ .563（sub-.600 = 替補水平）。**同意 🟢 Weak，方向偏下修**：對 Scholtens（軟球 RHP）若無第三輪 timing 加成，3-5 棒缺乏清壘能力。
- **chain_break / heat_vs_babip 信號**：dossier fire `🔴 HOME chain breaks at #2-3 (落差 0.313, high)`
  - 影響核心：chain top3 OBP 區段。Contreras/Abreu 上壘但 #3 Story (.492) 帶不動 → 即使前兩棒製造機會也常以無效殘壘收尾。**這是 Sox 進攻的結構性瓶頸**，本場若 Scholtens 撐到第三輪以前，3-5 棒清壘能力薄弱會壓制總分。

### AWAY — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - season vs matchup 一致 🟢 Weak，但**近 10 場 RS 3.50 / 近 30 場 RS 4.43** + **+7 連勝**顯示實戰產出比評級樂觀。top5 vs LHP 只 Caminero (.815) / Díaz (.897) 強，Aranda (.664) / Simpson (.582) / Mullins (.426) 偏弱。**同意 🟢 Weak 但方向略上修**（last7 動能 + Caminero/Díaz 正面對位 LHP Early）。
- **chain_break / heat_vs_babip 信號**：dossier fire `🔴 AWAY chain breaks at #4-5 (落差 0.452, high)`
  - 影響核心：chain mid SLG 區段。Caminero (#1)/Aranda(#2)/Díaz(#4) 是火力來源，但 #5 Mullins .426 vs LHP（左對左明顯吃癟）→ 對 Early 的後段清壘崩盤。**Aranda last7 BABIP .550 是異常運氣熱手（Flag 3 邊緣）**，AI 判讀本場可能回歸但仍偏 hot；不入 ±run，列風險敘事。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.47 / 7 / **2** (Coulombe IL15d, Slaten IL15d) | 4.11 / 8 / **3+** (Uceta IL60d, Rodríguez IL60d, +) |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（BOS）：core IL ×2 → 🔴 高（吃緊）。整體 ERA 3.47 比 TB 佳，但失去左投 Coulombe + setup Slaten → 中後段 vs Rays LHB（Aranda / Mullins / Simpson）少一張左投對位牌。Scholtens 若被早換（TTO3 風險），BOS 可吃下他但 Rays 牛棚反過來補位也會吃緊 → 雙方僵局。對 Rays 末段威脅：中等（Sox 牛棚 last 30 RA 4.23 → 3.60 趨勢向下）。
- AWAY 牛棚（TB）：core IL ×3+ → 🔴🔴 極高（崩盤級）。但**警示資料對立**：Rays 近 10 場 RA 1.50（防守極佳）+ 近 30 場 RA 3.33 → 實戰結果遠優於 ERA 4.11 + IL 名單暗示。可能解釋：(a) 弱對手干擾 (Jays / Giants 都不算強攻擊)；(b) bullpen 替補新秀爆發。**保守判讀**：對 Sox 主場攻擊群 vs RHP 仍偏吃緊，Sox 6-9 局有縫可鑽，但不應全盤押 TB 牛棚崩盤。對 Sox 末段威脅：高（IL 名單 + Fenway 主場攻勢 timing）。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-1.56):
  - **混合判讀偏向「運氣加持」+ 部分結構問題**。barrel% 14.0（聯盟頂段）+ FF/SI RV/100 偏正（被打）顯示球質有問題，但 hard_hit% 27.0（聯盟低段）+ 高 LOB 紀錄拉低 ERA。樣本仍小（GS 7）→ xERA 5.35 不可全信但方向對。**不自動下修本場預測**（Flag 8 紀律），但風險意識：若 Rays 核心 Caminero / Díaz 把 SI（RV +2.3）打中 → 預期 BOS 失分可能比 ERA 3.79 暗示更多。

### 額外信號
- 🔴 AWAY TTO3 penalty：OPS Δ +0.388（TTO1 0.632 → TTO3 1.020），第三輪明顯衰退；K% 從 22.8% 掉到 9.8%（Δ -13.0pp）（career fallback）
- 🔴 HOME chain breaks at #2-3：OPS 落差 0.313
- 🔴 AWAY chain breaks at #4-5：OPS 落差 0.452
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
- 🔴 ⏳ AWAY 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
  - 受影響但**幅度被近 10 場 RA 1.50 緩衝**。與 Flag 8 Early 同向：雙方後段都吃緊 → 中場後得分波動率上升。⏳ 短半衰期（每天異動）→ 帶懷疑解讀，不取上界。

## 條件修正

- Park Factor: 104.0 → +0.20 run（已含於 base formula，不重複加）
- Fenway HR -15%：抑制長打型得分 → Caminero / Abreu / Contreras 拉打傾向打者本場 HR 機率小幅下降，但 Green Monster 反向利二壘打 → 整體得分手段轉為串聯 single/double，**chain_break 信號的負面影響在此球場放大**
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：無 doubleheader；雙方先發 tier 落差不大（Early 真實 ~🟢 Back-end vs Scholtens ~🟢 Back-end / ⚪ Below），Scholtens 第三輪後拐點明顯，Early 較全程平穩

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.8 | +0.3（Scholtens TTO3 +0.5 ⊕ TB core IL +0.3 → 同向取 max+0.1 = +0.6；HOME chain_break -0.3 → 淨 +0.3） | **4.1** |
| AWAY | 4.9 | 0（BOS core IL +0.3 與 AWAY chain_break -0.3 互抵） | **4.9** |
| Total | 8.7 | +0.3 | **9.0** |

## 整體判斷

- **方向（基本面）**：**AWAY (Tampa Bay Rays)** — 修正後 TB 4.9 vs BOS 4.1（差 0.8 run）
- **總分（基本面）**：**~9.0**（base 8.7 + 信號淨 +0.3）— 中性偏多，未脫離聯盟基準
- **方向信心**：**~58%**（TB 領先）。理由：base formula 已偏 TB（4.9 vs 3.8），加上 +7 連勝動能與 Caminero/Díaz vs LHP 正面對位；但兩位 starter 樣本都小 + Sox 牛棚較不吃緊 → 不到 60% 上界
- **風險**：
  1. **雙方 starter 樣本量極小**（Early 7 GS / Scholtens 1 GS）→ ERA 與 advanced metrics 都不穩定，單場分散度高
  2. **Aranda last7 BABIP .550 極端熱（Flag 3 邊緣）**+ last7 OPS 1.019 → 若本場回歸均值，TB 中段攻擊弱化（敘事，不入 ±run）
  3. **Early Flag 8 ERA-xERA 落差 1.56**：若 batted-ball 回歸 xERA 5.35 方向，BOS 投手端比 ERA 3.79 暗示更脆弱 → 本場 BOS 失分上修風險
  4. **TB 牛棚數據對立**（core IL ×3+ ⏳ 但近 10 場 RA 1.50）→ 對 Sox 主場攻擊群實際對位是本場最大不確定性；若 IL 名單壓力浮現 + Sox chain_break 同時發作，總分上下偏移 1.0+ run 都合理

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組