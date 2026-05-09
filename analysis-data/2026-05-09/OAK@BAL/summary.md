## 投手對決

### Shane Baz (HOME, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p59, K-BB% p47），gap vs ERA-only = +28.8
  - 同意 Solid Starter（不採 ERA-only 給的 Back-end）。ERA 4.99 vs FIP 3.96 / xERA 4.70 / xFIP 3.99 落差近 1.0 → 部分壞運（BABIP / LOB% 偏負）+ 結構性兩半混合：whiff% 8.7 / K-BB% 9.7 偏低、velo 巔峰但 swing-and-miss 不夠。Flag 8 紀律不自動下修預測，但承認 ERA 4.99 高估其結構困難度。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 沒 fired，但 vs LHB .341/.417/.518（103 BF）vs RHB .235/.274/.456（73 BF）OPS 差近 0.20 — 是真實「右投對左打反向弱」profile（FF 32.9% + KC 32.9% 平衡但不夠殺左打）。OAK 核心預估含 Kurtz / Soderstrom / McNeil 3 名左打 → 風險實質存在；未自動加 ±run（門檻不到 reverse_platoon 觸發定義）。
- **對手打線威脅**：OAK season 🟡 Average + matchup 🟡 vs RHP，但 Kurtz vs RHP .948、Langeliers vs RHP .983 是 power threat；近 7 天 Langeliers 1.315（BABIP .440 偏熱含運氣）/ Wilson .914（BABIP .348）/ Soderstrom .804。Baz 第 3 球種 FC RV +3.7 是被打標的 → 中段對 Kurtz / Langeliers 有失分風險。預期 5-6 IP / 3-4 ER 區間，第三輪後（OPS .899 跳級至聯盟平均水準）為失分集中點。

### Aaron Civale (AWAY, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p42, K-BB% p54），gap vs ERA-only = -23.8
  - 同意 Solid Starter（不採 ERA-only 給的 Strong Ace）。ERA 2.95 vs xERA 4.21 / FIP 3.86 / xFIP 4.25 一致顯示運氣壓 ERA：hard_hit% 31.6 偏高、barrel% 6.7 中等、whiff% 7.7 偏低、velo 85.9 avg 為 30 歲退化中。本場若回歸 xERA 4.21 對應 4-5 ER 區間；近 3 場 1.72 ERA 顯示短期續熱中，本場是否回歸隨機。Flag 8 紀律不自動下修預測，但記錄此偏差。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 沒 fired。vs LHB .272/.343/.402（102 BF）vs RHB .269/.273/.423（56 BF）OPS 差 < 0.05 → 平均 RHP profile，不放大此風險。
- **對手打線威脅**：BAL matchup 🟠 Strong vs RHP > season 🟡 Average — Alonso vs RHP .836（last7 1.117 BABIP .353）/ Basallo .805（last7 1.028 BABIP .471 純運氣）/ Ward .788 是 vs RHP 利多，集中在 1-5 棒。但 Henderson .654 / Jackson .687 拉低平均 + 近 7 天多人冷（Henderson .285 / Ward .424 / Jackson .297）+ last7 BABIP .271 略冷（未達 Flag 3）。Civale FC + CU（top 2 球種 RV 各 +2.1）對 Alonso 級 power 危險；但 BAL chain_break #6-7 後段斷層會抒壓。預期 5-6 IP / 3-4 ER。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong
  - 同意上修一檔。Alonso / Basallo / Ward vs RHP 都偏強（OPS .788-.836），但 Henderson / Jackson 拖累 → 攻擊集中在 1-5 棒、後段斷層。對位 Civale 短期續熱但 xERA 4.21 真實水準偏弱 → 中段威脅明確。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #6-7（🟠 OPS Δ 0.152）fired：壓 BAL 中後段串聯，得分上限受限；對位 Civale TTO3 penalty 🟠（OPS Δ +0.115、K% 從 25 掉到 6.2）→ 若 BAL 撐到第三輪攻擊，TTO3 衰退補上中段斷層、淨抵銷部分壓制效果。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - 同意（與 season 一致）。短期熱（Langeliers 1.315 / Wilson .914 / Soderstrom .804）但夾雜 BABIP 偏高（Langeliers .440 / Wilson .348 / McNeil .400）→ 含運氣成分；last7 整體 BABIP .346 仍在常態。Kurtz vs RHP .948 是真實 power 威脅，對 Baz vs LHB 弱項放大效應顯著。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #7-8（🔴 OPS Δ 0.400）fired：OAK 後段斷層嚴重，威脅集中在 1-6 棒。但 Baz TTO3 penalty 🔴（OPS Δ +0.237、K% 從 25 掉到 14.6）→ 1-6 棒在第三輪有顯著攻擊機會，對沖一部分 7-8 棒劣勢。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.63 / 8 / 2 (Bautista IL60d, Helsley IL15d) | 4.76 / 1 / 0 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：core IL ×2（Bautista IL60d 已是長傷、Helsley IL15d 5/9 不會回）→ 🔴 高影響。closer + setup 雙缺，後段 7-8-9 局 leverage 直接掉一檔。Baz TTO3 penalty 🔴 會逼提前換投到 5-6 局 → 牛棚薄將直接放大失分風險。中段 OAK 1-6 棒（Kurtz / Langeliers / Soderstrom）爆發機會顯著放大。
- AWAY 牛棚：core IL 0，整體 ERA 4.76 中性偏弱但角色完整。Civale TTO3 penalty 🟠（K% 從 25 掉到 6.2）會逼第三輪換人，但 setup / closer 完整可吸收 → 不放大失分。對 BAL 後段 Henderson / Jackson cold 反而是中性偏正面，能直接壓末段串聯。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🔴 HOME TTO3 penalty：OPS Δ +0.237（TTO1 0.662 → TTO3 0.899），第三輪明顯衰退；K% 從 25.0% 掉到 14.6%（Δ -10.4pp）
- 🟠 AWAY TTO3 penalty：OPS Δ +0.115（TTO1 0.746 → TTO3 0.861），第三輪明顯衰退；K% 從 25.0% 掉到 6.2%（Δ -18.8pp）
- 🟠 HOME chain breaks at #6-7：OPS 落差 0.152
- 🔴 AWAY chain breaks at #7-8：OPS 落差 0.400
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - 本場高度相關。Baz TTO3 penalty 🔴 + BAL core IL ×2 雙重壓力同隊 fire：教練被迫 5-6 局換投但 closer / setup 都缺、選項薄 → 後段失分風險顯著放大、總分判讀偏多。⏳ 屬短期信號但 IL60d Bautista 不會回、IL15d Helsley 5/9 也不會回 → 實質 medium-term 利空，正常引用。

## 條件修正

- Park Factor: 96.0 → -0.20 run（Camden Yards Runs 96 / HR +7%，分裂型球場：抑制總得分但加成 HR；對 Alonso / Kurtz / Langeliers 等 power 打者為隱形利多，未入 formula）
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：兩位都 🟡 Solid Starter，tier 對位中性；非 doubleheader

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.1 | -0.05（PF -0.10 + signals +0.05） | 4.05 |
| AWAY | 4.3 | +0.20（PF -0.10 + signals +0.30） | 4.50 |
| Total | 8.4 | +0.15（PF -0.20 + signals +0.35） | 8.55 |

> AWAY signals net 計算：Baz TTO3 high (+0.40) + BAL core_il_count high (+0.50) 同隊同向 fire → 累積規則取 single max +0.50 + interaction +0.10 = +0.60；扣 OAK chain_break high (-0.30) = +0.30。
> HOME signals net：Civale TTO3 medium (+0.20) − BAL chain_break medium (-0.15) ≈ +0.05；BAL last7 BABIP .271 略冷（敘事，不入 ±run）。
> tier_mismatch（Baz +28.8 / Civale -23.8）依 Table A 不入 ±run，已在投手對決段敘事處理。

## 整體判斷

- **方向（基本面）**：略偏 AWAY（OAK）— Baz 三項利空疊加：vs LHB 弱項（split .341，OAK 預估 3 名核心左打）+ TTO3 penalty 🔴 + BAL 牛棚 core IL ×2；對位 Civale 雖續熱但 xERA 4.21 顯示真實水準偏弱、TTO3 penalty 🟠、BAL 後段冷 + chain_break #6-7
- **總分（基本面）**：8.55 run（formula 8.4 + signals net +0.35 - PF -0.20，落在 8.3-8.8 區間，輕微偏 over 但接近 base）
- **方向信心**：55%（OAK ML 略傾向但不到 high conviction — Civale 短期延續好運 + OAK 後段 chain_break #7-8 🔴 嚴重 + Camden HR PF +7% 對 BAL power 主力 Alonso 為隱形利多 → 抵銷部分 OAK 利好）
- **風險**：
  1. Baz vs LHB .341/.417/.518（103 BF 樣本足）為真實反向 split — OAK 排出 Kurtz / Soderstrom / McNeil 連續左打串可能放大；但若 BAL 教練調整投捕配球壓抑左打 leverage（FF 用更少、KC 用更多）可緩解
  2. Civale ERA 2.95 vs xERA 4.21 / FIP 3.86 — 任何單場都可能是回歸點，但近 3 場 1.72 ERA 顯示短期續熱中；本場是否回歸隨機，可能仍延續到下一場才反彈
  3. 兩隊均 projected lineup（13:00-14:00 ET 才會公布實際打序）— Henderson / Alonso / Kurtz / Langeliers 任一缺陣會大幅改變方向。建議在打線公布後重看 dossier
  4. Camden HR PF +7%（dossier 已標）— 對 Alonso（vs RHP .836）/ Kurtz（vs RHP .948）/ Langeliers（vs RHP .983）為隱形 over 上行 catalyst，未入 formula

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
