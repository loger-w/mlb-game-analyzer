## 投手對決

### Tyler Mahle (HOME, RHP, 31 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p75），gap vs ERA-only = +70.7
  - 不完全同意 Elite Ace 定位。tier_v2 由 xFIP 3.34 拉高，但 ERA 6.10 / WHIP 1.57 / K-BB% 14.4 / hard_hit% 22.5 之間明顯打架；近 3 場 7 ER / 14.7 IP（4.29 ERA）也未顯示 elite。gap +70.7 偏結構性訊號（xFIP / 球種 RV 略高於 ERA 反映），但 Flag 8 紀律下不自動下修預測，本場仍以 Solid-Mid 偏弱看待。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 🔴 fired，Δ +0.206 屬 high。AWAY 1-9 棒以右打為主（Marte 切換打、Carroll 左打，其餘 Perdomo/Arenado/Waldschmidt/Moreno/Troy 等右打為核心），與 Mahle 弱側（vs RHB OPS .954）完全重疊，風險明顯放大。
- **對手打線威脅**：AWAY 打線雖 matchup tier 僅 🟡 Average，但搭配 reverse platoon + 主球種 FF 47.2% 單一依賴 + TTO3 衰退 + 近 10 場 RS 6.4，威脅指數實高於表面。預期中段以後 Mahle 投球品質快速下滑。

### Eduardo Rodriguez (AWAY, LHP, 33 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p48, K-BB% p36），gap vs ERA-only = -42.6
  - 同意 Solid Starter。ERA 2.24 與 xERA 4.26 落差 -2.02（Flag 8），K-BB% 僅 8.1 / xFIP 4.15 / hard_hit% 23.1 都不符合 Strong Ace 條件，明顯為 BABIP / LOB% 運氣型壓低。Flag 8 紀律下不自動下修；近 3 場 1 ER / 18 IP 雖亮眼但屬尾部結果，本場以 Solid 看待。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 未 fired。E.Rod vs LHB / vs RHB OPS 各為 .625 / .620 區間相當，手別對位無放大空間。
- **對手打線威脅**：HOME 打線 vs LHP matchup tier 🟢 Weak（season OPS .693），核心打者中僅 Schmitt vs LHP OPS .965 較有威脅；其餘 Adames .411 / Devers .619 / Arraez .755 / Chapman .765 大致被壓制。近 7 BABIP .278 normal，無回歸動能，整體對 E.Rod 威脅有限。

## 打線評級

### HOME — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - matchup tier 與 season tier 同為 🟢 Weak，無落差，維持 Weak 評估。對 E.Rod 不具威脅，需仰賴單點突破（Schmitt vs LHP OPS .965 為唯一明顯優勢點）。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #8-9 fired（落差 0.277, medium）。HOME 為 projected 打序，僅影響尾段串聯效率（影響相對小，因為已 weak）。heat_vs_babip 未 fired，無雜訊。整體下修 -0.1 到 -0.2 run。

### AWAY — season tier 🟠 Strong / heat 🔥 Hot
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - season Strong → matchup Average，輕度下修。原因：核心 Perdomo .668 / Vargas .731 / Moreno .710 vs RHP 並非優勢點，僅 Marte / Carroll / Arenado 仍維持高水準。本場仍偏正面但非 Strong 級爆發力。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #2-3 fired（落差 0.300, high）：Carroll 1.185 → Perdomo .513 last7，破壞 1-3 棒串聯。lucky-hot ⏳ fired：last7 BABIP .354 偏高，熱度含運氣成分，對手投手有時間調整。AWAY 攻擊 chain 受 #2-3 切斷壓制 -0.15，但 reverse platoon + bullpen IL 的正向 signal 更強，淨向上。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.32 / 7 / 3+（🔴🔴 極高） | 4.09 / 6 / 3+（🔴🔴 極高） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 3.32 本質中上，但核心 IL ×3（Birdsong, Buttó 等 60-day）等同崩盤級。Mahle TTO3 已明顯衰退（Δ +0.167），預期 5 局後就需轉入 thin bullpen，對 AWAY 後段供應 +0.4~+0.6 run 失分風險。
- AWAY 牛棚：ERA 4.09 本就不出色，核心 IL ×3（A.J. Puk, Saalfrank 等 60-day）使後段更脆弱。但本場 E.Rod 近 3 場連續 6 IP 內僅失 1 ER，可能撐到 6-7 局壓低牛棚使用量，緩衝部分風險。對 HOME 後段供應 +0.2~+0.3 run。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=+2.04):
  - 偏結構性 + 部分運氣。Mahle xERA 4.06 / xFIP 3.34 都比 ERA 6.10 低 2 整段，hard_hit% 22.5 與 barrel% 9.1 都中性，反映 BABIP 與 LOB% 拖累 ERA。但 vs RHB OPS .954 + WHIP 1.57 顯示結構問題真實存在（reverse platoon、單一球種依賴）。Flag 8 紀律不自動下修，但敘事上預期實際 ERA 接近 xERA（~4.0）而非 6.10。
- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-2.02):
  - 偏運氣型。E.Rod ERA 2.24 / xERA 4.26 反向 2 整段，K-BB% 8.1（一般）/ xFIP 4.15 / barrel% 7.2 都顯示真實水平偏 Solid 而非 Strong Ace。BABIP 偏低 + LOB% 偏高把 ERA 壓得太漂亮。Flag 8 紀律不自動下修，但敘事上不該預期 0.50 ERA 的近 3 場結果延續。

### 額外信號
- 🔴 HOME reverse platoon Δ +0.206（vs RHB OPS 0.954 > vs LHB OPS 0.748）— RHP 對非預期手別反而吃虧
- 🟠 HOME single-pitch dependent：主球種使用率 47.2%（≥45.0%）
- 🔴 HOME TTO3 penalty：OPS Δ +0.167（TTO1 0.838 → TTO3 1.005），第三輪明顯衰退；K% 從 24.2% 掉到 12.0%（Δ -12.2pp）
- 🟠 HOME chain breaks at #8-9：OPS 落差 0.277
- 🟠 AWAY chain breaks at #2-3：OPS 落差 0.300
- 🔴 ⏳ HOME 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
- 🔴 ⏳ AWAY 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
  - 雙方牛棚同時崩盤級，後段比拚誰先換人。HOME Mahle TTO3 衰退快會先暴露牛棚弱點，加上 reverse platoon 對右打林立的 AWAY 打線形成雙重壓力 → 預期 AWAY 中段以後得分機會明顯偏多；反觀 E.Rod 與 Flag 8 落差屬運氣型，後續可能在中段失分，但 HOME 打線 vs LHP 🟢 Weak 限縮上檔。整體 signal 雙重壓力偏 AWAY scoring + total 偏高。

## 條件修正

- Park Factor: 91.0 → -0.45 run
- 天氣：Clear, 60°F, wind 22 mph, Out To RF
  - 影響判讀：60°F 中性偏微利投（聯盟基準 70°F）；**風 22 mph Out To RF 屬強風（>20 mph），風險段必提**。Oracle Park 本身 HR -17%，但 RF 順風 22 mph 在物理上可顯著抵消 RF 方向飛球的壓制（Carroll/Devers/Schmitt 等左打拉打 RF 受益）。淨判讀：HR 機率回到中性偏小幅利攻，PF -0.45 run 修正稍微偏保守，整體 +0.1~+0.2 run 校正（敘事不入信號表）。
- 先發 tier / doubleheader：非 doubleheader，單場 G2 of 系列賽（前一日 G1 D-backs 6-2 勝）。先發 tier 對比 Mahle ERA 6.10 表面 vs E.Rod ERA 2.24 表面，但真實水準（xFIP / xERA）兩者較接近（4.0 區間 vs 4.15），表面差距被 Flag 8 削弱，AWAY 真實先發優勢小於表面。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.4 | +0.2 | 3.6 |
| AWAY | 5.2 | +0.35 | 5.55 |
| Total | 8.6 | +0.55 | 9.15 |

## 整體判斷

- **方向（基本面）**：AWAY（Arizona Diamondbacks）
- **總分（基本面）**：9.15 run（HOME 3.6 / AWAY 5.55）
- **方向信心**：68%（多重 signal 同向支持 AWAY 得分：reverse platoon high + single-pitch + TTO3 high + bullpen IL ×3，且 AWAY 近 10 場 9-1 / RS 6.4 動能強；風險主要在 E.Rod 真實水準偏 Solid 而非近 3 場帳面 0.50 ERA）
- **風險**：
  1. **E.Rod Flag 8 反向**：ERA 2.24 與 xERA 4.26 落差 -2.02，本場若 BABIP / LOB% 回歸均值，HOME 得分可能比 3.6 預期略高；但 HOME 打線 vs LHP 🟢 Weak 為天花板硬上限。
  2. **AWAY lucky-hot ⏳**：last7 BABIP .354 偏高，對手投手有時間調整 mix，熱度含運氣，AWAY 5.55 上修偏樂觀情境。
  3. **強風 22 mph Out To RF**：Oracle Park 雖 HR -17%，但 RF 順風強勁可能放大左打拉打飛球；雙方都有左打核心（Carroll / Devers / Schmitt），總分上行可能超過 9.15。
  4. **HOME 打線 projected 而非 official**：實際先發 9 人若與 PA 排序近似有差異，chain_break #8-9 信號可能變化。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組