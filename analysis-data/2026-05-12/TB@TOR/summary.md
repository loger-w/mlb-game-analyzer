## 投手對決

### Patrick Corbin (HOME, LHP, 36 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p69, K-BB% p47），gap vs ERA-only = -5.5
  - 不同意 Solid Starter。**era_xera_delta = -2.63（Flag 8 全卡最大）**：ERA 3.60 高估真實水平 — 結構性 ERA/xERA gap 巨型，xERA **6.23** vs FIP 3.90 / K-BB 9.6 / velo **85.1 avg** / vs RHB .273/.347/.409 — 全部證據指向 ⚪ Below Average / 🟢 Back-end 下緣。ERA 3.60 純粹是低 BABIP + 殘壘運氣 + 6 GS 小樣本疊出。不下修預測（Flag 8 紀律），但敘事按 Back-end / Below Avg。
- **Reverse platoon 信號**：未 fired（vs LHB .250/.318/.300 / vs RHB .273/.347/.409 — vs RHB SLG 略高但未過 reverse 標準）。
  - n/a
- **對手打線威脅**：高。TB top 5 vs LHP（Díaz .872 / Aranda last7 1.091 / Caminero .805 / Vilade .793 / DeLuca .780）+ Corbin xERA 6.23 + vs RHB SLG .409 → TB 預期 5+ 分。

### Shane McClanahan (AWAY, LHP, 29 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p80, K-BB% p75），gap vs ERA-only = -6.9
  - 同意 Strong Ace（可能 Elite 邊緣）。ERA 2.60 / xERA 3.58 / FIP 2.73 / xFIP 3.66 / K-BB% 14.2 / WHIP 1.07 / velo 89.5 / 97.8 max — 結構頂尖。era_xera_delta -0.98（接近 Flag 8 邊界但未 fire）顯示 xERA 略高於 ERA — McClanahan 表現比 ERA 暗示稍多運氣，但 K-BB% 14.2 是真實。
- **Reverse platoon 信號**：未 fired（vs LHB .148/.233/.185 / vs RHB .192/.273/.273 — 都極壓）。
  - n/a，整支壓制。
- **對手打線威脅**：低。TOR top 5 vs LHP（Springer .797 / Guerrero 1.023 / Okamoto .819 / Valenzuela .993）數字漂亮但 McClanahan vs LHB SLG .185 + vs RHB SLG .273 — 整支對 McClanahan 結構性弱勢；單槍 HR 機率（Guerrero EV95 43.8 / Okamoto 53.5）。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟠 Strong
  - 不完全同意 Strong。Top 5 vs LHP OPS 數字好但對 McClanahan Elite 級壓制不適用一般 LHP 數據 — 維持 Average，本場 distribution 偏 🟢 Weak 結果。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - HOME chain_break #4-5 gap 0.213（vs LHP）fired — Okamoto .819 → Sosa .606 vs LHP，middle chain 中斷，−0.1 run。

### AWAY — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟢 Weak
  - 上修同意 Weak（season 一致）。但對 Corbin Flag 8 結構性弱 + vs RHB SLG .409 → TB 多右打 Top 4（Díaz/Aranda(L)/Caminero/Vilade/DeLuca）有實質 edge — 本場上修至 🟡 Average vs Corbin 特定 matchup。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - AWAY chain_break #6-7 gap 0.240 fired — Williamson .662 → Mullins .422 vs LHP，chain 尾微斷，−0.1 run。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.2 / 7 / 1 | 3.97 / 7 / 3 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：TOR 4.20 ERA + 1 core IL（García）— 中等深度，後段稍變薄。Corbin 5-6 局後高機率被換投，TOR 中繼需擋 3-4 局。
- AWAY 牛棚：**TB 3.97 ERA + 3 core IL（Uceta + M. Rodríguez + 1）= 🔴🔴 崩盤級**。McClanahan 若 TTO3 K% 大跌（-8.3pp）被早換投，TB 中繼防守全面失能 — TOR 後段加分機率極高。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-2.63):
  - **全卡最大運氣假象**。ERA 3.60 與 xERA 6.23 差距 2.63，K-BB% 9.6 + FIP 3.90 + velo 85.1 全部證據指向 ERA 將回升至 4.5+。單場仍可能因 BABIP 運氣延續低 ERA，但 distribution 厚尾向 TB 高分。不自動下修預測。

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ +0.068（TTO1 0.819 → TTO3 0.887），第三輪明顯衰退；K% 從 20.7% 掉到 14.3%（Δ -6.4pp）（career fallback）
- 🟠 AWAY TTO3 penalty：OPS Δ +-0.004（TTO1 0.585 → TTO3 0.581），第三輪明顯衰退；K% 從 31.6% 掉到 23.3%（Δ -8.3pp）（career fallback）
- 🟠 HOME chain breaks at #4-5：OPS 落差 0.262
- 🟠 AWAY chain breaks at #6-7：OPS 落差 0.253
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
- 🔴 ⏳ AWAY 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
  - **本場最大訊號 = Corbin Flag 8 -2.63 結構性弱 vs TB 3 core IL 崩盤雙重疊加**。前段 TB edge（Corbin 易被打）+ 後段 TOR edge（TB 牛棚崩盤）— 矛盾訊號使 distribution 寬。Aranda last7 1.091 + Caminero / Díaz 仍是 TB 主要威脅；TOR Guerrero 33 BF 1.023 vs LHP 可能單槍翻盤。

## 條件修正

- Park Factor: 99.0 → -0.05 run
- 天氣：室內（Roof Closed，不適用）
- 先發 tier / doubleheader：McClanahan Strong Ace > Corbin Back-end / Below Avg（兩級以上 mismatch by 真實水平）；TB 牛棚 3 core IL 崩盤是後段最大 downside。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 2.9 | +0.5（AWAY core IL ×3 崩盤 +0.5 + AWAY TTO3 +0.1 互動 max+0.1 −0.1 chain HOME = +0.5） | 3.4 |
| AWAY | 4.0 | +0.0（HOME core IL +0.1 + HOME TTO3 +0.1 互動 max+0.1 −0.1 chain AWAY = 0.0） | 4.0 |
| Total | 6.9 | +0.5 | 7.4 |

## 整體判斷

- **方向（基本面）**：AWAY (TB)
- **總分（基本面）**：7.4（Flag 8 厚尾向上，可能 8.0+ 因 Corbin 結構性弱 vs TB top 5）
- **方向信心**：62% — McClanahan 等級明顯優勢 + Corbin Flag 8 結構性弱 = SP edge 偏 TB；信心受 TB 3 core IL 崩盤級牛棚反向 hedge。
- **風險**：
  1. 🔴 **TB 牛棚 3 core IL 崩盤級** — McClanahan 若 TTO3 K% 大跌被早換，TB 中繼吃緊；TOR 後段可能炸開
  2. ⚠️ Corbin Flag 8 -2.63 — 單場仍可能延續低 ERA（運氣分佈不平均），TOR 主場身體舒適
  3. McClanahan 自身 -0.98 era_xera 也是 over-performer（程度小），可能 6 局後失準
  4. Rogers Centre HR +4%，Caminero / Aranda 右打拉打火力 1 球翻盤

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
