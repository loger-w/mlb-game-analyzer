## 投手對決

### Randy Vásquez (HOME, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p69, K-BB% p67），gap vs ERA-only = -4.5
  - |gap| < 15，同意 score-derived tier。Vásquez ERA 2.96 但 xERA 5.44，era_xera_delta=-2.48 走 Flag 8 紀律：xFIP 3.85 與 K-BB% 13% 都顯示真實水準較接近 Strong Ace 邊緣而非 Elite，敘事判讀偏「ERA 含運氣成分」，但不自動下修本場預測。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 未 fired。raw 數據 vs LHB OPS .749 / vs RHB OPS .630，正常 RHP 對 RHB 較強之 platoon 走向，無反向風險。
- **對手打線威脅**：PHI 打線 vs RHP 為 🟢 Weak（season OPS .712 但 vs RHP .549/.518/.585 多數中下段），核心威脅集中於 #2 Schwarber (.975) 與 #3 Harper (.995)；Vásquez FF/FC/SI 配置 RV/100 皆正向（FF +1.3 / FC +0.4 / SI +0.7），對 Weak vs RHP 打線壓制力符合 Strong Ace 等級。

### Aaron Nola (AWAY, RHP, 32 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p77），gap vs ERA-only = +70.4
  - |gap| ≥ 20 → tier_mismatch high severity。raw ERA 6.04 vs xERA 4.77（Δ +1.27 不利 ERA），xFIP 3.36 + K-BB% 14.8% 確認真實水平接近 Strong/Elite 邊緣，ERA 含運氣偏差（hard_hit% 26.1 偏低）。但 32 歲 📉 初期退化、velo 86.1 偏弱、近 3 場 ER/IP=7/17.3（ERA ~3.64）顯示反彈中但仍有結構性下行壓力。敘事判讀為「真實水平優於 ERA 但低於 tier_v2 純 Elite」，不自動下修預測（Flag 8 紀律）。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 未 fired。raw vs LHB OPS .884 / vs RHB OPS .825，正常輕度 reverse 但未達 0.080 門檻；SD 打線 9 人皆右打為主，無放大風險。
- **對手打線威脅**：SD 打線 vs RHP 為 🟡 Average，威脅分布於 #3 Sheets (.902) / #5 Andujar (.814) / #7 France (.992)；Nola 主球種 KC RV/100 +2.0 為武器球但 FF -5.3 / SI -1.7 為負值，被打球質風險高於 tier 顯示。考量 32 歲退化 + ERA 6.04 結構性壓力，本場預計被打到 4-5 分區間，非 Elite Ace 標準壓制。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak
  - season Average → matchup Weak 為下修一檔，#1 Tatis Jr. .598 / #4 Machado .527 / #9 Fermin .344 vs RHP 形成連續軟肋。本場面對 Nola 雖有結構性退化但 Elite Ace tier，HOME 攻擊上限受壓制，敘事方向：下修。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break fired at #3-4：Sheets .902 → Machado .527（Δ 0.375）斷裂嚴重，#3 Sheets 上壘後 #4 Machado 清壘乏力 → 壓制中段 RBI 串聯。heat_vs_babip：last7 BABIP 0.231 偏低（Flag 3），可能反彈但敘事不入 ±run。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - season Average → matchup Average，一致無調整。但前 5 棒 vs RHP 結構為「兩極化」：#2 Schwarber .975 + #3 Harper .995 為 Strong 雙核，#1 Turner .614 / #4 García .549 / #5 Bohm .518 為 Weak 三點，整體 chain 不順。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break fired at #3-4：Harper .995 → García .549（Δ 0.446）為超強斷裂，Harper 上壘後接續清壘乏力，壓制 RBI 上限。heat_vs_babip：last7 BABIP 0.239 偏低（Flag 3），#3 Harper last7 OPS .467（season .878）異常滑落但 last7 BABIP .211 顯示含厄運成分，有回歸機會但不自動上修。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.13 / 6 / 0 | 3.93 / 3 / 0 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 3.13 為聯盟前段，核心 IL 0 名（無累計效應），整體深度健康；對 AWAY 末段威脅高，配合 Vásquez TTO3 penalty fired（OPS Δ -0.157、K% drop 5.7pp）教練 likely 第 6 局前換投，後段交給高品質牛棚壓制 PHI 弱勢 chain，對 AWAY 末段得分形成有效防火牆。
- AWAY 牛棚：ERA 3.93 為聯盟中段，核心 IL 0 名（無累計效應），但品質明顯不及 HOME；AWAY TTO3 penalty 雖 OPS Δ 微幅但 K% drop 3.6pp 顯示 Nola 第三輪效率下滑，PHI 牛棚需提早銜接但中等品質遇 Petco（PF 95）部分緩衝，對 SD Average 打線壓制力屬一般。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-2.48):
  - 偏運氣面（ERA 低於 xERA 2.48 run）：Vásquez hard_hit% 30.7 / barrel% 12.5 雙偏高顯示被擊球品質不佳，未來 ERA 有上修壓力。但本場面對 PHI 🟢 Weak vs RHP 打線是低風險場合，被擊球品質惡化效應被打線軟弱稀釋。不自動下修本場預測。
- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.231):
  - 可能反彈：season tier Average 但近 7 天 BABIP .231 偏低顯示含厄運成分；面對 Nola 退化中（ERA 6.04 / 32 歲），有反彈窗口但 matchup tier 🟢 Weak vs RHP 結構性限制反彈幅度。不自動 ±run value。
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.239):
  - 可能反彈：BABIP .239 偏低且 Harper last7 BABIP .211 / OPS .467 為核心球員的厄運期，理論上有回歸空間；但 Vásquez 為 Strong Ace 且 chain_break #3-4 結構性問題抑制串聯，反彈幅度有限。不自動 ±run value。

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ +-0.157（TTO1 0.743 → TTO3 0.586），第三輪明顯衰退；K% 從 25.3% 掉到 19.6%（Δ -5.7pp）
- 🟠 AWAY TTO3 penalty：OPS Δ +0.005（TTO1 0.886 → TTO3 0.891），第三輪明顯衰退；K% 從 25.3% 掉到 21.7%（Δ -3.6pp）
- 🟠 HOME chain breaks at #3-4：OPS 落差 0.243
- 🟠 AWAY chain breaks at #3-4：OPS 落差 0.270
  - 兩側 chain_break 同向 fire，且雙打線 Flag 3 BABIP 偏低，形成「Strong 核心 + 串聯斷裂 + 厄運期」三層壓力，本場總得分上限受結構性壓制；TTO3 penalty 兩側 fire 但 HOME 牛棚品質明顯優於 AWAY，後段失分風險偏 AWAY（Table B core_il_count 未 fire 故不單獨計 ±run，TTO3 penalty 兩側對沖部分效應）。

## 條件修正

- Park Factor: 95.0 → -0.25 run
- 天氣：Partly Cloudy, 64°F, wind 9 mph, L To R
  - 影響判讀：64°F 落在 60-85°F 中性區間，無明顯偏移；風 9 mph 為輕度且 L To R 橫風（matchup-factors §風表「影響有限」），對 HR 與飛球軌道無顯著影響。整體天氣為中性，不入 ±run。
- 先發 tier / doubleheader：本場非 doubleheader（系列 G2 為單場）。先發 tier 差距明顯（Vásquez 🟠 Strong Ace vs Nola 🔴 Elite Ace 但 era 6.04 結構性壓力），AI 評估實質落差不如 tier 標籤顯示之大，雙方真實水平接近 Strong Ace 邊緣。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.6 | -0.1 | 4.5 |
| AWAY | 3.6 | -0.2 | 3.4 |
| Total | 8.2 | -0.3 | 7.9 |

> 信號明細：
> - HOME chain_break #3-4 (medium, Δ 0.243) → -0.2（壓制 HOME 攻擊）；Nola TTO3 K% drop 3.6pp（OPS Δ 僅 +0.005 結構不算典型 TTO3 penalty）→ +0.1（受惠 HOME 末段攻擊）。淨 -0.1。
> - AWAY chain_break #3-4 (medium, Δ 0.270) → -0.2（壓制 AWAY 攻擊）；Vásquez TTO3 OPS Δ -0.157 實為「越投越好」不入 +run（K% drop 5.7pp 但 OPS 改善 → 對手未獲利）→ 0。淨 -0.2。
> - Table A (heat_vs_babip 雙側 / tier_mismatch AWAY / strong_park) 全部寫 0（敘事段已處理）。
> - 單側 cap ±0.8 / 場 紀律：兩側均在 -0.2 內，遠未碰頂。

## 整體判斷

- **方向（基本面）**：HOME（SD Padres）勝。adjusted HOME 4.5 vs AWAY 3.4，差距 1.1 run > 0.5 閾值。
- **總分（基本面）**：7.9 run（base 8.2 - 信號修正 0.3）
- **方向信心**：62%。HOME 結構性優勢明確（PHI 打線 🟢 Weak vs RHP + 中性天氣 + Petco PF 95 壓制 + HOME 牛棚品質明顯優於 AWAY），但 Nola 雖 ERA 6.04 仍為 Elite tier 真實水平且近 3 場 ER/IP=7/17.3 反彈中，SD 打線 vs RHP 僅 🟡 Average + chain_break，HOME 上限受制。落在 50-75% 區間中段。
- **風險**：
  1. **Flag 8 雙刃**：Vásquez ERA 2.96 含運氣（xERA 5.44），本場若 hard_hit% 集中化可能單局崩盤；Nola ERA 6.04 含厄運，若真實水平回歸可能壓制 SD 至 2-3 分壓低 HOME 勝出空間。
  2. **Flag 3 雙側 BABIP 偏低**：兩打線都在厄運期，若 PHI 端 Schwarber/Harper 任一爆發回歸 + chain_break 修補，AWAY 可能跳分至 5+ 改變方向。
  3. **打線來源不對稱**：AWAY projected（非實際打序），實際公布若 Schwarber/Harper 順序調整可能緩解 chain_break，需賽前確認。
  4. **Petco 早場低溫 64°F + 風 9 mph 橫風**：對 HR 影響有限，但若臨場升 wind speed 至 15+ mph 需重新評估。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組