## 投手對決

### Landen Roupp (HOME, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p84），gap vs ERA-only = +16.9
  - 部分同意但保留：xFIP 3.06 / FIP 2.47 / K-BB% 16.9 / 被打 .183/.276/.245、whiff 10% / hard_hit 16.7% / barrel 0% — Statcast 質量背書沒有運氣紅利。低均速側翼（avg 86.7、SI 38.6%-heavy 滾地球路線）讓 ERA-only 模型用速度向下打分，xFIP-blend 修正後跳級。**結構性偏 🟠 Strong Ace 接近 Elite 的下沿**，非典型 Ace velo profile，gap 主要來自模型對「軟控制 + 高指令」profile 的低估，不是運氣。
- **Reverse platoon 信號**：未觸發（vs LHB .278 / vs RHB .197 屬正常順位）
- **對手打線威脅**：中等。Pirates 屬 🟡 Average vs RHP（xwOBA .337 / OPS .736），Top 5 全員 OPS .720+（Reynolds .796、Cruz .775、O'Hearn .818、Lowe .926、Gonzales .724），**Lowe vs LHP 1.073** 但對 RHP 仍 Average。Roupp SI/CU 滾地球路線對 Pirates Top 4 高 EV95% 群（Cruz 59.8%、O'Hearn 45.6%、Reynolds 44.7%、Lowe 43.8%）有風險：球進入 SI 區 + 90+ EV → Oracle 也壓不下二三壘打。但 vs RHB 67 BF .180/.254/.197 樣本顯示對右打有壓制力（5/9 棒右打為主）。

### Braxton Ashcraft (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +17.2
  - 大致同意：xFIP 3.01 / FIP 2.88 / K-BB% 18.8 / whiff 13.2% / barrel 6.7% / avg velo 92.2 max 99.1 — 是現代 Ace velo + stuff 的標配。ERA 3.02 與 xERA 2.64 落差 0.38 屬正常隨機；gap 主要來自對手 BABIP 分布偶然性而非結構性 luck。**判定：紮實 🟠 Strong Ace 邊際 Elite**，不下修預測。
- **Reverse platoon 信號**：未觸發（vs LHB .671 OPS / vs RHB .564 OPS — 對左打略弱但屬正常 RHP 順位）
- **對手打線威脅**：低偏弱。Giants 屬 🟢 Weak vs RHP（xwOBA .273 / OPS .635），Top 5 全員 sub-.740 OPS，**Devers last7 1.081（BABIP .400）是孤立熱手**，其餘 Chapman .606 / Adames .577 / Lee .702 / Arraez .732 都不對 RHP 構成威脅。Giants 整體 last7 BABIP .241 unlucky-cold 雖有反彈空間（見風險段），但 Top 5 平均 EV95% 31.4%（除 Devers 45.5%）擊球質量本就偏弱，反彈幅度受限。Ashcraft FF 32.1% / CU 28.5% / SL 18.4% 三球種平衡，難以被 weak 打線 squared up。

## 打線評級

### HOME — season tier 🟢 Weak / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak
  - matchup 與 season tier 一致 🟢 Weak，**同意維持 Weak**。chain OBP top3 .270 / SLG mid .391 都在弱打線水準，近 10 場 RS 2.30（場均不到 3 分）也呼應。
- **chain_break / heat_vs_babip 信號**：
  - 🔴 #8-9 OPS 落差 0.443（high）：1-3 棒 OBP .270 即使上壘，4-5 棒 SLG .391 推進有限，到 7 棒之後 chain 徹底斷裂，**單局壓制力強的 RHP 很容易把 inning 結束在第 8-9 棒**；對 Ashcraft 這種三球種平衡型 RHP 是優勢匹配。
  - 🟠 ⏳ unlucky-cold（BABIP .241）：Flag 3 範疇，敘事另列風險段，**不入錨點**。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - matchup 與 season tier 一致 🟡 Average，**同意維持 Average**。1-3 棒 OBP .361（穩健上壘）/ 4-5 棒 SLG .455（中段尚可），近 10 場 RS 5.00 攻擊穩定，整體穩於 Giants。
- **chain_break / heat_vs_babip 信號**：
  - 🟠 #8-9 OPS 落差 0.230（medium）：相較 Giants 0.443 輕微，1-5 棒（Reynolds/Cruz/O'Hearn/Lowe/Gonzales）chain 完整，斷層發生在 8-9 棒；**只要 1-5 棒任一輪有人上壘 + 推進，得分機率高於 Giants**。對 Roupp 的 SI/CU 滾地球路線，需提防雙殺結束 chain。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.35 / 8 / **4（3+ 級 🔴🔴 崩盤）** | 4.07 / 2 / **0（健康）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：表面 ERA 3.35 看似不錯，但 **core IL ×4（Erik Miller、Hayden Birdsong 等）已達崩盤級**，後段 high-leverage 火力被拔光。實際可用 = 中段 / 低 leverage 為主。**對 Pirates 末段威脅明顯下修**：若 Roupp 6 IP 後（K% TTO3 -9pp 暗示體力下滑）退場，Giants 必須用降級 RP 守 7-9 局，Top 5 OPS .720+ 的 Pirates 打線在這段是高 EV 對象。**這是本場最大基本面 swing 因子**。
- AWAY 牛棚：ERA 4.07 中段水準但 **core IL 0（健康完整）**，setup / closer 鏈完整；**對 Giants 末段是平均到偏優勢匹配**。考慮到 Giants weak 打線 + last7 BABIP .241（即使部分反彈），Pirates RP 守 4-5 分領先穩定度高。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.241):
  - **半結構性 + 部分壞運**。Giants Top 5 EV95% 31.4%（除 Devers 45.5% 外集中在 18-38% 中低段）說明擊球質量本就偏弱，BABIP 即使從 .241 反彈到聯盟均值 .295，受惠的也是接近 league-avg 區間（.700 OPS）而非顯著爆發；**反彈空間有限、不必下修對 Ashcraft 的壓制預期**，但要留意 Devers 個人熱手延續可能單局打破比分。本場判斷仍以 Giants weak lineup 為主，**不自動 ±run value**。

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ +0.010（TTO1 0.526 → TTO3 0.536），第三輪明顯衰退；K% 從 30.2% 掉到 21.2%（Δ -9.0pp）
- 🟠 AWAY TTO3 penalty：OPS Δ +0.002（TTO1 0.553 → TTO3 0.555），第三輪明顯衰退；K% 從 34.9% 掉到 23.1%（Δ -11.8pp）
- 🔴 HOME chain breaks at #8-9：OPS 落差 0.443
- 🟠 AWAY chain breaks at #8-9：OPS 落差 0.230
- 🔴 ⏳ HOME 牛棚 core IL ×4：🔴🔴 極高（牛棚崩盤級）
  - **本場主要 leverage swing factor**。與 Flag 3 形成**反向抵消**：HOME 打線可能反彈（+ 攻）但 HOME 牛棚崩盤（+ 失），淨影響仍偏向 AWAY 受惠（Pirates 後段對降級 RP 攻擊期望值 ↑）。⏳ 短半衰期信號需留意賽前 IL 狀態異動。

## 條件修正

- Park Factor: 91.0 → -0.45 run
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：雙 Elite Ace 對決（tier_v2 同級），formula base 已反映；**雙方先發深度同級 → 不額外修正**，但 Roupp velo 86.7 偏低、SI-heavy 滾地球依賴防守，Ashcraft 92.2 + 平衡球種更現代化，實戰可控變數略偏 Pirates。Doubleheader 否（系列賽 G2）。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 2.4 | -0.15（chain_break high -0.25 + Ashcraft tto3 +0.10） | 2.25 |
| AWAY | 2.6 | +0.55（core_il +0.5 + Roupp tto3 +0.10，cap 同向 max+0.1 = 0.6；chain_break -0.05） | 3.15 |
| Total | 5.0 | +0.40 | 5.40 |

## 整體判斷

- **方向（基本面）**：**AWAY (Pirates) 略佔**（adjusted 3.15 vs 2.25，邊際 +0.9 run）。核心驅動：(1) Pirates 打線 Average vs Giants Weak、(2) Pirates 牛棚健康 vs Giants core IL ×4 崩盤級、(3) Ashcraft velo / 球種組合略佔現代化邏輯優勢。
- **總分（基本面）**：**5.4 run**（base 5.0 + 0.4，受 Giants 牛棚信號驅動上修，但雙 Elite Ace + Oracle Park PF 91 仍把絕對值壓在 5-6 區間）。
- **方向信心**：**約 58%（low-confidence AWAY）**。Pirates 結構性優勢在牛棚 + 打線匹配，但 Giants 連勝中（streak +1，G1 已贏）+ 主場 + Roupp 對 Pirates 滾地球路線的具體適配性帶來足夠 noise，不到 60%。
- **風險**：
  1. **Roupp 滾地球路線 vs Cruz/O'Hearn 高 EV95% 群**：球被 squared up 到 SI 區 + 90+ EV → Oracle Park 也壓不住二三壘打，可能單局打破基本面預測。
  2. **Giants BABIP 反彈 + Devers 熱手延續**：last7 BABIP .241 與 Devers 1.081 OPS 是雙刃 — 若 Devers 帶起 #2 棒 chain，1-3 棒 OBP 可能超過 .270 預測。
  3. **Pirates 連敗 streak (-1) + G1 已輸**：心理 / momentum 因素是 noise 但實戰常見 swing。
  4. **打線未公布（projected）**：兩隊都是 PA 排序近似，若 Giants 把 Devers 上抬至 #2 或 Pirates 把 Lowe 上抬至 #3，chain 結構會優於 dossier 預測。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組