## 投手對決

### Cam Schlittler (HOME, RHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +1.1
  - 同意 Elite Ace。ERA 1.51 與 xFIP 2.33 / FIP 1.49 / K-BB% 27.6 / WHIP 0.74 全部對齊頂尖水準，gap 僅 +1.1 表示 ERA 並未顯著失真。velo 95.1 (max 100.1) + whiff 14.7% + 主球種 FF/FC/SI 全為正 RV，結構性支撐強。
- **Reverse platoon 信號**：未 fire（vs LHB .520 OPS / vs RHB .331 OPS，正向 platoon 差距 +0.189 屬正常 RHP 模式）。
- **對手打線威脅**：BAL 打線以 RHB 為主，正撞 Schlittler vs RHB .113/.154/.177（66 BF）的吸血區。Ward / Alonso / Basallo 雖 vs RHP 季節 OPS .77-.81 不算弱，但面對 Elite Ace 級 xFIP 2.33 +TTO3 OPS Δ -0.023（撐得住第三輪）→ BAL 攻勢結構性壓制，預期得分上限低。

### Shane Baz (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p85, K-BB% p61），gap vs ERA-only = +32.7（high）
  - |gap| ≥ 20 → tier_mismatch 高度發散：xFIP 3.55 / FIP 3.51 vs ERA 4.50，FIP 系列一致指向 Strong Ace 級真實水平，ERA 4.50 屬運氣 / sequencing 偏差（BABIP 高、HR/FB 不利）。**敘事修正不下修預測**——但要注意他並非 ERA 看起來那麼差。
- **Reverse platoon 信號**：未 fire，但 vs LHB .361/.424/.542（85 BF）vs RHB .230/.262/.426（65 BF）→ 正向 platoon 但 vs LHB 數字異常糟糕（.966 OPS）。CU 32.0% + KC（curve-heavy）對 LHB 弱化，RV CU -2.0 印證。
- **對手打線威脅**：🔴 高度警戒。NYY 前 5 棒中 4 名 LHB（Bellinger / Rice / Chisholm / Grisham）+ 1 名 RHB Judge（.983 vs RHP）。Rice 1.173 vs RHP / Bellinger .836 / Judge .983 — 三名核心都在 LHB-friendly 球場 + Baz vs LHB 弱點上集火，xFIP 雖好但本場手別匹配對 Baz 極不利。

## 打線評級

### HOME — season tier 🟠 Strong / heat 🔥 Hot
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong
  - season 與 matchup 一致為 Strong，且本場 vs Baz 形成「LHB 集火 vulnerable 投手」結構優勢。Hot 熱度 + last7 BABIP 0.291（中性）→ 熱度有實質支撐而非運氣堆出來，**評估方向 = 同意（略傾上修）**。
- **chain_break 信號**：🔴 fire #3-4（OPS 落差 0.584）— Rice 1.214 → Chisholm .630 出現大斷層。但落差來源是 Rice 太強而非 Chisholm 太弱，#1-3（Judge / Bellinger / Rice）OBP top3 = .410 仍提供穩定壘上人，後段（Caballero / Wells / McMahon / Rosario）OPS .59-.83 也不算空洞 → **chain 對總分壓制效應有限**，主要靠前 3 棒 + 散打輸出。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak
  - season Average 但 matchup vs RHP 降為 Weak — 本場對 Schlittler（vs RHB .113/.154/.177）匹配極差。前 5 棒多為 RHB，正撞投手吸血區。**評估方向 = 下修**。
- **chain_break 信號**：🔴 fire #8-9（OPS 落差 0.347）— Taveras .819 → Alexander .472 末段崩塌。前段 Ward / Henderson / Alonso vs RHP OPS 約 .70-.81 也只屬中等，整體串聯性弱 → 若無法在前 5 棒掛分將出現大段空白局。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.51 / 4 / 0 | 4.58 / 7 / 2 (Bautista 60d / Helsley 15d) |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 3.51 屬聯盟中上，0 名 core IL（健康）。NYY 近 10 場 RA 3.20 顯示牛棚與守備合力壓制有效，後段防守可靠 → 對 BAL 末段攻勢威脅高，BAL 若進入 7 局後仍未掛分機會驟降。
- AWAY 牛棚：ERA 4.58 偏高 + Bautista（Closer）/ Helsley（Setup）雙核心 IL → 對應 §牛棚累計效應「2 名核心 = 🔴 高」。Schlittler 預期能撐第三輪（TTO3 OPS Δ -0.023），但 Baz xFIP 雖好其 ERA 4.50 + last 3 場 ER 8 / 16.0 IP 顯示中段就可能下莊；若 Baz 5-6 局退場，BAL 將被迫提前用 high-leverage 替補，NYY hot 打線在弱化牛棚前易擴大比分。

## 風險提示

- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.26):
  - BAL 連敗 -3 期間 BABIP 0.260 偏低，理論上有向 .300 聯盟均值反彈空間（運氣成分）。**但**：(a) 樣本僅 7 天，雜訊極大；(b) 對手 Schlittler vs RHB .113/.154/.177 結構性壓制 → BABIP 即便回升，仍受先發投手品質框死。**敘事判讀傾向「回歸機會存在但本場不太可能兌現」**，不調整 ±run value。

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ +-0.023（TTO1 0.475 → TTO3 0.452），第三輪明顯衰退；K% 從 31.9% 掉到 20.0%（Δ -11.9pp）
- 🔴 AWAY TTO3 penalty：OPS Δ +0.261（TTO1 0.625 → TTO3 0.886），第三輪明顯衰退；K% 從 25.4% 掉到 17.5%（Δ -7.9pp）
- 🔴 HOME chain breaks at #3-4：OPS 落差 0.584
- 🔴 AWAY chain breaks at #8-9：OPS 落差 0.347
- 🔴 ⏳ AWAY 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - 本場直接放大：Baz 中段下莊機率不低（近 3 場 8 ER / 16 IP），BAL 替補後段 ERA 4.58 對上 NYY 🔥 Hot 打線（last 10 RS 6.20）→ 6-9 局失分風險顯著上升。與 Flag 3（BAL BABIP 偏低）方向相反但量級不對等：bullpen 吃緊壓 BAL 守，BABIP 回歸幫 BAL 攻 — 淨效應仍偏 NYY。

## 條件修正

- Park Factor: 96.0 → -0.20 run（runs 略偏投手友善；但 Yankee Stadium HR +12% 對 LHB 友善 → 對 NYY 前 5 棒中 4 名 LHB vs Baz 是隱性加成，formula 內 PF 已平均化處理，敘事提醒即可）
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：兩位先發皆為 RHP / 巔峰期（25/26 歲），無 doubleheader、無 TJ 復出階段。tier 落差 Schlittler Elite Ace vs Baz Strong Ace（真實水平）— 紙面 +1 檔差，雖然 Baz 比 ERA 顯示的好，但仍處於下風位。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.1 | +0.3（AWAY tto3 high +0.3 + AWAY core_il×2 high +0.3，interaction 取 max 不相加，再扣 HOME chain_break -0.2，cap 內）| 4.4 |
| AWAY | 1.6 | -0.1（HOME tto3 medium 但 OPS Δ 為負 + 微小 K% 利得 +0.05；AWAY chain_break #8-9 high -0.15）| 1.5 |
| Total | 5.7 | +0.2 | 5.9 |

## 整體判斷

- **方向（基本面）**：HOME (NYY)
- **總分（基本面）**：5.9
- **方向信心**：~70%（NYY +4 streak / 昨日 11-3 大勝 / Schlittler Elite vs Baz LHB-vulnerable / BAL 牛棚 2 核心 IL — 多重結構性優勢同向）
- **風險**：
  1. **Baz 真實水平被 ERA 低估**（xFIP 3.55 / FIP 3.51 → Strong Ace 級）— 若本場 BABIP / sequencing 修正回歸真實水平，NYY 4.4 base 可能高估 0.5-1.0 run
  2. **Judge last7 BABIP .462 不可持續**（含明顯運氣成分）— hot streak 一旦 cool down，NYY 攻勢上限收斂
  3. **BAL last7 BABIP .260 反彈機會** — 雖被 Schlittler 對 RHB 結構壓制，仍存在零星散打開洞風險
  4. **單場隨機性 40-45%** — 量化信號全同向 → 但 single-game variance 仍可吃掉所有結構優勢，BAL 連敗中亦可能爆冷反彈

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組