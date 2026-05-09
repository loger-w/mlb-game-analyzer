## 投手對決

### Kumar Rocker (HOME, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 score-derived = 🟢 Back-end Starter（ERA 4.71 / xERA 4.85 對齊，無 luck gap）
  - **FIP-blend 視角偏 🟡 Solid Starter**（FIP 3.83 / xFIP 3.61 / K-BB% 11.0 / WHIP 1.47），ERA 與球質之間有結構性落差：xFIP 把他壓到 3.6，但實際結果 4.71；多半是 hard_hit% 32.0 + barrel% 8.7 + WHIP 1.47 偏高的「Stuff 不差但被打中能轉化得分」型 sinker baller。GS = 6 樣本仍小，不下修預測，採 🟡 Solid Starter 為敘事基準。
- **Reverse platoon 信號**：未 fire（vs LHB OPS .830 vs RHB OPS .650 → 正常右投 platoon，且 vs LHB 大幅吃虧 .295/.371/.459）
  - vs LHB 偏弱對 Cubs 兩名 LHB 核心（Busch / PCA）特別不利。
- **對手打線威脅**：🟠 Strong（vs RHP）+ 🟢 Cubs 近 10 場 9-1、RS 5.90、9 連勝中。Rocker 主球 SI 38.9% 但 RV/100 = -2.2（主球種被打），Cubs 上修打線（Happ vs RHP .969、Busch last7 1.266、PCA last7 .877）正好能吃 SI/SL。Rocker TTO splits 反向（TTO3 .657 < TTO1 .817，70 BF 小樣本），第一輪反而最危險。

### Ben Brown (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 score-derived = 🟠 Strong Ace（ERA 2.10 / xERA 2.79，gap ~0.7 → Flag 8 中度，ERA 領先球質）
  - FIP 2.67 / xFIP 3.38 → **真實水平偏 🟡 Solid Starter 上緣**（K-BB% 15.9 / WHIP 1.01 / barrel% 4.3 確實是優於平均的接觸壓制能力，但不到 Strong Ace 的 K-BB%）。Flag 8 紀律：不自動下修預測，採 🟠 Strong（球質支撐尚可）為敘事基準，惟 base score 4.3 已含 Strong tier 加成，提醒運氣回歸可能讓單場失分多 0.5-1.0 run。
- **Reverse platoon 信號**：🟠 fired（vs RHB OPS .549 > vs LHB OPS .461，Δ +0.088）
  - **本場有利 Brown**：Rangers top 5 中 3 名 LHB（Seager #1 / Nimmo #2 / Carter #5），Brown 對 LHB 反而壓制更強（.175/.261/.200, 46 BF）。KC（knuckle curve）35.5% 對 LHB 破壞下沉，是 reverse platoon 的物理依據。
- **對手打線威脅**：🟠 Strong（vs RHP）但 matchup-specific 被 Brown 反向 platoon 抵銷。Rangers last7 BABIP 0.258 偏冷，但 Burger / Jung（RHB）反而是 Brown vs RHB 較難壓制的一邊，#3-4 棒 Burger / Jung 是真實得分點。Brown TTO3 career-fallback 87 BF Δ +0.294 → 🔴 high 級 TTO3 fade 是本場最大破口（見風險段）。

## 打線評級

### HOME (TEX) — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong → 較 season tier **上修一檔**
  - Seager / Nimmo / Jung / Carter 對 RHP OPS 都比 season 高（Δ +0.10 ~ +0.12），但 Brown 反向 platoon + Burger 季 OPS 僅 .617 拖累 chain，整體不到 Strong Ace 對手等級的「壓制不住」。本場對 Brown 期望接近 season tier 🟡。
- **chain_break / heat_vs_babip 信號**：
  - 🟠 chain break #4-5（Jung .875 → Carter .631, Δ 0.244）→ 進攻 chain 在中段斷裂，4-5-6 棒回不到頭頂。
  - last7 BABIP 0.258 偏冷（Flag 3）但 heat 為 Normal → 非 unlucky_cold 標準觸發，可能是樣本偏低尚未爆發；Brown 假如壓制力如 ERA 表面那麼強，今天可能延續這個低 BABIP 模式。

### AWAY (CHC) — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong（與 season tier 相同 → 不修正）
  - 但 last7 個別熱度顯著高於 Normal 標籤：Happ last7 OPS 1.180 / Busch 1.266 / PCA .877 — 三人連續 1 週爆打。Cubs 整體 9-1、RS 5.90、9 連勝 → 進攻引擎運作中。對 Rocker 主球 SI（RV/100 -2.2）有結構性放大優勢。
- **chain_break / heat_vs_babip 信號**：
  - 🟠 chain break #3-4（Happ .876 → Busch .699 season → 但 last7 兩人都 1.18+）→ season 數據看 chain 斷在 #3，但近 7 天兩人同步爆熱 → **本場 chain break 信號實際被 last7 hotness 抵消**，AI 判讀為「season 結構性弱點，但短期被熱度填補」。

## 牛棚

| | HOME (TEX) | AWAY (CHC) |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 2.98 / 6 / 0 | 3.90 / 10 / 4 |

### 牛棚影響判讀
- **HOME (TEX)**：2.98 ERA 在聯盟前段，0 名核心 IL → 後段防守完整。Brown 若 TTO3 fade 撐不過 5-6 局，Cubs 走進 TEX 健康牛棚是「投手強隊接力」，TEX 後段把比分鎖死的能力強。
- **AWAY (CHC)**：3.90 ERA 中等偏上，**核心 IL ×4（Caleb Thielbar / Hunter Harvey 等）→ 🔴🔴 牛棚崩盤級**。Brown 一旦提早被換下（TTO3 Δ +0.294 是非常大的衰退量），Cubs 沒有 high-leverage 替補可用，Rangers 後段（7-9 局）對 CHC 牛棚是 plus matchup。

## 風險提示

- ⚠️ **HOME (TEX) 打線 Flag 3 (last7 BABIP=0.258)**：
  - 偏向「冷得名實相符」（heat=Normal 沒有 hot 但打不出去，多半是球員整體狀態壓抑而非運氣偏低）。對 Brown 這種接觸壓制型 RHP，BABIP 短期更可能延續低位（Brown barrel% 4.3 / hard_hit% 22.7 是聯盟前段壓制力）→ **不大幅上修 TEX 得分期望**。

### 額外信號
- 🟠 **AWAY reverse platoon Δ +0.088**（已併入投手分析）→ **本場有利 Brown**（Rangers 3 LHB 在 top 5），抵銷 Strong matchup tier 的部分壓力。
- 🔴 **AWAY TTO3 penalty Δ +0.294**（87 BF career fallback, severity high）→ Brown 第三輪打擊區明顯衰退，K% 從 28.3 → 24.1（Δ -4.2pp）。本場 Cubs 教練組大概率會在 5-6 局看到 TTO3 信號就準備換投，但 **CHC 牛棚 core IL ×4 的雙重壓力放大此風險** — 沒有可信 high-leverage 接手，Rangers 7-8 局（Seager / Nimmo / Jung 第二三次面對牛棚 RP）是最大失分窗口。
- 🟠 **HOME chain breaks at #4-5（OPS Δ 0.244）**：Jung → Carter 落差大，Rangers 4 號棒打完後續攻擊串聯難 — 即使 Brown TTO3 失分，得分量也受 chain 結構限制，難得大局（≥ 4 run/inning）。
- 🟠 **AWAY chain breaks at #3-4（OPS Δ 0.177）**：season 結構，但 last7 Busch 1.266 已實質填補；本場若 last7 熱度延續 → 信號失效；若回歸 → chain break 重新成立。
- 🔴 ⏳ **AWAY 牛棚 core IL ×4**（已併入牛棚段）→ 與 Brown TTO3 信號 **同側同向 fire**，依 §量級錨點累積規則：取單側 max 區間 + 0.1，cap +0.8 / 場（不直接相加）。

## 條件修正

- **Park Factor**：Globe Life Field PF 96.0 → -0.20 run（投手友善端、HR +6% 略補）；retractable roof，公布前默認 indoor。
- **天氣**：未公布 + 多半 indoor（roof closed）→ 跳過天氣分析。
- **先發 tier / doubleheader**：單場、無 doubleheader；兩位先發都 26 歲（peak），無 TJ / age decline / role conversion 議題。
- **Cubs 連勝動能**：9-1 last 10、+9 streak、RS 5.90 vs season 5.73（差距僅 +0.17 → 熱度不誇張、屬實打）→ 不額外修正，已隱含於 base score 4.3。
- **Rangers 近況**：3-7 last 10、RS 2.70 vs season 3.47（攻擊熄火 -0.77）→ Rangers 進攻趨勢向下，但 Flag 3 BABIP 低顯示有運氣成份；不額外修正，base 2.8 已反映。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (TEX) | 2.8 | **+0.4**（TTO3 + core_il 同側同向 max+0.1 cap +0.8 - chain_break -0.2 - reverse_platoon 抵銷 -0.2） | **3.2** |
| AWAY (CHC) | 4.3 | **-0.2**（season chain_break #3-4 -0.2，last7 熱度部分抵銷但 conservative 取下界） | **4.1** |
| Total | 7.1 | +0.2 | **7.3** |

### + 信號計算說明
- **HOME side（TEX 得分）**：
  - Brown TTO3 penalty (high) +0.2 ~ +0.5
  - Brown reverse platoon (medium, 對 TEX 不利方向 -0.1 ~ -0.3)
  - CHC core_il ×4 (3+ 名級, +0.4 ~ +0.8)
  - HOME chain_break #4-5 (medium, -0.1 ~ -0.3)
  - 同側同向 fire（TTO3 + core_il 都 push TEX up）：取單側 max 區間 + 0.1 = 0.8 + 0.1 = 0.9 → **cap +0.8**
  - 反向減項：reverse_platoon -0.2 + chain_break -0.2 = -0.4
  - 淨：+0.8 - 0.4 = **+0.4**
- **AWAY side（CHC 得分）**：
  - AWAY chain_break #3-4 (medium, -0.1 ~ -0.3)
  - 取保守下界 -0.2（last7 熱度有抵銷）
  - 淨：**-0.2**

## 整體判斷

- **方向（基本面）**：**AWAY (Cubs)** 略優
- **總分（基本面）**：**~7.3 run**（TEX 3.2 / CHC 4.1）
- **方向信心**：**60%**（CHC 略優方向，但下方四點風險拉低信心）
- **風險**：
  1. **CHC 牛棚 core IL ×4 + Brown TTO3 fade 雙重壓力**：若 Brown 撐不過 5-6 局（career TTO3 OPS .965 是聯盟平均打者上緣），Cubs 走進 high-leverage 真空 → Rangers 7-9 局可能反咬，本場最大下方 tail risk（CHC 即便取得領先也守不住的情境）。
  2. **TEX last7 BABIP 0.258 + Brown 接觸壓制**：方向不明的 toss-up — 若 BABIP 延續低位，TEX 進攻仍熄火（Brown 也壓得住）；若 Burger / Jung（vs RHP 較強的 RHB）逮到 KC/SI 變化球的失投 → 一發逆轉。
  3. **兩位先發樣本都小**（Rocker GS 6、Brown 季初）：Tier 判定 fragile，本場真實表現可能落在預期之外 ±0.5 run。
  4. **Cubs 9-1 連勝壓力**：規律的均值回歸概率存在（雖然 RS 5.90 vs 5.73 季均差距僅 +0.17，純粹「實打 hot」而非「BABIP lucky hot」），但 9 連勝後第 10 場心理動能可能反轉，特別是 Rangers 主場 + Globe Life Field（CHC 客場）。
