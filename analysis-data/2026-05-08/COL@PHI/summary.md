## 投手對決

### Jesús Luzardo (HOME, LHP, 28 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +62.0
  - 同意 Elite Ace 判定。FIP 2.53 / xFIP 2.02 / K-BB% 24.5% / Hard Hit% 20.0%（聯盟頂級壓制接觸品質）四項一致指向 elite 層級。ERA 5.09 與 xERA 3.17 落差 +1.92（Flag 8）+ 近 3 場 ER/IP 12/17.3 ≈ 6.23 ERA → 屬於 BABIP / sequencing 運氣偏差延續未回歸，**不視為結構性退化、不下修預測**；但要承認運氣回歸時點不可控，今天可能還沒到 mean-revert
- **Reverse platoon 信號**：未 fired。vs LHB OPS .631 (37 BF) < vs RHB OPS .726 (135 BF) 為正常 LHP 順向 platoon
- **對手打線威脅**：低—中。COL 打線 vs LHP 評為 🟢 Weak（Rumfield .423 / Goodman .645 / Tovar .646 / Johnston .399 vs LHP 全低），對 Luzardo balanced 球種組合（ST 36.1% / FF 25.0% / CH 21.2%）難找突破口。主要威脅僅 Karros (.755 vs LHP) 一棒+ Tovar/Goodman 偶發長打

### Chase Dollander (AWAY, RHP, 24 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +24.7
  - 部分同意。xFIP 3.03 / K-BB% 19.2% / FIP 3.21 一致指向 Solid-to-Strong 區間，腳本評 Elite Ace 偏樂觀。GS 1 為本季先發場次極少，advanced split（vs LHB 69 BF / vs RHB 87 BF）多為 career fallback，單場波動性偏高。Hard Hit% 27.3% / Barrel% 10.1% 略高於投手友善值，CBP 球場放大下需留意；gap 解讀為「指標彼此一致但樣本未飽和」而非運氣偏差
- **Reverse platoon 信號**：未 fired。vs LHB OPS .747 > vs RHB OPS .595（Δ ~0.150 偏向 LHB 側但仍屬 RHP 正常順向 platoon，未達 reverse 門檻）。但 PHI 打線前段 Schwarber/Harper 為 LHB 主力，會放大 vs-LHB 較弱那一側
- **對手打線威脅**：中。PHI 打線 vs RHP 為 🟡 Average，但前段 Schwarber .983 / Harper 1.027 vs RHP OPS 為頂尖區間，且 Harper last7 OPS 1.262 處於熱潮；CBP 主場 + HR PF +16% 一發長打風險明顯

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - 同意 matchup tier 與 season tier 一致。但內部分裂明顯：top 3 (Schwarber/Turner/Harper) 三人 vs RHP OPS 約 .917（Strong+ 區間）負責絕大多數攻擊，#4-5 出現懸崖（Bohm .419 / Garcia .666 vs RHP）。本場攻擊主要靠前段三棒推進，後段串聯薄弱
- **chain_break / heat_vs_babip 信號**：HOME chain breaks at #4-5（OPS Δ 0.268，🟠 medium）
  - 影響：top 3 結束後得分 chain 中斷，4-5 棒 OPS 落差顯著。靠 Schwarber/Harper 單發 HR 才能跳過缺口；若 Dollander 能避開 LHB 主場景，PHI 中段難長串聯。Schwarber/Turner last7 OPS .554/.549 處冷期（last7 BABIP 偏低 ⏳，可能反彈）但 Harper 1.262 last7 抵銷一部分

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - **下修**。Matchup tier 比 season tier 低一檔，反映 COL 打線對 LHP 結構性弱（Rumfield .423 / Johnston .399 / Goodman .645 vs LHP）。本場 Luzardo LHP elite metrics + COL vs LHP weak → 攻擊上限明顯壓低，AWAY 得分預期應低於 base 2.8
- **chain_break / heat_vs_babip 信號**：AWAY chain breaks at #6-7（OPS Δ 0.320，🔴 high）；last7 BABIP 0.355（接近 ≥ .350 lucky-hot 門檻 ⏳ short half-life）
  - chain break：6-7 棒位 OPS 大斷層 → 後段幾無攻擊力，pitcher 過 5 棒後幾乎進入 reset 區間
  - heat_vs_babip：Karros last7 BABIP .615 / Johnston .500 完全不可持續，Rumfield last7 .903 OPS 也含 BABIP 運氣成分。一旦回歸，COL 攻擊上限再壓低；判讀為「last7 表面熱但結構不可持續，本場任何回歸都壓 AWAY 得分」

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.24 / 3 / 0 | 4.31 / 4 / 0 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：4.24 ERA 屬聯盟中段，core 角色全員可用（IL 0）。鑒於 Luzardo TTO3 OPS Δ +0.628 fire（高機率第三輪前換投），第 5-6 局後牛棚對 COL 後段（chain break #6-7 後幾無攻擊力）相對好守。Phillies last 10 RA 3.40 顯示守備端整體在熱期，後段防守可信
- AWAY 牛棚：4.31 ERA 接近 HOME 但 last 10 RA 5.90 顯示守備端正在被打開（可能含先發爆掉因素，但牛棚消耗會被連帶放大）。Dollander TTO3 Δ +0.267 + GS=1 樣本小 → 教練可能更早換投 → 牛棚負擔重；面對 PHI 前段 Schwarber/Harper LHB 主力組合，COL 牛棚 LHP 對應不足會被點名打。後段失分風險明顯偏 AWAY 牛棚這側

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=+1.92):
  - 判運氣偏差，非結構性退化。FIP 2.53 / xFIP 2.02 / K-BB% 24.5% / Hard Hit% 20.0% / Barrel% 5.5% 五項全聯盟頂級壓制接觸品質、三振保送 — 唯獨 ERA 與 BABIP/HR-rate/sequencing 連動的部分膨脹。**不自動下修預測**；但近 3 場 ER/IP 12/17.3 顯示運氣回歸尚未發生，本場仍有「再爆一次」尾巴風險，計入風險段而非預期值

### 額外信號
- 🔴 HOME TTO3 penalty：OPS Δ +0.628（TTO1 0.438 → TTO3 1.066），第三輪明顯衰退；K% 從 29.7% 掉到 24.0%（Δ -5.7pp）
- 🔴 AWAY TTO3 penalty：OPS Δ +0.267（TTO1 0.739 → TTO3 1.006），第三輪明顯衰退；K% 從 24.1% 掉到 19.6%（Δ -4.5pp）（career fallback）
- 🟠 HOME chain breaks at #4-5：OPS 落差 0.268
- 🔴 AWAY chain breaks at #6-7：OPS 落差 0.320
  - 雙方 TTO3 penalty 都 fire（Luzardo 50 BF 小樣本 / Dollander career fallback）→ 比賽進入 5-6 局後牛棚對決決定後段；但 HOME 投手 TTO3 penalty 對應的對手是 COL 弱打 vs LHP，penalty 釋放出的得分機會被 COL 自身 chain break #6-7 吃掉；AWAY 投手 TTO3 penalty 對應的對手是 PHI 強前段 + chain break #4-5，penalty 釋放反而落在 PHI 中段 cliff 也部分被吃。整體 TTO3 penalty 雙向 fire 的場景下，因 chain break 落點不對稱（COL 後段更弱），淨效應仍小幅偏向 HOME

## 條件修正

- Park Factor: 104.0 → +0.20 run
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：雙先發都 tier_v2 評為 🔴 Elite Ace，但 Luzardo 樣本飽和（GS 7）、metrics 五項一致；Dollander GS 1 多為 career fallback，分級偏樂觀，依紀律不下修但要計入單場波動；非 doubleheader

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.6 | +0.2 (AWAY TTO3 penalty, medium) − 0.2 (HOME chain break #4-5, medium) = 0.0 | 3.6 |
| AWAY | 2.8 | +0.1 (HOME TTO3 penalty, 但 COL vs LHP 弱取下界) − 0.3 (AWAY chain break #6-7, high) = −0.2 | 2.6 |
| Total | 6.4 | −0.2 | 6.2 |

> ⛔ 不入 + 信號（Table A）：Luzardo Flag 8（已敘事於 §風險提示）、COL last7 BABIP .355 heat_vs_babip（敘事）、Park Factor 104（已含於 base 公式倍率）。雙側修正均在 cap ±0.8 內。

## 整體判斷

- **方向（基本面）**：HOME（費城人）
- **總分（基本面）**：6.2（HOME 3.6 / AWAY 2.6）— Under-leaning 區間（vs 球迷直覺的「兩 Elite Ace + CBP」可能對應的中性 7-7.5 lookup line）
- **方向信心**：65%。理由：Luzardo metrics 五項一致 elite + COL 是聯盟下游 vs LHP 對戰之一 + Phillies last 10 (7-3, RA 3.40) 整體在熱期 + 主場 + COL last 10 (2-8, RA 5.90) 守備崩。未到 75%+ 是因 Luzardo 近 3 場 ER 仍未回歸 + Dollander 樣本極小 + CBP HR 因子 +16% 風險
- **風險**：
  1. **Luzardo 運氣回歸尚未發生**：metrics elite 但近 3 場 ER/IP=12/17.3 ≈ 6.23 ERA，BABIP/sequencing 運氣偏差延續，本場仍有「再爆一次」尾巴；若爆掉時點與 PHI chain break 不同步（亦即非 HR 一發）會拉長失分局
  2. **CBP HR Park Factor +16%**：Schwarber / Harper LHB 一發 + COL Karros (vs LHP .755) 偶發長打都被放大；HR 風險獨立於 base 公式，可能單發改寫總分
  3. **Dollander GS=1 樣本飽和度低**：xFIP 3.03 / K-BB% 19.2% / vs LHB 69 BF 多為 career fallback，單場波動性偏高；PHI 前段 Schwarber/Harper LHB 主力配對其 vs-LHB OPS .747 該側，是放大波動的對位點
  4. **雙短半衰期信號 ⏳ 反向**：COL last7 BABIP .355 lucky-hot（Karros .615 / Johnston .500 不可持續）+ PHI Schwarber/Turner last7 OPS .554/.549 cold；任一回歸發生本場都偏 HOME 方向（COL 回歸 → COL 得分更低；PHI 回歸 → PHI 得分更高）

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組