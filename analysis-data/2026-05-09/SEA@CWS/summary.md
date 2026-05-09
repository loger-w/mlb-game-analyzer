## 投手對決

### Anthony Kay (HOME, LHP, 31 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = ⚪ Below Average（xFIP p13, K-BB% p12），gap vs ERA-only = +6.9
  - 同意 ⚪ Below Average。gap +6.9 微幅，雙模型一致。值得注意的是 ERA 5.70 vs xERA 7.64（Flag 8 Δ -1.94）— ERA **高估** Kay 的真實品質（xERA 顯示更糟）。近 3 場 4 ER / 14.7 IP（2.45 ERA）是好區間，但 xERA 不認帳：whiff% 7.7 極低、hard_hit% 31.2、barrel% 12.6，球質壓不住打者。**結構性退化（年齡 31 + velo 90.7 + K-BB% 3.5）多於運氣**，本場對 RHB-heavy 打線高風險。
- **Reverse platoon 信號**：未 fired（vs LHB 30 BF .077/.200/.077 樣本未達 reverse 觸發），但 vs RHB 113 BF .347/.434/.632 是大樣本實錘 — **本場 Mariners 主打線 Julio (.982 vs LHP)、Arozarena (.819)、Young (.798) 三位 vs LHP 數據都比 season OPS 高**，Kay 對 RHB 的崩盤對位放大。
- **對手打線威脅**：⚠️ **高威脅**。Kay 真實品質 ⚪ Below Average 但 xERA 7.64 比腳本 tier 更悲觀；vs RHB .632 SLG 是 113 BF 大樣本，Mariners 1-2-5 棒（Julio/Arozarena/Young）vs LHP 全在 .798 以上，會直接攻擊 Kay 的弱手別。FF（31% 使用）RV/100 -2.5 是其最差球種，Mariners 打者吃 FF 機會多。

### Luis Castillo (AWAY, RHP, 33 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p63, K-BB% p57），gap vs ERA-only = **+49.0**（high tier_mismatch）
  - 同意 🟡 Solid Starter，**gap +49.0 是運氣 + 樣本噪音為主**：ERA 6.29 / xERA 5.86 / FIP 3.94 / xFIP 3.94 — peripheral 數據（K-BB% 11.3、xFIP 3.94）支持 Solid Starter，被打 .366/.423/.451 vs RHB（78 BF）BABIP 必有偏高。近 3 場 10 ER / 13.0 IP 是 ERA 主要拖累來源；但 velo 91.6（生涯 avg ~95-96）+ 年齡 33 📉 是 **隱含結構性風險訊號**。**本場給 🟡 Solid Starter 待遇但帶警示**：xFIP 3.94 該值得相信，BUT velo 退化是真的。
- **Reverse platoon 信號**：未 fired（vs LHB .493 SLG / vs RHB .451 SLG 差 < 0.080）。但 vs LHB 0.826 OPS（小樣本 82 BF）值得留意 — 與 sweeper-heavy RHP 預期一致（SL 26.5%）。Sox 打線 LHB 較少（Vargas、Benintendi），影響有限。
- **對手打線威脅**：⚠️ **中等**。Sox vs RHP 整體偏弱（top 5 平均 ~0.700 OPS），Murakami（.963）是唯一爆破點，他的 EV95% 63.3 / Barrel% 22.8 是真材實料的力量威脅；剩餘 Vargas（.638）、Meidroth（.673）、Benintendi（.652）vs RHP 都偏弱。Castillo 的 SL（26.5% 使用、xwOBA .323、whiff 31.1%）是 swing-and-miss 武器，剋制 Sox 中段。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - matchup tier = season tier，**同意維持 Average**。Sox vs RHP top 5 OPS 表現分歧：Murakami .963（elite）/ Montgomery .769 撐起前段，但 Vargas .638、Meidroth .673、Benintendi .652 屬於聯盟均值偏下 — Castillo SL 主武器（whiff 31%）對中段 vs RHP 弱者更有效。
- **chain_break / heat_vs_babip 信號**：chain_break #5-6 fired（OPS Δ 0.217）
  - 影響 **末段攻擊 chain**（#6-9 棒）— Murakami 單棒爆破能 RBI 自己，但回頭再湊一輪需要末段串聯，Sox 在 Murakami / Montgomery 後若無人上壘，Castillo 配合 SEA 牛棚壓制可逐輪壓低期望分。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - matchup tier 比 season tier 弱一階（🟡 → 🟢）— **下修方向但需細看**：Mariners 5 人 vs LHP 走兩極 — Julio .982 / Arozarena .819 / Young .798 三位 vs LHP **比 season OPS 高 ≥ 0.05**（接近觸發 platoon_advantage 邊緣），但 Cal Raleigh .453 / Naylor .419 vs LHP 是中段毀滅性弱點。**敘事：上下對立 — 1-2-5 棒可開發 Kay，3-4 棒（兩位主力）vs LHP 嚴重壓不住，整體攻擊上限被 #3-4 RBI 點壓低。**
- **chain_break / heat_vs_babip 信號**：chain_break #6-7 fired（OPS Δ 0.272，high 邊緣）
  - 影響 **末段 chain 完全斷裂** — Mariners 末段（projected #6-9）vs LHP 只有極弱補位，加上 Flag 3 last7 BABIP 0.229（Cal Raleigh 近 7 天 BABIP .000 是極端冷期）— 整體攻擊只能靠 1-2-5 棒。冷期可能反彈但 7 天樣本太小，不主動 ±run。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.68 / 4 / 1（Mike Vasil IL60） | 3.28 / 5 / 3（Vargas IL60 / Speier IL15 / +1） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（Sox 4.68 ERA / 1 core IL）：🟠 中高。整體 ERA 偏高（4.68 全聯盟末段），核心 IL 1 名（Vasil）使後段 high-leverage 變薄。系列首戰 G1 已輸 8-12，連 3 敗 streak 心理層面壓力，**對 Mariners 末段攻擊（若 1-2-5 棒上壘 Murakami/Montgomery 之後）防守能力存疑**。
- AWAY 牛棚（SEA 3.28 ERA / **3 core IL**）：🔴🔴 極高（崩盤級）。雖然當前 ERA 3.28 看起來健康，但 **Vargas（IL60）、Speier（IL15）+1 共 3 名核心 IL** 是真正的後段隱患 — 數據是「靠剩下的人撐」的結果。本場若 Castillo TTO3 早退（K% drop -8.2pp 預示效率掉），中後段牛棚要面對 Sox top 段（Murakami 二打席）+ 連戰消耗，**是本場最大下注點 — Sox 後段反撲機會 ↑**。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-1.94):
  - **結構性退化為主**。ERA 5.70 比 xERA 7.64 低（這次方向是 ERA 高估了 Kay 的真實品質） — whiff% 7.7、hard_hit% 31.2、barrel% 12.6 + velo 90.7 + 年齡 31 + K-BB% 3.5 全是「球質壓不住打者」的結構訊號。近 3 場好球率（4 ER / 14.7 IP）是 BABIP 偏低的好運區間，xERA 不認帳。**本場對 RHB 主力 Mariners 高機率衰退**，但 skill 紀律：不自動下修 Kay 預測；以敘事呈現。
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.229):
  - **混合 — Cal Raleigh 個案極端冷 + 整體可能小反彈**。Raleigh last7 BABIP .000（換言之 7 天無安打入場）是 Mariners 主力捕手的小樣本崩跌；Naylor last7 BABIP .435 / OPS .917 反向（已在反彈）。整體看 Mariners 1-2-5 棒（Julio/Arozarena/Young）OPS 健康，主要冷的是 Raleigh 一人。**本場對 LHP Kay**，Raleigh 對 LHP 整季 OPS .453 也低（雙重弱點），冷期回歸機率不高 — 採信 chain_break #6-7 的敘事方向，**不主動加 ±run**。

### 額外信號
- 🟠 AWAY TTO3 penalty：OPS Δ -0.132 反向；觸發點是 K% 從 25.3% → 17.1%（Δ -8.2pp）
  - 解讀：Castillo TTO3 OPS 反而更低（運氣 / 樣本噪音），但 K% 大降是真實效率掉。**意味著 Castillo TTO3 變「靠 contact suppression」而非主動 K** — 教練看到苗頭可能 5-6 局就換投。SEA 牛棚 3 core IL → 提早接手變高風險決策。**這是雙重壓力**（Flag 8 對 Sox 是 Kay；Castillo TTO3 + 牛棚薄）。
- 🟠 HOME chain breaks at #5-6：OPS 落差 0.217（Sox 攻擊在 Murakami / Montgomery 之後變薄）
- 🟠 AWAY chain breaks at #6-7：OPS 落差 0.272（Mariners 末段 vs LHP 串聯斷裂；# 3-4 Raleigh/Naylor vs LHP 也弱化主串聯）
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（Sox 末段防守變薄）
- 🔴 ⏳ AWAY 牛棚 core IL ×3：🔴🔴 極高（SEA 牛棚崩盤級 — 本場最關鍵變數）
  - **本場核心矛盾**：SEA 牛棚 3 core IL 與 Castillo TTO3 早退傾向疊加 → Sox 後段（7-9 局）攻擊機會被放大；同時 Sox 牛棚 1 core IL + ERA 4.68 也不是穩固後盾。**Flag 3（Mariners 打線冷）+ Flag 8（Kay 結構衰）兩端壓力同時存在**：Kay 端壓制更弱 / Mariners 攻擊更冷 → 互相抵消部分；最終本場走勢可能呈現「先發局數雙方 ER 壓低 → 末段牛棚決勝負」格局。

## 條件修正

- Park Factor: 97.0 → -0.15 run（Rate Field 略偏投手友善，HR -1%）
- 天氣：未公布（跳過天氣分析）
- 先發 tier：Castillo 🟡 Solid Starter（xFIP 3.94） vs Kay ⚪ Below Average（xERA 7.64） — Castillo 結構性質量明顯較佳；雙方都 31+ 📉 但 Castillo 近 3 場（10 ER / 13 IP）區間性難堪、Kay 近 3 場（4 ER / 14.7 IP）反向漂亮 — recency 不對齊 underlying，**以 underlying 為準**。doubleheader：N/A。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.2 | +0.4（core_il 3+ Mariners 牛棚 +0.5 / chain_break #5-6 -0.2 / tto3 K% drop +0.1，同向 max + 0.1） | 4.6 |
| AWAY | 6.5 | -0.1（chain_break #6-7 high 邊緣 -0.2 / core_il Sox 1 +0.1） | 6.4 |
| Total | 10.7 | +0.3 | 11.0 |

## 整體判斷

- **方向（基本面）**：**AWAY（Mariners）小幅優勢**。Castillo 真實品質（xFIP 3.94 / 🟡 Solid Starter）vs Kay 真實品質（xERA 7.64 / ⚪ Below Average）落差顯著，Mariners 1-2-5 棒（Julio/Arozarena/Young）vs LHP 全部加成；但 #3-4（Raleigh/Naylor）vs LHP 重大弱點 + Mariners 牛棚 3 core IL 是放大 Sox 末段反撲的隱患。
- **總分（基本面）**：~11.0 run（base 10.7 + 信號修正 +0.3）— **偏多得分傾向**（兩位先發都帶結構或近期失投傾向，雙方牛棚都不穩）。
- **方向信心**：**~58%** 偏 AWAY（基本面有結構性優勢，但近 3 場 recency / Mariners 打線冷期 / Castillo TTO3 早退 + SEA 牛棚崩盤級 IL 都是反向風險，信心不到強推等級）。
- **風險**：
  1. **SEA 牛棚 3 core IL 是本場最大下注點** — Castillo TTO3 K% 大降（-8.2pp）若提早被換，後段 high-leverage 缺口給 Sox（尤其 Murakami 二打席）反撲機會。
  2. **Castillo recency 拖累** — 近 3 場 6.92 ERA，velo 91.6（生涯偏低），年齡 33 退化期；underlying 強但對位若失準，單場可能爆掉。
  3. **Mariners 中段 vs LHP 弱點** — Raleigh / Naylor vs LHP 雙位 OPS < .460，攻擊上限被 #3-4 RBI 點壓制，總分若高需靠 Murakami 對 Castillo 單棒爆破而非 Mariners 打線連串。
  4. **Sox -3 連敗 streak** — G1 已輸 8-12，連敗心理壓力 + Kay 結構衰退，總得分上限預期不會比 Mariners 高，但 Kay 撐越久勝差越小。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組