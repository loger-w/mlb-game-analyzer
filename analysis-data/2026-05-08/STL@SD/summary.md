## 投手對決

### Griffin Canning (HOME, RHP, 29 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier = 🔴 Elite Ace（ERA 1.80 / K-BB% 21.0），但 tier_v2 未輸出（GS 1 樣本不足）。
  - **不同意 script tier**：GS = 1 完全不可作 tier 認定。Statcast peripheral 全面背離：xERA 4.09 / FIP 4.70 / barrel% 22.2（聯盟 RHP 平均 ~7-8%，**極端離群**）/ hard_hit 38.5（偏高）/ velo 90.9（不快）。真實水準合理推估 🟡 Solid Starter（ERA ~3.7-4.2）。ERA 1.80 主要是 1 場 5 IP 1 ER 的隨機運氣 + 樣本噪音，**不下修預測但需在風險段標明**。
- **Reverse platoon 信號**：未 fired（vs LHB 15 BF / vs RHB 4 BF 兩側都太少，無法判讀；vs RHB .250/.250/.250 4 BF 完全是噪音）。
- **對手打線威脅**：Cardinals 打線季級 OPS .730 / xwOBA .328（vs RHP Average tier），但 top 5 中 Walker last7 OPS 1.357、Burleson .923 兩人火燙；Canning 球種 CH 34.2% + FF 32.9% + SL 24.7% 三球分布，barrel% 22.2 顯示被擊球品質危險，遇到 hot middle order 容易爆。**威脅評估：高**。

### Michael McGreevy (AWAY, RHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p70, K-BB% p55），gap vs ERA-only = -18.1（tier_mismatch fired，high）。
  - **接受 tier_v2 結論**：真實是 Strong Ace 等級而非 ERA 帳面 Elite。然而即便 Strong Ace，xERA 5.78（!）已顯著高於 ERA 2.52，gap -3.26 屬於 **Flag 8 紀律下的高量級警告**：軟投（velo 86.5）+ 低 whiff（6.9%）+ 高 ground-ball play 模式仰賴 weak contact / 防守 / 對手手氣，被擊球品質在惡化（hard_hit 23.5 OK 但 BABIP 風險還在累積）。**運氣偏差為主、結構性風險為輔；不自動下修預測，但本場真實壓制力 ≈ 3.5-4 ERA 水準而非 2.52**。
- **Reverse platoon 信號**：未 fired（vs LHB .210/.248/.340 105 BF 與 vs RHB .179/.238/.359 42 BF，兩側差異 |Δ| < 0.080，正常 RHP 風格）。
- **對手打線威脅**：Padres vs RHP 季級 OPS .670 / xwOBA .313（Weak tier），last7 BABIP 0.224 整體 unlucky-cold。但 top 5 中 Bogaerts last7 OPS .946、Merrill .919 兩位實熱（BABIP .286 / .381 不算離譜）；Tatis、Machado、Laureano last7 都低迷且 BABIP 極端（.118 / .100），可能反彈。**威脅評估：中**（季級偏弱壓過 last7 反彈空間）。

## 打線評級

### HOME (Padres) — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak
  - matchup tier 比 season tier 低一檔 → 本場對打線評估**下修方向**。Padres vs RHP 是真實弱項（OPS .670 在 vs RHP 樣本中沒有特別優勢），不靠 BABIP 反彈幾乎難得分。
- **chain_break / heat_vs_babip 信號**：
  - 🔴 chain_break #7-8 OPS Δ 0.311（high severity）→ 後段 7-8 棒幾乎無生產，1-6 棒被處理掉就難再串。對 McGreevy 這種 GB / weak contact 投手，整局可能接近 1-2-3 出局頻繁。
  - last7 BABIP 0.224（Flag 3）由風險段處理。

### AWAY (Cardinals) — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - matchup tier 與 season tier 一致 → **同意評估**。Cardinals 季級 OPS .730 + chain OBP top3 .369 是健康水準，vs RHP 沒有 platoon 劣勢。
- **chain_break / heat_vs_babip 信號**：
  - 🔴 chain_break #4-5 OPS Δ 0.304（high severity）→ Walker (#4 OPS .971) 與 Gorman (#5 .667) 之間斷層大；如果 Walker 上壘無法靠 Gorman 推進，對中段串聯傷害最重。**但 #1-3 棒（Herrera .811 / Wetherholt .776 / Burleson .811）三人都健康**，前段串聯仍會給 McGreevy 壓力。

## 牛棚

| | HOME (Padres) | AWAY (Cardinals) |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.74 / 5 / 0 | 4.77 / 1 / 0 |

### 牛棚影響判讀
- **HOME (Padres) 牛棚**：ERA 3.74 + Core IL 0，後段是 Padres 的明確優勢。對 Cardinals top 3（Herrera/Wetherholt/Burleson）末段壓制力夠，可逐局拆掉 Cardinals chain。Petco 投手友善加成。
- **AWAY (Cardinals) 牛棚**：ERA 4.77 + Core IL 0，**比對手差約 1.0 ERA**。如果 McGreevy 6-7 局被換下，後段對 Padres 的 hot Bogaerts/Merrill 壓制力打折。雖無 IL 損耗，整體品質就是劣勢。**這是 Padres 後段反擊的關鍵窗口**。

## 風險提示

- ⚠️ **HOME 投手 Flag 8 (era_xera_delta=-2.29)**：
  - **以運氣 + 樣本噪音為主**。GS 1 帶 ERA 1.80 沒任何 predictive 意義，Statcast 周邊（barrel 22.2 / hard_hit 38.5 / xERA 4.09 / FIP 4.70）一致指向 Solid Starter 水準。**不自動下修預測**，但本場若 Canning 真實壓制力是 3.5-4 ERA 而非 1.80，AWAY 得分上修空間就在這裡。
- ⚠️ **AWAY 投手 Flag 8 (era_xera_delta=-3.26)**：
  - **運氣偏差為主、軟投風格 inherent**。McGreevy velo 86.5 / whiff 6.9% 仰賴 weak contact + 防守，xERA 5.78 是 BABIP 累積與 HR 風險的訊號。GS 7 樣本中等，**結構性風險明顯但不自動下修**。本場若 Padres top 5 BABIP 反彈（Tatis/Machado/Laureano 上 .250+ 區間），ERA 2.52 帳面會被打回原形。
- ⚠️ **HOME 打線 Flag 3 (last7 BABIP=0.224)**：
  - **可能反彈、不會大幅拉抬**。整體 BABIP 0.224 是運氣低端但 Padres season-level OPS .670 才是上限；個別球員（Machado .118 / Laureano .100）的 last7 BABIP 嚴重 unlucky 反彈空間最大，但即便回歸到 .280 區間，整體攻擊力仍受 vs RHP Weak tier 限制。**不自動 ±run value**。

### 額外信號
- 🟠 HOME TTO3 penalty（career fallback）：OPS 反向（TTO1 0.770 → TTO3 0.728）但 K% 從 23.4% drop 到 19.8%（Δ -3.6pp）。**不是典型 TTO3 衰退**——壓制力沒掉、只是三振能力第三輪降。對 Cardinals 4-5 棒 Walker（OPS .971 / 1.357 hot）影響：第三輪面對 Walker 缺 K 工具會放大被破壞風險。career fallback 信心低，量級給保守 +0.1。
- 🔴 HOME chain breaks at #7-8（OPS 落差 0.311，high）→ 信號量級 -0.2 ~ -0.3，取 -0.2。
- 🔴 AWAY chain breaks at #4-5（OPS 落差 0.304，high）→ 信號量級 -0.2 ~ -0.3，取 -0.2。
- **與 Flag 3/8 雙重壓力**：Padres 打線同時吃 chain_break 後段 + Flag 3 BABIP 反彈，淨效應大致打平（chain_break 壓制 vs BABIP 微反彈），summary 不雙重計算。

## 條件修正

- **Park Factor**: Petco Park 95.0（Runs 抑制 5%）→ -0.25 run / 場（base formula 已含）
  - HR PF +7%（HR 略加成），但 Runs PF 95 為主導。整體仍偏 pitcher's park。
- **天氣**：未公布（merged.weather = None / Petco 為室外但 API 未填）→ 跳過天氣分析。
- **先發 tier / doubleheader**：兩 SP 名義帳面（Canning 1.80 / McGreevy 2.52）vs 真實水準（peripheral）落差大；非 doubleheader（單場系列 G2）。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` Table B 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ Table A 不入欄：tier_mismatch（McGreevy）/ heat_vs_babip 衍生的 BABIP 反彈（Padres）/ strong_park（PF 已含於 base）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (Padres) | 4.2 | -0.2（chain_break #7-8 high） | 4.0 |
| AWAY (Cardinals) | 5.0 | -0.2（chain_break #4-5 high）+0.1（TTO3 penalty career fallback，弱信心） | 4.9 |
| Total | 9.2 | -0.3 | 8.9 |

## 整體判斷

- **方向（基本面）**：**AWAY (Cardinals)** 略優
  - 打線季級 OPS .730 > .670、last7 hot（Walker / Burleson 兩位 OPS > .920）、chain top 3 OBP .369 健康
  - McGreevy 季級 ERA 2.52 / WHIP 0.92 即便有 Flag 8 風險，仍勝過 Canning GS 1 樣本完全不可信的 1.80
  - 但 Padres 牛棚（3.74）+ Petco 後段是反擊窗口，差距不會拉太大
- **總分（基本面）**：**8.9**（base 9.2 - 0.3 chain_break 信號修正）
- **方向信心**：**58%**（介於 50-75%）
  - 不到 65% 的原因：兩 SP 都有 Flag 8 風險（特別 Canning GS 1 真實水準完全不確定），small-sample 噪音可雙向爆發
- **風險**（4 點）：
  1. **Canning GS 1 樣本歸零信任**——真實水準若是 Solid Starter（ERA ~3.8）AWAY 得分上修；若是真實 Elite（peripheral 翻盤）HOME 防守碾壓 → 雙向風險最大。
  2. **McGreevy 軟投 + xERA 5.78 結構性風險**——若 Padres top 5 BABIP 從 0.224 反彈至 .280+，帳面 2.52 ERA 會被破壞；本場最容易意外失分的劇本。
  3. **Padres 牛棚優勢 vs Cardinals 牛棚劣勢**——若雙方在 6-7 局戰平或 Cardinals 1-2 分小領先，Padres 牛棚（3.74 vs 4.77）有 ~1.0 ERA 結構優勢翻盤窗口。
  4. **Chain breaks 雙邊 high**（HOME #7-8 / AWAY #4-5）——兩隊得分上限都被壓制，total 不易爆衝；本場結構偏向 small-ball / 低分對決。
