## 投手對決

### Andre Pallante (HOME, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p66, K-BB% p40），gap vs ERA-only = -0.1
  - 同意。ERA 3.73 / xERA 4.41 / FIP 4.44 / xFIP 3.90 四個指標一致落在 Solid 區間，gap 僅 -0.1 屬正常噪音範圍。GB% 60.9 是真實壓制工具，配合 Busch HR -13% 是好對位。樣本 31.3 IP / 6 GS 已過小樣本門檻。
- **Reverse platoon 信號**：未 fired
  - Pallante vs LHB OPS .725 / vs RHB .626 — 標準 RHP platoon 走向，無異常。
- **對手打線威脅**：⚠️ MIL top 4（Mitchell-Chourio-Turang-Contreras）有真實威脅，但下半段 5-9 棒（Bauers/Vaughn/Frelick/Hamilton/Ortiz）OPS 全在 .759 以下，串聯性弱。最大警報是 **Turang vs RHP 1.055 OPS（last7 1.185）+ Contreras last7 .988**。Slider RV/100 +3.2 是制勝武器（whiff 40.6%），可用來壓 Turang 這種高接觸 LHB；但 SI / FF 兩顆 fastball xwOBA 都在 .421 / .461，被打偏重。Pallante 近 3 場最多投 5 IP（pitch limit / 早季控制），TTO3 OPS .815 雖未觸發 tto3_penalty signal（Δ +0.038），但實質第三輪會落在牛棚交接點。

### Brandon Sproat (AWAY, RHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 ERA-only tier = ⚪ Below Average；tier_v2 (xFIP-blend) 因 IP 26.67 < 30 未產出
  - **不同意 ERA-only 表面**。ERA 6.75 / xERA 5.27 / FIP 6.32 / xFIP 4.18 — 四指標分歧巨大。HR/9 2.36 是極端值（聯盟平均 ~1.2），HR/FB 大概率回歸帶動 ERA 收斂；K-BB% 8.7 與 Pallante 同分位，球速 92.1 也屬正常範圍。**結構性實力應落在 🟢 Back-end ~ 🟡 Solid 區間**，市場 ML pick'em 定價已反映這個判讀（不是純按 ERA）。Flag 8 紀律：不自動下修預測，但風險帶寬要拉開。
- **Reverse platoon 信號**：未 fired
  - vs LHB 1.035 / vs RHB .793 — 標準 RHP platoon（LHB 吃 RHP），無 reverse。
- **對手打線威脅**：🔴 STL 1-4 棒對位 Sproat 是這場最大尾風來源。Walker（vs RHP .959, last7 1.166, EV95% 59.1, Barrel 20.5）、Burleson（.929 / last7 1.049）、Wetherholt（.852）、Herrera（.868 / last7 .917）四人都對 RHP 有確切優勢；Sproat vs LHB 1.035 OPS 意味 Burleson/Gorman/Wetherholt 三位 LHB 會吃到擴大 platoon。BB% 13.0 加上 HR/9 2.36 兩個老問題都會被 STL 熱打線放大。近 3 場 IP 序列 3.0/3.7/3.7 → 預期 4 IP 以下，等於把比賽提早交給薄掉的牛棚（見下）。

## 打線評級

### HOME — season tier 🟡 Average / heat 🔥 Hot
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong
  - matchup tier 比 season tier 高一檔 → 認可上修。STL 全季打線受 9 棒 Scott II（.491）、6 棒 Winn（.729）拖低，但對 RHP 的 vs-RHP OPS 主要來自 Burleson .929、Walker .959、Herrera .868、Wetherholt .852 四人，本場是 STL 真正吃對位的場景。
- **chain_break / heat_vs_babip 信號**：HOME chain breaks at #4-5 (OPS Δ 0.298, 🟠 medium)
  - Walker（.962）→ Gorman（.664）的 0.3 OPS 跳水是真實串聯弱點，會壓制 5-7 棒的二次得分機會。Sproat 高 BB% 容易讓 1-4 棒上壘成串，但 Gorman/Winn 段難收尾 → 滿壘 / 一二壘有人時殺傷力受限。

### AWAY — season tier 🔴 Elite / heat 🔥 Hot
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🔴 Elite
  - **下修**：season tier 與 matchup tier 都被 **#2 Chourio 5 PA 樣本（OPS 2.500 / EV95% 100 / Barrel 50）嚴重通膨**，去掉 Chourio 真實打線是 🟠 Strong 偏 🟡 Average。Turang 1.055 vs RHP / Contreras .797 / Bauers .873 是真實威脅，但 6-9 棒 Vaughn .533 / Frelick .733 / Hamilton .557 / Ortiz .361 vs RHP 全在 average 以下。
- **chain_break / heat_vs_babip 信號**：
  - 🔴 AWAY chain breaks at #2-3 (OPS Δ 1.544) — **artifact**：Chourio 5 PA 通膨導致，**不可信**。真實的串聯斷點是 #3 Turang（.956）→ #5 Bauers（.759）以後直接摔到 #6-9 全部 OPS .596 以下。
  - 🟠 ⏳ AWAY lucky-hot：last7 BABIP 0.360（Turang .476 / Contreras .407 driving）— Flag 3 紀律不自動 ±run。Pallante 60.9% GB% + 滑球 whiff 40% 有壓制熱手回歸的工具（GB → 弱接觸 BABIP 自然回歸），對手有 1 天就能調整 mix 但 Pallante 已是 GB-heavy 不需大調整。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.92 / 1 / 0 核心 | 3.67 / 5 / 2 核心 (Zerpa + Koenig) |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 4.92 表面平庸但 0 核心 IL → 全陣容可用。Pallante 近 3 場僅投 5 IP，後段需要 4 IP 來自 bullpen，4.92 ERA 對 MIL top 4 仍有失分風險，但中性局面（不必處理高槓桿火災）。
- AWAY 牛棚：ERA 3.67 是表面強，但 **2 名核心 IL**（Zerpa + Koenig）= `matchup-factors.md` 🔴 高（吃緊）。Sproat 預期 4 IP 以下 → MIL 牛棚要承擔 5+ IP，等於對 STL 1-4 棒走完整輪 LHP/RHP 配對；正常情況本季可以靠 Zerpa（LHP）對 Burleson/Gorman 等 LHB 卡位，現在 LHP 高槓桿空缺由 B-tier 接，**這場最關鍵的 +run 槓桿**。對應 Table B core_il_count = 2 → STL 得分 +0.2~+0.5，本場條件取上界（早退 + 對手熱手 + LHB 槓桿空缺）→ 取 **+0.4**。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🟠 HOME chain breaks at #4-5：OPS 落差 0.298 — 真實，medium，壓 STL 連線得分上限
- 🔴 AWAY chain breaks at #2-3：OPS 落差 1.544 — **Chourio 5 PA 樣本 artifact，視為 noise，本場不入 ±run 修正**
- 🟠 ⏳ AWAY lucky-hot：last7 BABIP 0.360 — 熱手由 Turang/Contreras 拖（BABIP .476/.407 都顯著高於 career），Pallante GB-heavy 適合壓制，Flag 3 不入 ±run
- 🔴 ⏳ AWAY 牛棚 core IL ×2：Zerpa（LHP）+ Koenig 雙缺，對位 STL Burleson/Gorman 等 LHB 高槓桿失守 → 本場主要 +run 槓桿（已在牛棚段量化 +0.4 至 STL 得分）

## 條件修正

- Park Factor: 98.0 → -0.10 run（已含於 base）
- 天氣：Overcast, **129°F（API 異常值，不採用）**, wind 7 mph, In From RF
  - 影響判讀：溫度 129°F 為 API 資料異常（5 月 St. Louis 不可能達此溫度），不納入修正；wind 7 mph < 8 mph 屬噪音；In From RF 對 LHB 拉打方向（STL Burleson/Gorman 為 LHB）有極輕微抑制但不足以列為信號。整體天氣視為中性。
- 先發 tier / doubleheader：非 doubleheader；先發 tier 落差（Pallante 🟡 vs Sproat ⚪/真實 🟢-🟡 區間）已反映在 base formula。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 7.0 | +0.4 (MIL 核心 IL ×2) - 0.1 (STL chain break #4-5) = **+0.3** | **7.3** |
| AWAY | 6.0 | -0.0 (Chourio chain artifact 不採) - 0.0 (BABIP narrative only) = **0** | **6.0** |
| Total | 13.0 | +0.3 | **13.3** |

> ⚠️ 絕對得分數值較高（formula 校準偏寬），市場總分線 8.0 與 formula 基準不同尺度；**側向（HOME +1.3）的方向訊號比絕對總分可信**。

## 整體判斷

- **方向（基本面）**：HOME（STL）
- **總分（基本面）**：13.3（formula 尺度；市場校準 → 偏多側但量級不可直接套用）
- **方向信心**：60%
- **風險**：
  1. Sproat ERA 6.75 表面值與 xFIP 4.18 結構值落差大，今日如果 HR/FB 回歸（這場 Busch HR -13% 也在幫他）有可能投出表面意外的 4-5 IP / 2-3 ER 內容，會把 STL 得分壓到 base 7.0 以下
  2. Chourio 樣本通膨導致 MIL 打線 tier 評估有偏高風險，**但若他 5 PA 是真實能力起步**（季初新人 callup），AWAY 6.0 base 可能反而被低估
  3. Pallante 近 3 場固定 5 IP 上限 + STL 牛棚 ERA 4.92 → 第 6 局後 MIL 熱手 Turang/Contreras 對 B-tier RP 有反撲空間
  4. 單場樣本噪音：兩位先發合計 58 IP（雙位數樣本起步），本場 evidence weight 偏弱，不宜押超過 60% 信心

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
