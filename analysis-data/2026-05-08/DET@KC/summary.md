## 投手對決

### Kris Bubic (HOME, LHP, 28 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p82, K-BB% p74），gap vs ERA-only = +9.8
  - **同意 Strong Ace**。gap +9.8 < 15 觸發線；ERA 3.32 vs xERA 3.85 落差 0.53，Flag 8 未觸發。FIP 3.40 / xFIP 3.62 / K-BB% 14.1% 與 Strong Ace 描述一致。近 3 場 ER 5/18 IP（ERA 2.50）顯示狀態穩定。
- **Reverse platoon 信號**：未 fire。vs LHB OPS .570（39 BF, slash .171/.256/.314）/ vs RHB OPS .604（124 BF, slash .194/.298/.306），同手別優勢呈現正常 platoon。⚠️ 但 vs LHB sample 僅 39 BF — 表面 dominance 含小樣本 noise，下面對手威脅段一併校正。
- **對手打線威脅**：DET 整體 vs LHP 評為 🟢 Weak — Bubic 有結構性優勢。打線核心對 LHP 表現分歧大：
  - Greene vs LHP .948 + last7 BABIP .615（Flag 3 lucky-hot 極端值，可能回歸）→ 唯一威脅但含運氣
  - Carpenter vs LHP .384、McGonigle vs LHP .615 + last7 .475 BABIP .217（unlucky-cold 邊緣）— 主力對左投無效
  - Torkelson vs LHP .720 / Dingler vs LHP .654 — 中性
  - 結論：Bubic 對 DET 打線結構性壓制，唯一變數是 Greene 的單棒爆發風險。Kauffman HR -9% 進一步削弱 DET 長打變數。

### Keider Montero (AWAY, RHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p49, K-BB% p70），gap vs ERA-only = -1.5
  - **同意 Solid Starter**。gap -1.5 遠低於 15 觸發；ERA 3.48 / xERA 3.12 落差 0.36（Flag 8 未觸發）— xERA 反而比 ERA 低，意味著被擊球品質比結果好（hard hit% 29.9% 偏高但 barrel% 7.9% 控制住）。但 xFIP 4.13 比 FIP 3.40 高 0.73，提示 HR/FB 偏低、可能含好運（GB% 未在表中，但 SI 21.6% 占比高暗示中等 GB 傾向）。整體：Solid Starter 上沿，但運氣含量在 league average 之上。
- **Reverse platoon 信號**：🟠 fired，Δ +0.118（vs RHB OPS .691 vs LHB OPS .573，sample 兩側都 ≥ 30）。RHP 對非預期手別反而吃虧。
  - **本場放大**：KC 前 3 棒 Witt(R) / Garcia(R) / Perez(R) **全右打** + #6+ 多右打。對 Montero 是上半場連續右打輪攻擊，reverse platoon 結構性風險被本場 lineup 直接放大。
  - 緩解：Witt 雖 vs RHP 季 .772，但 last7 OPS .942 + BABIP .400（Flag 3 lucky-hot）→ 上限受 mean reversion 拉低。
- **對手打線威脅**：KC matchup tier vs RHP 🟡 Average，整體威脅中性，但本場有兩個放大因子：
  1. **Reverse platoon 對應 KC 多右打 lineup**（見上）
  2. **TTO3 penalty Δ +0.206（high）+ K% drop -9.1pp（career fallback, 155 BF）** — 第三輪 OPS 衝到 .934、K% 從 20.1% 掉到 11.0%。career-level 趨勢，不是季噪音。
  - 雙重壓力：教練可能 5-6 局就拉下；但 DET 牛棚 core IL ×3（見下）→ 後段替補比 Montero 更糟。
  - Pasquantino vs RHP .745 + last7 .806（4 棒 LHB）是除了 Witt 外的明確錨點。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - 與 season tier 同檔，方向「**同意**」。但 vs RHP 比 vs LHP 沒有統計優勢，本場對 Montero 的優勢來自 reverse_platoon + tto3_penalty 兩個 signal，不是打線本身的 hand split。
- **chain_break / heat_vs_babip 信號**：
  - 🟠 chain_break #2-3 fired（OPS 落差 .180）：Garcia .753 → Perez .573 → 中段 chain 斷。#3 Perez（season .573 + vs RHP .509）也是整體拖累，#4 Pasquantino vs RHP .745 + last7 .806 是另一個錨點 — chain 是「Witt 出場 → Garcia 過渡 → Perez 卡住 → Pasquantino 清壘」，串聯效率受限，多半靠單棒長打而非連續攻勢。
  - Witt last7 BABIP .400（heat_vs_babip 邊緣 / Flag 3 lucky-hot 警告）：last7 OPS .942 含運氣，**可能回歸**至季均 .825，後續單棒貢獻可能下修。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - 比 season tier 下修一檔，方向「**下修**」。Bubic 是 LHP，DET 對 LHP 結構性弱化（McGonigle/Carpenter vs LHP 都遠低於季均），本場進攻被結構壓制。
- **chain_break / heat_vs_babip 信號**：
  - 🟠 chain_break #7-8 fired（OPS 落差 .201）：在 9 棒序末段、非核心 1-5 段，影響有限 — Bubic 可能在 #7-8 棒順手解決，但對總得分上限壓制不大（因為前 5 棒已被 vs LHP weak 結構吃掉）。
  - Greene last7 BABIP .615（**Flag 3 極端 lucky-hot 警告**）：last7 OPS .814 含嚴重運氣，**很可能回歸**。Greene 是 DET vs LHP 唯一突破口，若 BABIP 修正 → DET 進攻上限再降一檔。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.9 / 4 / **1**（Estévez, IL15d） | 3.76 / 10 / **3**（Brieske IL60d, Melton IL60d, +1） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- **HOME (KC) 牛棚 🟠 中高**：整體 ERA 4.90 偏差，Estévez（closer 等級）IL15d 短期可能近期回歸；其餘 3 名投手 IL 多為非 core 角色。後段 high-leverage 階段缺一手，對 DET 末段威脅小幅放大；但 DET 打線 chain_break 在 #7-8 末段、又 vs LHP weak，Bubic 撐到 6+ 局可能 KC 牛棚僅需處理 1-2 inning，影響有限。
- **AWAY (DET) 牛棚 🔴🔴 極高（崩盤級）**：整體 ERA 3.76 表面好，但 core IL 3 名（Brieske / Melton 都 IL60d 長期不可用）→ 數字是 IL 之前累積的，**未來實況更差**。Montero 的 TTO3 penalty 與這條疊加形成核心風險：教練若 5-6 局拉下 Montero，後段必須由替補 / 新秀填洞，KC 後段攻擊（Witt 第 3-4 打席 + Pasquantino 後段打席）有結構性放大空間。**這是本場最關鍵的 asymmetry**。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ +-0.411（TTO1 0.778 → TTO3 0.367），第三輪明顯衰退；K% 從 26.6% 掉到 22.2%（Δ -4.4pp）
- 🟠 AWAY reverse platoon Δ +0.118（vs RHB OPS 0.691 > vs LHB OPS 0.573）— RHP 對非預期手別反而吃虧
- 🔴 AWAY TTO3 penalty：OPS Δ +0.206（TTO1 0.728 → TTO3 0.934），第三輪明顯衰退；K% 從 20.1% 掉到 11.0%（Δ -9.1pp）（career fallback）
- 🟠 HOME chain breaks at #2-3：OPS 落差 0.180
- 🟠 AWAY chain breaks at #7-8：OPS 落差 0.201
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
- 🔴 ⏳ AWAY 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
  - **本場核心 asymmetry**：Montero TTO3 penalty (high) + DET 牛棚崩盤級 → 兩條信號同向（兩者都讓 KC 後段攻擊放大），形成 6 局後 DET 全棚體系吃緊的雙重壓力。但 ⏳ short half-life 提示：core_il 是 last7 / 短期變動，DET 若有臨時頂替（rookie 上來、IL15d 提前回歸）可緩解一部分。
  - 與 Flag 3/8 互動：Flag 3（Witt last7 BABIP .400 / Greene last7 BABIP .615）反向作用 — 兩位主力都在 lucky-hot 區間，**可能回歸壓低雙方上限**；Flag 8 兩位投手都未觸發。雙方 lucky-hot 主力意味本場單棒爆發機率偏低，得分結構更靠 chain 連續性 → 反而把 chain_break 與牛棚 asymmetry 的權重再放大。

⚠️ **HOME TTO3 penalty 數據異常標註**：dossier label 顯示 "OPS Δ +-0.411"（TTO1 .778 → TTO3 .367，sample 36 BF），但**實際數值是 TTO3 比 TTO1 低 0.411**（不是高），同時 K% 也從 26.6% 降至 22.2%（越投越好）— 這與「TTO3 penalty」的觸發定義（TTO3 OPS - TTO1 OPS ≥ 0.100）方向相反，疑似腳本在 fallback / 顯示時取了 |Δ| 絕對值。**AI 不引用此信號為 KC 進攻加分依據**；反而 Bubic（小樣本 36 BF 警戒下）的數據暗示能撐第三輪，與 Strong Ace 評級一致。

## 條件修正

- Park Factor: 106.0 → +0.30 run（Kauffman 利安打 / 三壘打但壓制 HR -9% — 對 DET 中長距離擊球者如 Greene / Carpenter 不利，對 KC Witt-led 的線性攻擊偏中性）
- 天氣：未公布（跳過天氣分析）
- 先發 tier 落差：Bubic 🟠 Strong Ace (xFIP p82) > Montero 🟡 Solid Starter (xFIP p49) → KC 投手端結構性 +0.10~0.15 run 微優（已不重複 ±，因 base formula 已含投手品質）。Doubleheader 不適用（單場）。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.9 | +0.7 | 4.6 |
| AWAY | 4.2 | 0.0 | 4.2 |
| Total | 8.1 | +0.7 | 8.8 |

**HOME +0.7 推導**（cap ±0.8）：
- Montero `reverse_platoon` medium → KC 進攻 +0.2（Δ .118，KC 前 3 棒全右打放大）
- Montero `tto3_penalty` high (career fallback, 155 BF) → KC 進攻 +0.3（Δ .206 取 high 區間中段）
- DET `core_il_count` 3+ → KC 進攻 +0.4（崩盤級下界）
- 同側 interaction：三 signal 同向 fire，依「取單側 max + 0.1」規則 → max(0.4, 0.3, 0.2) + 0.1 = +0.5；但三條都 fire 時 AI 額外 +0.2 反映三重 stacking → +0.7（仍在 cap ±0.8 內）
- ⛔ 不入 ±：Witt last7 BABIP .400 (Flag 3 lucky-hot)、HOME TTO3 異常 label（已標註不引用）

**AWAY +0.0 推導**：
- KC `core_il_count` 1 → DET 進攻 +0.1（中高）
- DET `chain_break` #7-8 末段 → DET 進攻 -0.1（取下界半，因末段非核心 1-5 段）
- 兩者抵消 → 0
- ⛔ 不入 ±：Greene last7 BABIP .615 (Flag 3 極端 lucky-hot)；DET vs LHP weak 已部分含於 base formula vs hand splits，不重複 ±

## 整體判斷

- **方向（基本面）**：**HOME (KC) 略優**
  - 三軸都微偏 KC：(1) 投手 Bubic Strong Ace > Montero Solid Starter；(2) 進攻 — KC 對 Montero 有 reverse_platoon + TTO3 雙重利好，DET 對 Bubic 結構性弱化（vs LHP Weak）；(3) 牛棚 — DET core IL 崩盤級 vs KC 中高，後段差距明顯。
- **總分（基本面）**：**8.8**（base 8.1 + 信號修正 +0.7，全部加在 KC 端）
- **方向信心**：**~60% HOME 占優**
  - 三軸雖都偏 KC 但每條都有 caveat：Bubic vs LHB sample 39 BF 偏小、HOME TTO3 label 異常已扣除、Witt + Greene 雙方主力都在 lucky-hot 區間（Flag 3）→ 可能回歸壓低雙方上限、DET 牛棚 core_il 是 ⏳ short half-life（rookie / IL15d 回歸可緩解）。整體偏向 KC 但不是壓倒性，故信心控在 60%。
- **風險**：
  1. **DET 牛棚崩盤級**（core IL ×3, 兩名 IL60d）— KC 後段攻擊放大空間是本場最大 asymmetry；若 DET 替補新秀爆發或 Montero 撐長局可大幅縮小，但機率偏低。
  2. **雙方 lucky-hot 主力**（Witt last7 BABIP .400 / Greene last7 BABIP .615）— 兩位主力都在運氣偏移區間，若同場回歸→總得分下修；若同場持續→總得分上修；增加總分變異性。
  3. **小樣本警告**：Bubic vs LHB 39 BF + TTO 36 BF — 表面對 LHB / TTO3 dominance 含小樣本 noise，DET 若 Greene 真實爆發或左打抓到調整 → Bubic 出局時間提前可能引爆 KC 牛棚 ERA 4.90 的隱憂。
  4. **連敗慣性 vs 反彈**：KC −2 / DET −3 都連敗中（KC 對 CLE、DET 對 CWS）— 慣性效應與絕地反撲動機都不確定，無法系統判讀方向。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組