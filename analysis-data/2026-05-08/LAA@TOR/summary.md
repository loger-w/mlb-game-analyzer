## 投手對決

### Dylan Cease (HOME, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +15.3
  - 同意 v2 Elite Ace。ERA 3.05 vs xERA 3.03 對齊（**非** Flag 8）、FIP 1.93 比 ERA 還更低、hard_hit% 18.8 / barrel% 3.3 都聯盟頂段、K-BB% 22.9 + SL RV/100 +3.3 都 Elite 水準 → gap 是「ERA 還沒完全反映 K-BB% / soft contact 優勢」的結構性偏差，**不是運氣**。本場不下修。
- **Reverse platoon 信號**：未 fire（vs LHB OPS .626 / vs RHB .575，Δ 0.051 < 0.080 門檻）
  - 微 reverse 趨勢但不過門檻；TOR 主力 Vladdy / Okamoto / Clement 多為 RHB → 即使 reverse 也未放大為威脅。
- **對手打線威脅**：中等偏低。LAA matchup tier 🟡 Average vs RHP，但 chain_break #2-3（Trout → Adell, Δ 0.330）+ K% 26.7（高）對上 Cease whiff 15.8 / SL RV/100 +3.3 → 容易吃 K。Trout（vs RHP .971 / last7 .938 / EV95% 49.5 / Barrel% 24.2）是唯一明顯威脅，需小心一棒解決一局。

### Reid Detmers (AWAY, LHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +43.1
  - 結構面同意（FIP 2.55 / xERA 2.83 / K-BB% 20.8 / barrel% 6.3 都 Elite 區段），但 |ERA−xERA| = 1.45 接近 Flag 8 門檻（≥1.5）+ 近 3 場 ER 8/15.7 (4.59 ERA recent) → 結構性指標真但短期手感偏不利累積。**判讀運氣為主、結構為次**：保留 v2 不下修，但敘事上不全採上修；Flag 8 紀律下不自動 ±run。
- **Reverse platoon 信號**：未 fire（vs LHB .494 / vs RHB .734 是 LHP 正常 platoon）
  - 但這是大幅正向 platoon — vs RHB OPS 比 vs LHB 高 .240。TOR 預估打線 Vladdy / Okamoto / Clement / Schneider 多為 RHB（僅 Varsho LHB、Giménez switch）→ Detmers 對 RHB 弱點被放大，這是本場 TOR 的核心優勢點。
- **對手打線威脅**：中等偏上。TOR season tier 🟢 Weak vs LHP，**但 tier 是平均值掩蓋了上限**：Vladdy vs LHP .996（極佳）+ Okamoto vs LHP .790 / last7 OPS 1.364（極熱）→ 兩個高威脅點對應 Detmers RHB 弱點（OPS .734）。Clement / Giménez 把 tier 拉低但不會抵消 Vladdy/Okamoto 的單發威脅。

## 打線評級

### HOME — season tier 🟢 Weak / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - 與 season tier 一致 → 方向同意 Weak，但內部分布不平均：Vladdy (vs LHP .996) + Okamoto (vs LHP .790, last7 1.364) 是兩個 outlier 高威脅，Clement / Giménez (vs LHP .576 / .396) 把均值拉低。**本場實質威脅應視為 Weak 上緣 / 中等偏弱**，而非典型 Weak。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break 在 TOR 段未 fire → chain 連續性 OK；heat_vs_babip ⏳ unlucky-cold (BABIP .214) fire → last7 OPS 1-5 號平均約 .626 是真冷，但 .214 BABIP 是過度不利的運氣 → **可能反彈但帶 ⏳ 反身性**（Detmers 有 7 天時間調 mix），不過度押反彈。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - 與 season 一致，方向同意 Average。但 chain break 拉低實質得分能力 → 偏 Average 下緣。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #2-3 high (OPS Δ 0.330) → Trout (.971 vs RHP) 後立即接 Adell (.539 vs RHP / last7 .509)，**4 棒前的攻擊上限被切斷**：Trout 上壘但難得分，2 出局後容易回到弱棒重啟。heat_vs_babip 未 fire（BABIP 0.287 正常），手感數據可信。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.2 / 7 / 1（Yimi García, Setup/HL） | 5.3 / 5 / 0 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（TOR, ERA 4.20）：中等品質、有 Yimi García 60-day IL → 後段 high-leverage 變薄但仍可承擔。對 LAA 末段威脅：Trout 第 4 PA（TTO3 之外的 7-8 局段）若由替補 setup 處理是危險點，但整體 ERA 4.20 仍優於 LAA 牛棚 1.10 點，後段相對防守占優。
- AWAY 牛棚（LAA, ERA 5.30）：弱牛棚，雖無核心 IL 但 ERA 顯示輪換深度差 → high-leverage 與 mop-up 都可能丟分。Detmers 因 TTO3 Δ+0.750 極端會早下車（預估 5-6 IP）→ **LAA 弱牛棚被早暴露 6 局以上的負擔**，這是本場 TOR 後段攻擊的核心 edge。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.214):
  - 0.214 vs 聯盟均值 ~0.295 → 過度不利的運氣偏差，**結構性可能反彈**。但 ⏳ 半衰期短（7 天樣本 + 對手有時間調整），且 TOR 打線 vs LHP 本質就 Weak → 反彈期望保守。本場判讀：**不自動 ±run**（Table A 紀律），但敘事上接受「TOR 攻擊端有反彈空間，但 Detmers 是高品質 LHP，反彈 ceiling 受限」。

### 額外信號
- 🔴 HOME TTO3 penalty：OPS Δ +0.329（TTO1 0.504 → TTO3 0.833），第三輪明顯衰退；K% 從 37.5% 掉到 19.5%（Δ -18.0pp）
- 🔴 AWAY TTO3 penalty：OPS Δ +0.750（TTO1 0.379 → TTO3 1.129），第三輪明顯衰退；K% 從 29.2% 掉到 11.9%（Δ -17.3pp）
- 🔴 AWAY chain breaks at #2-3：OPS 落差 0.330
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - 受影響有限：LAA 打線 vs RHP 本質 Average + chain break #2-3 → 即使 TOR 後段防守變薄，LAA 串聯能力受限難爆量。**與 Flag 3 不同向**（Flag 3 是攻端可能反彈、core IL 是防端薄）→ 不疊加，僅各自折抵。

## 條件修正

- Park Factor: 99.0 → -0.05 run（中性，HR +4% 微利長打）
- 天氣：未公布（Rogers Centre 室內，可關閉屋頂 → 不分析）
- 先發 tier / doubleheader：兩 Elite Ace 對決 → 抑制總分上半場（1-5 局段）；但雙方 TTO3 都 fire（Cease Δ+0.329 / Detmers Δ+0.750）→ 6 局之後牛棚介入早，後半場被推回平均。Detmers Δ 極端 → 預估 5-6 IP；Cease 預估 6-7 IP。**非 doubleheader**。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 2.7 | +0.4（Detmers TTO3 Δ+0.750 high → +0.4 取上界 / 同向 LAA 牛棚 5.30 加 +0.0~+0.1，取單側 max + 0.0 保守） | 3.1 |
| AWAY | 2.0 | +0.1（Cease TTO3 Δ+0.329 medium → +0.2 / TOR core IL ×1 → +0.1，同側取 max + 0.1 = +0.3；扣 LAA chain_break high → −0.2，淨 +0.1） | 2.1 |
| Total | 4.7 | +0.5 | 5.2 |

## 整體判斷

- **方向（基本面）**：HOME（TOR）
- **總分（基本面）**：5.2（HOME 3.1 / AWAY 2.1）
- **方向信心**：60%（依據：Detmers TTO3 Δ+0.750 極端 + Detmers RHB platoon 弱點對上 Vladdy/Okamoto + LAA 弱牛棚 5.30 vs TOR 4.20；但 TOR 打線 vs LHP tier Weak + last7 cold 拖累信心 → 不到 75%）
- **風險**：
  1. **Detmers Flag 8 風險**：|ERA−xERA| = 1.45 接近觸發。若結構面（FIP 2.55 / xERA 2.83）才是真實水平，Detmers 可能壓制 TOR 全場（TOR 打線 vs LHP 本就 Weak） → 預測完全反向。
  2. **TOR Flag 3 持續冷打**：last7 BABIP 0.214 + heat Cold，若不反彈而持續冷期 → TOR 攻擊難破 3 分，總分掉到 4 以下，方向可能失靈。
  3. **Trout 一棒解決一局**：vs RHP .971 / last7 .938 / EV95% 49.5 / Barrel% 24.2 → 一發 HR 即繞過 chain_break 限制，把 LAA 拉回 +1。
  4. **TTO3 信號為估計（heuristic）**：Detmers 42 BF 樣本不算大；若教練不照預期早換、Detmers 撐到 7 IP → AWAY 牛棚弱點未暴露，total 與方向都受影響。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組