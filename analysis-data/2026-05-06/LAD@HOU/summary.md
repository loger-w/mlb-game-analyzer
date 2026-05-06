## 投手對決

### Lance McCullers Jr. (HOME, RHP, 32 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p76, K-BB% p61），gap vs ERA-only = +57.2
  - **不同意 v2 tier**。實際 ERA 6.32 / xERA 4.38 / FIP 4.28，三項都不在 Strong Ace 區間（Strong Ace 標準 ERA 2.50-3.20 + ERA+ 130-170）。v2 把他抬到 Strong Ace 是 xFIP 3.74（K%/BB%/GB% 估）拉高的結果，但 K-BB% 12.0 + WHIP 1.40 + whiff 9.5% + velo 87.1 mph 通通不像 Strong Ace。**判讀為 🟢 Back-end → 🟡 Solid Starter 之間**（依 xERA 4.38 取較合理錨）。Flag 8 走 AI 敘事，**不直接下修預測**，但本場敘事以 xERA 4.38 為功能層級的真實水準。
- **Reverse platoon 信號**：dossier 未 fire（無此標籤）。但實際拆分 vs LHB .246/.338/.477 (74 BF) > vs RHB .224/.356/.367 (59 BF) 在 SLG 上反向，左打更能打他長球（FC/SI 為主、ST 對右打才是強項）。對手道奇打線左打 dominant（Ohtani / Freeman / Tucker / Muncy / Rushing 全左打），**這個結構性對 McCullers 不利**。
- **對手打線威脅**：道奇 1-7 棒 vs RHP OPS 中位數 .756，Muncy .910 / Freeman .796 / Pages .844 / Rushing 1.225 都是中軸實質威脅。McCullers 主球種 FC RV/100 = -0.7（被打）、SI +0.9（中性）、ST +1.2（對左打效果有限），其餘配球零碎。**威脅評等：🔴 高**。

### Tyler Glasnow (AWAY, RHP, 32 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +3.1
  - **同意 Elite Ace**。ERA 2.56 / xERA 2.40 / FIP 3.02 / xFIP 2.79 一致；K-BB% 23.6 + WHIP 0.83 + velo 90/98.3 + barrel 7.1% + hard_hit 22.7% 全項頂級。Gap 僅 +3.1 屬統計噪音，無 Flag。
- **Reverse platoon 信號**：未 fire。vs LHB .146/.245/.280（94 BF）vs RHB .146/.180/.313（50 BF）— 雙側 sub-.350 OBP，無左右破口。
- **對手打線威脅**：太空人 1-7 棒 vs RHP OPS 中位數 .727；Alvarez 1.093 / Walker .995 是兩個真正威脅，其餘 Paredes / Smith / Cole / Matthews 都是中段 .682-.752 區間。Glasnow 球種組合 FF 30.5% / KC 23.2% / SI 22.2%，SI RV/100 = +3.8（對右打 sweet 武器），KC +0.7 對抗 Alvarez / Walker 兩位左打/右打皆有效。**威脅評等：🟡 中**（Alvarez 是唯一高槓桿風險點）。

## 打線評級

### HOME — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - matchup tier 比 season tier 降一檔，與 lineup 結構一致：太空人主力多右打（Walker/Paredes/Smith/Cole/Matthews），對 RHP 較吃虧；唯一左打巨炮 Alvarez 對 RHP 是真強，但 Glasnow vs LHB OPS only .525 — 連 Alvarez 都會被壓制。**本場打線評估：下修一檔，採 🟡 Average 為作業基準**。
- **chain_break / heat_vs_babip 信號**：🔴 chain breaks at #8-9 (落差 1.077)
  - #8 Shewmake 1.077 → #9 Salazar .000，但 Salazar 為 0 PA fallback（新人/大聯盟初登場）— **信號為資料假象，不真實納入**。1-7 棒實際 chain 完整，主要 break 在 #5 Smith (.717) → #6 Cole (.834) 反向，無壓制。

### AWAY — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong
  - 與 season tier 一致。道奇 1-9 vs RHP 中位數 .796 OPS，雖然個別人 vs RHP 略低於 vs LHP（Ohtani / Freeman 等），但整體仍維持 Strong 級別 — McCullers 不是壓制型 RHP，**評級維持 Strong**。
- **chain_break / heat_vs_babip 信號**：🔴 chain breaks at #7-8 (落差 0.385)
  - Rushing 1.225 (vs RHP) → Kim .830 → Freeland .728，落差中度但屬「上半棒打完強度緩慢遞減」常見型，#1-7 連續性完整，**對攻擊效率輕度負影響、但可忍受**。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 6.2 / 8 / 1（Hader, Closer，IL60d）| 3.69 / 9 / 1（Díaz, Closer，IL15d）|

### 牛棚影響判讀
- **HOME 牛棚**：ERA 6.20 屬全聯盟末段；Hader 60-day IL 等於 closer 缺席整段考察期，後段 leverage 高度壓力大。McCullers 又有 TTO3 OPS Δ +0.132（career fallback），第三輪明顯衰退 → 預期 5-6 局必須換投，落入 ERA 6.20 牛棚 + 缺 closer 的環境。**對道奇後段加分明顯（Table B `tto3_penalty` medium + `core_il_count` 1人 medium，同向取 max + 0.1，估 +0.3 ~ +0.4 run）**。
- **AWAY 牛棚**：ERA 3.69 中段水準；Díaz 15-day IL 是 closer 缺席但短期；Glasnow 自身 TTO 樣本太小（37 BF，confidence: heuristic）+ 平均能撐 6 局。預期道奇牛棚負擔輕，後段壓制力足以守住領先。**對太空人加分微（信號樣本不足 + Díaz 短期可回，估 +0.0 ~ +0.1 run）**。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=+1.94)：
  - **判讀：偏結構性，非運氣**。McCullers 32 歲、TJ 復出後第二段（過往多次傷病），velo 87.1 mph 屬聯盟下段（聯盟 RHP 平均 ~93），whiff 9.5% 也低於聯盟 ~25%。xERA 4.38 仍非 Strong Ace 水準，ERA 6.32 是「真實偏弱 + 運氣放大」雙重結果。**不自動下修預測（per Flag 8 紀律），但本場敘事採 xERA 4.38 為功能層級錨點**，已內含於上方 Tier 判讀。

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ +0.132（TTO1 0.732 → TTO3 0.864），career fallback；配合 HOME 牛棚 ERA 6.20 + Hader IL → 後段失分風險顯著放大（→ AWAY 加分區間取上界）。
- 🟠 AWAY TTO3 penalty：OPS Δ -0.135（TTO1 .503 → TTO3 .368）但 K% drop -10.5pp；37 BF 樣本太小（confidence: small_sample），且 OPS 反向（TTO3 反而更強），**判讀為訊號雜訊，不採納**。
- 🔴 HOME chain breaks at #8-9 (1.077)：Salazar 0 PA fallback，**資料假象不採納**。
- 🔴 AWAY chain breaks at #7-8 (0.385)：中度，#1-7 完整 → 對 chain 影響輕微，不調整。
- 🟠 ⏳ HOME 牛棚 core IL ×1 (Hader)：medium 1人區間 +0.0 ~ +0.2，與 TTO3 同向取交集，已合併於牛棚判讀。
- 🟠 ⏳ AWAY 牛棚 core IL ×1 (Díaz)：medium 但短期 IL15d 可回 + 牛棚整體 ERA 3.69 → 取下界 0.0。

## 條件修正

- Park Factor: 98.0 → -0.10 run（Daikin Park 中性偏投手，HR PF +2% 微利長球但屬噪音）
- 天氣：室內（Roof Closed，不適用）
- 先發 tier / doubleheader：兩名先發休息天數同為 GS 6（季中正常輪值），無影響；非 doubleheader。

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.5 | +0.0（Glasnow Elite + 太空人 vs RHP 弱，無 fired 加分信號）| 3.5 |
| AWAY | 5.1 | +0.3（HOME tto3_penalty + core_il_count 同向取 max）| 5.4 |
| Total | 8.6 | +0.3 | 8.9 |

## 整體判斷

- **方向（基本面）**：AWAY（道奇）
- **總分（基本面）**：8.9（base 8.6 + 0.3 牛棚/TTO3 信號）
- **方向信心**：65%（道奇贏面明顯但非壓倒性 — 主因 McCullers 真實能力反差導致下限變寬，加上系列 G2 太空人主場心理因素）
- **風險**：
  1. McCullers tier 反差（v2 Strong vs ERA 6.32）— 若回歸 xERA 4.38 真實水準，太空人主投不會崩太大，HOME 預期 3.5 可能高估道奇打線壓制 → 實際 HOME 得分可能更低（壓 Total 但仍利道奇贏球）
  2. 道奇 last7 攻擊偏冷（RS 3.50 vs season 5.37），Alvarez 在太空人陣中也 last7 OPS 僅 .591 — 雙方主力打者都偏冷，Total 上限不易撐起
  3. Glasnow 近 3 場 ER/IP = 8/18 (4.00 ERA) 比 season 2.56 略高，可能進入小幅疲勞期 — 雖仍 Elite tier，但「壓制 0-1 分」這種結果機率降
  4. 系列脈絡：G1 太空人 2-1 險勝，HOME 主場 G2 心理續航 + Glasnow 對太空人歷史對戰（dossier 未提供，需注意是否有 BvP ≥ 15 PA 樣本未 surface）
