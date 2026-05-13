## 投手對決

### Erick Fedde (HOME, RHP, 33 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟢 Back-end Starter（xFIP p29, K-BB% p29），gap vs ERA-only = -23.7
  - 同意 Back-end Starter，甚至偏 Below Avg。gap -23.7 = ERA 3.79 高估真實水平（運氣假象）；K-BB% 7.1 + FIP 5.60 + vs RHB SLG .556 結構性差。ERA 將回升。不自動下修（已含於 Back-end），但敘事按 Below Avg。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - fired Δ +0.311（vs RHB OPS .864 > vs LHB OPS .553）— 巨型 reverse platoon！Fedde 對右打吃虧明顯。KC 打線多右打（Witt / Garcia / Perez / Caglianone / Massey / Isbel）— **KC 右打整體 hunting zone**，這是本場最強 edge。
- **對手打線威脅**：高。Witt .803 vs RHP / Caglianone .773 / Collins .821 / Massey .777 + Fedde vs RHB .556 SLG → KC 預期 5+ 分。

### Stephen Kolek (AWAY, RHP, 29 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = —（樣本 < 30 BF，未打分），gap vs ERA-only = —
  - 樣本只 1 GS，tier_v2 無法判斷。原始數據：ERA 4.50 / xERA 3.53 / FIP 4.27 / K-BB% 13.6（好）/ velo 90.4 avg / 95.5 max。看起來 Solid Starter 上限有希望，但 1 GS 樣本不可信賴。實質按 🟢 Back-end Starter 處理。
- **Reverse platoon 信號**：未 fired（樣本不足無法計算）。
  - n/a
- **對手打線威脅**：中。CWS season Strong / vs RHP Strong — Murakami .933 / Antonacci .886 / Vargas .691 / Montgomery .782 / Romo 1.096（last 7 1.107，但 EV95 僅 26.3）。Kolek 1 GS 樣本，球質球速看起來壓制 — 結果不可預期，distribution 寬。

## 打線評級

### HOME — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong
  - 同意。Murakami / Antonacci / Vargas / Montgomery 整支 vs RHP 強，但 Kolek 樣本不足無法精準對位 → 維持 Strong。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - 未 fired chain_break；heat_vs_babip last7 BABIP 0.328 normal。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average
  - 上修同意。對 Fedde reverse platoon Δ +0.311 + vs RHB SLG .556 → KC 右打對位優勢極大，本場評估上修至 🟠 Strong vs Fedde 特定 matchup。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - AWAY chain_break #2-3 gap 0.207 — Witt .803 → Pasquantino .744 vs RHP，落差不大；−0.1 run。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.5 / 4 / 1 | 4.79 / 5 / 1 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：CWS 4.50 ERA + 1 core IL（Vasil）— 中等深度但後段稍變薄。Fedde 可能 5 局後被換投，CWS 牛棚需擋 4 局。
- AWAY 牛棚：KC 4.79 ERA + 1 core IL（Estévez closer）— **Estévez 是 closer，IL 影響度升級**。9 局 close-out 能力下降。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🔴 HOME reverse platoon Δ +0.311（vs RHB OPS 0.864 > vs LHB OPS 0.553）— RHP 對非預期手別反而吃虧
- 🟠 HOME TTO3 penalty：OPS Δ +-0.013（TTO1 0.723 → TTO3 0.710），第三輪明顯衰退；K% 從 18.8% 掉到 15.6%（Δ -3.2pp）（career fallback）
- 🟠 AWAY TTO3 penalty：OPS Δ +-0.051（TTO1 0.698 → TTO3 0.647），第三輪明顯衰退；K% 從 18.6% 掉到 13.4%（Δ -5.2pp）（career fallback）
- 🟠 AWAY chain breaks at #2-3：OPS 落差 0.207
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
- 🟠 ⏳ AWAY 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - 雙方各 1 core IL 中高影響相抵；本場主訊號是 Fedde reverse platoon Δ +0.311（巨型）+ Estévez closer IL — 對 KC 右打有結構性 edge。

## 條件修正

- Park Factor: 97.0 → -0.15 run
- 天氣：Partly Cloudy, 76°F, wind 21 mph, R To L
  - 影響判讀：21mph 是強橫風（R→L），會影響擊球軌跡準確性與內外野判斷，但**對 HR 拉打方向影響有限**。整體增加亂場性質，distribution 變寬，無顯著方向 ±run（中性）。
- 先發 tier / doubleheader：Fedde Back-end（運氣假象）vs Kolek SSS — 雙方都不出色；但 Fedde reverse platoon + KC 多右打給 KC 結構性 edge。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.1 | +0.1（AWAY core IL Estévez closer +0.1） | 5.2 |
| AWAY | 5.8 | +0.3（reverse platoon Δ +0.311 → +0.3 + HOME core IL +0.1 互動 max+0.1 −0.1 chain AWAY） | 6.1 |
| Total | 10.9 | +0.4 | 11.3 |

## 整體判斷

- **方向（基本面）**：AWAY (KC 微傾)
- **總分（基本面）**：11.3（強 OVER 訊號）
- **方向信心**：58% — Fedde reverse platoon +0.311 + 多右打 KC 是結構性 edge，但 Fedde Flag 8 ERA -23.7 可能延續低 ERA（運氣再次站他這邊），且 Kolek 1 GS 樣本完全不可預期。
- **風險**：
  1. Kolek 1 GS 樣本 — distribution 極寬，任何結果都可能
  2. Fedde Flag 8 -23.7 — 可能單場再次運氣好，ERA 3.79 延續
  3. 21mph 強橫風影響擊球（亂場係數 ↑）
  4. KC 進攻火力中等（season Average），需要 Fedde 真的崩盤才能領先

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
