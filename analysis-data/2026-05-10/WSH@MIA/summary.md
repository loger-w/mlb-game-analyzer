## 投手對決

### Sandy Alcantara (HOME, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p47, K-BB% p37），gap vs ERA-only = -6.2
  - **同意**：ERA 4.01 / xERA 3.45 / FIP 3.76 / xFIP 4.16 / K-BB% **8.2%**（中等偏低）/ velo 92.0 — 各項一致 🟡 Solid。30 歲初期退化但本季數據撐住。
- **TTO3 penalty（🔴）**：OPS Δ **+0.293**（TTO1 0.510 → TTO3 0.803），K% 從 23.8% 掉到 14.1% — 嚴重第三輪衰退。
  - 本場 WSH 強打 Wood / Abrams 第三輪可能爆分；若主隊延後換投，Alcantara 5 IP 後危險。
- **對手打線威脅**：🟡 中等。WSH matchup tier 🟢 Weak (vs RHP) — Wood vs RHP **.977** + Abrams **1.084** + House last7 .882 — 前 4 棒真實威脅，後段 Vivas/Millas/Nuñez 黑洞。

### Cade Cavalli (AWAY, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p73），gap vs ERA-only = +32.3
  - **謹慎同意**：ERA 4.15 看起來中段，但 FIP **2.64** / xFIP 3.35 / xERA 3.79 / K-BB% 13.9% — FIP-base 真的 elite，ERA 高估主因 BABIP 偏高。但 WHIP **1.70** 是嚴重警訊，控球可能崩。
  - 本場按 🟠 Strong Ace 對待較合理（FIP 真實 elite 但 WHIP 顯示 inning-by-inning 不穩）。
- **vs LHB 嚴重弱點**：vs LHB **.380/.444/.506**（90 BF）— 真實結構問題，OPS .950 vs LHB。
  - MIA 中段 Stowers (LHB #4) / Marsee (LHB #5) / Mack (LHB #8 vs RHP **1.084**) — Mack 是真實 LHB 攻擊點。

## 打線評級

### HOME — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟡 Average — 比 season tier 上修一檔；前 3 棒 Edwards/Hicks/Lopez vs RHP 全 .830+ OPS。
- **chain_break 信號（🟠 #3-4）**：Lopez .908 → Stowers .663 — 中度斷層；前 3 棒密集後 Stowers 接不上但仍是 LHB 對 Cavalli 弱點的攻擊點。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak — 比 season tier 下修一檔，但 Wood + Abrams vs RHP 火力齊備。
- **chain_break 信號（🔴 #1-2）**：Wood .921 → García Jr. .616 — 嚴重斷層；Wood 之後 García/House 接不下，攻擊密度依賴 Abrams (#4)。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.44 / 3 / **2 名（Fairbanks closer + Henriquez setup）** | **4.76** / 7 / **2 名（Beeter + Kranick）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（MIA）：ERA 3.44 不錯但 **2 核心 IL → 🔴 高**，Fairbanks (closer) + Henriquez 缺陣 → 後段 leverage 被弱化，WSH Wood/Abrams 末段威脅放大。
- AWAY 牛棚（WSH）：ERA **4.76** 偏弱 + **2 核心 IL → 🔴 高**（Beeter + Kranick）→ 對 MIA 中段（Edwards/Hicks/Lopez）末段攻擊放大。雙方都崩。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- ℹ️ HOME balanced 4+ pitches：Alcantara 最高球種僅 22.9% — 球種多元，難被 sit fastball；對 WSH 強打前段是優勢。
- 🔴 HOME TTO3 penalty +0.293 — 嚴重，Alcantara 第三輪是 WSH 反彈關鍵。
- 🟠 AWAY TTO3 penalty -0.022（career fallback，本季樣本不足）— 影響輕。
- 🟠 HOME chain breaks #3-4 / 🔴 AWAY chain breaks #1-2 — 雙方攻擊密度都受限。
- 🔴 雙方牛棚 core IL ×2 — 雙向 🔴 高，總分上行壓力大；雙邊均後段易失分。

## 條件修正

- Park Factor: 106.0 → +0.30 run（loanDepot park runs 106 偏輕度打者友善但 HR -6%）
- 天氣：室內（Roof Closed，不適用）
- 先發 tier：HOME Alcantara 🟡 Solid vs AWAY Cavalli 🟠 Strong Ace（FIP 2.64 真實但 WHIP 1.70 不穩）→ 雙弱配對中等水平
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 2.8 | +0.5（AWAY 牛棚 2 核心 IL → MIA 末段攻擊 ↑） | 3.3 |
| AWAY | 4.3 | +0.5（HOME 牛棚 2 核心 IL → WSH 末段攻擊 ↑） | 4.8 |
| Total | 7.1 | +1.0 | 8.1 |

## 整體判斷

- **方向（基本面）**：**AWAY (WSH) 略有利**。Cavalli FIP 2.64 比 Alcantara 3.76 強壓制基礎；雖 Cavalli vs LHB 1.044 是真實弱點但 MIA LHB 火力分散（Mack 第 8 棒、Stowers OPS .720 vs RHP 平凡）。WSH Wood + Abrams 對 Alcantara TTO3 penalty 是真實爆分機會。
- **總分（基本面）**：**8.1（base 7.1 + 雙方牛棚 🔴 +1.0）**，落點 7.5-9.0。雙方牛棚都崩，Total 上行；Alcantara TTO3 衰退是上行助力。
- **方向信心**：~58%（AWAY），微偏有利但 Cavalli vs LHB 弱點 + WHIP 1.70 限制信心。
- **風險**：
  1. **Cavalli WHIP 1.70**：控球若崩，可能 4 IP 5R 早下；MIA 進攻面爆分機率上升
  2. **Alcantara TTO3 +0.293**：若 MIA 早換投則信號失效；若死撐 6+ IP 則 WSH 第三輪爆分
  3. WSH 連敗 1 + RA 4.60 近 10 / 守 weak — 信心面對立 MIA 主場連勝 1
  4. 雙方牛棚 🔴 高 — 後段 7-9 局可能成為總分主戰場

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
