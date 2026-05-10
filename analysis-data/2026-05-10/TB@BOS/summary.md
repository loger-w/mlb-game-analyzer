## 投手對決

### Payton Tolle (HOME, LHP, 23 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 沒給定（樣本 GS 3 / 17.7 IP 不足）。原始 tier 🔴 Elite Ace — ERA 2.04 / xERA 2.03 / FIP 2.25 / xFIP 2.99 / K-BB% **25.3%** / WHIP 0.74 / velo 92.7 — 各項都頂級。
  - **同意 🔴 Elite Ace（小樣本警告）**：所有指標一致 elite 但僅 17.7 IP，本場按 Strong Ace ~ Elite Ace 對待較合理。FF 46.3% 主球種 RV +3.0（極正值）。
- **單一球種依賴（🟠 FF 46.3%）**：FF-heavy LHP 但 FF run value 極高，TB RHB 多打線可 sit fastball 但 Tolle 球質夠強。
- **vs RHB 嚴重壓制（.070/.184/.093，49 BF）**：對 RHB 接近完美壓制，TB 多 RHB 中段（Caminero / Williamson / Díaz / Mullins / DeLuca）。
- **對手打線威脅**：🟢 低。TB matchup tier 🟢 Weak (vs LHP) — Caminero vs LHP .739 / Díaz .866 是兩個攻擊點，其他人 vs LHP 弱。

### Nick Martinez (AWAY, RHP, 35 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p40, K-BB% p54），gap vs ERA-only = -43.2
  - **同意 + 強調 ERA 嚴重高估真實水平**：ERA 1.71 看起來像 🔴 Elite Ace 但 xERA **3.90** / FIP 3.41 / xFIP 4.28 / K-BB% **10.7%** / whiff% 7.3% / velo 86.7（極低）— 真實水平 🟢 Back-end ~ 🟡 Solid（4.0 ERA 區間）。35 歲明顯退化。-43.2 gap 是嚴重運氣加持。
- **TTO3 penalty（🟠 -0.049）**：表面看反向（TTO3 OPS 反而高），但 K% 從 19.2% 掉到 8.9% (-10.3pp) 顯示真實壓制力第三輪嚴重崩盤。BOS 第三輪反彈關鍵。
- **對手打線威脅**：🟡 中等。BOS matchup tier 🟢 Weak (vs RHP) — Contreras vs RHP .755 / Abreu **.851** last7 1.068 / Yoshida .740 — Abreu 是真實爆分點，Story / Mayer / Durbin 黑洞拖累。

## 打線評級

### HOME — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak — 與 season tier 一致；BOS 整體進攻面是聯盟下段。
- **chain_break 信號（🔴 #5-6）**：Story .514 → Gasper .000 — 嚴重斷層（Gasper 無 OPS 數據，可能首次 MLB）；BOS 攻勢密集前 3-4 棒 (Duran/Contreras/Abreu/Yoshida)，#5 之後完全熄火。

### AWAY — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟢 Weak — 與 season tier 一致；TB 對 LHP 整體弱。
- **chain_break 信號（🟠 #5-6）**：Vilade .662 → Mullins .430 — 中度斷層；TB 攻勢需 Caminero/Díaz 中心打通，後段難延續。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.42 / 6 / **1 名（Coulombe IL15d）** | 4.02 / 7 / **3 名（Uceta + M. Rodríguez 等高 leverage RP）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（BOS）：ERA 3.42 中段穩定，1 名核心 IL（Coulombe LHP setup）→ 🟠 中高，影響輕。Chapman (closer) 健康，後段對 TB 弱進攻仍 OK。
- AWAY 牛棚（TB）：ERA 4.02 中段 + **3 核心 IL → 🔴🔴 極高**（牛棚崩盤級），多名 high-leverage RP 缺陣。若 Martinez 早下（35 歲退化 + xERA 3.90），TB 後段對 BOS Abreu 等中心可能爆分。

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-2.19):
  - **嚴重運氣加持**：Martinez ERA 1.71 vs xERA 3.90 是明確 -2.19 gap；K-BB% 10.7% / whiff% 7.3% / velo 86.7 結構性都顯示真實水平偏弱。本場按 ERA 4.0 區間對待。**不自動下修**，敘事上 BOS 失分基準偏低（base 3.7 偏低，實際可能 4.5+）。

### 額外信號
- 🟠 HOME single-pitch dependent FF 46.3% — Tolle FF 球質強（RV +3.0），影響輕。
- 🟠 AWAY TTO3 penalty K% 崩跌 -10.3pp — BOS 第三輪重要反彈關鍵。
- 🔴 HOME chain break #5-6 落差 0.514 — BOS 後段熄火，限制總分上限。
- 🟠 AWAY chain break #5-6 — TB 後段斷層，影響輕（前段已弱）。
- 🟠 HOME 牛棚 1 核心 IL — 影響輕。
- 🔴 AWAY 牛棚 3 核心 IL — 🔴🔴 極高，BOS 末段攻擊放大關鍵。

## 條件修正

- Park Factor: 104.0 → +0.20 run（Fenway runs 104 但 HR -15%，利安打/二壘打）
- 天氣：Partly Cloudy 70°F, wind 9 mph **Out To CF** — 順風中外野推 HR + 雙打 → 中等推升得分
- 先發 tier：HOME Tolle 🔴 Elite Ace（小樣本）vs AWAY Martinez 真實 🟡 Solid（被 ERA 假象掩蓋）→ 嚴重不對稱
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.7 | +0.8（AWAY 牛棚 3 核心 IL → BOS 末段攻擊極大化，cap 上限） | 4.5 |
| AWAY | 2.3 | 0（核心 IL 1 名，未達 ≥2 門檻） | 2.3 |
| Total | 6.0 | +0.8 | 6.8 |

## 整體判斷

- **方向（基本面）**：**HOME (BOS) 中強有利**。Tolle elite vs Martinez 退化（xERA 3.90）+ TB 對 LHP 弱 + TB 牛棚 3 核心 IL → BOS 投手戰 + 進攻雙優。雖然 BOS 進攻面整體弱（chain break #5-6 後熄火），但前 4 棒對 Martinez 退化 + TB 牛棚崩盤可吃。
- **總分（基本面）**：**6.8（base 6.0 + +0.8 信號）**，落點 6.0-8.0。雙弱打線 + Tolle 強壓制 + Martinez 真實水平不差 → Total 中性；TB 牛棚崩 + Out To CF 風 → 微上行。
- **方向信心**：~65%（HOME），結構性數據支撐（Martinez ERA 1.71 不可持續 + TB 牛棚崩盤）。
- **風險**：
  1. **Tolle 樣本小（17.7 IP）**：可能本場反向回歸（被 TB Caminero/Díaz 打 4 IP 4R）— 但所有 underlying 數據都 elite，回歸機率低
  2. Martinez ERA 1.71 vs xERA 3.90 — 本場可能繼續運氣加持（5 IP 1R）也可能爆掉（4 IP 5R），但 xERA 3.90 是中性錨點
  3. BOS Abreu last7 1.068 + EV/Barrel 真實 — 本場可能單棒爆分 2-3 分推 Total 上行
  4. Fenway HR -15% + Out To CF 風 — 風 vs 球場效應抵消，HR 預期中性

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
