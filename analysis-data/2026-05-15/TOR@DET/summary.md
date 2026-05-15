## 投手對決

### Brenan Hanifee (HOME, RHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 沒給定（GS 1 / 3.7 IP 樣本太薄）。原始 tier 🟠 Strong Ace 是 ERA 1.08 假象。FIP **4.06** / xFIP 3.90 / xERA 3.86 / K-BB% **9.1%** / whiff% 5.8 / SI 59.5%（single-pitch） — 數據面是 🟢 Back-end ~ 🟡 Solid（ERA 4.0 區間）。
  - **不同意 Strong Ace**：3.7 IP 是極端小樣本，Flag 8 era_xera_delta -2.78 是運氣加持。本場按 🟡 Solid（4.0+ ERA 區間）對待。
- **Single-pitch dependent（🟠）**：SI 59.5% — 對打者第二輪後可預測，TOR 中段可能適應。
- **對手打線威脅**：🟢 低。TOR matchup tier 🟢 Weak (vs RHP) — Guerrero Jr. vs RHP .695 last7 .306（極冷 BABIP .087）/ Okamoto .784 / Varsho .746 — Guerrero Jr. 是 anchor 但 last7 嚴重冷期；整體威脅有限。

### Trey Yesavage (AWAY, RHP, 22 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 沒給定（GS 3 / 13.3 IP 樣本不足）。原始 tier 🟠 Strong Ace — ERA 0.68 / xERA 2.17 / FIP **1.98** / xFIP 3.63 / K-BB% **17.5%** / whiff% 14.0 / hard_hit% **10.8%**（極低）— 數據面真實 elite，barrel% 5.6% 接觸品質壓制好。
  - **同意按 🟠 Strong Ace 對待**：但 22 歲新秀 GS 3，本場按 🟠 Strong Ace 區間（ERA 2.5-3.0）較合理（FIP 1.98 不可持續）。
- **Single-pitch dependent（🟠）**：FF 48.6% — 邊緣門檻；FS 36.6% 補強雙球種威脅。
- **對手打線威脅**：🟡 中等。DET matchup tier 🟡 Average (vs RHP) — Greene vs RHP .858 last7 1.292（BABIP **.700** 火燙警訊）/ Dingler .886 / McGonigle .876 — 對 Yesavage 新秀小樣本，DET 可能適應後爆分。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak — 比 season tier 下修一檔；DET vs RHP 個別 Greene/Dingler 強但深度差。
- **chain_break 信號（🟠）**：#2-3 OPS 落差 0.216（Greene .916 → Torkelson .700）— 中度，但 Dingler #4 接得上，影響可控。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致，但 Guerrero Jr. last7 .306 冷期 + 整體 last7 BABIP .269 偏冷。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.08 / 10 / **3 名（🔴🔴 極高）** | 4.20 / 7 / **1 名（🟠 中高）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（DET）：ERA 4.08 中段但 **3 名核心 IL**（Brieske/Melton + 1）→ 🔴🔴 崩盤級。Hanifee 預期 4-5 IP 早下，DET 中繼會被 TOR 中心吃較多。
- AWAY 牛棚（TOR）：ERA 4.20 中段，1 名核心 IL（Yimi García）→ 🟠 中高。Yesavage 預期 5-6 IP，後段對 DET 中段（Greene/Dingler）仍有壓制。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-2.78):
  - **嚴重小樣本噪音**：Hanifee 3.7 IP 不足以建立 baseline，ERA 1.08 vs xERA 3.86 gap 主因樣本失真。本場對待按 🟡 Solid（4.0 ERA 區間），DET 失分基準應該偏高 — base 4.6（AWAY）合理偏低，可能往 5.0+ 走。

### 額外信號
- 🟠 HOME single-pitch dependent：Hanifee SI 59.5% — TOR 第二輪後可能適應。
- 🟠 AWAY single-pitch dependent：Yesavage FF 48.6% — 邊緣，FS 補強。
- 🟠 HOME chain breaks at #2-3：OPS 落差 0.216 — 中度。
- 🔴 HOME 牛棚 core IL ×3：🔴🔴 極高 — Hanifee 4 IP 後 DET 中繼崩盤，TOR 末段攻擊放大。
- 🟠 AWAY 牛棚 core IL ×1：🟠 中高 — TOR 後段仍 OK。

## 條件修正

- Park Factor: 106.0 → +0.30 run（Comerica 中性偏輕度打者友善，HR +5%）
- 天氣：未公布（跳過天氣分析）— 5 月中 Detroit 春末，溫度可能偏低
- 先發 tier：HOME Hanifee 真實 🟡 Solid（被 ERA 1.08 假象抬升 + 樣本太薄）vs AWAY Yesavage 真實 🟠 Strong Ace（樣本仍有限但數據面真實 elite）→ AWAY 投手戰有利
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 2.4 | 0（Yesavage Strong Ace 預期壓制 DET 弱攻擊；core IL 1 名未達門檻） | 2.4 |
| AWAY | 4.6 | +0.4（DET 牛棚 ×3 核心 IL 崩盤 + Hanifee 真實 🟡 Solid，TOR 末段攻擊放大） | 5.0 |
| Total | 7.0 | +0.4 | 7.4 |

## 整體判斷

- **方向（基本面）**：**AWAY (TOR)**。Yesavage 真實 Strong Ace（FIP 1.98 / hard_hit% 10.8% 接觸壓制）vs Hanifee 被 ERA 1.08 + 3.7 IP 小樣本掩蓋的真實 🟡 Solid；DET 牛棚 3 核心 IL 崩盤 + DET 主力 Guerrero Jr. 冷期 → TOR 投手戰 + 進攻雙優。
- **總分（基本面）**：**7.4 落點 6.5-8.5**。Yesavage 強壓制 DET 弱進攻 + Hanifee 雖然真實 Solid 但 DET 牛棚崩盤後段 → Total 中等略低。
- **方向信心**：**60-65%**（AWAY 有利）— Hanifee 樣本太薄是最大變數，但 DET 進攻面（攻↓ 連敗 3 + Greene BABIP .700 不可持續）+ 牛棚崩盤 + Yesavage 真實 elite 三重利空疊加。
- **風險**：
  1. Yesavage 22 歲新秀 GS 3 — 本場可能 5 IP 控不住，TOR 進攻仍可吃 Hanifee
  2. Greene last7 BABIP **.700** — 火燙不可持續，部分回歸但季度 OPS .916 仍真實威脅
  3. Hanifee SI 59.5% single-pitch — TOR 第二輪後可能適應，AWAY base 4.6 偏低
  4. DET 連敗 3 + 攻↓ 嚴重 — 心理面崩盤可能讓 base 2.4 (HOME) 更低

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
