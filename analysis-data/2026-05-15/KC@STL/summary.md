## 投手對決

### Dustin May (HOME, RHP, 28 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p78, K-BB% p60），gap vs ERA-only = +37.0
  - **不完全同意**：xFIP 3.71 / FIP 3.59 / K-BB% 11.8% 看起來 Solid 但 ERA **4.85** / xERA 4.64 / **近 3 場 ER 14 / IP 13.3**（區間 ERA 9.46）+ vs LHB **.343/.411/.505**（OPS .916 嚴重弱點）— 真實水平 🟢 Back-end ~ 🟡 Solid（ERA 4.5+ 區間），近期表現崩盤。
  - **本場按 🟢 Back-end** 對待（ERA 4.5-5.0 區間），May 復出後仍未恢復巔峰。
- **TTO 反向**：OPS Δ **-0.223**（TTO1 .875 → TTO3 .652）— TTO1 高（首輪被打）但 TTO3 反而 K% 提升，May 越投越穩；本場若撐住首 5-10 BF，後段可控。
- **vs LHB 嚴重弱點**：OPS .916（112 BF 真實樣本）— KC 中段 LHB 多（Pasquantino L / Jensen L）— Pasquantino vs RHP .757 / Jensen vs RHP .804 — 可吃。
- **對手打線威脅**：🟠 高。KC matchup tier 🟡 Average (vs RHP) — Witt vs RHP **.854** last7 **1.510**（BABIP .400 火燙）— Witt 是 May vs LHB-style 弱點以外的真威脅。

### Michael Wacha (AWAY, RHP, 34 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p49, K-BB% p64），gap vs ERA-only = -27.8
  - **同意 + ERA 高估**：ERA 2.63 看起來像 🟠 Strong 但 xERA **3.65** / FIP 3.72 / xFIP 4.13 / K-BB% 12.4% / velo 87.3（極低）/ vs LHB OPS **.559** 強壓制（樣本 125 BF 大）— 數據面真實 🟡 Solid（ERA 3.5-4.0 區間）。
  - **本場按 🟡 Solid Starter** 對待。
- **TTO3 penalty**：OPS Δ +0.089（K% Δ -14.4pp） — 中度衰退。
- **Reverse platoon（🟠 Δ +0.085）**：vs RHB OPS .644 > vs LHB .559 — RHB 表現微優於 LHB。STL 多 RHB 中段（Wetherholt/Herrera/Burleson RHB；Walker RHB）— 配對良好。
- **對手打線威脅**：🟠 高。STL matchup tier 🟡 Average (vs RHP) — Walker vs RHP **.927** + Burleson **.900** last7 .802 + Wetherholt .845 + Herrera .812 — 前 4 棒齊備。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致；Walker / Burleson 是 anchor，攻↓ 連 10 是隱憂（RS 3.30）。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average — 與 season tier 一致；Witt last7 1.510 是 anchor，但連敗 4 心理面壓力。
- **chain_break 信號（🟠）**：#4-5 OPS 落差 0.251 — 中度後段斷層。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | **4.70** / 1 / **0 名核心** | 4.64 / 5 / **1 名（Estévez closer IL15d，🟠 中高）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（STL）：ERA 4.70 偏弱但無核心 IL → 完整可用但整體中後段不穩。Helsley (closer) 健康。配合 May 真實 🟢 Back-end 早下，STL 中繼對 KC 中心 (Witt) 容錯低。
- AWAY 牛棚（KC）：ERA 4.64 偏弱 + Estévez (closer) IL15d → 1 核心 IL 🟠 中高。配合 Wacha TTO3 + 老投手限制，KC 後段對 STL 火力配對弱。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🟠 HOME TTO3 反向（-0.223）— May 越投越穩，但近 3 場 ER 14 / IP 13.3 顯示首輪壓制弱。
- 🟠 AWAY reverse platoon Δ +0.085 — Wacha 對 STL RHB-heavy 中段配對良好（Wacha vs RHB OPS 微弱）。
- 🟠 AWAY TTO3 penalty：OPS Δ +0.089 — Wacha 5-6 IP 後 STL 攻勢爆。
- 🟠 HOME chain breaks at #4-5：OPS 落差 0.251 — 中度。
- 🟠 AWAY 牛棚 core IL ×1：🟠 中高 — Estévez 缺陣 + ERA 4.64 後段對 STL Walker/Burleson 容錯低。

## 條件修正

- Park Factor: 98.0 → -0.10 run（Busch 中性偏輕度投手友善，HR -13% 壓 HR）
- 天氣：未公布（跳過天氣分析）
- 先發 tier：HOME May 🟢 Back-end vs AWAY Wacha 🟡 Solid → AWAY 投手戰略優
- doubleheader：無

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.0 | +0.3（KC 牛棚 ERA 4.64 + Estévez IL + Wacha TTO3 → STL 末段攻擊放大） | 4.3 |
| AWAY | 3.8 | +0.4（May 近期崩盤 ER 14/13.3 IP + vs LHB OPS .916 弱點 + Witt last7 1.510 火燙） | 4.2 |
| Total | 7.8 | +0.7 | 8.5 |

## 整體判斷

- **方向（基本面）**：**持平**（極微偏 HOME）。雙弱 starter 對峙（May 🟢 Back-end vs Wacha 🟡 Solid），雙方牛棚都不穩；STL 中心 (Walker/Burleson) 對 Wacha + KC 中心 (Witt) 對 May 兩端都有真實威脅。base 接近平手（4.0 / 3.8）。
- **總分（基本面）**：**8.5 接近實際，落點 7.5-9.5**。雙弱 starter + 雙方牛棚都偏弱 + Busch HR -13% 部分壓制 → Total 中等。
- **方向信心**：**55%**（HOME 微利）— STL 主場 + 連敗 4 vs KC 連敗 4 雙方狀況低迷 + May 近期崩盤但本場可能反彈、Witt 火燙但 KC 連敗心理面差。
- **風險**：
  1. May 近 3 場 ER 14 / IP 13.3 — 嚴重崩盤期，本場可能繼續被打或大幅反彈，最大變數
  2. Witt last7 OPS **1.510** + BABIP .400 — 火燙不可持續但 vs RHP 季度 .854 真實，本場仍危險
  3. KC 連敗 4 + STL 連敗 1 — 雙方狀況都低迷，雙方都有反彈空間
  4. STL 主場攻↓ 嚴重（近 10 RS 3.30 vs 季 4.53）— base 4.0 可能偏低 OR 反彈空間

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
