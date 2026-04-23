# Phase 3 Summary — MIL @ MIA (2026-04-18)

## 先發投手

### Sandy Alcántara (MIA, 主) — 🔴 Elite Ace (new-version)

| 指標 | 2026 (3 GS, 24.33 IP) | 2025 (174.2 IP) | 判定 |
|------|-----------------------|-----------------|------|
| ERA | 0.74 | 5.36 | 表面 |
| xERA | 2.24 | 4.64 | **真實水平** |
| avg_velo | 91.8 | 92.3 | −0.5 mph 微退 |
| whiff% | 11.9 | 9.3 | +2.6 改善 |
| csw% | 28.9 | 25.8 | +3.1 改善 |
| barrel% | 4.6 | 8.6 | **−4.0 顯著改善** |
| ev95% | 37.9 | 45.0 | **−7.1 顯著改善** |
| hard_hit% | 27.6 | 27.8 | 持平 |

**YoY 結論**：whiff+csw+barrel+ev95+xERA **五項一致改善** → 判定 **new-version**。配球重組（FC 使用率降 8%，FF 升 4%）。球速微降但接觸品質壓制大幅提升。真實水平 xERA 2.24 = Elite Ace 級別。ERA 0.74 是運氣/樣本噪音。

Platoon：vs L .184/.259/.306（54 BF），vs R .183/.210/.250（62 BF）— 左右手通殺。

### Brandon Woodruff (MIL, 客) — 🟠 Strong Ace (退化版)

| 指標 | 2026 (3 GS, 16.67 IP) | 2025 (64.2 IP) | 判定 |
|------|-----------------------|----------------|------|
| ERA | 4.32 | 3.20 | 表面 |
| xERA | 2.82 | 2.22 | **真實水平** |
| avg_velo | 88.9 | 90.1 | **−1.2 mph 顯著退** |
| whiff% | 10.2 | 12.2 | −2.0 退步 |
| csw% | 27.2 | 30.6 | −3.4 退步 |
| barrel% | 10.2 | 7.1 | +3.1 退步 |
| hard_hit% | 24.1 | 19.5 | +4.6 退步 |

**YoY 結論**：velo + whiff/csw + barrel/hardhit **四項一致退化** → 判定 **退化版**，但 xERA 2.82 仍屬 Strong Ace 區間（K/9 8.6、BB/9 1.6 控卻仍佳）。33 歲已進入退化期。

Platoon：vs L .229/.263/**.457**（38 BF，長打風險），vs R .250/.290/.429（31 BF）。

## 打線

### MIA (🟢 Weak, Normal)
OPS .719 / xwOBA .300 / BABIP **.304（正常，無需回歸修正）**/ K% 22.1。
傷兵重創：Stowers、Conine、Morel、Ruiz、Acosta 全缺陣（主力 OF + SS）。
Top 棒次：Edwards OPS .910 / Lopez .983 / Hicks .876 — 有零星亮點但 1-3 棒串聯能力受傷兵牽制。

### MIL (🟡 Average, 🥶 **Cold**)
OPS .685 / xwOBA .314 / BABIP **.306（正常，Cold 判定有效）**/ K% 21.3 / BB% 12.3（高）。
傷兵：Yelich、Chourio、Vaughn（主力打者三缺）。
Top 棒次：Turang OPS .941 / Contreras .868 / Mitchell .883 / Bauers .858 — 核心仍在，但整體節奏 Cold。

### BvP
雙方所有 BvP 樣本皆 < 15 PA → **⛔ PA-gate 禁用**。單日 Heriberto Hernández vs Woodruff（5 PA, 2 HR）樣本不可引用。

## 牛棚

| | ERA | 核心 IL | 影響 |
|--|-----|---------|------|
| MIA | 3.21（強） | 僅 Mazur/Henriquez 60-day（非核心） | 無修正 |
| MIL | 3.99（均） | Yoho + Koenig（LH setup）+ Priester (15-day) | 2–3 核心 IL，**對手 +0.5~0.7，信號 +1，MIL ML −3~4%** |

## 條件修正

| 信號 | Run 修正 |
|------|---------|
| 雙方皆 🟠 Strong Ace+ | **−1.0（總分下修）** |
| MIL 牛棚 2+ 核心 IL | **+0.5（加到 MIA 得分）** |
| Park Factor 98（loanDepot 有頂） | −0.1（可忽略） |
| 前一場背靠背（MIL 7-5 贏 MIA） | 無特殊影響 |

**天氣**：loanDepot park 有可收放式頂蓋 → 中性，天氣變數忽略。
**主審**：未查詢（best-effort 跳過）。

## 修正後預期得分（formula 粗估）

League baseline 4.4 R/team。

- E[MIA R] = 4.4 × (.300/.315) × (2.82/4.20) × (98/100) ≈ **2.75**
  - + 牛棚 IL 信號 +0.5 → **3.25**
- E[MIL R] = 4.4 × (.314/.315) × (2.24/4.20) × (98/100) ≈ **2.29**
  - + MIL Cold 不扣（BABIP 正常不觸發回歸，但近期狀態冷靜下來） → **2.29**

**修正後總分 ≈ 5.54**（已含 −1.0 雙 Ace 效應已隱含在 xERA 低數值中，不重複扣除）。

若未隱含雙 Ace 修正則再 −1.0 → 約 4.54；以保守中位估 **5.0 ± 0.5**。

## 整體判斷

- **方向**：基本面傾向 **MIA 微優勢**（主場 + 投手 new-version Alcantara 略勝退化版 Woodruff 0.58 xERA + MIL 牛棚受創 + MIL 打線 Cold 連敗）。
- **總分**：基本面**強烈傾向低分**（雙 Ace 壓制 + 雙方打線皆有傷 + Park 微利投手 + 有頂無天氣助攻）。
- **信心**：MEDIUM（雙方先發皆僅 3 GS，樣本薄；開季第三週數據仍在建構）。
- **風險**：
  - Alcantara 0.74 ERA 是運氣成分，真實 xERA 2.24 仍可能被 MIL 上半打線（Turang/Contreras/Mitchell）敲出 2-3 分。
  - Woodruff vs L SLG .457 有長打風險，但 MIA 打線 Weak 難以利用。
  - 雙方打線同時 Cold/Weak → 低分保險度高。

## 盤口交叉驗證提示

**Pinnacle (rec-time snapshot, 2026-04-18 12:00 ET, ~4h before first pitch)**：
- ML: MIA 1.93 (51.8%), MIL 2.00 (50.0%) — 近乎 pick'em，MIA 微favorite
- O/U: 7.5 @ ~−109/−108（Over/Under）
- RL: **MIA −1.5 @ 2.95（33.9%）**/ MIL +1.5 @ 1.45（69%）— Pinnacle 把 MIA 當 fav

**使用者盤口**：
- 讓分：MIL −1.95（讓分方）@ HK 0.950，MIA +1.95（受讓）@ HK 0.950
- 大小：7.85 @ HK 0.940/0.940

⚠️ **方向矛盾警訊**：Pinnacle 把 MIA 當 favorite（giving 1.5），使用者盤口把 MIL 當 heavy fav（giving 1.95）— 方向完全相反。Phase 4 將以 Pinnacle 市場為 Kelly 計算基準，並為使用者盤口獨立評估 EV。
