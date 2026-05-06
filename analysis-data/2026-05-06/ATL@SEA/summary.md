## 投手對決

### Bryan Woo (HOME, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p50, K-BB% p73），gap vs ERA-only = +24.3
  - 同意 v2 tier。Woo 的 ERA 4.61 被 .322 BABIP-against / 偏低 LOB% 拉高，但 xERA 4.06、FIP 4.03、K-BB% 13.9、WHIP 1.07 都站在中段先發，平均球速 92.4 mph 也健康，本場以 🟡 Solid Starter 對待，**不因表面 ERA 高而下修**
- **Reverse platoon 信號**：未 fire（vs LHB OPS .701 vs RHB OPS .678，Δ +0.023 < 0.080），不放大本場風險
- **對手打線威脅**：高。ATL 全季打線 🟠 Strong（vs RHP 🟠 Strong），Top 5 對 RHP 全在 .535-1.193，且 Olson last7 OPS 1.452 / Albies 1.183 / Baldwin .977 處於發燒等級。Woo single-pitch dependent (FF 48.3%) 加上 TTO3 penalty Δ +0.168（K% 17.8%→12.2%）→ 第三輪如直接吃 Olson / Albies 一輪，被打爆風險高

### Martín Pérez (AWAY, LHP, 35 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier (ERA-only) = 🟠 Strong Ace（ERA 2.22），但 v2 tier 未顯示；Flag 8 已標 ERA-xERA gap = -2.25
  - **不同意 ERA-only 的 Strong Ace**。xERA 4.47 / FIP 4.19 / xFIP 4.24 / K-BB% 8.1 全部指向中後段先發水準；ERA 2.22 是 BABIP-against / LOB% / sequencing 三方面同時走運的產物。本場以實質 🟢 Back-end / 弱 🟡 Solid 對待，**但不自動下修預測 run value**（Flag 8 紀律），改寫進風險段
- **Reverse platoon 信號**：fire（Δ +0.139，vs LHB .714 > vs RHB .575，兩側 BF 充足）。SEA top 5 中 J-Rod (R) / Aroz (R) 是核心 RHB → 對 Pérez 反而吃他平常壓制 RHB 的數據；但 L 棒 Naylor / Young 反而在 Pérez 的 reverse split 中佔便宜
- **對手打線威脅**：中。SEA matchup tier vs LHP 🟢 Weak（Naylor vs LHP .453、Raleigh vs LHP .501 都掉），Pérez 名義上有手別優勢；但 J-Rod vs LHP .924 / last7 1.022、Aroz vs LHP .864 兩個熱門 RHB 直接抵消優勢，加上 reverse platoon 讓 L 棒 Naylor last7 .850 也具威脅

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - Season 🟡 → vs LHP 🟢 是**下修一檔**：Naylor / Raleigh 對 LHP 是核心痛點，整體 SLG 預期偏低；但 J-Rod (.924) / Aroz (.864) 兩位 vs LHP 強勢者位於 1-2 棒，串聯起點仍可接受。本場攻擊「上限被壓、下限有保底」
- **chain_break / heat_vs_babip 信號**：HOME chain_break #6-7 OPS 落差 0.286（medium），#6-7 之後 chain 斷掉 → 二三巡之後得分串聯吃緊；Flag 3 last7 BABIP .256 雖未觸發 heat_vs_babip（heat = ⚖️ Normal），但 BABIP 可能向上回歸 → 本場攻擊有隱性 tailwind

### AWAY — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong
  - Season 🟠 = vs RHP 🟠，**評級維持**且 Top 5 last7 全部處於熱門段：Olson 1.452 / Albies 1.183 / Baldwin .977 三人連續發燒，對 Woo 的 RHP 又是擅長手別。本場攻擊上限非常高
- **chain_break / heat_vs_babip 信號**：AWAY chain_break #6-7 OPS 落差 0.342（high），#6 Dubón 之後接 #7 Pillar/Sanchez 等深度球員 chain 斷得更兇 → 1-5 棒火力集中、6-9 棒拖累。但因 1-5 太強，整體 chain_break 對 ATL 的壓制比一般情境溫和

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.14 / 5 / **3** (Carlos Vargas IL60d, Gabe Speier IL15d, +1) | 3.21 / 7 / **1** (Danny Young IL60d) |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：核心 IL ×3 對應 🔴🔴 **崩盤級** — Mariners 帳面 ERA 3.14 看似不錯，但是含 IL 中三位核心；今天 Woo 若如預期在第三輪走人（TTO3 嚴重），SEA 必須交給後段中繼，整段橋樑變薄。**ATL 後段 (7-9 局) 得分機率明顯放大**
- AWAY 牛棚：核心 IL ×1（Danny Young IL60d）對應 🟠 中高 — Braves 牛棚 ERA 3.21 與深度尚可；Pérez 屬「能撐 5-6 IP」但 xFIP 4.24 + 5 GS 樣本本季易失分，意味中段交棒。整體比 HOME 牛棚穩，**SEA 末段攻勢遇到的是相對完整的 setup → closer 鏈**

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-2.25):
  - **判讀運氣偏多**。ERA 2.22 vs xERA 4.47 的 -2.25 gap 屬極端區間，配合 K-BB% 8.1（聯盟均值約 13）、FIP 4.19、xFIP 4.24 顯示 35 歲 LHP 的 stuff 層面是中後段水準，2.22 ERA 的支撐主要來自 BABIP-against 偏低 + LOB% 偏高 + 軟弱接觸（hard_hit 26.8% 確實偏低，唯一具結構性的點）。**不自動下修 run prediction**，但本場若 SEA 把球打出去，Pérez 再走運空間不大，預期失分回歸 4.0+ 區間
- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.256):
  - **判讀偏向回歸**。.256 顯著低於聯盟基準（.290-.300）且 SEA Top 5 EV95% 多在 30-42（J-Rod 41.8 / Aroz 41.8 健康），擊球品質不差，BABIP 偏低多半是運氣 + T-Mobile 大外野壓打數。但**不自動 +run value**；本場仍在 T-Mobile Park（PF 82），即使 BABIP 回升，Park 抑制也會吃掉一部分

### 額外信號
- 🟠 HOME single-pitch dependent：主球種使用率 48.3%（≥45.0%）
- 🔴 HOME TTO3 penalty：OPS Δ +0.168（TTO1 0.663 → TTO3 0.831），第三輪明顯衰退；K% 從 17.8% 掉到 12.2%（Δ -5.6pp）
- 🟠 AWAY reverse platoon Δ +0.139（vs LHB OPS 0.714 > vs RHB OPS 0.575）— LHP 對非預期手別反而吃虧
- 🟠 AWAY TTO3 penalty：OPS Δ +0.145（TTO1 0.666 → TTO3 0.811），第三輪明顯衰退；K% 從 20.4% 掉到 15.2%（Δ -5.2pp）（career fallback）
- 🟠 HOME chain breaks at #6-7：OPS 落差 0.286
- 🔴 AWAY chain breaks at #6-7：OPS 落差 0.342
- 🔴 ⏳ HOME 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
- 🟠 ⏳ AWAY 牛棚 core IL ×1：🟠 中高（後段防守變薄）
- 🔴 投手友善球場 PF 82（≤90）
  - T-Mobile Park PF 82 + HR PF -18% 是本季倒數一二的 run-suppressor，**已包在 formula PF 倍率中**，不重複加 ±run。但其與 Flag 3 (SEA last7 BABIP .256) 雙重壓力下，SEA 打線即使回歸也難複製 5+ run 的爆發；本場總分上限被球場硬壓

## 條件修正

- Park Factor: 82.0 → -0.90 run
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：單場（doubleHeader=N），系列賽 ATL 1-0（5/5 G1 ATL 3-2 險勝）；ATL 連勝 +1 / SEA 連敗 -1。ATL 季戰績 26-11 / 近 10 7-3 RS 5.70；SEA 17-20 / 近 10 5-5 RS 3.80。投手對決名義 SEA 略佔（Woo > Pérez on peripherals），但牛棚與打線狀態雙雙 ATL 佔上風

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (SEA) | 3.9 | +0.1 | 4.0 |
| AWAY (ATL) | 3.9 | +0.5 | 4.4 |
| Total | 7.8 | +0.6 | 8.4 |

**HOME +0.1 拆解**（同側多 signal interaction，取 max + 0.1，再扣 chain_break）：
- max anchor: `reverse_platoon` (medium, +0.2) — Pérez 對 L 棒 OPS 反向偏高，Naylor / Young 受惠
- + 0.1 interaction（`tto3_penalty` AWAY career fallback + `core_il_count` AWAY ×1 同向）→ +0.3
- chain_break HOME (medium -0.2) → -0.2
- 淨 +0.1

**AWAY +0.5 拆解**（同側多 signal interaction，取 max + 0.1，再扣 chain_break）：
- max anchor: `core_il_count` HOME ×3（崩盤級 +0.4~+0.8 區間，取 +0.6）— SEA 牛棚 3 核心傷與 Woo TTO3 結合，後段必爆漏洞
- + 0.1 interaction（`tto3_penalty` HOME high + `pitch_mix_concentration` HOME single-pitch 同向）→ +0.7
- chain_break AWAY (high 但 1-5 棒太強，取 -0.2) → -0.2
- 淨 +0.5

## 整體判斷

- **方向（基本面）**：AWAY (ATL) 偏多
- **總分（基本面）**：8.4（formula 7.8 + 信號修正 +0.6）
- **方向信心**：62-65%（中等偏高）
  - 支持：ATL 1-3 棒 last7 全發燒（Olson 1.452 / Albies 1.183 / Baldwin .977）對 Woo single-pitch + TTO3 弱點；SEA 牛棚 3 核心 IL；Pérez ERA 雖花俏但 xERA 4.47 顯示真實水準允許 SEA 打到（雖然 SEA 整體 vs LHP 偏弱）
  - 折衷：T-Mobile Park PF 82 是強壓制器；Pérez 名義對 SEA 弱 vs LHP 打線確有手別優勢；Woo 26 yo 巔峰球速 92.4 / xERA 4.06 也比 ERA 4.61 真實面好不少 → 對 ATL 不是放鞭炮局
- **風險**：
  1. **Pérez 走運再延續一場**：xERA 4.47 是 season-to-date 平均，單場運氣可能再持續一場（hard_hit 26.8% 確實是結構性壓制）→ ATL 攻勢不一定如預期
  2. **Woo TTO3 樣本量 41 BF 不大**：本季 4 GS，第三輪數據有 sample noise；若教練早換投改成 TTO2 出場，penalty 信號失效
  3. **打線 projected 而非 official**：本場 16:10 開球，當下 12:45 ET 打序未出，若 ATL 主力如 Olson / Albies / Riley 任一輪休或 SEA 補進左投救援，本場讀法需重看
  4. **T-Mobile Park 對 ATL 飛球型打線壓制**：Olson / Riley / Baldwin 屬高 fly-ball 體型，T-Mobile HR PF -18% 可能把 1-2 個本壘打縮成飛球出局，總分上限被硬壓
