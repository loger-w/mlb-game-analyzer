# Phase 3 Summary — TB @ CLE (2026-04-28)

## 投手對決

### Tanner Bibee (CLE, RHP, 27 ⚡)
- **真實水平**：🟢 Back-end Starter
- ERA 4.45 / xERA 4.64 / FIP 4.62 / xFIP 3.85（一致 back-end）
- velo 87.3 avg / max 96.1（中庸）、whiff% 12.1、hard_hit% 34.5、barrel% 8.5
- vs LHB 0.238/.314/.492（SLG 偏高，左打開砲風險）
- vs RHB 0.316/.361/.386（AVG 偏高但 SLG 較低）
- 球種 FF 27.8 / FC 27.5 / CH 17.9 / CU 13.7 / SI 13.2（平衡）
- 近 3 場：5 ER / 13.7 IP，每場短局數（耗球 78/74/87，未能完成 5 IP）

### Nick Martínez (TB, RHP, 35 📉📉)
- **真實水平**：🟢 Back-end Starter（腳本 tier 🟠 Strong Ace 為 ERA 假象，**已修正**）
- ERA 2.10 / xERA 4.64 / FIP 3.87 / xFIP 4.34
- 觸發 Flag 13：era_xera_delta = −2.54（≥ 1.5 閾值）
- velo 86.8 avg / max 94.4（慢）、whiff% 7.1（極低）、hard_hit% 25.8、barrel% 8.5
- vs LHB 0.239/.311/.388（中等）
- vs RHB 0.196/.208/.283（74 BF；BB% 2.1 異常低，回歸後預期變差）
- 球種 SI 31.3 / CH 27.1 / FC 18.8 / FF 10.9 / CU 7.5

## YoY 對比結論（B7 完成）

| 指標 | 2025 | 2026 | Δ | 結構性？ |
|------|------|------|----|--------|
| avg velocity | 86.9 | 86.8 | −0.1 | 持平 |
| max velocity | 95.1 | 94.4 | −0.7 | ⚠️ 微跌 |
| whiff% | 8.4 | 7.1 | −1.3 | ⚠️ 退化 |
| csw% | 25.9 | 26.2 | +0.3 | 持平 |
| hard_hit% | 23.6 | 25.8 | +2.2 | ⚠️ 變差 |
| barrel% | 6.9 | 8.5 | +1.6 | ⚠️ 變差 |
| EV ≥95% | 34.5 | 33.0 | −1.5 | 持平 |
| xERA | 4.04 | 4.64 | +0.60 | ⚠️ 變差 |
| GB% | 45.7 | 40.6 | −5.1 | ⚠️ 退化 |
| 主球種 | FC 21.0 | SI 31.3 | +14.1 | ⚠️ 配球轉換 |

**判定**：5 項 Statcast 全部沒有改善信號，多項一致退化（whiff↓ + hard_hit↑ + barrel↑ + GB%↓）。配球大幅轉向 SI（沉球）想拿滾地球，但 GB% 反而下降，策略沒奏效。**ERA 2.10 純粹是 5 場運氣樣本，必須按 xERA ~4.64 估算 CLE 預期得分**。Tier 從腳本給的 🟠 修正為 🟢 Back-end Starter（接近 average）。

## 打線評級

### CLE — 🟡 Average / ⚖️ Normal heat
- xwOBA 0.331、OPS 0.707、chain SLG mid 0.452
- 主力：JRam（vs LHP 1.016 / vs RHP 0.703）、DeLauter（RHP 0.752）、Schneemann（RHP 0.926；但 last7 BABIP .500 警訊）
- 黑洞：Kwan（RHP 0.652，且近 7 BABIP 0.150 極端低）、Bo Naylor（RHP 0.329，近 7 OPS .393）
- 對 RHP 中等，vs Martínez 沒有歷史 BvP 樣本（PA<15 全數退場）
- 近 7 BABIP 0.284 → 接近常態，數據可信

### TB — 🟢 Weak / ⚖️ Normal heat（但近 7 BABIP 觸發回歸）
- xwOBA 0.304、OPS 0.711、chain SLG mid 0.299（中段空洞）
- 主力（vs RHP）：Caminero 0.830、Yandy Díaz 0.921、Aranda 0.863
- 黑洞（vs RHP）：Mullins 0.451、Walls 0.545、DeLuca 0.592
- 兩極化打線：上半棒次有威脅，下半棒次崩盤

## BABIP 回歸判定（B10 完成）

- **TB last7 BABIP = 0.241**（≤ 0.260 觸發閾值）
- 個體層面：Caminero last7 BABIP .227（vs season .247）、Aranda .167（vs .235）、Mullins .125（vs .152）、Walls .167（vs .297） — 多名球員打點集中爆發但 BABIP 低
- 季 BABIP 0.288 已接近聯盟均值，僅近 7 偏低
- **判定**：last7 BABIP 0.241 是 7 天樣本噪音，回歸後預期 ~.288；**TB 不算 Cold，不扣 Cold run value**。Streak +5 連勝顯示打點轉換效率高（OPS 偏高但 BABIP 偏低 → 多打長球補償），但長球率回歸 + BABIP 微升將更穩定。

## 牛棚

| | CLE (主) | TB (客) |
|---|---------|---------|
| Bullpen ERA | 4.57 | 5.18 |
| IL 投手 | 2 (Walters 15-Day, Armstrong 15-Day) | 8 (Cleavinger, Boyle, Englert, Pepiot, Wilson, Uceta, Rodríguez, Grove) |
| 核心 IL 估計 | 1 名（Walters high-leverage setup） | 2+ 名（Cleavinger setup + Boyle 等高槓桿） |

### 牛棚雙向修正值

- **CLE 牛棚（1 核心 IL）**：對手 (TB) +0.3 run | TB ML +2%（CLE ML −2%）
- **TB 牛棚（2+ 核心 IL，且 ERA 5.18 已被 predict.py 自動 +0.5）**：在 +0.5 基礎上額外 +0.3 累計效應 = 對手 (CLE) 合計 +0.8 run | CLE ML +4%（TB ML −4%）

雙向皆計入，O/U +1.1 / ML 淨值 CLE +2%。

## 條件修正

- Park Factor 101 → (101−100)×0.05 = +0.05 run（含主客均分）
- Progressive Field 2024 改造後 LHB HR +16% — TB 左打：Aranda、Simpson、DeLuca、Mullins（4 人）；CLE 左打：Kwan、Naylor、Manzardo（3 人）。微弱 HR 風險上修，已包含在 Park Factor 內。
- 4 月底 Cleveland 17:10 ET 開賽：通常 60-65°F，無顯著修正
- 雙方先發未達 🟡 Solid+ → 不適用「-0.5/-1.0 投手戰」修正
- 無 Doubleheader / 無 Platoon 全打線劣勢 / 無休息日異常

## 修正後預期得分

| | base (formula) | + Martínez xERA 修正 | + 牛棚 IL 累計 | + Park | adjusted |
|---|--------------|-------------------|---------------|--------|----------|
| CLE 得分 | 4.4 | +0.6（Martínez 真實 xERA 4.64 vs ERA 2.10 假象） | +0.3（TB 牛棚 IL 額外 — 已含 0.5 自動） | +0.05 | **5.4** |
| TB 得分 | 4.8 | — | +0.3（CLE 牛棚 IL 額外） | +0.05 | **5.2** |
| Total | 9.2 | +0.6 | +0.6 | +0.10 | **10.5** |

> 註：模型自動加 +0.5（TB 牛棚 ERA ≥ 5.0）已在 base 之外，故修正後總分視為 9.2 + 0.6 (Martínez) + 0.3 (CLE 牛棚) + 0.3 (TB 額外牛棚) + 0.1 (Park) ≈ 10.5。

## 整體判斷

- **方向（基本面）**：CLE HOME 略佔優（log5 52%）。差距非常小（adj 比分差 0.2）。
- **總分（基本面）**：明顯偏 OVER（adj total 10.5 vs line 7.5，差距 +3.0，但需 predict.py 最終確認）。
- **信心**：MEDIUM
- **風險**：
  - Martínez ERA 假象修正幅度為估算（5 場樣本），實際 xERA 是否回歸有不確定性
  - TB 連勝 5 場可能有反彈動能，但 BABIP 偏低顯示已透支運氣
  - CLE 連敗 3 + Kwan/Naylor 個別冷檔（last7 OPS .380/.388）可能拖累
  - 5 場樣本（Martínez）vs 6 場（Bibee）— 雙方 SP 都尚未穩定
  - O/U 7.5 偏低，市場可能已部分反映 Martínez ERA 假象 → 實際盤口空間需 predict.py 驗證
