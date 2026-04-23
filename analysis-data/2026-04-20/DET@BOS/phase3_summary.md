# Phase 3 Summary — DET @ BOS (2026-04-20, Fenway)

> 基本面快照。**不含盤口推薦/星級** — 盤口以 Phase 4 `prediction.json` 為準。

## 1. 先發投手對決

| 項目 | Jack Flaherty (DET, R) | Sonny Gray (BOS, R) |
|------|-----------------------|---------------------|
| 分級 | 🟡 Solid Starter | 🟢 Back-end Starter |
| ERA | 4.05 | 4.43 |
| FIP | **3.75** | 4.53 |
| xERA | 4.72 | **5.31** |
| xwOBA (被) | .344 | .362 |
| HR/9 | **0.45** | 1.33 |
| Whiff% | 9.0 | 8.6 |
| Barrel% (被) | 8.0 | 11.1 |
| Hard Hit% | 26.4 | 24.2 |
| Prior-year ERA | 4.64 | 4.28 |
| 年齡 | 30 | 36 📉 |

**關鍵判讀**：
- Flaherty FIP/xERA 皆優於 Gray（3.75 vs 4.53；4.72 vs 5.31），**壓制力差約 1 檔**。
- Gray **被 RHH 壓制差**：.333/.366/.590（大樣本 BF 41）；被 LHH 壓制強：.205/.255/.295（BF 47）。DET 中軸 RHH 多（Torres、Dingler、Báez、Vierling）— Gray 面對不利配對。
- Flaherty **控球 vs LHH 爆炸**：BB% **20.4%**（BF 49），OBP .417；vs RHH 卻宰殺：.152/.282/.182 K% 33.3%。BOS LHH（Anthony、Abreu、Duran）可能藉此累積保送。
- ERA vs xERA 閘門未觸發（Flaherty Δ=0.67、Gray Δ=0.88 皆 <1.5）。無需 YoY Statcast。
- Gray 36 歲 📉 年齡退化中；Flaherty 30 歲仍在帶。
- 3 場 game log：Flaherty 4.1/4.0/5.2 IP 都 <6 IP，早退；Gray 最新 4/8 對 MIL 6.1 IP 0ER 亮眼，但樣本小。

## 2. 打線評級

| 球隊 | Tier | avg_OPS | avg_xwOBA | K% / BB% | BABIP | Recent Heat |
|------|------|---------|-----------|----------|-------|-------------|
| DET | 🟠 Strong | .752 | .353 | 21.5 / 10.0 | .319 | ⚖️ Normal |
| BOS | 🟢 Weak | .628 | .302 | 25.1 / 8.0 | .288 | ⚖️ Normal |

**DET 打線**（對上 RHP Gray）：
- 熱打者：McGonigle (.892 recent 1.084)、**Dingler (.983 recent 1.178, Barrel% 23.5% elite)**、Carpenter (.802 recent 1.157 🔥)、Greene (.712 recent .817)。
- BABIP 回歸警訊：**Keith 季 .417 + 近 7d .333**（過熱，預期下修）；Dingler 季 .304 recent .286 合理。
- 串聯：obp_top3 .371、slg_mid .415 — 健康的傳鏈。
- 缺陣：Meadows (OF, 60d)、Sweeney (SS, 10d)、McKinstry (3B, 10d) — 替補深度考驗，但替補近期成績維持。

**BOS 打線**（對上 RHP Flaherty）：
- 亮點：**Contreras .902**（但 BABIP 季 .375 + recent **.583** = 嚴重過熱，預期顯著回歸，不予 Hot 加權）；Abreu .867（recent .421 cold + BABIP .200 → 運氣差但基底仍 OK）。
- Cold/低水平：Story .522、Durbin .439、Narváez .449、Duran .520、Mayer .542、Rafaela .739（近 7d .637）。
- 缺陣衝擊：**Casas + Romy Gonzalez 兩位 1B 同列 IL**，打線少一個核心長打點；Roman Anthony OBP .341 撐住上壘但 SLG .333 偏弱。
- 串聯：obp_top3 .324、slg_mid .355 — 明顯弱於 DET。

## 3. 牛棚

| 球隊 | Bullpen ERA | 核心 IL | 修正 |
|------|-------------|---------|------|
| BOS | 3.52 | **5 名投手 IL**（Houck 60d、Oviedo 60d、Slaten 15d、Crawford 15d、Sandoval 15d）— 核心 Houck + Slaten = 2 名核心缺陣 | O/U **+0.5 run**（對手 DET 得分上修）、BOS ML **-3%** |
| DET | 3.24 | Horn 15d、Brieske 60d、Melton 60d — 多為邊緣/復健中；Chapman/Jansen/Finnegan/Holton/Vest 核心完整 | 無修正（近 10 場 RA 2.1 驗證牛棚強勢） |

**牛棚雙向閘門**：BOS 2 名核心缺陣 → O/U +0.5 **且** BOS ML -3%（雙側皆修）。

## 4. 條件修正（信號）

| 信號 | 觸發 | Run Value / 備註 |
|------|------|-----------------|
| `bos_bullpen_il` | BOS 2 名核心 IL（Houck、Slaten） | O/U +0.5（上修 DET 得分），BOS ML -3% |
| `env_cold` | 49°F（每 10°F ≈ 1% 飛距） | 氣溫 49 vs 基準 70 → 飛距 -2%，O/U **-0.3 run** |
| `env_wind_unknown_dir` | 16.1 mph 但風向未知 | 保守處理：不修正（若 out = +0.3、若 in = -0.3，資訊不足不提修） |
| `flaherty_wildness_vs_lhh` | Flaherty BB% 20.4% vs LHH（BF 49 達引用門檻） | O/U +0.3 run（BOS LHH 保送累積） |
| `gray_rhh_vulnerability` | Gray vs RHH .333/.366/.590（BF 41） | O/U +0.3 run（DET RHH 攻擊） |
| `park_fenway` | PF 105 | 已於 park factor 公式內乘；不重複信號 |
| `rain_risk` | 80% precip | 建議用戶留意延賽 — 不進模型 |
| `early_season` | BOS 21 場 / DET 22 場 | **D4 受讓盤偏見防護觸發**（<30 場）。星級受限。 |
| `umpire_unknown` | HP 未確認 | 跳過主審修正 |

**BABIP 回歸閘門**（Hot/Cold 判定前已檢查）：
- Contreras recent BABIP .583 → 不予 Hot 加權
- Keith 季 BABIP .417 → Hot 加權打折
- Other `recent_heat = ⚖️ Normal`（BOS / DET 兩邊） — 無額外 Hot/Cold 修正

## 5. 條件與環境

- 球場：Fenway Park，PF 105（偏大分友善）
- 天氣：49°F、風 16.1 mph（方向未知）、降雨機率 80%
- 主審：**未確認**（WebSearch 命中的 4/20 ESPN-IN 文章可能是 4/19 ET 比賽時區錯置）
- 賽季階段：🌱 開季（雙方 <30 場）— 投影系統權重高、D4 觸發

## 6. 近期狀態與動能

| 項目 | DET | BOS |
|------|-----|-----|
| 近 10 場 | **8-2**（RS 4.1 / RA 2.1）🔥 | 5-5（RS 4.2 / RA 4.1） |
| 近 30 / 本季 | 12-10（RD +19） | 8-13（RD -15）📉 |
| 連勝/敗 | +2 連勝 | -2 連敗 |
| 趨勢 | ↑ | ↓ |

**前場 H2H**（4/19）：DET 6-2 BOS（Flaherty 對手 4/17 未登板；Flaherty 上次登板 4/9；Gray 上次 4/8 — 輪值常規休息）。

## 7. 修正後預期得分（Phase 3 初估，供 Phase 4 比對）

基於 league-avg ~4.4 R/G 起點，修正：

| 方向 | 計算 | 預估 |
|------|------|------|
| DET 得分 | 4.4 × PF(1.05) × DET_off(Strong, +0.2) × Gray_weak_vs_R(+0.3) + bos_pen_il(+0.5) + cold(-0.15) + wind(0) | **≈ 5.0-5.3** |
| BOS 得分 | 4.4 × PF(1.05) × BOS_off(Weak, -0.3) × Flaherty_wildness_vs_L(+0.3) + cold(-0.15) + det_pen(0) | **≈ 4.0-4.3** |
| 總分 | — | **≈ 9.0-9.5** |

⚠️ 此為 Phase 3 手算參考值；最終以 `predict.py` 的 `formula_prediction` 為準。

## 8. 整體判斷（方向性，無星級）

- **ML 方向**：基本面偏 **DET**（投手差 1 檔、打線差 1 檔、牛棚差 + IL、動能差、4/19 剛勝）
- **O/U 方向**：基本面偏 **OVER**（Fenway PF 105、DET 進攻強 + Gray 被 RHH 痛擊、BOS 牛棚 2 核 IL、Flaherty BB vs LHH 問題）；反向壓力：49°F + 16 mph 風 + 雨勢
- **Run Line 方向**：若 DET 被定為讓分方（-1.5），以 D4（<30 場受讓偏見）為警戒 — 星級需護欄
- **主要風險**：
  1. Flaherty 早退（3 場均 <6 IP）→ DET 牛棚提早登板雖強但仍有變數
  2. Gray 最近 1 場 6.1 IP 0ER 可能延續壓制
  3. 雨延/風向未知 — 預測環境信心度降級
  4. D4 早季樣本 — 護欄預計觸發
