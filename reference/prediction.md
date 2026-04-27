# 預測公式 & 信號修正

## 比分預測方法

> 勝率與比分皆來自 `formula_prediction`（Log5 + 期望得分公式）。
> XGBoost 路徑於 2026-04 重構移除（spec 2026-04-26-mlb-skill-slimming-design）；
> 舊 `cross_validation` / `ml_prediction` / `xgb_raw_home_pct` 欄位不再產出。

### 步驟 1 — 計算雙方期望得分

```
E[R_A] = 聯盟平均得分 × (A 隊打線 xwOBA / 聯盟平均 xwOBA) × (B 隊投手 ERA / 聯盟平均 ERA) × (PF / 100)
```

### 步驟 1.5 — 套用 Run Value 修正（信號表）

```
修正後主隊得分 = 基礎主隊得分 + Σ(影響主隊的信號修正值)
修正後客隊得分 = 基礎客隊得分 + Σ(影響客隊的信號修正值)
修正後總分 = 修正後主隊得分 + 修正後客隊得分
```

---

## 信號 → Run Value 修正表

### 總分上修信號

| 信號 | Run 修正值 |
|------|-----------|
| 牛棚前日重操（5+ IP） | +0.5（加到對手得分） |
| 牛棚核心 2+ 人 IL | +1.0（加到對手得分） |
| Park Factor 修正 | (PF - 100) × 0.05（用 5 年回歸 PF） |
| 雙方打線近 7 天 Hot（場均 ≥ 5） | +0.5（需 BABIP 反向檢查） |
| Platoon 劣勢（全打線 vs 同手投手） | +0.4（加到投手得分） |
| Doubleheader 第二場 | +0.3 |
| 投手多/少休息日（vs 5 天） | ±0.04/day |

### 總分下修信號

| 信號 | Run 修正值 |
|------|-----------|
| 雙方先發皆 🟠 Strong Ace+ | -1.0 |
| 雙方先發皆 🟡 Solid+ | -0.5 |
| 雙方打線近 7 天 Cold（場均 ≤ 2） | -0.5（需 BABIP 反向檢查） |
| 季後賽得分壓縮 | ×0.84-0.86 |


---

## O/U 推薦（差距制）

```
差距 = 修正後總分 - O/U line
```

| 差距（絕對值） | 星級 |
|---------------|------|
| > 3.0 run | ⭐⭐⭐⭐⭐ 強烈推薦 |
| 2.0-3.0 | ⭐⭐⭐⭐ 推薦 |
| 1.5-2.0 | ⭐⭐⭐ 中度推薦 |
| < 1.5 | 不推薦（SD ≈ 4.5，在噪音範圍） |

---

## ML 星級

| 真實勝率 vs 隱含勝率差距 | 星級 |
|-------------------------|------|
| >= 15% | ⭐⭐⭐⭐⭐ |
| 10-15% | ⭐⭐⭐⭐ |
| 5-10% | ⭐⭐⭐ |
| < 5% | ⭐⭐（僅供參考） |

---

## Run Line -1.5 機率計算

```
P(win by 2+) = P(win) × P(margin ≥ 2 | win)
```

**P(margin ≥ 2 \| win) 查表**

`[Source: Run Line -1.5 table range midpoints (58-60% / 60-63% / 63-67% / 67-72%); pending empirical calibration via pybaseball schedule_and_record game-level margins — P2 TODO]`

| 熱門方 American ML | P(margin ≥ 2 \| win) |
|--------------------|---------------------|
| −130 ~ −110        | 0.59                |
| −170 ~ −131        | 0.615               |
| −220 ~ −171        | 0.65                |
| ≤ −221             | 0.695               |

**重要**：此表條件於 **bookmaker favorite**（American ML 較負方），不是 model predicted favorite。當 model 與 market 分歧時，bucket key 一律用 market ML — 否則查到錯的條件機率。

**Run Line -1.5 星級（區分主/客場）**：

| 條件 | P(cover) | 星級 |
|------|---------|------|
| 客場熱門 ML ≤ -200 | ~48-52% | ⭐⭐⭐ |
| 客場熱門 ML -150~-200 | ~42-46% | ⭐⭐ |
| 主場熱門 ML ≤ -200 | ~44-48% | ⭐⭐（主隊可能不打 9 局下半） |
| ML > -150 | < 40% | PASS |

---

## 讓分方向交叉驗證（輸出前強制執行）

```
1. 確認讓分方 = ML 負值方 = 投手/主場/牛棚綜合優勢方
2. 確認受讓方 = ML 正值方 = 綜合劣勢方
3. 預測差距 > 讓分值 → 推薦讓分方
4. 預測差距 < 讓分值 → 推薦受讓方
5. 差距 ±0.5 → 不推薦或降低星級
```

---

## 比賽敘事觸發條件

| 劇本 | 觸發條件 | 敘事方向 |
|------|---------|---------|
| 投手戰 | 雙方先發 🟠+ 且 FIP < 3.20 | 5 局前 1-0 或 2-1，牛棚決勝 |
| 打線互爆 | 雙方 xwOBA ≥ .340 + PF ≥ 105 | 先發撐不過 5 局先崩 |
| 單方碾壓 | 投手差 ≥ 2 級 + 打線差 ≥ 1 級 | 中段拉開，可能 cover -1.5 |
| 牛棚崩盤 | 一方牛棚核心 2+ IL + 前日 5+ IP 消耗 | 6 局後大量失分 |
| 硬幣翻轉 | ML 差 < 5% + 投手同級 | 均勢，單場隨機性 ~45% |

---

## 分析紀律

### D1：模型輸出紀律

`formula_prediction.lean`（HOME 或 AWAY）為唯一決定方向的依據。

- 可調整：勝率幅度 ±5%、信心降級、星級降級
- 可覆蓋：模型未計入的重大因素（先發臨時更換等）、用戶明確要求
- 不可覆蓋：軟性因素（Platoon / 連勝動能 / H2H 等）影響強度，不影響方向
- ML 路徑（XGBoost）於 2026-04 重構移除，`cross_validation` 欄位不再產出

> 預測紀錄歷史檔仍含 `cross_validation` 欄位（pre-2026-04），僅供觀察，新預測不寫入。

### D2：信號修正紀律

信號因子必須量化為 **Run Value 修正值**，不得獨立給 O/U 方向。

- 修正後總分 > O/U line → Over
- 修正後總分 < O/U line → Under
- 差距 < 1.5 run → 不推薦（SD ≈ 4.5 run）
- **不允許「信號說 Over 但比分說 Under」的矛盾。**

### D3：禁止同場對立方向推薦（硬性規則）

同一場比賽不得同時推薦 ML 勝方 A + A 的受讓（盤口邏輯上互斥會互咬）。

| formula home_win_pct | ML 推薦 | 受讓推薦 |
|----------------------|---------|---------|
| ≥ 60% | 可推 ML 勝方 | ⛔ 不得推「對方受讓」 |
| 55%-60% | 二選一（ML 或對方受讓） | 二選一（ML 或對方受讓） |
| < 55% | 不推 ML，可考慮受讓或 PASS | 允許 |

**原則**：ML 勝率越高，模型訊號越強，應該走 ML；勝率不夠高時走受讓盤才有價值。

### D5：比分與盤口一致性（硬性規則）

O/U 推薦方向必須與 D2 修正後總分一致：

| 修正後總分 vs O/U line | 允許推薦 |
|----------------------|---------|
| 修正後總分 ≤ O/U line | ⛔ 不得推 Over（允許 Under 或 PASS） |
| 修正後總分 ≥ O/U line | ⛔ 不得推 Under（允許 Over 或 PASS） |
| 差距 < 1.5 run | 僅允許 PASS |

此規則是 D2 的強化表述 — 當信號與比分矛盾時，信號必須讓步給比分。

---

## PASS 門檻 + 星級護欄（速查）

下表是 `predict.py` guardrail 自動執行的 PASS / 降級規則。分析者應**閱讀 predict.py 輸出的降級警告**，不要自創判斷。

### 自動 PASS 條件

| 指標 | PASS 條件 | 出處 |
|------|----------|------|
| O/U | 修正後總分與 O/U line 差距 < 1.5 run | D2 / D5 |
| ML | ml_pct vs 隱含勝率差距 < 5% | ML 星級 |
| ML | ml/formula 方向分歧（`ml_lean != formula_lean`） | D1（α 實作） |
| Run Line -1.5 | 熱門方 ML > -150（P(cover) < 40%） | Run Line 星級 |

### 星級上限護欄

| 觸發條件 | ml_stars 上限 | ou/rl 影響 |
|---------|--------------|-----------|
| `\|ml_pct − formula_log5_pct\| > 20%` | — | confidence 降 LOW |

---

## 預測紀錄存放位置

- **Per-game（真相來源）**：`analysis-data/{YYYY-MM-DD}/{AWAY}@{HOME}/prediction.json`
  單筆 JSON、pretty-printed。由 `predict.py --save` 產生。**屬於 mlb-game-analyzer skill**。
- **Per-date summary（快取）**：`analysis-data/{YYYY-MM-DD}/predictions.jsonl`
  當日所有場次 JSONL，由 `mlb-post-game-review` skill 重建。
- **賽後回填**：`actual_*` / `verified=true` 由 `mlb-post-game-review` skill 回填。

## 預測紀錄格式（prediction.json / predictions.jsonl）

```json
{
  "date": "YYYY-MM-DD",
  "game": "AWAY vs HOME",
  "home_team": "XXX",
  "away_team": "XXX",
  "home_sp": "Name",
  "away_sp": "Name",
  "home_sp_starts": 0,
  "away_sp_starts": 0,
  "predicted_winner": "HOME/AWAY",
  "predicted_home_pct": 0.0,
  "predicted_home_score": 0.0,
  "predicted_away_score": 0.0,
  "predicted_total": 0.0,
  "adjusted_total": 0.0,
  "signal_adjustments": {},
  "ou_line": 0.0,
  "ou_rec": "OVER/UNDER/PASS",
  "run_line_rec": "PASS",
  "ml_rec": "XXX",
  "ml_stars": 0,
  "confidence": "HIGH/MEDIUM/LOW",
  "tags": [],
  "umpire_name": null,
  "umpire_ou_rate": null,
  "park_factor": 100,
  "temperature_f": null,
  "wind_mph": null,
  "wind_direction": null,
  "actual_winner": null,
  "actual_home_score": null,
  "actual_away_score": null,
  "actual_total": null,
  "verified": false
}
```

賽後回填 `actual_*` 並設 `verified: true`。
