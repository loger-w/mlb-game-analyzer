# 預測公式 & 信號修正

## 比分預測方法

> ⚠️ **total_model（xgb_total_model.pkl）訓練資料有結構性缺陷，比分預測不可靠。**
> 勝率使用 XGBoost win_model，比分使用 formula 公式計算。
> predict.py 已實作此邏輯：`ml_prediction` 用於勝率，`formula_prediction` 用於比分。

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

**P(margin ≥ 2 \| win) 查表** → 見「Kelly Sizing & Unit Output」章節 §「P(margin ≥ 2 \| win) 查表」。

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

### D1：模型覆蓋紀律

ML (XGBoost) 與 Log5 (Formula) 方向一致時（即 `ml_lean == formula_lean`），**不得因軟性因素翻轉勝方**（Platoon 劣勢、連勝動能、H2H 等）。

- 可調整：勝率幅度 ±5%、信心降級、星級降級
- 可覆蓋：模型未計入的重大因素（先發臨時更換等）、用戶明確要求
- **不可覆蓋**：方向分歧（`ml_lean != formula_lean`）→ ML 強制 PASS
- **原則**：模型方向 > 直覺。軟性因素影響幅度，不影響方向。
- **實作**：`predict.py` 當場比對 `ml_lean` / `formula_lean`，不讀 `cross_validation` 字串（α 實作，見 spec 2026-04-22-mlb-skill-slimming-design.md §3.2）。`cross_validation` 欄位仍寫入（含 `INSUFFICIENT_SAMPLE` / `DIVERGENT` / `CONSISTENT` / `NO_ML_MODEL`）但僅供觀察。

### D2：信號修正紀律

信號因子必須量化為 **Run Value 修正值**，不得獨立給 O/U 方向。

- 修正後總分 > O/U line → Over
- 修正後總分 < O/U line → Under
- 差距 < 1.5 run → 不推薦（SD ≈ 4.5 run）
- **不允許「信號說 Over 但比分說 Under」的矛盾。**

### D3：禁止同場對立方向推薦（硬性規則）

同一場比賽不得同時推薦 ML 勝方 A + A 的受讓（盤口邏輯上互斥會互咬）。

| XGBoost home_win_pct | ML 推薦 | 受讓推薦 |
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

## Kelly Sizing & Unit Output

### 公式

Fractional Kelly 以真實勝率 `p` 與 American odds 計算：

```
b = 100/|ml|            (ml < 0)  或  ml/100  (ml > 0)
raw_kelly = max(0, (p × (b+1) − 1) / b)
fractional = raw_kelly / divisor
capped     = min(fractional, cap_pct)
units      = round(capped / unit_size, 0.5)
```

**預設參數**（由 `predict.py` args 控制）：

| 參數 | 預設值 | Source |
|------|-------|--------|
| `--kelly-divisor` | 4 (quarter-Kelly) | Thorp (2006) "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"; Poundstone (2005) *Fortune's Formula* ch.14 — fractional Kelly reduces drawdown when p̂ carries ±5-10% estimation error |
| `--kelly-cap` | 3.0 (% of bankroll) | Ruin-risk heuristic; tightened in V1 due to synthetic-label p̂ uncertainty (P1 blocker). Revisit post-P1. |
| `--unit-size` | 1.0 (1u = 1% bankroll) | UX convention; rounds `capped / unit_size` to nearest 0.5 unit |

### Odds 來源

`predict.py --save` 自動讀 `odds_snapshots/` 中推薦時間最近的 Pinnacle snapshot：
- Snapshot time 必須早於比賽開打時間
- 隊名對照用 `TEAM_ABBREV`（全名 → 縮寫）
- **ET 日期來源**：優先從 `args.game_data` 路徑（`analysis-data/YYYY-MM-DD/`）取；fallback 從 `_meta.game_date`（UTC ISO）轉 ET
- Doubleheader 需 `--game-index 1` 或 `2`；缺此 arg 時 `ValueError` 會 surface 給使用者（不吞）

CLI override（優先於 snapshot）：
- `--ml-odds-home-dec` / `--ml-odds-away-dec`
- `--ou-odds-over-dec` / `--ou-odds-under-dec`
- `--rl-odds-home-dec` / `--rl-odds-away-dec`

若 snapshot 與 CLI 都無對應市場 → 該市場 `kelly.*` = `null`。

### 機率來源

| 市場 | p 來源 | Source / Note |
|------|-------|---------------|
| ML | `ml_prediction.home_win_pct / 100`（XGBoost） | 不用 Log5，避免和 cross_validation 紀律打架 |
| O/U | `1 − Φ(line; μ=formula_prediction.total, σ=4.5)` | σ=4.5 `[Source: reference/prediction.md D2/D5 baseline; pending empirical calibration from MLB 2020-2024 totals — P2 TODO]` |
| RL -1.5 | `P(win) × P(margin ≥ 2 \| win)`，後者查表 | 熱門方用**市場 ML** 判定（非 model margin） — C2 修正 |

### P(margin ≥ 2 \| win) 查表

`[Source: reference/prediction.md Run Line -1.5 table range midpoints (58-60% / 60-63% / 63-67% / 67-72%); pending empirical calibration via pybaseball schedule_and_record game-level margins — P2 TODO]`

| 熱門方 American ML | P(margin ≥ 2 \| win) |
|--------------------|---------------------|
| −130 ~ −110        | 0.59                |
| −170 ~ −131        | 0.615               |
| −220 ~ −171        | 0.65                |
| ≤ −221             | 0.695               |

**重要（C2）**：此表條件於 **bookmaker favorite**（American ML 較負方），不是 model predicted favorite。當 model 與 market 分歧時，bucket key 一律用 market ML — 否則查到錯的條件機率。

### Side 標籤來源（C3）

`kelly.rl.favorite_side` 的 `"HOME_-1.5"` / `"AWAY_-1.5"` 優先用 Pinnacle snapshot 的 `rl.home_point`（±1.5 是 Pinnacle 設定的事實）；snapshot 缺 point 時才用 market ML 推測。

### prediction.json `kelly` 區塊 schema

```jsonc
"kelly": {
  "snapshot_source": "odds_snapshots/2026-04-18_16-00-ET.json" | null,
  "snapshot_time_et": "2026-04-18 16:00 ET" | null,
  "params": {"divisor": 4, "cap_pct": 3.0, "unit_size_pct": 1.0},
  "ml": {
    "direction": "HOME" | "AWAY",
    "decimal_odds": 1.83,
    "raw_kelly_pct": 2.34, "fractional_pct": 0.59,
    "capped_pct": 0.59, "units": 0.5
  } | null,
  "ou": {
    "direction": "OVER" | "UNDER" | "PASS",
    "line": 8.5,
    "over": { ... } | null,
    "under": { ... } | null
  } | null,
  "rl": {
    "favorite_side": "HOME_-1.5" | "AWAY_-1.5",
    "favorite": { ... } | null,
    "underdog": { ... } | null
  } | null,
  "warnings": [
    // e.g. "ml_guardrail_pass", "no_matching_snapshot", "team_name_mismatch: ..."
  ]
}
```

### 紀律

- **Kelly 完全對齊 D1-D5 guardrail**：若 `final_ml_rec == "PASS"` / `final_ou_rec == "PASS"` / `final_rl_rec == "PASS"`，對應市場的 `kelly.*` 為 `null`，`warnings` 紀錄觸發原因（`ml_guardrail_pass` / `ou_guardrail_pass` / `rl_guardrail_pass`）
- 反向保證：`kelly.<market>` 有數字時對應市場必然非 PASS — direction / stars 由既有 guardrail 決定，Kelly 不改方向只決定注碼
- 負 edge（raw ≤ 0）→ 該市場的 Kelly 欄位全 `0`（非 null；0 是合法的「不下注」訊號）
- **Snapshot 4h 延遲**：快速變盤（steam move）下 Kelly 可能在推薦與下注之間過時。V1 接受此限制。

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
  "cross_validation": "CONSISTENT/DIVERGENT/INSUFFICIENT_SAMPLE/NO_ML_MODEL",
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
