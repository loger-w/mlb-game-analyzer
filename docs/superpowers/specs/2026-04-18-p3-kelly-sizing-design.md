# P3: Kelly Sizing — Design Spec

**Date:** 2026-04-18
**Status:** Draft — pending user review
**Parent roadmap:** MLB Betting Skill — Root-Cause Fixes (P3 Kelly → P2 CLV → P1 Retrain)

---

## 1. Goal

讓 skill 輸出 **下注尺寸**（quarter-Kelly + 3% hard cap + unit 顯示），不只是方向 + 星級。使用者拿到 prediction.json 後能直接照 `units` 欄位下注。

## 2. Non-Goals（明確排除，避免 scope creep）

- 真實 label 重訓（→ P1）
- CLV / closing line value 追蹤（→ P2）
- Line movement / reverse line movement 訊號（→ P2 + M5）
- Props / SGP / player props（skill 目前不支援）
- Bankroll state 管理（使用者自己套用 fraction 到 bankroll）
- 多 book line shopping（`fetch_odds.py` 限定 Pinnacle，不變）

## 3. 既有基建盤點（spec 前提）

| 既有 | 位置 | 狀態 |
|---|---|---|
| `calc_kelly(model_prob, ml)` → raw Kelly % | `scripts/odds_analyzer.py:92` | ✓ 可用，輸入 American odds |
| `_normal_cdf`, `_p_at_most`, `_p_exactly`, `_p_at_least` | `scripts/odds_analyzer.py:37-53` | ⚠️ σ=`_MLB_TOTAL_STD = 3.5`，但 `prediction.md` D2/D5 寫 4.5（既有 bug，見 §11） |
| `ml_to_implied_prob`, `hk_to_american`, `american_to_hk`, `calc_ev` | `scripts/odds_analyzer.py` | ✓ 全部到位 |
| `analyze_moneyline` | 同上 | ✓ 已回傳 `kelly` 欄位（raw） |
| `analyze_over_under` / `analyze_run_line` | 同上 | ✗ 只回 direction + stars，無 Kelly |
| Pinnacle snapshot 基建 | `scripts/fetch_odds.py` + `odds_snapshots/*.json` | ✓ Task Scheduler 4h 自動抓 |
| Snapshot 檔格式 | `odds_snapshots/2026-04-17_08-00-ET.json` | ✓ ML / OU / RL 三市場 + decimal odds + implied_pct |
| `TEAM_ABBREV` 隊名對照 | `scripts/predict.py:22` | ✓ 需反向 lookup |

**缺口（本 spec 要填）**：
1. fractional Kelly wrapper（divisor + cap + units）
2. O/U 和 RL 的 P(side) → Kelly 計算
3. `predict.py` 自動讀 snapshot + 整合 Kelly
4. prediction.json schema 新增 `kelly` 區塊
5. reference 文件補 Kelly 章節

---

## 4. Architecture Decision

**方案 C（Kelly 數學全部在 `odds_analyzer.py`）+ 方案 β（`predict.py` 自動讀 snapshot）+ γ fallback override CLI args**

理由（brainstorming 對話確認）：
- C 維持 single-responsibility：odds 相關計算集中
- β 自動化：Task Scheduler 已在收，使用者不用多跑 `odds_analyzer.py`
- γ override 保留：測試 / what-if 分析 / snapshot 壞掉時的逃生艙

---

## 5. API 變更

### 5.1 `scripts/odds_analyzer.py` — 新增 / 修改

#### 新 helper：`calc_fractional_kelly`

```python
def calc_fractional_kelly(
    model_prob: float,
    ml: int,
    divisor: int = 4,
    cap_pct: float = 3.0,
    unit_size_pct: float = 1.0,
) -> dict:
    """Fractional Kelly with hard cap + unit conversion.

    Args:
        model_prob: 模型估計勝率 (0.0-1.0)
        ml: American moneyline (正數或負數)
        divisor: Kelly 分數係數（4 = quarter）
        cap_pct: 每注上限（% of bankroll，3.0 = 3%）
        unit_size_pct: 1 單位代表幾 % bankroll（1.0 = 1u = 1%）

    Returns:
        {
            "raw_kelly_pct": float,       # calc_kelly 輸出
            "fractional_pct": float,      # raw / divisor
            "capped_pct": float,          # min(fractional, cap_pct)
            "units": float,               # capped / unit_size, rounded to 0.5
        }
    無 edge 時全部 0（不是 None — 0 是合法的「不下注」訊號）。
    """
```

測試值驗算：
- `calc_fractional_kelly(0.55, -110, 4, 3.0, 1.0)`
  - b = 100/110 ≈ 0.909
  - raw = (0.55 × 1.909 − 1) / 0.909 × 100 ≈ **5.5%**
  - fractional = 5.5 / 4 = **1.375%**
  - capped = min(1.375, 3.0) = **1.375%**
  - units = round(1.375 / 1.0 × 2) / 2 = **1.5u**

#### 新 helper：`decimal_to_american`

```python
def decimal_to_american(dec: float) -> int:
    if dec <= 1.0:
        raise ValueError(f"Invalid decimal odds: {dec}")
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    return int(round(-100 / (dec - 1)))
```

#### 新 helper：`p_margin_ge_2_given_win`

對齊 `reference/prediction.md` 的 Run Line -1.5 機率表：

```python
def p_margin_ge_2_given_win(favorite_ml: int) -> float:
    """P(margin >= 2 | win) 基於熱門方 American ML 的查表值。"""
    ml = abs(favorite_ml)
    if ml <= 130:   return 0.59
    if ml <= 170:   return 0.615
    if ml <= 220:   return 0.65
    return 0.695
```

#### 修改 `analyze_moneyline`

簽章 append：`..., kelly_params: dict = None`。回傳加：

```python
"kelly_fractional": {
    "direction": "HOME" | "AWAY",  # 與 direction 同
    "raw_kelly_pct": 2.34,
    "fractional_pct": 0.59,
    "capped_pct": 0.59,
    "units": 0.5,
}
```

若 `kelly_params is None` → 用預設 `{divisor: 4, cap_pct: 3.0, unit_size_pct: 1.0}`。

#### 修改 `analyze_over_under`

簽章加：`..., over_odds_ml: int = None, under_odds_ml: int = None, kelly_params: dict = None`。

```python
# 標準 MLB O/U 幾乎都是 .5 整數線（8.5、9.5），無 push
# 整數線簡化：忽略 push，按 strict >/< 處理
p_over  = 1.0 - _normal_cdf(line, predicted_total, _MLB_TOTAL_STD)
p_under = _normal_cdf(line, predicted_total, _MLB_TOTAL_STD)
# 註：_MLB_TOTAL_STD 目前 = 3.5，偏低；P3 附帶修正 → 4.5（見 §11）
```

回傳加 `kelly_fractional` 區塊，分 `over` / `under` 兩邊各自的 raw/fractional/capped/units。若對應側 `*_odds_ml is None` → 該側 Kelly 欄位為 `null`（保持整體 dict 存在）；若雙側皆 None → 整個 `kelly_fractional` 為 `null`。

#### 修改 `analyze_run_line`

新簽章（清理後）：

```python
def analyze_run_line(
    predicted_margin: float,
    model_home_win_pct: float = None,   # P(home win) from ml_prediction
    home_ml: int = None,                # American ML（市場熱門判定 + P(margin≥2|win) 查表）
    away_ml: int = None,
    home_rl_odds_ml: int = None,        # 主隊 RL 那邊的 American odds
    away_rl_odds_ml: int = None,        # 客隊 RL 那邊的 American odds
    home_point: float = None,           # Pinnacle snapshot 主隊 RL point（±1.5），用於標註 side
    kelly_params: dict = None,
) -> dict:
    # 市場熱門方：American ML 較負那方（不是 model favorite）
    fav_is_home = home_ml < away_ml
    fav_win_pct = model_home_win_pct if fav_is_home else (1 - model_home_win_pct)
    fav_ml      = home_ml if fav_is_home else away_ml
    fav_rl_odds = home_rl_odds_ml if fav_is_home else away_rl_odds_ml
    dog_rl_odds = away_rl_odds_ml if fav_is_home else home_rl_odds_ml
    p_cover_fav = fav_win_pct * p_margin_ge_2_given_win(fav_ml)
    p_cover_dog = 1 - p_cover_fav

    # Side 標籤：優先用 Pinnacle snapshot 的 point（source of truth）
    if home_point is not None:
        fav_side = "HOME_-1.5" if home_point < 0 else "AWAY_-1.5"
    else:
        fav_side = "HOME_-1.5" if fav_is_home else "AWAY_-1.5"
    # Kelly 算兩側（favorite_cover 用 fav_rl_odds，underdog_cover 用 dog_rl_odds）
```

**關鍵決策（C2 + C3）**：
- 熱門方判定一律用**市場 ML**（`home_ml < away_ml`），**不**用 `predicted_margin`。原因：`p_margin_ge_2_given_win` 查表 key 來自 `reference/prediction.md:91-96`「熱門方 ML」分布 — 這是條件於 bookmaker favorite 的歷史分佈，與 model 預測方向無關。當 model / market 分歧（e.g. model 推 home 贏但 market 熱門是 away），仍查 market bucket 才對齊表格定義。
- Side 標籤優先用 Pinnacle snapshot 回傳的 `home_point`（±1.5 是 Pinnacle 設定的事實）；若 snapshot 缺 point，才用 market ML 推測。

回傳加 `kelly_fractional.favorite_cover` / `kelly_fractional.underdog_cover`，每個含 raw/fractional/capped/units + `side`（"HOME_-1.5" / "AWAY_-1.5"）+ `decimal_odds`。若對應 RL odds 為 None → 該側 null。

---

### 5.2 `scripts/predict.py` — 新增 CLI args + 整合邏輯

#### 新增 CLI args（全部 optional）

| Arg | Default | 用途 |
|---|---|---|
| `--kelly-divisor` | `4` | Quarter-Kelly |
| `--kelly-cap` | `3.0` | 每注上限 % |
| `--unit-size` | `1.0` | 1u = 1% bankroll |
| `--no-auto-odds` | flag | 跳過 snapshot 自動查，走 CLI args only |
| `--ml-odds-home-dec` | `None` | Override decimal odds |
| `--ml-odds-away-dec` | `None` | 同上 |
| `--ou-odds-over-dec` | `None` | 同上 |
| `--ou-odds-under-dec` | `None` | 同上 |
| `--rl-odds-home-dec` | `None` | 同上 |
| `--rl-odds-away-dec` | `None` | 同上 |
| `--game-index` | `None` | Doubleheader 時指定 G1/G2 |

#### `compute_kelly_block` 簽章

```python
def compute_kelly_block(
    args,
    merged: dict,
    ml_prediction: dict | None,
    formula_prediction: dict,
    final_ml_rec: str,     # 既有 predict.py guardrail 算出（見 predict.py:407-413）
    final_ou_rec: str,     # 同上（見 predict.py:424-452）
    final_rl_rec: str,     # 同上（見 predict.py:455-468）
) -> dict | None:
    ...
```

#### 新邏輯（在 `--save` 路徑內）

**前置：ET 日期取得 (C1)**
- **主要來源**：從 `args.game_data` 路徑 regex 抽取 `analysis-data/YYYY-MM-DD/...` 的日期段。此 convention 保證是 ET（見 `reference/prediction.md:237`），與 snapshot 檔名的 `YYYY-MM-DD_HH-00-ET` 定義一致。
- **Fallback**：`_meta.game_date`（MLB API `gameDate`，UTC ISO）→ 用 `timezone(timedelta(hours=-4))`（複用 `fetch_odds.py:21` 常數）轉 ET 後切日期。
- **不再用** `game_date_iso[:10]` 切 UTC 日期當 ET — 西岸晚場（ET 22:00 = UTC 隔日 02:00）會跨日，造成 snapshot 對不上 → Kelly 沉默 null。
- `game_start_utc` 仍從 `_meta.game_date` 直接取（UTC ISO），供 `load_closest_snapshot` 做「snapshot 時間 < 開打時間」比較。

**前置：Guardrail PASS 對齊 (I1)**
- `compute_kelly_block` 讀 `final_ml_rec / final_ou_rec / final_rl_rec`（由 `predict.py main()` 既有 guardrail 算出）。
- 若 `final_ml_rec == "PASS"` → `kelly.ml = null` + `warnings.append("ml_guardrail_pass")`
- 若 `final_ou_rec == "PASS"` → `kelly.ou = null` + `warnings.append("ou_guardrail_pass")`
- 若 `final_rl_rec == "PASS"` → `kelly.rl = null` + `warnings.append("rl_guardrail_pass")`
- 僅對 non-PASS market 進行 Kelly 計算；反向保證：kelly 有數字時對應市場必然非 PASS。

**主流程：**

1. **Odds 收集**
   - 若 `--no-auto-odds` → 只用 CLI args；沒有 args → Kelly = null + warning
   - 否則 `load_closest_snapshot(game_date_et, game_start_utc)`（用前置取得的 ET 日期 + UTC 開打時間）
   - CLI args 覆寫 snapshot 值

2. **隊名對照**：snapshot 全名 → `TEAM_ABBREV` 反向 map → abbrev；比對 merged.json `_meta` 的 home/away abbrev

3. **Doubleheader**：若同日同兩隊有 2 筆 snapshot game → require `--game-index`；否則 raise `ValueError`。此 error **不被** `compute_kelly_block` call site 的 except 吞（見 5.2 錯誤處理），surface 給使用者。

4. **Odds 轉換 + RL point 傳遞 (C3)**：decimal → American（`decimal_to_american`）→ 餵給 odds_analyzer 三個 analyze 函數。RL 額外從 snapshot 取 `rl.home_point` 傳入 `analyze_run_line(home_point=...)` — side 標籤靠 Pinnacle 實際 point 決定，不靠 model / ML 推測。

5. **機率來源**
   - ML：`ml_prediction.home_win_pct / 100`（XGBoost，除 100 得 fraction）
   - O/U：從 `formula_prediction.total` 套常態分佈（`analyze_over_under` 內部用 `1 − Φ(line, μ, σ=_MLB_TOTAL_STD)`）
   - RL：`ml_prediction.home_win_pct / 100` + 市場 ML（判 market favorite）+ `p_margin_ge_2_given_win(fav_ml)`

6. **錯誤處理 (I4)**：call site 的 `try/except` 只捕 `(KeyError, IOError, json.JSONDecodeError)` — `ValueError`（doubleheader 無 `--game-index` / 壞 decimal odds）**不**被吞，使用者能看到實際錯誤並修復。

7. **寫入 prediction.json** — 新 `kelly` 區塊（見 5.3）

#### 新 helper：`load_closest_snapshot(game_date_et, game_start_time, snapshot_dir)`

```python
def load_closest_snapshot(game_date_et: str, game_start_time_utc: str,
                          snapshot_dir: str = None) -> dict | None:
    """Find newest snapshot with time < game_start_time, matching game_date_et.

    Returns None if no match. Caller handles fallback.
    """
```

邏輯：
- Glob `odds_snapshots/*.json`
- 解析檔名時間（`YYYY-MM-DD_HH-00-ET`）
- 過濾 `snapshot_time_utc < game_start_time_utc`
- 取最新者

### 5.3 `prediction.json` schema 擴充

```jsonc
{
  // ... 既有欄位 ...

  "kelly": {
    "snapshot_source": "odds_snapshots/2026-04-18_16-00-ET.json" | null,
    "snapshot_time_et": "2026-04-18 16:00 ET" | null,
    "params": {"divisor": 4, "cap_pct": 3.0, "unit_size_pct": 1.0},

    "ml": {
      "direction": "HOME",
      "decimal_odds": 1.83,
      "raw_kelly_pct": 2.34,
      "fractional_pct": 0.59,
      "capped_pct": 0.59,
      "units": 0.5
    } | null,

    "ou": {
      "direction": "OVER",
      "line": 8.5,
      "over": {"decimal_odds": 1.91, "raw_kelly_pct": ..., "fractional_pct": ..., "capped_pct": ..., "units": ...},
      "under": {"decimal_odds": 1.95, ...}
    } | null,

    "rl": {
      "favorite_side": "HOME_-1.5",
      "favorite": {"decimal_odds": 1.56, "raw_kelly_pct": ..., ...},
      "underdog": {"decimal_odds": 2.58, ...}
    } | null,

    "warnings": []  // e.g. ["team_name_mismatch", "no_snapshot", "doubleheader_ambiguous"]
  }
}
```

**向下相容**：`kelly` 整個 key 可省略；`assemble_analysis.py` / `upload_prediction.py` 消費時用 `.get("kelly")`。

---

## 6. Snapshot-to-Game 對照規則

### 6.1 隊名映射

Snapshot 用全名（"New York Mets"），prediction 用 abbrev（"NYM"）。`TEAM_ABBREV` 既有於 `predict.py:22`，格式為 `{full_name: abbrev}`，可直接 `dict.get(snapshot.home_team)` 查 abbrev。

**決策**：`TEAM_ABBREV` 保留在 `predict.py`。snapshot 對照邏輯寫在 `predict.py` 內部（odds_analyzer 不需要隊名映射）。

### 6.2 Doubleheader

若 snapshot 同日同兩隊有 2 場：
- `commence_et` 時間不同 → 選較接近 game_start 的
- 若 merged.json 有 G1/G2 後綴 → 用 `--game-index 1` or `2` 指定

### 6.3 Pinnacle 停盤後

Pinnacle 比賽開打後暫停（`fetch_odds.py` 註解提到）。找最後一筆 snapshot time < game_start_time，不找現時的。

---

## 7. Fallback / Error 矩陣

| 情境 | 行為 |
|---|---|
| 沒 `odds_snapshots/` 資料夾 | `kelly: null` + WARN log「no_snapshot_dir」 |
| 有資料夾但無檔 | `kelly: null` + WARN「no_snapshot_files」 |
| 有檔但無當日 game（以 ET 日期比對） | `kelly: null` + WARN「no_matching_snapshot」 |
| 隊名 snapshot 找不到對照 | `kelly: null` + WARN「team_name_mismatch: {name}」 |
| Pinnacle 只有 ML 沒 OU/RL | `kelly.ml` 正常，`kelly.ou`/`kelly.rl` 為 `null` |
| Raw Kelly ≤ 0（無 edge） | 該市場的 raw/fractional/capped/units 全 `0`（非 null） |
| Doubleheader 無 `--game-index` | raise `ValueError`（**不**被 compute_kelly_block call site 的 except 吞；使用者看到錯誤並加 `--game-index` 重跑） |
| `--no-auto-odds` + 無 CLI args | `kelly: null` + WARN「no_odds_available」 |
| Model prob 不可用（predict.py graceful fallback） | `kelly.ml: null`，formula-only 時不產 ML Kelly |
| `final_ml_rec == "PASS"`（DIVERGENT / INSUFFICIENT_SAMPLE 方向分歧等 D1-D5 guardrail 觸發） | `kelly.ml = null` + WARN「ml_guardrail_pass」（I1）|
| `final_ou_rec == "PASS"`（OU-1 差距<1.5 / OU-2 方向矛盾 / OU-3 無星級） | `kelly.ou = null` + WARN「ou_guardrail_pass」（I1）|
| `final_rl_rec == "PASS"`（LOW confidence / 未指定 stars / D4 大熱門受讓等） | `kelly.rl = null` + WARN「rl_guardrail_pass」（I1）|

---

## 8. Reference 文件更新

### 8.1 `reference/prediction.md`

新章節「## Kelly Sizing & Unit Output」：
- Fractional Kelly 公式
- 預設參數（quarter + 3% cap + 1u=1%）
- P(margin ≥ 2 | win) 查表（對齊既有章節）
- prediction.json `kelly` 區塊 schema 說明
- 使用範例（snapshot auto vs CLI override）

### 8.2 `reference/output-format.md`

TL;DR 的「盤口速查」表格新增一行：

```
💰 建議注碼（Quarter-Kelly, cap 3%）
| 市場 | 方向 | 單位 | Pinnacle odds |
|------|------|------|---------------|
| ML   | HOME | 1.5u | 1.83 |
| O/U  | OVER | 0.5u | 1.91 |
| RL   | PASS | —    | — |
```

### 8.3 `reference/workflow.md` Phase 4.0

註記：
- `predict.py --save` 自動讀 `odds_snapshots/` 中推薦時間最近的 Pinnacle snapshot
- CLI override args 存在但非 primary 流程
- **Kelly 完全對齊 D1-D5 guardrail**：若 `final_ml_rec == "PASS"` / `final_ou_rec == "PASS"` / `final_rl_rec == "PASS"`，對應市場的 `kelly.*` 為 `null`，`kelly.warnings` 紀錄觸發原因（`ml_guardrail_pass` / `ou_guardrail_pass` / `rl_guardrail_pass`）
- 反向保證：`kelly.<market>` 有數字時，對應市場必然非 PASS — direction / stars 仍由既有 guardrail 決定，Kelly 不改方向只決定注碼

---

## 9. Testing Strategy

### 9.1 單元測試（新 `scripts/tests/test_kelly.py`）

```python
# 1. Positive edge, quarter-Kelly, no cap engaged
calc_fractional_kelly(0.55, -110, 4, 3.0, 1.0)
# Expect raw~5.5, fractional~1.375, capped=1.375, units=1.5

# 2. Zero edge
calc_fractional_kelly(0.524, -110, 4, 3.0, 1.0)
# Expect all ~0

# 3. Negative edge
calc_fractional_kelly(0.45, -110, 4, 3.0, 1.0)
# Expect all 0 (calc_kelly returns 0 via max(0, ...))

# 4. Cap engaged (high edge long odds)
calc_fractional_kelly(0.50, +250, 4, 3.0, 1.0)
# b=2.5, raw=(0.50×3.5−1)/2.5×100 = 30%
# fractional=7.5, capped=3.0 (cap engaged), units=3.0

# 5. decimal_to_american
assert decimal_to_american(1.83) == -120
assert decimal_to_american(2.50) == 150

# 6. p_margin_ge_2_given_win
assert p_margin_ge_2_given_win(-120) == 0.59
assert p_margin_ge_2_given_win(-250) == 0.695
```

### 9.2 整合測試

用 `odds_snapshots/2026-04-17_08-00-ET.json`（真實現有 snapshot）跑：

```bash
$PYTHON scripts/predict.py --game-data sample_merged.json --save \
    --ou-line 8 --ou-rec OVER --ml-rec CLE --ml-stars 3 --run-line-rec PASS
```

驗證 `prediction.json` 的 `kelly.ml.capped_pct` 非 null 且 ≤ 3.0。

### 9.3 邊界手測

- Doubleheader 測試：假造 2 場同日 snapshot，不加 `--game-index` 應 error
- 隊名 mismatch：假造 snapshot 用「Oakland Athletics」（現實是「Athletics」），應 WARN + kelly=null
- `--no-auto-odds` 無 override args：應 WARN + kelly=null 但預測仍產出

---

## 10. Rollout

1. 實作 `scripts/odds_analyzer.py` 新 helpers + analyze 函數擴充 + 單元測試
2. 實作 `scripts/predict.py` CLI args + snapshot loader + 整合
3. 更新 reference 三個文件
4. 跑既有 analysis-data 樣本回歸測試（確認 kelly 區塊可選時不壞）
5. `upload_prediction.py` 不改（prediction.json 向下相容）

**不需要 DB migration、不需要重新跑歷史場次**。

---

## 11. Risk / Known Limits

- **⚠️ 既有 bug：`_MLB_TOTAL_STD = 3.5`**（`odds_analyzer.py:34`）
  - 現行 code 用 3.5，但 `prediction.md` D2/D5 全寫 SD ≈ 4.5
  - 影響：O/U 機率過度尖（Over 和 Under 同時被低估遠端），Kelly 會偏高
  - **P3 附帶修正**：改為 4.5（與文件對齊）。影響既有 `analyze_weighted_ou` 的 EV，需手測回歸確認沒亂。
  - 若要保守，可新增常數 `_MLB_TOTAL_STD_KELLY = 4.5` 只給新 Kelly 計算用，保留舊函數 3.5。**決策建議：統一改 4.5，文件才是事實**。
  - `[Source: reference/prediction.md D2/D5 baseline; pending empirical calibration from MLB 2020-2024 totals — tracked as P2 TODO]`
- **Kelly 品質 = model p 品質**：C1（synthetic label）未修前 p 可能偏差 3-5%，quarter-Kelly + 3% cap 是當前 safety margin。P1 完成後可考慮開到 half-Kelly。
  - Quarter-Kelly 選擇 `[Source: Thorp (2006) "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"; Poundstone (2005) Fortune's Formula ch.14 — fractional Kelly reduces drawdown when p̂ carries ±5-10% estimation error]`
  - 3% hard cap `[Source: ruin-risk heuristic; additionally tightened here due to synthetic-label p̂ uncertainty (P1 blocker). Revisit post-P1.]`
- **ML Kelly 僅用 XGBoost p**（`ml_prediction.home_win_pct`）；若 XGBoost 模型不存在（`predict.py` graceful fallback 到 formula-only），則 `kelly.ml = null`。不用 formula log5 算 ML Kelly，避免和 XGBoost 的 cross_validation 紀律打架。
- **Snapshot 4h 延遲**：快速變盤（steam move）下 Kelly 可能在推薦與實際下注之間過時。V1 接受；P2 M5 將加 line movement 偵測 + CLV tracking，屆時以實證資料量化延遲對 ROI 的影響（V1 不給具體 % 數字，避免無根據假設）。
- **σ=4.5 league-wide**：Coors 高變異球場的 Over Kelly 會略偏低（under-sized）。V1 接受；未來可依 park 調整。`[Source: reference/prediction.md D2/D5 baseline; pending empirical calibration from MLB 2020-2024 totals — P2 TODO]`
- **P(margin≥2|win) 查表僅 4 bucket**：解析度低，-149 vs -151 結果跳動。可接受；P1 完成後用實際模型的 margin 分佈取代。`[Source: reference/prediction.md Run Line -1.5 table range midpoints (58-60% / 60-63% / 63-67% / 67-72%); pending empirical calibration via pybaseball schedule_and_record game-level margins — P2 TODO]`
- **Market vs Model favorite (C2 解決方案)**：RL Kelly 用**市場 ML** 判熱門方查 `p_margin_ge_2_given_win` bucket，不用 model 的 `predicted_margin`。當 model / market 分歧，bucket 一律以 market 為準 — 因為表格本身就是條件於 bookmaker favorite 的歷史分佈，用 model favorite 查表會拿到錯的條件機率。
- **RL 只處理 ±1.5**：不支援 alternative run lines（±2.5）。目前 skill 也只處理 ±1.5，保持一致。
- **Pinnacle-only**：不做多 book line shopping（P2/M5 也不做，Pinnacle 是 sharp book，Kelly 對它的 line 才有意義）。

---

## 12. 完成定義 (Definition of Done)

**核心實作**
- [x] `calc_fractional_kelly`、`decimal_to_american`、`p_margin_ge_2_given_win` 單元測試全綠
- [x] `analyze_moneyline` / `analyze_over_under` / `analyze_run_line` 回傳含 `kelly_fractional` 區塊
- [x] `_MLB_TOTAL_STD` 改為 4.5（與文件對齊），既有 `analyze_weighted_ou` 回歸測試通過（EV 不跳離譜）
- [x] `predict.py --save` 用當日 snapshot 產出含 `kelly` 區塊的 prediction.json

**Critical bug 修復測試**
- [x] **C1**：西岸晚場 fixture（UTC `2026-04-19T02:00:00Z` / ET `2026-04-18`）→ `load_closest_snapshot` 用路徑 `analysis-data/2026-04-18/` 抽的 ET 日期找到 `2026-04-18_*-ET.json` snapshot，不再 null
- [x] **C2**：Market/Model 分歧 fixture（`home_ml=+140, away_ml=-150, predicted_margin=+0.5`）→ RL Kelly 用 `away_ml=-150` 查 bucket（得 0.615），`favorite_cover` 對應 away_rl_odds
- [x] **C3**：Pinnacle `home_point=+1.5`（home 是 dog）→ `kelly.rl.favorite_side == "AWAY_-1.5"`，不再靠 `fav_is_home` 推測
- [x] **I1**：DIVERGENT 場景 → `kelly.ml is None` + `"ml_guardrail_pass" in kelly.warnings`；INSUFFICIENT_SAMPLE 方向分歧同結果

**向下相容**
- [x] 無 snapshot / 隊名 mismatch / 無 odds 場景皆 graceful fallback（不 crash）
- [x] `assemble_analysis.py` 不改但消費新 prediction.json 不壞
- [x] 三份 reference 文件更新（prediction.md Kelly 章節含 source tags、output-format.md 注碼表、workflow.md Phase 4 註記）
- [x] Doubleheader 與 `--no-auto-odds` edge case 手測通過
- [x] `compute_kelly_block` call site `except (KeyError, IOError, json.JSONDecodeError)` 不吞 `ValueError`（I4）
