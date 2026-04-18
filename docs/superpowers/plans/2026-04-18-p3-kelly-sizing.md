# P3 Kelly Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓 `predict.py --save` 產出的 prediction.json 加入 quarter-Kelly + 3% hard cap 的注碼建議，資料來源從 `odds_snapshots/` 自動抓最近的 Pinnacle snapshot。

**Architecture:** Kelly 計算擴充既有 `scripts/odds_analyzer.py`（方案 C），`scripts/predict.py` 新增 snapshot 讀取邏輯（方案 β）與 CLI override args（方案 γ fallback）。向下相容既有 prediction.json schema。

**Tech Stack:** Python 3、pytest、既有 `odds_analyzer.py` / `predict.py` / `fetch_odds.py`、MLB Stats API snapshot 格式（Pinnacle decimal odds）。

**Spec:** `docs/superpowers/specs/2026-04-18-p3-kelly-sizing-design.md`

**Execution context:** main repo 根目錄（未使用 worktree）。Windows bash shell；`$PYTHON` 在本機 = `python`。

---

## File Structure

**Creates:**
- `scripts/tests/__init__.py` — 空檔，讓 pytest 視為 package
- `scripts/tests/test_kelly.py` — Kelly helper 單元測試
- `scripts/tests/test_odds_analyzer_extended.py` — `analyze_*` 擴充測試
- `scripts/tests/test_predict_snapshot.py` — snapshot loader 測試
- `scripts/tests/fixtures/sample_snapshot.json` — 小型 Pinnacle snapshot fixture
- `scripts/tests/fixtures/sample_merged.json` — 小型 `merged.json` fixture

**Modifies:**
- `scripts/odds_analyzer.py` — 新 3 helpers + 3 analyze 函數擴充 + σ 常數 3.5 → 4.5
- `scripts/predict.py` — 新 CLI args + snapshot loader + Kelly 整合 + prediction.json `kelly` block
- `scripts/requirements.txt` — 加 `pytest>=7.0.0`
- `reference/prediction.md` — 新 Kelly 章節
- `reference/output-format.md` — 盤口速查新增注碼行
- `reference/workflow.md` — Phase 4 註記自動 odds lookup

---

## Tasks

### Task 1: Bootstrap test infra + 驗證 pytest 可用

**Files:**
- Create: `scripts/tests/__init__.py`
- Modify: `scripts/requirements.txt`

- [ ] **Step 1: 建空 `scripts/tests/__init__.py`**

```bash
mkdir -p scripts/tests scripts/tests/fixtures
```
寫檔：
```
# scripts/tests/__init__.py
# empty — makes pytest treat this dir as a package
```

- [ ] **Step 2: 加 `pytest>=7.0.0` 到 requirements.txt**

`scripts/requirements.txt` 結尾加一行：
```
pytest>=7.0.0
```

- [ ] **Step 3: 安裝 pytest**

Run: `$PYTHON -m pip install pytest>=7.0.0`
Expected: `Successfully installed pytest-...`

- [ ] **Step 4: 驗證 pytest 能收集**

Run: `$PYTHON -m pytest scripts/tests/ --collect-only 2>&1 | tail -5`
Expected: `no tests ran` / `collected 0 items`（空目錄正確）

- [ ] **Step 5: Commit**

```bash
git add scripts/tests/__init__.py scripts/requirements.txt
git commit -m "test(mlb-skill): bootstrap pytest infrastructure for P3 Kelly sizing"
```

---

### Task 2: `calc_fractional_kelly` helper (TDD)

**Files:**
- Create: `scripts/tests/test_kelly.py`
- Modify: `scripts/odds_analyzer.py` (append helpers)

- [ ] **Step 1: 寫失敗測試**

建 `scripts/tests/test_kelly.py`：
```python
"""Unit tests for fractional Kelly helpers."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from odds_analyzer import calc_fractional_kelly


def test_positive_edge_quarter_kelly_no_cap():
    """p=0.55, ml=-110, quarter Kelly, no cap engaged."""
    result = calc_fractional_kelly(0.55, -110, divisor=4, cap_pct=3.0, unit_size_pct=1.0)
    # b = 100/110 ≈ 0.909
    # raw = (0.55 * 1.909 - 1) / 0.909 * 100 ≈ 5.5
    assert result["raw_kelly_pct"] == 5.5
    # fractional = 5.5 / 4 = 1.375
    assert result["fractional_pct"] == 1.375
    # cap not hit
    assert result["capped_pct"] == 1.375
    # units = 1.375 / 1.0 rounded to nearest 0.5 = 1.5
    assert result["units"] == 1.5


def test_zero_edge_returns_zero():
    """p at implied prob — no edge."""
    # at -110, implied = 110/210 ≈ 0.5238; use exactly that
    result = calc_fractional_kelly(0.5238, -110, divisor=4, cap_pct=3.0, unit_size_pct=1.0)
    # raw should round to ~0 (boundary)
    assert result["raw_kelly_pct"] <= 0.1
    assert result["fractional_pct"] <= 0.1
    assert result["capped_pct"] <= 0.1
    assert result["units"] == 0.0


def test_negative_edge_returns_zero():
    """p < implied — no bet."""
    result = calc_fractional_kelly(0.45, -110, divisor=4, cap_pct=3.0, unit_size_pct=1.0)
    assert result["raw_kelly_pct"] == 0
    assert result["fractional_pct"] == 0
    assert result["capped_pct"] == 0
    assert result["units"] == 0.0


def test_cap_engaged_high_edge_long_odds():
    """p=0.50, ml=+250 → big raw Kelly, cap should trigger."""
    result = calc_fractional_kelly(0.50, +250, divisor=4, cap_pct=3.0, unit_size_pct=1.0)
    # b=2.5, raw = (0.50 * 3.5 - 1) / 2.5 * 100 = 30
    assert result["raw_kelly_pct"] == 30.0
    # fractional = 30 / 4 = 7.5
    assert result["fractional_pct"] == 7.5
    # cap at 3.0
    assert result["capped_pct"] == 3.0
    # units = 3.0 / 1.0 = 3.0
    assert result["units"] == 3.0


def test_half_kelly_divisor():
    """divisor=2 doubles fractional output."""
    result = calc_fractional_kelly(0.55, -110, divisor=2, cap_pct=3.0, unit_size_pct=1.0)
    # raw same = 5.5
    assert result["raw_kelly_pct"] == 5.5
    # fractional = 5.5 / 2 = 2.75
    assert result["fractional_pct"] == 2.75
    assert result["capped_pct"] == 2.75
    # units = 2.75, round to nearest 0.5 = 3.0
    assert result["units"] == 3.0
```

- [ ] **Step 2: 驗證測試先失敗**

Run: `$PYTHON -m pytest scripts/tests/test_kelly.py -v`
Expected: `ImportError: cannot import name 'calc_fractional_kelly'` (5 failures)

- [ ] **Step 3: 實作 helper**

在 `scripts/odds_analyzer.py` 既有 `calc_kelly`（約 line 101）後方加入：
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
        {raw_kelly_pct, fractional_pct, capped_pct, units}
        無 edge 時全部 0（不是 None — 0 是合法的「不下注」訊號）。
    """
    raw = calc_kelly(model_prob, ml)          # already returns 0 if negative
    fractional = round(raw / divisor, 4)
    capped = round(min(fractional, cap_pct), 4)
    # units：以 unit_size_pct 為 1u，round 到最近 0.5
    units = round(capped / unit_size_pct * 2) / 2 if unit_size_pct > 0 else 0.0
    return {
        "raw_kelly_pct": raw,
        "fractional_pct": fractional,
        "capped_pct": capped,
        "units": units,
    }
```

- [ ] **Step 4: 驗證測試通過**

Run: `$PYTHON -m pytest scripts/tests/test_kelly.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/odds_analyzer.py scripts/tests/test_kelly.py
git commit -m "feat(mlb-skill): add calc_fractional_kelly with cap + unit output"
```

---

### Task 3: `decimal_to_american` helper (TDD)

**Files:**
- Modify: `scripts/tests/test_kelly.py`
- Modify: `scripts/odds_analyzer.py`

- [ ] **Step 1: 新增失敗測試**

附加到 `scripts/tests/test_kelly.py` 底部：
```python
from odds_analyzer import decimal_to_american


def test_decimal_to_american_favorite():
    """dec=1.83 → American -120."""
    assert decimal_to_american(1.83) == -120


def test_decimal_to_american_underdog():
    """dec=2.50 → American +150."""
    assert decimal_to_american(2.50) == 150


def test_decimal_to_american_even():
    """dec=2.00 → American +100."""
    assert decimal_to_american(2.00) == 100


def test_decimal_to_american_invalid():
    """dec<=1.0 should raise ValueError."""
    import pytest
    with pytest.raises(ValueError):
        decimal_to_american(1.0)
    with pytest.raises(ValueError):
        decimal_to_american(0.5)
```

- [ ] **Step 2: 驗證測試失敗**

Run: `$PYTHON -m pytest scripts/tests/test_kelly.py::test_decimal_to_american_favorite -v`
Expected: `ImportError: cannot import name 'decimal_to_american'`

- [ ] **Step 3: 實作**

在 `scripts/odds_analyzer.py` 的 `american_to_hk` 後方（約 line 79）加入：
```python
def decimal_to_american(dec: float) -> int:
    """Decimal odds → American moneyline."""
    if dec <= 1.0:
        raise ValueError(f"Invalid decimal odds: {dec}")
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    return int(round(-100 / (dec - 1)))
```

- [ ] **Step 4: 驗證測試通過**

Run: `$PYTHON -m pytest scripts/tests/test_kelly.py -v`
Expected: `9 passed`（Task 2 的 5 個 + 本 task 4 個）

- [ ] **Step 5: Commit**

```bash
git add scripts/odds_analyzer.py scripts/tests/test_kelly.py
git commit -m "feat(mlb-skill): add decimal_to_american odds conversion"
```

---

### Task 4: `p_margin_ge_2_given_win` helper (TDD)

**Files:**
- Modify: `scripts/tests/test_kelly.py`
- Modify: `scripts/odds_analyzer.py`

- [ ] **Step 1: 新增失敗測試**

附加：
```python
from odds_analyzer import p_margin_ge_2_given_win


def test_p_margin_bucket_shallow_favorite():
    """-120 → 0.59"""
    assert p_margin_ge_2_given_win(-120) == 0.59


def test_p_margin_bucket_mid():
    """-150 → 0.615"""
    assert p_margin_ge_2_given_win(-150) == 0.615


def test_p_margin_bucket_heavy():
    """-200 → 0.65"""
    assert p_margin_ge_2_given_win(-200) == 0.65


def test_p_margin_bucket_monster():
    """-250 → 0.695"""
    assert p_margin_ge_2_given_win(-250) == 0.695


def test_p_margin_positive_ml_treated_same():
    """+250 underdog favorite scenario (hypothetical) → treat by magnitude."""
    # abs value drives bucket
    assert p_margin_ge_2_given_win(+250) == 0.695
```

- [ ] **Step 2: 驗證失敗**

Run: `$PYTHON -m pytest scripts/tests/test_kelly.py::test_p_margin_bucket_shallow_favorite -v`
Expected: `ImportError: cannot import name 'p_margin_ge_2_given_win'`

- [ ] **Step 3: 實作**

在 `scripts/odds_analyzer.py` 的 `decimal_to_american` 後方加入：
```python
def p_margin_ge_2_given_win(favorite_ml: int) -> float:
    """P(margin >= 2 | win)，對齊 reference/prediction.md 的 Run Line -1.5 機率表。"""
    ml = abs(favorite_ml)
    if ml <= 130:
        return 0.59
    if ml <= 170:
        return 0.615
    if ml <= 220:
        return 0.65
    return 0.695
```

- [ ] **Step 4: 驗證通過**

Run: `$PYTHON -m pytest scripts/tests/test_kelly.py -v`
Expected: `14 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/odds_analyzer.py scripts/tests/test_kelly.py
git commit -m "feat(mlb-skill): add p_margin_ge_2_given_win table lookup"
```

---

### Task 5: Fix `_MLB_TOTAL_STD` 3.5 → 4.5 + regression check

**Files:**
- Modify: `scripts/odds_analyzer.py:34`

- [ ] **Step 1: 記錄當前 σ=3.5 下的 baseline 值**

Run: `$PYTHON -c "import sys; sys.path.insert(0, 'scripts'); from odds_analyzer import _p_at_least; print(round(_p_at_least(9, 10.0), 4))"`
Expected: `0.6664`（σ=3.5 下 P(total ≥ 9 | μ=10.0)：`1 − Φ((8.5−10)/3.5) = 1 − Φ(−0.4286) ≈ 0.666`）

- [ ] **Step 2: 改常數為 4.5**

Edit `scripts/odds_analyzer.py:34`，替換：
```python
_MLB_TOTAL_STD = 3.5  # MLB 比賽總分標準差（典型值）
```
為：
```python
_MLB_TOTAL_STD = 4.5  # 對齊 reference/prediction.md D2/D5 紀律（原 3.5 為既有 bug）
```

- [ ] **Step 3: 驗證新 σ 下的輸出變更**

Run: `$PYTHON -c "import sys; sys.path.insert(0, 'scripts'); from odds_analyzer import _p_at_least; print(round(_p_at_least(9, 10.0), 4))"`
Expected: `0.6306`（σ=4.5 下同樣條件，分佈變寬 → 尾巴機率略降；數值算式 `1 − Φ(−0.333) ≈ 0.631`）

- [ ] **Step 4: 跑既有 kelly 測試確認無誤傷**

Run: `$PYTHON -m pytest scripts/tests/test_kelly.py -v`
Expected: `14 passed`（Kelly 測試不依賴 σ）

- [ ] **Step 5: Commit**

```bash
git add scripts/odds_analyzer.py
git commit -m "fix(mlb-skill): correct _MLB_TOTAL_STD from 3.5 to 4.5 to align with prediction.md"
```

---

### Task 6: `analyze_moneyline` 加 `kelly_fractional` block (TDD)

**Files:**
- Create: `scripts/tests/test_odds_analyzer_extended.py`
- Modify: `scripts/odds_analyzer.py`（`analyze_moneyline`，約 line 138）

- [ ] **Step 1: 建測試檔**

`scripts/tests/test_odds_analyzer_extended.py`：
```python
"""Tests for analyze_moneyline/over_under/run_line Kelly extensions."""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from odds_analyzer import analyze_moneyline


def test_analyze_ml_has_kelly_fractional():
    """ML 分析回傳含 kelly_fractional 區塊，對方向與金額有正確數值。"""
    # Home at -150 (implied 60%), model says 65% → edge on home
    result = analyze_moneyline(home_ml=-150, away_ml=+140, model_win_pct=0.65)
    assert "kelly_fractional" in result
    kf = result["kelly_fractional"]
    assert kf["direction"] == "HOME"  # 與 result["direction"] 同
    # raw Kelly at -150 with p=0.65:
    # b = 100/150 ≈ 0.6667; raw = (0.65*1.6667 - 1)/0.6667 = 0.0833/0.6667 ≈ 12.5%
    assert kf["raw_kelly_pct"] > 10
    assert kf["raw_kelly_pct"] < 15
    assert kf["fractional_pct"] == round(kf["raw_kelly_pct"] / 4, 4)
    assert kf["capped_pct"] <= 3.0
    assert kf["units"] >= 0


def test_analyze_ml_no_edge_zero_kelly():
    """若 model 跟 implied 一致 → Kelly 0。"""
    # Home -110 implied ~52.4%; model says exactly 52.4% → zero edge
    result = analyze_moneyline(home_ml=-110, away_ml=-110, model_win_pct=0.524)
    kf = result["kelly_fractional"]
    # direction 由 EV 比較決定，但 Kelly 應接近 0
    assert kf["raw_kelly_pct"] <= 0.1
    assert kf["units"] == 0.0


def test_analyze_ml_custom_kelly_params():
    """kelly_params override 預設 divisor/cap。"""
    result = analyze_moneyline(
        home_ml=-150, away_ml=+140, model_win_pct=0.65,
        kelly_params={"divisor": 2, "cap_pct": 5.0, "unit_size_pct": 1.0},
    )
    kf = result["kelly_fractional"]
    # half-Kelly: fractional = raw / 2
    assert kf["fractional_pct"] == round(kf["raw_kelly_pct"] / 2, 4)
```

- [ ] **Step 2: 驗證失敗**

Run: `$PYTHON -m pytest scripts/tests/test_odds_analyzer_extended.py -v`
Expected: `KeyError: 'kelly_fractional'`（3 failures）

- [ ] **Step 3: 修改 `analyze_moneyline`**

找到 `scripts/odds_analyzer.py:138` 的 `analyze_moneyline`，改簽章並在 return 前加入計算：

```python
def analyze_moneyline(
    home_ml: int,
    away_ml: int,
    model_win_pct: float,
    kelly_params: dict = None,
) -> dict:
    """分析 Moneyline 盤口"""
    home_implied = ml_to_implied_prob(home_ml)
    away_implied = ml_to_implied_prob(away_ml)

    home_ev = calc_ev(model_win_pct, home_ml)
    away_ev = calc_ev(1 - model_win_pct, away_ml)

    home_kelly = calc_kelly(model_win_pct, home_ml)
    away_kelly = calc_kelly(1 - model_win_pct, away_ml)

    # 推薦方向：取 EV 較高的一方
    if home_ev > away_ev:
        direction = "HOME"
        best_ev = home_ev
        best_kelly = home_kelly
        prob_diff = (model_win_pct - home_implied) * 100
        kelly_prob = model_win_pct
        kelly_ml = home_ml
    else:
        direction = "AWAY"
        best_ev = away_ev
        best_kelly = away_kelly
        prob_diff = ((1 - model_win_pct) - away_implied) * 100
        kelly_prob = 1 - model_win_pct
        kelly_ml = away_ml

    stars = get_stars_ml(prob_diff)

    # Fractional Kelly
    kp = kelly_params or {}
    kf = calc_fractional_kelly(
        kelly_prob, kelly_ml,
        divisor=kp.get("divisor", 4),
        cap_pct=kp.get("cap_pct", 3.0),
        unit_size_pct=kp.get("unit_size_pct", 1.0),
    )
    kf["direction"] = direction

    return {
        "home_ml": home_ml,
        "away_ml": away_ml,
        "home_implied_pct": round(home_implied * 100, 1),
        "away_implied_pct": round(away_implied * 100, 1),
        "model_home_pct": round(model_win_pct * 100, 1),
        "model_away_pct": round((1 - model_win_pct) * 100, 1),
        "home_ev": home_ev,
        "away_ev": away_ev,
        "direction": direction,
        "prob_diff": round(prob_diff, 1),
        "kelly": round(best_kelly, 2),       # 既有 raw 欄位保留
        "kelly_fractional": kf,               # 新區塊
        "stars": stars,
    }
```

- [ ] **Step 4: 驗證通過**

Run: `$PYTHON -m pytest scripts/tests/test_odds_analyzer_extended.py -v`
Expected: `3 passed`

Run: `$PYTHON -m pytest scripts/tests/ -v`
Expected: 所有測試通過 (14 + 3 = 17)

- [ ] **Step 5: Commit**

```bash
git add scripts/odds_analyzer.py scripts/tests/test_odds_analyzer_extended.py
git commit -m "feat(mlb-skill): add kelly_fractional block to analyze_moneyline"
```

---

### Task 7: `analyze_over_under` 加 `kelly_fractional` block (TDD)

**Files:**
- Modify: `scripts/tests/test_odds_analyzer_extended.py`
- Modify: `scripts/odds_analyzer.py`（`analyze_over_under`，約 line 179）

- [ ] **Step 1: 新增失敗測試**

附加到 `test_odds_analyzer_extended.py`：
```python
from odds_analyzer import analyze_over_under


def test_analyze_ou_kelly_both_sides():
    """line=8.5, predicted=10.0 → Over 有 edge；Under 無 edge。"""
    result = analyze_over_under(
        line=8.5, predicted_total=10.0,
        over_odds_ml=-110, under_odds_ml=-110,
    )
    assert result["direction"] == "OVER"
    assert "kelly_fractional" in result
    kf = result["kelly_fractional"]
    assert "over" in kf and "under" in kf
    # Over 應該有正 Kelly
    assert kf["over"]["raw_kelly_pct"] > 0
    # Under 應該 0
    assert kf["under"]["raw_kelly_pct"] == 0


def test_analyze_ou_no_odds_kelly_null():
    """未傳 odds → kelly_fractional 為 null。"""
    result = analyze_over_under(line=8.5, predicted_total=10.0)
    assert result["kelly_fractional"] is None


def test_analyze_ou_partial_odds():
    """只有 Over odds → Under 側 null，Over 側有值。"""
    result = analyze_over_under(
        line=8.5, predicted_total=10.0, over_odds_ml=-110,
    )
    kf = result["kelly_fractional"]
    assert kf is not None
    assert kf["over"] is not None
    assert kf["under"] is None
    assert kf["over"]["raw_kelly_pct"] > 0
```

- [ ] **Step 2: 驗證失敗**

Run: `$PYTHON -m pytest scripts/tests/test_odds_analyzer_extended.py -v`
Expected: 3 個新 test 失敗（`TypeError: unexpected keyword argument 'over_odds_ml'`）

- [ ] **Step 3: 修改 `analyze_over_under`**

替換整個函數（約 line 179-197）：
```python
def analyze_over_under(
    line: float,
    predicted_total: float,
    over_odds_ml: int = None,
    under_odds_ml: int = None,
    kelly_params: dict = None,
) -> dict:
    """分析直線大小分盤口（無拆注）— 使用 run 差距制。

    O/U 幾乎都是 .5 整數線，忽略 push 處理。
    """
    diff = predicted_total - line
    stars = get_stars_ou(diff)

    if stars == 0:
        direction = "PASS"
    elif diff > 0:
        direction = "OVER"
    else:
        direction = "UNDER"

    # 機率：P(Over) = 1 - Φ(line; μ=predicted_total, σ=_MLB_TOTAL_STD)
    p_over = 1.0 - _normal_cdf(line, predicted_total, _MLB_TOTAL_STD)
    p_under = 1.0 - p_over

    # Kelly（若有 odds）
    kelly_fractional = None
    if over_odds_ml is not None or under_odds_ml is not None:
        kp = kelly_params or {}
        kelly_fractional = {"over": None, "under": None}
        if over_odds_ml is not None:
            kf = calc_fractional_kelly(
                p_over, over_odds_ml,
                divisor=kp.get("divisor", 4),
                cap_pct=kp.get("cap_pct", 3.0),
                unit_size_pct=kp.get("unit_size_pct", 1.0),
            )
            kf["decimal_odds"] = round(american_to_hk(over_odds_ml) + 1, 3)
            kelly_fractional["over"] = kf
        if under_odds_ml is not None:
            kf = calc_fractional_kelly(
                p_under, under_odds_ml,
                divisor=kp.get("divisor", 4),
                cap_pct=kp.get("cap_pct", 3.0),
                unit_size_pct=kp.get("unit_size_pct", 1.0),
            )
            kf["decimal_odds"] = round(american_to_hk(under_odds_ml) + 1, 3)
            kelly_fractional["under"] = kf

    return {
        "line": line,
        "predicted_total": round(predicted_total, 1),
        "diff": round(diff, 1),
        "direction": direction,
        "stars": stars,
        "p_over": round(p_over, 4),
        "p_under": round(p_under, 4),
        "kelly_fractional": kelly_fractional,
    }
```

- [ ] **Step 4: 驗證通過**

Run: `$PYTHON -m pytest scripts/tests/ -v`
Expected: 20 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/odds_analyzer.py scripts/tests/test_odds_analyzer_extended.py
git commit -m "feat(mlb-skill): add kelly_fractional to analyze_over_under with P(Over/Under)"
```

---

### Task 8: `analyze_run_line` 加 `kelly_fractional` block (TDD)

**Files:**
- Modify: `scripts/tests/test_odds_analyzer_extended.py`
- Modify: `scripts/odds_analyzer.py`（`analyze_run_line`，約 line 282）

- [ ] **Step 1: 新增失敗測試**

附加：
```python
from odds_analyzer import analyze_run_line


def test_analyze_rl_kelly_favorite_cover():
    """margin=+2.5, model_home_win=0.65, home ML -150 熱門（market 與 model 同向）。"""
    result = analyze_run_line(
        predicted_margin=2.5,
        model_home_win_pct=0.65,
        home_ml=-150,
        away_ml=+140,
        home_rl_odds_ml=-110,  # home -1.5 at -110
        away_rl_odds_ml=-110,  # away +1.5 at -110
        home_point=-1.5,       # Pinnacle: home 熱門 → home point = -1.5
    )
    assert "kelly_fractional" in result
    kf = result["kelly_fractional"]
    assert "favorite_cover" in kf
    assert "underdog_cover" in kf
    # Market favorite = home (ml=-150 更負), fav_ml=-150 → bucket 0.615
    # P(cover_fav) = 0.65 × 0.615 ≈ 0.3998；implied at -110 ≈ 0.5238 → Kelly ~0
    # P(cover_dog) ≈ 0.6002, implied 0.5238 → edge ~7.6%
    assert kf["underdog_cover"]["raw_kelly_pct"] > 0
    assert kf["favorite_cover"]["raw_kelly_pct"] >= 0  # 可能 0
    assert kf["favorite_cover"]["side"] == "HOME_-1.5"


def test_analyze_rl_market_favorite_when_model_disagrees():
    """C2 bug 測試：model 認為 home 贏 +0.5 分，但 market 熱門是 away。
    查 bucket 必須用 away_ml（市場熱門），不是 home（model 預測贏）。
    """
    # market: away 熱門 (ml=-150)，home 冷門 (ml=+140)
    # model: predicted_margin=+0.5（home 小勝，但 model P(home)=0.55 也不強）
    result = analyze_run_line(
        predicted_margin=0.5,
        model_home_win_pct=0.55,
        home_ml=+140,           # home 冷門
        away_ml=-150,           # away 熱門（market favorite）
        home_rl_odds_ml=+200,   # home +1.5 at +200（dog RL odds）
        away_rl_odds_ml=-260,   # away -1.5 at -260（fav RL odds）
        home_point=+1.5,        # Pinnacle: home 拿 +1.5 → home 是 dog
    )
    kf = result["kelly_fractional"]
    # fav_is_home 必須是 False（market favorite = away，不是 home）
    # fav_ml = away_ml = -150 → bucket = 0.615
    # p_cover_fav = (1 - 0.55) × 0.615 = 0.45 × 0.615 ≈ 0.2768（away win × margin 條件）
    # p_cover_dog = 1 - 0.2768 ≈ 0.7232（home 拿 +1.5 cover 機率）
    # Side 標註來自 home_point=+1.5 → fav_side = "AWAY_-1.5"
    assert kf["favorite_cover"]["side"] == "AWAY_-1.5"
    # favorite (away -1.5 @ -260) implied ≈ 72.2%, model p_cover ≈ 27.7% → raw Kelly = 0（負 edge）
    assert kf["favorite_cover"]["raw_kelly_pct"] == 0
    # underdog (home +1.5 @ +200) implied ≈ 33.3%, model p_cover ≈ 72.3% → 強正 edge
    assert kf["underdog_cover"]["raw_kelly_pct"] > 0
    # favorite Kelly 用的 odds 應該是 away_rl_odds_ml (-260)，不是 home_rl_odds_ml (+200)
    # 若 C2/C3 bug 未修，code 會誤用 home_rl_odds_ml=+200 當 favorite odds，
    # 導致 favorite_cover.raw_kelly_pct 反而顯示大正值 —— 此斷言會 fail
    assert kf["favorite_cover"]["decimal_odds"] == pytest.approx(1.385, abs=0.01)  # -260 → 1.385


def test_analyze_rl_side_label_falls_back_when_home_point_missing():
    """home_point 未傳 → 用 market ML 推 side（fallback path）。"""
    result = analyze_run_line(
        predicted_margin=2.5,
        model_home_win_pct=0.65,
        home_ml=-150, away_ml=+140,
        home_rl_odds_ml=-110, away_rl_odds_ml=-110,
        # home_point 省略
    )
    kf = result["kelly_fractional"]
    assert kf["favorite_cover"]["side"] == "HOME_-1.5"  # fav_is_home=True → home 是 -1.5


def test_analyze_rl_no_odds_kelly_null():
    """沒傳 RL odds → kelly_fractional 為 null。"""
    result = analyze_run_line(predicted_margin=2.5, model_home_win_pct=0.65)
    assert result["kelly_fractional"] is None


def test_analyze_rl_missing_ml_kelly_null():
    """有 RL odds 但沒 ML（無法判 market favorite / 查 bucket）→ null。"""
    result = analyze_run_line(
        predicted_margin=2.5, model_home_win_pct=0.65,
        home_rl_odds_ml=-110, away_rl_odds_ml=-110,
        # home_ml / away_ml 未傳
    )
    assert result["kelly_fractional"] is None
```

（`pytest` 已 import 過；確保檔案頂部有 `import pytest`。）

- [ ] **Step 2: 驗證失敗**

Run: `$PYTHON -m pytest scripts/tests/test_odds_analyzer_extended.py -v`
Expected: 3 個 fail（`TypeError: unexpected keyword argument`）

- [ ] **Step 3: 修改 `analyze_run_line`**

替換整個函數（約 line 282-302）。**關鍵改動（C2 + C3）**：
- `fav_is_home` 用 **market ML**（`home_ml < away_ml`），不再用 `predicted_margin`
- 新增 `home_point` 參數，side 標籤優先從 Pinnacle snapshot 的 point 推
- Fallback 保留：snapshot 無 point 時用 market ML 推 side

```python
def analyze_run_line(
    predicted_margin: float,
    model_home_win_pct: float = None,
    home_ml: int = None,
    away_ml: int = None,
    home_rl_odds_ml: int = None,
    away_rl_odds_ml: int = None,
    home_point: float = None,       # Pinnacle snapshot 主隊 RL point（±1.5）
    kelly_params: dict = None,
) -> dict:
    """分析讓分盤（-1.5）。

    C2/C3 fix: 熱門方用市場 ML 判定（非 model margin）；side 標籤優先用 Pinnacle point。
    """
    if abs(predicted_margin) < 1.5:
        direction = "NEUTRAL"
        stars = 1
    elif predicted_margin >= 2.5:
        direction = "FAVORITE_COVER"
        stars = min(int(predicted_margin), 5)
    elif predicted_margin <= -2.5:
        direction = "UNDERDOG_COVER"
        stars = min(int(abs(predicted_margin)), 5)
    else:
        direction = "LEAN_FAVORITE" if predicted_margin > 0 else "LEAN_UNDERDOG"
        stars = 2

    # Kelly：需要 model_home_win_pct + 市場 ML（判熱門）+ RL odds
    kelly_fractional = None
    have_ml = home_ml is not None and away_ml is not None
    have_rl_odds = home_rl_odds_ml is not None or away_rl_odds_ml is not None
    if model_home_win_pct is not None and have_ml and have_rl_odds:
        # C2: 市場熱門方判定用 American ML 較負那方（不用 predicted_margin）
        fav_is_home = home_ml < away_ml
        fav_win_pct = model_home_win_pct if fav_is_home else (1 - model_home_win_pct)
        fav_ml      = home_ml if fav_is_home else away_ml
        fav_rl_odds = home_rl_odds_ml if fav_is_home else away_rl_odds_ml
        dog_rl_odds = away_rl_odds_ml if fav_is_home else home_rl_odds_ml

        p_cover_fav = fav_win_pct * p_margin_ge_2_given_win(fav_ml)
        p_cover_dog = 1 - p_cover_fav

        # C3: Side 標籤優先用 Pinnacle snapshot 的 point（source of truth）
        if home_point is not None:
            fav_side = "HOME_-1.5" if home_point < 0 else "AWAY_-1.5"
            dog_side = "AWAY_+1.5" if home_point < 0 else "HOME_+1.5"
        else:
            fav_side = "HOME_-1.5" if fav_is_home else "AWAY_-1.5"
            dog_side = "AWAY_+1.5" if fav_is_home else "HOME_+1.5"

        kp = kelly_params or {}
        kelly_fractional = {"favorite_cover": None, "underdog_cover": None}

        if fav_rl_odds is not None:
            kf = calc_fractional_kelly(
                p_cover_fav, fav_rl_odds,
                divisor=kp.get("divisor", 4),
                cap_pct=kp.get("cap_pct", 3.0),
                unit_size_pct=kp.get("unit_size_pct", 1.0),
            )
            kf["decimal_odds"] = round(american_to_hk(fav_rl_odds) + 1, 3)
            kf["side"] = fav_side
            kelly_fractional["favorite_cover"] = kf

        if dog_rl_odds is not None:
            kf = calc_fractional_kelly(
                p_cover_dog, dog_rl_odds,
                divisor=kp.get("divisor", 4),
                cap_pct=kp.get("cap_pct", 3.0),
                unit_size_pct=kp.get("unit_size_pct", 1.0),
            )
            kf["decimal_odds"] = round(american_to_hk(dog_rl_odds) + 1, 3)
            kf["side"] = dog_side
            kelly_fractional["underdog_cover"] = kf

    return {
        "predicted_margin": round(predicted_margin, 1),
        "direction": direction,
        "stars": stars,
        "kelly_fractional": kelly_fractional,
    }
```

- [ ] **Step 4: 驗證通過**

Run: `$PYTHON -m pytest scripts/tests/ -v`
Expected: 25 passed（Task 7 之前 20 + 本 task 新增 5 個 RL 測試：`favorite_cover` / `market_favorite_when_model_disagrees` / `side_label_falls_back` / `no_odds_null` / `missing_ml_null`）

- [ ] **Step 5: Commit**

```bash
git add scripts/odds_analyzer.py scripts/tests/test_odds_analyzer_extended.py
git commit -m "feat(mlb-skill): add kelly_fractional to analyze_run_line with market-favorite logic (C2/C3)"
```

---

### Task 9: `load_closest_snapshot` helper in `predict.py` (TDD)

**Files:**
- Create: `scripts/tests/fixtures/sample_snapshot.json`
- Create: `scripts/tests/test_predict_snapshot.py`
- Modify: `scripts/predict.py`

- [ ] **Step 1: 建 snapshot fixture**

`scripts/tests/fixtures/sample_snapshot.json`：
```json
{
  "snapshot_time_utc": "2026-04-18T20:00:00+00:00",
  "snapshot_time_et": "2026-04-18 16:00 ET",
  "credits_remaining": "455",
  "credits_used": "45",
  "game_count": 2,
  "games": [
    {
      "game": "New York Mets @ Chicago Cubs",
      "away_team": "New York Mets",
      "home_team": "Chicago Cubs",
      "commence_utc": "2026-04-18T23:00:00Z",
      "commence_et": "2026-04-18 19:00 ET",
      "game_date_et": "2026-04-18",
      "bookmakers": {
        "pinnacle": {
          "title": "Pinnacle",
          "ml": {
            "Chicago Cubs": {"odds": 1.74, "implied_pct": 57.5},
            "New York Mets": {"odds": 2.24, "implied_pct": 44.6}
          },
          "ou": {
            "Over": {"odds": 1.93, "point": 8.0, "implied_pct": 51.8},
            "Under": {"odds": 1.94, "point": 8.0, "implied_pct": 51.5}
          },
          "rl": {
            "Chicago Cubs": {"odds": 1.56, "point": -1.5, "implied_pct": 64.1},
            "New York Mets": {"odds": 2.58, "point": 1.5, "implied_pct": 38.8}
          }
        }
      }
    },
    {
      "game": "Baltimore Orioles @ Cleveland Guardians",
      "away_team": "Baltimore Orioles",
      "home_team": "Cleveland Guardians",
      "commence_utc": "2026-04-18T22:11:00Z",
      "commence_et": "2026-04-18 18:11 ET",
      "game_date_et": "2026-04-18",
      "bookmakers": {
        "pinnacle": {
          "title": "Pinnacle",
          "ml": {
            "Baltimore Orioles": {"odds": 2.19, "implied_pct": 45.7},
            "Cleveland Guardians": {"odds": 1.76, "implied_pct": 56.8}
          },
          "ou": {
            "Over": {"odds": 1.93, "point": 8.0, "implied_pct": 51.8},
            "Under": {"odds": 1.94, "point": 8.0, "implied_pct": 51.5}
          },
          "rl": {
            "Baltimore Orioles": {"odds": 1.56, "point": 1.5, "implied_pct": 64.1},
            "Cleveland Guardians": {"odds": 2.58, "point": -1.5, "implied_pct": 38.8}
          }
        }
      }
    }
  ]
}
```

- [ ] **Step 2: 建第二個 snapshot fixture（早 4 小時）**

`scripts/tests/fixtures/sample_snapshot_earlier.json`（內容同上，只改 `snapshot_time_utc` 到 `2026-04-18T16:00:00+00:00`、`snapshot_time_et` 到 `12:00 ET`）：
```json
{
  "snapshot_time_utc": "2026-04-18T16:00:00+00:00",
  "snapshot_time_et": "2026-04-18 12:00 ET",
  "credits_remaining": "458",
  "credits_used": "42",
  "game_count": 1,
  "games": [
    {
      "game": "New York Mets @ Chicago Cubs",
      "away_team": "New York Mets",
      "home_team": "Chicago Cubs",
      "commence_utc": "2026-04-18T23:00:00Z",
      "commence_et": "2026-04-18 19:00 ET",
      "game_date_et": "2026-04-18",
      "bookmakers": {
        "pinnacle": {
          "title": "Pinnacle",
          "ml": {
            "Chicago Cubs": {"odds": 1.70, "implied_pct": 58.8},
            "New York Mets": {"odds": 2.30, "implied_pct": 43.5}
          },
          "ou": {},
          "rl": {}
        }
      }
    }
  ]
}
```

- [ ] **Step 3: 寫測試**

`scripts/tests/test_predict_snapshot.py`：
```python
"""Tests for snapshot loading and team-name resolution in predict.py."""
import sys
import os
import json
import shutil
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predict import load_closest_snapshot

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _make_snapshot_dir(tmpdir):
    """Copy fixtures into a temp snapshot dir with expected filename format."""
    shutil.copy(
        os.path.join(FIXTURES, "sample_snapshot.json"),
        os.path.join(tmpdir, "2026-04-18_16-00-ET.json"),
    )
    shutil.copy(
        os.path.join(FIXTURES, "sample_snapshot_earlier.json"),
        os.path.join(tmpdir, "2026-04-18_12-00-ET.json"),
    )


def test_load_closest_picks_newest_before_gametime():
    """game_start 19:00 ET → should pick 16:00 ET (newest before start)."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_snapshot_dir(tmp)
        snap = load_closest_snapshot(
            game_date_et="2026-04-18",
            game_start_utc="2026-04-18T23:00:00Z",
            snapshot_dir=tmp,
        )
        assert snap is not None
        assert snap["snapshot_time_et"] == "2026-04-18 16:00 ET"


def test_load_closest_ignores_snapshots_after_gametime():
    """game_start 15:00 ET → 16:00 ET snapshot 在 start 之後，只能用 12:00 ET。"""
    with tempfile.TemporaryDirectory() as tmp:
        _make_snapshot_dir(tmp)
        snap = load_closest_snapshot(
            game_date_et="2026-04-18",
            game_start_utc="2026-04-18T19:00:00Z",  # 15:00 ET
            snapshot_dir=tmp,
        )
        assert snap is not None
        assert snap["snapshot_time_et"] == "2026-04-18 12:00 ET"


def test_load_closest_no_snapshots_returns_none():
    """Empty snapshot dir → None."""
    with tempfile.TemporaryDirectory() as tmp:
        snap = load_closest_snapshot(
            game_date_et="2026-04-18",
            game_start_utc="2026-04-18T23:00:00Z",
            snapshot_dir=tmp,
        )
        assert snap is None


def test_load_closest_ignores_other_dates():
    """Snapshot 日期對不上 → None。"""
    with tempfile.TemporaryDirectory() as tmp:
        _make_snapshot_dir(tmp)
        snap = load_closest_snapshot(
            game_date_et="2026-04-19",  # 不是同一天
            game_start_utc="2026-04-19T23:00:00Z",
            snapshot_dir=tmp,
        )
        assert snap is None
```

- [ ] **Step 4: 驗證失敗**

Run: `$PYTHON -m pytest scripts/tests/test_predict_snapshot.py -v`
Expected: `ImportError: cannot import name 'load_closest_snapshot'`

- [ ] **Step 5: 實作 `load_closest_snapshot`**

在 `scripts/predict.py` 接近頂部（`pythagorean_runs` 函數之前）加入：
```python
import glob
import re
from datetime import datetime, timezone

_SNAPSHOT_FILENAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})-00-ET\.json$")


def load_closest_snapshot(
    game_date_et: str,
    game_start_utc: str,
    snapshot_dir: str = None,
) -> dict | None:
    """Find newest Pinnacle snapshot with snapshot_time < game_start_utc
    and containing games on game_date_et.

    Returns None if no match.
    """
    if snapshot_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        snapshot_dir = os.path.join(base, "odds_snapshots")

    if not os.path.isdir(snapshot_dir):
        return None

    try:
        game_start_dt = datetime.fromisoformat(game_start_utc.replace("Z", "+00:00"))
    except ValueError:
        return None

    candidates = []
    for path in glob.glob(os.path.join(snapshot_dir, "*.json")):
        name = os.path.basename(path)
        m = _SNAPSHOT_FILENAME_RE.match(name)
        if not m:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                snap = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        # 檢查 snapshot 時間 < game_start
        snap_time = datetime.fromisoformat(snap["snapshot_time_utc"].replace("Z", "+00:00"))
        if snap_time >= game_start_dt:
            continue

        # 檢查是否含當日 game
        has_date = any(g.get("game_date_et") == game_date_et for g in snap.get("games", []))
        if not has_date:
            continue

        candidates.append((snap_time, snap))

    if not candidates:
        return None

    # 取最新者
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]
```

- [ ] **Step 6: 驗證測試通過**

Run: `$PYTHON -m pytest scripts/tests/test_predict_snapshot.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict_snapshot.py scripts/tests/fixtures/
git commit -m "feat(mlb-skill): add load_closest_snapshot for Pinnacle odds lookup"
```

---

### Task 10: `resolve_pinnacle_odds` — snapshot → decimal odds dict (TDD)

**Files:**
- Modify: `scripts/tests/test_predict_snapshot.py`
- Modify: `scripts/predict.py`

- [ ] **Step 1: 新增失敗測試**

附加到 `test_predict_snapshot.py`：
```python
from predict import resolve_pinnacle_odds


def test_resolve_odds_matches_teams():
    """用 snapshot 找 CHC@NYM 的 Pinnacle odds。"""
    with open(os.path.join(FIXTURES, "sample_snapshot.json")) as f:
        snap = json.load(f)

    result = resolve_pinnacle_odds(snap, home_abbrev="CHC", away_abbrev="NYM")
    assert result is not None
    assert result["ml"]["home_decimal"] == 1.74
    assert result["ml"]["away_decimal"] == 2.24
    assert result["ou"]["line"] == 8.0
    assert result["ou"]["over_decimal"] == 1.93
    assert result["ou"]["under_decimal"] == 1.94
    assert result["rl"]["home_point"] == -1.5
    assert result["rl"]["home_decimal"] == 1.56
    assert result["rl"]["away_decimal"] == 2.58


def test_resolve_odds_team_mismatch_returns_none():
    """隊名對不上 → None。"""
    with open(os.path.join(FIXTURES, "sample_snapshot.json")) as f:
        snap = json.load(f)

    # Miami Marlins 不在 fixture
    result = resolve_pinnacle_odds(snap, home_abbrev="MIA", away_abbrev="ATL")
    assert result is None


def test_resolve_odds_missing_ou_market():
    """早 snapshot 只有 ML 沒 OU/RL → ml 有值但 ou/rl 為 None。"""
    with open(os.path.join(FIXTURES, "sample_snapshot_earlier.json")) as f:
        snap = json.load(f)

    result = resolve_pinnacle_odds(snap, home_abbrev="CHC", away_abbrev="NYM")
    assert result is not None
    assert result["ml"]["home_decimal"] == 1.70
    assert result["ou"] is None
    assert result["rl"] is None
```

- [ ] **Step 2: 驗證失敗**

Run: `$PYTHON -m pytest scripts/tests/test_predict_snapshot.py -v`
Expected: 3 新 test fail（ImportError）

- [ ] **Step 3: 實作 `resolve_pinnacle_odds`**

`predict.py` `load_closest_snapshot` 後方加入：
```python
# 反向 map：full team name → abbrev
_NAME_TO_ABBREV = dict(TEAM_ABBREV)  # 既有 TEAM_ABBREV 格式正是 {full_name: abbrev}


def resolve_pinnacle_odds(
    snapshot: dict,
    home_abbrev: str,
    away_abbrev: str,
) -> dict | None:
    """Extract Pinnacle decimal odds for a specific matchup.

    Returns:
        {
            "ml": {home_decimal, away_decimal},
            "ou": {line, over_decimal, under_decimal} or None,
            "rl": {home_point, home_decimal, away_point, away_decimal} or None,
            "snapshot_time_et": str,
        }
        or None if no matching game.
    """
    for g in snapshot.get("games", []):
        home_full = g.get("home_team")
        away_full = g.get("away_team")
        gh = _NAME_TO_ABBREV.get(home_full)
        ga = _NAME_TO_ABBREV.get(away_full)
        if gh != home_abbrev or ga != away_abbrev:
            continue

        pin = g.get("bookmakers", {}).get("pinnacle")
        if not pin:
            continue

        ml = pin.get("ml", {})
        ou = pin.get("ou", {})
        rl = pin.get("rl", {})

        result = {
            "snapshot_time_et": snapshot.get("snapshot_time_et"),
            "ml": None,
            "ou": None,
            "rl": None,
        }

        if home_full in ml and away_full in ml:
            result["ml"] = {
                "home_decimal": ml[home_full]["odds"],
                "away_decimal": ml[away_full]["odds"],
            }

        if "Over" in ou and "Under" in ou:
            result["ou"] = {
                "line": ou["Over"].get("point"),
                "over_decimal": ou["Over"]["odds"],
                "under_decimal": ou["Under"]["odds"],
            }

        if home_full in rl and away_full in rl:
            result["rl"] = {
                "home_point": rl[home_full].get("point"),
                "home_decimal": rl[home_full]["odds"],
                "away_point": rl[away_full].get("point"),
                "away_decimal": rl[away_full]["odds"],
            }

        return result

    return None
```

- [ ] **Step 4: 驗證通過**

Run: `$PYTHON -m pytest scripts/tests/ -v`
Expected: 32 passed（Kelly 14 + odds_analyzer_extended 11 [ML 3 + OU 3 + RL 5] + snapshot 4 [Task 9] + resolve_pinnacle 3 = 32）

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict_snapshot.py
git commit -m "feat(mlb-skill): add resolve_pinnacle_odds for snapshot → decimal odds extraction"
```

---

### Task 11: `predict.py` 新 CLI args + Kelly block wiring

**Files:**
- Modify: `scripts/predict.py`

- [ ] **Step 1: 新增 CLI args 到 argparser**

找到 `scripts/predict.py` 底部 `main()` 函數中定義 args 的區塊，加入：
```python
# Kelly sizing parameters
parser.add_argument("--kelly-divisor", type=int, default=4,
                    help="Kelly fraction divisor (default 4 = quarter-Kelly)")
parser.add_argument("--kelly-cap", type=float, default=3.0,
                    help="Hard cap per bet, %% of bankroll (default 3.0)")
parser.add_argument("--unit-size", type=float, default=1.0,
                    help="1 unit = this %% of bankroll (default 1.0)")
parser.add_argument("--no-auto-odds", action="store_true",
                    help="Skip snapshot auto-lookup; use only CLI odds overrides")
parser.add_argument("--ml-odds-home-dec", type=float, default=None,
                    help="Override: decimal odds for home ML")
parser.add_argument("--ml-odds-away-dec", type=float, default=None,
                    help="Override: decimal odds for away ML")
parser.add_argument("--ou-odds-over-dec", type=float, default=None,
                    help="Override: decimal odds for Over")
parser.add_argument("--ou-odds-under-dec", type=float, default=None,
                    help="Override: decimal odds for Under")
parser.add_argument("--rl-odds-home-dec", type=float, default=None,
                    help="Override: decimal odds for home RL")
parser.add_argument("--rl-odds-away-dec", type=float, default=None,
                    help="Override: decimal odds for away RL")
parser.add_argument("--game-index", type=int, default=None,
                    help="Doubleheader game number (1 or 2)")
```

- [ ] **Step 2: 新增 Kelly 計算函數**

**注意 merged.json 結構**：真實 merged.json 把賽事 meta 放在 `_meta` 區塊下（參見 `predict.py:340` `data.get("_meta", {})`），而非 root。`home_team` / `away_team` 存的是 abbrev（"CHC" / "NYM"）。`game_date` 是 MLB API `gameDate` 的 **UTC ISO** 字串（C1 fix：不再 `[:10]` 切 UTC 當 ET）。

**簽章新增 (I1)**：加 `final_ml_rec / final_ou_rec / final_rl_rec` 三參數，接受 `predict.py main()` guardrail 算出的最終推薦（見 `predict.py:407-468`）；PASS 市場的 Kelly 強制 null。

**檔案頂部 imports 補齊**：確認 `predict.py` 已 import `re`、`os.path`、`datetime.timezone`、`datetime.timedelta`（task 9 Step 5 已加 `import glob, re`，Task 11 此處補 `from datetime import timezone, timedelta`）。

`predict.py` 在 `main()` 之前加入：
```python
from datetime import timezone, timedelta

# C1: ET timezone（對齊 fetch_odds.py:21 — MLB 球季 EDT = UTC-4）
_ET_TZ = timezone(timedelta(hours=-4))
_ANALYSIS_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")


def compute_kelly_block(
    args,
    merged: dict,
    ml_prediction: dict | None,
    formula_prediction: dict,
    final_ml_rec: str,
    final_ou_rec: str,
    final_rl_rec: str,
) -> dict | None:
    """Build the kelly block for prediction.json.

    I1: PASS markets → kelly.{market} = None (kelly must align with D1-D5 guardrail).
    C1: ET date extracted from analysis-data/YYYY-MM-DD/ path (fallback: UTC→ET).
    C3: Pass Pinnacle rl.home_point into analyze_run_line for truthful side labeling.

    Returns dict with {snapshot_source, snapshot_time_et, params, ml, ou, rl, warnings}.
    """
    from odds_analyzer import (
        analyze_moneyline, analyze_over_under, analyze_run_line,
        decimal_to_american,
    )

    warnings = []
    meta = merged.get("_meta", {})
    home_abbrev = meta.get("home_team") or "HOME"
    away_abbrev = meta.get("away_team") or "AWAY"
    game_date_iso = meta.get("game_date") or ""  # UTC ISO 或 ""

    # === C1: ET 日期取得 ===
    # 主要來源：args.game_data 路徑的 analysis-data/YYYY-MM-DD/ 段（convention = ET）
    game_date_et = None
    if args.game_data:
        for part in os.path.normpath(args.game_data).split(os.sep):
            if _ANALYSIS_DATE_RE.match(part):
                game_date_et = part
                break
    # Fallback：UTC ISO → ET 轉換
    if not game_date_et and game_date_iso:
        try:
            utc_dt = datetime.fromisoformat(game_date_iso.replace("Z", "+00:00"))
            game_date_et = utc_dt.astimezone(_ET_TZ).strftime("%Y-%m-%d")
        except ValueError:
            pass
    # game_start_utc 直接用 UTC ISO（給 load_closest_snapshot 做時間比較）
    game_start_utc = game_date_iso if "T" in game_date_iso else None

    # === I1: Guardrail PASS 對齊 ===
    ml_is_pass = final_ml_rec == "PASS"
    ou_is_pass = final_ou_rec == "PASS"
    rl_is_pass = final_rl_rec == "PASS"
    if ml_is_pass:
        warnings.append("ml_guardrail_pass")
    if ou_is_pass:
        warnings.append("ou_guardrail_pass")
    if rl_is_pass:
        warnings.append("rl_guardrail_pass")

    # 1) Snapshot auto-lookup
    snap_odds = None
    snap_source = None
    if not args.no_auto_odds:
        snap = load_closest_snapshot(game_date_et, game_start_utc) if game_date_et and game_start_utc else None
        if snap:
            snap_odds = resolve_pinnacle_odds(
                snap, home_abbrev, away_abbrev,
                game_index=args.game_index,  # doubleheader support (ValueError surfaces)
            )
            snap_source = snap.get("snapshot_time_et")
            if snap_odds is None:
                warnings.append(f"team_name_mismatch: {home_abbrev} vs {away_abbrev}")
        else:
            warnings.append("no_matching_snapshot")

    # 2) CLI overrides take precedence
    def _pick(override_dec, snap_value):
        if override_dec is not None:
            return decimal_to_american(override_dec), override_dec
        if snap_value is not None:
            return decimal_to_american(snap_value), snap_value
        return None, None

    s_ml = (snap_odds or {}).get("ml") or {}
    s_ou = (snap_odds or {}).get("ou") or {}
    s_rl = (snap_odds or {}).get("rl") or {}

    ml_home_ml, ml_home_dec = _pick(args.ml_odds_home_dec, s_ml.get("home_decimal"))
    ml_away_ml, ml_away_dec = _pick(args.ml_odds_away_dec, s_ml.get("away_decimal"))
    ou_over_ml, ou_over_dec = _pick(args.ou_odds_over_dec, s_ou.get("over_decimal"))
    ou_under_ml, ou_under_dec = _pick(args.ou_odds_under_dec, s_ou.get("under_decimal"))
    ou_line = s_ou.get("line") if s_ou else None
    rl_home_ml, rl_home_dec = _pick(args.rl_odds_home_dec, s_rl.get("home_decimal"))
    rl_away_ml, rl_away_dec = _pick(args.rl_odds_away_dec, s_rl.get("away_decimal"))
    # C3: home_point 從 snapshot 取（Pinnacle ±1.5 事實）
    rl_home_point = s_rl.get("home_point") if s_rl else None

    kelly_params = {
        "divisor": args.kelly_divisor,
        "cap_pct": args.kelly_cap,
        "unit_size_pct": args.unit_size,
    }

    # 3) 若完全沒 odds → 回 null kelly block + warnings
    have_any = any([ml_home_dec, ml_away_dec, ou_over_dec, ou_under_dec, rl_home_dec, rl_away_dec])
    if not have_any:
        warnings.append("no_odds_available")
        return {
            "snapshot_source": None,
            "snapshot_time_et": None,
            "params": kelly_params,
            "ml": None, "ou": None, "rl": None,
            "warnings": warnings,
        }

    out = {
        "snapshot_source": snap_source,
        "snapshot_time_et": snap_source,
        "params": kelly_params,
        "ml": None, "ou": None, "rl": None,
        "warnings": warnings,
    }

    # ML Kelly：僅用 XGBoost model prob；PASS 時強制 null (I1)
    # 注意：ml_prediction["home_win_pct"] 是百分比 (0-100)，Kelly 公式需要 fraction (0-1)
    model_p_home = None
    if ml_prediction is not None:
        pct = ml_prediction.get("home_win_pct")
        if pct is not None:
            model_p_home = pct / 100.0
    if (not ml_is_pass
            and model_p_home is not None
            and ml_home_ml is not None and ml_away_ml is not None):
        ml_res = analyze_moneyline(ml_home_ml, ml_away_ml, model_p_home, kelly_params)
        kf = ml_res["kelly_fractional"]
        out["ml"] = {
            "direction": kf["direction"],
            "decimal_odds": ml_home_dec if kf["direction"] == "HOME" else ml_away_dec,
            "raw_kelly_pct": kf["raw_kelly_pct"],
            "fractional_pct": kf["fractional_pct"],
            "capped_pct": kf["capped_pct"],
            "units": kf["units"],
        }

    # OU Kelly；PASS 時強制 null (I1)
    predicted_total = formula_prediction.get("total")
    if (not ou_is_pass
            and predicted_total is not None and ou_line is not None
            and (ou_over_ml or ou_under_ml)):
        ou_res = analyze_over_under(ou_line, predicted_total, ou_over_ml, ou_under_ml, kelly_params)
        kf = ou_res["kelly_fractional"]
        if kf:
            over_block = kf["over"] and {
                "decimal_odds": ou_over_dec,
                "raw_kelly_pct": kf["over"]["raw_kelly_pct"],
                "fractional_pct": kf["over"]["fractional_pct"],
                "capped_pct": kf["over"]["capped_pct"],
                "units": kf["over"]["units"],
            }
            under_block = kf["under"] and {
                "decimal_odds": ou_under_dec,
                "raw_kelly_pct": kf["under"]["raw_kelly_pct"],
                "fractional_pct": kf["under"]["fractional_pct"],
                "capped_pct": kf["under"]["capped_pct"],
                "units": kf["under"]["units"],
            }
            out["ou"] = {
                "direction": ou_res["direction"],
                "line": ou_line,
                "over": over_block,
                "under": under_block,
            }

    # RL Kelly；PASS 時強制 null (I1)；C3: 傳 home_point
    predicted_margin = formula_prediction.get("margin")
    if (not rl_is_pass
            and predicted_margin is not None and model_p_home is not None
            and ml_home_ml is not None and ml_away_ml is not None
            and (rl_home_ml or rl_away_ml)):
        rl_res = analyze_run_line(
            predicted_margin, model_p_home,
            home_ml=ml_home_ml, away_ml=ml_away_ml,
            home_rl_odds_ml=rl_home_ml, away_rl_odds_ml=rl_away_ml,
            home_point=rl_home_point,   # C3: Pinnacle source-of-truth
            kelly_params=kelly_params,
        )
        kf = rl_res["kelly_fractional"]
        if kf:
            out["rl"] = {
                "favorite_side": (kf.get("favorite_cover") or {}).get("side"),
                "favorite": kf.get("favorite_cover"),
                "underdog": kf.get("underdog_cover"),
            }

    return out
```

- [ ] **Step 3: 呼叫 `compute_kelly_block` 於 `--save` 流程**

**精確位置**：`scripts/predict.py` 內 `main()` 函數，`record = {...}` dict（當前約 line 470）**之前**、且**在 `final_ml_rec / final_ou_rec / final_rl_rec` 都已算出之後**（`predict.py:407-468` guardrail 完成處）加入 Kelly 計算。

在 `record = {` 行之前加入：
```python
# === Kelly Sizing 計算（I1: 對齊 guardrail；I4: tighten except） ===
kelly_block = None
try:
    kelly_block = compute_kelly_block(
        args, data, ml_pred, formula_pred,
        final_ml_rec=final_ml_rec,
        final_ou_rec=final_ou_rec,
        final_rl_rec=final_rl_rec,
    )
except (KeyError, IOError, json.JSONDecodeError) as e:
    # Data / IO issues → 非致命，kelly block 為 None，主流程繼續
    print(f"⚠️ Kelly computation failed: {e}", file=sys.stderr)
    kelly_block = None
# ValueError (doubleheader 無 --game-index / 壞 decimal odds) 故意不吞 →
# 使用者看到錯誤並加對應 CLI arg 重跑
```

然後在 `record = {...}` dict 裡、`"verified": False,` 那行之前加入：
```python
            "kelly": kelly_block,
```

**變數對應關係**（`predict.py` main() 裡的既有變數名）：
- `args` → 既有 argparse 物件
- `data` → 既有從 merged.json 讀入的 dict（參見 line 340 附近 `data.get("_meta", {})`）
- `ml_pred` → 既有 XGBoost 預測 dict（參見 line 255）
- `formula_pred` → 既有 Log5 / formula dict（參見 line 258）
- `final_ml_rec` → 既有 guardrail 結果（參見 line 410-413）
- `final_ou_rec` → 既有 guardrail 結果（參見 line 425-452）
- `final_rl_rec` → 既有 guardrail 結果（參見 line 455-468）

這些變數都已在 `--save` 區塊內作用域可見，不用額外傳參。**注意**：呼叫位置必須在 guardrail 三個 `final_*_rec` 都算完之後（I1 依賴這些值）。

- [ ] **Step 4: 快速整合 smoke test**

Run:
```bash
$PYTHON scripts/predict.py --test
```
Expected: 既有 test 模式不受影響，回傳 OK。

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py
git commit -m "feat(mlb-skill): wire Kelly block into predict.py --save flow"
```

---

### Task 12: Doubleheader handling + fallback matrix

**Files:**
- Modify: `scripts/predict.py`（`load_closest_snapshot` / `resolve_pinnacle_odds`）
- Modify: `scripts/tests/test_predict_snapshot.py`

- [ ] **Step 1: 寫 DH 測試**

附加到 `test_predict_snapshot.py`：
```python
from predict import resolve_pinnacle_odds


def test_resolve_odds_doubleheader_without_index_errors():
    """同日同兩隊出現 2 場 → 需要 --game-index，未指定應 raise."""
    snap = {
        "snapshot_time_et": "2026-04-18 16:00 ET",
        "games": [
            {
                "game": "NYM @ CHC",
                "home_team": "Chicago Cubs",
                "away_team": "New York Mets",
                "commence_et": "2026-04-18 13:00 ET",
                "game_date_et": "2026-04-18",
                "bookmakers": {"pinnacle": {"ml": {
                    "Chicago Cubs": {"odds": 1.80}, "New York Mets": {"odds": 2.10},
                }, "ou": {}, "rl": {}}},
            },
            {
                "game": "NYM @ CHC",
                "home_team": "Chicago Cubs",
                "away_team": "New York Mets",
                "commence_et": "2026-04-18 19:00 ET",
                "game_date_et": "2026-04-18",
                "bookmakers": {"pinnacle": {"ml": {
                    "Chicago Cubs": {"odds": 1.75}, "New York Mets": {"odds": 2.20},
                }, "ou": {}, "rl": {}}},
            },
        ],
    }
    import pytest
    with pytest.raises(ValueError, match="doubleheader"):
        resolve_pinnacle_odds(snap, home_abbrev="CHC", away_abbrev="NYM")


def test_resolve_odds_doubleheader_with_index():
    """--game-index 指定 G2 → 取第二場 (19:00)。"""
    snap = {
        "snapshot_time_et": "2026-04-18 16:00 ET",
        "games": [
            {
                "game": "NYM @ CHC (G1)",
                "home_team": "Chicago Cubs",
                "away_team": "New York Mets",
                "commence_et": "2026-04-18 13:00 ET",
                "game_date_et": "2026-04-18",
                "bookmakers": {"pinnacle": {"ml": {
                    "Chicago Cubs": {"odds": 1.80}, "New York Mets": {"odds": 2.10},
                }, "ou": {}, "rl": {}}},
            },
            {
                "game": "NYM @ CHC (G2)",
                "home_team": "Chicago Cubs",
                "away_team": "New York Mets",
                "commence_et": "2026-04-18 19:00 ET",
                "game_date_et": "2026-04-18",
                "bookmakers": {"pinnacle": {"ml": {
                    "Chicago Cubs": {"odds": 1.75}, "New York Mets": {"odds": 2.20},
                }, "ou": {}, "rl": {}}},
            },
        ],
    }
    res = resolve_pinnacle_odds(snap, home_abbrev="CHC", away_abbrev="NYM", game_index=2)
    assert res["ml"]["home_decimal"] == 1.75  # G2
```

- [ ] **Step 2: 驗證失敗**

Run: `$PYTHON -m pytest scripts/tests/test_predict_snapshot.py -v`
Expected: 2 個新 test fail

- [ ] **Step 3: 擴充 `resolve_pinnacle_odds` 加 `game_index`**

替換 `resolve_pinnacle_odds` 的主迴圈邏輯：
```python
def resolve_pinnacle_odds(
    snapshot: dict,
    home_abbrev: str,
    away_abbrev: str,
    game_index: int = None,
) -> dict | None:
    """Extract Pinnacle decimal odds. For doubleheaders, game_index (1 or 2) required."""
    matches = []
    for g in snapshot.get("games", []):
        home_full = g.get("home_team")
        away_full = g.get("away_team")
        gh = _NAME_TO_ABBREV.get(home_full)
        ga = _NAME_TO_ABBREV.get(away_full)
        if gh != home_abbrev or ga != away_abbrev:
            continue
        matches.append(g)

    if not matches:
        return None

    if len(matches) > 1:
        if game_index is None:
            raise ValueError(
                f"doubleheader detected for {away_abbrev}@{home_abbrev}; "
                f"pass game_index=1 or 2"
            )
        # 按 commence_et 排序，game_index 1 基底
        matches.sort(key=lambda g: g.get("commence_et", ""))
        if game_index < 1 or game_index > len(matches):
            raise ValueError(f"game_index {game_index} out of range (have {len(matches)} games)")
        g = matches[game_index - 1]
    else:
        g = matches[0]

    pin = g.get("bookmakers", {}).get("pinnacle")
    if not pin:
        return None

    ml = pin.get("ml", {})
    ou = pin.get("ou", {})
    rl = pin.get("rl", {})

    home_full = g["home_team"]
    away_full = g["away_team"]

    result = {
        "snapshot_time_et": snapshot.get("snapshot_time_et"),
        "ml": None,
        "ou": None,
        "rl": None,
    }

    if home_full in ml and away_full in ml:
        result["ml"] = {
            "home_decimal": ml[home_full]["odds"],
            "away_decimal": ml[away_full]["odds"],
        }

    if "Over" in ou and "Under" in ou:
        result["ou"] = {
            "line": ou["Over"].get("point"),
            "over_decimal": ou["Over"]["odds"],
            "under_decimal": ou["Under"]["odds"],
        }

    if home_full in rl and away_full in rl:
        result["rl"] = {
            "home_point": rl[home_full].get("point"),
            "home_decimal": rl[home_full]["odds"],
            "away_point": rl[away_full].get("point"),
            "away_decimal": rl[away_full]["odds"],
        }

    return result
```

- [ ] **Step 4: 把 `--game-index` 傳入 `compute_kelly_block`**

修改 `compute_kelly_block` 裡 `resolve_pinnacle_odds` 呼叫：
```python
snap_odds = resolve_pinnacle_odds(
    snap, home_abbrev, away_abbrev,
    game_index=args.game_index,
)
```

- [ ] **Step 5: 驗證所有測試通過**

Run: `$PYTHON -m pytest scripts/tests/ -v`
Expected: 34 passed（Task 10 結束 32 + DH 2 = 34）

- [ ] **Step 6: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict_snapshot.py
git commit -m "feat(mlb-skill): handle doubleheader via --game-index arg in snapshot resolver"
```

---

### Task 13: End-to-end integration test with real snapshot

**Files:**
- Modify: `scripts/tests/test_predict_snapshot.py`

- [ ] **Step 1: 建 minimal merged.json fixture**

**注意**：真 merged.json 把賽事 meta 放在 `_meta` 區塊下，fixture 必須照此結構，否則 `compute_kelly_block` 取不到 meta。

`scripts/tests/fixtures/sample_merged.json`：
```json
{
  "_meta": {
    "home_team": "CHC",
    "away_team": "NYM",
    "game_date": "2026-04-18T23:00:00Z",
    "home_sp": "Test Pitcher A",
    "away_sp": "Test Pitcher B",
    "home_sp_starts": 10,
    "away_sp_starts": 10,
    "venue": "Wrigley Field"
  },
  "home_starter_fip": 3.80,
  "home_starter_k_bb": 15.0,
  "home_starter_whip": 1.25,
  "away_starter_fip": 4.10,
  "away_starter_k_bb": 12.0,
  "away_starter_whip": 1.30,
  "home_batting_xwoba": 0.335,
  "home_batting_ops": 0.755,
  "home_batting_k_pct": 22.0,
  "away_batting_xwoba": 0.320,
  "away_batting_ops": 0.725,
  "away_batting_k_pct": 24.0,
  "home_bullpen_era": 3.80,
  "away_bullpen_era": 4.20,
  "home_recent_rs": 4.8,
  "home_recent_ra": 4.2,
  "away_recent_rs": 4.3,
  "away_recent_ra": 4.5,
  "home_season_rs": 4.5,
  "home_season_ra": 4.4,
  "away_season_rs": 4.2,
  "away_season_ra": 4.3,
  "park_factor": 100
}
```

- [ ] **Step 2: 寫整合測試**

附加到 `test_predict_snapshot.py`。新增的 helper `_make_args` 用於產生標準 argparse Namespace；4 個測試涵蓋 happy-path + 3 個 critical bug (C1/C2+C3/I1)：

```python
import argparse


def _make_args(game_data_path, **overrides):
    """Build argparse.Namespace with all defaults compute_kelly_block expects."""
    ns = argparse.Namespace(
        game_data=str(game_data_path),
        kelly_divisor=4, kelly_cap=3.0, unit_size=1.0,
        no_auto_odds=False, game_index=None,
        ml_odds_home_dec=None, ml_odds_away_dec=None,
        ou_odds_over_dec=None, ou_odds_under_dec=None,
        rl_odds_home_dec=None, rl_odds_away_dec=None,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_end_to_end_predict_with_snapshot(tmp_path):
    """Happy path: 路徑含 ET 日期 → 抓 snapshot → kelly block 完整。"""
    # merged.json 放在含 ET 日期段的路徑下（模擬 analysis-data/2026-04-18/...）
    game_dir = tmp_path / "2026-04-18" / "NYM@CHC"
    game_dir.mkdir(parents=True)
    merged_path = game_dir / "merged.json"
    shutil.copy(os.path.join(FIXTURES, "sample_merged.json"), merged_path)

    snap_dir = tmp_path / "odds_snapshots"
    snap_dir.mkdir()
    shutil.copy(
        os.path.join(FIXTURES, "sample_snapshot.json"),
        snap_dir / "2026-04-18_16-00-ET.json",
    )

    from predict import compute_kelly_block
    import predict
    orig = predict.load_closest_snapshot
    predict.load_closest_snapshot = lambda gde, gsu, snapshot_dir=None: orig(gde, gsu, snapshot_dir=str(snap_dir))

    try:
        with open(merged_path) as f:
            merged = json.load(f)
        ml_pred = {"home_win_pct": 60.0}
        formula_pred = {"total": 9.5, "margin": 0.8}
        args = _make_args(merged_path)
        kelly_block = compute_kelly_block(
            args, merged, ml_pred, formula_pred,
            final_ml_rec="CHC", final_ou_rec="OVER", final_rl_rec="PASS",
        )
    finally:
        predict.load_closest_snapshot = orig

    assert kelly_block is not None
    assert kelly_block["ml"] is not None
    assert kelly_block["ml"]["raw_kelly_pct"] > 0
    assert kelly_block["ml"]["capped_pct"] <= 3.0
    assert kelly_block["ou"] is not None
    assert kelly_block["ou"]["line"] == 8.0
    # RL final_rl_rec=PASS → null + warning (I1)
    assert kelly_block["rl"] is None
    assert "rl_guardrail_pass" in kelly_block["warnings"]


def test_c1_west_coast_late_game_finds_snapshot(tmp_path):
    """C1 regression: UTC 2026-04-19T02:00:00Z（ET 22:00 前一天）應仍找到 ET 2026-04-18 的 snapshot。

    舊 bug：game_date_iso[:10] = "2026-04-19"，但 snapshot.game_date_et = "2026-04-18" → 對不上 → kelly null。
    新邏輯：路徑 analysis-data/2026-04-18/ 是 source of truth，能找到 snapshot。
    """
    # 模擬西岸晚場：merged.json 放在 2026-04-18 ET 資料夾，但 _meta.game_date 是 UTC 隔日
    game_dir = tmp_path / "2026-04-18" / "LAD@SF"
    game_dir.mkdir(parents=True)
    merged_path = game_dir / "merged.json"
    with open(os.path.join(FIXTURES, "sample_merged.json")) as f:
        merged = json.load(f)
    merged["_meta"]["game_date"] = "2026-04-19T02:00:00Z"  # UTC (ET 2026-04-18 22:00)
    merged["_meta"]["home_team"] = "CHC"  # 沿用 fixture 裡的隊伍
    merged["_meta"]["away_team"] = "NYM"
    with open(merged_path, "w") as f:
        json.dump(merged, f)

    snap_dir = tmp_path / "odds_snapshots"
    snap_dir.mkdir()
    # Snapshot 存成 ET 2026-04-18 的檔案
    shutil.copy(
        os.path.join(FIXTURES, "sample_snapshot.json"),
        snap_dir / "2026-04-18_20-00-ET.json",
    )
    # Snapshot 內部 game_date_et = "2026-04-18"（fixture 已如此設定）

    from predict import compute_kelly_block
    import predict
    orig = predict.load_closest_snapshot
    predict.load_closest_snapshot = lambda gde, gsu, snapshot_dir=None: orig(gde, gsu, snapshot_dir=str(snap_dir))

    try:
        args = _make_args(merged_path)
        kelly_block = compute_kelly_block(
            args, merged,
            ml_prediction={"home_win_pct": 60.0},
            formula_prediction={"total": 9.5, "margin": 0.8},
            final_ml_rec="CHC", final_ou_rec="OVER", final_rl_rec="PASS",
        )
    finally:
        predict.load_closest_snapshot = orig

    # 若 C1 未修，snapshot 會對不上 → ml is None；修好後應有值
    assert kelly_block["ml"] is not None, "C1 bug: west-coast late game snapshot not found"
    assert kelly_block["snapshot_time_et"] is not None
    assert "no_matching_snapshot" not in kelly_block["warnings"]


def test_c2_c3_model_market_split_uses_market_favorite(tmp_path):
    """C2/C3 regression: model 與 market 熱門方分歧時，RL Kelly 必須查 market bucket。

    Setup: model 覺得 home 贏（predicted_margin=+0.5），但 market 覺得 away 熱門（ml=-150）。
    舊 bug：fav_is_home = +0.5 >= 0 = True → 查 home_ml=+140 bucket（錯），favorite_cover 對應 home_rl_odds（錯）。
    新邏輯：fav_is_home = home_ml(+140) < away_ml(-150) = False → 查 away_ml=-150 bucket，favorite_cover 對應 away_rl_odds。
    """
    game_dir = tmp_path / "2026-04-18" / "CHC@NYM"
    game_dir.mkdir(parents=True)
    merged_path = game_dir / "merged.json"
    with open(os.path.join(FIXTURES, "sample_merged.json")) as f:
        merged = json.load(f)
    # 使用 fixture 的 home=CHC / away=NYM；覆寫 _meta 確認對齊 snapshot
    with open(merged_path, "w") as f:
        json.dump(merged, f)

    # 客製 snapshot：home=CHC 冷門 (+140), away=NYM 熱門 (-150), home_point=+1.5
    snap = {
        "snapshot_time_utc": "2026-04-18T20:00:00+00:00",
        "snapshot_time_et": "2026-04-18 16:00 ET",
        "games": [{
            "game": "New York Mets @ Chicago Cubs",
            "away_team": "New York Mets",
            "home_team": "Chicago Cubs",
            "commence_utc": "2026-04-18T23:00:00Z",
            "commence_et": "2026-04-18 19:00 ET",
            "game_date_et": "2026-04-18",
            "bookmakers": {"pinnacle": {
                "title": "Pinnacle",
                "ml": {
                    "Chicago Cubs": {"odds": 2.40, "implied_pct": 41.7},    # +140 underdog
                    "New York Mets": {"odds": 1.67, "implied_pct": 59.9},   # -150 favorite
                },
                "ou": {"Over": {"odds": 1.91, "point": 8.5}, "Under": {"odds": 1.95, "point": 8.5}},
                "rl": {
                    "Chicago Cubs": {"odds": 3.00, "point": 1.5},    # home +1.5 @ +200
                    "New York Mets": {"odds": 1.385, "point": -1.5}, # away -1.5 @ -260
                },
            }},
        }],
    }
    snap_dir = tmp_path / "odds_snapshots"
    snap_dir.mkdir()
    with open(snap_dir / "2026-04-18_16-00-ET.json", "w") as f:
        json.dump(snap, f)

    from predict import compute_kelly_block
    import predict
    orig = predict.load_closest_snapshot
    predict.load_closest_snapshot = lambda gde, gsu, snapshot_dir=None: orig(gde, gsu, snapshot_dir=str(snap_dir))

    try:
        args = _make_args(merged_path)
        kelly_block = compute_kelly_block(
            args, merged,
            ml_prediction={"home_win_pct": 55.0},     # model 覺得 home 小勝
            formula_prediction={"total": 9.0, "margin": 0.5},  # predicted_margin=+0.5
            final_ml_rec="CHC", final_ou_rec="OVER",
            final_rl_rec="NYM",  # 非 PASS，允許算 RL
        )
    finally:
        predict.load_closest_snapshot = orig

    rl = kelly_block["rl"]
    assert rl is not None
    # C3: favorite_side 來自 Pinnacle home_point=+1.5 → fav_side = "AWAY_-1.5"
    assert rl["favorite_side"] == "AWAY_-1.5"
    # C2: favorite_cover 的 decimal_odds 應該是 away_rl_odds = 1.385（-260）
    assert rl["favorite"]["decimal_odds"] == pytest.approx(1.385, abs=0.01)
    # favorite_cover raw Kelly 應為 0（market favorite prob ≈ 72% implied，model 給 away win 僅 45% × 0.615 ≈ 28%）
    assert rl["favorite"]["raw_kelly_pct"] == 0
    # underdog (home +1.5 @ +200) 有強正 edge
    assert rl["underdog"]["raw_kelly_pct"] > 0


def test_i1_divergent_forces_ml_kelly_null(tmp_path):
    """I1 regression: final_ml_rec=PASS → kelly.ml 必為 null + warning 紀錄。"""
    game_dir = tmp_path / "2026-04-18" / "NYM@CHC"
    game_dir.mkdir(parents=True)
    merged_path = game_dir / "merged.json"
    shutil.copy(os.path.join(FIXTURES, "sample_merged.json"), merged_path)

    snap_dir = tmp_path / "odds_snapshots"
    snap_dir.mkdir()
    shutil.copy(
        os.path.join(FIXTURES, "sample_snapshot.json"),
        snap_dir / "2026-04-18_16-00-ET.json",
    )

    from predict import compute_kelly_block
    import predict
    orig = predict.load_closest_snapshot
    predict.load_closest_snapshot = lambda gde, gsu, snapshot_dir=None: orig(gde, gsu, snapshot_dir=str(snap_dir))

    try:
        with open(merged_path) as f:
            merged = json.load(f)
        args = _make_args(merged_path)
        # ml_rec = PASS 模擬 DIVERGENT / INSUFFICIENT_SAMPLE 方向分歧
        kelly_block = compute_kelly_block(
            args, merged,
            ml_prediction={"home_win_pct": 60.0},  # 有正 edge，但被 guardrail PASS 掉
            formula_prediction={"total": 9.5, "margin": 0.8},
            final_ml_rec="PASS", final_ou_rec="OVER", final_rl_rec="PASS",
        )
    finally:
        predict.load_closest_snapshot = orig

    assert kelly_block["ml"] is None, "I1: PASS 市場的 kelly 必為 null"
    assert "ml_guardrail_pass" in kelly_block["warnings"]
    assert "rl_guardrail_pass" in kelly_block["warnings"]
    # OU 非 PASS → 仍應有值
    assert kelly_block["ou"] is not None
```

- [ ] **Step 3: 驗證新增的 4 個整合測試**

Run: `$PYTHON -m pytest scripts/tests/test_predict_snapshot.py -v -k "end_to_end or c1_ or c2_c3_ or i1_"`
Expected: 4 passed (`end_to_end`, `c1_west_coast_late_game_finds_snapshot`, `c2_c3_model_market_split_uses_market_favorite`, `i1_divergent_forces_ml_kelly_null`)

- [ ] **Step 4: 跑全部測試**

Run: `$PYTHON -m pytest scripts/tests/ -v`
Expected: 38 passed（Task 12 結束 34 passed：Kelly 14 + odds_analyzer_extended 11 [原 9 + RL 新增 2] + snapshot 9；Task 13 新增 4 = 38）

- [ ] **Step 5: Commit**

```bash
git add scripts/tests/fixtures/sample_merged.json scripts/tests/test_predict_snapshot.py
git commit -m "test(mlb-skill): add Kelly integration tests covering C1/C2/C3/I1 fixes"
```

---

### Task 14: 更新 `reference/prediction.md` — 新 Kelly Sizing 章節

**Files:**
- Modify: `reference/prediction.md`

- [ ] **Step 1: 定位插入點**

Run: `grep -n "## 預測紀錄存放位置" reference/prediction.md`
Expected: 印出行號（當前約 236 行）。新章節要插入這行**之前**，且自己前後各有一個水平分隔符 `---`。

- [ ] **Step 2: 插入新章節**

使用 Edit 工具，把 `"## 預測紀錄存放位置"` 那一行替換為「新章節內容 + 原本的章節標題」。完整插入內容：
```markdown
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
- **Snapshot 4h 延遲**：快速變盤（steam move）下 Kelly 可能在推薦與下注之間過時。V1 接受；P2 M5 將加 line movement + CLV tracking，屆時以實證資料量化延遲對 ROI 的影響。
```

- [ ] **Step 3: 驗證 markdown 無語法壞掉**

Run: `$PYTHON -c "import pathlib; print(len(pathlib.Path('reference/prediction.md').read_text(encoding='utf-8')))"`
Expected: 應印出新的總字數（比原本多 ~2000 字元）

- [ ] **Step 4: Commit**

```bash
git add reference/prediction.md
git commit -m "docs(mlb-skill): add Kelly Sizing section to prediction.md"
```

---

### Task 15: 更新 `reference/output-format.md` 和 `reference/workflow.md`

**Files:**
- Modify: `reference/output-format.md`
- Modify: `reference/workflow.md`

- [ ] **Step 1: 擴充 `reference/output-format.md` 的 TL;DR 代碼區塊**

現有 TL;DR 包在 ` ``` ... ``` ` 代碼區塊內（當前 line 5-17）。結尾是 `| Run Line | ✅/⚠️/PASS | ⭐⭐⭐ | ... |` 然後 ` ``` ` 閉合。

用 Edit 工具，把：
```
| Run Line | ✅/⚠️/PASS | ⭐⭐⭐ | ... |
```
（單一行，完整匹配）替換為：
```
| Run Line | ✅/⚠️/PASS | ⭐⭐⭐ | ... |

💰 建議注碼（Quarter-Kelly, cap 3% of bankroll）
| 市場 | 方向 | 注碼 | Pinnacle odds |
|------|------|------|---------------|
| ML   | {方向} | {units}u | {decimal_odds} |
| O/U  | {方向} | {units}u | {decimal_odds} |
| RL   | {方向或 PASS} | {units}u | {decimal_odds} |
```
（保持縮排一致，結尾的 ``` 不動）

- [ ] **Step 2: 擴充 `reference/workflow.md` Phase 4.0**

Run: `grep -n "^### 4.0 執行預測腳本" reference/workflow.md`
Expected: 印出該標題行號。在此段落下方找到 `--save` 參數表結束的下一行（空行前）。

用 Edit 工具，在既有 `| --umpire-ou-rate | 若有 | 主審 Over% |` 這行下方（表格最後一列之後）插入：
```markdown

> **自動 Odds 查詢**：`predict.py --save` 會自動從 `odds_snapshots/` 撈推薦時間最近的 Pinnacle snapshot 作為 Kelly 計算來源（Kelly 區塊詳見 `reference/prediction.md` Kelly Sizing 章節）。若需手動覆寫，加 `--ml-odds-home-dec 1.83` / `--ou-odds-over-dec 1.91` / `--rl-odds-home-dec 1.56` 等 args。Doubleheader 需指定 `--game-index 1` 或 `2`。
```

- [ ] **Step 3: 驗證文件皆含新字串**

Run: `grep -c "Kelly" reference/output-format.md reference/workflow.md reference/prediction.md`
Expected: 三檔皆非 0。output-format ≥ 1，workflow ≥ 1，prediction ≥ 5（Task 14 已加 Kelly Sizing 整個章節）

- [ ] **Step 4: Commit**

```bash
git add reference/output-format.md reference/workflow.md
git commit -m "docs(mlb-skill): wire Kelly output into output-format + workflow references"
```

---

### Task 16: Acceptance run with real project data

**Files:** 無 code 修改；只驗證。

- [ ] **Step 1: 跑全部測試**

Run: `$PYTHON -m pytest scripts/tests/ -v`
Expected: 38 passed, 0 failed（Task 13 結束 38）

- [ ] **Step 2: 跑 `predict.py --test` 不失敗**

Run: `$PYTHON scripts/predict.py --test`
Expected: stdout 印 `{"test": "OK", ...}` 類似既有行為

- [ ] **Step 3: 如果有當日比賽，做真實 smoke test**

若 `analysis-data/2026-04-18/` 下有任何 `merged.json`：
```bash
# 假設存在 CHC@NYM
$PYTHON scripts/predict.py --game-data analysis-data/2026-04-18/NYM@CHC/merged.json --save \
    --ml-rec CHC --ml-stars 2 --ou-line 8.0 --ou-rec OVER --run-line-rec PASS
```
Expected: `analysis-data/2026-04-18/NYM@CHC/prediction.json` 含 `kelly` 區塊。

若當日無比賽，跳過這步。

- [ ] **Step 4: 手檢 prediction.json**

打開 `analysis-data/**/prediction.json`，確認 `kelly` 區塊：
- `params` 反映預設 `{4, 3.0, 1.0}`
- `ml.capped_pct` ≤ 3.0
- 無 snapshot → `warnings` 含 `no_matching_snapshot` 或 `no_odds_available`

- [ ] **Step 5: 更新 spec 勾 DoD 清單**

編輯 `docs/superpowers/specs/2026-04-18-p3-kelly-sizing-design.md` §12，把完成的項目從 `- [ ]` 改為 `- [x]`。

- [ ] **Step 6: 最終 commit**

```bash
git add docs/superpowers/specs/2026-04-18-p3-kelly-sizing-design.md
git commit -m "chore(mlb-skill): mark P3 Kelly sizing DoD as complete"
```

---

## 完成後後續

P3 完成後可開始 **P2（CLV 基礎設施 + M5 line movement）**。現有基建已在（`fetch_odds.py` + `odds_snapshots/` + Task 9-10 的 snapshot loader），P2 主要工作：
- Snapshot time = 推薦當下 → prediction.json 記錄「推薦時 line」
- 比賽開打前最後一個 snapshot → 回填 closing line
- 計算 CLV（cents beat closing）
- Line movement / reverse movement 訊號偵測

P2 spec 啟動前先跑 brainstorming skill（現有 infra 重點確認 → 縮小範圍）。
