# 確定性預測 (Model B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把方向/總分/信心從 AI 自由心證改成 script 確定性計算（信心 = winprob(gap)），AI 退化成敘事層；信號持久化供未來 ablation；回測改從凍結 merged.json 重算預測。

**Architecture:** 新增 `predict.py`（純函式：得分差→勝率→方向/信心）。`summary_renderer` 改填 script 數字、敘事段留 placeholder。`prepare_game` 存 `signals.json`。回測 `load.py` 改從 merged.json 用 predict() 算預測（不再 parse summary.md 的數字），達成「改公式一鍵重算」。

**Tech Stack:** Python 3.13、stdlib `statistics.NormalDist`（無需 scipy）、pytest、pandas。

參考 spec：`docs/superpowers/specs/2026-05-28-deterministic-prediction-design.md`

---

## File Structure

**新增：**
- `scripts/predict.py` — winprob 曲線 + 方向 + 信心 + 持平判定（純函式）
- `scripts/backfill_signals.py` — 為既有比賽重算補 `signals.json`
- `scripts/tests/test_predict.py`

**修改：**
- `scripts/summary_renderer.py` — `_render_expected_runs_section`（+信號=0, adjusted=base）、`_render_overall_section`（吃 prediction 填數字）、`render_summary`（算 prediction）
- `scripts/prepare_game.py` — `step_g` 寫 `signals.json`
- `scripts/lib/load.py` — 預測改用 merged.json + predict()；slice flags 改讀 signals.json
- `scripts/tests/test_summary_renderer.py` — 更新斷言（數字段不再是 placeholder）
- `scripts/tests/test_load.py` — 更新為 merged.json 驅動的預測

**常數（v1 起步值，spec §15）：** `MARGIN_SD = 4.0`、`PUSH_FLOOR = 0.53`、bucket 邊界 `0.58 / 0.67`。

---

## Task 1: predict.py — winprob 曲線 + bucket（純函式）

**Files:**
- Create: `scripts/predict.py`
- Test: `scripts/tests/test_predict.py`

- [ ] **Step 1: 寫 failing test**

```python
# scripts/tests/test_predict.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import winprob, confidence_bucket


def test_winprob_known_points():
    assert abs(winprob(0.0) - 0.500) < 0.005
    assert abs(winprob(0.81) - 0.580) < 0.005   # MEDIUM 下界
    assert abs(winprob(1.76) - 0.670) < 0.005    # HIGH 下界
    assert abs(winprob(0.30) - 0.530) < 0.005    # 持平邊界
    # 對稱性
    assert abs(winprob(-1.0) - (1 - winprob(1.0))) < 1e-9


def test_confidence_bucket_boundaries():
    assert confidence_bucket(0.55) == "LOW"
    assert confidence_bucket(0.579) == "LOW"
    assert confidence_bucket(0.58) == "MEDIUM"
    assert confidence_bucket(0.669) == "MEDIUM"
    assert confidence_bucket(0.67) == "HIGH"
    assert confidence_bucket(0.80) == "HIGH"
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `python -m pytest scripts/tests/test_predict.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'predict'`）

- [ ] **Step 3: 寫最小實作**

```python
# scripts/predict.py
"""確定性預測：得分差 → 勝率 → 方向 / 信心。

設計見 docs/superpowers/specs/2026-05-28-deterministic-prediction-design.md。
信心 = 預測那一側的單場勝率，由 winprob 曲線換算，無 AI 介入、無信號 penalty。
"""
from statistics import NormalDist

MARGIN_SD = 4.0   # 單場 run-margin 標準差，歷史 MLB 先驗，非 fit 回測樣本
PUSH_FLOOR = 0.53  # 勝率低於此 → 持平（無方向）

_NORM = NormalDist()


def winprob(gap: float) -> float:
    """P(主隊勝) = Φ(gap / S)，gap = home_score − away_score。"""
    return _NORM.cdf(gap / MARGIN_SD)


def confidence_bucket(p: float) -> str:
    """勝率 → LOW / MEDIUM / HIGH（沿用既有 _effective_confidence_bucket 邊界）。"""
    if p < 0.58:
        return "LOW"
    if p < 0.67:
        return "MEDIUM"
    return "HIGH"
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `python -m pytest scripts/tests/test_predict.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict.py
git commit -m "feat(predict): winprob curve + confidence bucket (pure fns)"
```

---

## Task 2: predict.py — predict() 方向/總分/信心/持平

**Files:**
- Modify: `scripts/predict.py`
- Test: `scripts/tests/test_predict.py`

- [ ] **Step 1: 加 failing test**

```python
# append to scripts/tests/test_predict.py
from predict import predict


def test_predict_home_favored():
    r = predict(home_score=5.5, away_score=3.0)  # gap +2.5
    assert r["direction"] == "HOME"
    assert r["total"] == 8.5
    assert abs(r["confidence_pct"] - 0.734) < 0.005
    assert r["confidence_bucket"] == "HIGH"


def test_predict_away_favored():
    r = predict(home_score=3.0, away_score=5.0)  # gap -2.0
    assert r["direction"] == "AWAY"
    assert abs(r["confidence_pct"] - 0.691) < 0.005  # winprob(+2.0)
    assert r["confidence_bucket"] == "HIGH"


def test_predict_pickem_is_push():
    r = predict(home_score=4.1, away_score=4.0)  # gap +0.1 → winprob ~0.51 < 0.53
    assert r["direction"] == "持平"
    assert r["confidence_bucket"] is None
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `python -m pytest scripts/tests/test_predict.py -k predict -v`
Expected: FAIL（`ImportError: cannot import name 'predict'`）

- [ ] **Step 3: 加實作**

```python
# append to scripts/predict.py

def predict(home_score: float, away_score: float) -> dict:
    """確定性預測。回傳 {direction, total, confidence_pct, confidence_bucket}。

    direction ∈ {HOME, AWAY, 持平}；持平時 confidence_bucket = None。
    confidence_pct 一律是「預測那一側」的勝率（持平時為較高側勝率，仍 < PUSH_FLOOR）。
    """
    gap = home_score - away_score
    p_home = winprob(gap)
    p_away = 1.0 - p_home

    if p_home >= PUSH_FLOOR:
        direction, conf = "HOME", p_home
    elif p_away >= PUSH_FLOOR:
        direction, conf = "AWAY", p_away
    else:
        direction, conf = "持平", max(p_home, p_away)

    return {
        "direction": direction,
        "total": round(home_score + away_score, 1),
        "confidence_pct": round(conf, 4),
        "confidence_bucket": confidence_bucket(conf) if direction != "持平" else None,
    }
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `python -m pytest scripts/tests/test_predict.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict.py
git commit -m "feat(predict): predict() direction/total/confidence/持平"
```

---

## Task 3: summary_renderer — 得分表 +信號=0, adjusted=base

**Files:**
- Modify: `scripts/summary_renderer.py:261-281`（`_render_expected_runs_section`）
- Test: `scripts/tests/test_summary_renderer.py`

- [ ] **Step 1: 寫 failing test**

```python
# append to scripts/tests/test_summary_renderer.py
def test_expected_runs_no_ai_placeholder():
    from summary_renderer import render_summary
    output = render_summary(_minimal_bundle(), {"home_score": 4.0, "away_score": 3.0})
    section = output.split("## 修正後預期得分", 1)[1].split("\n## ", 1)[0]
    # +信號 欄全 0、adjusted = base、該段無 AI placeholder
    assert "<!-- AI 補 -->" not in section
    assert "| HOME | 4.0 | 0 | 4.0 |" in section
    assert "| AWAY | 3.0 | 0 | 3.0 |" in section
    assert "| Total | 7.0 | 0 | 7.0 |" in section
```

（`_minimal_bundle` 已存在於該測試檔，沿用。）

- [ ] **Step 2: 跑測試確認 fail**

Run: `python -m pytest scripts/tests/test_summary_renderer.py::test_expected_runs_no_ai_placeholder -v`
Expected: FAIL（目前是 `<!-- AI 補 -->` placeholder）

- [ ] **Step 3: 改實作** — 替換 `_render_expected_runs_section` 的 return（行 269-281）

```python
    return [
        "## 修正後預期得分",
        "",
        "> v1：信號只進敘事、不進數字（+信號 欄一律 0、adjusted = base）。",
        "> 哪個信號該進數字由未來 ablation 決定（見 spec §10）。",
        "",
        "| | base (formula) | + 信號 | adjusted |",
        "|---|---|---|---|",
        f"| HOME | {home_base} | 0 | {home_base} |",
        f"| AWAY | {away_base} | 0 | {away_base} |",
        f"| Total | {total_base} | 0 | {total_base} |",
        "",
    ]
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `python -m pytest scripts/tests/test_summary_renderer.py::test_expected_runs_no_ai_placeholder -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/summary_renderer.py scripts/tests/test_summary_renderer.py
git commit -m "feat(summary): score table +信號=0 / adjusted=base (v1 no signals in numbers)"
```

---

## Task 4: summary_renderer — 整體判斷段由 script 填數字

**Files:**
- Modify: `scripts/summary_renderer.py:284-307`（`_render_overall_section` + `render_summary`）
- Test: `scripts/tests/test_summary_renderer.py`

- [ ] **Step 1: 寫 failing test**

```python
# append to scripts/tests/test_summary_renderer.py
def test_overall_section_filled_by_script():
    from summary_renderer import render_summary
    # gap +2.5 → HOME / HIGH / 73%
    output = render_summary(_minimal_bundle(), {"home_score": 5.5, "away_score": 3.0})
    overall = output.split("## 整體判斷", 1)[1]
    assert "**方向（基本面）**：HOME" in overall
    assert "**總分（基本面）**：8.5" in overall
    assert "73%" in overall and "HIGH" in overall
    # 方向/總分/信心 不再是 placeholder
    assert "AI 補 HOME / AWAY" not in overall
    # 風險 仍是 AI placeholder
    assert "**風險**：<!-- AI 補 1-4 點 -->" in overall
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `python -m pytest scripts/tests/test_summary_renderer.py::test_overall_section_filled_by_script -v`
Expected: FAIL

- [ ] **Step 3: 改實作** — `_render_overall_section` 改吃 prediction，`render_summary` 算 prediction

```python
def _render_overall_section(prediction: dict) -> list[str]:
    d = prediction["direction"]
    total = prediction["total"]
    pct = prediction["confidence_pct"]
    bucket = prediction["confidence_bucket"]
    conf_str = f"{pct*100:.0f}%（{bucket}）" if bucket else f"{pct*100:.0f}%（持平）"
    return [
        "## 整體判斷",
        "",
        f"- **方向（基本面）**：{d}",
        f"- **總分（基本面）**：{total}",
        f"- **方向信心**：{conf_str}",
        "- **風險**：<!-- AI 補 1-4 點 -->",
        "",
        "⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組",
        "ℹ️ 方向/總分/信心由 scripts/predict.py 確定性計算；AI 僅補風險敘事，不得改數字。",
    ]
```

`render_summary` 改（行 297-307）：

```python
def render_summary(bundle: dict, formula_pred: dict) -> str:
    """主入口：渲染 summary.md template，回傳 markdown 字串（不寫檔；caller 寫檔）。"""
    from predict import predict
    prediction = predict(
        home_score=formula_pred.get("home_score", 0.0),
        away_score=formula_pred.get("away_score", 0.0),
    )
    lines: list[str] = []
    lines += _render_pitcher_matchup_section(bundle)
    lines += _render_lineup_section(bundle)
    lines += _render_bullpen_section(bundle)
    lines += _render_risk_section(bundle)
    lines += _render_conditional_section(bundle)
    lines += _render_expected_runs_section(bundle, formula_pred)
    lines += _render_overall_section(prediction)
    return "\n".join(lines)
```

- [ ] **Step 4: 跑測試確認 pass + 全 renderer 測試不破**

Run: `python -m pytest scripts/tests/test_summary_renderer.py -v`
Expected: PASS（含舊測試；若有舊測試斷言整體判斷是 placeholder，更新它）

- [ ] **Step 5: Commit**

```bash
git add scripts/summary_renderer.py scripts/tests/test_summary_renderer.py
git commit -m "feat(summary): 整體判斷段由 predict.py 填數字，AI 只留風險敘事"
```

---

## Task 5: prepare_game — step_g 持久化 signals.json

**Files:**
- Modify: `scripts/prepare_game.py`（`step_g`，約行 412-419）
- Test: `scripts/tests/test_prepare_game_steps.py`

- [ ] **Step 1: 寫 failing test**

```python
# append to scripts/tests/test_prepare_game_steps.py
def test_step_g_writes_signals_json(tmp_path, monkeypatch):
    import types, sys, json
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import prepare_game

    # stub renderer + formula + signals so test 不依賴真實資料
    fake_renderer = types.ModuleType("summary_renderer")
    fake_renderer.render_summary = lambda bundle, fp: "# S <!-- AI 補 -->"
    monkeypatch.setitem(sys.modules, "summary_renderer", fake_renderer)
    fake_pred = types.ModuleType("scoring_formula")
    fake_pred.predict_with_formula = lambda merged: {"home_score": 4, "away_score": 3}
    monkeypatch.setitem(sys.modules, "scoring_formula", fake_pred)
    fake_sig = types.ModuleType("signals_lib")
    fake_sig.signals_for_bundle = lambda bundle: {"signals": [{"name": "x", "fired": False}], "fired_count": 0}
    monkeypatch.setitem(sys.modules, "signals_lib", fake_sig)

    (tmp_path / "merged.json").write_text("{}", encoding="utf-8")
    summary_path = tmp_path / "summary.md"
    prepare_game.step_g(output_dir=tmp_path, summary_path=summary_path, force=True,
                        bundle={"merged": {}})

    sig_path = tmp_path / "signals.json"
    assert sig_path.exists()
    data = json.loads(sig_path.read_text(encoding="utf-8"))
    assert "signals" in data and "fired_count" in data
```

（若 `step_g` 簽名不同，依實際簽名調整 stub 呼叫；核心斷言是 signals.json 產生且結構正確。）

- [ ] **Step 2: 跑測試確認 fail**

Run: `python -m pytest scripts/tests/test_prepare_game_steps.py::test_step_g_writes_signals_json -v`
Expected: FAIL（signals.json 不存在）

- [ ] **Step 3: 改實作** — 在 step_g 寫完 summary 後加 signals.json 持久化

```python
    # 在 step_g 內 summary_path.write_text(...) 之後加：
    from signals_lib import signals_for_bundle
    import json as _json
    signals_result = signals_for_bundle(bundle)
    (output_dir / "signals.json").write_text(
        _json.dumps(signals_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[G] signals  → {output_dir / 'signals.json'}", file=sys.stderr)
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `python -m pytest scripts/tests/test_prepare_game_steps.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_game.py scripts/tests/test_prepare_game_steps.py
git commit -m "feat(prepare): step_g 持久化 signals.json（為未來 ablation 凍結特徵）"
```

---

## Task 6: backfill_signals.py — 既有比賽補 signals.json

**Files:**
- Create: `scripts/backfill_signals.py`
- Test: `scripts/tests/test_backfill_signals.py`

- [ ] **Step 1: 寫 failing test**

```python
# scripts/tests/test_backfill_signals.py
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backfill_signals import backfill_one


def test_backfill_one_writes_signals(tmp_path, monkeypatch):
    import types
    fake_sig = types.ModuleType("signals_lib")
    fake_sig.signals_for_bundle = lambda bundle: {"signals": [], "fired_count": 0}
    monkeypatch.setitem(sys.modules, "signals_lib", fake_sig)

    (tmp_path / "merged.json").write_text("{}", encoding="utf-8")
    ok = backfill_one(tmp_path)
    assert ok is True
    assert (tmp_path / "signals.json").exists()


def test_backfill_one_skips_when_no_merged(tmp_path):
    ok = backfill_one(tmp_path)  # 無 merged.json
    assert ok is False
    assert not (tmp_path / "signals.json").exists()
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `python -m pytest scripts/tests/test_backfill_signals.py -v`
Expected: FAIL（`No module named 'backfill_signals'`）

- [ ] **Step 3: 寫實作**

```python
# scripts/backfill_signals.py
"""為既有比賽重算補 signals.json（best-effort）。

TTO3 / pitch_mix 依賴 Statcast 逐球資料，當時若未凍結則不 fire（樣本受限），
core_il / reverse_platoon / chain_break / platoon 用 lineup/roster 算，應可補回。
用法：python scripts/backfill_signals.py --month 2026-05
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _load_bundle(game_dir: Path) -> dict:
    bundle = {}
    for key, fname in [
        ("home_pitcher", "home_pitcher.json"),
        ("away_pitcher", "away_pitcher.json"),
        ("home_lineup", "home_lineup.json"),
        ("away_lineup", "away_lineup.json"),
        ("merged", "merged.json"),
    ]:
        p = game_dir / fname
        if p.exists():
            try:
                bundle[key] = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                bundle[key] = None
    return bundle


def backfill_one(game_dir: Path) -> bool:
    """重算單場 signals.json。回傳是否成功（無 merged.json → False）。"""
    if not (game_dir / "merged.json").exists():
        return False
    from signals_lib import signals_for_bundle
    bundle = _load_bundle(game_dir)
    result = signals_for_bundle(bundle)
    (game_dir / "signals.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="YYYY-MM")
    args = ap.parse_args(argv)
    data_dir = SKILL_ROOT / "analysis-data"
    done = skipped = 0
    for date_dir in sorted(data_dir.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith(args.month):
            continue
        if date_dir.name.endswith(".local-backup"):
            continue
        for game_dir in sorted(date_dir.iterdir()):
            if not game_dir.is_dir():
                continue
            if backfill_one(game_dir):
                done += 1
            else:
                skipped += 1
    print(f"backfill signals: done={done} skipped={skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑測試確認 pass**

Run: `python -m pytest scripts/tests/test_backfill_signals.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 對 5 月實跑 + commit**

```bash
python scripts/backfill_signals.py --month 2026-05
git add scripts/backfill_signals.py scripts/tests/test_backfill_signals.py "analysis-data/2026-05-*/*/signals.json"
git commit -m "feat(backfill): 既有比賽補 signals.json + 5 月實跑"
```

---

## Task 7: 回測改從 merged.json 用 predict() 算預測

**Files:**
- Modify: `scripts/lib/load.py`（`_build_row`，行 84-171）
- Test: `scripts/tests/test_load.py`

**理由：** spec §10.1「script 重算預測」。從凍結 merged.json 算 → 改公式可一鍵全月重算，不需重生成 summary（也不洗掉 AI 敘事）。slice flags 改讀 signals.json（Task 5/6 已凍結）。

- [ ] **Step 1: 寫 failing test**

```python
# append to scripts/tests/test_load.py — 用既有 fixture 風格建一個 game dir
def test_build_row_prediction_from_merged(tmp_path, monkeypatch):
    import sys, json
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib import load

    matchup = tmp_path / "2026-05-02" / "BAL@NYY"
    matchup.mkdir(parents=True)
    (matchup / "game_data.json").write_text(json.dumps({
        "game": {"gamePk": 1, "home": {"team": "New York Yankees"},
                 "away": {"team": "Baltimore Orioles"}}}), encoding="utf-8")
    (matchup / "merged.json").write_text(json.dumps({
        "home_batting_xwoba": 0.340, "away_batting_xwoba": 0.300,
        "home_starter_fip": 3.5, "away_starter_fip": 4.5, "park_factor": 100}), encoding="utf-8")
    (matchup / "signals.json").write_text(json.dumps({
        "signals": [{"name": "reverse_platoon", "fired": True, "side": "HOME"}],
        "fired_count": 1}), encoding="utf-8")
    (matchup / "result.json").write_text(json.dumps({
        "winner": "HOME", "total": 9, "home_score": 5, "away_score": 4}), encoding="utf-8")

    row = load._build_row("2026-05-02", matchup, "BAL", "NYY")
    # 預測來自 predict(merged formula)，非 parse summary
    assert row["skill_direction"] in ("HOME", "AWAY", "持平")
    assert row["skill_confidence_pct"] is not None
    assert row["has_reverse_platoon"] is True  # 來自 signals.json
```

- [ ] **Step 2: 跑測試確認 fail**

Run: `python -m pytest scripts/tests/test_load.py::test_build_row_prediction_from_merged -v`
Expected: FAIL（目前 `_build_row` 從 summary.md parse，且需要 summary.md 存在）

- [ ] **Step 3: 改 `_build_row`** — 預測改用 merged.json + predict()，flags 改讀 signals.json

在 `load.py` 頂部 import 區加：
```python
from predict import predict as _predict
from scoring_formula import predict_with_formula as _formula
```
（並確保 `SCRIPT_DIR`（=scripts/）在 sys.path；load.py 已 `sys.path` 設定。）

把 `_build_row` 中「Parse summary」與 skill_direction/total/confidence 來源段，改為：
```python
    merged = _read_game_data(matchup_dir)  # 既有：讀 game_data.json
    merged_json = _read_json(matchup_dir / "merged.json")
    if merged_json is None:
        return None
    formula_pred = _formula(merged_json)
    pred = _predict(formula_pred["home_score"], formula_pred["away_score"])

    skill_direction = pred["direction"]
    skill_total = pred["total"]
    skill_conf_pct = pred["confidence_pct"]
    skill_conf = pred["confidence_bucket"]
    parse_failed = False  # 確定性計算不會 parse_failed；只要 merged.json 在就成立

    sig = _read_json(matchup_dir / "signals.json") or {"signals": []}
    fired = {s["name"] for s in sig.get("signals", []) if s.get("fired")}
    has_reverse_platoon = "reverse_platoon" in fired
    has_chain_break_300 = "chain_break" in fired   # 註：嚴格 ≥0.300 判定見下方 note
    has_bullpen_il_2plus = any(
        s["name"] == "core_il_count" and (s.get("value") or 0) >= 2
        for s in sig.get("signals", []) if s.get("fired"))
```

新增小工具（load.py，若無）：
```python
def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
```

> **chain_break ≥0.300 note：** 若 signals.json 的 chain_break 帶 `value`（OPS 落差），用 `value >= 0.300` 判 `has_chain_break_300`；否則退化為 fired 即 True。實作時依 signals_lib 實際欄位調整。

回傳 dict 中 `park_factor` 改從 `merged_json.get("park_factor")`；移除對 `parse_summary` 的依賴（該段刪除）。保留 `closing_missing` / `result_missing` 等既有欄位邏輯不變。

- [ ] **Step 4: 跑測試確認 pass + 全 load 測試**

Run: `python -m pytest scripts/tests/test_load.py -v`
Expected: PASS（更新任何假設「從 summary parse」的舊測試）

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/load.py scripts/tests/test_load.py
git commit -m "feat(backtest): 預測改從凍結 merged.json 用 predict() 算（一鍵重算）+ flags 讀 signals.json"
```

---

## Task 8: 持平 gate + 全月重跑 + 比較

**Files:**
- 無新檔；跑既有 `scripts/backtest.py` + 人工檢視

- [ ] **Step 1: 量化「持平」比例（spec §14.1 gate）**

Run:
```bash
python -c "import sys; sys.path.insert(0,'scripts'); from pathlib import Path; import json; from scoring_formula import predict_with_formula as F; from predict import predict as P; root=Path('analysis-data'); n=d=0;
import itertools
for dd in sorted(root.glob('2026-05-*')):
    if dd.name.endswith('.local-backup') or not dd.is_dir(): continue
    for g in dd.iterdir():
        mj=g/'merged.json'
        if not mj.exists(): continue
        try: m=json.loads(mj.read_text(encoding='utf-8'))
        except: continue
        fp=F(m); pr=P(fp['home_score'],fp['away_score']); n+=1; d+= (pr['direction']!='持平')
print('total',n,'directional',d,'持平',n-d)"
```
Expected: 印出總場數 / 有方向場數 / 持平場數。

- [ ] **Step 2: Gate 判定**

- 若「有方向」≥ ~60 場 → 通過，續 Step 3。
- 若 < 60 → 「持平」砍太兇。回 `scripts/predict.py` 把 `PUSH_FLOOR` 從 0.53 調降（如 0.52），重跑 Step 1，記錄理由於 commit message。

- [ ] **Step 3: 全月重跑回測**

Run: `python scripts/backtest.py run --month 2026-05`
Expected: `Valid: N / M` 印出；報告生成於 `analysis-data/backtest/2026-05-report.md`。

- [ ] **Step 4: 驗收 — bucket 不再 by-day clustering（spec §16）**

Run:
```bash
python -c "import csv; from collections import defaultdict; rows=list(csv.DictReader(open('analysis-data/backtest/2026-05-details.csv',encoding='utf-8'))); byday=defaultdict(lambda:[0,0]);
for r in rows:
    p=r.get('skill_confidence_pct');
    if not p: continue
    hi = float(p)>=0.67; byday[r['date']][0]+=hi; byday[r['date']][1]+=1
[print(d, f'{h}/{t} HIGH') for d,(h,t) in sorted(byday.items())]"
```
Expected: HIGH 比例在各天**平均分布**（不再 5/08-09 全 0、5/10-26 爆量）。人工確認 clustering 消失。

- [ ] **Step 5: Commit 重跑結果**

```bash
git add analysis-data/backtest/2026-05-report.md analysis-data/backtest/2026-05-details.csv
git commit -m "data(backtest): 確定性預測全月重跑 — 乾淨零-drift baseline"
```

---

## Self-Review

**Spec coverage：**
- §4 Model B 分工 → Task 4（整體判斷 script 填）+ Task 7（回測算）✓
- §5 數字模型（winprob/方向/持平/bucket）→ Task 1, 2 ✓
- §6 winprob 曲線（S=4.0 先驗）→ Task 1（`MARGIN_SD`）✓
- §7 AI 敘事層約束 → Task 4（風險留 placeholder + ℹ️ 註記）✓
- §8 signals.json 持久化 + backfill → Task 5, 6 ✓
- §9 summary 結構（+信號=0）→ Task 3, 4 ✓
- §10.1 重跑乾淨 baseline → Task 7, 8 ✓；§10.2 ablation → 未來，spec 已載，無 task（正確，非 v1）
- §14.1 持平 gate → Task 8 Step 1-2 ✓
- §16 成功標準（重現性、clustering 消失）→ Task 1（純函式必重現）、Task 8 Step 4 ✓

**Placeholder scan：** 無 TBD/TODO。Task 7 的 chain_break value note 是「依實際欄位調整」的合理彈性，非 placeholder（給了 fallback）。

**Type consistency：** `predict()` 回傳 `{direction, total, confidence_pct, confidence_bucket}` — Task 2 定義、Task 4 / Task 7 使用一致。`signals_for_bundle` 回傳 `{signals, fired_count}` — Task 5/6/7 一致。

**已知彈性點（實作時確認）：**
1. `step_g` 真實簽名（Task 5 stub 依實際調整）
2. signals_lib `chain_break` / `core_il_count` 的 `value` 欄位名（Task 7 flags 判定依實際欄位）
3. 舊 `test_load.py` / `test_summary_renderer.py` 若有「從 summary parse」「整體判斷是 placeholder」的斷言需更新

---

## 風險

- Task 7 改回測預測來源是本 plan 最大改動；務必跑全 `test_load.py` 確認 closing_line / result 邏輯未被波及。
- 「持平」gate（Task 8）是上線前真 gate；砍太兇要調 PUSH_FLOOR，不要硬上。
