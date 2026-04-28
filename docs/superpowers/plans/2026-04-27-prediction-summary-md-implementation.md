# Phase 4 prediction_summary.md Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓 Phase 4 的 `predict.py --save` 在輸出 `prediction.json` 同時，額外產生 ready-to-paste markdown summary（含 TL;DR + Section 8-10），AI 不再 Read JSON 與 `reference/output-format.md`（後者整個刪除）。

**Architecture:** 在 `scripts/predict.py` 內新增 7 個純函式（渲染各區塊）+ 1 個 assembler；`main()` 寫完 prediction.json 後額外寫 `prediction_summary.md`。新增 ~28 個 tests 至既有 `scripts/tests/test_predict.py`。最後同步 `SKILL.md` / `reference/workflow.md` Phase 4 章節並刪除 `reference/output-format.md`。

**Tech Stack:** Python 3, pytest（既有專案慣例：`sys.path.insert` import + plain `def test_*` 函式）。

**Spec:** `docs/superpowers/specs/2026-04-27-prediction-summary-md-design.md`

**Existing utilities to reuse (already in `scripts/predict.py`):**
- `TEAM_ABBREV`（30 隊全名 → 縮寫，line 19-30）
- `compute_signal_table(data)`（returns `{"signals": list, "total_run_adjustment": float}`，line 365-422）
- `_inactive_rl_override()`（rl_override 預設 dict 結構，line 258-273）
- `apply_rl_guardrail()`（產生 final_rl_rec / final_rl_stars / rl_override，line 276-362）

---

### Task 1: `_format_pct_with_flip` pure function

**Files:**
- Modify: `scripts/predict.py`（在 `compute_signal_table` 後、`predict_with_formula` 前新增）
- Modify: `scripts/tests/test_predict.py`（追加測試）

- [ ] **Step 1: Write the failing test**

在 `scripts/tests/test_predict.py` 末尾追加：

```python


def test_format_pct_no_adjusted_home_winner():
    from predict import _format_pct_with_flip
    result = _format_pct_with_flip(51.9, "HOME", 0.0, 0.0, has_adjusted=False)
    assert result == "Formula log5: **51.9% (HOME)**"


def test_format_pct_no_adjusted_away_winner():
    """pct 永遠是 home 勝率；side label 為 HOME (主隊勝率視角)"""
    from predict import _format_pct_with_flip
    result = _format_pct_with_flip(44.2, "AWAY", 0.0, 0.0, has_adjusted=False)
    assert result == "Formula log5: **44.2% (HOME)**"


def test_format_pct_adjusted_no_flip():
    """adjusted 比分仍與 formula 同方向 → 維持 formula log5 顯示"""
    from predict import _format_pct_with_flip
    result = _format_pct_with_flip(51.9, "HOME", 5.0, 4.0, has_adjusted=True)
    assert result == "Formula log5: **51.9% (HOME)**"


def test_format_pct_adjusted_flip_home_to_away():
    from predict import _format_pct_with_flip
    result = _format_pct_with_flip(51.9, "AWAY", 4.4, 4.85, has_adjusted=True)
    assert result.startswith("⚠️")
    assert "51.9% (HOME)" in result
    assert "4.4 < 4.85" in result
    assert "AWAY 勝" in result
    assert "pct 未隨翻轉重算" in result


def test_format_pct_adjusted_flip_away_to_home():
    from predict import _format_pct_with_flip
    result = _format_pct_with_flip(44.2, "HOME", 5.0, 4.0, has_adjusted=True)
    assert result.startswith("⚠️")
    assert "44.2% (HOME)" in result
    assert "5.0 > 4.0" in result
    assert "HOME 勝" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_pct" -v
```
Expected: 5 FAIL with `ImportError: cannot import name '_format_pct_with_flip'`

- [ ] **Step 3: Write minimal implementation**

在 `scripts/predict.py` 的 `compute_signal_table` 函式結尾後（約 line 423 之後）插入：

```python


def _format_pct_with_flip(
    formula_pct: float,
    predicted_winner: str,
    adj_home: float,
    adj_away: float,
    has_adjusted: bool,
) -> str:
    """渲染勝率行；adjusted 比分翻轉方向時加 ⚠️ 註明。

    formula_pct 是 home_win_pct（永遠以主隊視角）；side label 固定為 HOME。
    翻轉條件：has_adjusted=True 且 (formula_pct > 50) != (predicted_winner == "HOME")
    """
    if not has_adjusted:
        return f"Formula log5: **{formula_pct:.1f}% (HOME)**"
    formula_winner = "HOME" if formula_pct > 50 else "AWAY"
    if formula_winner == predicted_winner:
        return f"Formula log5: **{formula_pct:.1f}% (HOME)**"
    cmp = "<" if adj_home < adj_away else ">"
    return (
        f"⚠️ Formula {formula_pct:.1f}% (HOME) → adjusted 比分 "
        f"{adj_home:.1f} {cmp} {adj_away:.1f} 判 {predicted_winner} 勝"
        "（pct 未隨翻轉重算）"
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_pct" -v
```
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 4 summary - 新增 _format_pct_with_flip

渲染勝率行；formula vs adjusted 比分方向翻轉時加 ⚠️ 註明
(pct 未隨翻轉重算)。屬於 prediction_summary.md 組件。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `format_signal_table_md` pure function

**Files:**
- Modify: `scripts/predict.py`（在 `_format_pct_with_flip` 後新增）
- Modify: `scripts/tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

在 `scripts/tests/test_predict.py` 末尾追加：

```python


def test_format_signal_table_both_populated():
    from predict import format_signal_table_md
    auto = [
        {"signal": "Park Factor 106（修正 +0.30）", "run_value": 0.30},
        {"signal": "雙方先發 FIP ≤ 3.0（Ace 級）", "run_value": -1.0},
    ]
    user = {"bullpen_il_away": 0.5, "pitcher_yoy_home": 0.3}
    result = format_signal_table_md(auto, user)
    assert "### Auto signals" in result
    assert "### User-supplied signals" in result
    assert "Park Factor 106" in result
    assert "+0.30" in result
    assert "-1.00" in result
    assert "`bullpen_il_away`" in result
    # auto 總和 = -0.70；user 總和 = +0.80
    assert "**-0.70**" in result
    assert "**+0.80**" in result


def test_format_signal_table_auto_empty():
    from predict import format_signal_table_md
    result = format_signal_table_md([], {"foo": 0.5})
    assert "### Auto signals" in result
    # auto empty → 「（無）」一行
    assert "（無）" in result
    assert "`foo`" in result


def test_format_signal_table_user_empty():
    from predict import format_signal_table_md
    auto = [{"signal": "X", "run_value": 0.5}]
    result = format_signal_table_md(auto, {})
    assert "X" in result
    assert "（無）" in result


def test_format_signal_table_both_empty():
    from predict import format_signal_table_md
    result = format_signal_table_md([], {})
    # 兩段都顯示「（無）」
    assert result.count("（無）") == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_signal_table" -v
```
Expected: 4 FAIL with `ImportError: cannot import name 'format_signal_table_md'`

- [ ] **Step 3: Write minimal implementation**

在 `_format_pct_with_flip` 後新增：

```python


def format_signal_table_md(auto_signals: list[dict], user_signals: dict) -> str:
    """組 auto + user 兩個 mini-table；各自空時顯示「（無）」一行。"""
    lines = ["### Auto signals"]
    if auto_signals:
        lines.append("| 信號 | ±run |")
        lines.append("|------|------|")
        total = 0.0
        for s in auto_signals:
            rv = s["run_value"]
            lines.append(f"| {s['signal']} | {rv:+.2f} |")
            total += rv
        lines.append(f"| **總和** | **{total:+.2f}** |")
    else:
        lines.append("（無）")

    lines.append("")
    lines.append("### User-supplied signals")
    if user_signals:
        lines.append("| Key | ±run |")
        lines.append("|-----|------|")
        total = 0.0
        for k, v in user_signals.items():
            lines.append(f"| `{k}` | {v:+.2f} |")
            total += v
        lines.append(f"| **總和** | **{total:+.2f}** |")
    else:
        lines.append("（無）")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_signal_table" -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 4 summary - 新增 format_signal_table_md

Auto signals (compute_signal_table) + user signals (--signal-adjustments)
各自渲染 mini-table，含 ±run 顯示與總和；空段顯示「（無）」。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `format_recommendation_rows` pure function

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

在 `scripts/tests/test_predict.py` 末尾追加：

```python


def _make_minimal_record(**overrides):
    """Phase 4 test record factory."""
    base = {
        "date": "2026-04-26",
        "home_team": "Kansas City Royals",
        "away_team": "Los Angeles Angels",
        "predicted_winner": "HOME",
        "predicted_home_pct": 51.9,
        "predicted_home_score": 3.1,
        "predicted_away_score": 2.6,
        "formula_home_score": 3.1,
        "formula_away_score": 2.6,
        "adjusted_total": 5.7,
        "signal_adjustments": {},
        "ou_line": 9.0,
        "ou_rec": "UNDER",
        "ou_stars": 2,
        "ml_rec": "KC",
        "ml_stars": 2,
        "original_ml_stars": 2,
        "run_line_rec": "PASS",
        "run_line_stars": None,
        "rl_override": {
            "active": False, "path": None, "diff": None, "stars": None,
            "tags": None, "warnings": None, "thresholds": None,
        },
        "tags": [],
        "temperature_f": None,
        "wind_mph": None,
        "wind_direction": None,
        "umpire_name": None,
        "umpire_ou_rate": None,
    }
    base.update(overrides)
    return base


def test_format_recommendation_rows_full_pass():
    from predict import format_recommendation_rows
    record = _make_minimal_record(
        ml_rec="PASS", ml_stars=None,
        ou_rec="PASS", ou_stars=None,
        run_line_rec="PASS",
    )
    tldr, full = format_recommendation_rows(record, [])
    # TL;DR 表頭存在
    assert "| 市場 |" in tldr
    # 三行都 PASS
    assert tldr.count("PASS") >= 3
    assert full.count("PASS") >= 3


def test_format_recommendation_rows_ml_with_audit_tag():
    from predict import format_recommendation_rows
    record = _make_minimal_record(tags=["home-2star-risk"])
    tldr, full = format_recommendation_rows(record, [])
    assert "audit" in tldr
    assert "`home-2star-risk`" in tldr
    assert "audit" in full
    # ml direction = KC
    assert "| ML | KC |" in tldr or "| KC |" in tldr


def test_format_recommendation_rows_ml_cap_appended_in_full():
    """ml_stars < original_ml_stars → full rows 附加降級原因"""
    from predict import format_recommendation_rows
    record = _make_minimal_record(ml_stars=2, original_ml_stars=4)
    cap_reasons = ["formula 勝率 51.9%（50-55%）上限 2"]
    _tldr, full = format_recommendation_rows(record, cap_reasons)
    assert "原" in full or "降為" in full or "上限" in full
    # cap reason text 必須出現
    assert "50-55%" in full


def test_format_recommendation_rows_rl_inactive():
    """rl_override.active=False → reason 提及 RL_DIFF_MIN"""
    from predict import format_recommendation_rows
    record = _make_minimal_record()
    tldr, full = format_recommendation_rows(record, [])
    assert "RL_DIFF_MIN" in tldr or "1.5" in tldr
    assert "PASS" in tldr


def test_format_recommendation_rows_rl_active_big_diff():
    """rl_override.active=True big-diff → reason 含 path + diff + tags"""
    from predict import format_recommendation_rows
    record = _make_minimal_record(
        run_line_rec="LAA",
        run_line_stars=2,
        predicted_home_score=2.0,
        predicted_away_score=4.6,
        rl_override={
            "active": True,
            "path": "big-diff",
            "diff": 2.6,
            "stars": 2,
            "tags": ["home-bullpen-slump", "home-pitching-slump"],
            "warnings": [],
            "thresholds": {"diff_min": 1.5, "diff_big": 2.2, "diff_star": 2.0},
        },
    )
    tldr, full = format_recommendation_rows(record, [])
    assert "big-diff" in tldr
    assert "2.6" in tldr
    # tags 折進 RL row 一句話理由
    assert "home-bullpen-slump" in tldr or "home-pitching-slump" in tldr


def test_format_recommendation_rows_ou_pass_due_to_small_gap():
    """ou_rec=PASS 且 |adj_total - ou_line| < 1.5 → reason 提及差距"""
    from predict import format_recommendation_rows
    record = _make_minimal_record(
        ou_rec="PASS", ou_stars=None,
        adjusted_total=8.8, ou_line=9.0,
    )
    tldr, _full = format_recommendation_rows(record, [])
    assert "0.2" in tldr  # gap
    assert "PASS" in tldr
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_recommendation_rows" -v
```
Expected: 6 FAIL with `ImportError: cannot import name 'format_recommendation_rows'`

- [ ] **Step 3: Write minimal implementation**

在 `format_signal_table_md` 後新增：

```python


def _stars_str(stars: int | None) -> str:
    """渲染星級；None / 0 → '—'，否則 ⭐ * stars。"""
    if not stars:
        return "—"
    return "⭐" * stars


def format_recommendation_rows(
    record: dict, cap_reasons: list[str]
) -> tuple[str, str]:
    """產生 (tldr_table_md, full_rows_md)。共用 reason 字串確保 TL;DR 與
    推薦結果 section 一致（同來源避免漂移）。

    一句話理由規則見 spec section 2「推薦行 一句話理由 + tag 折進規則」。
    """
    home_team = record.get("home_team", "")
    away_team = record.get("away_team", "")
    home_abbr = TEAM_ABBREV.get(home_team, home_team[:3].upper())
    away_abbr = TEAM_ABBREV.get(away_team, away_team[:3].upper())
    pct = record.get("predicted_home_pct", 0.0)
    pw = record.get("predicted_winner", "HOME")
    side = "HOME" if pct > 50 else "AWAY"
    tags = record.get("tags") or []

    # ===== ML row =====
    ml_rec = record.get("ml_rec") or "PASS"
    ml_stars = record.get("ml_stars")
    original_ml_stars = record.get("original_ml_stars")

    # Folded tags for ML row
    ml_folded = [t for t in tags if t in ("divergent", "direction-override", "home-2star-risk")]

    if ml_rec == "PASS":
        ml_dir = "PASS"
        ml_stars_str = "—"
        ml_reason = f"Log5 {pct:.1f}% ({side})"
    else:
        ml_dir = ml_rec
        ml_stars_str = _stars_str(ml_stars)
        reason_parts = [f"Log5 {pct:.1f}% ({side})"]
        if ml_folded:
            reason_parts.append("audit " + ", ".join(f"`{t}`" for t in ml_folded))
        ml_reason = "，".join(reason_parts)

    # ===== O/U row =====
    ou_rec = record.get("ou_rec") or "PASS"
    ou_stars = record.get("ou_stars")
    ou_line = record.get("ou_line")
    adj_total = record.get("adjusted_total")

    if ou_line is not None and adj_total is not None:
        gap = abs(adj_total - ou_line)
        gap_str = f"adj_total {adj_total:.1f} vs line {ou_line}，差距 {gap:.1f} run"
    else:
        gap = None
        gap_str = "—"

    if ou_rec == "PASS":
        ou_dir = "PASS"
        ou_stars_str = "—"
        if gap is not None and gap < 1.5:
            ou_reason = f"差距 {gap:.1f} < 1.5 run"
        else:
            ou_reason = gap_str
    else:
        ou_dir = ou_rec
        ou_stars_str = _stars_str(ou_stars)
        ou_reason = gap_str

    # ===== Run Line row =====
    rl_rec = record.get("run_line_rec") or "PASS"
    rl_stars = record.get("run_line_stars")
    rl_override = record.get("rl_override") or {}

    if rl_override.get("active"):
        rl_dir = rl_rec
        rl_stars_str = _stars_str(rl_stars)
        path = rl_override.get("path", "?")
        diff = rl_override.get("diff", 0.0)
        ov_tags = rl_override.get("tags") or []
        if ov_tags:
            tag_str = ", ".join(f"`{t}`" for t in ov_tags)
            rl_reason = f"override `{path}`，|diff|={diff:.1f}，tags={tag_str}"
        else:
            rl_reason = f"override `{path}`，|diff|={diff:.1f}"
    else:
        rl_dir = "PASS"
        rl_stars_str = "—"
        adj_home = record.get("predicted_home_score") or 0
        adj_away = record.get("predicted_away_score") or 0
        diff = abs(adj_home - adj_away)
        rl_reason = f"|diff|={diff:.1f} < 1.5（RL_DIFF_MIN）"

    # ===== TL;DR table =====
    tldr_lines = [
        "| 市場 | 方向 | 推薦指數 | 一句話理由 |",
        "|------|------|----------|-----------|",
        f"| ML | {ml_dir} | {ml_stars_str} | {ml_reason} |",
        f"| O/U | {ou_dir} | {ou_stars_str} | {ou_reason} |",
        f"| Run Line | {rl_dir} | {rl_stars_str} | {rl_reason} |",
    ]
    tldr = "\n".join(tldr_lines)

    # ===== Full rows =====
    def _full_row(market: str, direction: str, stars_str: str, reason: str) -> str:
        if stars_str == "—":
            head = f"**{direction}**"
        else:
            head = f"**{direction} {stars_str}**"
        return f"- **{market}**: {head} — {reason}"

    full_lines = [_full_row("ML", ml_dir, ml_stars_str, ml_reason)]
    # ML cap reason appended
    if (original_ml_stars is not None and ml_stars is not None
            and original_ml_stars > ml_stars and cap_reasons):
        full_lines[0] += f"（原 {_stars_str(original_ml_stars)} 降為 {_stars_str(ml_stars)}：{'; '.join(cap_reasons)}）"

    full_lines.append(_full_row("O/U", ou_dir, ou_stars_str, ou_reason))
    full_lines.append(_full_row("Run Line", rl_dir, rl_stars_str, rl_reason))

    full = "\n".join(full_lines)
    return tldr, full
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_recommendation_rows" -v
```
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 4 summary - 新增 format_recommendation_rows

回傳 (tldr_table_md, full_rows_md)；共用 reason 字串確保 TL;DR 與推薦
結果 section 一致。ML 行折進 divergent / direction-override /
home-2star-risk tags；RL 行 active 時折進 override.tags；ml_stars 降級
時 full row 附 cap_reasons。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `format_discipline_check` pure function

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

在 `scripts/tests/test_predict.py` 末尾追加：

```python


def test_format_discipline_check_all_pass():
    from predict import format_discipline_check
    record = _make_minimal_record()
    result = format_discipline_check(record)
    # 4 行（D1/D2/D3/D5；D4 已棄用）
    assert result.count("\n") == 3
    assert "✅ D1" in result
    assert "✅ D2" in result
    assert "✅ D3" in result
    assert "✅ D5" in result


def test_format_discipline_check_d1_direction_override():
    from predict import format_discipline_check
    record = _make_minimal_record(tags=["direction-override"])
    result = format_discipline_check(record)
    assert "⚠️ D1" in result
    assert "direction-override" in result


def test_format_discipline_check_d1_ml_pass():
    from predict import format_discipline_check
    record = _make_minimal_record(ml_rec="PASS", ml_stars=None)
    result = format_discipline_check(record)
    assert "✅ D1" in result
    assert "PASS" in result


def test_format_discipline_check_d3_violation():
    """ml 推主隊 + run_line 推客隊 → D3 ⚠️"""
    from predict import format_discipline_check
    record = _make_minimal_record(
        ml_rec="KC", run_line_rec="LAA", run_line_stars=2,
    )
    result = format_discipline_check(record)
    assert "⚠️ D3" in result


def test_format_discipline_check_d5_violation():
    """ou_rec=OVER 但 adj_total <= ou_line → D5 ⚠️"""
    from predict import format_discipline_check
    record = _make_minimal_record(
        ou_rec="OVER", adjusted_total=8.5, ou_line=9.0,
    )
    result = format_discipline_check(record)
    assert "⚠️ D5" in result


def test_format_discipline_check_d5_ou_pass_skipped():
    """ou_rec=PASS → D5 永遠 ✅"""
    from predict import format_discipline_check
    record = _make_minimal_record(ou_rec="PASS", ou_stars=None)
    result = format_discipline_check(record)
    assert "✅ D5" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_discipline_check" -v
```
Expected: 6 FAIL with `ImportError: cannot import name 'format_discipline_check'`

- [ ] **Step 3: Write minimal implementation**

在 `format_recommendation_rows` 後新增：

```python


def format_discipline_check(record: dict) -> str:
    """渲染 D1-D5 紀律檢查 4 行（D4 已棄用）。"""
    home_team = record.get("home_team", "")
    away_team = record.get("away_team", "")
    home_abbr = TEAM_ABBREV.get(home_team, home_team[:3].upper())
    away_abbr = TEAM_ABBREV.get(away_team, away_team[:3].upper())
    pw = record.get("predicted_winner", "")
    ml_rec = record.get("ml_rec") or "PASS"
    ou_line = record.get("ou_line")
    ou_rec = record.get("ou_rec") or "PASS"
    adj_total = record.get("adjusted_total")
    rl_rec = record.get("run_line_rec") or "PASS"
    tags = record.get("tags") or []

    lines = []

    # D1: predicted_winner 方向是否與 ml_rec 一致
    if "direction-override" in tags:
        lines.append(
            f"- ⚠️ D1 模型方向：direction-override（ml_rec={ml_rec}, predicted_winner={pw}）"
        )
    elif ml_rec == "PASS":
        lines.append("- ✅ D1 模型方向：ml_rec=PASS")
    else:
        winner_abbr = home_abbr if pw == "HOME" else away_abbr
        if ml_rec == winner_abbr:
            lines.append(
                f"- ✅ D1 模型方向：predicted_winner={pw}({winner_abbr}) 與 ml_rec={ml_rec} 一致"
            )
        else:
            lines.append(
                f"- ⚠️ D1 模型方向：predicted_winner={pw}({winner_abbr}) 與 ml_rec={ml_rec} 不一致"
            )

    # D2: 信號量化（永遠 ✅，predict.py 只接受 run_value 形式）
    lines.append("- ✅ D2 信號量化：所有信號已轉為 run value")

    # D3: 同場無對立推薦
    if rl_rec == "PASS" or ml_rec == "PASS":
        lines.append("- ✅ D3 同場無對立推薦")
    else:
        opposite = (
            (ml_rec == home_abbr and rl_rec == away_abbr)
            or (ml_rec == away_abbr and rl_rec == home_abbr)
        )
        if opposite:
            lines.append(
                f"- ⚠️ D3 同場推對立：ml_rec={ml_rec} + run_line_rec={rl_rec}"
            )
        else:
            lines.append("- ✅ D3 同場無對立推薦")

    # D5: 比分盤口一致
    if ou_rec == "PASS" or ou_line is None or adj_total is None:
        lines.append("- ✅ D5 比分盤口一致：ou_rec=PASS 或無 line")
    else:
        if ou_rec == "OVER" and adj_total > ou_line:
            lines.append(
                f"- ✅ D5 比分盤口一致：adj_total {adj_total} > ou_line {ou_line} vs ou_rec=OVER"
            )
        elif ou_rec == "UNDER" and adj_total < ou_line:
            lines.append(
                f"- ✅ D5 比分盤口一致：adj_total {adj_total} < ou_line {ou_line} vs ou_rec=UNDER"
            )
        else:
            lines.append(
                f"- ⚠️ D5 比分盤口矛盾：adj_total {adj_total} vs ou_line {ou_line}, ou_rec={ou_rec}"
            )

    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_discipline_check" -v
```
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 4 summary - 新增 format_discipline_check

D1/D2/D3/D5 4 行 ✅/⚠️ checkbox（D4 已棄用）。D1 偵測
direction-override；D3 同場 ml/rl 對立；D5 比分盤口一致性。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `format_rl_override_block` pure function

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

在 `scripts/tests/test_predict.py` 末尾追加：

```python


def test_format_rl_override_active_big_diff():
    from predict import format_rl_override_block
    rl = {
        "active": True, "path": "big-diff", "diff": 2.6, "stars": 2,
        "tags": ["home-bullpen-slump", "home-pitching-slump"],
        "warnings": [],
        "thresholds": {"diff_min": 1.5, "diff_big": 2.2, "diff_star": 2.0},
    }
    result = format_rl_override_block(rl)
    assert result is not None
    assert "## Run Line override 細節" in result
    assert "big-diff" in result
    assert "2.6" in result
    assert "home-bullpen-slump" in result


def test_format_rl_override_active_with_warnings():
    from predict import format_rl_override_block
    rl = {
        "active": True, "path": "mid-diff+strong-tag", "diff": 1.8, "stars": 1,
        "tags": ["away-pitching-slump"],
        "warnings": ["pw_diff_direction_mismatch"],
        "thresholds": {"diff_min": 1.5, "diff_big": 2.2, "diff_star": 2.0},
    }
    result = format_rl_override_block(rl)
    assert result is not None
    assert "pw_diff_direction_mismatch" in result
    assert "⚠️" in result


def test_format_rl_override_inactive_returns_none():
    from predict import format_rl_override_block
    rl = {
        "active": False, "path": None, "diff": None, "stars": None,
        "tags": None, "warnings": None, "thresholds": None,
    }
    assert format_rl_override_block(rl) is None


def test_format_rl_override_empty_dict_returns_none():
    from predict import format_rl_override_block
    assert format_rl_override_block({}) is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_rl_override" -v
```
Expected: 4 FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

在 `format_discipline_check` 後新增：

```python


def format_rl_override_block(rl_override: dict) -> str | None:
    """RL override active=True 時渲染細節 section；inactive 回 None。"""
    if not rl_override or not rl_override.get("active"):
        return None
    path = rl_override.get("path")
    diff = rl_override.get("diff")
    stars = rl_override.get("stars")
    tags = rl_override.get("tags") or []
    warnings = rl_override.get("warnings") or []
    thr = rl_override.get("thresholds") or {}

    lines = ["## Run Line override 細節"]
    lines.append(f"- 路徑: `{path}`")
    if diff is not None:
        lines.append(f"- |diff|: {diff:.2f}")
    if stars is not None:
        lines.append(f"- stars: {stars}")
    if tags:
        lines.append(f"- 觸發 tags: {', '.join(f'`{t}`' for t in tags)}")
    if warnings:
        lines.append(f"- ⚠️ warnings: {', '.join(warnings)}")
    if thr:
        lines.append(
            f"- thresholds: diff_min={thr.get('diff_min')}, "
            f"diff_big={thr.get('diff_big')}, diff_star={thr.get('diff_star')}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_rl_override" -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 4 summary - 新增 format_rl_override_block

active=true 時渲染 path / |diff| / tags / warnings / thresholds；
inactive 回 None 讓 caller 整段省略。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `format_env_block` pure function

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

在 `scripts/tests/test_predict.py` 末尾追加：

```python


def test_format_env_block_all_present():
    from predict import format_env_block
    record = _make_minimal_record(
        temperature_f=72.0, wind_mph=8.5, wind_direction="LF→RF",
        umpire_name="John Doe", umpire_ou_rate=0.51,
    )
    result = format_env_block(record)
    assert result is not None
    assert "## 環境補充" in result
    assert "72" in result
    assert "8.5" in result
    assert "LF→RF" in result
    assert "John Doe" in result
    assert "0.51" in result


def test_format_env_block_partial():
    from predict import format_env_block
    record = _make_minimal_record(temperature_f=68.0, umpire_name="Jane")
    result = format_env_block(record)
    assert result is not None
    assert "68" in result
    assert "Jane" in result
    # 沒有 wind / ou_rate row
    assert "風速" not in result
    assert "Over%" not in result


def test_format_env_block_all_null_returns_none():
    from predict import format_env_block
    record = _make_minimal_record()  # all env fields None
    assert format_env_block(record) is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_env_block" -v
```
Expected: 3 FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

在 `format_rl_override_block` 後新增：

```python


def format_env_block(record: dict) -> str | None:
    """渲染環境補充 section；所有欄位皆 null → None（整段省略）。"""
    fields = [
        ("氣溫 (°F)", record.get("temperature_f")),
        ("風速 (mph)", record.get("wind_mph")),
        ("風向", record.get("wind_direction")),
        ("主審", record.get("umpire_name")),
        ("主審 Over%", record.get("umpire_ou_rate")),
    ]
    non_null = [(k, v) for k, v in fields if v is not None]
    if not non_null:
        return None
    lines = ["## 環境補充"]
    for k, v in non_null:
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_env_block" -v
```
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 4 summary - 新增 format_env_block

5 個環境欄位（溫度 / 風速 / 風向 / 主審 / 主審 Over%）非 null 時列出；
全 null 回 None（整段省略）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `format_trend_tags_block` pure function

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

在 `scripts/tests/test_predict.py` 末尾追加：

```python


def test_format_trend_tags_block_pure_trend():
    from predict import format_trend_tags_block
    tags = ["home-hot-offense", "home-pitching-slump", "away-bullpen-strong"]
    result = format_trend_tags_block(tags, set())
    assert result is not None
    assert "## 趨勢標記" in result
    assert "`home-hot-offense`" in result
    assert "`home-pitching-slump`" in result
    assert "`away-bullpen-strong`" in result


def test_format_trend_tags_block_all_folded_returns_none():
    """所有 tags 都已折進推薦 → None"""
    from predict import format_trend_tags_block
    tags = ["divergent", "home-2star-risk"]
    folded = {"divergent", "home-2star-risk"}
    assert format_trend_tags_block(tags, folded) is None


def test_format_trend_tags_block_partial_fold():
    """部分折進 → 剩下的列出"""
    from predict import format_trend_tags_block
    tags = ["home-hot-offense", "divergent", "home-bullpen-slump"]
    folded = {"divergent"}
    result = format_trend_tags_block(tags, folded)
    assert result is not None
    assert "`home-hot-offense`" in result
    assert "`home-bullpen-slump`" in result
    assert "`divergent`" not in result


def test_format_trend_tags_block_empty_tags_returns_none():
    from predict import format_trend_tags_block
    assert format_trend_tags_block([], set()) is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_trend_tags_block" -v
```
Expected: 4 FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

在 `format_env_block` 後新增：

```python


def format_trend_tags_block(tags: list[str], recommendation_tags: set[str]) -> str | None:
    """扣除已折進推薦行的 tags；剩下空 → None。"""
    remaining = [t for t in tags if t not in recommendation_tags]
    if not remaining:
        return None
    lines = ["## 趨勢標記"]
    lines.append("- " + "、".join(f"`{t}`" for t in remaining))
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_trend_tags_block" -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 4 summary - 新增 format_trend_tags_block

扣除已折進推薦行的 tags（divergent / direction-override /
home-2star-risk / RL override.tags）；剩下空 → None。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `format_prediction_summary_md` assembler

**Files:**
- Modify: `scripts/predict.py`
- Modify: `scripts/tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

在 `scripts/tests/test_predict.py` 末尾追加：

```python


def test_format_prediction_summary_md_smoke_full():
    """完整 record → markdown 含所有 hard sections"""
    from predict import format_prediction_summary_md
    record = _make_minimal_record(
        tags=["home-hot-offense", "home-pitching-slump", "away-bullpen-slump", "home-2star-risk"],
    )
    signal_table = {
        "signals": [{"signal": "Park Factor 106（修正 +0.30）", "run_value": 0.30}],
        "total_run_adjustment": 0.30,
    }
    md = format_prediction_summary_md(record, signal_table, [])
    assert "# Prediction Summary — LAA @ KC (2026-04-26)" in md
    assert "## TL;DR" in md
    assert "## 比分預測" in md
    assert "## 勝率預測" in md
    assert "## 信號修正表" in md
    assert "## 推薦結果" in md
    assert "## 紀律檢查" in md
    # narrative placeholder
    assert "<!-- narrative:" in md
    # auto signal in table
    assert "Park Factor 106" in md
    # 趨勢標記 soft section（含未折進的 tags）
    assert "## 趨勢標記" in md
    assert "`home-hot-offense`" in md


def test_format_prediction_summary_md_all_pass():
    """全 PASS 場景 → 三行 PASS，無對立 / 無 cap"""
    from predict import format_prediction_summary_md
    record = _make_minimal_record(
        ml_rec="PASS", ml_stars=None, original_ml_stars=None,
        ou_rec="PASS", ou_stars=None,
        run_line_rec="PASS",
    )
    signal_table = {"signals": [], "total_run_adjustment": 0.0}
    md = format_prediction_summary_md(record, signal_table, [])
    assert md.count("PASS") >= 3


def test_format_prediction_summary_md_rl_override_active():
    """rl_override.active → Run Line override 細節 section 出現"""
    from predict import format_prediction_summary_md
    record = _make_minimal_record(
        run_line_rec="LAA", run_line_stars=2,
        predicted_home_score=2.0, predicted_away_score=4.6,
        rl_override={
            "active": True, "path": "big-diff", "diff": 2.6, "stars": 2,
            "tags": ["home-bullpen-slump"],
            "warnings": [],
            "thresholds": {"diff_min": 1.5, "diff_big": 2.2, "diff_star": 2.0},
        },
    )
    signal_table = {"signals": [], "total_run_adjustment": 0.0}
    md = format_prediction_summary_md(record, signal_table, [])
    assert "## Run Line override 細節" in md


def test_format_prediction_summary_md_adjusted_flip():
    """adjusted 翻轉 → 勝率行有 ⚠️ 註明"""
    from predict import format_prediction_summary_md
    record = _make_minimal_record(
        predicted_winner="AWAY",
        predicted_home_pct=51.9,
        predicted_home_score=4.4,
        predicted_away_score=4.85,
        formula_home_score=3.1,  # 與 predicted 不同 → has_adjusted=True
        formula_away_score=2.6,
        adjusted_total=9.25,
    )
    signal_table = {"signals": [], "total_run_adjustment": 0.0}
    md = format_prediction_summary_md(record, signal_table, [])
    assert "⚠️" in md
    assert "未隨翻轉重算" in md


def test_format_prediction_summary_md_all_soft_omitted():
    """soft sections 不適用 → 全省略，只剩 hard sections"""
    from predict import format_prediction_summary_md
    record = _make_minimal_record(tags=[])
    signal_table = {"signals": [], "total_run_adjustment": 0.0}
    md = format_prediction_summary_md(record, signal_table, [])
    assert "## Run Line override 細節" not in md
    assert "## 環境補充" not in md
    assert "## 趨勢標記" not in md


def test_format_prediction_summary_md_raises_on_missing_home_team():
    from predict import format_prediction_summary_md
    import pytest as _pytest
    bad = _make_minimal_record()
    del bad["home_team"]
    with _pytest.raises(ValueError):
        format_prediction_summary_md(bad, {"signals": [], "total_run_adjustment": 0}, [])


def test_format_prediction_summary_md_raises_on_missing_away_team():
    from predict import format_prediction_summary_md
    import pytest as _pytest
    bad = _make_minimal_record()
    del bad["away_team"]
    with _pytest.raises(ValueError):
        format_prediction_summary_md(bad, {"signals": [], "total_run_adjustment": 0}, [])


def test_format_prediction_summary_md_raises_on_missing_predicted_winner():
    from predict import format_prediction_summary_md
    import pytest as _pytest
    bad = _make_minimal_record()
    del bad["predicted_winner"]
    with _pytest.raises(ValueError):
        format_prediction_summary_md(bad, {"signals": [], "total_run_adjustment": 0}, [])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_prediction_summary_md" -v
```
Expected: 8 FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

在 `format_trend_tags_block` 後新增：

```python


def format_prediction_summary_md(
    record: dict, signal_table: dict, cap_reasons: list[str]
) -> str:
    """組合 prediction_summary.md 完整內容。
    Hard sections（必出現）：TL;DR / 比分預測 / 勝率預測 / 信號修正表 / 推薦結果 / 紀律檢查
    Soft sections（缺資料省略）：Run Line override 細節 / 環境補充 / 趨勢標記
    Fail-fast：缺 home_team / away_team / predicted_winner → raise ValueError
    """
    if "home_team" not in record or "away_team" not in record:
        raise ValueError("record missing home_team / away_team")
    if "predicted_winner" not in record:
        raise ValueError("record missing predicted_winner")

    home_team = record["home_team"]
    away_team = record["away_team"]
    home_abbr = TEAM_ABBREV.get(home_team, home_team[:3].upper())
    away_abbr = TEAM_ABBREV.get(away_team, away_team[:3].upper())
    date = record.get("date", "—")

    pw = record["predicted_winner"]
    pct = record.get("predicted_home_pct", 0.0)
    home_score = record.get("predicted_home_score", 0.0)
    away_score = record.get("predicted_away_score", 0.0)
    formula_home = record.get("formula_home_score", 0.0)
    formula_away = record.get("formula_away_score", 0.0)
    formula_total = round(formula_home + formula_away, 1)
    adj_total = record.get("adjusted_total", 0.0)
    ou_line = record.get("ou_line")

    # has_adjusted: predicted_score 與 formula_score 不同 = 用戶有傳 --adjusted-*
    has_adjusted = (
        abs(formula_home - home_score) > 0.01 or abs(formula_away - away_score) > 0.01
    )

    tldr_table, full_rows = format_recommendation_rows(record, cap_reasons)

    lines = [
        f"# Prediction Summary — {away_abbr} @ {home_abbr} ({date})",
        "",
        "## TL;DR",
        f"- 預測比分: **{home_abbr} {home_score:.1f} − {away_score:.1f} {away_abbr}**"
        f"（{pw} 勝，勝率 {pct:.1f}%）",
        "- 比賽走勢: <!-- narrative: AI 依 reference/prediction.md「比賽敘事觸發條件」選 1-2 句填入 -->",
        "",
        "📊 推薦速查:",
        "",
        tldr_table,
        "",
        "---",
        "",
        "## 比分預測",
        f"- Formula 比分: {home_abbr} {formula_home:.1f} / {away_abbr} {formula_away:.1f}"
        f"（總分 {formula_total:.1f}）",
    ]
    if has_adjusted:
        lines.append(
            f"- Adjusted 比分: {home_abbr} {home_score:.1f} / {away_abbr} {away_score:.1f}"
            f"（總分 {adj_total:.1f}）"
        )
    if ou_line is not None:
        gap = abs(adj_total - ou_line)
        lines.append(
            f"- O/U gap: |adj_total {adj_total:.1f} − line {ou_line}| = {gap:.1f}"
        )

    lines.extend([
        "",
        "## 勝率預測",
        f"- {_format_pct_with_flip(pct, pw, home_score, away_score, has_adjusted)}",
        "",
        "## 信號修正表",
        "",
        format_signal_table_md(
            signal_table.get("signals") or [],
            record.get("signal_adjustments") or {},
        ),
        "",
        "## 推薦結果",
        full_rows,
        "",
        "## 紀律檢查 (D1-D5)",
        format_discipline_check(record),
    ])

    # Soft sections
    rl_block = format_rl_override_block(record.get("rl_override") or {})
    if rl_block:
        lines.extend(["", rl_block])

    env_block = format_env_block(record)
    if env_block:
        lines.extend(["", env_block])

    # Build folded tags set: 已被推薦行折入的不再進趨勢標記
    tags = record.get("tags") or []
    folded: set[str] = set()
    for t in ("divergent", "direction-override", "home-2star-risk"):
        if t in tags:
            folded.add(t)
    rl_override = record.get("rl_override") or {}
    if rl_override.get("active") and rl_override.get("tags"):
        folded.update(rl_override["tags"])

    trend_block = format_trend_tags_block(tags, folded)
    if trend_block:
        lines.extend(["", trend_block])

    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest scripts/tests/test_predict.py -k "format_prediction_summary_md" -v
```
Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py scripts/tests/test_predict.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 4 summary - 新增 format_prediction_summary_md

組合 prediction_summary.md 完整內容。Hard sections 6（TL;DR / 比分 /
勝率 / 信號表 / 推薦 / 紀律 D1-D5）必出；Soft sections 3（RL override
/ 環境 / 趨勢標記）不適用整段省略。Fail-fast on missing home_team /
away_team / predicted_winner。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: `main()` 整合 — 寫出 summary 檔

**Files:**
- Modify: `scripts/predict.py`（修改 `main()` 末段，緊接 prediction.json 寫入後）

無單元測試（純 I/O 整合，靠下一步手動執行驗證）。

- [ ] **Step 1: 收集 cap_reasons 並修改 `main()`**

在 `scripts/predict.py` 的 `main()` 中，找到這段（約 line 831-834）：

```python
        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
        with open(prediction_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"Saved to {prediction_path}", file=sys.stderr)
```

替換為：

```python
        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
        with open(prediction_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"Saved to {prediction_path}", file=sys.stderr)

        # 額外輸出 prediction_summary.md（同目錄）
        summary_path = Path(prediction_path).parent / "prediction_summary.md"
        try:
            summary_md = format_prediction_summary_md(record, signal_table, cap_reasons)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_md)
            print(f"Saved summary to {summary_path}", file=sys.stderr)
        except ValueError as e:
            print(f"Skipped summary (data incomplete): {e}", file=sys.stderr)
```

確認 `Path` 已 import（檔頂應有 `from pathlib import Path`，line 11）。

- [ ] **Step 2: 手動驗證 — 用 LAA@KC 2026-04-26 fixture**

```bash
python scripts/predict.py \
  --game-data analysis-data/2026-04-26/LAA@KC/merged.json \
  --save \
  --adjusted-home 3.1 --adjusted-away 2.6 \
  --ou-line 9.0 --ou-rec UNDER --ou-stars 2 \
  --ml-rec KC --ml-stars 2 \
  --skip-phase3-check \
  --skip-yoy-check
```

Expected stderr (last 2 lines):
```
Saved to analysis-data/2026-04-26/LAA@KC/prediction.json
Saved summary to analysis-data/2026-04-26/LAA@KC/prediction_summary.md
```

> ⚠️ 若 LAA@KC merged.json 不存在，改用任意當日已存在的 merged.json 路徑替代驗證。

- [ ] **Step 3: 檢查 summary 內容**

```bash
cat analysis-data/2026-04-26/LAA@KC/prediction_summary.md
```

Expected：6 個 hard sections 全在；TL;DR 含 narrative HTML 註解；推薦速查表 3 行；信號修正表 2 段；紀律檢查 4 行（D1/D2/D3/D5）。

- [ ] **Step 4: 全測試回歸**

```bash
python -m pytest scripts/tests/ -v
```
Expected: 全部 PASS（含舊測試 + 新增 ~38 個）

- [ ] **Step 5: Commit**

```bash
git add scripts/predict.py
git commit -m "$(cat <<'EOF'
feat(mlb-skill): Phase 4 summary - main() 整合輸出 summary md

寫完 prediction.json 後額外輸出 prediction_summary.md 至同目錄。
ValueError 時 stderr warning 但不 fail（保留 JSON 輸出）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: SOP 更新 + 刪除 `output-format.md`

**Files:**
- Modify: `SKILL.md`（Quick Reference Phase 4 行）
- Modify: `reference/workflow.md`（Phase 4 章節）
- Delete: `reference/output-format.md`

無單元測試（純文件變更）。

- [ ] **Step 1: 修改 `SKILL.md` Quick Reference**

找到 Quick Reference 區段（約 line 24-31）的 Phase 4 行：

```markdown
| 4. 預測輸出 | `prediction.json` + 報告（`predict.py`；紀律 D1-D5 自動執行） |
```

改為：

```markdown
| 4. 預測輸出 | `prediction.json` + `prediction_summary.md`（`predict.py`；紀律 D1-D5 自動執行） |
```

- [ ] **Step 2: 修改 `reference/workflow.md` Phase 4.0**

找到 Phase 4.0 末尾的 `> ⚠️ **勝率與比分皆用 predict.py 的 ...` 提醒（約 line 310）。

在這行**之後**插入：

```markdown

> 腳本同時輸出 `prediction_summary.md` 至同目錄（含 ready-to-paste TL;DR + Section 8-10）。
```

- [ ] **Step 3: 修改 `reference/workflow.md` Phase 4.7 標題下**

找到 `### 4.7 輸出前驗證` 區段（約 line 346），在原本的 `⛔ **輸出前必須逐項檢查：**` 那行**之前**插入：

```markdown

✅ Read `$GAME_DIR/prediction_summary.md`，確認 `## 紀律檢查` section 全 ✅；TL;DR + Section 8-10 直接複製進最終報告。

ℹ️ 一般情況下無需 Read `prediction.json`；僅在 summary 缺漏 / 除錯 / 使用者明確要求查驗時 Read JSON。

```

- [ ] **Step 4: 修改 `reference/workflow.md` Phase 4.7 閘門加一條**

在 4.7 既有 checkbox 的最開頭插入：

```markdown
- [ ] `prediction_summary.md` 已輸出
```

完整 4.7 閘門應變為：
```
- [ ] `prediction_summary.md` 已輸出
- [ ] D1 / D2 紀律通過？
- [ ] D3 同場無對立推薦？
...
```

- [ ] **Step 5: 重寫 `reference/workflow.md` Phase 4.8**

找到（約 line 357-359）：

```markdown
### 4.8 輸出格式

完整模板見 `reference/output-format.md`（TL;DR + 10 段完整報告）。
```

替換為：

```markdown
### 4.8 輸出格式

完整 TL;DR + Section 8-10 模板已內化於 `prediction_summary.md`，AI 直接複製貼上。Section 1-7（基本面：球場、戰績、投手、打線、牛棚、條件修正等）由 AI 從 `game_data_summary.md` / `merged_summary.md` / `phase3_summary.md` 補充。
```

- [ ] **Step 6: 刪除 `reference/output-format.md`**

```bash
git rm reference/output-format.md
```

- [ ] **Step 7: 驗證 SOP 內容**

```bash
grep -n "prediction_summary" SKILL.md reference/workflow.md
```
Expected：`SKILL.md` 至少 1 條（Quick Reference Phase 4），`reference/workflow.md` 至少 4 條（Phase 4.0 / 4.7 開頭 / 4.7 閘門 / 4.8）。

```bash
ls reference/output-format.md
```
Expected：`No such file or directory`

```bash
grep -rn "output-format.md" SKILL.md reference/ scripts/
```
Expected：無 hit（歷史 spec / plan 中的提及不算）。

- [ ] **Step 8: Commit**

```bash
git add SKILL.md reference/workflow.md
git commit -m "$(cat <<'EOF'
docs(mlb-skill): Phase 4 SOP 對齊 prediction_summary.md，刪 output-format.md

SKILL.md Quick Reference Phase 4 主要產出加 prediction_summary.md；
workflow.md Phase 4.0 補腳本同時輸出 summary、Phase 4.7 開頭改 Read
summary 而非 JSON、4.7 閘門新增 summary 檢查、4.8 重寫指向 summary
（Section 1-7 仍由 AI 從上游 summary 補）。

刪除 reference/output-format.md（模板與資料源同源於 predict.py）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- ✅ Section 2 規格 → Task 8 (`format_prediction_summary_md`)
- ✅ Section 2 TL;DR + Section 8-10 內化 → Task 3 (`format_recommendation_rows` 回傳 tuple) + Task 8 assembler
- ✅ Section 2 紀律檢查規則（D1/D2/D3/D5） → Task 4 (`format_discipline_check`)
- ✅ Section 2 推薦行 tag 折進規則 → Task 3 + Task 8 folded set
- ✅ Section 2 _format_pct_with_flip → Task 1
- ✅ Section 2 信號修正表 → Task 2
- ✅ Section 2 RL override 細節 → Task 5
- ✅ Section 2 環境補充 → Task 6
- ✅ Section 2 趨勢標記 → Task 7
- ✅ Section 3 邊界條件 → Task 8 hard/soft 邏輯 + 各純函式 edge case 測試
- ✅ Section 3 Fail-fast → Task 8 raise tests
- ✅ Section 4.1 main() 整合 → Task 9
- ✅ Section 4.2 測試 → Tasks 1-8 各自含測試
- ✅ Section 4.3 SKILL.md / workflow.md → Task 10
- ✅ Section 4.4 刪除 output-format.md → Task 10 Step 6
- ✅ Section 4.5 不動的部分 → 計畫不觸碰

**Placeholder scan:**
- ✅ 無 TBD / TODO / "implement later"
- ✅ 每個 step 含完整可執行的代碼或命令

**Type consistency:**
- ✅ `_format_pct_with_flip(formula_pct: float, predicted_winner: str, adj_home: float, adj_away: float, has_adjusted: bool) → str` 在 Task 1 定義，Task 8 呼叫一致
- ✅ `format_signal_table_md(auto_signals: list[dict], user_signals: dict) → str` 在 Task 2 定義，Task 8 呼叫一致
- ✅ `format_recommendation_rows(record: dict, cap_reasons: list[str]) → tuple[str, str]` 在 Task 3 定義，Task 8 呼叫一致
- ✅ `format_discipline_check(record: dict) → str` 在 Task 4 定義，Task 8 呼叫一致
- ✅ `format_rl_override_block(rl_override: dict) → str | None` 在 Task 5 定義，Task 8 呼叫一致
- ✅ `format_env_block(record: dict) → str | None` 在 Task 6 定義，Task 8 呼叫一致
- ✅ `format_trend_tags_block(tags: list, recommendation_tags: set) → str | None` 在 Task 7 定義，Task 8 呼叫一致
- ✅ `format_prediction_summary_md(record: dict, signal_table: dict, cap_reasons: list[str]) → str` 在 Task 8 定義，Task 9 main() 呼叫一致
- ✅ `_make_minimal_record(**overrides)` test factory 在 Task 3 定義（測試專用），Tasks 4 / 5 / 6 / 7 / 8 重用一致
