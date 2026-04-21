# 2026-04-21 — RL 對稱化（Symmetrize RL guardrail with ML/OU markets）

## Context

4/20 上線的 `RL-1b` 放寬規則（`docs/specs/2026-04-20-rl-threshold-relaxation.md`）在 4/20 9 場 MLB 比賽**完全未觸發**。9/9 場全部走 RL = PASS，未升級任何 RL override。

**診斷時間軸**：
- 4/20 session 觀察到「RL 門檻太高」症狀
- 初步假設：DIFF_MIN=1.5 / DIFF_BIG=2.2 門檻過高
- 讀 code + 全 9 場 prediction.json 驗證：門檻本身沒擋住任何一場；擋住 9/9 的是 `apply_rl_guardrail` 的**入口閘門** `user_rl_rec is None`。CLI 傳入 `--run-line-rec PASS` 時 `args.run_line_rec = "PASS"` ≠ `None`，RL-1b 整塊跳過
- 進一步探查：ML / OU guardrail 不讀 `confidence`；只有 RL 讀 `confidence == "LOW"` 當 hard gate
- 結論：根因不是「入口閘門 gate 語義」（PASS vs None），而是更底層的**市場間不對稱性** — RL 是唯一一個「因為 ML 市場的 confidence 標籤而被特殊處理」的市場

**目標**：把 RL guardrail 重寫成跟 ML / OU 對稱的結構 — **每個市場只讀自己市場的證據**。
- ML 讀 `cross_validation`（DIVERGENT / INSUFFICIENT_SAMPLE）
- OU 讀 `gap`（`|adj_total - ou_line| >= 1.5`）
- RL 讀 `diff` + `strong_tags`（既有 RL-1b 規則，但不綁 confidence）

本 spec **supersede** `docs/specs/2026-04-20-rl-threshold-relaxation.md` 裡兩個具體規則：
- 被 supersede：RL-1b 的 `confidence == "LOW"` 條件（放寬條件之一）
- 被 supersede：RL-1 硬閘門（`confidence == "LOW" and user_rl_rec != "PASS" → PASS`）

不被 supersede 的部分：DIFF_MIN=1.5 / DIFF_BIG=2.2 / DIFF_STAR=2.0、STRONG_TAGS 集合、`rl_override` schema、Q3 `kelly_available` 語義、Q4 `pw_diff_direction_mismatch` defensive check。

## 現況觀察

### 不對稱證據（verbatim quote）

```python
# ML (predict.py:991-1054 摘要)
# D1: cross_validation == "DIVERGENT" → force_ml_pass
# D1.5: INSUFFICIENT_SAMPLE + ml 與 formula_log5 方向不一致 → force_ml_pass
# 都不以 "confidence == LOW" 為 hard gate

# OU (predict.py:1066-1092)
# NEUTRAL → PASS
# |adj_total - ou_line| < 1.5 → PASS
# adj_total 與 rec 方向矛盾 → PASS
# 缺 --ou-stars → PASS
# 完全不讀 confidence

# RL (predict.py:280-337 現況)
# RL-1b: confidence == "LOW" AND user_rl_rec is None AND diff >= DIFF_MIN → 升級
# RL-1 : confidence == "LOW" AND user_rl_rec != "PASS" → 砍成 PASS
# RL-2 : 非 PASS 但 stars unspecified → PASS
# ↑ RL-1b + RL-1 兩條都綁 confidence
```

### 4/20 回放（9 場 prediction.json）

| 場次 | \|diff\| | 強 tag | confidence | user_rl_rec | rl_override.active |
|---|---:|---|---|---|---|
| HOU@CLE | 3.30 | 3 | LOW | PASS | ❌ false |
| ATH@SEA | 2.40 | - | LOW | PASS | ❌ false |
| ATL@WSH | 2.40 | home-bullpen-slump | LOW | PASS | ❌ false |
| LAD@COL | 2.30 | home-pitching-slump | LOW | PASS | ❌ false |
| STL@MIA | 1.50 | away-bullpen-slump | LOW | PASS | ❌ false |
| DET@BOS | 1.00 | - | LOW | PASS | ❌ false |
| BAL@KC | 0.00 | home-bullpen-slump | LOW | PASS | ❌ false |
| PHI@CHC | 0.40 | - | LOW | PASS | ❌ false |
| TOR@LAA | 0.30 | - | LOW | PASS | ❌ false |

9/9 場 `kelly.warnings = ["rl_guardrail_pass"]`（`final_rl_rec = "PASS"` 的旗號）。

## 根因

| 層級 | 問題 | 解決層面 |
|------|------|---------|
| 表層 | 4/20 skill 每場傳 `--run-line-rec PASS` | workflow.md:232 標「必填」，skill 無 RL 結論時預設 PASS |
| 中層 | `apply_rl_guardrail` 入口閘門 `user_rl_rec is None` 把 `PASS` 當「使用者明確否決」 | gate 語義無法區分「skill 預設 PASS」vs「user 明確否決」 |
| 底層（真正根因） | RL 綁 `confidence`，而 ML/OU 不綁 | **市場間不對稱性**：把 ML 市場的 confidence 標籤當 RL 的 hard gate |

只解決表層/中層 = 治標。治本 = 砍掉底層不對稱性。

## 設計原則

**Symmetric guardrail 原則**：每個市場只讀自己市場的證據品質。

| 市場 | 證據品質來源 | 不讀的東西 |
|------|------------|----------|
| ML | `cross_validation` (DIVERGENT / INSUFFICIENT_SAMPLE) | - |
| OU | `\|adj_total - ou_line\|` gap + stars 完整性 | confidence |
| **RL（新）** | `diff` + `strong_tags` + stars 完整性 | **confidence（砍）** |

confidence 是 ML 市場的 output，不該跨市場當 gate。跨市場用，就是把一個市場的噪音傳染到另一個市場。

## 解決方案

### Step 1 — `apply_rl_guardrail` 重寫

**目前（predict.py:225-345）**：`confidence` 出現兩處
- `:283` RL-1b 升級條件
- `:332` RL-1 硬閘門

**重寫**：整段不讀 `confidence`。對應參數從簽章移除（呼叫點只有 `predict.py:1097` 一處，同步更新即可；保留 deprecated no-op 違反對稱原則）。

```python
def apply_rl_guardrail(
    *,
    # confidence 參數移除（原 predict.py:227）— 不再讀 ML 市場的 confidence
    adj_home: float,
    adj_away: float,
    trend_tags: list[str],
    user_rl_rec: str | None,
    user_rl_stars: int | None,
    predicted_winner: str,
    home_team: str,
    away_team: str,
    kelly_rl_available: bool = False,
) -> tuple[str, int | None, dict]:
    """Apply RL guardrails and produce rl_override audit dict.

    Rules (Symmetric to ML/OU — see docs/superpowers/specs/2026-04-21-rl-symmetrization-design.md):
      RL-1b (auto-推薦): user_rl_rec in (None, "PASS") AND diff >= DIFF_MIN
                        → auto-upgrade to team-abbr + 1/2 stars.
                        A |diff| >= DIFF_BIG                  → big-diff (no tag needed)
                        B DIFF_MIN <= |diff| < DIFF_BIG + strong tag → mid-diff+strong-tag
                        Stars: |diff| <= DIFF_STAR → 1; else 2.
      RL-2  (existing): 非 PASS 但 stars unspecified → PASS.

    PASS-as-unspecified: user_rl_rec == "PASS" 視同 None（skill workflow 預設值）。
    user 傳 team abbr (如 "NYY") 則被尊重，RL-1b 不 override。

    Defensive (Q4): if predicted_winner != diff_side → warning but do not block.
    """
    final_rl_rec = user_rl_rec if user_rl_rec is not None else "PASS"
    final_rl_stars = user_rl_stars
    rl_override = _inactive_rl_override()

    diff = abs(adj_home - adj_away)
    diff_side = "HOME" if adj_home > adj_away else "AWAY"
    strong_rl = RL_STRONG_TAGS & set(trend_tags)

    # RL-1b: auto-推薦（不讀 confidence — 對稱 ML/OU）
    override_path = None
    if user_rl_rec in (None, "PASS") and diff >= RL_DIFF_MIN:
        if diff >= RL_DIFF_BIG:
            override_path = "big-diff"
        elif strong_rl:
            override_path = "mid-diff+strong-tag"

    if override_path is not None:
        warnings: list[str] = []
        if predicted_winner != diff_side:
            warnings.append("pw_diff_direction_mismatch")
            print(
                f"⚠️ RL-1b: predicted_winner={predicted_winner} 與 diff_side={diff_side} 不一致",
                file=sys.stderr,
            )

        fav_team = home_team if diff_side == "HOME" else away_team
        fav_abbr = TEAM_ABBREV.get(fav_team, "")

        if fav_abbr:
            stars = 2 if diff > RL_DIFF_STAR else 1
            final_rl_rec = fav_abbr
            final_rl_stars = stars
            rl_override = {
                "active": True,
                "path": override_path,
                "diff": round(diff, 2),
                "stars": stars,
                "tags": sorted(strong_rl),
                "kelly_available": bool(kelly_rl_available),
                "warnings": warnings,
                "thresholds": {
                    "diff_min": RL_DIFF_MIN,
                    "diff_big": RL_DIFF_BIG,
                    "diff_star": RL_DIFF_STAR,
                },
            }
            print(
                f"ℹ️ RL-1b 自動推薦（{override_path}）：|diff|={diff:.2f} "
                f"tags={sorted(strong_rl) or '(pure-diff)'} → {fav_abbr} {stars}★",
                file=sys.stderr,
            )

    # RL-1（舊）整段刪除 — 不對稱的 confidence hard gate 不再存在

    # RL-2: 非 PASS 但 stars 未指定 → PASS（sanity check，對等 OU）
    if final_rl_rec != "PASS" and final_rl_stars is None:
        print(f"⚠️ 讓分盤從 {final_rl_rec} 改為 PASS（未指定 --run-line-stars）", file=sys.stderr)
        final_rl_rec = "PASS"
        final_rl_stars = 0

    return final_rl_rec, final_rl_stars, rl_override
```

**關鍵變更**：
1. 簽章移除 `confidence` 參數（若保留為向後相容，函式內部不用）
2. RL-1b gate 條件改為 `user_rl_rec in (None, "PASS") and diff >= RL_DIFF_MIN`
3. RL-1 硬閘門（舊 predict.py:329-337）整段刪除
4. 訊息字串「RL-1b 放寬（...）」改為「RL-1b 自動推薦（...）」（語義變化：從「LOW 情境的特殊放寬」變成「RL 市場的預設推薦邏輯」）

**呼叫點更新**（predict.py:1097）：
```python
final_rl_rec, final_rl_stars, rl_override = apply_rl_guardrail(
    # confidence=result["final"]["confidence"],  ← 移除此參數傳遞
    adj_home=adj_home,
    adj_away=adj_away,
    trend_tags=trend_tags,
    user_rl_rec=args.run_line_rec,
    user_rl_stars=args.run_line_stars,
    predicted_winner=result["final"]["recommended_winner"],
    home_team=home_team,
    away_team=away_team,
)
```

### Step 2 — Fixture 重構（scripts/tests/test_predict_snapshot.py:486+）

既有 fixture（`def test_rl1b_*`）審查與調整：

| # | 既有函式名 | 情境 | 新設計下語義 | 動作 |
|---|-----------|------|-------------|------|
| 1 | `test_rl1b_mid_diff_strong_tag_1star` | LOW + diff=1.8 + bullpen-slump | **不變** | **保留**（移除 `confidence=LOW` kwarg，因為簽章不再有此參數） |
| 2 | `test_rl1b_big_diff_no_tag_2star` | LOW + diff=2.3 + 無 tag | **不變** | **保留**（同上） |
| 3 | `test_rl1b_mid_diff_strong_tag_just_over_star_boundary` | LOW + diff=2.1 + pitching-slump | **不變** | **保留** |
| 4 | `test_rl1b_diff_below_min_not_triggered` | LOW + diff=1.4 + bullpen-slump | **不變**（diff < DIFF_MIN） | **保留** |
| 5 | `test_rl1b_mid_diff_without_strong_tag_not_triggered` | LOW + diff=1.8 + 無 tag | **不變**（中分差需 tag） | **保留** |
| 6 | `test_rl1b_high_confidence_path_unchanged` | **HIGH + 任意 diff/tag → 不觸發** | **語義變**（HIGH + no user rec + diff 符合 → 會觸發） | **重寫為 6a/6b** |
| 7 | `test_rl1b_respects_user_supplied_rec` | user 傳 NYY + LOW → 不 override | **不變**（尊重 user 仍成立，但 reason 從「LOW」變「user 明確傳 non-PASS non-None」） | **保留**，移除 `confidence` kwarg |
| 8 | `test_rl1b_defensive_direction_mismatch_still_triggers` | Q4 defensive check | **不變** | **保留**，移除 `confidence` kwarg |

**新增 fixture**：

- **6a** `test_rl1b_high_confidence_user_supplied_abbr_respected` — HIGH（情境層面）+ user 傳 `NYY` → 尊重 user，不 override
- **6b** `test_rl1b_high_confidence_no_user_rec_auto_triggers` — HIGH（情境層面）+ user 省略 + diff=2.3 + home-pitching-slump → 觸發 big-diff 2★（**新對稱行為**）
- **9** `test_rl1b_user_pass_treated_as_unspecified` — user 傳 `PASS` + diff=2.3 + home-pitching-slump → 觸發 big-diff 2★（PASS-as-unspecified）
- **10** `test_rl1b_not_gated_by_confidence_high_case` — HIGH-like 情境 + user 傳 PASS + diff=2.3 → 觸發（對稱性核心測試，等價 6b 但顯式用 PASS）
- **11** `test_rl1b_not_gated_by_confidence_medium_case` — MEDIUM-like 情境 + user 傳 PASS + diff=1.8 + bullpen-slump → 觸發 mid-diff+strong-tag 1★

> 注意：簽章移除 `confidence` 後，fixture 的「HIGH / MEDIUM / LOW」情境不再能透過 kwarg 直接表達。既有 fixture 1-5, 7, 8 需移除 `confidence=...` kwarg；情境層面（HIGH/MEDIUM/LOW）的命名保留在函式名稱與 docstring 即可。對稱性核心測試（10, 11）透過「相同 `diff + tag + user_rl_rec` 組合得到相同結果」來驗證 — 等價於確認 function body 不再對 confidence 產生行為差。

共 12 個 fixture（6 拆成 6a/6b 算 2 個）。

### Step 3 — 文檔清理

**必改**：

| 檔案 | 位置 | 變更 |
|------|------|------|
| `reference/workflow.md` | 第 232 列 Phase 4 參數表 `--run-line-rec` 那列 | 「必填」→「可選」；說明「Phase 3 有明確 RL 結論時傳 team abbr；無結論時省略或傳 PASS，RL-1b 依 diff/tag 自主評估」 |
| `scripts/predict.py` | `apply_rl_guardrail` docstring 第 240-255 行 | 更新 Rules 區塊，刪 `confidence=LOW` 條件描述，加「Symmetric to ML/OU」說明 |

**不改**：

| 檔案 | 原因 |
|------|------|
| `docs/specs/2026-04-20-rl-threshold-relaxation.md` | 歷史決策記錄；本 spec 以 supersede 章節註記取代部分 |
| `reference/prediction.md` | grep 結果未見「RL 在 LOW 一律 PASS」的描述（policy 原本只在 code 裡）；無需改 |
| `scripts/review_stats.py` / `scripts/upload_prediction.py` / `scripts/odds_analyzer.py` | schema 不變，透傳邏輯不受影響 |
| `scripts/clv.py` / `scripts/backfill_clv.py` | 不依賴 confidence |

**需 grep 確認**（實作時執行）：`SKILL.md`、`reference/pitfalls.md`、`reference/matchup-factors.md` 是否有「RL 在開季/LOW 一律 PASS」描述；若有同步清理。

## 行為差（vs 現況）

### Matrix

| 情境 | 現況 | 新設計 | 差異 |
|------|------|--------|------|
| confidence=LOW + user PASS + diff/tag 符合 | PASS | **RL-1b 觸發** | ✅ 4/20 本議題核心 |
| confidence=LOW + user team abbr | RL-1 砍成 PASS | **尊重 user abbr** | ✅ 尊重 Phase 3 判斷 |
| confidence=HIGH + user PASS + diff/tag 符合 | PASS | **RL-1b 觸發** | ✅ 對稱性帶來的新行為 |
| confidence=HIGH + user team abbr | 尊重 | 尊重 | 不變 |
| confidence=MEDIUM（INSUFFICIENT_SAMPLE）+ user PASS + diff 符合 | PASS | **RL-1b 觸發** | ✅ 對稱性新行為 |
| 任意 confidence + user 省略 + diff < DIFF_MIN | PASS | PASS | 不變（自然 PASS） |

### 4/20 9 場重跑預測

基於現有 `adj_home` / `adj_away` / `tags` 模擬新 guardrail：

| 場次 | 結果 |
|------|------|
| HOU@CLE | ✅ **AWAY 2★** big-diff |
| ATH@SEA | ✅ **HOME 2★** big-diff（免 tag） |
| ATL@WSH | ✅ **AWAY 2★** big-diff |
| LAD@COL | ✅ **AWAY 2★** big-diff |
| STL@MIA | ✅ **HOME 1★** mid-diff+strong-tag |
| DET@BOS | ❌ PASS（diff=1.00 < 1.5） |
| BAL@KC | ❌ PASS（diff=0.00） |
| PHI@CHC | ❌ PASS（diff=0.40） |
| TOR@LAA | ❌ PASS（diff=0.30） |

5/9 解鎖，4/9 自然 PASS。

## 回測與驗證

### 既有回測資料

`scripts/_backtest_rl_relaxation.py` 已驗證 4/18–4/19 25 場 LOW 場次，方案 C 觸發 5 場 4W-1L（80%）。此結果仍適用於新 spec 的 LOW 路徑（因為新規則在 LOW 的行為跟舊 RL-1b 相同：diff/tag 符合就觸發）。

### HIGH / MEDIUM 路徑無回測樣本

**誠實揭露**：
- 4/18–4/19 25 場全部是 LOW（新 skill schema 分界後的樣本）
- HIGH / MEDIUM 路徑在當前資料集**沒有任何回測**
- 短期內（4–5 月）實務影響為 0（因為賽季場次 < 30 都會觸發 `INSUFFICIENT_SAMPLE`，confidence ≠ HIGH）
- 5 月後隨 `cross_validation` 從 INSUFFICIENT_SAMPLE 過渡到 CONSISTENT / DIVERGENT，HIGH 路徑的觸發情境才會自然出現

**觀察責任歸於使用者既有 `mlb-post-game-review` 每日流程**；若 HIGH 路徑長期表現顯著差於 LOW，回來加 `cross_validation == CONSISTENT` 等 filter（見 §風險）。

### 驗證檢查清單

- [ ] `pytest scripts/tests/test_predict_snapshot.py` 全綠（含 12 fixture）
- [ ] `pytest scripts/tests/` 全綠（test_clv / test_kelly / test_backfill_clv 等不受影響）
- [ ] 4/20 9 場用新 `predict.py` 重跑 `--save`：5 場 `rl_override.active=true`，4 場 `rl_override.active=false`，模型得分與輸出結構與現況一致（除 `rl_override` 區塊）
- [ ] 新 `rl_override.path` 分布：`big-diff` × 4、`mid-diff+strong-tag` × 1（對照 4/20 matrix）
- [ ] `kelly.warnings` 狀態：觸發場移除 `rl_guardrail_pass`（因為 `final_rl_rec != "PASS"`）；PASS 場保留

## 風險

| 風險 | 嚴重度 | 緩解 |
|------|--------|------|
| HIGH / MEDIUM 路徑無回測樣本，新觸發場次可能洗單 | 中 | 4–5 月全部 INSUFFICIENT_SAMPLE（MEDIUM），HIGH 預計 5 月後才出現；每日 `mlb-post-game-review` 觀察；若顯著劣化加 `cross_validation in ("CONSISTENT", "INSUFFICIENT_SAMPLE")` filter |
| LOW + user team abbr 原擋、現放行 | 低 | 尊重 Phase 3 判斷是對的方向；若洗單明顯再考慮對 user rec 做 sanity check（例如 user team abbr 與 diff_side 反向時警告） |
| 砍掉 `confidence` 參數破壞呼叫端 | 低 | 呼叫點只有 `predict.py:1097` 一處；Step 1 同步更新 |
| `kelly.warnings=["rl_guardrail_pass"]` 觸發情境變窄 | 低 | semantic 不變（仍代表 final_rl_rec=PASS）；觸發場次變少 = 更準確 |
| 下游 `judge_rl` / `upload_prediction` / `review_stats` 相容性 | 低 | schema 不變（`rl_override` / `run_line_rec` 值域皆不改）；不需改下游 |
| 歷史 spec（2026-04-20-rl-threshold-relaxation）未 supersede 清楚 | 低 | 本 spec Context 章節明示 supersede 範圍；舊 spec 不改 |
| `predict.py` docstring / workflow.md 文字不同步 | 中 | Step 3 列出所有清理位置；實作階段 grep 全倉 double-check |

**回滾策略**：單一 commit 回滾。若發現顯著劣化：
1. `git revert <commit>` 回到現況 RL-1b（LOW 專屬放寬）
2. 或在 code 層加 `cross_validation in ("CONSISTENT", "INSUFFICIENT_SAMPLE")` filter（保留對稱原則但縮小觸發範圍）

## 執行順序（TDD）

1. **測試先行**：更新 `test_predict_snapshot.py`
   - 既有 fixture 1-5, 7, 8：移除 `confidence` kwarg（保留情境）
   - fixture 6 拆成 6a/6b（或改名 + 新增）
   - 新增 fixture 9, 10, 11
   - 此時**應 FAIL**（code 未改）
2. **Code 重寫**：`scripts/predict.py`
   - `apply_rl_guardrail` 簽章移除 `confidence` 參數
   - Function body 按 Step 1 重寫（刪 RL-1、改 RL-1b gate）
   - 呼叫點（`predict.py:1097`）同步更新
   - Docstring 更新
3. **驗證測試**：全部 fixture 應 PASS
4. **Regression**：`pytest scripts/tests/` 全綠
5. **E2E**：4/20 9 場重跑 `predict.py --save` → 5 場 override 觸發、4 場自然 PASS
6. **文檔清理**：
   - `reference/workflow.md:232` 參數表
   - grep `SKILL.md` / `reference/pitfalls.md` / `reference/matchup-factors.md` 找「RL / 開季 / LOW」同義描述
7. **Commit**：**單一 commit** 方便回滾，commit message 引用本 spec 路徑

## 關鍵檔案

**直接修改**：
- `scripts/predict.py`
  - `apply_rl_guardrail` (225-345) — 重寫（簽章 + 邏輯 + docstring）
  - `main()` (~1097) — 呼叫點 kwargs 更新
- `scripts/tests/test_predict_snapshot.py`
  - (486+) — fixture 1-5, 7, 8 更新；6 拆成 6a/6b；新增 9, 10, 11
- `reference/workflow.md` (232) — 參數表「必填」→「可選」

**已驗證下游相容（不改）**：
- `scripts/review_stats.py` — `judge_rl` team abbr 解析
- `scripts/upload_prediction.py` — 透傳 `run_line_rec`
- `scripts/odds_analyzer.py` — 內部 label 與 rec 字串解耦
- `scripts/clv.py` / `scripts/backfill_clv.py` — 不讀 confidence

**新增**：無（本 spec 不引入新檔案）

## Open questions

### ✅ 全部已在 brainstorming 階段拍板（2026-04-21 session）

- **~~根因共識~~** → 真正根因是市場間不對稱性（RL 綁 confidence，ML/OU 不綁）。PASS-as-unspecified 只是中層症狀。
- **~~修法範圍~~** → A（RL 對稱化，scope 包含：gate 重寫 + fixture 重構 + workflow 清理）。
- **~~是否留「強制關閉 RL-1b」逃生門~~** → **不留**。RL 是下注決策，使用者永遠可 veto（看到推薦但不跟單）。若未來真有需求再加 `--no-rl-override` flag（5 行 code 成本，符合 YAGNI）。
- **~~保留 4/20 spec 的 Q3/Q4 語義~~** → 完全保留（`kelly_available`、`pw_diff_direction_mismatch`）。
- **~~保留 DIFF_MIN/BIG/STAR 門檻~~** → 是，門檻本身沒被證實錯誤，動它會讓本 spec 跨入另一個 scope。
- **~~是否同時加入 edge-vs-Pinnacle 的 odds-driven RL 推薦~~** → 否，YAGNI；若未來需要走另一個 spec（例如 `rl-edge-based-recommendation`）。
