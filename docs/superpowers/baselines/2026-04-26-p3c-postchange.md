# P3c Post-Change Run — Current HEAD State

> Subagent dispatch on current repo (HEAD = 89d448d, post-P3a + P3b + P3b polish).
> Date: 2026-04-26
> Subagent type: general-purpose (sonnet)

## T1 — BABIP 高極端

Setup: 主隊 PHI 近 7 天 BABIP = .395，連勝 5 場。

- [YES] Phase 3.4 偵測 BABIP 高極端 → 觸發 B10 TaskCreate：workflow.md §B10 明確規定 BABIP ≥ .370 時立即 TaskCreate
- [YES] phase3_summary.md 含 §BABIP 回歸判定：B10 task description 強制
- [YES] 不將 PHI 標為 Hot：matchup-factors.md §BABIP 回歸檢查 明確
- [YES] 預測 Run Value 不加 +0.5 Hot 修正：prediction.md 信號表標注「需 BABIP 反向檢查」

**PASS**

## T2 — BABIP 低極端

Setup: 客隊 NYM lineup BABIP = .245，連敗 4 場。

- [YES] Phase 3.4 偵測 BABIP 低極端 → 觸發 B10 TaskCreate
- [YES] phase3_summary.md 含 §BABIP 回歸判定
- [YES] 不將 NYM 標為 Cold
- [YES] 預測 Run Value 不扣 -0.5 Cold 修正

**PASS**

## T3 — ERA-xERA 落差

Setup: 主隊投手 ERA 2.80 / xERA 4.50，IP = 38。

- [YES] Phase 2 Step 2 閘門：偵測 |ERA-xERA| ≥ 1.5 → 必須補跑 pitcher_stats.py --year 2025
- [YES] TaskCreate B7（補跑 YoY 對比）
- [YES] phase3_summary.md §YoY 對比結論
- [YES] 不通過閘門前不得進 Phase 3.5

**PASS**

## T4 — 牛棚雙向閘門

Setup: 客隊 NYM 牛棚 2 名核心 IL（Closer + Primary Setup）。

- [YES] 同時計算 OU 修正 +0.5 run + ML 修正 -3%
- [YES] TaskCreate B9
- [YES] phase3_summary.md §牛棚雙向修正值

**PASS**

## T5 — D3 對立方向

Setup: formula home_win_pct = 65%。

- [YES] ml_rec 為主隊縮寫 (PHI)：workflow.md §Phase 4 明確「HOME / AWAY 字面值會被 reject」
- [YES] run_line_rec ∉ {NYM, NYM +1.5, AWAY +1.5}：D3 ≥ 60% 不得推「對方受讓」

**PASS**

## T6 — D5 比分一致性

Setup: adjusted_total = 8.2，OU line = 9.5（差距 1.3，< 1.5 噪音閾值）。

- [YES] 推 ou_rec: PASS：D2 + D5 + O/U 星級表三層
- [YES] 不推 OVER（adjusted < line）：D5「修正後總分 ≤ O/U line → 不得推 Over」

**PASS**

---

## Summary

- T1: PASS
- T2: PASS
- T3: PASS
- T4: PASS
- T5: PASS
- T6: PASS

**Total: 6/6 PASS — matches baseline (pre-P3 6/6 PASS)**

P3 trim did NOT break any rule discoverability. 紀律規則精煉後仍正確觸發。

## Post-trim rule discoverability verdict

| 規則 | 觸發點檔 | Canonical 檔 | 第三層防護 |
|------|---------|------------|-----------|
| BABIP 閾值 (.260 / .370) | `workflow.md §B10` | `matchup-factors.md §BABIP 回歸檢查` | `flags-checklist.md §3` |
| ERA-xERA ≥ 1.5 | `workflow.md §Phase 2 Step 2 閘門` | `matchup-factors.md §YoY Statcast 驗證` | `flags-checklist.md §13` |
| 牛棚雙向 (B9) | `workflow.md §B9`（含「同時計算 ML 修正 + OU 修正」） | `matchup-factors.md §牛棚傷兵累計效應` | `flags-checklist.md §4` |
| D3 對立方向 | `workflow.md §Phase 4 --ml-rec`（拒 HOME 字面值） | `prediction.md §D3`（含 60%/55% 表格） | `flags-checklist.md §5` |
| D5 噪音底線 (1.5 run) | `prediction.md §D2`（prose） | `prediction.md §D5`（表格） | `prediction.md §O/U 星級表` |

REFACTOR not needed. P3c can commit without further changes.
