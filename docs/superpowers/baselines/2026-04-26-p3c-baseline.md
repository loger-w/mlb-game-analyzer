# P3c Baseline Run — Pre-P3 State

> Subagent dispatch on `/tmp/mlb-baseline/` (git worktree at commit `1e35517` = P2 末尾，pre-P3a/P3b state).
> Date: 2026-04-26
> Subagent type: general-purpose (sonnet)

## T1 — BABIP 高極端

Setup: 主隊 PHI 近 7 天 BABIP = .395（高於 .370 極端閾值），連勝 5 場，分析是否正確抑制 Hot 標記與 +0.5 Run Value 修正。

Expected behavior checklist:
- [YES] Phase 3.4 偵測 BABIP 高極端 → 觸發 B10 TaskCreate：workflow.md B10 明確規定「偵測任一打線近 7 天 BABIP ≥ .370 時，立即 TaskCreate」，PHI recent_babip = .395 ≥ .370，觸發條件清楚滿足。
- [YES] phase3_summary.md 含 §BABIP 回歸判定：workflow.md B10 規定 task description 要求「結論寫入 phase3_summary.md §BABIP 回歸判定」。
- [YES] 不將 PHI 標為 Hot：matchup-factors.md 明確規定 BABIP ≥ .370 時「不加 Hot run value，標註『回歸預期下降』」。
- [YES] 預測 Run Value 不加 +0.5 Hot 修正：prediction.md 信號表規定「需 BABIP 反向檢查」，BABIP 反向檢查否決 Hot → 修正不適用。

**PASS**

## T2 — BABIP 低極端

Setup: 客隊 NYM 近 7 天 BABIP = .245，連敗 4 場。

- [YES] Phase 3.4 偵測 BABIP 低極端 → 觸發 B10 TaskCreate
- [YES] phase3_summary.md 含 §BABIP 回歸判定
- [YES] 不將 NYM 標為 Cold
- [YES] 預測 Run Value 不扣 -0.5 Cold 修正

**PASS**

## T3 — ERA-xERA 落差

Setup: 主隊投手 ERA = 2.80 / xERA = 4.50（差 1.70），IP = 38.0，prior_year ERA = 3.50。

- [YES] Phase 2 Step 2 閘門：偵測 |ERA-xERA| ≥ 1.5 → 必須補跑 pitcher_stats.py --year 2025
- [YES] TaskCreate B7（補跑 YoY 對比）
- [YES] phase3_summary.md §YoY 對比結論
- [YES] 不通過閘門前不得進 Phase 3.5

**PASS**

## T4 — 牛棚雙向閘門

Setup: 客隊 NYM 牛棚 core_il_count = 2（Closer + Primary Setup IL）。

- [YES] 同時計算 OU 修正 +0.5 run + ML 修正 -3%（該隊勝率下修）
- [YES] TaskCreate B9（牛棚雙向修正值）
- [YES] phase3_summary.md §牛棚雙向修正值

**PASS**

## T5 — D3 對立方向

Setup: formula home_win_pct = 65%（PHI 主場強勢）。

- [YES] ml_rec 為主隊縮寫 (PHI)，不得是字面值 'HOME'：workflow.md Phase 4 明確說明「HOME / AWAY 字面值會被 reject」
- [YES] run_line_rec 為 PHI / PHI -1.5 / PASS 任一，不得為 NYM / NYM +1.5：D3 home_win_pct ≥ 60% 時不得推「對方受讓」

**PASS**

## T6 — D5 比分一致性

Setup: formula adjusted_total = 8.2，OU line = 9.5（差距 1.3，< 1.5 噪音閾值）。

- [YES] 推 ou_rec: PASS（差距 < 1.5）：D2 + D5 雙重觸發 PASS
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

**Total: 6/6 PASS**

PRE-P3 規則文件對這 6 個場景的覆蓋是完整的，可作為 POST-P3 改版後行為對比的 baseline。

| 場景 | 主要規則來源 |
|------|------------|
| T1/T2 BABIP | `matchup-factors.md §BABIP 回歸檢查` + `workflow.md B10` + `flags-checklist #3` |
| T3 ERA-xERA | `workflow.md Phase 2 Step 2 閘門` + `B7 TaskCreate` + `flags-checklist #13` |
| T4 牛棚雙向 | `matchup-factors.md §牛棚傷兵累計效應` + `workflow.md B9` + `flags-checklist #4` |
| T5 D3 | `prediction.md D3` + `workflow.md Phase 4 W2（ml-rec reject HOME）` |
| T6 D5 | `prediction.md D2 + D5` + `O/U 星級表` |
