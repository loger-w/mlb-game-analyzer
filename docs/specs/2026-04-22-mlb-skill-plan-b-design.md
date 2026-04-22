# MLB Skill Plan B 重構設計（瘦身後三層防線）

> **狀態**：Design — 待使用者 review
> **日期**：2026-04-22
> **作者**：Claude Opus 4.7 + Loger
> **前置文件**：
> - `2026-04-22-mlb-skill-rule-inventory.md`（pre-slim 規則全景圖）
> - `2026-04-22-mlb-skill-slimming-design.md`（Phase 0 瘦身 spec）
> - `2026-04-22-mlb-skill-slimming-deletion-list.md`（精確刪除清單）
> - `docs/superpowers/plans/2026-04-22-mlb-skill-slimming.md`（瘦身執行 plan）
> - Brainstorm 工作稿：`C:/Users/Loger/.claude/plans/silly-petting-valiant.md`

---

## 1. Context / Motivation

Phase 0 瘦身已於 2026-04-22 完成（commit `ebaac44 / c947445 / 0c9366c / a8145e8 / 5f64d99`）：刪除 CLV 追蹤、後端上傳、開季保守規則（D1.5 / A4 / D4）、WebSearch 天氣/主審、`assemble_analysis.py`。D1 改 α 實作（直接比 `ml_lean` vs `formula_lean`，不讀 `cross_validation` 字串）。

### Post-slim baseline

- `SKILL.md` 162 行 / `reference/*.md` 1035 行 / `predict.py` 1072 行
- 原 inventory 診斷的 **Y（規則打架/冗餘）/ V（條件觸發無 anchor）/ W（繞過路徑）** 問題仍存在
- cumulative 追蹤問題（#1 主場 2★ / #3 近身戰 / #4 divergent tag / #8 xgb-predicted 矛盾 / #9 `ml_rec` 字面值 / #10 RL 人工繞過）未被瘦身 cover

### 目標

用「三層防線」機制解決剩餘 Y/V/W 問題。**不拆 sub-skill**（B-α 方案：保持 1 個 skill + 內部重構）。

### 為什麼不拆 sub-skill

原 inventory 候選 4 個（analyzer / guardrails / deferred-checks / rationalizations）。post-slim 規模下各候選都 ≤ 80 行，拆分會造成：
- 多一層 invocation overhead
- 規則散在多處反而難找
- guardrails 本身是 code，skill 化沒意義
- rationalizations 去重後只剩 13 條，單獨 skill 過小

B-α（1 skill + 內部重構）最能兼顧 YAGNI 與 enforcement 強度。

---

## 2. 三層防線架構

### 2.1 白話定義（用 ERA-xERA YoY 情境貫穿）

**情境**：Phase 2 Step 2 偵測到投手 `|ERA − xERA| ≥ 1.5`，要求補跑 prior year YoY 對比。此規則在 post-slim 仍靠 skill 自律，Claude 有時跳過。三種堵法：

| 層 | 白話 | 強制來源 | 適用規則類型 |
|----|------|---------|------------|
| **1. python 硬擋** | `predict.py --save` 開跑前自己驗：當季有 `ERA−xERA ≥ 1.5` 但 `$GAME_DIR/{side}_pitcher_{YYYY-1}.json` 不存在 → 直接 exit error。Claude 完全沒機會跳。必要時**連動佐證腳本**（指示該跑哪個腳本） | python 層（無法繞過，除非改 code） | W 類繞過全部 + 可從 JSON 推斷的 Y 類 |
| **2. md 檔擴充** | Claude 判斷結果寫進現有 `phase3_summary.md` 的新 section；`predict.py --save` 用 regex 檢查 section header 存在，缺則拒跑 | python grep 檢查（結構檢查，不驗語義） | V 類需 Claude 人工判斷但結果可序列化 |
| **3. TaskCreate forcing function** | `workflow.md` 觸發點加「TaskCreate 指示」；Claude 自己 create，Phase 轉換前 TaskList 檢查清空 | Claude workflow 層（skill prompt + `superpowers:using-superpowers` 強制） | 跨 phase / 跨工具呼叫的 V 類 |

### 2.2 重要決議（2026-04-22 user confirmed）

- **檔案落地 = 擴現有 `phase3_summary.md`**，不新增 `phase3_checks.json` 或其他 anchor 檔
- **A13 D4 不補**；瘦身已整條刪，同類補洞走第 1 層處理 cumulative Y 類新缺口
- **RL 統一走 RL-1b + Strong tag**，廢除 `--run-line-rec` / `--run-line-stars` CLI args（W1 徹底消除）
- **TaskCreate 不用 blocks/blockedBy**（依 Claude TaskList 自查 + skill prompt 硬規則）

---

## 3. Scope

### 3.1 In Scope

**W 類繞過（第 1 層）**
- W1 `--run-line-rec` 繞 gate → **廢除 CLI args**
- W2 `ml_rec` 字面值 "HOME"/"AWAY" → schema hard exit
- W3 `--signal-adjustments` JSON 無 schema → allowlist stderr warning
- W4 `--game-data` 路徑腦補 → regex hard exit

**cumulative Y 類新缺口（第 1 層）**
- Y-new-2（#3 近身戰 <0.5）→ `ml_stars_cap = 1`
- Y-new-3（#4 divergent tag）→ `ml_stars_cap = 2`
- Y-new-1（#1 主場 2★）→ 只加 tag `"home-2star-risk"` 不 cap
- Y2（#8 xgb-predicted 矛盾）→ **force PASS** + tag `"xgb-predicted-divergent"`

**V 類 anchor 化（第 2 + 3 層）**
- B7 ERA-xERA YoY 補跑（第 1 + 2 + 3 全套）
- B9 牛棚雙向閘門（第 2 + 3）
- B10 BABIP 回歸（第 2 + 3）

**Y1 三清單去重（Section 1）**
- 新檔 `reference/flags-checklist.md`（13 條獨立）
- `pitfalls.md` 砍 Common Mistakes
- `SKILL.md` 刪 Rationalizations + Red Flags，改為 top 3 flags + link

### 3.2 Out of Scope

- 重新引入 D4 或任何開季保守規則（user 瘦身時明確刪除）
- cumulative #2 OU PASS 場誤差（模型問題，非 guardrail）
- cumulative #6 冷天氣訊號（天氣已刪）
- cumulative #7 小樣本 xERA 權重（L4 係數校準、另外 track）
- Kelly 參數 tune / RL 門檻重評
- 1★ cap 0 規則（cumulative 未觸發）
- Sub-skill 拆分（B-β / B-γ 方案 2026-04-22 駁回）

### 3.3 保留不動

- 核心 guardrail G1-G9（D1 α / A5 / A6 / OU-1/2/3 / RL-1b auto / RL-2 / A14 Kelly PASS）
- Phase 1-4 流程骨架
- BvP / `role_change` / YoY 觸發條件（強化 anchor 機制、不改邏輯本身）
- `fetch_results.py` / `summarize_predictions.py` / `review_stats.py`
- `mlb-post-game-review` skill

---

## 4. Design Decisions

### 4.1 三清單去重合併（Section 1）

**新檔 `reference/flags-checklist.md`**（~60 行，13 條獨立規則）

去重後的 13 條 Flags（每條一段簡述，不分「想法/行為/Phase 檢查」三欄）：

1. 記憶/訓練資料代替腳本 API
2. BvP 樣本 <15 硬推結論
3. Hot/Cold 判定未查 BABIP
4. 牛棚傷兵只修 O/U 未修 ML
5. 同場推對立方向
6. 不寫 `phase3_summary.md`
7. 跳過 Roster 檢查
8. Agent 子代理跑 WebSearch
9. 省 `--game-data` / 腦補路徑
10. shell redirect `>` 取代 `--output/-o`
11. WebSearch 失敗繼續分析
12. 中文對話用英文輸出
13. ERA-xERA 落差寫「風險提示」代替驗證

**`reference/pitfalls.md`**（55 → ~40 行）
- 刪 `## Common Mistakes` 整節（12 條已併入 `flags-checklist.md`）
- 保留 `## Edge Cases`（11 條）+ `## 具體修正係數備忘`

**`SKILL.md`**（162 → ~110 行）
- 刪 §Rationalizations 表（L55-L71）
- 刪 §Red Flags 清單（L73-L91）
- §Common Pitfalls 改為「最高優先 3 項 + link 到 `reference/flags-checklist.md` 完整清單」

### 4.2 W 類 code 下沉（Section 2，第 1 層）

**W1 RL 統一走 RL-1b + Strong tag**

`predict.py` argparse 移除：
- `--run-line-rec`
- `--run-line-stars`

`apply_rl_guardrail` 函數簽名移除 `user_rl_rec` / `user_rl_stars` 參數（函數內部 `if user_rl_rec in (None, "PASS")` 整段 gate 刪除，一律走 auto override path）。

`reference/workflow.md` Phase 4 參數表刪除兩個 args 相關描述。`SKILL.md` 同步刪除提及。

**W2 `ml_rec` schema validation**

`predict.py` main() argparse 後加：

```python
valid_ml_rec = set(TEAM_ABBREV.values()) | {"PASS", None}
if args.ml_rec not in valid_ml_rec:
    sys.exit(
        f"⛔ --ml-rec 必須是 team abbr（如 NYY）或 PASS，收到 {args.ml_rec!r}\n"
        f"  合法值: {sorted(v for v in valid_ml_rec if v) + ['PASS']}"
    )
```

**W3 `signal_adjustments` schema allowlist**

定義（具體 key 集合在 implementation phase scan 現有 prediction.json 後確定）：

```python
SIGNAL_KEYS_ALLOWLIST = {
    "bullpen_il_home", "bullpen_il_away",
    "platoon_home", "platoon_away",
    "park_factor_adj",
    "hot_offense_home", "hot_offense_away",
    "cold_offense_home", "cold_offense_away",
    "pitcher_rest_home", "pitcher_rest_away",
    "doubleheader_g2",
    "yoy_home", "yoy_away",
    # ... 其他現行 key
}
```

解析 `args.signal_adjustments` JSON 後，unknown key → stderr warning（不 exit，允許新 signal 擴充）。

**W4 `--game-data` 路徑 regex**

```python
GAME_DATA_PATTERN = re.compile(
    r"analysis-data[/\\]\d{4}-\d{2}-\d{2}[/\\][A-Z]{2,3}@[A-Z]{2,3}(-G[12])?[/\\]merged\.json$"
)

if not GAME_DATA_PATTERN.search(args.game_data.replace("\\", "/")):
    sys.exit(f"⛔ --game-data 路徑不符規範: {args.game_data}")
```

支援 Windows 反斜線 + absolute / relative path。

### 4.3 連動 YoY 佐證腳本（Section 2，第 1 層）

`predict.py --save` 讀 `merged.json` 後：

```python
def pitcher_triggers_yoy(pitcher_data: dict) -> bool:
    """回傳 True 如果觸發 B7 YoY 補跑紀律。"""
    era = pitcher_data.get("era")
    xera = pitcher_data.get("xera")
    ip = pitcher_data.get("ip")
    prior_era = pitcher_data.get("prior_year", {}).get("era")
    if era is not None and xera is not None and abs(era - xera) >= 1.5:
        return True
    if (ip is not None and ip < 30
        and era is not None and prior_era is not None
        and era < prior_era - 1.0):
        return True
    return False
```

若任一投手觸發 + `$GAME_DIR/{side}_pitcher_{YYYY-1}.json` 不存在 → exit error + 印具體命令：

```
⛔ {pitcher_name} |ERA-xERA|=1.62 觸發 B7 YoY 紀律，但缺 prior year data：
請先跑：
  pitcher_stats.py --name "{pitcher_name}" --year 2025 -o $GAME_DIR/home_pitcher_2025.json
```

保留 `--skip-yoy-check` flag（edge case 測試用）。

### 4.4 cumulative Y 類新 L1 code 規則（Section 2，第 1 層）

實作位置：`predict.py` `if args.save:` 區塊內。以下為意圖性代碼；精確 line 數交由 `writing-plans` 決定。

**Y2（xgb-predicted 矛盾 force PASS）** — 在星級 cap 計算區塊，D1 α 之後、A5 之前：

```python
# Y2: xgb_home_lean vs predicted_winner 矛盾（signal 翻轉 xgb）→ 強制 PASS
y2_triggered = False
if ml_pred and formula_pred:
    xgb_home_lean = "HOME" if ml_pred["home_win_pct"] > 50 else "AWAY"
    pred_winner = result["final"]["recommended_winner"]
    if xgb_home_lean != pred_winner:
        ml_stars_cap = 0
        force_ml_pass = True
        y2_triggered = True
        cap_reasons.append(
            f"xgb_home_lean={xgb_home_lean} vs predicted_winner={pred_winner} "
            f"方向矛盾 強制 PASS"
        )
```

**注意**：Y2 跟 D1 α 觸發情境不同且獨立：
- D1 α：`ml_lean != formula_lean`（兩 model 意見不同）
- Y2：`xgb_home_lean != predicted_winner`（`signal_adjustments` 翻轉 xgb 方向）
- A6（既有）：`args.ml_rec != predicted_winner`（user input vs model 輸出）

三者可同時或獨立觸發。

**Y-new-2（近身戰 cap）** — 在 A5 之後：

```python
# Y-new-2: 近身戰（|adj 比分差| < 0.5）上限 1（cumulative #3 連 4 天觸發）
if abs(adj_home - adj_away) < 0.5:
    ml_stars_cap = min(ml_stars_cap, 1)
    cap_reasons.append(
        f"近身戰 |adj 比分差|={abs(adj_home - adj_away):.2f} < 0.5 上限 1"
    )
```

**Y-new-3（divergent tag cap）** — 在 A5 之後。檢查 `args.tags`（user-supplied）而非 `all_tags`，因為 "divergent" 是 Phase 3 Claude 手動加的 user tag，不是 `compute_trend_tags(data)` 產生的 trend tag。此規則可不依賴 `all_tags` 構造順序：

```python
# Y-new-3: user-supplied 'divergent' tag → 上限 2（cumulative #4 推薦場 0W-4L）
user_tags_raw = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
if "divergent" in user_tags_raw:
    ml_stars_cap = min(ml_stars_cap, 2)
    cap_reasons.append("'divergent' tag 上限 2")
```

**Y-new-1（主場 2★ audit tag）** — 在 `all_tags` 構造後、record dict 組成前。條件加 `final_ml_rec != "PASS"`（若 ml 已 PASS 推薦已消，tag 無意義）：

```python
# Y-new-1: 主場 2★ 觀察 tag（cumulative #1 連 4 天觸發，條件難定義、先觀察）
if (result["final"]["recommended_winner"] == "HOME"
    and final_ml_stars == 2
    and final_ml_rec != "PASS"
    and "home-2star-risk" not in all_tags):
    all_tags.append("home-2star-risk")
```

**Y2 audit tag** — 在 `all_tags` 構造後。用 `y2_triggered` flag 避免重算：

```python
# Y2 audit tag（觸發 flag 在前面 force PASS block 設定）
if y2_triggered and "xgb-predicted-divergent" not in all_tags:
    all_tags.append("xgb-predicted-divergent")
```

### 4.5 `phase3_summary.md` 擴充（Section 3，第 2 層）

**`reference/workflow.md` Phase 3.5 章節**加新 MUST contain 條件式 3 項：

| 觸發條件（predict.py 自判） | 必要 section header |
|-----------------------|-------------------|
| `pitcher_triggers_yoy(home_pitcher)` 或 `pitcher_triggers_yoy(away_pitcher)` | `## YoY 對比結論` |
| `lineup_triggers_babip(home_lineup)` 或 `lineup_triggers_babip(away_lineup)`（近 7 天 BABIP ≤ .260 或 ≥ .370） | `## BABIP 回歸判定` |
| `args.signal_adjustments` 含 `bullpen_il_home` 或 `bullpen_il_away` key | `## 牛棚雙向修正值` |

**觸發判斷函式**：

```python
def pitcher_triggers_yoy(pitcher: dict) -> bool:
    """B7 YoY 補跑觸發條件（同 §4.3）。"""
    era = pitcher.get("era")
    xera = pitcher.get("xera")
    ip = pitcher.get("ip")
    prior_era = (pitcher.get("prior_year") or {}).get("era")
    if era is not None and xera is not None and abs(era - xera) >= 1.5:
        return True
    if (ip is not None and ip < 30
        and era is not None and prior_era is not None
        and era < prior_era - 1.0):
        return True
    return False


def lineup_triggers_babip(lineup: dict) -> bool:
    """B10 BABIP 回歸觸發條件（近 7 天 BABIP 極端）。"""
    recent_babip = lineup.get("recent_babip")
    if recent_babip is None:
        return False
    return recent_babip <= 0.260 or recent_babip >= 0.370
```

**`predict.py --save` 檢查邏輯**：

```python
from pathlib import Path
import re

game_dir = Path(args.game_data).parent
phase3_path = game_dir / "phase3_summary.md"
if not phase3_path.exists():
    sys.exit(f"⛔ {phase3_path} 不存在 — Phase 3 未完成")

content = phase3_path.read_text(encoding="utf-8")

required_sections = []
if (pitcher_triggers_yoy(data.get("home_pitcher", {}))
    or pitcher_triggers_yoy(data.get("away_pitcher", {}))):
    required_sections.append("## YoY 對比結論")
if (lineup_triggers_babip(data.get("home_lineup", {}))
    or lineup_triggers_babip(data.get("away_lineup", {}))):
    required_sections.append("## BABIP 回歸判定")
sig_adj = json.loads(args.signal_adjustments or "{}")
if "bullpen_il_home" in sig_adj or "bullpen_il_away" in sig_adj:
    required_sections.append("## 牛棚雙向修正值")

missing = [s for s in required_sections
           if not re.search(rf"^{re.escape(s)}\b", content, re.M)]
if missing:
    sys.exit(f"⛔ phase3_summary.md 缺 section: {missing}")
```

**原則**：只檢 section header 存在（結構檢查），不驗語義（避免誤殺）。

保留 `--skip-phase3-check` flag（edge case 測試用）。

### 4.6 `merge_game_data.py` 擴欄（Section 3 依賴）

為讓 `predict.py` 能判觸發條件，merged.json 加新欄位：

- `home_pitcher.era` / `.xera` / `.ip`（可能已有，確認）
- `home_pitcher.era_xera_delta` = `abs(era - xera)`（新）
- `home_pitcher.prior_year.era`（若 `home_pitcher_{YYYY-1}.json` 存在，merge 時 include；否則 `null`）
- `home_lineup.recent_babip` = 近 7 天 BABIP（從 `lineup_analyzer.py` 輸出取）
- away 同理

### 4.7 V 類 TaskCreate 樣板（Section 4，第 3 層）

`reference/workflow.md` 三個觸發點加 instruction block：

**B7（Phase 2 Step 2）**

```markdown
### Step 2 閘門擴充（B7 YoY 補跑）

⛔ 任一投手 `|ERA − xERA| ≥ 1.5` 或 `IP < 30 且 ERA 低於 prior year ≥ 1.0` 時，**同步**：

1. 執行 `pitcher_stats.py --year {YYYY-1} -o $GAME_DIR/{side}_pitcher_{YYYY-1}.json`
2. TaskCreate（forcing function）：
   - `subject`: "補跑 {side} YoY 對比（{pitcher_name}）"
   - `description`: "對比 5 項 Statcast 指標（avg_velo / pitch_types / whiff_pct / hard_hit_pct / xera）；結論寫入 phase3_summary.md §YoY 對比結論"

此 task 必須 complete 才能進 Phase 3.5（summary 存檔）。
```

**B9（Phase 3.2 牛棚）**

```markdown
### 牛棚雙向閘門擴充（B9）

⛔ 偵測核心（Closer / Primary Setup / High-leverage）IL 任一人時，TaskCreate：
- `subject`: "牛棚雙向修正值（核心 {N} 人 IL）"
- `description`: "同時計算 ML 修正（-%）+ OU 修正（+run）；寫入 phase3_summary.md §牛棚雙向修正值；呼叫 predict.py 時 --signal-adjustments 含 bullpen_il_{side}"

此 task 必須 complete 才能進 Phase 3.5。
```

**B10（Phase 3.4 BABIP）**

```markdown
### BABIP 回歸閘門擴充（B10）

⛔ 偵測任一打線近 7 天 BABIP ≤ .260 或 ≥ .370，TaskCreate：
- `subject`: "BABIP 回歸判定（{team} 近 7 天 {value}）"
- `description`: "回歸 ~.300 後判定 Hot/Cold 是否調整；結論寫入 phase3_summary.md §BABIP 回歸判定"

此 task 必須 complete 才能進 Phase 3.5。
```

**Phase 轉換檢查**（Phase 2→3 和 Phase 3→4 章節都加）：

```markdown
⛔ 進入 Phase {N+1} 前必須：
1. TaskList 檢查前 Phase 產生的 V 類 tasks（YoY / 牛棚雙向 / BABIP 回歸）全部 complete
2. 有 pending task 不得進下 Phase
```

---

## 5. Implementation Phases

| Phase | 內容 | Dependencies | 可並行 |
|-------|------|------------|-------|
| **1** | 三清單去重 + `SKILL.md` + `pitfalls.md` 重構 + 新檔 `flags-checklist.md` | 無 | 與 Phase 2 並行 |
| **2** | W + Y 類 code 下沉 + 廢除 RL args（`predict.py`） | 無 | 與 Phase 1 並行 |
| **3** | `merge_game_data.py` 擴欄 + `predict.py` grep 檢查 `phase3_summary.md` | Phase 2 完成（predict.py 基礎清理完） | 否 |
| **4** | `workflow.md` TaskCreate 樣板 + Phase 轉換檢查 | Phase 2 完成（確保 CLI args 廢除同步反映） | 否 |
| **5** | 測試 + 全流程驗證 | Phase 1-4 全完成 | 終點 |

詳細 task 分解 + line numbers 交由 `superpowers:writing-plans` 生成。

---

## 6. 測試策略

### 6.1 單元測試（`scripts/tests/test_predict_snapshot.py` +~85 LOC）

- W2 `ml_rec` schema（"HOME"/"AWAY" 字面值 → sys.exit）
- W3 `signal_adjustments` allowlist（unknown key → stderr warning）
- W4 `--game-data` regex（錯誤路徑 → sys.exit）
- 廢除 `--run-line-rec` / `--run-line-stars`（argparse 不應再含此 args）
- Y-new-2 近身戰 cap（`adj_home=4.2, adj_away=4.5` → `ml_stars_cap=1`）
- Y-new-3 divergent tag cap（tags 含 "divergent" → cap=2）
- Y-new-1 audit tag（HOME predicted + 2★ → tags 含 "home-2star-risk"）
- Y2 xgb-predicted force PASS（xgb 61% HOME + predicted=AWAY → force_ml_pass=True）
- phase3_summary grep（觸發 B7 但缺 `## YoY 對比結論` → sys.exit）

### 6.2 Integration test（手動）

- **場景 A**：挑一場 2026-04-23 比賽，人為觸發 W2（`--ml-rec HOME`）、Y-new-2（比分 4.2/4.5）、Y2（signal 翻轉 xgb）確認正確
- **場景 B 回測 4/21 NYY@BOS**：新 Y2 force PASS 應把原 1W-0L 的 ML 推薦改為 PASS（cumulative 會變 5W-10L）— 驗證行為符合設計
- **場景 C 舊 jsonl 相容**：`review_stats.py --date 2026-04-20` 不 crash

### 6.3 驗收準則

- [ ] 所有 pytest 綠
- [ ] `SKILL.md` ≤ 110 行
- [ ] `pitfalls.md` ≤ 40 行
- [ ] `flags-checklist.md` 13 條 + `pitfalls.md` Edge Cases 搬家完成
- [ ] `grep "run_line_rec\|run_line_stars" scripts/predict.py` 無結果
- [ ] `grep "Rationalizations\|Red Flags" SKILL.md` 無結果
- [ ] 手動跑一場新資料，所有新 guardrail 正確觸發
- [ ] `analysis-logs/cumulative.md` 狀態更新（#10 W1 消除 / #9 W2 修復 / #8 Y2 force PASS）

---

## 7. LOC 變動預估

| 檔案 | Before | After | Delta |
|------|--------|-------|-------|
| `SKILL.md` | 162 | ~110 | **-52** |
| `reference/pitfalls.md` | 55 | ~40 | -15 |
| `reference/flags-checklist.md`（新） | 0 | ~60 | +60 |
| `reference/workflow.md` | 292 | ~330 | +38 |
| `reference/prediction.md` | 345 | ~350 | +5 |
| `scripts/merge_game_data.py` | 272 | ~305 | +33 |
| `scripts/predict.py` | 1072 | ~1140 | +68 |
| `scripts/tests/test_predict_snapshot.py` | 現有 | +85 | +85 |

**核心**：reference/scripts 規則機器「更硬但更短」；`SKILL.md` 純散文規則 -52 行，W 繞過 + Y 類缺口全部 code 下沉。

---

## 8. 相容性與風險

### 8.1 相容性

| 項目 | 策略 |
|------|------|
| 舊 `prediction.json` / `predictions.jsonl` | 保留；新欄位 optional，`review_stats.py` 讀用 `.get()` tolerate |
| `--run-line-rec` / `--run-line-stars` CLI args 廢除 | 現行 skill prompt / workflow.md 引用**必須同步刪**；舊 CLI 呼叫 → argparse 報 unknown arg error（預期行為） |
| 新 merged.json 欄位（`era_xera_delta` / `recent_babip` / `prior_year.era`） | `merge_game_data.py` 擴欄；來源資料缺 → 欄位 `null`；predict.py tolerance |
| `phase3_summary.md` 格式 | 現有 6 個 MUST contain 不動；新 3 個是條件式（觸發才要） |

### 8.2 風險與監控

**R1. W1 廢除 `--run-line-rec` 後，Claude 失去手動 RL 表達能力**
- 影響：Phase 3 若覺得某場讓分特別有價值，無法強制推薦；一律靠 RL-1b auto gate
- 監控：5/1-5/15 RL 戰績若明顯衰退（<20%），考慮引入更寬鬆的 RL-1b 門檻而非回復手動 CLI
- user 2026-04-22 明確接受此 trade-off（理由：前幾天 RL-1b 自動場效果不錯）

**R2. Y2 xgb-predicted force PASS 可能過殺**
- 影響：回測 4/21 NYY@BOS 會從 WIN 變 PASS；cumulative #8 前 3 筆中有 1 筆 WIN（33%）
- 緩解：接受；Y2 規則是為消除方向矛盾優於保留少量 upside，user 明確選 force PASS 而非 audit-only

**R3. `phase3_summary.md` hard exit 可能阻擋測試 / edge case**
- 影響：測試時若沒寫全 MUST contain sections，`predict.py --save` 跑不動
- 緩解：保留 `--skip-phase3-check` flag；正式流程不加

**R4. `merge_game_data.py` 新欄位可能相容舊 merged.json**
- 影響：pre-Plan-B 的 merged.json 無新欄位，predict.py 觸發邏輯全 False（不誤觸發）
- 緩解：讀新欄位用 `.get()`，無值視為不觸發

**R5. TaskCreate forcing function 依賴 Claude 自律使用 TaskList**
- 影響：若 Claude 忽略 TaskList 檢查，第 3 層失效
- 緩解：`superpowers:using-superpowers` skill 會強制 task list；日後若發現遺忘，加 `blocks/blockedBy` 關係

---

## 9. References

- Rule inventory（pre-slim）：`docs/specs/2026-04-22-mlb-skill-rule-inventory.md`
- 瘦身 design：`docs/specs/2026-04-22-mlb-skill-slimming-design.md`
- 瘦身 deletion list：`docs/specs/2026-04-22-mlb-skill-slimming-deletion-list.md`
- 瘦身 implementation plan：`docs/superpowers/plans/2026-04-22-mlb-skill-slimming.md`
- Plan B brainstorm 工作稿：`C:/Users/Loger/.claude/plans/silly-petting-valiant.md`
- Cumulative tracking：`analysis-logs/cumulative.md`
- RL-1b 放寬 spec：`docs/specs/2026-04-20-rl-threshold-relaxation.md`
- RL 對稱化 plan：`docs/superpowers/plans/2026-04-21-rl-symmetrization.md`

---

## 10. 下一步

1. User review 本 spec（此步驟）
2. Approved 後 invoke `superpowers:writing-plans` 生 implementation plan 到 `docs/superpowers/plans/2026-04-22-mlb-skill-plan-b.md`
3. 新 session 執行 implementation plan（TDD，每 task 獨立 commit）
