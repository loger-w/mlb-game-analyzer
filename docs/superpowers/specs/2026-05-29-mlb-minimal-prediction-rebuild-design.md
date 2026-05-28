# MLB 極簡預測重構（Model C / team-level）— 2026-05-29

> 接續並**取代** `2026-05-28-deterministic-prediction-design.md` 的數字模型（§5 `scoring_formula`）。
> **保留**其核心哲學：預測數字由 script 確定性計算、AI 只做敘事;不再用 xwOBA×FIP 乘法公式與信心檔位。

## 1. 目的

把預測從「打線進階數據(xwOBA) × 先發 FIP × 一堆信號」的複雜路線,**砍回最直接的「得分 vs 失分」結果型模型**。同時:

1. 用**一支 orchestrator** 統一整條流程(取代現有約 20 支腳本的鬆散組合)。
2. 輸出改為 **RL(讓分)+ O/U(大小分)**,並與**當下盤口**的 no-vig 機率比,標出 edge。**不輸出 Money line**。
3. 抓預測當下的**兩隊成績 + 盤口**凍結存檔,讓回測可重現、未來可 ablation。
4. SKILL.md 縮到只剩「跑哪行指令 / 在哪建資料夾 / 把結果念給使用者」。

## 2. 背景（為什麼做）

2026-05 回測(n=278,Pinnacle no-vig baseline)證實舊模型壞掉:

| 指標 | 值 | 解讀 |
|---|---|---|
| 方向命中 skill vs market | 54.5% vs 56.2%(edge **−1.7%**) | skill 連「跟著盤口熱門」都打不贏 |
| 信心校準 | LOW 57.4% > MEDIUM 54.9% > **HIGH 52.2%** | **反向**:越有信心越不準,HIGH 宣稱 0.76 實際 0.52 |
| Brier skill vs market | 0.27 vs 0.24 | skill 機率品質更差 |
| 反市場時 | 45.2%(n=62) | 跟市場唱反調時更常錯 |

舊 SKILL.md 的「HIGH 80%」來自早期 n=20 選樣偏差,整月攤開即破。**結論:刪掉舊數字模型重建。**

額外發現(指導本設計):
- 進階數據(xwOBA / 信號 / tier)**從未進過預測數字**——它們只餵 dossier + AI 敘事。約 6,500 行在做「給人看的輸出」,不在做預測。
- platoon / 打線細緻度在現有樣本**無法被證明有用**(舊模型有看打線仍爛;聚合後效應 ~0.2–0.5 R/G,多半在雜訊地板下)。故 v1 不納入,但**凍結存檔留待 ablation**。

## 3. 不在 scope

- **打線進預測**:v1 進攻側用球隊平均得分(RS),不用今日打線。打線**凍結但不進數字**(§8),日後 ablation 裁決。
- **進階數據進預測**:xwOBA / wRC+ / Statcast / 信號 / tier 一律不進 v1 數字。
- **Money line 輸出**:ML 內部算(供方向 sanity),但不對使用者輸出。只給 RL + O/U。
- **係數在 5 月 tune**:所有係數(§5.4)取自先驗,標記為**待回測重新擬合**。v1 先求「乾淨可重現 + 簡單」,不是「證明更準」。
- **Poisson/Skellam 機率**:v1 用常態近似。離散分配當未來升級(§14)。
- **smart-money 盤口移動報告**:`odds/analyze_smart_money.py` + `odds/reports/` 維持為獨立工具,**不接進** predict_game(預設保留、不動)。

## 4. 架構：orchestrator + 小模組

```
scripts/
  predict_game.py   # orchestrator:單場 (--matchup A@H) 或當日全部 (--all)
  fetch_inputs.py   # 回傳一場的模型輸入 + 凍結用的打線快照
  run_model.py      # 純函數:輸入 → 期望得分 → RL/OU/ML 機率。零 I/O
  odds_compare.py   # 找當下 snapshot + 抽 Pinnacle no-vig(RL+總分) + 算 edge
  report.py         # 凍結 features.json + 產 prediction.md(給 AI 敘事)
  config.py         # 所有先驗係數集中於此(便於回測後重新擬合)
```

**資料流:**

```
單場:  predict_game --date D --matchup TB@CLE
          → fetch_inputs → run_model → odds_compare → report
          → 寫 analysis-data/D/TB@CLE/{features.json, prediction.md}

當日:  predict_game --date D --all
          → 拉 MLB schedule(D 當天 gameType=R)→ 逐場跑同一條流程
          → 單場失敗只記錄、不中斷整批
```

**模組契約(各自單一職責、可獨立測試):**

| 模組 | 輸入 | 輸出 | 依賴 |
|---|---|---|---|
| `fetch_inputs` | date, away, home (game_pk) | dict:兩隊 RS(近/季)、RA(近/季,存檔用)、雙方先發 FIP+組件、雙方牛棚 ERA、PF、今日打線快照 | MLB Stats API、`park_factors_lib`、`_team_resolver` |
| `run_model` | 上述 inputs(純值) | μ_home, μ_away, μ_margin, μ_total, p_home_ml, p_home_cover_rl, p_over | `config`、stdlib `statistics.NormalDist` |
| `odds_compare` | date, home, away, model 機率 | no-vig(RL/總分)、edges、所用 snapshot 檔名 | `lib/closing_line`(擴充 RL) |
| `report` | inputs + model + odds + edges | 寫 `features.json` + `prediction.md` | — |

## 5. 數字模型規格

### 5.1 進攻(得分火力)

```
RS_blend(team) = RECENT_W × recent_RS + (1 − RECENT_W) × season_RS
```
`recent_RS` = 近 N 場每場得分;`season_RS` = 整季每場得分。

### 5.2 守備(今日壓制力)

```
pitch_today(team) = SP_W × starter_FIP + BP_W × bullpen_ERA
```
- `starter_FIP` 從 MLB API 季成績組件算:
  ```
  FIP = (13×HR + 3×(BB+HBP) − 2×K) / IP + FIP_CONSTANT
  ```
  不需 pybaseball / Statcast。
- 對方守備刻意**只用先發+牛棚,不用對方 RA**——RA 已含先發過去場次,與 starter_FIP 並用會**雙重計算先發**。RA 仍凍結存檔供 ablation。

### 5.3 期望得分

```
μ_home = RS_blend(home) × pitch_today(away) / LEAGUE_RG × (PF/100)
μ_away = RS_blend(away) × pitch_today(home) / LEAGUE_RG × (PF/100)
```
直覺:我隊火力 × 對方今日投手孱弱度 ÷ 聯盟基準 × 球場。PF 對兩隊同時作用(同一球場)。

### 5.4 期望得分 → 機率(常態近似)

單一旋鈕 `SIGMA_TEAM`,推出 margin 與 total 的 SD:
```
SIGMA = SIGMA_TEAM × √2          # 主客各自得分獨立的近似
μ_margin = μ_home − μ_away
μ_total  = μ_home + μ_away

P(主過 RL)  = 1 − Φ((−rl_point_home − μ_margin) / SIGMA)
              # 主 −1.5 → P(margin>1.5);主 +1.5 → P(margin>−1.5)
P(Over)     = 1 − Φ((total_line − μ_total) / SIGMA)
P(主 ML)    = Φ(μ_margin / SIGMA)        # 內部算,不輸出
```
- 客方機率 = 1 − 主方(±1.5 / X.5 線無 push)。
- 整數總分線(罕見)的 push:常態近似下 P(push)≈0,v1 接受;Poisson 升級時再原生處理。

### 5.5 先驗係數（集中於 `config.py`，全部標記待回測重新擬合）

| 常數 | 起步值 | 來源 / 備註 |
|---|---|---|
| `LEAGUE_RG` | 4.4 | 聯盟每場均分;可由當季 schedule 算 |
| `RECENT_W` | 0.35 | RS 近期權重(季佔 0.65) |
| `SP_W` / `BP_W` | 0.6 / 0.4 | 先發約 6/9 局、牛棚 3/9 |
| `SIGMA_TEAM` | 3.0 | 單隊單場得分 SD(歷史先驗)→ margin/total SD ≈ 4.24 |
| `FIP_CONSTANT` | 3.10 | FIP 聯盟正規化常數 |
| `RECENT_N` | 10 | 近期窗口場數 |

### 5.6 v1 明確排除
- ❌ 信心檔位 LOW/MED/HIGH(回測證實反向,移除)。新「信心」= model 機率本身。
- ❌ 任何信號 / tier / platoon ±run。
- ❌ AI 對數字的 override。

## 6. 輸出規格

`prediction.md`（AI 敘事素材）必含:

```
## {AWAY} @ {HOME} — {date}
- 期望得分:HOME μ.μ / AWAY μ.μ(total μ.μ)
| 市場 | 線 | model 機率 | 市場 no-vig | edge(pp) |
|------|----|-----------|-------------|----------|
| RL HOME {±1.5} | … | 35.5% | 41.0% | −5.5 |
| RL AWAY {∓1.5} | … | 64.5% | 59.0% | +5.5 |
| Over {line}    | … | 47.9% | 52.0% | −4.1 |
| Under {line}   | … | 52.1% | 48.0% | +4.1 |
- 所用盤口 snapshot:{filename}
<!-- AI 敘事:哪邊有正 edge、量級、需注意什麼。不喊「下哪邊」、不硬掰 EV、不提 ML。 -->
```

**紀律(沿用舊 odds workflow):** AI 給 lean + edge,不給明確下注指令;不無中生有 fair odds / EV%;數字一律 script 算。

## 7. 盤口整合

- **抓取**:`odds/fetch_odds.py` 已抓 Pinnacle ML/O-U/**RL** 且寫好 `no_vig_pct`,**不改**。使用者要最新盤口 → SKILL 先跑 fetch_odds 再 predict_game。
- **取用**:`odds_compare` 找「預測當下可用的最新 snapshot」(≤ now、未開打),抽該場 Pinnacle 的 RL line/no-vig + 總分 line/no-vig。
- **擴充** `lib/closing_line.py`:現有 `extract_pinnacle_no_vig` 只抽 ML+總分,**新增抽 RL**(home/away point + no_vig)。
- **回測 baseline 一致性**:features.json **凍結當下實際用的那組盤口**;回測直接讀凍結值,不再回頭找固定 12:00 ET 線(going-forward 更忠實)。

## 8. `features.json` 凍結 schema（回測 + ablation 用）

```jsonc
{
  "schema_version": 2,
  "generated_at_utc": "2026-05-29T...",
  "game": { "date", "game_pk", "home", "away", "venue" },
  "inputs": {
    "home_rs_recent", "home_rs_season", "away_rs_recent", "away_rs_season",
    "home_ra_recent", "home_ra_season", "away_ra_recent", "away_ra_season",  // 存檔,v1 不用
    "home_starter": { "name", "id", "fip", "ip", "k", "bb", "hbp", "hr" },
    "away_starter": { "...同上" },
    "home_bullpen_era", "away_bullpen_era",
    "park_factor", "league_rg_used"
  },
  "lineup_frozen": {            // 供日後 ablation,v1 模型不讀
    "source": "official|projected",
    "home": [ { "order", "name", "id", "ops", "woba" }, "...9 人" ],
    "away": [ "..." ]
  },
  "model": {
    "constants_used": { "...§5.5 當下值..." },   // 重現關鍵:係數重擬合後,舊場仍記得用了什麼
    "mu_home", "mu_away", "mu_margin", "mu_total",
    "p_home_ml", "p_home_cover_rl", "p_over"
  },
  "odds": {
    "snapshot_file",
    "rl":    { "home_point", "home_no_vig", "away_point", "away_no_vig" },
    "total": { "line", "over_no_vig", "under_no_vig" }
  },
  "edges": { "home_rl_pp", "over_pp" }
}
```
缺資料(先發未定 / 盤口未開)→ 對應欄位 `null` + `prediction.md` 標註跳過,不中斷。

## 9. SKILL.md 重寫

縮到骨架:
```markdown
# MLB Game Predictor
## 場景路由
- Step 0:建立 ET_NOW(1 次 tool call,時間正確性,不可省)
- 解析:單場 or 當日(ET)?是否「先抓盤口」?
## 指令
- 先抓盤口(若使用者要):python odds/fetch_odds.py
- 單場: python scripts/predict_game.py --date {ET} --matchup {AWAY}@{HOME}
- 當日: python scripts/predict_game.py --date {ET} --all
  (建立 analysis-data/{date}/{AWAY}@{HOME}/)
## 給使用者
讀該資料夾 prediction.md 念給使用者。AI 只敘事(RL/OU + edge),不改數字、不提 ML。
```
**移除**:校準表、腳本導覽、信心檔位、所有數字邏輯說明(`reference/` 文件退役見 §10)。保留 ET_NOW 時間紀律。

## 10. 刪除範圍（約 6,500 行）

**完全退役(連同各自 tests):**
`pitcher_stats.py`、`dossier_renderer.py`、`summary_renderer.py`、`signals_lib.py`、`lib_tier_v2.py`、`lib_role_tagging.py`、`roster_checker.py`、`merge_game_data.py`、`scoring_formula.py`、`predict.py`、`refresh_baselines.py`、`backfill_signals.py`。
文件:`reference/matchup-factors.md`、`reference/flags-checklist.md`、`reference/workflow-fundamentals.md`(舊基本面流程已刪)。`reference/workflow-odds.md` 若 smart-money 工具續留則保留、否則一併退役。

**瘦身(不全死):**
- `lineup_analyzer.py` → 砍成輕量「抓今日 9 人 + ops/woba」,只供 `fetch_inputs` 凍結(移除 Statcast 排行榜 merge / tier / platoon / BvP / last7 等重機器)。
- `fetch_game_data.py` → salvage「賽程 + 兩隊 RS/RA + probable starter」併進 `fetch_inputs`,其餘退役。

**保留 / 重用:** `park_factors_lib`、`_team_resolver`、`_utils`、`odds/fetch_odds`(+`odds/lib/odds_math`)、`lib/closing_line`(擴充 RL)、`fetch_results`、`backtest.py` + `lib/{load,metrics,render,diagnostic}`(改 schema)。

**獨立保留、不接入:** `odds/analyze_smart_money.py` + `odds/reports/`。

## 11. 回測整合 + ablation

- `lib/load.py`:改讀 `features.json`(v2 schema)+ `result.json` → 每場一列。因預測值已凍結,load 變單純(不再重算)。
- `lib/metrics.py`:聚焦 **RL 過盤命中率**、**O/U 命中率**、**edge 校準**(正 edge 那側是否真的較常贏)、vs market。**移除**方向信心檔位校準。
- **ablation 流程**:`run_model` 為純函數;features.json 凍結了打線 / RA / 進階輸入,日後可「用 alt 輸入重跑 run_model 對同一批比賽」比較命中率,裁決打線 / platoon / RA 是否升級進模型。

## 12. 檔案組織

**新增:** `predict_game.py`、`fetch_inputs.py`、`run_model.py`、`odds_compare.py`、`report.py`、`config.py` + 各自 `tests/test_*.py`。
**修改:** `lib/closing_line.py`(加 RL)、`lib/load.py`、`lib/metrics.py`(新 schema/指標)、`SKILL.md`。
**刪除/瘦身:** 見 §10。

## 13. 測試

| 測試 | 內容 |
|---|---|
| `test_run_model` 期望得分 | §5.3 worked example → 斷言 μ_home≈4.10、μ_away≈4.18 |
| `test_run_model` 機率 | 已知 μ + SIGMA → 斷言 P(RL)、P(Over)、P(ML) 到小數點 |
| `test_run_model` RL 邊界 | 主 −1.5 vs 主 +1.5 的 cover 公式方向正確 |
| `test_fetch_inputs` FIP | 給組件 → 斷言 FIP 公式;IP 過小 → fallback |
| `test_fetch_inputs` RS blend | recent/season → 斷言加權 |
| `test_odds_compare` | snapshot fixture → 斷言 RL/總分 no-vig + edge(含「找最新 snapshot」邏輯) |
| 重現性 | 同 inputs 跑兩次 → 數字完全一致 |
| 當日模式 | schedule fixture → 逐場、單場失敗不中斷 |

純函式 + fixture,不做 mock-heavy。

## 14. 邊界處理

| 情況 | 處理 |
|---|---|
| 先發未定 | starter 欄 null;該場 prediction.md 標「先發未公布,跳過」 |
| 先發 IP 過小(< MIN_IP,如 10) | FIP 不穩 → 用 league 替代並標註 |
| 盤口 snapshot 缺 / 該場未開盤 | odds/edges 欄 null;只輸出 model 機率,標「無盤口可比」 |
| RS/RA API 缺 | fallback 聯盟均分 4.4 並標註 |
| 整數總分線 push | v1 常態近似 P(push)≈0;Poisson 升級再處理 |
| doubleheader | 沿用 `-G1/-G2` 後綴資料夾 |

## 15. 開放參數（implementation plan / 回測後定稿）

§5.5 全部先驗:`LEAGUE_RG`、`RECENT_W`、`SP_W`/`BP_W`、`SIGMA_TEAM`、`FIP_CONSTANT`、`RECENT_N`、`MIN_IP`。
**SIGMA_TEAM 是 edge 的關鍵**:σ 太小 → 機率過度自信 → edge 虛高。上線前必須用回測重新擬合 σ,才可信任 edge 數字(gate)。

## 16. 風險與已知限制

1. **σ 與權重是先驗**:RL/OU 的 edge 對 σ 敏感,須回測重擬合前不可當下注依據(§15 gate)。
2. **常態近似**:低得分離散分布用常態是近似;極端線與 push 不精確。Poisson/Skellam 為已知升級路徑。
3. **team-level 對今日陣容無感**:當家球星輪休/傷,模型不知道(設計取捨;打線已凍結供日後驗)。
4. **未證明打贏市場**:v1 目標是乾淨、簡單、可量測,如同前一版 spec;是否有 alpha 等 going-forward 回測。
5. **FIP 依賴 API 組件**:季初樣本小 → FIP 抖動,靠 MIN_IP guard。

## 17. 成功標準

1. 同一場同 inputs 跑兩次 → 數字完全一致(零 drift)。
2. 單場 + 當日模式都產出 `features.json` + `prediction.md`。
3. 回測能在凍結的 features.json 上一鍵重跑,輸出 RL/OU 命中 + edge 校準。
4. 程式碼足跡較現狀減少約 6,500 行。
5. features.json 凍結了打線 + RA + 先發組件,使打線 / platoon / RA 三項 ablation 日後可行。
