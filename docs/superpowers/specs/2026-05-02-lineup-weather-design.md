# Official Lineup + Weather 整合 Design

**Date**: 2026-05-02
**Scope**: `mlb-game-analyzer` skill — 在資料收集階段（步驟 1）優先取用球隊公布的當日打序，並在天氣資訊已公布時納入 dossier / summary 的條件修正段；皆採「公布則用、未公布則跳過或 fallback」語義。

---

## 1. Motivation

現行 `lineup_analyzer.py` 取「active roster 排除投手 / IL，按本季 PA 降序前 9 人」當打線近似（`projected`）。MLB Stats API 公布實際打序後（賽前 ~2-4 小時）並未被採用，造成 `chain.obp_top3` / `slg_mid` 等串聯指標、BvP 對位以及 Tier 評級都基於季度近似而非當日實況。

天氣資料同樣存在於 `feed/live` 端點的 `gameData.weather`（賽前 ~1 小時始填齊），但 pipeline 完全未取用，summary `## 條件修正` 段的天氣判讀目前純為 AI placeholder，缺資料支撐。

---

## 2. Goals

1. 在 `prepare_game.py` pipeline 自動優先取用 official lineup；無資料時 silent fallback 至現行 PA proxy。
2. 在同一 pipeline 中自動取用 weather；無資料時跳過天氣分析（不阻斷流程、不報錯）。
3. 兩條 official 路徑的成功 / 失敗在 dossier 與 summary 中清楚標記（`lineup_source` 標籤、weather 三狀態列）。
4. 既不改變 scoring formula，也不改變 Flag 體系，避免 noisy 訊號自動修正得分。
5. 既有 `merged.json` / `home_lineup.json` / `away_lineup.json` 缺新欄位時 renderer 仍能 graceful 渲染（向下相容）。

---

## 3. Non-Goals

- 不引入第三方天氣 API（OpenWeatherMap / NWS）。資料源限縮為 MLB Stats API `feed/live`。
- 不對天氣 / 主力缺陣自動 ±run value（與 BABIP / ERA-xERA gap 處理一致）。
- 不偵測「主力打者休息」並產生新 Flag（保留為未來工作；方法論記於 §11）。
- 不調整公式（`scoring_formula.py` 零異動）。
- 不變更 schema migration 策略（既有 `analysis-data/` 內舊 merged.json 不重產）。

---

## 4. 整體架構與資料流

```
prepare_game.py
  step_a → fetch_game_data.py → game_data.json
           （ids 回傳值多一個 game_pk —— 從現有 game_data.json 既有的 gamePk 讀出）

  step_d → lineup_analyzer.py × 2
           新增 --game-pk 參數
           內部新增 fetch_official_lineup(game_pk, team_id)
           分支：official_ids 完整 9 人 → 跑那 9 人；否則走原本 PA fallback
           回傳新增 lineup_source / lineup_source_detail 欄位、batter 多 batting_order

  step_e → merge_game_data.py
           新增 fetch_weather(game_pk)（與 fetch_bullpen_era 同層級）
           merged.json 新增 weather block

  step_f → dossier_renderer.py
           打線 section：official 時 9 棒 vs 對方先發 table；projected 時維持現行 Top 5
           新增 weather row（在 venue / PF 同段，缺資料整行省略）

  step_g → summary_renderer.py
           ## 條件修正 段：weather 三狀態 pre-fill
           ## 打線評級 段：lineup_source 標記
```

### 4.1 檔案異動清單

| 檔 | 異動 | 規模 |
|---|---|---|
| `scripts/lineup_analyzer.py` | 新 `fetch_official_lineup`、`analyze_team` 加 game_pk 分支、output schema +2 欄 | M |
| `scripts/prepare_game.py` | step_a ids 多 game_pk、step_d cmd 多 `--game-pk` | S |
| `scripts/merge_game_data.py` | 新 `fetch_weather`、merged 多 weather block | M |
| `scripts/dossier_renderer.py` | 打線 section 分支、weather row | M |
| `scripts/summary_renderer.py` | `## 條件修正` 加 weather row、`## 打線評級` 加 source 標記 | S |
| `reference/matchup-factors.md` | 新 `### 天氣修正` 子段、`## 打線分析` 段首加 source 說明 | S |
| `SKILL.md` | Quick Reference 提及 official lineup + weather、加「條件式資料」段 | XS |
| `scripts/tests/*` | 新 fixture + 測試（~25 case） | M |

---

## 5. Lineup 整合細節

### 5.1 新函式 `fetch_official_lineup`（在 `lineup_analyzer.py`）

```python
def fetch_official_lineup(game_pk: int, team_id: int) -> list[int] | None:
    """從 feed/live 取該隊公布打序的 player_id list（按 1-9 棒順序）。

    回傳：
      - list[int] 長度 9：官方公布完整打序
      - list[int] 長度 0~8：部分公布（caller 自行決定 fallback）
      - None：API 失敗 / team_id 不在 boxscore

    side 自動判斷：比對 boxscore 的 home.team.id / away.team.id。
    """
    try:
        resp = requests.get(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        boxscore = data.get("liveData", {}).get("boxscore", {})
        for side in ("home", "away"):
            t = boxscore.get("teams", {}).get(side, {})
            if t.get("team", {}).get("id") == team_id:
                return list(t.get("battingOrder", []))
        print(f"[lineup_analyzer] team_id {team_id} not in boxscore (game_pk={game_pk})", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[lineup_analyzer] feed/live fetch failed: {e}", file=sys.stderr)
        return None
```

> 注意：partial / empty 兩種狀態（None / []  vs 1–8 人）的 stderr 訊息由 caller (`analyze_team`) 在決定 fallback 時印出，因為函式本身不該知道「9 才算完整」這個語義。
```

### 5.2 `analyze_team` 改寫骨架

```python
def analyze_team(team, year, opposing_pitcher_id=None, game_pk=None):
    team_id = resolve_team_id(team)

    official_ids = fetch_official_lineup(game_pk, team_id) if game_pk else None

    if official_ids and len(official_ids) == 9:
        lineup_source = "official"
        # 用 official 順序拿基本資料
        # position 復用既有 fetch_team_roster(team_id, year)：official lineup 球員必在 active roster，
        # 用 player_id → position dict 對應；如有極少數查不到位置（晚進名單尚未同步），fallback 到 ""
        roster = fetch_team_roster(team_id, year)
        position_map = {p["id"]: p["position"] for p in roster}
        core_lineup = build_lineup_from_official(official_ids, year, position_map)
        # 每個 batter 帶 batting_order = 1..9
    else:
        lineup_source = "projected"
        # 原本流程：active roster → IL filter → PA 排序 top 9
        core_lineup = build_lineup_from_pa_proxy(team_id, year, ...)

    # 下游一致：Statcast / Platoon / Last7 / BvP（per-batter loop 不變）
    enrich_lineup(core_lineup, year, opposing_pitcher_id)

    return {
        ...,  # 既有 tier / chain / aggregates
        "lineup_source": lineup_source,
        "lineup_source_detail": {
            "fetched_at": ISO_UTC,
            "game_pk": game_pk,
        } if lineup_source == "official" else None,
        "lineup": core_lineup,
    }
```

**partial lineup（< 9 人）**：完整才算 official，否則整套 fallback；不混合。

### 5.3 lineup 內每個 batter 新增欄位

```python
{
    ...,  # 既有：mlbam_id, name, position, pa, avg, obp, slg, ops, xwoba, ...
    "batting_order": 1,  # 1–9；projected 路徑此欄為 None
}
```

### 5.4 fallback / failure 矩陣

| game_pk | feed/live API | battingOrder | 結果 | stderr |
|---|---|---|---|---|
| None | — | — | projected | （無） |
| 有 | 成功 | 長度 9 | **official** | `[lineup_analyzer] official lineup fetched (9)` |
| 有 | 成功 | 長度 0 | projected | `[lineup_analyzer] official lineup not yet posted, fallback to PA proxy` |
| 有 | 成功 | 長度 1–8 | projected | `[lineup_analyzer] official lineup partial (N=X), fallback to PA proxy` |
| 有 | 成功 | team_id 不在 boxscore | projected | `[lineup_analyzer] team_id not in boxscore (game_pk=...), fallback` |
| 有 | 失敗 | — | projected | `[lineup_analyzer] feed/live fetch failed: <err>, fallback` |

**永遠不 abort pipeline。**

### 5.5 `prepare_game.py` 異動

```python
# step_a 回傳值多 "game_pk"（讀 game_data.json 既有的 gamePk）
return {
    ...,
    "game_pk": game_section.get("gamePk"),
}

# step_d signature 多 game_pk 參數，cmd 加 --game-pk
cmd = [..., "lineup_analyzer.py", "--team", team, "--year", season, "-o", out_path]
if opposing_id:
    cmd += ["--opposing-pitcher-id", str(opposing_id)]
if game_pk:
    cmd += ["--game-pk", str(game_pk)]
```

### 5.6 dossier 打線 section 分支

```python
def _render_lineup_overview(bundle):
    home_lu = bundle.get("home_lineup") or {}
    away_lu = bundle.get("away_lineup") or {}
    home_source = home_lu.get("lineup_source", "projected")
    away_source = away_lu.get("lineup_source", "projected")

    lines.append("## 打線")
    lines.append(f"- HOME 打線來源：{_source_label(home_source, home_lu)}")
    lines.append(f"- AWAY 打線來源：{_source_label(away_source, away_lu)}")

    # 主 9 人 table 邏輯不變（PA 或 batting_order 排序由 source 決定）

    # vs 對方先發 sub-block 分支：
    if home_source == "official":
        _render_full9_vs_pitcher(home_lu, ...)   # 新 helper：9 棒按 batting_order 全部顯示
    else:
        _render_top5_vs_pitcher(home_lu, ...)    # 既有：PA top 5 + 額外熱手（重構自現行 inline 邏輯）
    # away 同
```

`_source_label`：
- `official` → `🟢 official`
- `projected` → `🟡 projected（PA 排序近似 — 打線尚未公布）`

> 不用 `lineup_source_detail.fetched_at` 當「公布時間」顯示給使用者——那是我們撈取時間，不是 MLB 公布時間，會誤導。
> JSON 內保留 fetched_at 作技術記錄即可（debug / 重跑判斷用）。

### 5.7 summary `## 打線評級` 段加 source

```markdown
### HOME — 🔴 Elite / 🔥 Hot
- 打線來源：🟢 official
- **Tier 覆寫**：<!-- AI 補 -->
```

---

## 6. Weather 整合細節

### 6.1 新函式 `fetch_weather`（在 `merge_game_data.py`）

```python
def fetch_weather(game_pk: int) -> dict | None:
    """從 feed/live 取 gameData.weather。

    回傳：
      - dict：{condition, temp_f, wind_text, indoor}
      - None：API 失敗 / weather 欄位不存在或全空
    """
    try:
        resp = requests.get(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
            timeout=10,
        )
        resp.raise_for_status()
        w = resp.json().get("gameData", {}).get("weather", {}) or {}
        condition = (w.get("condition") or "").strip()
        temp = (w.get("temp") or "").strip()
        wind = (w.get("wind") or "").strip()

        if not condition and not temp and not wind:
            return None

        indoor = condition.lower() in ("roof closed", "dome")

        try:
            temp_f = int(temp) if temp else None
        except ValueError:
            temp_f = None

        return {
            "condition": condition or None,
            "temp_f": temp_f,
            "wind_text": wind or None,
            "indoor": indoor,
        }
    except Exception as e:
        print(f"[merge_game_data] weather fetch failed: {e}", file=sys.stderr)
        return None
```

**設計取捨**：
- 不解析 wind_text 成 speed / direction 欄位 — AI 直接讀原文判讀
- 不引入第三方 weather API
- 不對 None 補預設值 — None 語義就是「沒資料、跳過分析」

### 6.2 `merge_game_data.py` 主流程接入

```python
# 既有：
merged["home_bullpen_era"] = home_bp_era
merged["away_bullpen_era"] = away_bp_era
merged["park_factor"] = park_factor

# 新增：
game_pk = game_info.get("gamePk")
weather = fetch_weather(game_pk) if game_pk else None
merged["weather"] = weather  # dict | None
```

### 6.3 merged.json schema 新增段

```json
{
  ...,
  "weather": {
    "condition": "Sunny",
    "temp_f": 78,
    "wind_text": "10 mph, Out To CF",
    "indoor": false
  },
  "home_lineup": {
    ...,
    "lineup_source": "official",
    "lineup_source_detail": { "fetched_at": "...", "game_pk": 778345 }
  }
}
```

### 6.4 dossier weather row

放在現有 venue / park_factor 同段：

```markdown
**venue**: Yankee Stadium | **park_factor (runs)**: 105
**weather**: Sunny, 78°F, wind 10 mph Out To CF
```

三狀態：
- 有資料：`Sunny, 78°F, wind 10 mph Out To CF`
- 室內：`室內（Roof Closed，不適用天氣分析）`
- 缺資料：**整行省略**（與 dossier 簡潔風格一致）

### 6.5 summary `## 條件修正` 段 weather row

現行 template：
```markdown
## 條件修正
- Park Factor: 105 → +0.25 run
- 先發 tier / doubleheader / 天氣：<!-- AI 補 -->
```

改為：
```markdown
## 條件修正
- Park Factor: 105 → +0.25 run
- 天氣：{狀態列}
- 先發 tier / doubleheader：<!-- AI 補 -->
```

`{狀態列}` 三變體（實際渲染樣式）：

**有資料**：
```markdown
- 天氣：Sunny, 78°F, wind 10 mph Out To CF
  - 影響判讀：<!-- AI 補：對得分 / HR 影響判讀 -->
```

**室內**：
```markdown
- 天氣：室內（Roof Closed，不適用）
```

**缺資料**：
```markdown
- 天氣：未公布（跳過天氣分析）
```

只有「有資料」這條才需要 AI 判讀；室內 / 缺資料 placeholder 直接省略，避免 AI 寫廢話。

### 6.6 weather 不進公式

`scoring_formula.py` **零異動**。原則維持：
- park_factor 進公式（×PF/100）
- weather 不進公式 — 與 BABIP / ERA-xERA gap 同處理路線

### 6.7 失敗處理

| game_pk | feed/live API | weather 欄位 | merged.weather | summary 顯示 |
|---|---|---|---|---|
| None | — | — | None | 缺資料 |
| 有 | 失敗 | — | None | 缺資料 |
| 有 | 成功 | 全空 | None | 缺資料 |
| 有 | 成功 | condition=Roof Closed | `{indoor: true, ...}` | 室內 |
| 有 | 成功 | 有風溫條件 | `{indoor: false, ...}` | 有資料 |

**永遠不 abort。**

---

## 7. Reference Doc 異動

### 7.1 `reference/matchup-factors.md` 新增 `### 天氣修正` 子段

附在現有 `## 球場 & 天氣` 段尾（`### Park Factor` 之後）：

```markdown
### 天氣修正

資料源：MLB Stats API `feed/live` 的 `gameData.weather`，由 `merge_game_data.py` 自動撈取。
**未公布或室內球場 → 不分析**（merged.weather = None 或 indoor=true）。

> ⛔ 天氣**不進 scoring formula**（與 BABIP / ERA-xERA gap 同等級——研究存在但 noisy）。
> AI 在 summary `## 條件修正` 段以敘事方式判讀，**不自動 ±run value**。

#### 風（wind）

MLB API wind 欄位已含風向解讀（球場 orientation 已換算），形式：

| 文字 | 意義 |
|------|------|
| `Out To CF / LF / RF` | 順風出去（利 HR / 飛球） |
| `In From CF / LF / RF` | 逆風進來（壓 HR / 利投手） |
| `L To R` / `R To L` | 橫風（影響有限） |
| `Calm` / `Varies` | 無顯著影響 |

風速門檻（敘事用）：

| 速度 | 影響 |
|------|------|
| < 8 mph | 噪音，可忽略 |
| 8–15 mph | 輕度，順風略利攻 / 逆風略利投 |
| 15–20 mph | 中度，HR 機率明顯偏移 |
| > 20 mph | 強，**summary 風險段必提** |

#### 溫度

聯盟基準 ~70°F；偏離越多影響越大（球的飛行距離與空氣密度 / 球皮含水量相關）。

| 溫度 | 影響 |
|------|------|
| > 85°F | ⬆️ 球易飛，輕度利攻 |
| 60–85°F | 中性 |
| 50–60°F | 輕度利投 |
| < 50°F | ⬆️ 利投，球員肌肉表現也受影響 |

> Coors / Yankee Stadium / Wrigley 對風更敏感（球場 orientation + 大氣條件交互）。
> 球員適應性差異大（北方球隊冷天表現相對好）— **AI 判讀時優先看相對強度**，不直接套表。
```

### 7.2 `reference/matchup-factors.md` `## 打線分析` 段首加 source 說明

```markdown
## 打線分析

**打線來源**（由 `lineup_analyzer.py` 自動偵測）：
- 🟢 **official**：球隊已公布今日打序（賽前 ~2-4 小時 API 才填），9 人 1-9 棒順序為實際打序
- 🟡 **projected**：打序未公布，採 active roster（排除 IL）按 PA 降序取前 9 人作近似

**評級邏輯不分 source**：tier / chain / over_under_lean / 觸發條件對兩種來源一致。
**差異**：official 路徑下 `chain.obp_top3` / `slg_mid` 是真實 1-3 棒 / 4-5 棒；projected 是 PA 排序近似。

對打線核心（1-9 棒）查詢：xwOBA、OPS、OBP、SLG、ISO、K%/BB%、Hard Hit%、Barrel%、BABIP、xBA、xSLG。
（以下保持原樣...）
```

### 7.3 `SKILL.md` 異動

**Quick Reference 表第 1 步**：

```markdown
| 1. 資料收集 | `merged.json` + `dossier.md` + `summary.md`（含 AI 填空 placeholder）<br>**自動偵測**：official lineup（公布後）/ 天氣（公布後） | `prepare_game.py` |
```

**「資料來源優先順序」段後新增「條件式資料」段**：

```markdown
### 條件式資料（公布後才有）

| 資料 | 來源 | 缺資料行為 |
|------|------|-----------|
| 公布打線（battingOrder） | feed/live | fallback 至 PA proxy（lineup_source = "projected"） |
| 天氣（condition / temp / wind） | feed/live `gameData.weather` | summary 標「未公布（跳過天氣分析）」 |

**公布時機**：打線通常開賽前 2–4 小時、天氣前 1 小時 ~ 開賽後填齊。
**重跑取最新**：`prepare_game.py --force` 才會覆蓋已編輯的 summary.md（dossier 永遠重產）。
```

**步驟 1 後續動作 ℹ️ 區補一條**：

```markdown
ℹ️ **打線來源 / 天氣**：dossier 與 summary 都會標記。official 與 projected 分析架構相同，差異僅在 9 人組成是真實打序還是 PA 近似（見 `matchup-factors.md` §打線分析）。
```

**步驟 2.3 條件修正描述補「天氣」**：

```markdown
| 2.3 條件修正 | 傷病/TJ/角色轉換/年齡/球場/**天氣** | `matchup-factors.md` §天氣修正 |
```

### 7.4 不動的 reference

- `flags-checklist.md` — Flag 體系不動，weather / lineup_source 不是 Flag
- 公式相關 — `scoring_formula.py` 不動

---

## 8. Testing

### 8.1 新增 fixtures

**`scripts/tests/fixtures/feed_live_official_lineup.json`**（精簡版）：

```json
{
  "gameData": {
    "weather": {"condition": "Sunny", "temp": "78", "wind": "10 mph, Out To CF"}
  },
  "liveData": {
    "boxscore": {
      "teams": {
        "home": {
          "team": {"id": 147},
          "battingOrder": [592450, 519317, 624413, 519203, 670541, 543305, 596019, 624577, 656555],
          "players": {"ID592450": {"position": {"abbreviation": "DH"}}}
        },
        "away": {
          "team": {"id": 110},
          "battingOrder": [],
          "players": {}
        }
      }
    }
  }
}
```

額外變體：
- `feed_live_partial_lineup.json` — battingOrder 長度 5（觸發 fallback）
- `feed_live_empty_lineup.json` — battingOrder = [] 且 weather = {}
- `feed_live_indoor.json` — condition = "Roof Closed"
- `feed_live_weather_only.json` — battingOrder = [] 但 weather 有值（lineup fallback、weather 顯示）

### 8.2 `tests/test_lineup_analyzer.py`

| 測試 | 場景 | 驗證 |
|------|------|------|
| `test_fetch_official_lineup_full` | mock feed/live 回 9 人 | 回傳 list[int]，長度 9，順序保留 |
| `test_fetch_official_lineup_partial` | 回 5 人 | 回傳 list[int] 長度 5（caller 自行決定 fallback） |
| `test_fetch_official_lineup_empty` | 回 [] | 回傳 [] |
| `test_fetch_official_lineup_team_not_found` | team_id 不在 boxscore | 回傳 None |
| `test_fetch_official_lineup_api_fail` | requests 拋 exception | 回傳 None，stderr 有 warning |
| `test_analyze_team_official_path` | game_pk + 完整 9 人 | `lineup_source == "official"`，9 人含 batting_order=1..9，跳過 PA 排序 |
| `test_analyze_team_partial_falls_back` | game_pk + 5 人 | `lineup_source == "projected"`，走 PA proxy |
| `test_analyze_team_no_game_pk` | game_pk = None | `lineup_source == "projected"`，等同現行行為 |
| `test_analyze_team_api_fail_falls_back` | feed/live 失敗 | `lineup_source == "projected"`，stderr warning |

### 8.3 `tests/test_merge_game_data.py`

| 測試 | 場景 | 驗證 |
|------|------|------|
| `test_fetch_weather_full` | weather 三欄齊 | 回傳 dict，indoor=False |
| `test_fetch_weather_indoor` | condition="Roof Closed" | indoor=True |
| `test_fetch_weather_empty` | weather={} | 回傳 None |
| `test_fetch_weather_partial` | 只有 condition 沒 wind | 回傳 dict，缺欄為 None |
| `test_fetch_weather_api_fail` | requests 拋 exception | 回傳 None，stderr warning |
| `test_merged_weather_present` | end-to-end mock | merged.json 有 `weather` 區塊 |
| `test_merged_weather_absent` | end-to-end mock，weather=None | merged.json `weather` = None |

### 8.4 `tests/test_dossier_renderer.py`

| 測試 | 場景 | 驗證 |
|------|------|------|
| `test_lineup_section_official` | bundle 兩隊 lineup_source=official | 標題出現「打線來源：🟢 official」、9 人 vs 對方先發 table、無 Top 5 sub-block |
| `test_lineup_section_projected` | bundle 兩隊 lineup_source=projected | 標題「🟡 projected」、現行 Top 5 + 額外熱手 sub-block 維持 |
| `test_lineup_section_mixed` | home=official, away=projected | 兩邊各自渲染對應分支 |
| `test_lineup_section_no_source_field` | merged.json 缺 lineup_source（舊資料） | 預設 projected，向下相容 |
| `test_weather_row_present` | merged.weather 三欄齊 | dossier 出現「weather: Sunny, 78°F, ...」 |
| `test_weather_row_indoor` | merged.weather.indoor=True | 顯示「室內（Roof Closed，不適用天氣分析）」 |
| `test_weather_row_absent` | merged.weather=None | 整行省略 |

### 8.5 `tests/test_summary_renderer.py`

| 測試 | 場景 | 驗證 |
|------|------|------|
| `test_lineup_section_marks_source` | home=official | summary `## 打線評級` HOME 段含「打線來源：🟢 official」 |
| `test_conditional_weather_present` | merged.weather 有資料 | `## 條件修正` 出現「天氣：Sunny, 78°F, ...」+ AI placeholder |
| `test_conditional_weather_indoor` | indoor=True | 「天氣：室內（Roof Closed，不適用）」+ **無** AI placeholder |
| `test_conditional_weather_absent` | weather=None | 「天氣：未公布（跳過天氣分析）」+ **無** AI placeholder |

### 8.6 `tests/test_prepare_game.py`（既有，異動）

如有現行整合測試 mock subprocess + game_data fixture：
- 確認 step_a return 值新增 `game_pk`
- 確認 step_d cmd 含 `--game-pk`（當 game_pk 存在時）

### 8.7 不動的測試

- `scoring_formula` 公式不變 → 測試不動
- `pitcher_stats` / `roster_checker` / `fetch_game_data` 不動 → 測試不動
- Flag 偵測（Flag 3 / Flag 8）邏輯不變 → 測試不動

### 8.8 測試風格

- 全部 mock `requests.get`，**不打真 API**
- fixture 是真 API 回應的精簡版（保留必要欄位、刪掉 plays 等大欄位）
- 用 `pytest`（與既有測試一致）

---

## 9. 整體 sanity check

| 範疇 | 規模 | 風險 |
|------|------|------|
| 程式碼 | 5 檔異動（2 中度、3 小度）+ 1 新函式各 ~30 行 | 低，邏輯本地化 |
| Schema 變化 | merged.json 加 2 欄（weather block + lineup_source pass-through）；lineup.json 加 2 欄 | 低，renderer 用 `.get()` 向下相容 |
| Reference doc | matchup-factors.md 加一節 + 一段微調；SKILL.md 兩處小改 | 無 |
| 測試 | 新增 ~25 個單元測試 + ~5 個 fixture | 無 |
| API call 數 | 每場 +3 次 feed/live（lineup × 2 + weather × 1） | MLB API 無認證且 cache 友善，可接受 |

**沒有公式變更、沒有 schema 破壞性改動、所有失敗路徑都 fallback 不 abort。**

---

## 10. Out of Scope（本次不做、未來可加）

1. **主力打者休息偵測**：方法已存在（official_ids vs PA-top9 差集），但 user 決定本次不做。未來可在 `lineup_analyzer.analyze_team` 末端加一行 `absent_pa_top9` 並由 dossier / summary 顯示。
2. **`--require-lineup` flag**：強制要求 official lineup，否則 exit 非 0。本次採「自動 fallback」即可。
3. **第三方 pre-game weather API**：賽前 4+ 小時的天氣預報。本次限縮 MLB API only。
4. **主審 (home plate umpire) 分析**：feed/live 有 officials 資料，但本次不納入；未來如要加，與 weather 同管道。
5. **天氣自動 ±run value**：研究存在但 noisy，與 skill 「不自動修正 noisy 訊號」哲學一致，本次保留 AI 判讀。

---

## 11. Open Questions（none）

所有先前討論的開放問題皆已決議：

- A1（端點）→ feed/live（一次拿全）
- A2（partial）→ 完整 9 人才算 official
- A3（CLI）→ 自動 fallback
- A4（主力休息）→ 不做（方法記錄）
- A5（顯示打線）→ 是
- A6（game_pk 傳遞）→ ids 多 game_pk、step_d 加 --game-pk
- A7（API 共用）→ 各支腳本各自打
- B1（資料源）→ MLB API only
- B2（風向解讀）→ AI 判斷
- B3（室內）→ 識別 condition 標 indoor=true 跳過
- B4（公式）→ 不進
- B5（顯示位置）→ summary `## 條件修正`
- B6（dossier）→ 也顯示
- B7（落地腳本）→ merge_game_data.py
- B8（reference）→ 加 §天氣修正
- C1–C5（fallback / schema / SKILL.md / 測試 / 重產規則）→ 一致確認
- D1（official 9 人 BvP）→ 9 人全跑（dossier 改顯示）
- D2（feed/live 失敗）→ silent fallback projected

---

## 12. Implementation 順序建議（給 writing-plans 階段參考）

1. `lineup_analyzer.py` — `fetch_official_lineup` + `analyze_team` 分支 + 測試
2. `prepare_game.py` — step_a / step_d 傳 game_pk + 整合測試
3. `merge_game_data.py` — `fetch_weather` + merged.weather + 測試
4. `dossier_renderer.py` — 打線 source 標記 + 9 人 vs 對方先發 + weather row + 測試
5. `summary_renderer.py` — `## 打線評級` source 標 + `## 條件修正` weather 三狀態 + 測試
6. `reference/matchup-factors.md` — `### 天氣修正` + `## 打線分析` 段首
7. `SKILL.md` — Quick Reference + 條件式資料段 + 步驟 1 ℹ️
8. End-to-end smoke test：跑一場真實比賽（賽前打線已公布的場次）驗證 pipeline

各步驟獨立可測，commit 粒度 = 一檔一 commit（reference/SKILL 可合併一 commit）。
