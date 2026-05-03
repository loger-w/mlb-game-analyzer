# TTO3 Penalty Signal Design

**Date**: 2026-05-03（**2026-05-03 Plan B 修訂**——spike 結果證實 Plan A 不可行）
**Scope**: `mlb-game-analyzer` skill — 在現有 8 個 derived signals 之外新增第 9 個 `tto3_penalty`（先發投手第三輪面對打者 OPS 衰退幅度），覆蓋 CHANGELOG.md line 48 「第二批 signals」中第一個遺留項目。沿用 PR-3 既建立的 signal contract / dossier `## 🎯 訊號摘要` / summary `### 額外信號` 三層 surface 不變動既有架構。

---

## 0. Plan B Amendment（post-spike 2026-05-03）

原 §5.1 假設可走 MLB Stats API `statSplits + sitCodes`（Plan A）。Spike 階段對 `ot1,ot2,ot3` / `1,2,3` / `1f,2f,3f` 三組候選 sitCode 全部測過，結論：

- 三組都 200 OK 但**回 0 splits**（`1,2,3` 回的「March」是月份分裂、非 TTO）
- MLB API `/situationCodes` 元資料端點 602 個 codes **沒有任何 TTO 切面**（最近的是 First Inning Pitched / First 75 Pitches / First Batter RP Only，語意不對）
- `careerStatSplits` endpoint 同樣 0 splits

**結論**：MLB Stats API 不曝光 Times-Through-Order 數據，Plan A 路徑死。

**改走 Plan B**：用 `pybaseball.statcast_pitcher(start_dt, end_dt, mlbam_id)` 拉整季逐球 DataFrame，自己用 `(game_pk, batter)` 分組 + `at_bat_number` 排序計算每位打者在該場的 PA ordinal，再 PA 級加總成 TTO1/2/3 OPS / K% / BB%。

Spike 第二輪驗證 Plan B 可行（Skubal 2025 4 月前兩週 → 271 pitches、72 PAs、所有必要欄位 `at_bat_number / batter / events / game_pk` 都在）。

**設計差異總覽**：

| 項目 | 原 Plan A | Plan B |
|---|---|---|
| Fetch 方式 | MLB API `statSplits` + sitCodes | `pybaseball.statcast_pitcher` 整季逐球 |
| Career fallback | 第二支 `careerStatSplits` API | `statcast_pitcher` 多年區間 `(year-4, year)` |
| 回傳速度 | < 1 秒 | ~5–15 秒/投手 |
| 程式碼量 | ~30 行（fetch + parse） | ~80–100 行（fetch + PA 分組 + outcome 加總） |
| OPS / OBP / SLG | API 直給字串 | 從 events 自己算（OBP/SLG 公式） |
| 既有 codebase 風險 | 純 requests | 沿用 `_import_pybaseball()` 既有 lazy import 模式（同 `fetch_whiff_csw`） |

§5.1 / §5.3 已**就地改寫**為 Plan B 版本；§5.4 fallback 矩陣概念不變、實作機制改動；§5.5 是新增的 PA outcome 映射表。

---

## 1. Motivation

Times Through Order 3 penalty 是先發投手對抗第三次面對同一打線時的衰退現象——統計上常見「TTO3 OPS 比 TTO1 高 +0.070」（聯盟基準），由 stuff 多樣性、arsenal 深度、體力分配三個結構性因素合成。某些投手（low-K、single-pitch dependent、低 stuff diversity）的 TTO3 衰退顯著大於聯盟基準，賽前可預判：

1. **教練會提早換投** → 後段牛棚負擔增加
2. **若對方牛棚同時薄**（既有 `core_il_count` signal）→ 後段失分風險被放大、總分判讀偏多
3. **TTO3 強的投手**（不 fire 反向意義）→ 牛棚消耗少 → 對方總分壓力減

現行 signals_lib 涵蓋 stuff 結構（pitch_mix_concentration）但**沒有 stamina-curve / 第三輪衰退**這個獨立維度。新增 `tto3_penalty` 讓 dossier 第一次有「投手撐到第三輪能力」的可量化信號。

---

## 2. Goals

1. 新增 pure-function `signal_tto3_penalty(tto_splits)` 落在 `signals_lib.py`，nature 與既有 8 signals 完全一致。
2. 新增 `pitcher_stats.fetch_tto_splits(mlbam_id, year)`，沿用 `fetch_platoon_splits` 的 MLB Stats API `statSplits` 路徑，零新依賴。
3. 4 月小樣本（season TTO3 BF < 30）→ silent fallback 至 `careerStatSplits`，confidence 降為 `heuristic` + `source: "career"`，4 月也能上線。
4. Dossier `## 投手對決` table 加一 visible row「TTO splits」（**無條件**顯示，mirror `vs LHB / vs RHB` 行為），signal 只負責 flag interpretation。
5. `## 🎯 訊號摘要` + `### 額外信號` 兩處 fired 時 surface（同既有 8 signals）。
6. `reference/matchup-factors.md §Signals` 加 §9 條目，AI 判讀指引明確化（不自動 ±run value 紀律一致）。
7. 順手清理 `CHANGELOG.md` line 50 過時條目（wRC+ / Stuff+ 已上線）。

---

## 3. Non-Goals

- **不**進 `scoring_formula.py`（一致 §3 / §8 紀律——研究存在但 noisy / reflexive）。
- **不**新增 prepare_game step（沿用 step C `pitcher_stats.py` fetch tier）。
- **不**動 `merge_game_data.py`（signal 直接從 `bundle["home_pitcher" / "away_pitcher"]` 讀，不需 mirror 層）。
- **不**對非先發投手特別 guard（RP / opener TTO3 BF 永遠 < 30 → 自動走 small_sample no_fire 路徑）。
- **不**處理「投手實際被換掉的時點預測」（純資料信號，AI 在 summary 判讀）。
- **不**改 `_HALF_LIFE_BY_NAME` 既有 8 條 entry（只追加第 9 條 `"tto3_penalty": "structural"`）。

---

## 4. 整體架構與資料流

```
prepare_game.py
  step_c → pitcher_stats.py × 2
           現有 fetch_platoon_splits 同 tier 加一行 fetch_tto_splits
           home_pitcher.json / away_pitcher.json 頂層多 "tto_splits" key

  step_e → merge_game_data.py
           不動（signal 直讀 pitcher.json，不過 merge 層）

  step_f → dossier_renderer.py
           ## 投手對決 table 加 visible row「TTO splits」
           ## 🎯 訊號摘要 自動 pick up（既有 signals_for_bundle cache）

  step_g → summary_renderer.py
           ### 額外信號 自動 pick up（既有 signals_for_bundle cache）
```

### 4.1 檔案異動清單

| 檔 | 異動 | 規模 |
|---|---|---|
| `scripts/pitcher_stats.py` | 新 `fetch_tto_splits()` + `_compute_tto_from_statcast()` + `_pa_outcome_aggregates()` + `_has_sufficient_tto3()`；main 路徑加 call + JSON 寫入 | M |
| `scripts/signals_lib.py` | 新 `signal_tto3_penalty()`；`_HALF_LIFE_BY_NAME` 加 1 行；`compute_all_signals` per-pitcher 段加 1 行 | S |
| `scripts/dossier_renderer.py` | `## 投手對決` table 加 row「TTO splits」 helper + caller | S |
| `reference/matchup-factors.md` | §Signals 加 §9 條目；§Signals 半衰期表 structural 列加 `tto3_penalty` | S |
| `CHANGELOG.md` | 移除 line 50 過時條目；加新條目記 `tto3_penalty` 上線 | XS |
| `scripts/tests/test_pitcher_stats.py` | +6–8（_pa_outcome_aggregates / _compute_tto_from_statcast / fetch_tto_splits） | S |
| `scripts/tests/test_signals_lib.py` | +9（signal 觸發 / fallback / boundary） | S |
| `scripts/tests/test_dossier_renderer.py` | +5（TTO row 渲染 + 1 integration） | XS |

---

## 5. Data fetch 細節（Plan B）

### 5.1 新函式 `fetch_tto_splits`（在 `pitcher_stats.py`，緊接 `fetch_platoon_splits` 之後）

```python
def fetch_tto_splits(mlbam_id: int, year: int) -> dict:
    """C2.5：取得投手 Times-Through-Order Splits（TTO1 / TTO2 / TTO3）。

    Season 優先；TTO3 BF < 30 時 silent fallback 至 5-year career window。
    回傳：
      {
        "source": "season" | "career",
        "tto1": {"ops": float, "k_pct": float, "bb_pct": float, "bf": int},
        "tto2": {...},
        "tto3": {...},
      }
      或 {"error": "..."} 兩條路徑都失敗時。
    """
    season_data = _compute_tto_from_statcast(mlbam_id, year, year)
    if _has_sufficient_tto3(season_data):
        season_data["source"] = "season"
        return season_data

    # Career fallback：5-year window（year-4 ~ year）
    career_data = _compute_tto_from_statcast(mlbam_id, year - 4, year)
    if _has_sufficient_tto3(career_data):
        career_data["source"] = "career"
        return career_data

    # 兩條都不足 / 失敗：優先回 season（caller 走 small_sample no_fire）
    if "error" not in season_data:
        season_data["source"] = "season"
        return season_data
    if "error" not in career_data:
        career_data["source"] = "career"
        return career_data
    return {"error": season_data.get("error", "TTO splits unavailable")}


def _compute_tto_from_statcast(mlbam_id: int, year_start: int, year_end: int) -> dict:
    """從 pybaseball Statcast pitch-by-pitch 聚合成 TTO1/TTO2/TTO3 dict。

    假設 statcast_pitcher DataFrame 含欄位：
      `at_bat_number`, `batter`, `events`, `game_pk`
    （已在 spike 階段驗證：Skubal 2025 樣本 271 pitches 全有）。
    """
    _, statcast_pitcher_fn, _, _ = _import_pybaseball()
    try:
        start = f"{year_start}-03-20"
        end = f"{year_end}-11-05"
        df = statcast_pitcher_fn(start, end, mlbam_id)
        if df is None or df.empty:
            return {"error": "No Statcast data"}

        # PA = events 非 null（每 PA 最後一球的 events 才有值）
        pa_df = df[df["events"].notna()].copy()
        if pa_df.empty:
            return {"error": "No PA events in Statcast data"}

        # 為每個 PA 計算 TTO ordinal：
        # 同一場（game_pk）內，同一打者（batter）依 at_bat_number 升冪排，
        # cumcount + 1 即 1st PA / 2nd PA / 3rd PA
        pa_df = pa_df.sort_values(["game_pk", "at_bat_number"])
        pa_df["tto_ordinal"] = pa_df.groupby(["game_pk", "batter"]).cumcount() + 1

        result: dict = {}
        for ordinal in (1, 2, 3):
            bucket_pas = pa_df[pa_df["tto_ordinal"] == ordinal]
            if len(bucket_pas) == 0:
                continue
            result[f"tto{ordinal}"] = _pa_outcome_aggregates(bucket_pas)
        return result if result else {"error": "No TTO buckets computed"}
    except Exception as e:
        return {"error": f"statcast TTO compute failed: {e}"}


def _pa_outcome_aggregates(pa_df) -> dict:
    """從 PA-level DataFrame slice（一行一 PA，含 events 欄）算 OPS / K% / BB% / BF。

    OBP / SLG / AVG 由 events 計數 + sabermetric 公式合成（PA 不直接給）。
    """
    bf = len(pa_df)
    if bf == 0:
        return {"ops": None, "k_pct": 0.0, "bb_pct": 0.0, "bf": 0}

    events = pa_df["events"]
    h_singles = int((events == "single").sum())
    h_doubles = int((events == "double").sum())
    h_triples = int((events == "triple").sum())
    h_hrs = int((events == "home_run").sum())
    h = h_singles + h_doubles + h_triples + h_hrs

    bb = int((events == "walk").sum())
    hbp = int((events == "hit_by_pitch").sum())
    k = int(events.isin(["strikeout", "strikeout_double_play"]).sum())
    sf = int(events.isin(["sac_fly", "sac_fly_double_play"]).sum())
    sh = int(events.isin(["sac_bunt", "sacrifice_bunt_double_play"]).sum())

    ab = bf - bb - hbp - sf - sh
    if ab <= 0:
        return {"ops": None, "k_pct": round(k / bf * 100, 1),
                "bb_pct": round(bb / bf * 100, 1), "bf": bf}

    obp_denom = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_denom if obp_denom > 0 else 0.0
    tb = h_singles + 2 * h_doubles + 3 * h_triples + 4 * h_hrs
    slg = tb / ab if ab > 0 else 0.0
    ops = obp + slg

    return {
        "ops": round(ops, 3),
        "k_pct": round(k / bf * 100, 1),
        "bb_pct": round(bb / bf * 100, 1),
        "bf": bf,
    }


_TTO_MIN_BF = 30


def _has_sufficient_tto3(data: dict) -> bool:
    if "error" in data:
        return False
    tto3 = data.get("tto3") or {}
    return (tto3.get("bf") or 0) >= _TTO_MIN_BF
```

### 5.2 main 路徑接入（`pitcher_stats.py`）

緊接現有 `platoon_splits = fetch_platoon_splits(...)` 行之後：

```python
platoon_splits = fetch_platoon_splits(pitcher_id, args.year)
tto_splits = fetch_tto_splits(pitcher_id, args.year)
```

並加進寫入 JSON 的 dict：

```python
{
    ...,
    "platoon_splits": platoon_splits,
    "tto_splits": tto_splits,
    ...
}
```

### 5.3 Spike 結果（已完成 2026-05-03）

**Plan A 死亡確認**：
- `sitCodes=ot1,ot2,ot3` / `1,2,3` / `1f,2f,3f` 三組全部 0 splits 回傳
- `careerStatSplits` 同樣 0 splits
- MLB API `/situationCodes` 元資料 602 個 codes 無任何 TTO 切面

**Plan B 可行性確認**（同次 spike 第二輪）：
- `pybaseball.statcast_pitcher('2025-04-01', '2025-04-15', 669373)` → 271 pitches、72 PAs
- 必要欄位 `at_bat_number / batter / pitcher / events / description / game_pk / inning` 全部 PRESENT
- events 分布合理：`field_out 31, strikeout 23, single 11, walk 3, double 2, home_run 1, force_out 1`

下游 implementer 不需再 spike，直接走 §5.1 Plan B 程式碼。

### 5.4 fallback / failure 矩陣（Plan B）

| season Statcast | season tto3.bf | career Statcast | 結果 source | confidence | fired? |
|---|---|---|---|---|---|
| OK | ≥ 30 | (skip) | `"season"` | data | 看 ops/k delta |
| OK | < 30 | OK 且 tto3.bf ≥ 30 | `"career"` | heuristic | 看 ops/k delta |
| OK | < 30 | OK 但 tto3.bf < 30 | `"season"` | small_sample | no_fire |
| OK | < 30 | error | `"season"` | small_sample | no_fire |
| error | — | OK 且 tto3.bf ≥ 30 | `"career"` | heuristic | 看 ops/k delta |
| error | — | error | `{"error": "..."}` | small_sample | no_fire |

**永遠不 abort pipeline。**

### 5.5 PA outcome 映射表

`_pa_outcome_aggregates` 把 Statcast `events` 字串對應到 sabermetric 計數：

| events 值 | 計入 |
|---|---|
| `single` | H, 1B |
| `double` | H, 2B |
| `triple` | H, 3B |
| `home_run` | H, HR |
| `walk` | BB |
| `hit_by_pitch` | HBP |
| `strikeout` / `strikeout_double_play` | K |
| `sac_fly` / `sac_fly_double_play` | SF（不計 AB） |
| `sac_bunt` / `sacrifice_bunt_double_play` | SH（不計 AB） |
| `field_out` / `force_out` / `grounded_into_double_play` / `fielders_choice` / `fielders_choice_out` / `double_play` / `triple_play` / 其他 out 類 | AB（不計 H） |
| `catcher_interf` / `intent_walk` 等罕見 | 一律當非 AB（保守） |

公式：
- AB = BF - BB - HBP - SF - SH
- AVG = H / AB
- OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
- SLG = TB / AB，TB = 1B + 2×2B + 3×3B + 4×HR
- OPS = OBP + SLG
- K% = K / BF × 100
- BB% = BB / BF × 100

罕見 events（如 `catcher_interf`）若未列入計數可能讓 OPS 略偏，但對 TTO 比較信號影響可忽略（兩個 bucket 都同樣保守）。

---

## 6. Signal 細節

### 6.1 新函式 `signal_tto3_penalty`（在 `signals_lib.py`）

緊接現有 8 個 signals 之後（`signal_core_il_count` 之後、`compute_all_signals` 之前）：

```python
# ---------------------------------------------------------------------------
# 9. tto3_penalty — pitcher's TTO3 OPS uplift vs TTO1 (3rd-time-through curve)
# ---------------------------------------------------------------------------

_TTO3_OPS_DELTA_FIRE = 0.100   # ≥ 0.100 → medium fire
_TTO3_OPS_DELTA_HIGH = 0.150   # ≥ 0.150 → high fire
_TTO3_K_DROP_FIRE = 3.0        # K% drop ≥ 3 percentage points → medium fire
_TTO3_MIN_BF = 30              # require ≥ 30 BF in tto3 bucket


def signal_tto3_penalty(tto_splits: dict | None) -> dict:
    """Surface starters whose TTO3 OPS uplift exceeds league-typical curve.

    Fires when (any of):
      - tto3.ops - tto1.ops ≥ 0.100  → medium (≥ 0.150 → high)
      - tto3.k_pct - tto1.k_pct ≤ -3.0 (K% drop ≥ 3pp) → medium

    half_life: structural (multi-year stuff/arsenal/stamina trait).
    Confidence: data (season) or heuristic (career fallback).
    Small sample: tto3.bf < 30 → no_fire + confidence=small_sample.

    Pre-game data only — actual TTO this game is unknown.
    Does NOT auto-trigger run value adjustment; AI in summary judges
    bullpen-load / total-tilt implications.
    """
    name = "tto3_penalty"
    if not tto_splits or "error" in tto_splits:
        return _make(name, False, confidence="small_sample")

    tto1 = tto_splits.get("tto1") or {}
    tto3 = tto_splits.get("tto3") or {}
    bf3 = tto3.get("bf") or 0
    if bf3 < _TTO3_MIN_BF:
        return _make(name, False, confidence="small_sample",
                     details={"tto3_bf": bf3})

    ops1 = _to_float(tto1.get("ops"))
    ops3 = _to_float(tto3.get("ops"))
    if ops1 is None or ops3 is None:
        return _make(name, False, confidence="small_sample")

    k1 = _to_float(tto1.get("k_pct"))
    k3 = _to_float(tto3.get("k_pct"))
    has_k = k1 is not None and k3 is not None

    ops_delta = ops3 - ops1
    k_delta = (k3 - k1) if has_k else 0.0

    fired_ops = ops_delta >= _TTO3_OPS_DELTA_FIRE
    fired_k = has_k and k_delta <= -_TTO3_K_DROP_FIRE

    if not (fired_ops or fired_k):
        return _make(name, False, value=round(ops_delta, 3),
                     details={"tto3_bf": bf3,
                              "source": tto_splits.get("source", "season")})

    severity = "high" if ops_delta >= _TTO3_OPS_DELTA_HIGH else "medium"
    source = tto_splits.get("source", "season")
    confidence = "data" if source == "season" else "heuristic"

    label = (
        f"TTO3 penalty：OPS Δ +{ops_delta:.3f}（TTO1 {ops1:.3f} → TTO3 {ops3:.3f}），"
        f"第三輪明顯衰退"
    )
    if fired_k:
        label += f"；K% 從 {k1:.1f}% 掉到 {k3:.1f}%（Δ {k_delta:+.1f}pp）"
    if source == "career":
        label += "（career fallback）"

    return _make(
        name, True, value=round(ops_delta, 3), severity=severity, label=label,
        details={
            "ops_delta": round(ops_delta, 3),
            "k_delta": round(k_delta, 1) if has_k else None,
            "tto1_ops": ops1, "tto3_ops": ops3,
            "tto3_bf": bf3, "source": source,
        },
        confidence=confidence,
    )
```

### 6.2 註冊 half_life

`_HALF_LIFE_BY_NAME` 第 56 行新增：

```python
_HALF_LIFE_BY_NAME = {
    "tier_mismatch": "structural",
    "heat_vs_babip": "short",
    "platoon_advantage": "medium",
    "strong_park": "structural",
    "reverse_platoon": "medium",
    "chain_break": "medium",
    "pitch_mix_concentration": "medium",
    "core_il_count": "short",
    "tto3_penalty": "structural",  # ← 新增
}
```

### 6.3 `compute_all_signals` per-pitcher 段加一行

第 461–472 行 per-pitcher loop：

```python
for side, p in (("HOME", home_p), ("AWAY", away_p)):
    signals.append(_tag(signal_tier_mismatch(p.get("tier_gap")), side))
    signals.append(_tag(
        signal_reverse_platoon(p.get("platoon_splits"), p.get("pitch_hand")),
        side,
    ))
    statcast = p.get("statcast") or {}
    signals.append(_tag(
        signal_pitch_mix_concentration(statcast.get("pitch_types")),
        side,
    ))
    signals.append(_tag(signal_tto3_penalty(p.get("tto_splits")), side))  # ← 新增
```

---

## 7. Surface 細節

### 7.1 Dossier `## 投手對決` table 新 row（**無條件顯示**）

緊接現有「vs LHB / vs RHB」row 之後，加一 row：

```markdown
| TTO splits | TTO1 .700 / TTO2 .740 / TTO3 .810 (Δ+0.110, 42 BF) | TTO1 .680 / TTO2 .720 / TTO3 .735 (Δ+0.055, 38 BF) |
```

格式規則：
- `TTO1 {ops:.3f} / TTO2 {ops:.3f} / TTO3 {ops:.3f} (Δ{delta:+.3f}, {tto3_bf} BF)`
- `source == "career"` 時整 cell 後綴 `(career)`：
  - `TTO1 .680 / TTO2 .720 / TTO3 .735 (Δ+0.055, 38 BF) (career)`
- `tto_splits = {"error": ...}` 或 `tto3.bf < 30` → cell 顯示 `n/a (sample <30 BF)`
- 投手缺 `tto_splits` key（schema 向下相容）→ `n/a`

實作 helper（`dossier_renderer.py`）：

```python
def _render_tto_splits_cell(pitcher: dict | None) -> str:
    if not pitcher:
        return "n/a"
    tto = pitcher.get("tto_splits")
    if not tto or "error" in tto:
        return "n/a"  # 缺 key / fetch 失敗
    tto1, tto2, tto3 = tto.get("tto1") or {}, tto.get("tto2") or {}, tto.get("tto3") or {}
    bf3 = tto3.get("bf") or 0
    if bf3 < 30:
        return "n/a (sample <30 BF)"  # season + career 都 thin
    o1, o2, o3 = tto1.get("ops"), tto2.get("ops"), tto3.get("ops")
    if o1 is None or o3 is None:
        return "n/a"
    delta = o3 - o1
    suffix = " (career)" if tto.get("source") == "career" else ""
    o2_str = f"{o2:.3f}" if o2 is not None else "?"
    return f"TTO1 {o1:.3f} / TTO2 {o2_str} / TTO3 {o3:.3f} (Δ{delta:+.3f}, {bf3} BF){suffix}"
```

### 7.2 Dossier `## 🎯 訊號摘要`（自動 pick up）

`signals_for_bundle` cache 已涵蓋；renderer 不需修改。fired 時自動出現：

```markdown
🟠 TTO3 penalty：OPS Δ +0.110（TTO1 .700 → TTO3 .810），第三輪明顯衰退
🔴 TTO3 penalty：OPS Δ +0.155（TTO1 .700 → TTO3 .855），第三輪明顯衰退；K% 從 28.0% 掉到 23.0%（Δ -5.0pp）
🟠 TTO3 penalty：OPS Δ +0.105（TTO1 .680 → TTO3 .785），第三輪明顯衰退（career fallback）
```

`half_life: structural` → 不加 ⏳ badge。

### 7.3 Summary `## 風險提示 § 額外信號`（自動 pick up）

同既有 8 signals 處理：fired 時 standard 條目，半衰期 structural 不加 ⏳。`signals_for_bundle` cache 共用。

---

## 8. Reference doc 異動

### 8.1 `reference/matchup-factors.md` §Signals 新 §9 條目

緊接現有 §8 `core_il_count` 之後：

```markdown
#### 9. tto3_penalty（投手）
- 觸發：TTO3 OPS - TTO1 OPS ≥ 0.100 → medium，≥ 0.150 → high；OR K% drop ≥ 3pp
- 樣本：TTO3 BF ≥ 30；season 不足 fallback career（confidence: heuristic）
- 範例：starter TTO1 .700 / TTO3 .810（Δ +0.110）→ 第三輪 OPS 等同聯盟平均打者
- AI 判讀：
  - TTO3 弱（fire）→ 教練可能提早換投，後段牛棚負擔 ↑
  - 同時對手 `core_il_count` fire（牛棚薄）→ 後段失分風險 ↑、總分判讀偏多
  - TTO3 強（不 fire）→ 隱性訊號，AI 可從 dossier `## 投手對決` 表格直接讀「能撐第三輪 → 牛棚消耗少」
- ⛔ **不自動 ±run value**（與 §3 / §8 紀律一致）
```

### 8.2 `reference/matchup-factors.md` §Signals 半衰期表更新

第 273–278 行 structural 列加上 `tto3_penalty`：

```markdown
| structural | （無） | tier_mismatch / strong_park / **tto3_penalty** | 多年 / season-to-date 累計，反身慢，**正常引用** |
```

### 8.3 `CHANGELOG.md` 異動

**移除**過時條目（line 50）：

```diff
- - **wRC+ / Stuff+** — FanGraphs API non-free，不引入
```

理由：5/3 session 已實作 `wRC+`（commit `df165ab`）+ `Stuff+`（commit `ca7d8a1`），條目過時。

**新增**條目（在最頂端 `## 2026-05-03` 之上）：

```markdown
## 2026-05-04 — TTO3 penalty signal (signal #9)

第 9 個 derived signal，pitcher-side per-game。第三輪面對打者 OPS 衰退幅度，
覆蓋 PR-3 後 line 48「第二批 signals」第一項。

- **commit (TBD)** `feat(pitcher)`: 加 `fetch_tto_splits()` (season + career fallback)
- **commit (TBD)** `feat(signals)`: `signal_tto3_penalty` + half_life=structural
- **commit (TBD)** `feat(dossier)`: `## 投手對決` table 加 TTO splits row
- **commit (TBD)** `docs(reference)`: matchup-factors §Signals §9 + 半衰期表

### 紀律保留
- ✅ 信號**不入 scoring formula**
- ✅ 既有 8 signals 行為零變動（compute_all_signals 只追加一行）
- ✅ 4 月小樣本 silent fallback career，BF < 30 統一 small_sample no_fire
- ✅ Dossier TTO row 無條件顯示（mirror vs LHB / vs RHB pattern）
```

---

## 9. Testing

### 9.1 `tests/test_pitcher_stats.py` 新增（~8 tests，Plan B）

| 測試 | 場景 | 驗證 |
|------|------|------|
| `test_pa_outcome_aggregates_all_strikeouts` | DataFrame 全 strikeout | OPS=0、K%=100 |
| `test_pa_outcome_aggregates_basic_mix` | 1 single + 1 walk + 1 K + 1 fly out | OBP / SLG 數學正確 |
| `test_pa_outcome_aggregates_handles_sf_and_hbp` | 含 sac_fly + hit_by_pitch | AB 計算扣除 SF/HBP；OBP 含 HBP |
| `test_compute_tto_from_statcast_assigns_ordinals` | 1 場 9 打者各面對 3 次 | 3 桶各 9 BF、tto1/2/3 都有 |
| `test_compute_tto_from_statcast_empty_df` | mock 回空 DataFrame | `{"error": "No Statcast data"}` |
| `test_fetch_tto_splits_season_full` | mock 回 3 桶 tto3.bf ≥ 30 | source=season、不打 career |
| `test_fetch_tto_splits_falls_back_to_career` | season tto3.bf=15 + career tto3.bf=800 | source=career、打了兩次 statcast_pitcher |
| `test_fetch_tto_splits_both_thin` | season + career 都 < 30 BF | source=season、tto3.bf 仍 < 30（caller 走 small_sample） |
| `test_fetch_tto_splits_pybaseball_raises` | _import_pybaseball 拋 exception | 回 `{"error": ...}` |

### 9.2 `tests/test_signals_lib.py` 新增（~7 tests）

| 測試 | 場景 | 驗證 |
|------|------|------|
| `test_tto3_penalty_fires_ops_medium` | tto3-tto1 OPS Δ = 0.110 | fired=True、severity=medium、label 含 OPS Δ |
| `test_tto3_penalty_fires_ops_high` | OPS Δ = 0.155 | severity=high |
| `test_tto3_penalty_fires_k_drop` | OPS Δ = 0.050（< fire）+ K% drop = 4pp | fired=True（K trigger）、label 含 K% |
| `test_tto3_penalty_fires_both_ops_and_k` | OPS Δ = 0.130 + K% drop = 4pp | fired=True、severity=medium、label 同時含兩段 |
| `test_tto3_penalty_no_fire` | OPS Δ = 0.060 + K% drop = 1pp | fired=False、value 仍存 ops_delta |
| `test_tto3_penalty_small_sample_below_30_bf` | tto3.bf = 25 | fired=False、confidence=small_sample |
| `test_tto3_penalty_career_source_marks_heuristic` | source="career" + fire | confidence=heuristic、label 後綴「career fallback」 |
| `test_tto3_penalty_in_compute_all_signals` | bundle 含 home_pitcher.tto_splits | signals list 含 tto3_penalty + side="HOME" |
| `test_half_life_registry_includes_tto3` | `_HALF_LIFE_BY_NAME["tto3_penalty"]` | == "structural" |

### 9.3 `tests/test_dossier_renderer.py` 新增（~2 tests）

| 測試 | 場景 | 驗證 |
|------|------|------|
| `test_pitcher_table_includes_tto_row_season` | bundle 兩投手都有 tto_splits source=season | row 標題「TTO splits」、cell 含 `Δ+0.XXX` |
| `test_pitcher_table_tto_row_career_suffix` | source=career | cell 後綴「(career)」 |
| `test_pitcher_table_tto_row_small_sample` | tto3.bf=20 | cell == `"n/a (sample <30 BF)"` |

### 9.4 不動的測試

- `scoring_formula` 不變 → 測試不動
- 既有 8 signals 邏輯不變 → 測試不動
- merge_game_data 不變 → 測試不動
- summary_renderer：`signals_for_bundle` cache 已 cover，現有測試覆蓋足夠（不另加）

### 9.5 測試風格（Plan B）

- `_compute_tto_from_statcast` / `fetch_tto_splits` 用 `monkeypatch.setattr(pitcher_stats, "_import_pybaseball", lambda: (None, fake_statcast, None, None))` 注入假 statcast_pitcher（沿用 test_pitcher_stats.py 既有的 `_import_pybaseball` 注入模式 — 同 `lookup_pitcher_id` 那組測試）
- `_pa_outcome_aggregates` 直接 build pandas DataFrame slice 餵測試（無 fetch）
- `_HALF_LIFE_BY_NAME` 寫一個 schema 完整性 test 確保所有 9 條都有 entry
- 所有新測試用 `pytest`，與既有 439 tests 一致

**目標**：439 → ~457 tests（+18）

---

## 10. 整體 sanity check（Plan B）

| 範疇 | 規模 | 風險 |
|------|------|------|
| 程式碼 | 4 檔異動（1 中、3 小）+ 4 新函式（_pa_outcome_aggregates / _compute_tto_from_statcast / fetch_tto_splits / signal_tto3_penalty）| 中，pandas 邏輯需仔細 unit-test |
| Schema 變化 | pitcher.json 加 1 頂層 key（tto_splits）| 低，dossier 用 `.get()` 向下相容 |
| Reference doc | matchup-factors §Signals 加一節 + 半衰期表一格；CHANGELOG 一加一刪 | 無 |
| 測試 | +18 個 | 無 |
| Fetch 開銷 | 每場 +2 次 `statcast_pitcher` season pull（home + away pitcher，每次 ~5–15 秒）；4 月 fallback 額外 +0~2 次 5-year career pull（每次 ~20–40 秒） | 中——pybaseball 已 cache 友善；prepare_game 既有 ThreadPoolExecutor 架構可包；最壞情境 ~80 秒 fetch 時間 |

**沒有公式變更、沒有破壞性 schema 改動、所有失敗路徑都 fallback 不 abort。**

---

## 11. Out of Scope（本次不做、未來可加）

1. **休息天數 / 上一場用球數**（CHANGELOG line 48 第二批 signals 中的另兩項）— 留下批 implementation。
2. **TTO4+ penalty**（先發投手撐到第四輪）— 樣本太稀（聯盟平均 SP 一場 < 5 BF 在 TTO4），不單獨做 signal。
3. **Reliever inheritance penalty**（接班牛棚 inherited runner 數）— 不同維度，留待 bullpen 模組擴展。
4. **TTO penalty per pitch type**（特定球種第三輪衰退）— Statcast 級資料維度，現階段過細。
5. **動態調整觸發閾值**（根據投手 tier）— 例如 elite ace 用更嚴格閾值——維護成本高，留至 backtest 階段再決定。

---

## 12. Open Questions（none）

所有先前討論的開放問題皆已決議：

- Q1（資料來源）→ ~~A：MLB Stats API `statSplits` + `sitCodes`~~ → **Plan B：pybaseball.statcast_pitcher 自行聚合**（spike 證實 MLB API 無 TTO 切面）
- Q2（觸發條件）→ α + K% drop OR-trigger + BF ≥ 30
- Q3（half_life）→ structural（multi-year stuff/arsenal/stamina trait）
- Q4（Bundle key + fallback）→ A：season → 5-year career fallback，confidence drop
- Q5（Surface 整合）→ Dossier visible row + 🎯 訊號摘要 + § 額外信號 + matchup-factors §9
- 順手清理 → CHANGELOG line 50 過時條目移除

Spike 已完成（2026-05-03），Plan B 的 statcast_pitcher 欄位驗證通過，無剩餘不確定性。

---

## 13. Implementation 順序建議（Plan B；給 writing-plans 階段參考）

1. **Spike**：~~驗證 sitCodes~~ — **已完成**，結論寫進 §0 / §5.3。
2. `pitcher_stats.py` — `_pa_outcome_aggregates` helper + 測試
3. `pitcher_stats.py` — `_compute_tto_from_statcast` helper + 測試
4. `pitcher_stats.py` — `fetch_tto_splits` orchestrator + 測試
5. `pitcher_stats.py` — main 路徑接入（fetch 呼叫 + JSON 寫入）
6. `signals_lib.py` — `signal_tto3_penalty` + `_HALF_LIFE_BY_NAME` + 測試
7. `signals_lib.py` — `compute_all_signals` per-pitcher 段加 1 行 + 測試
8. `dossier_renderer.py` — `_render_tto_splits_cell` helper + table 接入 + 測試
9. `reference/matchup-factors.md` — §9 條目 + 半衰期表
10. `CHANGELOG.md` — 移除 line 50 + 加新條目
11. End-to-end smoke test：跑一場真實比賽（5/3 場次有 4 月以上樣本）驗證 dossier TTO row 與 fired signal 正確產出
12. （可選）`docs/follow-up-backlog.md` 加「下批 signals — 休息天數 / 上一場用球數」條目

各步驟獨立可測，commit 粒度 = 一檔一 commit（matchup-factors / CHANGELOG 可合併最終 commit）。
