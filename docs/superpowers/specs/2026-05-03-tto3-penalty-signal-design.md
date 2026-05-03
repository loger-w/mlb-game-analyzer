# TTO3 Penalty Signal Design

**Date**: 2026-05-03
**Scope**: `mlb-game-analyzer` skill — 在現有 8 個 derived signals 之外新增第 9 個 `tto3_penalty`（先發投手第三輪面對打者 OPS 衰退幅度），覆蓋 CHANGELOG.md line 48 「第二批 signals」中第一個遺留項目。沿用 PR-3 既建立的 signal contract / dossier `## 🎯 訊號摘要` / summary `### 額外信號` 三層 surface 不變動既有架構。

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
| `scripts/pitcher_stats.py` | 新 `fetch_tto_splits()`（season + career fallback）；main 路徑加 call + JSON 寫入 | M |
| `scripts/signals_lib.py` | 新 `signal_tto3_penalty()`；`_HALF_LIFE_BY_NAME` 加 1 行；`compute_all_signals` per-pitcher 段加 1 行 | S |
| `scripts/dossier_renderer.py` | `## 投手對決` table 加 row「TTO splits」 helper + caller | S |
| `reference/matchup-factors.md` | §Signals 加 §9 條目；§Signals 半衰期表 structural 列加 `tto3_penalty` | S |
| `CHANGELOG.md` | 移除 line 50 過時條目；加新條目記 `tto3_penalty` 上線 | XS |
| `scripts/tests/test_pitcher_stats.py` | +5（fetch_tto_splits 各路徑） | S |
| `scripts/tests/test_signals_lib.py` | +6–7（signal 觸發 / fallback / boundary） | S |
| `scripts/tests/test_dossier_renderer.py` | +2（TTO row 渲染） | XS |

---

## 5. Data fetch 細節

### 5.1 新函式 `fetch_tto_splits`（在 `pitcher_stats.py`，緊接 `fetch_platoon_splits` 之後）

```python
def fetch_tto_splits(mlbam_id: int, year: int) -> dict:
    """C2.5：取得投手 Times-Through-Order Splits（TTO1 / TTO2 / TTO3）。

    Season 優先；TTO3 BF < 30 時 silent fallback 至 careerStatSplits。
    回傳：
      {
        "source": "season" | "career",
        "tto1": {"ops": float, "k_pct": float, "bb_pct": float, "bf": int},
        "tto2": {...},
        "tto3": {...},
      }
      或 {"error": "..."} 兩條路徑都失敗時。
    """
    # sitCode 字串需在實作前 5 分鐘 spike 驗證。候選順序：
    #   1. "ot1,ot2,ot3"  (Order through 1/2/3)
    #   2. "1,2,3"        (Times faced 1st/2nd/3rd)
    #   3. "1f,2f,3f"     (Times Faced 1/2/3 alt syntax)
    # 用實際 200 OK + 三桶都有資料的 sitCodes 為準。

    season_data = _fetch_tto_one("statSplits", mlbam_id, year)
    if _has_sufficient_tto3(season_data, min_bf=30):
        season_data["source"] = "season"
        return season_data

    career_data = _fetch_tto_one("careerStatSplits", mlbam_id, year)
    if _has_sufficient_tto3(career_data, min_bf=30):
        career_data["source"] = "career"
        return career_data

    # season 不足且 career fetch 也失敗 / 無資料
    if "error" not in season_data:
        season_data["source"] = "season"
        return season_data  # caller 會走 small_sample 路徑（BF < 30）
    return {"error": season_data.get("error", "TTO splits unavailable")}


def _fetch_tto_one(stats_kind: str, mlbam_id: int, year: int) -> dict:
    """共用 helper。stats_kind = "statSplits" or "careerStatSplits"。"""
    try:
        params = {
            "stats": stats_kind,
            "group": "pitching",
            "sitCodes": "ot1,ot2,ot3",  # spike-verified；若不行 fallback "1,2,3"
        }
        if stats_kind == "statSplits":
            params["season"] = year
        resp = requests.get(
            f"{MLB_API_BASE}/people/{mlbam_id}/stats",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        result = {}
        for sg in data.get("stats", []):
            for split in sg.get("splits", []):
                desc = split.get("split", {}).get("description", "").lower()
                s = split.get("stat", {})
                bf = int(s.get("battersFaced", 0))
                k = int(s.get("strikeOuts", 0))
                bb = int(s.get("baseOnBalls", 0))
                ops = _parse_ops_with_fallback(s)  # ops 缺 → obp+slg fallback
                # 桶名解析：description 含 "1st" / "first" → tto1，依此類推
                key = _classify_tto_bucket(desc)
                if key is None:
                    continue
                result[key] = {
                    "ops": ops,
                    "k_pct": round(k / bf * 100, 1) if bf > 0 else 0.0,
                    "bb_pct": round(bb / bf * 100, 1) if bf > 0 else 0.0,
                    "bf": bf,
                }
        return result if result else {"error": f"No TTO split data ({stats_kind})"}
    except Exception as e:
        return {"error": f"{stats_kind} fetch failed: {e}"}


def _parse_ops_with_fallback(stat: dict) -> float | None:
    """OPS 優先；缺 → OBP+SLG fallback（mirror signal_reverse_platoon 第 257-262 行）。"""
    try:
        return float(stat["ops"])
    except (KeyError, TypeError, ValueError):
        try:
            return float(stat["obp"]) + float(stat["slg"])
        except (KeyError, TypeError, ValueError):
            return None


def _classify_tto_bucket(desc: str) -> str | None:
    """把 split description 對應到 tto1 / tto2 / tto3。
    MLB API 回的 description 字串可能是 "1st PA in G as P" / "Times Faced - 1st"
    /「Order Through 1」之類；統一靠 1st/2nd/3rd 子字串判斷。"""
    if "1st" in desc or "first" in desc:
        return "tto1"
    if "2nd" in desc or "second" in desc:
        return "tto2"
    if "3rd" in desc or "third" in desc:
        return "tto3"
    return None


def _has_sufficient_tto3(data: dict, min_bf: int) -> bool:
    if "error" in data:
        return False
    tto3 = data.get("tto3") or {}
    return (tto3.get("bf") or 0) >= min_bf
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

### 5.3 sitCode spike 驗證（implementation 第一步）

實作前先 5 分鐘人工 spike：

```python
import requests
r = requests.get(
    "https://statsapi.mlb.com/api/v1/people/621244/stats",
    params={
        "stats": "statSplits",
        "group": "pitching",
        "season": 2025,
        "sitCodes": "ot1,ot2,ot3",
    },
)
print(r.json())
```

驗證項目：
- HTTP 200
- 回 3 個 splits（每個 description 含 1st / 2nd / 3rd 字樣或可辨識的 TTO ordinal）
- BF / OPS / K / BB 等欄位齊全

若 `ot1,ot2,ot3` 不通 → 改試 `1,2,3` → 改試 `1f,2f,3f`。三組都失敗 → spec 改回 fallback **B**（pybaseball Statcast 路徑做 career-only），re-loop spec self-review。

### 5.4 fallback / failure 矩陣

| season fetch | season tto3.bf | career fetch | 結果 source | confidence | fired? |
|---|---|---|---|---|---|
| OK | ≥ 30 | (skip) | `"season"` | data | 看 ops/k delta |
| OK | < 30 | OK 且 tto3.bf ≥ 30 | `"career"` | heuristic | 看 ops/k delta |
| OK | < 30 | OK 但 tto3.bf < 30 | `"season"` | small_sample | no_fire |
| OK | < 30 | error | `"season"` | small_sample | no_fire |
| error | — | OK 且 tto3.bf ≥ 30 | `"career"` | heuristic | 看 ops/k delta |
| error | — | error | `{"error": "..."}` | small_sample | no_fire |

**永遠不 abort pipeline。**

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

### 9.1 `tests/test_pitcher_stats.py` 新增（~5 tests）

| 測試 | 場景 | 驗證 |
|------|------|------|
| `test_fetch_tto_splits_season_full` | mock 回 3 桶 BF ≥ 30 | source=season、tto1/tto2/tto3 三鍵齊、ops 解析正確 |
| `test_fetch_tto_splits_falls_back_to_career` | season tto3.bf=15 + career mock 200 OK | source=career、用 career 數值 |
| `test_fetch_tto_splits_career_also_thin` | season tto3.bf=10 + career tto3.bf=20 | source=season、回 thin season（caller 走 small_sample） |
| `test_fetch_tto_splits_obp_slg_fallback` | mock stat 缺 ops 但有 obp+slg | ops = obp + slg |
| `test_fetch_tto_splits_api_fail` | requests 拋 exception 兩次 | 回 `{"error": ...}` |

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

### 9.5 測試風格

- 全部 mock `requests.get`（既有 fixture style）
- `_HALF_LIFE_BY_NAME` 寫一個 schema 完整性 test 確保所有 9 條都有 entry
- 所有新測試用 `pytest`，與既有 432 tests 一致

**目標**：432 → ~446 tests（+14）

---

## 10. 整體 sanity check

| 範疇 | 規模 | 風險 |
|------|------|------|
| 程式碼 | 4 檔異動（1 中、3 小）+ 2 新函式（fetch + signal）| 低，邏輯本地化 |
| Schema 變化 | pitcher.json 加 1 頂層 key（tto_splits）| 低，dossier 用 `.get()` 向下相容 |
| Reference doc | matchup-factors §Signals 加一節 + 半衰期表一格；CHANGELOG 一加一刪 | 無 |
| 測試 | +14 個 | 無 |
| API call 數 | 每場 +2 次 statSplits（home + away pitcher）；fallback 多 +0~2 次 careerStatSplits | MLB API 無認證可接受 |

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

- Q1（資料來源）→ A：MLB Stats API `statSplits` + `sitCodes`
- Q2（觸發條件）→ α + K% drop OR-trigger + BF ≥ 30
- Q3（half_life）→ structural（multi-year stuff/arsenal/stamina trait）
- Q4（Bundle key + fallback）→ A：season → career fallback，confidence drop
- Q5（Surface 整合）→ Dossier visible row + 🎯 訊號摘要 + § 額外信號 + matchup-factors §9
- 順手清理 → CHANGELOG line 50 過時條目移除

唯一在實作階段需驗證的不確定性：**MLB Stats API TTO sitCode 字串**（`ot1,ot2,ot3` vs `1,2,3` vs `1f,2f,3f`）— 5 分鐘 spike 即可確認。三組都失敗 → 改 Plan B（pybaseball Statcast career-only 路徑），re-loop spec self-review。

---

## 13. Implementation 順序建議（給 writing-plans 階段參考）

1. **Spike**：`requests.get` 直接打 MLB Stats API 確認 TTO sitCodes 字串。固定 sitCode 寫進 spec 後再開始。
2. `pitcher_stats.py` — `fetch_tto_splits` + helpers + main 路徑接入 + 測試
3. `signals_lib.py` — `signal_tto3_penalty` + `_HALF_LIFE_BY_NAME` + `compute_all_signals` + 測試
4. `dossier_renderer.py` — `_render_tto_splits_cell` helper + caller + 測試
5. `reference/matchup-factors.md` — §9 條目 + 半衰期表
6. `CHANGELOG.md` — 移除 line 50 + 加新條目（commit hash 留空待最終 commit）
7. End-to-end smoke test：跑一場真實比賽（5/3 場次有 4 月以上樣本）驗證 dossier TTO row 與 fired signal 正確產出
8. （可選）`docs/follow-up-backlog.md` 加「下批 signals — 休息天數 / 上一場用球數」條目

各步驟獨立可測，commit 粒度 = 一檔一 commit（matchup-factors / CHANGELOG 可合併最終 commit）。
