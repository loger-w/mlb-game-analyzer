# 確定性預測設計（Model B）— 2026-05-28

## 1. 目的

把 mlb-game-analyzer 的**預測數字**（方向 / 總分 / 信心）從「AI 自由心證」改成「script 確定性計算」，消除跨 session drift，讓回測乾淨、可重現、可迭代。AI 退化成**敘事解讀層**，輔助使用者判讀與盤口比對，但不得碰任何數字。

## 2. 背景（為什麼做）

2026-05 軌 A 回測（n=114）揭露兩個問題：

1. **AI 預測沒 alpha**：skill 方向命中 56.6% 沒打贏 Pinnacle 58.4%，也沒打贏純確定性 baseline。AI 判斷對預測準度沒貢獻。
2. **信心是 session 產物，不是訊號**：HIGH 桶嚴重 clustering（5/08、5/09 共 28 場 0 個 HIGH；5/10、5/12、5/26 卻一堆）。每天 summary 由不同 session 補，AI 對信心的給法在 session 之間飄移。「HIGH 桶命中 73-80%」因此被選樣偏差污染，不可信。

結論：信心（與所有預測數字）必須由 script 確定性計算，AI 不碰。

## 3. 不在 scope

- **信號進數字**：v1 所有 9 個 signal 都**只進敘事**，不轉 ±run。哪個信號該進數字由未來 ablation 決定（§10），不在此 spec 拍板。
- **point-in-time 數據抓取**：只回測已凍結的比賽（2026-05 起）。4 月以前不回測（當時沒凍結 analysis-data，現在抓會污染未來資訊）。
- **Total / O-U 的信心**：v1 信心只針對方向。Total 仍輸出數值，O/U 命中在回測量測，但不另給 total 信心。
- **係數在 5 月上 tune**：v1 係數全取自先驗（winprob 曲線、§量級錨點中點）。5 月只當 sanity check，真正驗證等 6 月 out-of-sample。

## 4. 架構：Model B 分工

| 層 | 負責 | 產出 |
|---|---|---|
| **script（確定性）** | 方向 / 總分 / 信心 / 持平判定 | summary.md「整體判斷」段**填好的數字** |
| **AI（敘事）** | 投手 tier 判讀、Flag 8、天氣、9 個信號的風險敘事 | summary.md 敘事 placeholder |

**硬規則**：AI 不得修改 script 填的方向 / 總分 / 信心。AI 的敘事可以「提醒風險」（例「信心 70% 但牛棚空了,留意後段」）但不改數字。

## 5. 數字模型規格

### 5.1 預測得分

沿用既有 `scoring_formula.predict_with_formula`（已含 HOME +0.3 / Total −1.0 校正）：

```
home = LEAGUE_RPG × (home_xwoba / LG_xwoba) × (away_fip / LG_ERA) × (PF/100) + 0.3 − 0.5
away = LEAGUE_RPG × (away_xwoba / LG_xwoba) × (home_fip / LG_ERA) × (PF/100) − 0.5
total = home + away
```

v1 **無信號修正**，故 `adjusted = base`。summary 得分表「+ 信號」欄一律寫 `0`，`adjusted` 欄 = `base` 欄。

### 5.2 方向

```
gap = home − away          # signed
方向 = HOME  if winprob(gap) ≥ 持平門檻 且 gap > 0
       AWAY  if winprob(−gap) ≥ 持平門檻 且 gap < 0
       持平  otherwise（winprob 太接近 50%）
```

### 5.3 信心 = winprob(gap)

**核心改動**：信心不是獨立概念,就是「預測那一側的單場勝率」,由**得分差 → 勝率**曲線換算。

```
winprob(gap) = Φ(gap / S)
  Φ = 標準常態 CDF
  S = 單場 run-margin 標準差 ≈ 4.0（取自歷史 MLB 單場分差分布,非 fit 5 月）
```

起步參考值（S=4.0）：

| gap (run) | winprob | bucket |
|---|---|---|
| 0.30 | 53% | 持平邊界 |
| 0.81 | 58% | LOW↑ / MEDIUM↓ |
| 1.76 | 67% | MEDIUM↑ / HIGH↓ |
| 2.5 | 73% | HIGH |

### 5.4 信心 bucket（沿用既有邊界）

```
winprob < 53%        → 持平（不出方向,不進方向回測分母）
53% ≤ winprob < 58%  → LOW
58% ≤ winprob < 67%  → MEDIUM
winprob ≥ 67%        → HIGH
```

bucket 純粹是 winprob 的標籤,無 penalty、無 AI 介入。

### 5.5 v1 明確排除

- ❌ 無信號 ±run
- ❌ 無信心 penalty（信號衝突 / projected 打線 都不扣分 —— 那需要信號碰數字,與「信號只進敘事」矛盾）
- ❌ 無 AI 對數字的 override

## 6. winprob 曲線（待驗證參數）

- **形式**：常態 CDF `Φ(gap / S)`（或等價 logistic）。形式固定。
- **S 來源**：歷史 MLB 單場 run-margin 標準差,**不得 fit 本回測樣本**。起步 S=4.0,implementation plan 查公開數據定稿。
- **驗證方式**：5 月當 sanity check（看各 bucket 實際命中是否落在 winprob 區間附近）;真正校準等 6 月 out-of-sample。若 5 月嚴重偏離才微調 S,且須記錄理由。

## 7. AI 敘事層

- 角色不變：讀 dossier → 寫投手 tier 判讀、Flag 8（ERA-xERA 運氣 vs 結構）、天氣、9 個信號風險敘事。
- **新增約束**：summary.md「整體判斷」段的方向 / 總分 / 信心由 script 預先填好,AI **只填敘事 placeholder**,不得改動數字欄。
- 信號（reverse_platoon / chain_break / TTO3 / pitch_mix / platoon / core_il 等）全部在此層被敘述,作為使用者判讀 + 盤口比對的質化 context。

## 8. 資料基建（為未來 ablation 凍結特徵）

雖然 v1 信號不進數字,**仍要持久化結構化信號**,目的是讓 6/7 月能跑 ablation（§10）。

- prepare_game 跑完,把 `signals_lib.compute_all_signals(bundle)` 輸出存成 `{GAME_DIR}/signals.json`（含 name / fired / severity / value / side / half_life）。
- 「能存就存」原則：回測會用到的特徵都凍結（信號、formula 輸入已在 merged.json、odds 已存）。
- **5 月 backfill（best-effort）**：從凍結的零件 JSON（away_lineup.json / home_pitcher.json 等）重算 compute_all_signals 補存 signals.json。
  - ⚠️ **限制**：TTO3 / pitch_mix 依賴 Statcast 逐球資料,當時若未凍結則無法 backfill,只能 going-forward。core_il / reverse_platoon / chain_break / platoon 用 lineup/roster 算,應可 backfill。spec 接受此不完整性,ablation 時對缺資料的信號標註樣本受限。

## 9. summary.md 結構變化

| 段 | v1 變化 |
|---|---|
| 修正後預期得分表 | base 欄 = formula；「+ 信號」欄 = `0`；adjusted 欄 = base |
| 整體判斷 - 方向 | script 填（HOME/AWAY/持平） |
| 整體判斷 - 總分 | script 填（= total） |
| 整體判斷 - 方向信心 | script 填（winprob % + bucket） |
| 整體判斷 - 風險 | AI 填（敘事,可引用信號） |
| 其餘敘事段 | AI 填（不變） |

`summary_renderer.py` 改為：數字段直接寫值（非 placeholder）；只有敘事段保留 `<!-- AI 補 -->`。

## 10. 回測整合 + ablation 流程

### 10.1 重跑乾淨 baseline

- script 重算 5 月所有凍結比賽的預測 → 零-drift、可重現 baseline。
- 比較：新確定性版 vs 舊 AI 版 vs market（方向命中 / O/U / calibration）。
- 解決 5/26 hindsight 疑慮：script 不看結果,重算即乾淨。

### 10.2 ablation 流程（未來用,寫進 spec 供參考）

決定某信號是否該進數字：

```
for 每個候選信號 S:
    baseline   = 跑回測（無 S）
    with_S     = 跑回測（S 依 §量級錨點轉 ±run 進 adjusted）
    若 with_S 命中率顯著 > baseline（out-of-sample,n 足夠）→ S 升級進數字
    否則 → S 留在敘事層
```

需 signals.json 凍結（§8）才能跑。6/7 月樣本累積後執行。

## 11. 檔案組織

**新增**：
- `scripts/predict.py` — winprob 曲線 + 方向 + 信心 + 持平判定（純函式,吃 formula_pred 的 gap）
- `scripts/backfill_signals.py` — 5 月舊場重算補 signals.json（best-effort）
- `scripts/tests/test_predict.py`

**修改**：
- `scripts/summary_renderer.py` — 整體判斷段寫填好的數字；得分表 +信號=0；敘事段保留 placeholder
- `scripts/prepare_game.py` — 接 predict.py + 存 signals.json
- `scripts/scoring_formula.py` — 不變（校正已在）

**不動**：
- `signals_lib.py`（已是確定性,含 side 標記,直接用）
- 回測 pipeline `scripts/lib/*`（parse_summary 讀 script 填的數字,格式相容）

## 12. 測試

| 測試 | 內容 |
|---|---|
| `test_predict.py` winprob | 給 gap → 斷言預期 winprob %（對 S=4.0 的已知點:gap 0.81→58%, 1.76→67%） |
| `test_predict.py` 持平邊界 | gap 使 winprob < 53% → 斷言「持平」;≥53% → 斷言出方向 |
| `test_predict.py` bucket | winprob → 斷言正確 LOW/MED/HIGH 標籤 |
| 重現性 | 同一場 merged.json 跑兩次 → 數字完全一致 |
| `test_summary_renderer` | 整體判斷段為填好數字（非 placeholder）;敘事段仍是 placeholder |

不做 mock-heavy 測試,純函式 fixture 即可。

## 13. 邊界處理

| 情況 | 處理 |
|---|---|
| formula 輸入缺（xwoba/fip 為 None） | fallback 聯盟平均（既有行為） |
| gap 剛好落在 bucket 邊界 | 用 `<` / `≥` 明確界定,不模糊 |
| winprob < 53% | 方向 = 持平,信心仍輸出 % 供參考,但不進方向回測分母 |
| signals.json backfill 失敗（資料缺） | 該信號標 `backfill_unavailable`,ablation 時排除 |
| 既有 5 月 summary 已是 AI 數字 | backfill script 用凍結 merged.json 重算覆蓋整體判斷段數字;敘事段可留 |

## 14. 風險與已知限制

1. **「持平」可能變多**：純 winprob + home +0.3 壓過 → 接近五五的場變持平。**實作第一步必須先量化 5 月有方向場數**,若砍到 < ~60 場（方向回測樣本過小）則調「持平」門檻（53%）放寬。這是上線前的 gate。
2. **backfill 不完整**：TTO3 / pitch_mix 的 5 月 ablation 樣本可能不全（§8）。
3. **S 是先驗值**：4.0 取自歷史,單場勝率曲線在不同 run environment 會略移,6 月須驗證。
4. **n=114 仍小**：所有 calibration 結論偏「偵測性」,CI 寬。v1 目標是「乾淨可重現」,不是「證明更準」。
5. **失去 AI 的 context 細緻度**：v1 數字比舊 AI 判斷粗（無信號量級）。賭注是「確定性 + 乾淨回測」價值 > 失去的細緻度,且回測證明舊 AI 判斷沒 alpha,故賭注合理。

## 15. 開放參數（implementation plan 定稿）

- `S`（winprob 曲線標準差,起步 4.0,查公開 MLB 數據）
- 持平門檻（起步 winprob 53%,視 §14.1 量化結果調整）
- bucket 邊界（沿用 58% / 67%,與既有 `_effective_confidence_bucket` 對齊）

## 16. 成功標準

1. 同一場跑兩次,方向 / 總分 / 信心完全一致（零 drift）。
2. 5 月全月回測可一鍵重跑,結果可重現。
3. HIGH/MED/LOW bucket 的場次分布**不再 by-day clustering**（用 script 後,clustering 應消失 —— 這是本 spec 的直接驗收點）。
4. signals.json 對 going-forward 比賽 100% 凍結;5 月 backfill 對 lineup/roster 類信號完成。
