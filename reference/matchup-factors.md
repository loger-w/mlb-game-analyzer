# 對決修正因子

## 先發投手進階數據

### 核心指標（必查）

| 指標 | 說明 | 來源 |
|------|------|------|
| ERA / FIP / xERA / xFIP | 投球品質 | MLB Stats API / Statcast |
| K% / BB% / K-BB% | 三振保送（K-BB% 預測力最高） | MLB Stats API |
| WHIP / HR/9 / GB% | 被上壘與球質 | MLB Stats API |
| Hard Hit% / Barrel% | 被擊球品質 | Statcast |
| 球種組合、使用率、Run Value | 球種分析 | Baseball Savant |
| xwOBA（被） | Statcast 預期被上壘 | Statcast |

### 投手數據權重

| 資料來源 | 角色 |
|---------|------|
| 本季全部數據 | 主要基準（ERA/FIP/K-BB%） |
| 近 30 天趨勢 | 僅識別結構性改變（球速跳升、新球種） |
| 去年數據 | 本季樣本有限時回歸參考 |
| 投影系統 | ZiPS/Steamer 交叉驗證 |

### 投手實力分級

| 等級 | 定義 | 參考標準 |
|------|------|---------|
| 🔴 Elite Ace | 當代頂尖 | ERA < 2.50 + K-BB% > 20% |
| 🟠 Strong Ace | 全明星級 | ERA 2.50-3.20 + ERA+ 130-170 |
| 🟡 Solid Starter | MLB 穩定先發 | ERA 3.20-4.20 + ERA+ 100-130 |
| 🟢 Back-end | 中後段先發 | ERA 4.20-5.00 + ERA+ 80-100 |
| ⚪ Below Average | MLB 邊緣 | ERA > 5.00 或 ERA+ < 80 |

同時蒐集：投球手 R/L、年齡、休息天數、近 5 場用球數。

---

## 打線分析

**打線來源**（由 `lineup_analyzer.py` 自動偵測）：
- 🟢 **official**：球隊已公布今日打序（賽前 ~2-4 小時 API 才填），9 人 1-9 棒順序為實際打序
- 🟡 **projected**：打序未公布，採 active roster（排除 IL）按 PA 降序取前 9 人作近似

**評級邏輯不分 source**：tier / chain / over_under_lean / 觸發條件對兩種來源一致。
**差異**：official 路徑下 `chain.obp_top3` / `slg_mid` 是真實 1-3 棒 / 4-5 棒；projected 是 PA 排序近似。

對打線核心（1-9 棒）查詢：xwOBA、OPS、OBP、SLG、ISO、K%/BB%、Hard Hit%、Barrel%、BABIP、xBA、xSLG。

**打線評級**：🔴 Elite / 🟠 Strong / 🟡 Average / 🟢 Weak
**近期熱度**（近 7 天）：💪 Hot / ⚖️ Normal / 😓 Cold
**串聯分析**：1-3 棒上壘能力、4-5 棒清壘能力、弱點位置、左右手組成。

同時查詢：Platoon splits、BvP 歷史對決（≥ 15 PA 才有參考價值）、球種對決。

### BABIP 回歸風險標註

- 近 7 天 BABIP ≤ .260 或 ≥ .370 → 由 `prepare_game.py` 自動偵測，於 dossier 與 summary 的「## 風險提示」段標 ⚠️
- AI 在敘事中判讀「可能回歸 / 可能持續」，**不自動 ±run value**
- 聯盟平均 BABIP ≈ .300，需 ~800 AB 才穩定 — 7 天樣本噪音極大，自動修正等同賭運氣

---

## 牛棚分析

整體品質（ERA、FIP、K-BB%）、關鍵角色可用性、近 3 天用球消耗。

### 牛棚傷兵累計效應

「核心」定義：Closer + Primary Setup + High-leverage reliever（低槓桿角色不計入）

| 缺陣人數 | 影響度 | 分析提示 |
|---------|------|---------|
| 1 名核心 | 🟠 中高 | 該隊後段防守變薄，對手末段得分機會增加 |
| 2 名核心 | 🔴 高 | 牛棚明顯吃緊，AI 在風險段需明確判讀對總得分與勝率方向的影響 |
| 3+ 名核心 | 🔴🔴 極高 | 牛棚崩盤等級，本場不確定性顯著放大 |

> 牛棚傷兵在 summary 的「整體判斷 / 風險」段需明確指出對得分與方向的判讀（不限格式）。

### 牛棚替補品質反向檢查

- 替補 ERA < 被替換者 → 不扣分或微調
- 替補是新秀 → 搜尋 MiLB 數據
- 球隊牛棚深度前 10 → 核心缺 1 人影響較小

---

## 傷兵影響過濾

| 球員角色 | 影響度 | 處理 |
|---------|--------|------|
| 今日先發投手帶傷 | 🔴 高 | 搜尋近期數據變化 |
| 打線主力缺陣 | 🔴 高 | 評估替補 vs 主力落差 |
| 牛棚核心不可用 | 🟠 中高 | 見牛棚累計效應 |
| 輪值其他先發受傷 | ⚪ 無影響 | **不納入本場分析** |
| 板凳/替補 | ⚪ 低 | **不納入** |

### 反向檢查

- 搜尋球隊有/無該球員的近期戰績
- 缺該球員時勝率未下降 → 影響度降級
- 高薪低效球員缺陣 → 替補可能更好

---

## 傷病與手術復出

### Tommy John Surgery 復出分級

| 復出階段 | 定義 | 評估方式 |
|---------|------|---------|
| 🔴 復出首年 | 術後第一個完整賽季 | 搜尋術後 vs 術前 2 年數據 |
| 🟠 復出次年 | 第二個完整賽季 | 通常有改善趨勢 |
| 🟡 完全恢復 | 第三年起 | 正常評估 |
| 🔴 二次 TJ | 曾動 2 次 | HSS 研究：僅 65% 回到 MLB |

其他：肩部手術（搜尋術後數據）、膝蓋/腳踝（搜尋 sprint speed）、腦震盪（注意適應期）。

### 投手角色轉換（牛棚 → 先發）

| 面向 | 影響 |
|------|------|
| 體力分配 | 球速/銳度可能下降 |
| 投球局數 | 初期限制 4-5 IP |
| 球種組合 | 需展開 4-5 球種 |
| ERA 膨脹 | 前 3-5 場 ERA 偏高 |

**規則**：
1. 不得直接使用牛棚時期 ERA/FIP 評估先發表現
2. 搜尋最近一次先發時期數據作為基準
3. 回歸先發前 3 場降級一檔評等
4. 已有先發 game log → 以實際先發數據為主

---

## 球員年齡退化

**打者**：20-26 📈 / 27-29 ⚡ / 30-32 📉 / 33-35 📉📉 / 36+ 📉📉📉
**投手**：20-24 📈 / 25-29 ⚡ / 30-33 📉（球速年降 0.3-0.5 mph）/ 34-36 📉📉 / 37+ 📉📉📉

> 若 30+ 歲但 Statcast 維持/提升 → 降低退化修正。本季數據已反映退化 → 不額外修正。

---

## 球場 & 天氣

### Park Factor
資料源：`scripts/data/park_factors.json`（2023-2025 3 年加權，Baseball Savant）
- 修正公式：`E[R] × (PF / 100)`
- 解析：100 = 聯盟平均；> 100 打者友善；< 100 投手友善

**分裂型球場**（Runs PF 與 HR PF 反向，特別處理）：
- Kauffman Stadium：Runs 106 / HR 91 — 利安打與三壘打，壓制 HR
- PNC Park：Runs 102 / HR 83 — 利二三壘打，HR 嚴重壓制
- UNIQLO Field at Dodger Stadium：Runs 98 / HR 121 — 抑制總得分但加成 HR

**近期重大改造**（影響 PF 解讀）：
- Camden Yards 2025 季前左外野牆移近、降低 → 預期由投手友善（96）逐步轉為打者友善（3 年加權尚未反映完整效應）
- Progressive Field 2024 移除外野貨櫃 → 風洞效應，LHB HR +16%
- 臨時主場：Athletics（Sutter Health）/ Rays（Steinbrenner）— 樣本期短

> ⛔ Coors Field 4 月：物理上空氣密度比夏季高 ~8-10%，4 月 PF ≈ 112，5 月後恢復 131。

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

---

## Signals（輔助信號）

PR-3（2026-05-03）後新增 `signals_lib`，8 個 derived signals，dossier 頂部 `## 🎯 訊號摘要` 與 summary `## 風險提示 § 額外信號` 雙處 surface。**信號不入 scoring formula**，AI 在 summary 判讀。

### 信號規範

每個 signal 共用 contract：
```python
{name, fired, value, severity, label, details, confidence}
```
- `fired` True 才會在 dossier / summary 出現
- `severity` ∈ {low, medium, high}（dossier emoji 對應 ℹ️ / 🟠 / 🔴）
- `confidence` ∈ {data, heuristic, small_sample}

### 8 個 signals 觸發條件 + AI 判讀指引

#### 1. tier_mismatch（投手）
- 觸發：`|tier_v2.score − ERA-only_score| ≥ 15`（|gap| ≥ 20 → high）
- gap > 0 「ERA 低估真實水平」；gap < 0 「ERA 高估真實水平」
- AI 判讀：運氣偏差 vs 結構性突破。**不自動下修預測**（與 Flag 8 紀律一致）

#### 2. heat_vs_babip（打線）
- 觸發：🔥 Hot + last7 BABIP ≥ 0.350 → lucky-hot；🥶 Cold + ≤ 0.270 → unlucky-cold
- AI 判讀：熱度是否含運氣 / 冷期是否將反彈。**不自動 ±run value**（與 Flag 3 紀律一致）

#### 3. platoon_advantage（打線 vs 對手手別）
- 觸發：top 5 中 ≥ 4 人 vs-this-hand OPS 比 season OPS 高 ≥ 0.050
- AI 判讀：本場打線對該手別優勢明顯，是否影響 chain 連續性

#### 4. strong_park（球場）
- 觸發：Park Factor ≥ 110（打者友善）或 ≤ 90（投手友善）。≥ 115 / ≤ 85 → high
- AI 判讀：與既有 §Park Factor 條件修正一致對待，不重複加 ±run value

#### 5. reverse_platoon（投手）
- 觸發：vs LHB / vs RHB OPS 與 handedness 預期反向 |Δ| ≥ 0.080，兩側 BF ≥ 30
- 範例：sweeper-heavy RHP 對 RHB OPS 比對 LHB 還高
- AI 判讀：本場對手核心打者手別組成是否放大此風險

#### 6. chain_break（打線）
- 觸發：1-9 棒按 caller 順序，最大相鄰 OPS 落差 ≥ 0.150
- 範例：Alonso .367 / O'Neill .286 對 LHP，#4-5 chain breaks
- AI 判讀：對得分串聯性的影響，是否壓制總分上限

#### 7. pitch_mix_concentration（投手）
- 觸發：max usage % ≥ 45%（single-pitch dependent）或 < 25%（balanced 4+ pitches）
- AI 判讀：single-pitch 投手對 platoon-advantaged 打線抗性弱；balanced 投手難對位

#### 8. core_il_count（牛棚）
- 觸發：本隊 IL 上 core_role ∈ {Closer, Setup, High-leverage RP, Co-Closer} 計數 ≥ 1
- 階梯：1 = 🟠 中高、2 = 🔴 高、3+ = 🔴🔴 極高
- AI 判讀：對應 §牛棚傷兵累計效應 1/2/3+ 名分級

#### 9. tto3_penalty（投手）
- 觸發：TTO3 OPS - TTO1 OPS ≥ 0.100 → medium，≥ 0.150 → high；OR K% drop ≥ 3pp
- 樣本：TTO3 BF ≥ 30；season 不足 fallback 5-year career（confidence: heuristic）
- 資料：pybaseball Statcast pitch-by-pitch，依 (game_pk, batter) cumcount 自行算 TTO ordinal（MLB Stats API 不曝光此切面）
- 範例：starter TTO1 .700 / TTO3 .810（Δ +0.110）→ 第三輪 OPS 已達聯盟平均打者水準
- AI 判讀：
  - TTO3 弱（fire）→ 教練可能提早換投，後段牛棚負擔 ↑
  - 同時對手 `core_il_count` fire（牛棚薄）→ 後段失分風險 ↑、總分判讀偏多
  - TTO3 強（不 fire）→ 隱性訊號，AI 可從 dossier `## 投手對決` 表格直接讀「能撐第三輪 → 牛棚消耗少」
- ⛔ **不自動 ±run value**（與 §3 / §8 紀律一致）

### Signals 與紀律 Flag 的關係

| 層級 | 處理 | 自動 ±run value? |
|-----|------|---------------|
| Flag 3/8 | 紀律硬規則，summary `## 風險提示` 主段渲染 | ⛔ 不自動 |
| Signals | 輔助觀察，dossier `## 🎯 訊號摘要` + summary `### 額外信號` | ⛔ 不自動 |

**重疊處理**：tier_mismatch 與 Flag 8 同源、heat_vs_babip 與 Flag 3 同源 → 自動從 summary `### 額外信號` 排除避免雙列。dossier `## 🎯 訊號摘要` 兩者都列（不同層級的 surface）。

### Signals 半衰期（⏳ 標記）

每個 signal 帶 `half_life` 分類，對應「對手反應有多快會把這個 signal 治療掉」：

| 半衰期 | 標記 | 對應 signals | 判讀建議 |
|-------|------|-------------|---------|
| structural | （無） | tier_mismatch / strong_park / tto3_penalty | 多年 / season-to-date 累計，反身慢，**正常引用** |
| medium | （無） | platoon_advantage / reverse_platoon / chain_break / pitch_mix_concentration | season split / 季中可調，**通常可信但留意對手換人** |
| short | ⏳ | heat_vs_babip / core_il_count | last7 / 每天異動，**帶懷疑解讀** — 對手可能立即調整 |

⏳ 標記出現在 dossier `## 🎯 訊號摘要` 與 summary `### 額外信號` 的 signal label 前。AI 在 summary 對 ⏳ signal 的引用應該寫類似「last7 BABIP 偏高，但對手投手有 7 天時間調整 mix」這種帶反身性的解讀，而不是當成穩定信號。

源頭實作：`scripts/signals_lib.py:_HALF_LIFE_BY_NAME`。

