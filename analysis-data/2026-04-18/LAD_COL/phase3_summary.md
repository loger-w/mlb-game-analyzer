# Phase 3 Summary — LAD @ COL 2026-04-18 20:10 ET

**game_pk**: 824372 | **venue**: Coors Field (park 115) | **status**: Scheduled

## 投打對決

### LAD 打線 vs COL SP Ryan Feltner
- LAD 打線：🟠 Strong, OPS .835 / xwOBA .368 / BABIP .342 / K% 23.5
- Feltner 2026：ERA 7.30 / **xERA 8.60（比 ERA 更差）**, barrel 19.5%, EV95 53.7%, hard-hit 31.9%
- Feltner split：vs LHB .059/.111/.059 (18 PA 小樣本), vs RHB .424/.500/.848 (38 PA)
- LAD lineup 含多左打（Ohtani, Freeman, Muncy, Tucker）→ split 字面上 Feltner 壓左打，但 18 PA 樣本不足無法引用
- Prior year 4.75 ERA → 非菁英 baseline，無明顯回歸空間

### COL 打線 vs LAD SP Emmet Sheehan
- COL 打線：🟡 Average, OPS .700 / xwOBA .316 / BABIP .314 / K% 28.6
- Sheehan 2026：ERA 6.60 / xERA 5.48（Δ 1.12，未觸發 Step 2 gate A）, xFIP 3.30, GB% 65.5%
- **Sheehan ERA 比 prior year 2.82 高 3.78 分**（gate B 要求比 prior 低才觸發，未觸發；但反向偏離與高 GB% + 低 xFIP 暗示回歸空間）
- Sheehan split：vs LHB .300/.364/.725 (44 PA), vs RHB .211/.286/.211 (21 PA)
- COL lineup 偏右打（Tovar/Goodman/Karros/Doyle）→ Sheehan 對右打有優勢

### BvP
- last7_wOBA / BvP 樣本皆 `-`（無資料）→ 跳過 BvP 結論

## 牛棚

- **LAD**: 2.13 ERA / 50.7 IP（表面極佳）— 但 **5 位關鍵 IL**：Graterol, Evan Phillips, Brock Stewart, Jake Cousins, Ben Casparius
- **COL**: 2.86 ERA / 91.3 IP（健康穩定）
- **牛棚雙向閘門觸發（LAD）**：
  - O/U 方向：LAD 牛棚深度受損 → COL 後段得分上修 **+0.5**
  - 同隊 ML 方向：LAD ML 估計略下修（不足以改變 ★ 等級，但納入 D2 信號評估）

## 條件修正

- **Coors Field April**：park factor 115，總分基線 **+0.6**
- 4 月 Coors 仍有 elevation + 低濕度效應；humidor 部分平衡但無法完全抵消
- 無天氣數據（未抓風/溫）— `temperature_f`, `wind_mph` 留空

## 近期狀態 / BABIP 回歸

- **LAD**: 🔥 Hot, team BABIP .342 — 略高於 .300 平均，部分回歸風險；Pages 個人 BABIP .500 極端（小樣本）
- **COL**: ⚖️ Normal, team BABIP .314 — 正常範圍
- Hot/Cold 標記經 BABIP 檢查後：LAD Hot 訊號部分受回歸稀釋，但 xwOBA .368 底層仍支持強打

## 傷兵影響

### LAD（重大）
- **Mookie Betts** (SS, IL-10) — 去年 MVP 等級打線支柱
- **Tommy Edman** (2B, IL-10)
- **Kike Hernández** (1B, IL-60)
- 打線補位（Alex Freeland OPS .565）明顯降級
- 然則 team xwOBA .368 仍維持 🟠 Strong（Ohtani/Freeman/Muncy/Pages 支撐）

### COL（有限）
- Kris Bryant (DH, IL-60)、Kyle Freeland (SP, IL-15)
- 打線影響有限（Bryant 本季尚未建立價值）

## Signal Adjustments 彙整

| Signal | Direction | Magnitude | Rationale |
|--------|-----------|-----------|-----------|
| `coors_april_park` | total ↑ | +0.6 | park factor 115 |
| `lad_bullpen_3plus_il` | COL score ↑ / LAD ML ↓ | +0.5 to COL | 5 位牛棚 IL |
| `sheehan_xfip_regression` | COL score ↓ (mild) | -0.3 to COL | xFIP 3.30 vs ERA 6.60, GB% 65.5%, prior 2.82 |

**Tags 候選**：`insufficient-sample`, `early-season`, `coors-april`, `lad-bullpen-il`, `lad-core-hitters-il`, `sheehan-regression-signal`

## 閘門檢查

- [x] Step 1 roster gate：Sheehan + Feltner 皆 active
- [x] Step 2 gates：A/B 皆未觸發（數值記錄在案，未達閾值）
- [x] BvP：資料不足 → 不引用
- [x] 牛棚雙向閘門：LAD 側已處理
- [x] BABIP 回歸：已檢查

⛔ **不含盤口星級 / 初步推薦**（single source = Phase 4 `prediction.json`）
