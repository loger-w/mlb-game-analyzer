# Phase 3 綜合分析摘要 — TEX @ SEA (2026-04-18 19:15 ET)

> 盤口推薦（ML / O/U / Run Line / 星級）由 Phase 4 `prediction.json` 產生，本檔只保留基本面結論。

## 3.1 先發投手對決

**George Kirby (SEA, RHP, 28 ⚡ 巔峰期, 🟡 Solid Starter)**
- 2026：ERA 3.25 / xERA 2.90 / FIP 3.57 / xFIP 3.10 / WHIP 0.94 / K% 21.4 / BB% 5.8 / GB% 63.0 / Hard Hit 27.9% / Barrel 5.4% (27.7 IP / 4 GS)
- 平台分裂：vs LHH .180/.268/.260（宰制）/ vs RHH .239/.255/.370（同樣優）
- 近 3 場：6 IP 1 ER @CLE, 6 IP 4 ER @NYY, **8 IP 3 ER vs TEX (4/07)** — 對 TEX 有 8 IP quality start 實績
- 2025 FIP 3.16 / xFIP 2.81（4.21 ERA 是運氣低）— 真實實力在 Solid Starter 上緣

**Nathan Eovaldi (TEX, RHP, 36 📉📉, 表面⚪ / 實質 🟢 Back-end~🟡)**
- 2026：ERA 5.40 / **xERA 3.79** / FIP 4.07 / xFIP 2.87 / WHIP 1.48 / K% 26.8 / BB% 7.2 / GB% 47.4 / Hard Hit 35.5% / Barrel 7.9% (21.7 IP / 4 GS)
- ⛔ **|ERA−xERA| = 1.61 ≥ 1.5 閘門觸發**，已執行 YoY Statcast 對比（`away_pitcher_2025.json`）
- YoY：velo +0.5 / whiff +2.5 / K% +0.8 → **stuff 仍在**；BB% 翻倍 (4.2→7.2)、GB% -14 (61.7→47.4)、Hard Hit +11.6 → **接觸品質 + 控球退化**
- 判定：真實水準 ~xERA 3.79（不是 5.40），但非 2025 Elite 水準，視為 Back-end/Mid-rotation
- 平台：vs LHH 1.19 OPS（致命弱點）/ vs RHH .642（可用）
- 近 3 場：4.67 IP 5 ER @PHI, 4 IP 6 ER @BAL, **6 IP 2 ER vs SEA (4/07)** — 上次對 SEA 表現可

**投手差**：Kirby 勝 1-1.5 檔（skill 約 3.0 ERA 水準 vs Eovaldi 真實 ~3.8-4.0）

## 3.2 打線評級

**SEA 🟡 Average**：avg OPS .675 / xwOBA .328 / BABIP .291 / 🧊 Normal recent heat
- vs Eovaldi RHP：Eovaldi vs LHH 弱點。SEA LHH/S：Raleigh (S), Naylor (L), Young (L), Donovan (L), Raley (L), Rivas (S), Crawford (L) — 7/9 含左打 → **槓桿 Eovaldi vs LHH 缺陷**
- Raley 爆熱（last 7 .440 / 1.182 OPS, babip .625）but xwOBA .429 支撐 → 部分 hot 有效
- Julio last 7 babip .438 但 xwOBA .296 → 回歸下行
- Naylor last 7 babip .158 + season babip .148（xwOBA .304） → 回歸上行
- BvP vs Eovaldi（PA≥15）：Raleigh 21 PA .294/.429/.529 強 / Julio 21 PA .381 強 / Arozarena 31 PA .107/.161/.179 弱 / Raley 15 PA .267/.400 中 / Crawford 22 PA .150/.300 弱 — 混合

**TEX 🟡 Average**：avg OPS .708 / xwOBA .314 / BABIP .289 / 🧊 Normal recent heat
- vs Kirby RHP：Kirby vs LHH 宰制 .180 BA。TEX LHH：Nimmo (L), Seager (L), Carter (L), Pederson (L) — 4/9 左打 → **Kirby 優勢**
- Jung last 7 .407 / babip .435（xwOBA .346 部分支撐）→ 小幅回歸
- Seager last 7 babip .176（xwOBA .364）→ 回歸上行
- BvP vs Kirby（PA≥15）：Seager 20 PA .056/.150/.056 極弱 / Josh Smith 21 PA .200 弱 — TEX 核心打線對 Kirby 歷史壓制

## 3.3 牛棚

- **SEA 牛棚 ERA 3.16**（強）；IL 僅 Bryce Miller (SP)、Vargas/Evans (60-day SP)，**無核心 RP 缺陣**
- **TEX 牛棚 ERA 2.78**（菁英級）；IL 核心：**Chris Martin (high-leverage setup, 15-day)** → 1 名核心缺陣
- 牛棚傷兵累計修正：TEX 扣 1 核心 → **SEA 得分 +0.3 run / TEX ML -2%**（雙向已反映）

## 3.4 條件修正

| 信號 | 數值 | Run Value |
|------|------|----------|
| Park Factor (T-Mobile) | 97 | -3% 總分（~-0.23 run） |
| Eovaldi xERA vs ERA 落差已校正 | xERA 3.79 | 使用 3.79 為基準，非 5.40 |
| TEX 牛棚 Chris Martin IL | 1 核心 | SEA +0.3 run，TEX ML -2% |
| 年齡 Eovaldi 36 | 📉📉 | 已反映於本季數據 — 不額外扣 |
| Kirby last 3 對 TEX 8 IP QS | 實戰 | 加分（+0.1-0.2 Kirby 表現） |
| 天氣 | 未查 | 中性假設 |
| 主審 | 未查 | 中性假設 |

## 3.5 修正後預期得分

- **SEA 預期得分**：Eovaldi 允許 ~3.8 ERA × 5-6 IP + TEX 牛棚 ~3.0 ERA (含 Martin IL +0.3) × 3-4 IP ≈ **3.5-3.9 run**
- **TEX 預期得分**：Kirby 允許 ~2.9-3.2 ERA × 6-7 IP + SEA 牛棚 ~3.16 ERA × 2-3 IP ≈ **3.0-3.4 run**
- **總分預期**：**6.5-7.3**（中位 ~6.9）
- Park 修正後總分 × 0.97 ≈ **6.7**

## 3.6 整體判斷

- **方向性**：基本面 **偏 SEA（主場 + Kirby 投手檔次勝 1 檔 + vs Eovaldi LHH 弱點）**，但 **TEX 牛棚優勢縮小 gap + 本系列 3-0 hot H2H**
- **總分**：基本面 **偏 UNDER**（投手對決品質 + Park 97 + 雙牛棚強）
- **受讓方向交叉驗證**：TEX -1.25 意味市場認為 TEX 可望贏超 1 分；但 Kirby 在主場 vs Eovaldi RHP 差 1 檔，ML implied SEA 55.9% vs TEX 45.9%，讓分方向存在 **受讓盤偏見（反向）嫌疑** — TEX 給 1.25 過多，受讓 SEA +1.25 看起來有價值
- **信心**：Moderate（Eovaldi xERA 校正 + Kirby 對 TEX 實戰表現 + 雙牛棚強 + Park 為 UNDER 提供多路徑）
- **主要風險**：
  1. TEX 已 3-0 壓制 SEA（心理壓制）
  2. Kirby K% 下滑至 21.4（雖 GB% 補），遇 TEX 平均 OPS 較高打線有爆分可能
  3. Eovaldi 若運氣繼續差（BABIP 校正），ERA 可能持續偏 5.0+

## 3.7 Phase 4 預測輸入參數（建議）

- `--adjusted-home` 3.6（SEA）
- `--adjusted-away` 3.2（TEX）
- `--ou-line` 7.5（最近的有效線，7.45 quarter 取 7.5）或 7.0（取下限）
- 信號：`bullpen_martin_il: 0.3`（對 SEA 得分）

> Phase 4 將由 `predict.py --save` 決定最終 ML/OU/Run Line 推薦與星級。
