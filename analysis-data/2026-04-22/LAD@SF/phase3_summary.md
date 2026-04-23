# Phase 3 分析結論 — LAD @ SF (2026-04-22, Oracle Park, PF 96)

## 先發投手

### Ohtani (LAD, RHP) — 🟠 Strong Ace
- 2026: ERA 0.50 / WHIP 0.72 / xERA 2.30 / FIP 2.10 / xFIP 3.40
- K% 26.1 / BB% 8.7 / HR/9 0.00 / GB% 51.4 | 18.0 IP / 3 GS
- Platoon: vs L .083/.175/.167 (40 BF) | vs R .154/.241/.231 (29 BF)
- 球種：FF 40.7% / ST 19.0% / CU 17.6% / FS 11.8% / SI 8.0% (avg velo 89.0, max 100.4)
- Prior 2025: ERA 2.87 / FIP 1.87 / xFIP 2.06 / K% 33.0 / xERA 2.55

### YoY Statcast 對比結論（|ERA − xERA| = 1.80 ≥ 1.5 觸發）
| 指標 | 2025 | 2026 | Δ |
|---|---|---|---|
| avg_velo | 91.5 | 89.0 | −2.5 mph ⚠️ |
| whiff_pct | 15.0 | 14.6 | ≈ 持平 |
| hard_hit_pct | 22.2 | 19.2 | 改善 (−3.0) |
| barrel_pct | 3.4 | 4.5 | 微升 |
| xERA | 2.55 | 2.30 | 改善 (−0.25) |
| pitch_types | FF+ST+SL+CU+FC | FF+ST+CU+FS+SI | 新增 FS/SI，汰除 SL/FC |

**結論**：球速下滑 2.5 mph 是退化警訊（年齡 31，「初期退化」區段），但被擊球品質與 xERA 驗證本季 elite 非運氣；球種組合進化（新增 FS/SI）增加制球空間。綜合維持 🟠 **Strong Ace** tier；實際預期得分以 xERA 2.30 為基準（而非 0.50 ERA）。18 IP 小樣本下的超低 ERA 會有約 1.5-2.0 run 的回歸空間。

### Mahle (SF, RHP) — ⚪ Below Average
- 2026: ERA 7.23 / WHIP 1.93 / xERA 7.17 / FIP 6.96 / **xFIP 3.34** / HR/9 2.89 / GB% 74.2
- K% 23.9 / BB% 13.6 (過高) / 18.67 IP / 4 GS
- Platoon: vs L .257/.350/.429 (41 BF) | **vs R .375/.468/.750 (47 BF) — 災難**
- 球種：FF 44.4% / FS 28.0% / FC 12.7% / SL 10.2% / SI 4.8% (avg velo 89.0, max 94.9)
- Prior 2025: ERA 2.18 / FIP 3.33 / xFIP 4.20 / K% 19.1 / BB% 8.4 / HR/9 0.52（86.7 IP）

**關鍵 tension**：ERA 7.23 ≈ xERA 7.17（品質糟糕非運氣）vs xFIP 3.34（HR/FB 小樣本偏高，有回歸空間）。BB% 13.6 翻倍於去年 8.4 — 制球真實退化；GB% 74.2 小樣本噪音不可信。綜合：真實水平估計落在 ERA 4.5-5.0 區間（介於 xFIP 3.34 與本季 7.23 之間，考慮 age 31 退化）。

## 打線評級

### SF vs Ohtani (RHP) — 🟢 Weak
- 團隊 OPS .649 / xwOBA .294 / BABIP .311 / K% 20.9 / BB% 5.2
- 熱度：⚖️ Normal | L7 BABIP .311（無回歸觸發）
- Chain: OBP top3 .302 / SLG mid .365（串聯弱）
- 前段熱點：Adames vs RHP .857 (69 PA)、Chapman .711、Casey Schmitt .753
- 底段：Bailey .389、Encarnacion .352 — 近乎沒威脅
- BvP：Chapman 9 PA / Devers 12 PA — 均 < 15 不可引用
- **Bader (CF) IL** — 打線評級已反映（lineup_analyzer 只挑可上場 9 人）

### LAD vs Mahle (RHP) — 🟠 Strong (🔥 Hot)
- 團隊 OPS .830 / xwOBA .359 / BABIP .343 / K% 23.2 / BB% 9.7
- 熱度：🔥 Hot | L7 BABIP .343（< .370 無回歸觸發）
- Chain: OBP top3 .365 / SLG mid .603（串聯頂級）
- vs RHP 冷血：Freeman (LHB) .909、Pages .1.038、Muncy (LHB) .925、Ohtani (LHB) .845、Teoscar .843
- LHB 密度高：Ohtani/Tucker/Freeman/Muncy — 正對 Mahle 較好平台側（vs L .779 OPS 允許）
- BvP：Muncy 6 PA / Freeman 7 PA / Smith 8 PA — 均 < 15 不可引用
- **Mookie Betts + Tommy Edman + Kike Hernández IL** — 打線評級已扣抵此缺陣

## 牛棚

### SF BP (3.36 ERA) — B9 閘門觸發（2 名核心 IL）
- IL：Jason Foley、Randy Rodríguez（high-leverage setup）、Sam Hentges
- 核心計 2 人 IL → 修正值：**O/U +0.5 run（加 LAD 得分）、ML −3% SF**

### LAD BP (4.00 ERA) — B9 閘門觸發（3+ 名核心 IL 🔴🔴 極高）
- IL 核心：**Edwin Díaz (closer)**、**Evan Phillips (primary setup)**、**Brusdar Graterol (high-leverage)**
- 其他 IL：Brock Stewart、Jake Cousins、Ben Casparius、Blake Snell (SP)、Gavin Stone (SP)
- 3+ 核心 IL → 修正值：**O/U +1.0 run（加 SF 得分）、ML −5% LAD、信號 +2**

### 牛棚雙向修正值
| 修正來源 | O/U (+run) | ML (%) |
|---------|-----------|--------|
| SF BP IL (2 核心) | +0.5 → LAD | −3% SF |
| LAD BP IL (3+ 核心) | +1.0 → SF | −5% LAD |
| 淨 total 貢獻 | **+1.5 run 總分** | LAD 淨 −2% (彼此衰減) |

## 條件修正

| 信號 | Run Value |
|------|-----------|
| Park Factor 96 | −0.20 total (predict.py 自動) |
| SF BP IL (2 核心) | +0.5（加到 LAD 得分） |
| LAD BP IL (3+ 核心) | +1.0（加到 SF 得分） |
| Ohtani 球速 −2.5 mph (YoY) | 無 Run Value（Statcast 其它面向已驗證 elite）|
| Mahle ERA/xFIP 發散 | 無明確信號（xFIP 正向 vs xERA 負向抵銷） |

> 兩先發不同級（🟠 vs ⚪）— 不觸發「雙方 Ace−1.0」；LAD Hot 單側熱 — 不觸發「雙方 Hot +0.5」。

## 修正後預期得分

- **Base（predict.py formula，已含 PF −0.2）**：LAD 8.2 / SF 2.0 / total 10.2
- **疊加 BP 信號**：LAD 8.2 + 0.5 = 8.7；SF 2.0 + 1.0 = 3.0；total 11.7
- **Mahle 真實水平回歸調整**（本季 18.67 IP 小樣本，ERA 可能高估）：向下修 LAD 約 1.5-2 run → LAD 6.5-7.0
- **最終預期（折衷）**：LAD ~7.0 / SF ~3.0 / total ~10.0

## 整體判斷

- **方向**：基本面明確偏向 LAD（投手差 2 檔以上、打線差 1 檔、近 30 天戰績懸殊 16-7 vs 10-13）
- **信心**：MEDIUM — Ohtani 小樣本 (18 IP) + Mahle 小樣本 (18.67 IP) 均增加不確定性；YoY 驗證雙方都出現結構性變化
- **風險**：
  1. Ohtani 球速下滑是真實退化信號，雖然 Statcast 結果仍 elite，但若某場 velo 持續下探可能被擊出
  2. Mahle 真實水平可能比 ERA 7.23 更接近 4.5-5.0（回歸利多 LAD 以外的 SF 進攻預測）
  3. LAD 牛棚嚴重 IL — 若 Ohtani 提早下班（95 球限制約 6 IP），SF 打爆 LAD BP 的機率明顯上升
  4. SF 前場靠 Webb 贏 3-1（4/21），打線慣性未必延續；但 Ohtani 不是 Webb
  5. Oracle Park 降分效應（PF 96）會抑制 LAD 的大比分潛力
- **基本面方向總結**：LAD 偏大（ML 傾向 AWAY），O/U 偏 OVER（修正後總分 10.0 vs 線 8.5，差 +1.5 在有效門檻邊緣）
