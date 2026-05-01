## 投手對決

### Brandon Williamson (HOME, LHP, 28y ⚡)
- **Tier 覆寫**：⚪ Below Average（沿用腳本）
- 真實水平判斷：ERA 5.40 / xERA 6.93 / FIP 5.90 / xFIP 5.80 — 各層級指標一致指向 below average，無「運氣不好」空間。**K-BB% -0.9（保送大於三振）** 是結構性災難，球速 87.1 mph 不具壓制力，barrel% 10.3 偏高。GS 5 場樣本雖小，但所有進階指標互相佐證。
- 對手打線威脅：Rockies 整體 vs LHP 偏弱（top 5 中 Rumfield .358、Tovar .607、Johnston .423 都 vs LHP 不佳），但 Goodman .837 與 Karros .764 vs LHP 健康，Williamson 的高 BB% 對任何打線都是送禮。預期 Rockies 客場仍可打出 4-5 分。

### Tomoyuki Sugano (AWAY, RHP, 36y 📉📉)
- **Tier 覆寫**：🟢 Back-end（腳本給 🟡 Solid Starter，**降一檔**）
- 真實水平判斷：ERA 3.42 看似亮眼，但 **xERA 6.15（差距 -2.73！）**、hard_hit% 29.1、**barrel% 15.5（極高）**、whiff% 9.3 — Statcast 全面顯示他被打爆，是 BABIP/HR 運氣護盤。36 歲球速 88.5 mph，FF/FS/FC 三球種無壓制球。近 3 場 4ER/16.7IP 是樣本期紅利，回歸壓力極大。
- 對手打線威脅：Reds 全隊 vs RHP 火力可觀（xwOBA .365 / OPS .725 整體偏高），近 7 天 De La Cruz 1.231、Stewart .982、McLain .900、Friedl .920、Steer .962 群熱。Sugano 的 barrel rate 15.5% × GABP HR+29% = 高 HR 風險，預期失分 5.5-6.5。

## 打線評級

### HOME — 🟠 Strong / 🔥 Hot
- **Tier 覆寫**：沿用腳本。xwOBA .365 是真實水平（樣本 29 場），近 10 場 RS 6.10 與 xwOBA 一致。BB% 11.5 / K% 21.8 體質健康。對位 RHP Sugano 弱投是理想 matchup。

### AWAY — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用腳本。xwOBA .315 偏弱，近 10 場 RS 僅 3.50（季均 3.97）。整體進攻處於低潮。`last7 BABIP=0.370` 是運氣補丁——若 BABIP 回歸，得分將持續萎縮。

## 牛棚

| | HOME (CIN) | AWAY (COL) |
|---|---|---|
| ERA / IL 數 / 核心 IL 估計 | 2.83 / 3 / Caleb Ferguson 高槓桿（IL15d）為 1 名核心 | 3.73 / 4 / 多為 60d 長期 IL，影響度低 |

### 牛棚雙向修正值
- HOME 牛棚（CIN 2.83 elite，1 名核心 IL）：對手 +0.3 run | HOME ML -2%
- AWAY 牛棚（COL 3.73 平庸，IL 多為 60d 結構性）：對手 +0.0 run | AWAY ML 0%

## 風險提示

- ⚠️ HOME 投手 Flag 13 (era_xera_delta=-1.53):
  - **結構性，不是運氣**。xERA 6.93、FIP 5.90、xFIP 5.80 全部一致，K-BB% 為負是體質問題。腳本 Tier 已給 ⚪ 不需再下修，但提示：Williamson 不會因「運氣回歸」變強，他就是這個水平。
- ⚠️ AWAY 投手 Flag 13 (era_xera_delta=-2.73):
  - **強烈運氣成分**。Statcast 看 Sugano 是 ⚪/🟢 等級，但他憑 BABIP/HR/A 順序避開大局。結構性下行壓力高。**降一檔至 🟢 Back-end，預期得分上修 +0.5 run（Reds 額外得分）**。
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.37):
  - **可能回歸**。Rockies 近 10 場 RS 3.50 已遠低於 base 6.6，BABIP .370 是支撐這 3.50 的關鍵。BABIP 回歸 → 得分再下修。**AWAY 預期得分下修 -0.4 run**。

## 條件修正

- Park Factor: 104.0 → +0.20 run（已內含於 base）
- GABP HR +29%、Sugano barrel% 15.5 → HR 路徑放大，**單獨對 HOME 得分 +0.3 run**
- 雙方先發無 🟠+ 級別 → 不適用「Strong Ace 雙王 -1.0」下修
- 無 doubleheader、無極端天氣
- Williamson LHP × COL vs LHP 整體偏弱 → **HOME 失分下修 -0.3 run**

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (CIN) | 6.2 | +0.5 (Sugano xERA 真實水平) +0.3 (HR 放大) +0.0 (其他) | **7.0** |
| AWAY (COL) | 6.6 | -0.4 (BABIP 回歸) -0.3 (vs LHP 但 Williamson 仍爛維持基準) -1.0 (近 30 場 RS 3.97 vs base 6.6 修正) | **4.9** |
| Total | 12.8 | — | **11.9** |

> ⚠️ 修正後 11.9 仍遠高於 9.5 線（差距 +2.4 run）。但需考量：（a）兩位先發實際 IP 不長 → 牛棚承擔大部分局數；（b）CIN 牛棚 ERA 2.83 elite 會壓制 COL 後段得分；（c）COL 客場、面對 LHP，疊加因素易低於修正值。

**保守基本面區間：HOME 5.8-7.0 / AWAY 3.8-5.0 / Total 9.6-12.0**，中位數 ≈ 10.8。

## 整體判斷

- **方向（基本面）**：傾向 OVER 9.5，因雙方先發 xERA 都在 6+ 區間 + HR-friendly 球場 + Reds 打線發燙；但保守區間下緣（9.6）緊貼線，存在 PASS 可能。
- **總分（基本面）**：~10.8-11.9，差距 9.5 線 +1.3 ~ +2.4 run。
- **信心**：MEDIUM（Williamson 樣本只有 5 GS、Sugano 近期表現雖屬運氣仍是事實、雙方都有 Flag 13 不確定性高）。
- **風險**：
  1. Sugano 連續 3 場壓制不是 0% 概率延續，xERA 不等於下一場必然爆
  2. CIN 牛棚 elite (2.83) 會把 COL 在 4-9 局得分壓很低
  3. COL 近 10 場 RS 僅 3.50，可能直接打出 3 分以下
  4. 4 月份氣溫不利長打（GABP 春季 HR 比夏季少）
- **ML 方向**：CIN 較強（home + 打線 + 牛棚），但雙方先發都是 below-average，model 勝率不會比市場 implied 59% 強多少 → 大概率 PASS 等級。
