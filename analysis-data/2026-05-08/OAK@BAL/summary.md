## 投手對決

### Kyle Bradish (HOME, RHP, 29 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p72, K-BB% p42），gap vs ERA-only = +31.6
  - 部分同意 tier_v2 但保留疑慮。xFIP 3.81 vs ERA 5.03 落差 -1.22（HR/FB 抽中過頭），FIP 4.81 也比 ERA 略好 → 確有運氣成分；但 xERA 5.11 跟 ERA 幾乎吻合（gap -0.08），擊球品質統計不認為他「真的好」。Barrel% 12.9% 偏高 + hard_hit% 26.4% 偏低的組合表示「不常被結實打到，但中招就是大棒」。**真實水平介於 Below Average 與 Solid Starter 之間**，依 Flag 8 紀律不下修 base 預測，但也別把他當穩定 Solid Starter 用。
- **Reverse platoon 信號**：🔴 fired，vs RHB OPS 1.056 (55 BF) > vs LHB .813 (102 BF)，Δ +0.243 high
  - 對手 OAK 1-5 棒含 2 名 RHB（Wilson、Langeliers），其中 **Langeliers vs RHP .1011、last7 OPS 1.406（爆熱）+ Barrel% 16.8%** 是放大此風險的核心點；Wilson vs RHP .735 雖無熱手，仍直接吃 reverse 紅利。3 名 LHB（Kurtz / Soderstrom / McNeil）原本就有正常 platoon 優勢，**Bradish 不論手別都吃虧**，雙重打擊。注意 vs RHB sample 只 55 BF，存在小樣本噪音。
- **對手打線威脅**：🟠~🔴 中高。OAK matchup tier (vs RHP) 上修為 🟠 Strong + 🔥 Hot + Bradish reverse platoon → 上限可觀。但 OAK 打線 #7-8 OPS 落差 0.439（chain_break high）→ 後段串聯薄、3-5 局短票風險。實際得分動能取決於 Bradish 能否把 Langeliers 處理掉。

### Jacob Lopez (AWAY, LHP, 28 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = ⚪ Below Average（xFIP p5, K-BB% p5），gap vs ERA-only = +9.5
  - 同意 ⚪ Below Average，gap +9.5 < 15 不需特別解釋。**K-BB% 0.7%、WHIP 1.90、FIP 6.37、xFIP 5.72 全部一致指向「他真的不好」**。Flag 8 (xERA 3.46 vs ERA 6.60) 在獨立段處理 — 別被低 xERA 騙以為這是強投。avg velo 85.3 mph 對 LHP 屬底層，無壓制力。
- **Reverse platoon 信號**：未觸發（vs LHB OPS .608、vs RHB .944，正常 platoon 走向；vs LHB sample 42 BF 接近門檻但 Δ 是預期方向）
  - n/a
- **對手打線威脅**：🟡 中等。BAL 1-5 棒含 3 RHB（Ward / Alonso / Jackson）+ 2 LHB（Henderson / Basallo），對 LHP **Ward .955 / Basallo .769 / Henderson .748** 都不錯，**Alonso vs LHP 反而 .564** 是冷點；BAL 整體 vs Lopez 對位有優勢但 Alonso 拉低 chain。Lopez TTO3 OPS Δ +0.970（小樣本 31 BF）+ 近 3 場 11ER/13.3IP → **撐不到第 3 輪很可能**，BAL 的真正威脅是「逼 OAK 早換投 → 啃 OAK 牛棚（ERA 4.82）」。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟡 Average
  - 與 season tier 一致 → **同意維持 🟡 Average**。BAL 對 LHP 沒有顯著上修，但 Lopez 本身慘 → 攻擊力來源是「對手投手缺陷」而非「打線特別熱」。
- **chain_break / heat_vs_babip 信號**：HOME chain_break #8-9 OPS Δ 0.258（medium）；heat_vs_babip 未觸發
  - chain_break 出現在 #8-9 → 影響的是「下棒次的延伸火力」，對主力 1-5 棒的清壘效率影響小；medium 嚴重度，本場壓制總分 ~ -0.1 run。

### AWAY — season tier 🟡 Average / heat 🔥 Hot
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong
  - season tier 🟡 Average → matchup vs RHP 上修一檔到 🟠 Strong → **同意上修**。Kurtz vs RHP .932 / Soderstrom .886 / Langeliers vs RHP 1.011 都是強指標，配合 Bradish reverse platoon → **本場 OAK 攻擊上修方向明確**。但 OAK 整體 last7 BABIP .348 接近 0.350 門檻，熱度可能含部分運氣（雖未 fire heat_vs_babip，AI 仍要留心 Langeliers last7 .429 BABIP / McNeil .476 BABIP 的回歸風險）。
- **chain_break / heat_vs_babip 信號**：🔴 AWAY chain_break #7-8 OPS Δ 0.439（high）；heat_vs_babip 未 fire（隊整體 BABIP .348 < .350）
  - chain_break 嚴重在 #7-8 → 影響「中段串聯」，OAK 主力火力集中在 1-5 棒，6 棒以下接不上 → **如果 1-5 棒沒順利上壘清壘，後段難以補火**。對總分上限的壓制 ~ -0.3 run（high 取下界）。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.63 / 7 / **2** (Bautista IL60d Closer + Helsley IL15d Setup) → 🔴 高 | 4.82 / 1 / **0** → 🟢 完整 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 4.63 表面比 OAK 4.82 略好，但**少 closer + setup 兩名核心**（Bautista 60d、Helsley 15d）→ 對應 §牛棚傷兵累計效應 「2 名 → 🔴 高」。**6 局後對 OAK 強打段（Kurtz/Wilson/Langeliers/Soderstrom 4-5 名連環）面對的是非 high-leverage 相對弱投** → 後段失分機率明顯偏高。配合 Bradish 自身 reverse platoon → 「Bradish 撐 5-6 局 + 牛棚被打」是本場主要負面情境。
- AWAY 牛棚：ERA 4.82 中下水準但**核心完整無 IL** → 真正的問題是 Lopez 撐不到 5 局時，**牛棚要吃 4-5 局**。Lopez 近 3 場僅 13.3 IP（場均 4.4 IP）+ TTO3 OPS Δ +0.970 → 預期早換，OAK 牛棚會被消耗超量；雖然沒缺人但 4-5 IP 的負擔對任何牛棚都吃力，BAL 連續打擊機會 elevated。

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=+3.14):
  - **混合解讀**：xERA 3.46 看的是擊球品質（Lopez hard_hit% 20.7%、barrel% 4.0% 都很低 → 沒被打很結實），但 **K-BB% 0.7%、WHIP 1.90、FIP 6.37** 同步顯示「保送爛、自己給自己壓力」。即「擊球品質好但控球差到無法兌現」— 這是**結構性問題**（球速 85.3、球路無壓制力）而非單純運氣。本場判斷：**不下修 base 預測**，但留意「Lopez 可能少被打結實 → 但保送多 → 牛棚提早上 → 失分仍可能堆積」的劇本。

### 額外信號
- 🔴 HOME reverse platoon Δ +0.243（vs RHB OPS 1.056 > vs LHB OPS 0.813）— RHP 對非預期手別反而吃虧
- 🟠 HOME TTO3 penalty：OPS Δ +0.006（TTO1 0.867 → TTO3 0.873），第三輪明顯衰退；K% 從 22.2% 掉到 16.7%（Δ -5.5pp）
- 🔴 AWAY TTO3 penalty：OPS Δ +0.970（TTO1 0.745 → TTO3 1.715），第三輪明顯衰退；K% 從 21.4% 掉到 12.9%（Δ -8.5pp）
- 🟠 HOME chain breaks at #8-9：OPS 落差 0.258
- 🔴 AWAY chain breaks at #7-8：OPS 落差 0.439
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - **本場直接受影響**。Bradish 真實水平不穩 + reverse platoon 對 OAK 兩名 RHB（含爆熱 Langeliers）吃虧 → Bradish 中途下場機率高 → BAL 後段牛棚少 closer/setup → OAK 中後段打點機會 elevated。⏳ short half-life：對手 OAK 已經 5/4-5/7 連戰過 BAL 系列前不熟悉，今天等於 series 第 1 場，對手反應時間 = 1 場（最少緩衝），影響度滿打滿。

## 條件修正

- Park Factor: 96.0 → -0.20 run
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：兩位先發都偏弱（Bradish 🟡 Solid Starter 但 ERA 5.03 / Lopez ⚪ Below Average ERA 6.60），**對決淨值偏向利攻**；非 doubleheader、雙方都是正常輪值。Camden Yards HR +7%（球場原生 HR-friendly）— 雖然 PF 96 整體偏投，但對 Langeliers 這種 barrel% 16.8% 的右打有左外野親和力。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 6.8 | +0.0 | 6.8 |
| AWAY | 5.3 | +0.1 | 5.4 |
| Total | 12.1 | +0.1 | 12.2 |

**信號計算明細**：
- **HOME 攻擊**：AWAY (Lopez) tto3_penalty high 但 small_sample (31 BF) → +0.1；HOME chain_break #8-9 medium → -0.1；platoon_advantage 未 fire（5 人中只 2 達門檻）→ 淨 0
- **AWAY 攻擊**：HOME (Bradish) reverse_platoon high (partial, 2 RHB 中 1 爆熱) → +0.2；Bradish tto3_penalty medium (K% drop 5.5pp，OPS Δ 微) → +0.1；HOME 牛棚 core_il ×2 high → +0.3；同側同向取 max +0.3 加 +0.1 累積規則 → +0.4；AWAY chain_break #7-8 high → -0.3；淨 +0.1
- ⛔ Bradish tier_mismatch (+31.6) / Lopez Flag 8 (era_xera +3.14) 依 Table A 不入錨點，敘事另段

## 整體判斷

- **方向（基本面）**：略偏 HOME（BAL）— base 已給 +1.5 run 優勢，信號修正後仍 +1.4 偏 HOME
- **總分（基本面）**：12.2（信號修正後）— 偏中高，雙方先發都不強 + Camden Yards HR +7%
- **方向信心**：~58%（中性偏弱）— 兩個拉扯方向：base 偏 HOME 1.4，但 OAK 近10 5-5 / streak +1 + last7 .348 BABIP 熱手 vs BAL 近10 3-7 / RA 6.80 防守崩 + −1 streak，動能與近期 trend 反向。下修信心至 58% 而非 65%。
- **風險**：
  1. **Lopez 短票早退**：K-BB% 0.7%、近 3 場場均 4.4 IP、TTO3 OPS Δ +0.970 → 不太可能撐到第 3 輪；OAK 牛棚 ERA 4.82 中下 + 要吃 4-5 IP → BAL 中後段強打段機會多。
  2. **Langeliers 單棒帶走**：vs RHP OPS 1.011、last7 OPS 1.406、Barrel% 16.8% + Bradish reverse platoon (vs RHB 1.056) + Camden HR +7% → 一發 HR 改變 total 走向的可能性高。
  3. **BAL 後段失分**：少 Bautista (Closer) + Helsley (Setup) → 6 局後若 lead，blown save 風險 elevated；OAK 1-5 棒火力強 → 末段如能輪到對非 high-leverage 投手，雙位數失分情境存在。
  4. **chain_break 雙邊 fire + 短票波動**：HOME #8-9 (medium) + AWAY #7-8 (high) 同時觸發 → 兩隊都可能出現「3-5 局沒得分」斷火段，total 偏離 12.2 的方差比平均場大。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組