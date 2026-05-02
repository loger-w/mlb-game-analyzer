## 投手對決

### David Peterson (HOME, LHP, 30 yo)
- **Tier 覆寫**：沿用腳本 ⚪ Below Average，但**真實水平偏 🟢 Back-end / 🟡 Solid 邊緣**。
- 真實水平判斷：ERA 5.06 表面差，但 FIP 3.51 / xFIP 3.61 / xERA 4.78 都明顯優於 ERA — ERA-FIP gap = 1.55，顯示前段比賽有 BABIP/LOB 不順的成分。K-BB% 10.4% 屬中下，velo 88.1 mph LHP 偏低（30 歲已進入 📉 退化區），whiff% 10.7 一般，barrel% 僅 4.5 屬優。近 3 場 ER/IP = 10/14.7（5.95），明顯翻車。整體實力是「合格 5 號先發」，本場可期待 5 IP / 3 ER 區間。
- 對手打線威脅：WSH 主力 James Wood (LHB) season OPS .953 / vs LHP .914、Brady House (RHB) vs LHP .891 是兩大威脅；CJ Abrams (LHB) vs LHP 僅 .492，是 Peterson 可凹的一環。Peterson 自己 vs LHB OPS .879 / vs RHB OPS .792（樣本 40/85 BF 噪音大），沒有典型 LHP 對 LHB 優勢，所以 Wood 的左打優勢仍可放大。

### Cade Cavalli (AWAY, RHP, 27 yo)
- **Tier 覆寫**：沿用腳本 🟡 Solid Starter，**FIP / Statcast 顯示往 🟠 Strong Ace 方向探**。
- 真實水平判斷：ERA 4.01 / xERA 3.87 / FIP 2.29 / xFIP 3.29，FIP 與 ERA 落差大（-1.72）顯示運氣偏負面。K-BB% 13.8% 高於聯盟平均，velo 91.2 / max 99.1 mph、hard_hit% 19.5（極優）、barrel% 4.2（極優），27 歲 ⚡ 巔峰期。WHIP 1.66 偏高（保送 + 安打串聯）是唯一弱點 — 容易陷入「人在壘上但壓制擊球品質」的局面。近 3 場 ER/IP = 4/14.3（2.51），近期狀態好。
- 對手打線威脅：**極端 Platoon split** — vs LHB OPS .912 / vs RHB OPS .534（116 BF 樣本已不算小），對 RHB 嚴重壓制。NYM Top 5 是 Bichette (R) / Semien (R) / Robert (R) / Alvarez (R) / Baty (L)，4 R + 1 L，整條打線右打偏重 → **NYM 結構性吃虧**。Brett Baty (LHB) season OPS .568 也不是高威脅左打。Cavalli 對 NYM 打線是壓制等級。

## 打線評級

### HOME — NYM 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用 🟡 Average，**本場降一級看待**。NYM season OPS .642 / xwOBA .314 已偏聯盟尾段，近 10 場 RS 3.10 攻線冰冷（昨日 G1 雖 8-0 但對手是 Littell ERA 7.85），主流右打 Top 4 對 Cavalli (vs RHB OPS .534) 是 Platoon 雙重劣勢。

### AWAY — WSH 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用 🟡 Average。WSH OPS .704 / xwOBA .334 中游偏前段（聯盟 .315 為平均），James Wood 是 plus-plus 中軸（season OPS .953、Barrel% 29.4），對 Peterson 沒有 Platoon 弱勢，Brady House vs LHP .891 是第二把刀。

## 牛棚

| | HOME (NYM) | AWAY (WSH) |
|---|---|---|
| ERA / IL 數 / 核心 IL 估計 | 3.64 / 7 / **2 名核心**（A.J. Minter + Dedniel Núñez） | 5.11 / 6 / 1 名核心（Beeter / Henry 主要為 swing/depth） |

### 牛棚雙向修正值
- NYM 牛棚（雖品質好但 2 核心 IL）：對手 +0.5 run | NYM ML −2~3%
- WSH 牛棚（品質差 ERA 5.11）：對手 +0.5 run | WSH ML −3%

> 牛棚已雙向反映 — 兩隊各加 ~0.5 run 對手分（總分上升 ~1.0 run），但 WSH ML 受傷更重，因為其牛棚 ERA 已是聯盟尾段。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.254):
  - **可能部分回歸但結構性偏弱**：NYM season xwOBA .314（聯盟平均線）、season K% 20.3、近 10 場 RS 3.10 的攻線是「弱不是冷」，BABIP .254 既有運氣成分也反映擊球品質不佳（season barrel% top 主力 Alvarez 14.5 算高，但 Bichette 4.3、Robert 2.9、Semien 7.3 都偏低）。**輕度上修可能（+0.2 run）**，不全額補。

- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.234):
  - **回歸機率較高**：WSH season xwOBA .334 中游偏前、James Wood Barrel% 29.4 + EV95% 64.7 是聯盟頂尖擊球品質，BABIP .234 與其本季實力嚴重背離 — 偏向運氣樣本。**較大幅度上修（+0.3 run）**，但因面對 Peterson + Citi Field 抑制，不全額。

## 條件修正

- Park Factor: 96.0 → **−0.20 run**（Citi Field 投手友善，HR PF +7% 但 Runs 抑制）
- 先發 tier：Cavalli (🟡→🟠) > Peterson (⚪→🟢) — 強度落差 0.5-1 級 → 整體不到 "雙方皆 Solid+" 觸發 -0.5 的標準（Peterson 偏弱），不啟動雙先發抑制
- 連戰背景：本場為 Mets vs Nationals 系列 G2，G1 由 Holmes 完封 8-0，無 Doubleheader。
- 天氣：4 月底 Citi Field 一般 60-70°F，無極端氣象修正。

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| NYM (HOME) | 2.3 | +0.5 (WSH 牛棚弱) +0.2 (BABIP 微回歸) −0.3 (Cavalli 對 RHB 壓制 + Platoon) = **+0.4** | **2.7** |
| WSH (AWAY) | 3.8 | +0.5 (NYM 牛棚 2 核心 IL) +0.3 (BABIP 回歸) +0.2 (Wood 左打優勢) = **+1.0** | **4.8** |
| Total | 6.1 | **+1.4** | **7.5** |

## 整體判斷

- **方向（基本面）**：偏向 **WSH 領先**。Cavalli 是本場品質決勝因子（FIP 2.29，對 RHB OPS .534），完整壓制 NYM 主流右打；Peterson 雖近期翻車但 FIP 3.51 不糟，問題在 NYM 自家攻線冷 + Cavalli 質感太強，主場優勢被抵銷。投打結構 + 牛棚雙弱（WSH 5.11 / NYM 2 核心 IL）讓總分輕度上揚。
- **總分（基本面）**：修正後 7.5，O/U line 7.0 → 差距 +0.5 run，**遠低於 1.5 run 推薦門檻**，總分側趨向 PASS。
- **信心**：MEDIUM（WSH ML / Run Line 方向）— Cavalli > Peterson 質感差距清楚，但 NYM 主場 + WSH 牛棚 ERA 5.11 兩個反向因子限制信心。
- **風險**：
  1. Cavalli WHIP 1.66 偏高，若控球失常可能短局數退場，將比賽交給 WSH ERA 5.11 牛棚 → WSH 的優勢瞬間反轉。
  2. NYM 近 10 場 RA 3.10（守備改善），主場可能拖到延長或低分；Peterson 近 3 場 ER 10 是樣本噪音還是退化未明。
  3. NYM 昨日 G1 攻 8 分，攻線可能反彈，BABIP .254 純樣本噪音的話 NYM 得分會超出 2.7。
  4. James Wood 個人狀態決定 WSH 上限 — Top 4 之後 WSH 打線深度不及 NYM。

⛔ MUST NOT contain：星級、明確盤口推薦
