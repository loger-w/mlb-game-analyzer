## 投手對決

### Andrew Painter (HOME, RHP, 23 📈 成長期)
- **Tier 驗證**：腳本 score-derived tier ⚪ Below Average（ERA 5.28），但 FIP 3.44 / xFIP 3.76 / K-BB% 14.3% 屬 🟡 Solid Starter 區間，**xFIP-blend gap > +15**（ERA 嚴重低估真實水平）
  - 23 歲頂級新秀、IP 僅 29（觸發 Flag 8 小樣本），ERA 5.28 主要來自 BABIP 與序列運氣（barrel 6.3% / hard 19.5% 都遠優於 ERA），**不自動下修**：peripherals 顯示能撐第六局水準的先發體質
- **Reverse platoon 信號（🔴 Δ +0.214）**：vs RHB OPS .930 vs vs LHB OPS .716，FF 38.3% 為主球種對 RHB 命中率高，理論上吃虧
  - 本場 Athletics 上 5 棒：Kurtz / Soderstrom / McNeil / Butler 皆 LHB（4/5），**信號被打線結構大幅稀釋**——只剩 Wilson 與後段右打受惠，影響量級下修
- **對手打線威脅**：Athletics season tier 🟡 Average / matchup vs RHP 🟡 Average，Kurtz vs RHP .958、Soderstrom .893、McNeil .802 構成上段威脅；不過 Painter peripherals + LHB 友善 → 中等威脅、被 5 局內限制 2-3 分為合理基準

### J.T. Ginn (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 🟢 Back-end Starter（ERA 4.30 / xERA 3.75，gap 僅 +0.55），FIP 4.74 / xFIP 4.25 / K-BB% 7.3% 同意 Back-end 定位，**gap 微小、結構合理**
  - 主球種 SI 36.1% RV/100 +2.4 / whiff 22.5% 是亮點，但 K-BB% 7.3% 偏低代表壓制力有限；近 3 場 4 ER / 7 IP（≈2.3 IP/場）顯示常被早早換投
- **Reverse platoon 信號**：未 fire（vs LHB OPS .746 / vs RHB OPS .635，正常 platoon）
- **對手打線威脅**：Phillies vs RHP 🟡 Average，但 Schwarber vs RHP .946、Harper vs RHP 1.033 為兩位 elite tier 打者；Phillies 近 10 戰 RS 4.50、+4 連勝攻擊熱度足夠 → 高威脅，Ginn 撐到第五局以上難度高

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average，與 season tier 一致，無方向修正
  - 但 Schwarber + Harper 為 vs RHP elite 雙核（OPS .946 / 1.033），實質 ceiling 高於「Average」標籤
- **chain_break #4-5（🟠 Δ 0.274）**：Harper（#3）→ García .707（#4）→ Bohm .433（#5）OPS 斷層
  - 4-5 棒清壘能力不足，1-3 棒上壘後續打不回來機率上升 → 壓低本場上限約 -0.1 run

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average，與 season tier 一致
  - Kurtz / Soderstrom / McNeil 三人 vs RHP OPS 均 .800+，上段對 Painter 有威脅
- **chain_break #6-7（🔴 Δ 0.524）**：6-7 棒落差明顯，後段攻擊鏈幾乎斷掉
  - 上 5 棒打不到時整場啞火機率高 → 壓低本場上限約 -0.2 run

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.14 / 3 / 0 名 | 4.86 / 1 / 0 名 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 4.14 中段水準、核心 IL 0 名可全員投入；Ginn 近 3 場平均 2.3 IP/場 → Athletics 牛棚會在第 4-5 局接手，Phillies 後段牛棚 vs Athletics 相對脆弱牛棚是本場主要 leverage 點
- AWAY 牛棚：ERA 4.86 全聯盟後段、核心 IL 雖 0 但深度先天不足；Painter 若僅 5 IP 換下，Athletics 必須吃 4 局，後段被 Schwarber / Harper 二輪輪到 → 後段失分風險高

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🔴 HOME reverse platoon Δ +0.214（vs RHB OPS 0.930 > vs LHB OPS 0.716）— RHP 對非預期手別反而吃虧
- 🟠 AWAY TTO3 penalty：OPS Δ -0.037（TTO1 0.835 → TTO3 0.798），第三輪 OPS 微幅下降但 K% 從 25.2% 掉到 15.9%（Δ -9.3pp）（career fallback）
- 🟠 HOME chain breaks at #4-5：OPS 落差 0.274
- 🔴 AWAY chain breaks at #6-7：OPS 落差 0.524
  - reverse_platoon 因 Athletics 4/5 上段是 LHB，**實際放大效果有限**；TTO3 penalty 對 Ginn 而言意義不大（他通常撐不到第三輪）；HOME chain_break 是真實壓制（García/Bohm 連續冷打者）；AWAY chain_break 高度壓制 Athletics 上限

## 條件修正

- Park Factor: 104.0 → +0.20 run（Citizens Bank Park HR +16%，Schwarber 31 全壘打打者尤其受惠）
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：Painter peripheral tier 高於 ERA 表面（Solid Starter 體質），Ginn 近 3 場短局數 → 牛棚負荷不對等是主軸；非雙重賽

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.4 | -0.1（chain_break #4-5）| 5.3 |
| AWAY | 3.9 | -0.2（chain_break #6-7）+ 0.1（reverse_platoon 稀釋後）= -0.1 | 3.8 |
| Total | 9.3 | -0.2 | 9.1 |

## 整體判斷

- **方向（基本面）**：HOME（Phillies）
- **總分（基本面）**：9.1（adjusted；偏 over 中性區）
- **方向信心**：65-70%（依據：①Phillies 近 10 戰 8-2、+4 連勝、攻守俱優；②主場 + Painter peripheral 體質高於 ERA 顯示；③Schwarber / Harper vs RHP 雙 elite 對 Ginn K-BB% 7.3% 形成 mismatch；④Athletics 牛棚 ERA 4.86 在本場 leverage 落差最大）
- **風險**：
  1. **Painter 小樣本（IP 29）**：ERA 5.28 仍可能在本場真實兌現，若被 Kurtz / Soderstrom 開局打開，5 局 4-5 失分情境並非低機率
  2. **Athletics last7 BABIP .346**：略高於正常區間，部分擊球可能回歸（壓低 Athletics 期望）
  3. **Phillies chain_break #4-5**：García / Bohm 冷打者連續，1-3 棒上壘後續打不回來，導致 RS 卡在 4-5 區間
  4. **打線未公布**：projected 順序 PA 近似，正式打序公布後若 Bohm 移後段 / Marsh 進入則需上修 Phillies offense
