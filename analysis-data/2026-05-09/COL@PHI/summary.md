## 投手對決

### Aaron Nola (HOME, RHP, 32 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p83），gap vs ERA-only = +54.0
  - **判讀**：gap 巨大但 Elite Ace 帽子過大。xFIP 3.19 / FIP 4.09 / xERA 4.42 vs ERA 5.06 → ERA 確實被 ~0.6-1.0 run BABIP/HR luck 拉高（barrel% 9.3 / Hard Hit% 25.1 都優異），結構面合理 tier 在 🟡 **Solid Starter** 區間（K-BB% 16.7、whiff% 10.4 拉不到 Strong Ace 門檻）。Flag 8 未觸發（|ERA-xERA| 0.64 < 1.5），但 tier_v2 信號提醒**不自動以 ERA 5.06 下修預測**。
- **Reverse platoon 信號**：未 fire（vs LHB .320/.402/.507 vs vs RHB .236/.257/.431，是 normal platoon 而非反向）
  - 但 vs LHB 數據確實慘，Rockies LHB 威脅見「對手打線威脅」段
- **對手打線威脅**：Rockies projected lineup top 5 含 2 LHB（Rumfield #1 .866 vs RHP, Johnston #5 .971 vs RHP）— 直接吃 Nola vs LHB 罩門；3 RHB（Goodman/Tovar/Karros）對 Nola 相對安全（.236/.257 對 RHB）。**TTO3 penalty 高度觸發**：第 3 輪 OPS 1.026（聯盟頂級打者水準），K% 從 27.8 跌到 24.3 — 教練可能 5-6 局換投，PHI 牛棚需早接手。

### Kyle Freeland (AWAY, LHP, 32 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = —（v2 無清楚分類，可能因 xFIP 落在中段 + 樣本邊界），gap vs ERA-only 未顯示
  - **判讀**：自行檢視 — ERA 5.04 / xERA 4.60 / FIP 4.22 / xFIP 3.23 → 與 Nola 結構類似，ERA 比 xFIP 高 ~1.8 run（含 Coors Field 主場噪音與 BABIP 偏移）；近 3 場 4 ER / 15.7 IP = 2.29 ERA 顯示**近期狀況偏好**。barrel% 5.3（excellent）/ Hard Hit% 25.7 / whiff% 11.0 均優於 ERA 反映。合理 tier 上修至 🟢 **Back-end** 偏 🟡 Solid 之間。
- **Reverse platoon 信號**：未 fire
- **對手打線威脅**：Phillies top 5 中 Schwarber/Turner/Harper 都是 LHB，且 vs LHP 全部退步（OPS Δ −0.145 / −0.179 / −0.157）— 對 LHP Freeland **明顯弱化**。RHB Adolis García vs LHP .813（強化）+ Bohm vs LHP .464（弱）。整體 Phillies vs LHP 是真 weak。Freeland career TTO3 OPS .863 vs TTO1 .798（Δ +0.065，未觸發）→ 比 Nola 更能撐第 3 輪。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - **方向**：**下修**。Top 3 LHB（Schwarber/Turner/Harper）vs LHP 全部明顯退步，且 Turner vs LHP .448 / Bohm .464 都是替補級數字。Phillies vs LHP 弱化是結構性（去年同樣偏弱），不是樣本噪音。
- **chain_break / heat_vs_babip 信號**：🟠 HOME chain breaks at #1-2 (.267)
  - **判讀**：Schwarber → Turner 上壘能力斷裂（.749 → .448 vs LHP），導致 Harper 站打席時前面常無人 — 直接壓制 Phillies 製造大局能力，火力依賴 solo HR（CBP 利 HR 略補償）。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong
  - **方向**：**上修**。Rumfield/Goodman/Johnston vs RHP 都比 season 好（.866 / .894 / .971），3/5 platoon-advantaged。Nola 對 LHB 慘（OPS .909 against），Rumfield + Johnston 是高威脅。
- **chain_break / heat_vs_babip 信號**：🔴 AWAY chain breaks at #2-3 (.349)
  - **判讀**：Goodman .894 → Tovar .475 是嚴重斷裂，#3 Tovar vs RHP .475 接近自動出局。即使 #1-2 上壘，#3 出局 → #4 Karros / #5 Johnston 的清壘機會被吃掉 → **壓制 Rockies 上限**，火力主要靠 #1-2 + #4-5 連續打才能爆發。L7 BABIP .341 偏高但未觸發 hot/cold（< .370 門檻），不足以判運氣紅利。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.13 / 3 / 0 | 4.43 / 4 / 0 |

### 牛棚影響判讀
- **HOME (Phillies) 牛棚**：ERA 4.13 約聯盟均；core_il = 0，可用度正常。但 Nola TTO3 penalty 高度觸發，預期 5-6 局後接手 — 如果出現「Nola 早退 + Rockies LHB 連線」場景，後段中繼壓力大。對 Rockies 末段攻擊算尚可抵抗（Rockies vs RP K% 27.5 偏高）。
- **AWAY (Rockies) 牛棚**：ERA 4.43 高於聯盟；近 30 RA 5.10 / 近 10 RA 5.90 顯示**整體狀況下滑**。core_il = 0 名義上完整但 ERA 數據說明牛棚有結構性弱點。Freeland 若能撐到 6 局已是上策 — 一旦交給牛棚，Phillies 主場 + CBP HR 場 + Schwarber/Harper 仍可能爆 solo。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🔴 HOME TTO3 penalty (high)：見「Nola 對手打線威脅」段判讀 — 推升 PHI 牛棚負擔，AWAY 後段得分機會 ↑（Table B 中 +0.2 ~ +0.3 / 場）
- 🟠 HOME chain_break #1-2：見「HOME 打線 chain_break 信號」段判讀 — 壓制 PHI 攻線連續性（Table B 中 −0.1 ~ −0.3 / 場）
- 🔴 AWAY chain_break #2-3 (high)：見「AWAY 打線 chain_break 信號」段判讀 — 壓制 COL 上限（Table B 中 −0.3 / 場）
  - **雙重交互**：HOME chain_break + AWAY chain_break 都是壓制本側攻擊；TTO3 反向利 AWAY → 取單側 max + interaction，AWAY 側 chain_break (−0.3) 與 TTO3 (+0.3) **近抵消** → 淨 0；HOME 側純 chain_break −0.2

## 條件修正

- Park Factor: 104.0 → +0.20 run（HR +16%，對 Schwarber/Harper/Goodman 有利）
- 天氣：未公布（跳過天氣分析）
- 先發 tier：Nola 與 Freeland 結構面接近（xFIP 3.19 vs 3.23），名義 ERA 差距是 noise；Nola 整體仍略佔 tier 優勢（K-BB% 16.7 vs 14.4，sample 與信譽更穩）
- doubleheader：不適用

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.8 | −0.2（chain_break #1-2） | 4.6 |
| AWAY | 4.5 | +0.3（TTO3 penalty）− 0.3（chain_break #2-3）= 0 | 4.5 |
| Total | 9.3 | −0.2 | 9.1 |

## 整體判斷

- **方向（基本面）**：**HOME 略優**
- **總分（基本面）**：**9.1**（接近 base 9.3，雙方信號近相互抵消）
- **方向信心**：**55%**（兩位先發 xFIP 都比 ERA 好 ~1.8 run，誰先被打開鏈子是關鍵；Phillies vs LHP 弱化是結構性、Rockies 牛棚近狀況差，這兩點微傾 HOME；G1 Phillies 已輸，反彈動機 +）
- **風險**：
  1. **Nola TTO3 penalty 真的觸發**（5-6 局換投）→ PHI 牛棚扛 3-4 局，Rockies 近 7 RS 5.20 + Rumfield/Johnston LHB 威脅 → 後段失分情境若實現，HOME 略優判讀會反轉
  2. **Freeland 客場樣本干擾**：career TTO 數據相對穩定但本季 ERA 5.04 含 Coors 噪音，做客 CBP 真實表現未知；近 3 場 sharp 是好訊號但樣本小（15.7 IP）
  3. **Citizens Bank Park HR +16%** 對 Schwarber/Harper 有利，但他們 vs LHP 數據差 → 即使 squared up 機會少；反而 Goodman/García vs Nola 的 barrel% 高（15.2 / 9.0）更可能利用球場
  4. 兩隊均處低迷季 / Rockies 1 勝後反彈 + Phillies 連敗修復 → single-game variance 高，95% CI 大概 6-13 分區間都合理

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
