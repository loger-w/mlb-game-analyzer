## 投手對決

### Eric Lauer (HOME, LHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = ⚪ Below Average（xFIP p5, K-BB% p23），gap vs ERA-only = +14.3
  - 同意 ⚪ Below Average — ERA 6.69 / xERA 5.84 / FIP 6.93 / xFIP 5.07 / K-BB% 6.1% 全面偏弱，velo 86.1 mph 也屬偏低段；gap +14.3 略低於 15 觸發閾值，屬結構性弱化（年齡 30 起初期退化 + velo 低）而非運氣偏差。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 未觸發 reverse_platoon（vs LHB BF 僅 28 < 30 樣本門檻），但需注意 vs LHB 實際 OPS .812 反高於 vs RHB .837 並不明顯（一般 LHP vs LHB 應有優勢），可視為「LHP 對 LHB 護身符失靈」的隱性風險；惟 AWAY 打線左打主力少（核心五人多右打），實質衝擊有限。
- **對手打線威脅**：AWAY (🟡 Average vs LHP) 整體威脅中等偏弱，last7 🥶 Cold 且核心 OPS 多在 .700 以下（Tovar .578、Karros .586 較弱），Goodman / Johnston vs LHP 反而走 split 弱化（.728 / .305），對 Lauer fastball/changeup 主軸壓力可控；但 TTO3 penalty +0.111 fire → Lauer 第三輪必崩，配上 single-pitch 45.5% FF 易被讀，第二輪後段就要提早接力。

### Kyle Freeland (AWAY, LHP, 33 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p82, K-BB% p59），gap vs ERA-only = +57.8
  - gap +57.8 屬 high 級 tier_mismatch（ERA 7.04 vs xERA 5.73 缺口 1.31 接近 Flag 8 閾值 1.5；FIP 5.42 vs xFIP 3.62 缺口更大 1.80）。Tier_v2 引用 xFIP / K-BB%，給出 🟠 Strong Ace 偏樂觀；近 3 場 4 ER/15.7 IP（ERA 2.29）支持 xFIP 邏輯。判讀為「ERA 含運氣偏差（HR/FB 偏高 + Coors 樣本污染）」與「結構性退化（年齡 33 + velo 86.5）」並存，**不自動下修預測**，但給投手評等實質應取兩者中間（Solid 偏弱），不採完整 Strong Ace 水平。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - 未觸發 reverse_platoon。Freeland vs LHB OPS .915 / vs RHB OPS .926（皆偏高），手別差不顯著且兩側都被打；對 HOME 打線（左右混編、Ohtani/Freeman 左打主力）影響中性。
- **對手打線威脅**：HOME (🟠 Strong vs LHP) tier 高、xwOBA .360 / OPS .795，Ohtani last7 OPS 1.140、Freeman last7 OPS 1.278 均處 BABIP 偏高（.400 / .500）的爆熱期；Tucker vs LHP OPS .958 是天然 LHP killer。Freeland fastball RV -3.6 / curveball RV -4.0 主力球種皆負分，core 5 棒打者對 LHP 平均 OPS 高，威脅 🔴 高。

## 打線評級

### HOME — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟡 Average
  - matchup tier 比 season tier 低一檔，反映 HOME 整體 vs LHP 表現略遜於 vs RHP，但本場對手 Freeland 為 below-average LHP 且兩側 OPS 都偏高，**實際對位可上修回 🟠 Strong**——核心打者 Ohtani / Freeman 為左打 LHP killer（vs LHP OPS .865 / .773 仍佳），Tucker vs LHP .958 更是優勢明顯，整體看 chain 連續性無虞。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break fired at #6-7（OPS 落差 0.218）— 影響打線後段串聯，前 5 棒火力集中但 6 棒之後接續弱化，多輪打席優勢被中斷。heat_vs_babip 未觸發（heat ⚖️ Normal，整體不受 BABIP 極端值干擾），但 Ohtani last7 BABIP .400 / Freeman .500 個別屬可能回歸區間（敘事提示，不入 ±run）。

### AWAY — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - matchup tier 比 season tier 低一檔，且本場 Freeland 雖 ERA 高但 xFIP 屬中上，下修方向成立 — Rumfield vs LHP OPS .531、Johnston vs LHP OPS .305、Karros vs LHP OPS .633，前 5 棒有 3 人 vs LHP 明顯走弱，**下修確認 🟢 Weak**。Lauer 雖弱但 vs LHB 樣本小（28 BF）難給優勢。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break fired at #4-5（OPS 落差 0.243）— 打線中段 cleanup 區段斷裂，Goodman / Tovar / Johnston 後 Karros 接續 .586 OPS 嚴重壓制清壘能力。heat_vs_babip 未觸發 strict（last7 BABIP .271 略高於 .260 門檻），但 🥶 Cold + 低 BABIP 屬「冷期含部分運氣可能微反彈」敘事；惟 cold 同時遇 LHP weak split，反彈幅度受限。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 2.87 / 12 / 3（🔴🔴 極高） | 4.35 / 8 / 2（🔴 高） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：基準 ERA 2.87 屬聯盟前段，**但 core IL ×3（Brock Stewart / Edwin Díaz / + 1）已達崩盤級 🔴🔴**。Closer + 多名 high-leverage 同時缺陣等於沒有真正的 9th-inning lock-down，剩餘可用名單需大量 low-leverage 補位高槓桿輪次。對 AWAY 末段威脅顯著放大（+run 對手得分），尤其 5-7 局換投空窗最易被打穿。+0.4~0.6 run / 場（取 3+ 名級下半段，因基底 ERA 仍前段補償）。
- AWAY 牛棚：ERA 4.35 屬中後段，core IL ×2（Herget / Vodnik）屬 🔴 高吃緊級。HOME 打線中段 3-5 棒 vs RHP 火力強，後援補位品質差時末段保護不住。對 HOME 末段威脅明顯（+0.2~0.4 run / 場）。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🟠 HOME single-pitch dependent：主球種使用率 45.5%（≥45.0%）
- 🟠 HOME TTO3 penalty：OPS Δ +0.111（TTO1 0.704 → TTO3 0.815），第三輪明顯衰退；K% 從 25.9% 掉到 14.2%（Δ -11.7pp）（career fallback）
- 🟠 HOME chain breaks at #6-7：OPS 落差 0.218
- 🟠 AWAY chain breaks at #4-5：OPS 落差 0.243
- 🔴 ⏳ HOME 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
- 🔴 ⏳ AWAY 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - 兩隊牛棚均吃緊但 HOME 級數更高；本場兩位先發都屬偏弱 LHP（HOME ⚪ / AWAY tier-mismatch），都難撐 6 局以上 → 牛棚承擔局數高，IL 信號實質影響放大。HOME single-pitch + TTO3 penalty 雙重 fire 進一步推升提早換投機率，AWAY 後段攻擊極可能受惠（+run 對 AWAY）。⏳ 屬 short half-life，但今日無調整窗口可信引用。Flag 3/8 雙重壓力未硬觸發，主要壓力源仍是 IL ×3 + tier_mismatch + TTO3 三重結構，總分判讀偏多方向確立。

## 條件修正

- Park Factor: 98.0 → -0.10 run
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：非 doubleheader 場（系列賽 G2，G1 為 05-25 已賽）。先發 tier — HOME Lauer ⚪ Below Average，AWAY Freeland 名目 🟠 Strong Ace 但 ERA-xERA 缺口大實質取 🟡 偏弱 Solid；對位上 AWAY 投手實質強於 HOME，但 HOME 打線實質強於 AWAY，兩端抵銷後 HOME 攻擊端優勢仍勝。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 6.5 | +0.1 | 6.6 |
| AWAY | 7.2 | +0.4 | 7.6 |
| Total | 13.7 | +0.5 | 14.2 |

## 整體判斷

- **方向（基本面）**：AWAY（formula base 與信號修正後 AWAY 7.6 > HOME 6.6，gap 1.0 run）
- **總分（基本面）**：14.2（base 13.7 + 信號 +0.5）— 屬高分區間（UNIQLO Dodger Stadium HR +21% + 雙弱投 + HOME 牛棚 IL ×3 + AWAY 牛棚 IL ×2 同時崩盤）
- **方向信心**：52%（低信心 AWAY 方向）— 公式給 AWAY 高分但與球隊基本面實力（HOME 34-20 / AWAY 20-37、HOME 攻擊 🟠 Strong / AWAY 攻擊 🟢 Weak、近 10 場 +3 vs -3）方向相反；formula 結果主要由 Lauer ERA 6.69 推升 AWAY 期望分數，是 ERA-driven 而非綜合實力。實質可能更接近持平或 HOME 略勝。
- **風險**：
  1. **base formula 與基本面落差大** — AWAY base 7.2 主要由 Lauer ERA 6.69 推升；COL 打線 🟢 Weak vs LHP + 🥶 Cold + chain_break #4-5，實際得分上限可能受壓制至 5-6 分區，使 formula 高估 AWAY 約 1-1.5 run。
  2. **HOME 牛棚崩盤級風險** — core IL ×3（含 Closer Edwin Díaz），即使 Dodgers 攻擊強仍可能讓出末段；若領先 1-2 分仍有翻盤窗口。
  3. **Freeland tier_mismatch 不確定** — xFIP 3.62 vs ERA 7.04 缺口 +57.8 分高度模糊，可能本場走 xFIP 預期（壓制 LAD）也可能延續 ERA 慘況（送分），單場波動最高。
  4. **總分偏高側確信度高於方向** — 雙投均偏弱 + 雙牛棚崩盤 + Dodger Stadium HR friendly，總分上行（≥ 14）信心 ~70%，高於方向信心。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組