## 投手對決

### Jack Leiter (HOME, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p89），gap vs ERA-only = +68.0
  - **不完全同意 Elite Ace 標籤**。+68 大 gap 主要由 xFIP 3.03 撐起（HR/FB 被正規化），但 K-BB% 18.3% 落 Solid Starter 區間、whiff% 13.7 偏低、ERA 5.45 與 FIP 4.26 都顯示實戰結果差。較合理判讀：**「Solid Starter ＋ 偏差幸運值」** — ERA 確實高估其差，但「Elite Ace」也高估了壓制力。屬「ERA 低估真實水平」典型 case，但要回歸到 mid-tier 不是 ace。
- **Reverse platoon 信號**：dossier 未列入 fired 信號；不適用
- **對手打線威脅**：Cubs 整體 OPS .789 / xwOBA .344（🟠 Strong），但 vs RHP 矩陣偏弱（top 5 vs RHP 平均 ~.745；Bregman .658、Hoerner .711、Crow-Armstrong .695 都低於 season 水準）。Happ vs RHP .956 是首要威脅。Leiter FF 是唯一正 RV 球種（+1.5/100, 41% 使用），CH/SL 都在負 RV — 若 FF 命中率高、Cubs 多打不上去；若 FF 控不住，Happ / Busch 中段就能爆。

### Edward Cabrera (AWAY, RHP, 28 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p85, K-BB% p70），gap vs ERA-only = +8.7
  - **同意 Strong / Solid 之間的判讀**。ERA 3.27 / xERA 4.21（−0.94 落差，未達 Flag 8 ±1.5 門檻但偏 luck-supported）；K-BB% 13.4% 偏低（保送 BB% 高）才是 xFIP 3.55 vs ERA 3.27 落差小的真因。近 3 場 ER/IP = 3/16.7（1.62）有熱度但樣本小。實戰判讀：**Strong Ace 上緣，但 BB 控管是隱憂**。
- **Reverse platoon 信號**：dossier 未列入 fired；不適用（vs LHB .259/.340/.395 vs RHB .250/.289/.375 落差在常規 RHP 範圍內）
- **對手打線威脅**：Rangers 季 OPS .701 / xwOBA .315（🟠 Average）但近 7 天 BABIP 0.242（Flag 3 unlucky-cold）。top 5 vs RHP 矩陣不錯（Seager .815, Nimmo .816, Jung .907），但 last7 OPS 全線 < .700（Burger .547, Carter .590, Jung .590），季水準與近期狀態落差大。Cabrera CH 是 plus pitch（+1.7 RV/100, 33% 使用），可能壓制 Rangers 中軸；SI 被打硬（hard_hit 38.9%）但 whiff 僅 3.2%，是 contact-induce 球路。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - Matchup tier 與 season tier 持平 — **同意**。Rangers 季 OPS .701 落 Average 區段，vs RHP 沒明顯放大或壓縮。Seager / Nimmo / Jung 季 vs RHP 帳面好但 last7 全冷，**短期狀態壓過手別優勢**。
- **chain_break / heat_vs_babip 信號**：HOME chain_break #7-8 落差 0.237（🟠 medium）
  - 影響本場攻擊 chain：1-5 棒厚（Seager/Nimmo/Burger/Jung/Carter）+ 6-9 棒空（McCutchen / 替補 / 二線野手）。**Cabrera 若能撐到第 7-9 棒，幾乎能空轉一輪**；上半段失分後，下半段不易延續。Flag 3（last7 BABIP .242）narrative 上偏向「可能向上回歸」但不自動加 run。

### AWAY — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - Matchup tier 比 season tier 低一檔 — **同意下修方向**。Cubs 季 OPS .789 是 Strong，但 top 5 vs RHP 矩陣裡有 Bregman .658 / Hoerner .711 / Crow-Armstrong .695 三人低於季水準，整體 vs RHP 拖到 Average。Happ .956 vs RHP 是亮點，但單點不足以撐起 chain。
- **chain_break / heat_vs_babip 信號**：AWAY chain_break #7-8 落差 0.186（🟠 medium 偏低）
  - Cubs 1-5 棒厚（Bregman / Hoerner / Happ / Busch / PCA），#6-9 包 Suzuki / Swanson / Shaw / Amaya 也不算空，落差比 Rangers 小。Happ / Busch / PCA last7 都在 1.000+ OPS 但 BABIP 偏高（Happ .429, Busch .444）→ **lucky-hot 邊緣**，不自動加 run，但 narrative 上要警惕短期回歸。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 2.93 / 6 / 0 | 3.83 / 9 / 4 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- **HOME（Rangers）牛棚**：ERA 2.93 為聯盟前段，核心 IL 0 名 → **完整可用**。Chris Martin（IL15d）雖是 setup 級但腳本算 0，整體深度仍佳。Leiter 若被換 5-6 局後，Rangers 能拿出多個高品質中後段選擇給 Cubs 末段製造壓力。
- **AWAY（Cubs）牛棚**：ERA 3.83 中段，核心 IL ×4（Hunter Harvey、Caleb Thielbar、Porter Hodge、Riley Martin / + Matthew Boyd 等）→ **🔴🔴 極高（牛棚崩盤級）**。Closer 級 + 多名 HL RP 一次性缺陣，Cabrera 若提早被換（TTO3 Δ +0.245），Daniel Palencia / Phil Maton / Jacob Webb / Hoby Milner 等替補品質參差，後段 4-7 局是 Cubs 最大破口。**這是本場最大的單一風險點**。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.242):
  - 顯著低於 .260 門檻，**unlucky-cold** 邊緣。Rangers top 5 vs RHP 季 OPS 都在 .700+ 但近 7 天全 < .700 — 屬球運偏差累積，不是 talent 滑落。判讀：**今晚回歸機率不低，不自動加 run，但若 Cabrera BB 控管失常 → Rangers 早段有機會反彈**。Table A ⛔ 紀律不入「+ 信號」欄。

### 額外信號
- 🔴 HOME TTO3 penalty：Leiter OPS Δ +0.222（TTO1 0.535 → TTO3 0.757），第三輪明顯衰退；K% 從 32.4% 掉到 26.3%（Δ -6.1pp）
- 🔴 AWAY TTO3 penalty：Cabrera OPS Δ +0.245（TTO1 0.610 → TTO3 0.855），第三輪明顯衰退
- 🟠 HOME chain breaks at #7-8：Rangers 1-5 棒 vs 6-9 棒 OPS 落差 0.237
- 🟠 AWAY chain breaks at #7-8：Cubs 1-5 棒 vs 6-9 棒 OPS 落差 0.186
- 🔴 ⏳ AWAY 牛棚 core IL ×4：🔴🔴 極高（牛棚崩盤級）
  - **本場受此信號高度影響**：與 AWAY tto3_penalty（Cabrera 第三輪衰退 +0.245 = high）形成雙重壓力 — Cabrera 多半第 5-6 局被換下，後段交給 Daniel Palencia / Phil Maton / Hoby Milner 一線中繼，但無 Harvey / Hodge / Thielbar 撐 7-8 局橋段。**Rangers 末段反撲視窗放大**，若早段沒被 Cabrera 壓死，後 4 局得分上限提高。⏳ 半衰期 short — 對手有空間調整 mix，但今晚就在打、來不及。

## 條件修正

- Park Factor: 96.0 → -0.20 run（Globe Life Field 略偏投手友善，HR PF +6%）
- 天氣：未公布（Globe Life 是可開合屋頂球場，多半關閉 → 通常忽略）
- 先發 tier / doubleheader：非 doubleheader；Leiter（高 ERA-xERA gap）vs Cabrera（小 gap luck-supported）— 先發 tier 略偏 Cabrera 但不到一檔差距

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.0 | +0.4 | 4.4 |
| AWAY | 4.8 | +0.2 | 5.0 |
| Total | 8.8 | +0.6 | 9.4 |

**HOME +0.4 拆解**：
- AWAY tto3_penalty（high）+ AWAY core_il_count（×4 極高）同向（皆對 Cubs 投手後段壓力）→ interaction：取單側 max + 0.1 = +0.5（core_il_count 上界 0.8 但保守因 Cabrera 仍能撐 5-6 局）+ 0.1 = +0.6
- HOME chain_break（medium 0.237）→ −0.2（Rangers #7-9 弱壓制 chain）
- 淨 +0.4

**AWAY +0.2 拆解**：
- HOME tto3_penalty（high 0.222）→ +0.3（Leiter 第三輪 OPS 上升 + K% 掉 → Cubs 中後段機會）
- AWAY chain_break（medium-low 0.186）→ −0.1（Cubs #7-9 略弱）
- 淨 +0.2

## 整體判斷

- **方向（基本面）**：AWAY（Cubs）小幅領先
- **總分（基本面）**：~9.4（base 8.8 + 信號 +0.6）
- **方向信心**：~58%
- **風險**：
  1. **Leiter 真實水平不確定**：tier_v2 +68 gap 把他從 ERA-only Below Average 拉到 Elite Ace，但 K-BB% / whiff% 都不到 ace 級，FF 是唯一正 RV 球種。若 FF 控得好，Cubs 偏弱 vs RHP 的 top 5（Bregman/Hoerner/PCA）很容易被壓死，整場可能變低分纏鬥；若 FF 失準，Happ/Busch 一輪打回 5 分也可能。本場 outcome 對 Leiter FF 命中率敏感度極高。
  2. **Cubs 牛棚崩盤級（core IL ×4）+ Cabrera TTO3 penalty +0.245**：雙重壓力意味 Rangers 中後段反撲視窗大。如果 Rangers 跟著 last7 BABIP .242 unlucky 回歸（Flag 3）+ vs RHP 季水準（top 5 OPS 都 .700+），9.4 是保守估計，總分突破 10 分機率不低。
  3. **Cubs last7 lucky-hot 邊緣**（Happ BABIP .429, Busch .444）：與 Rangers Flag 3 反向 — 若同時回歸，Cubs 攻擊 / Rangers 守備兩端壓力都會緩解，總分壓低。
  4. **打線皆 projected**：開賽前 2-4 小時官方打序若大幅變動（特別 Rangers 是否啟用 Pederson / Higashioka 替換 Carter / Jung），影響 vs RHP top 5 矩陣。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
