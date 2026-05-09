## 投手對決

### Jose Quintana (HOME, LHP, 37 📉📉📉 快速退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter，但結構性面表現遠遜：ERA 4.07 vs xERA **5.30**（gap −1.23，Flag 8 觸發）、FIP 6.14 / xFIP 5.39、K-BB% **0.9%**（聯盟底）。ERA 明顯被 BABIP / LOB% 灌水，真實水平接近 ⚪ Below Average。Flag 8 紀律下不自動下修 Quintana 對手得分，但敘事偏 Mets 對他應壓制不住的方向。近 3 場 9 ER/13 IP（6.23）已開始顯現 regression。
- **Reverse platoon 信號**：未 fire（vs LHB 25 BF / vs RHB 80 BF，未達 30 BF 門檻）。但 vs LHB SLG **.667**（25 BF）為極端 outlier — Mets 上半段 LHB（Bichette 等）若取得對位機會仍有發揮空間。
- **對手打線威脅**：Mets 今日對 LHP 整體 🟢 Weak（top 5 OPS .539-.760），但 Quintana 球速 86.0 mph、whiff% 9.4%、K-BB% 0.9% — 屬於「不三振、不保送、靠球質壓制」型；面對 Coors 廣闊外野壓力反而吃虧。Mets 不需強對位也能靠連續安打串聯。

### Christian Scott (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟢 Back-end Starter（GS=2，極小樣本）。ERA 4.26 vs xERA 4.67 gap 不顯著；但 FIP 4.68 / **xFIP 3.86** 顯示 K-BB% 13.8% 與 hard_hit% 20.0% 有結構性向上空間，AI 同意 tier 判斷偏保守。但 GS=2 不足做穩定推論。
- **Reverse platoon 信號**：未 fire（兩側 BF < 30）。vs LHB 13 BF / vs RHB 16 BF 都太小，本場 Rockies 打線 RHB 為主，預期回歸正常 platoon。
- **對手打線威脅**：Rockies 對 RHP 🟠 Strong + 主場熱（last7 OPS 多在 .850+）+ platoon advantage（top 5 中 4 人 vs RHP OPS ≥ season +0.050）。Scott **single-pitch dependent（FF 51.3%）** 是本場核心風險 — Rockies 打者第二輪後容易鎖定 FF 時機；career TTO3 OPS .851（36 BF，sample 太小但方向警告）配合 Coors 廣外野，後段失分風險顯著。

## 打線評級

### HOME (Rockies) — season tier 🟡 Average / heat 🔥 Hot
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong
  - matchup tier (Strong) > season tier (Average) — top 5 對 RHP 平均上修 ≈ +0.080 OPS，本場應上修攻擊評估。Moniak（vs RHP 1.199）/ Rumfield（.889）/ Johnston（.952）三人都是 Scott 直球時可能爆發點。
- **chain_break / heat_vs_babip 信號**：
  - 🔴 chain_break #2-3（Δ 0.335）：Moniak 1.199 → Freeman .769 落差大，若 Moniak 出局 Freeman 銜接弱，2-3 棒清壘效率受限。
  - 🟠 ⏳ heat_vs_babip lucky-hot（last7 BABIP 0.381）：見 §風險提示。

### AWAY (Mets) — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - matchup tier (Weak) < season tier (Average) — top 5 對 LHP 集體 underperform（Bichette .671 / Baty .583 / Alvarez .644 / Benge .539），僅 Semien .760 撐著。本場應下修 Mets 攻擊評估，但 Quintana 結構性差會部分抵消。
- **chain_break / heat_vs_babip 信號**：
  - 🔴 chain_break #7-8（Δ 0.385）：projected 後段（8-9 棒）OPS 落差大，但 projected 排序非實際打序，影響打折。Quintana 不擅 K，Mets 整列接觸型打法仍可串聯。

## 牛棚

| | HOME (Rockies) | AWAY (Mets) |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.39 / 4 / **0** | 3.82 / 7 / **3** |

> 牛棚 IL：Rockies 4 名（無核心高槓桿）；Mets 7 名含 A.J. Minter (IL15d)、Dedniel Núñez (IL60d) 與第三名核心 → 對應 §牛棚傷兵累計效應 **3+ 名 🔴🔴 極高（崩盤級）**。

### 牛棚影響判讀
- HOME 牛棚（Rockies）：ERA 4.39 季帳面普通但完整核心可用，主場 Coors 對任何牛棚都不友善 — Quintana 撐 4-5 局後 6-9 局 4-5 名中繼接力，可預期失分但不至於崩盤。
- AWAY 牛棚（Mets）：ERA 3.82 紙面良好但 **核心 3 人 IL**，後段缺乏 setup / closer 級高槓桿。Scott 預期 4-5 IP（GS 2 用球管控），第 6 局起 Mets 將在 **Coors 主場 + Rockies 熱打線 + 核心牛棚薄弱** 的三重壓力下取分。本場最大方向性風險。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.381):
  - last7 BABIP .381 + last7 OPS 多在 .85+（Moniak 1.240 / McCarthy 1.076 / Johnston .940），明顯有 hot streak 含運氣成分。**但**：(1) 主場 Coors 廣外野自然推升 BABIP；(2) 本場對手 Quintana 球速衰退 + xFIP 5.39 + 近 3 場 6.23 ERA，弱投未必觸發回歸。AI 判讀 **可能延續**，不自動下修 HOME 預期得分。

### 額外信號
- 🟠 AWAY single-pitch dependent（FF 51.3%）+ 🟠 HOME platoon advantage：兩信號同向放大 — Rockies top 5 platoon 優勢碰上 Scott 單一球種，第 2-3 輪打席後配球單一更易被鎖定。本場若 Scott 第 5 局後仍在投，是高風險時段。
- 🔴 HOME chain breaks at #2-3：Moniak (1.199) → Freeman (.769) 是清壘段斷點，限制大量出局時的爆發上限，但 Moniak 自身爆發力可單棒解決問題。
- 🔴 AWAY chain breaks at #7-8：projected 排序，影響打折。
- 🔴 ⏳ AWAY 牛棚 core IL ×3（🔴🔴 崩盤級）：本場最高權重風險，已於 §牛棚影響判讀 詳述。
- 🔴 打者友善球場 PF 131：formula 已涵蓋，敘事補充：本場條件（cool 63°F + wind 7 mph In From LF）會輕度抵消，但 Coors 主場效應仍主導。

## 條件修正

- Park Factor: 131.0 → +1.55 run（已含於 formula base，5 月 PF 已恢復 131）
- 天氣：Partly Cloudy, 63°F, wind 7 mph, In From LF
  - 影響判讀：63°F 處 60-85°F 中性區間下緣（輕度利投）；風 7 mph 為「噪音可忽略」門檻邊緣，但風向 In From LF 對 LHB 拉打 HR 有輕微壓制（Moniak、McCarthy 為 LHB）。整體 −0.1 ~ −0.2 run，輕度抵消 Coors PF。
- 先發 tier / doubleheader：非 doubleheader；Scott GS=2 用球量管控 → 預期 4-5 IP，提早交給牛棚為本場結構性事實，不是 ad-hoc 風險。

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (Rockies) | 6.8 | +0.6（platoon +0.2 / pitch_mix +0.2 / Mets core_il×3 +0.5 / chain_break −0.2 / weather −0.1，cap ±0.8 ✓） | **7.4** |
| AWAY (Mets) | 8.7 | −0.3（chain_break −0.2 / weather −0.1；Flag 8 Quintana 結構差 ⛔ 不入錨點，敘事 only） | **8.4** |
| Total | 15.5 | +0.3 | **15.8** |

## 整體判斷

- **方向（基本面）**：AWAY (Mets) 略佔優，差距 ≈ 1.0 run
- **總分（基本面）**：15.8（base 15.5 微上修，Coors 高總分結構主導）
- **方向信心**：**55%**（AWAY 偏多但不強）— Mets 站在較好的投打對決面（Scott peak vs Quintana decline + Mets 結構好），但 **核心牛棚 3 人 IL 在 Coors** 是顯著拉低方向信心的對沖；若 Scott 提早退場，Mets 後段 4 局可能被 Rockies 反咬。
- **風險**：
  1. **Mets 核心牛棚 ×3 IL 在 Coors** — 第 6 局起方向風險最大；若 Scott 4 IP 退場，Rockies 後段可掃 3-5 分推翻領先。
  2. **Scott GS=2 樣本極小** — TTO3 .851 + single-pitch FF 51.3% 在 Rockies 熱打線前可能比預期更脆弱。
  3. **Quintana 反向** — 37 歲快速退化 + xERA 5.30 + 近 3 場 6.23，下限可能比 ERA 4.07 更低，Mets 打線即使對 LHP 弱也可能多得 1-2 分。
  4. **Rockies last7 BABIP .381 lucky-hot** — Flag 3 不入錨點，但若回歸時點剛好落在本場（弱投時不易），HOME 7.4 估計偏高。
