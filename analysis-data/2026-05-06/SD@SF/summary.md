## 投手對決

> ⚠️ **使用者指定 AWAY 先發為 Waldron, M，但 MLB Stats API 之 probable pitcher 為 Bradgley Rodriguez (#699134)**。Waldron 雖在 active roster 但今日非預定先發。本分析以 API 為準。

### Adrian Houser (HOME, RHP, 33 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟢 Back-end Starter（xFIP p43, K-BB% p15），gap vs ERA-only = +25.4 (high)
  - 同意 xFIP-blend 比 ERA-only 更接近真實水平：xFIP 4.23 < FIP 5.61 < ERA 7.12，落差來自 HR/FB 偏高（HR/9 1.78）。但即便採 xFIP，K-BB% 4.2、whiff 8.4、xwOBA .377 都偏弱，「Back-end」是合理上限非中位。**不自動下修，但結構面仍偏弱**。
- **Reverse platoon 信號**：未 fire（vs LHB 1.190 / vs RHB .602，正向 platoon 但極端 magnitude）
  - 不算 reverse，但 LHB 把 Houser 打爆等級（OBP .453 / SLG .737 in 87 BF）。SD 主打 RHB，Merrill (LHB #2) 是唯一手別優勢點，Devers/Lee/Eldridge 等 LHB 在 SF 端反而要面對 Rodriguez，無交集。
- **對手打線威脅**：SD 多 RHB（Tatis/Bogaerts/Machado/Laureano），對應 Houser vs RHB .602 OPS — Houser 在這個手別組合下有體面的可能。**但 Merrill 一棒落 LHB，加上 Houser GB% 62.2 + 球速跌至 90.8 avg，TTO2-3 一旦失準，Oracle Park 雖壓 HR (-17%) 仍擋不住串聯**。Houser 04-01 對同一支 SD 投出 5.3 IP / 1 ER 是基準參考。

### Bradgley Rodriguez (AWAY, RHP, 22 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = —（小樣本），ERA-only 顯示 🟠 Strong Ace（ERA 1.62）— 但**14 G / 0 GS / 16.67 IP，這是純 RP 數據**。
  - 依 `matchup-factors §投手角色轉換（牛棚 → 先發）`：不得直接以 RP ERA 評估先發，**回歸先發前 3 場降級一檔** → 真實預期 🟡 Solid 上限 / 🟢 Back-end 中位。先發體力分配、TTO2-3、4-5 球種展開都是首次承受。近 3 outings 僅 1.7~2.0 IP，今日預估上限 4-5 IP。
- **Reverse platoon 信號**：🔴 fired（Δ +0.222，vs RHB .654 OPS > vs LHB .432 OPS，CH 42.5% 主球種）
  - SF 預定 1-9 棒中 RHB 約 5-6 名（Ramos/Schmitt/Chapman/Adames/J.Rodriguez），其中 Schmitt（.914 OPS / vs RHP .897）是熱手。Rodriguez 改變球（CH-heavy）對 RHB 失去「左打通用 putaway」優勢，反而被 SF 中段火力點吃。**signal 在本場放大成立**。
- **對手打線威脅**：SF 整體 tier Weak / Cold，但 Schmitt + 1.166 OPS 的 J.Rodriguez（小樣本警告）是危險點；多數 RHB 對 reverse-platoon RHP 有結構性優勢。Rodriguez 真要把 SF 壓低於 base 2.4，需要其 RP-用的 100 mph 四縫線 + GB% 59.5 蟲爆球能延續到先發場景，**這是強烈未驗證假設**。

## 打線評級

### HOME — season tier 🟢 Weak / heat 🥶 Cold
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak — 與 season tier 一致，**同意維持 Weak 評級**。Rodriguez reverse-platoon 信號雖 fire（→ HOME 端 +0.1~0.3），但 SF 自身 chain_break #8-9（Δ 0.755）與 last7 BABIP .249（Flag 3）的壓力抵銷掉，淨效應接近 base。
- **chain_break / heat_vs_babip 信號**：
  - 🔴 chain_break #8-9 Δ 0.755（J.Rodriguez 1.166 [小樣本警告] → Bailey .411）：實質瓶頸落在 #9 Bailey、#7 Eldridge .453；後段難帶起串聯，#5-6 Devers/Adames（vs RHP .505/.650）也偏弱。**壓制 SF 大局攻擊上限**。
  - 🟠 ⏳ HOME unlucky-cold（last7 BABIP .249）：屬 Flag 3 — 可能反彈也可能持續。本場 Rodriguez GB% 59.5 + CH-heavy 是低反彈環境，**敘事中性，不入 ±run**。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak — 落差一級。**同意下修至 Weak/Average 之間**。Top 5 vs RHP 表現參差（Tatis .597、Machado .596、Bogaerts .788、Laureano .800、Merrill .652），無 elite 火力點；但 Houser vs RHB .602 OPS 雖然不差，FIP 5.61 + GB% 62.2 碰到 Tatis/Laureano 強拉打型（last7 EV95% 60.4 / 44.6）會出現高品質接觸。
- **chain_break / heat_vs_babip 信號**：
  - 🔴 chain_break #7-8 Δ 0.371：Top 5 後接續力下降，Houser 若能撐到 6 IP TTO3 會落在 SD 板凳區段，但 Houser TTO3 K% 從 19% 跌至 0%（signal 9 fire）反而難自己拿三振。
  - 🟠 ⏳ AWAY unlucky-cold（last7 BABIP .254）：對 Houser GB% 62.2 + Oracle Park HR -17% 環境，地球反彈須仰賴 SF 內野守備（紮實但非頂尖），**敘事偏「短期可能持續壓制」，不入 ±run**。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.21 / 8 / 4 (🔴🔴 極高) | 3.90 / 5 / 0 (健康) |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- **HOME 牛棚（SF）**：season ERA 3.21 看似中上，**但核心 IL ×4**（Erik Miller 15d、Hayden Birdsong 60d、Jason Foley 60d、Randy Rodríguez 60d 等高槓桿臂）— 對應 §牛棚累計效應「3+ 名 → 🔴🔴 極高 / 崩盤級」。可用 leverage 池只剩 Ryan Walker、Robbie Ray（先發 hybrid 用法?）、Tyler Mahle、JT Brubaker。**Houser TTO3 K% 跌到 0% + 牛棚薄** = 後段失分視窗最大化。SF 教練可能讓 Houser 撐久一點以保 leverage 臂，反而放大失分。
- **AWAY 牛棚（SD）**：season ERA 3.90，但**核心 IL = 0**（Mason Miller closer、Jason Adam setup、Jeremiah Estrada、Adrian Morejon 全可用）。Rodriguez 改打先發後 SD pen 流動性略受影響，但仍是聯盟中上。Rodriguez 預估 4-5 IP，後續 King 換工可走長中繼或正規 RP 接力。**SD 後段優勢明顯**。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.249)：
  - 中性偏「短期持續」。Rodriguez RP→GS 風險拉高 SF 接觸品質的不確定，但其 GB% 59.5 + CH 主球種對應 SF 過去一週低 BABIP 的成因（弱接觸/接殺率高）並未消失。**不影響方向判斷，但若 Rodriguez 早早被識破（球種無變化、4 IP 後 fastball 速度掉），BABIP 反彈可能集中在中後段**。
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.254)：
  - 中性偏「短期持續」。Houser 雖整體質地差，但本季 GB% 62.2、04-01 對 SD 出 5.3 IP / 1 ER / 6 H，弱接觸基礎打法在 Oracle Park 內野能延續。SD 上一場（5/5）打 Giants 出 10 分代表 SD 攻擊能爆，但那場是接力對手不同投手 — **本場 Houser/SF 守備組合的壓制環境延續概率仍高**。

### 額外信號
- 🟠 HOME TTO3 penalty：OPS Δ +-0.097（TTO1 1.073 → TTO3 0.976），第三輪明顯衰退；K% 從 19.0% 掉到 0.0%（Δ -19.0pp）
- 🔴 AWAY reverse platoon Δ +0.222（vs RHB OPS 0.654 > vs LHB OPS 0.432）— RHP 對非預期手別反而吃虧
- 🔴 HOME chain breaks at #8-9：OPS 落差 0.755
- 🔴 AWAY chain breaks at #7-8：OPS 落差 0.371
- 🔴 ⏳ HOME 牛棚 core IL ×4：🔴🔴 極高（牛棚崩盤級）
  - 直接放大 Houser TTO3 K% 失靈的後果：6-9 局 SD 火力可能進入 SF 第三、第四線中繼。**與 Houser tier_mismatch（Flag 8 紀律：不自動下修）無加倍風險，但與 tto3_penalty signal 同向 — 同側多 signal 取單側 max + 0.1，AWAY 攻擊 +0.5~0.8 區間（cap）**。

## 條件修正

- Park Factor: 91.0 → -0.45 run（Oracle Park 強投手球場，HR -17%）
- 天氣：Partly Cloudy, 59°F, wind 6 mph, Out To CF
  - 影響判讀：6 mph 順風到中外野屬「噪音」門檻（< 8 mph，可忽略）；59°F 在 50–60°F 「輕度利投」邊界。整體**風 + 溫度 + 球場三方一致壓得分**，符合 base 7.9 偏低設定的合理性。HR 被進一步壓制 → 雙方主要靠單打/二壘安打串聯，加大 chain_break signal 殺傷力。
- 先發 tier / doubleheader：非 doubleheader（系列賽 G2，G1 為 5/5）。先發落差大：Houser 結構偏弱已知 vs Rodriguez 是 RP→GS 首戰未知 — 本場最大不確定性集中在 Rodriguez 首啟可控局數與球種展開。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (SF) | 2.4 | +0.1（reverse_platoon medium，受 chain_break 內含部分抵銷）| 2.5 |
| AWAY (SD) | 5.5 | +0.5（tto3_penalty + core_il_count 同側 max 區間 + 0.1，cap 內）| 6.0 |
| Total | 7.9 | +0.6 | 8.5 |

> **±run 推導說明**：
> - HOME +0.1：Rodriguez reverse_platoon Δ +0.222 medium 級 → Table B +0.1~0.3，取下界 0.1 因 SF chain_break #8-9 與 last7 cold 同時壓低 ceiling。
> - AWAY +0.5：Houser tto3_penalty + SF core_il ×4 同側 fire，取 core_il (3+ 名 +0.4~0.8) max 區間下緣 + tto3 interaction 0.1 = +0.5（單側 cap 0.8 內）。
> - ⛔ 不入：Houser tier_mismatch（Flag 8/Table A）、雙方 BABIP（Flag 3/Table A）、Oracle PF（已含於 base）。

## 整體判斷

- **方向（基本面）**：**AWAY (SD)** 略偏向。Houser 結構面偏弱（FIP 5.61、K-BB% 4.2、vs LHB 1.190 OPS）+ SF 牛棚 core IL ×4 + Houser TTO3 K% 崩 = 後段 SD 失分視窗大。Rodriguez RP→GS 是反向不確定性，但 SD 牛棚健康可吃下後 4-5 IP。
- **總分（基本面）**：**8.5（adjusted）**，落在 Total 線上。Oracle Park HR -17% + 59°F + 微順風一致壓 HR、力推單打型得分；雙方 last7 BABIP 都偏低（Flag 3）但本場環境支持「持續低 BABIP」。
- **方向信心**：**SD 約 55–60%**。模型方向清楚但被 Rodriguez RP→GS 首戰、SD 自身打線 cold（vs RHP top 5 平均 OPS .654）兩個變數削弱信心，未達 > 65% 強信賴區間。
- **風險**：
  1. **Rodriguez 首啟成敗主導變異**：若球速速度站住 + CH 命中率高 → 4 IP 0~1 ER，AWAY 模型直接成立；若控球散 → 一輪過後 SF 中段 5 棒打回，方向反轉。
  2. **Houser 04-01 對 SD 已投出 5.3 IP / 1 ER**：同一支打線在過去一個月已示範可被 GB 球路壓制，AWAY 攻擊未必如 base 5.5 順利兌現。
  3. **SF 牛棚倘 Walker / Ray 滿狀態 + 比賽進入 high-leverage** → core IL 信號短期被吸收，AWAY 後段加成弱化。
  4. **薄盤環境**：開球前 3.8h，後續打線 / 雷諾換投公告可能改變 Tatis 健康度、Bailey 蹲與否、Eldridge 是否進等突發條件。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組