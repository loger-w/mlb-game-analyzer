## 投手對決

### Jake Bennett (HOME, LHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP-blend 3.88 拉抬），ERA-only tier 估 🔴 Elite（ERA 1.80）。**樣本僅 5.0 IP**，兩個 tier 都是噪音放大；xERA 10.24 vs ERA 1.80 → era_xera_delta -8.44（Flag 8 觸發）。
  - **判讀**：5 IP 樣本下 xERA 算法極度敏感於單一被擊球品質；**不下修 / 不上修預測**，視為「未驗證」。實質參考：velo 88.4 mph（LHP 偏慢）/ Whiff 11.8%（低）/ K-BB% 5.0%（極低）→ Stuff 端非 Ace 級。
- **Reverse platoon 信號**：vs LHB .200/.200/.200（5 BF）/ vs RHB .308/.400/.538（15 BF）— 兩側都未達 30 BF 門檻，**信號未 fire**，但帳面對 RHB 較吃虧。TB 上半棒（Caminero R / Díaz R）為 RHB → 帳面對位略不利於 Bennett。
- **對手打線威脅**：TB 整季 vs LHP = 🟢 Weak（season tier 與 matchup tier 一致），AWAY chain_break #4-5 OPS 落差 0.446（high）→ 1-3 棒（Caminero/Simpson/Aranda）能打但 4-5 棒（Díaz last7 .656 / Mullins last7 .325）斷層。Bennett 若能撐前兩輪到第 4 棒，串聯壓制空間大。

### Griffin Jax (AWAY, RHP, 31 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = ⚪ Below Average，ERA-only tier 也估 ⚪（ERA 5.14）→ tier 一致，era_xera_delta 僅 -0.73（無 Flag 8）。但 GS 2 / IP 14 屬小樣本。
  - **判讀**：tier 一致 + era≈xera → 噪音可控。**真正風險**：Jax 是牛棚→先發轉型期（matchup-factors §投手角色轉換 規則 3：回歸先發前 3 場降級一檔），現已是最低 tier 無可再降，但體力分配與球種展開（ST 28.0 / FF 25.7 / CH 18.8）尚未跑滿先發節奏 → IP 上限可能僅 4-5 局。
- **Reverse platoon 信號**：vs LHB .276/.382/.517（36 BF, 達門檻）/ vs RHB .190/.308/.333（26 BF, 未達門檻）— 信號未 fire 但帳面 OPS 落差 .258（嚴重反 platoon）。BOS 上半棒 Abreu (L) / Duran (L last7 .926) 是核心 LHB → **本場放大此風險**：兩名表現不錯的左打對位他被打爆機率高。
- **對手打線威脅**：BOS 整季 vs RHP = 🟢 Weak（matchup tier）vs season 🟡 Average → matchup 下修一檔，但 #1 Contreras (.781 vs RHP) / #2 Abreu (.811) 仍是穩定威脅；HOME chain_break #2-3 OPS 落差 0.319（high，#2 .848 → #3 .529）→ 上半三棒結束後串聯斷裂。Jax 若撐到第二輪，後段可控。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak
  - matchup tier 比 season 低一檔 → 本場 BOS 攻擊力**輕度下修**；但上修點：Jax 反 platoon + Abreu/Duran 為 LHB → 上半棒對位帳面有利，整體**走平**（matchup 下修 ↔ Jax 弱對 LHB 上修），不下調預期。
- **chain_break / heat_vs_babip 信號**：HOME chain_break #2-3 fired（high, OPS 0.319）
  - 影響：1-2 棒（Contreras/Abreu）若上壘，3 棒 Story (.529) 推進力差 → 大局形成需靠 #1-2 連線一波到底，**壓制大局上限**。

### AWAY — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - matchup 與 season tier 一致 → 本場 TB 攻擊力**維持低位**；Bennett 樣本太小無法當基準對位修正，敘事不再加減。
- **chain_break / heat_vs_babip 信號**：AWAY chain_break #4-5 fired（high, OPS 0.446）
  - 影響：1-3 棒（Caminero/Simpson/Aranda last7 1.060）能上壘，4-5 棒 Díaz (.656 last7) / Mullins (.325 last7) 清壘失靈 → 殘壘風險大。Bennett 若被 1-3 棒纏到，第二輪能否扛 Díaz/Mullins 兩個冷棒成為關鍵防線。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.3 / 7 / **2 名（高）** | 4.19 / 8 / **3 名（極高）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- **HOME 牛棚**：3.30 ERA 帳面好但 core IL ×2（Coulombe LHP setup + Slaten RHP setup）→ 高槓桿區雙缺。BOS 連勝 3（近 10 場 6-4）牛棚負擔已累，本場若 Bennett 5 IP 落地（IP 上限基於 5.0 IP 累計），中段（6-7 局）將進入薄弱 high-leverage 區，TB 上半棒能在末段擴大失分。
- **AWAY 牛棚**：4.19 ERA + core IL ×3（Uceta / M. Rodríguez / +1）→ **崩盤級**。TB 連勝 6（近 10 場 9-1）牛棚已被使用密集；Jax 若僅撐 4-5 IP，後段需 4 局以上接力，**牛棚崩盤型失分情境機率明顯升高**，這是本場最重要的單向不對稱風險。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=-8.44):
  - **5 IP 極小樣本下的算法噪音**為主因，混合部分結構性（velo 88.4 / whiff 11.8% / K-BB% 5.0% 都不像 1.80 ERA 該有的水平）。本場**不自動下修預測**，但敘事認定 Bennett 可能向 ERA 4.0+ 區間回歸，**場上若早段被擊出 1-2 個 hard hit 即視為回歸啟動**。

### 額外信號
- 🔴 HOME chain breaks at #2-3：OPS 落差 0.319（已於上方打線段判讀）
- 🔴 AWAY chain breaks at #4-5：OPS 落差 0.446（已於上方打線段判讀）
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（已於上方牛棚段判讀）
- 🔴 ⏳ AWAY 牛棚 core IL ×3：🔴🔴 極高（已於上方牛棚段判讀）
  - **疊加敘事**：Jax 短局數（先發轉型 GS 2）+ TB 牛棚崩盤級（3 core IL）→ **同向疊加**。教練若被迫 4 IP 換投，後段對 BOS 上半棒有利的對位（Abreu/Duran LHB last7 hot）即會 leverage 出來。

## 條件修正

- Park Factor: 104.0 → +0.20 run（兩側已含於 base）；Fenway HR -15% 微壓 HR 但 runs PF 才是公式錨，已涵蓋。
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：兩位先發都是小樣本（Bennett 5 IP / GS 1，Jax 14 IP / GS 2）+ 都是非典型先發背景 → 整體 IP 上限預期偏低（4-5.5 IP）→ 牛棚 leverage 提前進場，雙邊後段失分機率提高。**對 Total 偏多向**。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 6.5 | +0.5（AWAY core_il ×3 取 +0.6）−0.3（HOME chain_break high）+0.2（Jax reverse platoon 對 BOS LHB 核心放大）= **+0.5** | **7.0** |
| AWAY | 6.1 | +0.3（HOME core_il ×2 取 +0.3）−0.3（AWAY chain_break high）+0.0（Bennett 樣本 inconclusive，敘事處理不入欄）= **+0.0** | **6.1** |
| Total | 12.6 | +0.5 | **13.1** |

## 整體判斷

- **方向（基本面）**：**HOME (Boston Red Sox) 微傾**
- **總分（基本面）**：**13.1**（base 12.6 + 0.5；偏 Over 8.5 約 +4.6 run 容差）
- **方向信心**：**56%**（淨正信號集中於 BOS 端 — TB 牛棚崩盤級 + Jax 反 platoon 放大 + Jax 短局數轉型 — 但兩位先發都是極小樣本，本身分散性大）
- **風險**：
  1. **Bennett 5 IP 樣本噪音**：1.80 ERA 不可信，但 xFIP 3.88 也不該推到爆。若實況是「中段就被打出 hard contact」，HOME 防守端可能比預期差很多 → 修正方向反而是 TB 端上修（風險對稱）。
  2. **Jax 體力與球種展開未驗證**：GS 2 樣本太小，可能他這次撐到 6 IP 也可能 3 IP 就 KO。本場最大單向波動點。
  3. **雙邊打線 vs 對位手別都 🟢 Weak**：Total 13.1 已是上修後值，若兩邊先發剛好都是「Live ball + 守備兜得住」的場景，跌回 11-12 區間機率不低。
  4. **TB 牛棚崩盤級**疊加 Jax 短局數 = 後段失分爆發風險（單向偏 BOS / 偏 Over），是判斷中最 robust 的一條。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
