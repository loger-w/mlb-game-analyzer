## 投手對決

### Trey Yesavage (HOME, RHP, 22 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = —（樣本不足 GS 2 拒給 v2 score），ERA-only 給 score 78.2
  - **不同意 Strong Ace 表面分級**。GS 僅 2 場（career），ERA 0.96 / FIP 2.14 / xFIP 4.22 三層落差大 — xFIP 4.22 對應 Back-end 水準，現有亮眼 ERA 主要是 BABIP / HR 抑制運氣。22 歲 prospect 第三次先發，敵方錄影量極少是 short-term edge，但 sample 紀律不允許定 Strong Ace。**真實水平估介於 Solid Starter（K-BB% 15.0、whiff 12.2、barrel 3.6 是 ace-like 訊號）與 Back-end（xFIP / velo 88.6 平均偏低）之間**。Flag 8 紀律：不自動下修預測，但風險段必提 regression upside。
- **Reverse platoon 信號**：未 fired
- **對手打線威脅**：Angels 1-9 棒 vs RHP OPS 中位偏弱（Schanuel/Adell/Moncada/Rivero/Lowe 五人 vs RHP < 0.770），但前段 Trout (vs RHP .955)、Soler (.803)、Neto (.713) 三人對 fastball-heavy RHP 仍具威脅。Yesavage 主球種 FF 54.5% 對 Trout 級別打者有風險，但 forkball (FS 36.5%) 是制動工具。整體威脅度 **中**：top 3 隨時造成 1-2 失分但中下段（vs RHP < 0.600 三人）chain 斷得很快。

### Jack Kochanowicz (AWAY, RHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟢 Back-end Starter（xFIP p44, K-BB% p19），gap vs ERA-only = -30.8
  - **同意 tier_v2 判定**。K-BB% 5.1 是聯盟末段（既不三振也壓不住保送），SI 主球種 RV/100 +2.1（被打），hard_hit% 28.4 偏高。ERA 3.05 主因近期 BABIP 控住與被 HR 偏少，**屬運氣偏好**而非結構性突破，xFIP 4.21 / FIP 3.80 才是真實水準。Flag 8 紀律：不自動下修，但敘事上預期回歸到 ERA 4.0+ 區間，對打者友善的 Rogers Centre HR +4% 進一步放大下行風險。
- **Reverse platoon 信號**：未 fired（vs LHB SLG .253 vs RHB SLG .263，方向一致無反向）
- **對手打線威脅**：Jays 1-9 棒 5 名左打 + 1 SH 對 sinker-baller RHP 站位佳，但 Kochanowicz vs LHB SLG .253 抑制力其實不差。重點威脅是 #3-4 棒 Vlad Jr (vs RHP .753)、Okamoto (vs RHP .819 + 🔥 last7 OPS 1.432)：兩人對 SI 有滾地球 / 平飛差勁的 punish 能力。整體威脅度 **中高**：Okamoto 熱度 + 低 K-BB% pitcher 容易讓 Jays 中段累積壘上跑者。

## 打線評級

### HOME — season tier 🟢 Weak / heat 🥶 Cold
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak
  - Matchup tier 與 season tier 一致為 Weak。**本場評估方向：略上修**。原因：Kochanowicz 是 Back-end + 低 K-BB% pitcher（不會把好球員壓住），且 Okamoto last7 1.432 OPS / 53.3% EV95% / 16.3% Barrel% 是隊內熱手 — 對 sinker-baller 有 launch angle 優勢。Vlad Jr (last7 .375 / BABIP .150) 為極端 unlucky-cold，遇到沒能力鎖死的對手很容易反彈。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #1-2 high (Springer .612 / Barger .279, 落差 0.333) — Barger 23 PA 樣本太小且季 OBP .174 是隊內最弱 #2 棒；意味 Vlad/Okamoto 上場常見「無人 / 1 出局空壘」起手。對總分壓制顯著但對 #3-4 棒 solo HR 機率不影響。heat_vs_babip unlucky-cold（last7 BABIP .210，全隊多人 BABIP < .200 last7）— 對 sinker-baller 反彈時機合理，敘事 +0.1~+0.2 但 Flag 3 紀律不入量化。

### AWAY — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak
  - Matchup tier 從 Average 降到 Weak。**本場評估方向：下修**。Yesavage 主 FF + FS 對位上 Angels #5-9 棒（Adell vs RHP .569、Moncada vs RHP .669 last7 .263、Rivero 13 PA 菜鳥、Lowe vs RHP .524 last7 .000）幾乎是 auto-out 區。實際攻擊集中在 Trout / Soler / Neto / Schanuel 四人。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #7-8 high (Grissom .785 / Rivero .462, 落差 0.323) — Rivero 季 13 PA 菜鳥捕手，Yesavage forkball 對菜鳥滾出三振容易，#7 之後 chain 直接斷。heat_vs_babip unlucky-cold (last7 BABIP .251) 程度比 Jays 輕，且 Trout last7 BABIP .538 / Schanuel .391 已經反彈中，主要是 #6-9 棒在拖。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.15 / 7 / 1（Yimi García IL60d，Setup 等級） | 5.14 / 5 / 0 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 4.15 中段水準，但 Yimi García IL60d 屬 Setup 級長傷 → 對應 §牛棚傷兵累計效應 1 名核心 = 🟠 中高，後段（7-8 局）對 Trout-Soler 段一旦逼出 high-leverage 情境會吃緊。前一晚 G1 完封贏 2-0，牛棚消耗低（投手只有先發吃完局 + 必要時收尾），可用度 OK。
- AWAY 牛棚：ERA 5.14 為近期聯盟後段，但 core IL 0 名 — 沒有名單級警訊，是 quality 問題而非 availability 問題。此值代表整體後段失分壓力本就偏大；遇到對手主場、4-5 局 Kochanowicz 大概率被換下後，6-8 局 Jays 攻擊機會 ↑。最近 10 場 RA 4.10 比 30 天 4.70 改善，但仍是隊內最大破口。

## 風險提示

- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.21):
  - **可能反彈**。多人同步偏低（Vlad .150 / Springer .250 / Sanchez .222 / Gimenez .200）— 全隊性偏移很少持續超過 10-14 天。但「對 Kochanowicz 反彈」與「下一場才反彈」是兩回事，今晚不保證。Flag 3 紀律：不自動 +run，敘事上代表 TOR 進攻有額外 upside（保守 +0.0~+0.2，已含於上方 matchup tier 略上修敘事）。
- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.251):
  - **部分反彈中**。隊內 Trout/Schanuel 已 last7 BABIP > .390（反彈完成），剩 #6-9 拖累；對 Yesavage 樣本不足，#5-9 vs RHP 結構性弱才是主因。不自動 +run，本場 BABIP 反彈空間較有限。

### 額外信號
- 🟠 HOME single-pitch dependent：主球種使用率 54.5%（≥45.0%）
  - 對 Angels 影響有限：top 3 (Trout / Soler / Neto) 對 fastball-timing 有威脅，但 #5-9 對 forkball 制動容易吃三振。AI 量級：+0.1（取下界，因 Yesavage K-BB% 15% 高有自我緩衝）
- 🟠 AWAY TTO3 penalty：OPS Δ +-0.169（TTO1 0.534 → TTO3 0.365），第三輪明顯衰退；K% 從 22.2% 掉到 16.7%（Δ -5.5pp）
  - **方向矛盾的 signal**：K% 下降（投手第三輪三振率掉）符合 penalty 方向，但 OPS 反而下降（被打更弱）。GS 7 sample 過小，TTO3 樣本估約 30 BF，雜訊主導。實務上意義是「Kochanowicz 第三輪不再三振，靠軟弱接觸」— Jays 教練看到要早換對位。AI 量級：+0.05（保守，sample 紀律下不取 medium 區間）
- 🔴 HOME chain breaks at #1-2：OPS 落差 0.333
  - Springer-Barger 落差大，#1-2 棒上壘不一致 → Vlad / Okamoto 攻擊以 solo 為主。對中段 chain 影響顯著，已含 -0.3 量級。
- 🔴 AWAY chain breaks at #7-8：OPS 落差 0.323
  - Grissom-Rivero 落差，#7-8 之後 inning 容易 1-2-3。對 chain 完整性 -0.3。
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - Yimi García 是 Setup 級長傷，對 Angels 後段（特別 Trout-Soler 二次上壘）有放大效果。AI 量級：+0.1（單名核心 IL，medium）。**short half-life ⏳**：對手已知此狀況，今天進場前可能有戰術調整（早段消耗 Yesavage / 把 Trout-Soler 推到 7-8 局）。

## 條件修正

- Park Factor: 99.0 → -0.05 run（中性，HR +4% 略利攻 HR 不利安打）
- 天氣：室內（Roof Closed，不適用）
- 先發 tier / doubleheader：Yesavage GS 2 極小樣本是本場最大不確定性 — 任何「投得像 ERA 0.96」與「投得像 xFIP 4.22」對總分都是 ±1.5 run 級的擺盪。Kochanowicz 沒有意外（Back-end Starter 已是穩定基準），偏離的話只會更差不會更好。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.9 | +0.05（TTO3 K% drop, 保守）− 0.3（chain_break #1-2 high） | 3.65 |
| AWAY | 2.3 | +0.1（pitch_mix_concentration, 取下界）+ 0.1（core_il_count ×1）取單側 max 規則 → +0.2 − 0.3（chain_break #7-8 high） | 2.2 |
| Total | 6.2 | — | **5.85** |

## 整體判斷

- **方向（基本面）**：HOME（Toronto Blue Jays）
- **總分（基本面）**：~5.9（5.6-6.2 區間）— 兩隊打線冷 + Yesavage 表面數據強 + Kochanowicz 並不算被打爆型 sinker-baller 共同壓低總分上限
- **方向信心**：58%（中等偏低 — 主場優勢 + Kochanowicz Back-end + Angels 牛棚差三項成立，但被 Yesavage GS 2 sample 風險抵消約 8-10pp）
- **風險**：
  1. **Yesavage regression risk**（最大反向因子）：xFIP 4.22 與 ERA 0.96 落差暗示真實水平距離 0.96 ERA 很遠。若 Angels 把 Trout / Soler / Neto 的長打打出來，短局數失 3-4 分劇本完全可能（GS 2 沒有可靠 floor）。
  2. **Okamoto last7 OPS 1.432 是隊內熱手**：對 Kochanowicz sinker-baller 滾地導向有 launch angle 優勢，single HR 即可改變總分敘事，但這是上修 HOME 不是反向。
  3. **TOR last7 全隊 BABIP .210 隨時反彈**：Vlad / Springer / Sanchez / Gimenez 多人同步 unlucky-cold，反彈節點若落在今晚會打破預期低分。
  4. **Kochanowicz TTO3 K% drop + Angels 牛棚 ERA 5.14**：Kochanowicz 大概率 5 局內被換下，Angels 牛棚 6-9 局是 Jays 累積分數的 prime window。
