## 投手對決

### Kyle Harrison (HOME, LHP, 24 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = —（樣本僅 6 GS，xFIP-blend 不給數），ERA-only 推得 🔴 Elite Ace（ERA 2.12）但 xERA 3.36 顯示真實水平在 🟠 Strong ~ 🟡 Solid 邊界，gap ≈ -1.24 run。
  - **判讀**：不同意 ERA-only Elite Ace 標籤。低 hard_hit% 17.6 + barrel% 5.5 是結構性正面，但 BB% 偏高、vs LHB 反吃虧、單一球種依賴 58.6% → 真實水平更貼近 Strong Ace。**不自動下修預測**，但風險段需明示其 ERA 含 BABIP / 守備運氣成分。
- **Reverse platoon 信號**：🔴 fired（vs LHB OPS .826 > vs RHB OPS .542，Δ +0.284），vs LHB 樣本僅 31 BF — small sample，但訊號方向與 single-pitch FF 依賴吻合（FF 對 LHB 沒位移）。
  - **判讀**：本場 NYY 預期出戰 LHB 至少 3-4 人（Bellinger / Grisham / Chisholm，外加可能 Volpe/Wells 等候補），其中 Bellinger vs LHP .990 / Grisham vs LHP .672 — 兩位 LHB 打 LHP 不僅未顯弱、Bellinger 反而打更兇 → 嚴重放大此風險。Judge (RHB) vs LHP 1.072，但 RHB 對 Harrison 反而是「弱對抗」邊（vs RHB OPS .542）。
- **對手打線威脅**：🔴 高。Top 5 全員 vs LHP OPS ≥ .462，其中 Judge / Bellinger 雙巨砲 OPS > .990，加上 reverse platoon + 單一球種 + TTO3 penalty 三重放大 → Harrison 5 局內被攻陷機率明顯偏高，MIL 教練很可能 4-5 局就要走牛棚。

### Cam Schlittler (AWAY, RHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +1.1（< 15，不觸發 mismatch）。ERA 1.52 / xERA 2.71 雖有 1.19 run gap，但 xERA 仍 sub-3.00 → 即使回歸真實水平仍是 🟠 Strong Ace 起跳。
  - **判讀**：同意 Elite Ace。velo 95.4 / max 101.3 + balanced 三球種（FF / FC / SI 三球 RV/100 全正，FC +4.0 是 putaway 主武器）+ vs LHB / RHB 兩側都極壓 → 結構性真材實料。8 場 GS 樣本仍偏小是唯一保留，但近 3 場 ER/IP = 3/16.7（ERA 1.62）顯示無 regression 跡象。
- **Reverse platoon 信號**：未 fired。vs LHB / RHB 兩側 OPS 都低（.543 / .384），符合 RHP balanced arsenal 的常規 platoon。
- **對手打線威脅**：🟡 中。MIL 對 RHP 是 🟡 Average tier（season .715 OPS、xwOBA .328），熱度 ⚖️ Normal。Top 3 之中只有 Turang vs RHP 1.052 是真威脅，Contreras .792 中等，#3-5 棒 OPS 全跌至 .631-.903 區間且 chain_break #1-2（Δ .172）讓 Turang 的上壘難以串聯成大局 → Schlittler 預期至少壓制至 6 IP / 2 ER 內。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - **判讀**：matchup tier 與 season tier 一致（持平）→ 對 Schlittler 既無上修也無下修，對「Elite Ace RHP」對位 Average 打線。Turang vs RHP 1.052 是唯一真威脅，但被 chain_break #1-2 卡住串聯 → 期待單局 1-2 分為主，難打大局。
- **chain_break / heat_vs_babip 信號**：🟠 chain breaks at #1-2（Δ .172, medium）
  - **判讀**：Turang（OPS .928）→ Contreras（.756）落差中等，但更關鍵是 Contreras → Frelick（.756 → .636）這段繼續衰退，意味即使 Turang 上壘，#2-3 棒打回率不高 → 壓制 chain 上限約 -0.2 run。

### AWAY — season tier 🟠 Strong / heat 🔥 Hot
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟠 Strong
  - **判讀**：matchup tier 與 season tier 一致 + heat 🔥 Hot 但 last7 BABIP .333（< .350 未 fire heat_vs_babip）→ 熱度可信、無回歸警報。對 Harrison 這位 reverse-platoon LHP，AI 應隱性上修評估方向：Top 5 vs LHP OPS 加權平均 ≈ .788（Judge 1.072 拉高、Chisholm .462 拉低），核心 Judge / Bellinger 對 LHP 都比 season 兇 → 維持 Strong tier 但風險偏多打。
- **chain_break / heat_vs_babip 信號**：🔴 chain breaks at #6-7（Δ .541, high）
  - **判讀**：Top 5 中 Caballero（.720）之後跌入 Volpe / Wells / Stanton 等候補 PA 區，#6-7 棒可能是 .180 級 OPS。對 chain 連續性的影響：前 5 棒打完後若沒清空壘包，下半棒回到 1-3 棒前的「斷點」會直接結束 inning → 壓制本隊得分上限約 -0.3 run（high 取下界）。但前 5 棒對 Harrison 已具足夠 carry power，主要影響大局上限而非場均得分。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.6 / 5 / **2**（Zerpa, Koenig）| 3.18 / 4 / **0** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（MIL）：ERA 3.60 中等、core IL ×2（Zerpa + Koenig 兩位 LH setup 皆 IL15d）→ 🔴 高吃緊。對 Harrison TTO3 penalty fired 的劇本特別不利：教練可能 4-5 局就要下 Harrison，但後段 high-leverage LHP 全空，只能用低槓桿 RHP 對付 Judge / Bellinger 這種 OPS > .898 的核心 → 第 6-8 局是 NYY 的攻擊窗口。
- AWAY 牛棚（NYY）：ERA 3.18 全聯盟前段、core IL = 0 → 完整可用。Schlittler 本身 TTO3 強（.543 OPS）能撐至少 6 IP，後段交給完整 closer / setup 鎖局 → MIL 後段反撲機率被壓低，本場「越打到後面越偏 NYY」結構成立。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🔴 HOME reverse platoon Δ +0.284（vs LHB OPS 0.826 > vs RHB OPS 0.542）— LHP 對非預期手別反而吃虧
- 🟠 HOME single-pitch dependent：主球種使用率 58.6%（≥45.0%）
- 🟠 HOME TTO3 penalty：OPS Δ +0.057（TTO1 0.777 → TTO3 0.834），第三輪明顯衰退；K% 從 26.2% 掉到 17.7%（Δ -8.5pp）（career fallback）
- 🟠 AWAY TTO3 penalty：OPS Δ +0.055（TTO1 0.488 → TTO3 0.543），第三輪明顯衰退；K% 從 30.9% 掉到 16.2%（Δ -14.7pp）
- 🟠 HOME chain breaks at #1-2：OPS 落差 0.172
- 🔴 AWAY chain breaks at #6-7：OPS 落差 0.541
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - **判讀**：⏳ short half-life signal — 對手有 24h 調整空間，但兩位 IL15d 不會本場恢復、也無 trade deadline 補強空間 → 信號穩定。與 Harrison TTO3 penalty 形成「先發撐不久 + 牛棚補不上」的雙重壓力，將 NYY 中後段攻擊期望值上修約 +0.4 run（2 名 core IL 中段值）。同時與 Harrison ERA 含運氣（Flag 8 風險）形成連動：若 Harrison 早段 BABIP 回歸（被打強）→ 提早出局 → 牛棚壓力暴露更早。

## 條件修正

- Park Factor: 97.0 → -0.15 run（American Family Field 投手友善，但 HR PF +11% 對 Judge / Bellinger 等高 power 打者單側補償，PF 倍率已含）
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：雙方 SP 同列 🔴 Elite Ace tier（Schlittler 結構性 / Harrison ERA-only），對決級數對等。無 doubleheader。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (MIL) | 1.9 | -0.2（chain_break #1-2 medium） | 1.7 |
| AWAY (NYY) | 3.0 | +0.5（core_il +0.4 取 max + interaction +0.1，再扣 chain_break #6-7 -0.3）淨 +0.2；reverse_platoon / pitch_mix / tto3 同向 fire 不疊加 | 3.2 |
| Total | 4.9 | +0.0 ~ +0.1 | ≈ 4.9 ~ 5.0 |

## 整體判斷

- **方向（基本面）**：偏 **AWAY (NYY)**。雖然 base formula HOME 1.7 vs AWAY 3.2 差距不大，但 NYY 累計多重對 Harrison 的結構性優勢（reverse platoon + 單一球種 + TTO3 + MIL 牛棚 core IL ×2），加上自家 SP Schlittler 是真 Elite Ace 能 carry → 結構面明顯偏 NYY。
- **總分（基本面）**：**4.9 ~ 5.0**（兩位 Elite Ace tier 對決壓低總分上限，但 MIL 牛棚薄 + Harrison 半 ERA 含運氣為向上風險）
- **方向信心**：**60%**（不到 75% 因 G1 MIL 6-0 大勝顯示 NYY 打線可能有對 MIL 投手群的盲點 + 兩位 SP 樣本都偏小 6-8 場 + 打序未公布 projected 噪音）
- **風險**：
  1. **Harrison ERA 含運氣**（Flag 8 紀律）：ERA 2.12 vs xERA 3.36 = -1.24 run gap，僅 6 GS 樣本，BABIP / 守備偏好可能向均值回歸；若回歸發生在本場 → AWAY 上修空間大，反之 Harrison 仍能壓制 → AWAY 下修。
  2. **Bellinger last7 含運氣**：last7 OPS 1.516 + last7 BABIP .417（接近 heat_vs_babip 觸發門檻），可能回歸 → 若 Judge 不單獨 carry，NYY 前 5 棒攻擊力會打折。
  3. **Series momentum**：G1 MIL 6-0 大勝、MIL 連勝 +2 / NYY 連敗 -1 — 短期心理 / 打擊節奏可能延續到 G2，且 MIL 近 10 RA 僅 2.20（守備極好）對 NYY 攻擊面是隱性壓力。
  4. **打序 projected**：兩隊打線都是 PA 排序近似，若 MIL 把 Frelick 提到 #2 取代 Contreras（vs RHP 更佳）或 NYY 調動順序，chain_break 信號可能轉移 / 失效。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組