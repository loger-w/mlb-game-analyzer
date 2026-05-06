## 投手對決

### Jack Flaherty (HOME, RHP, 30 📉 初期退化)
- **Tier 驗證**：腳本 tier = ⚪ Below Average（ERA 5.90 / FIP 5.72 / K-BB% 5.0 / WHIP 1.79），xFIP 4.91 顯示結構面接近 🟢 Back-end，但 K-BB% 5.0 是 league bottom 區，WHIP 1.79 = 實質壞透
  - 同意 ⚪ Below Average。xFIP-ERA gap 約 -1.0（結構好過 ERA），但 BB-rate 高 + 球速 87.4 平均（已退化）+ 近 3 場 8ER/14IP（5.14）→ 沒有運氣回歸跡象，是真正的 below-average
- **Reverse platoon 信號**：未 fire（vs LHB OPS .898 / vs RHB OPS .780，正常 platoon）
  - 但 vs LHB OBP .434 警示 BB 失控（與 K-BB% 5.0 一致）
- **對手打線威脅**：BOS 🟡 Average vs RHP，但 top 5 last7 OPS 三人 ≥ .847（Contreras 1.036 / Abreu .958 / Duran .847）— Flaherty 的 FF 45.8%（single-pitch dependent ≥ 45%）+ TTO3 penalty K% 從 28% 掉到 23.4% → BOS 打 2-3 輪後吃定他

### Sonny Gray (AWAY, RHP, 36 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier = 🟢 Back-end Starter（ERA 4.30 / FIP 4.32 / xFIP 3.70 / K-BB% 8.0），xERA 5.66 vs ERA 4.30 = +1.36 gap → ERA **高估**（運氣偏好）
  - 同意 🟢 Back-end，但要往下修。xERA 5.66 對齊 Below Average 區間；FIP 4.32 跟 ERA 一致；xFIP 3.70 = swing-through 預期但 36 歲 + whiff 8.3% 低 → 偏向 xFIP 機率低。**真實水準靠近 ⚪/🟢 邊界**，本場以 🟢- 看待
- **Reverse platoon 信號**：🔴 fire — Δ +0.438（vs RHB OPS .974 > vs LHB OPS .536）。樣本 47/52 BF 已過門檻
  - DET top 5 中 RHB 三人（Torres / Torkelson / Dingler）都會吃到 Gray 的反向劣勢；Dingler vs RHP .884、Torkelson EV95% 46.7、Dingler EV95% 52.9 + Barrel 16.1 → high-impact contact 機率 ↑
  - LHB McGonigle / Greene 雖然 normal platoon 上是占優，但 Gray 對 LHB OPS .536 → 這 2 人本場反而被壓制
- **對手打線威脅**：DET 是 🟠 Strong tier（xwOBA .353），但 vs Gray 變更危險 — 因為威脅集中在 RHB 三人段（3-5 棒）。**Gray 第三輪 OPS 從 .600 跳到 .789（Δ +0.189，high）+ K% 掉 5.2pp** → 90+ 球後就崩，而 DET bullpen IL ×3 反向意味 Gray 也撐不到後段被換到崩盤牛棚

## 打線評級

### HOME — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟡 Average
  - season 🟠 → matchup 🟡 是降一檔，原因是 DET 整體 vs RHP xwOBA 沒特別突出（top 5 中 Torkelson .748 / Greene .794 平庸）。**但本場應該往回升 — 因為 Gray 的 reverse platoon + tto3 雙信號讓「vs Gray」≠「vs RHP 平均」**。本場我 lean 比 🟡 Average 高半檔
- **chain_break / heat_vs_babip 信號**：🟠 chain breaks at #8-9（OPS 落差 0.186，medium）
  - 影響 7-9 棒銜接，1-2 出局後段續攻能力下降；Comerica 大球場利打深安打，這個落差比常規球場放大一點

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak
  - season 🟡 → matchup 🟢 = 降一檔。Story / Duran vs RHP 都是 .501 / .572 災難，Anthony .683 也 below average。但 last 7 三人在熱（Contreras 1.036 / Abreu .958 / Duran .847）+ Flaherty 是 ⚪ Below Average + single-pitch dependent → **本場 lean 從 🟢 Weak 上修一檔到 🟡-，靠 1-2 棒 + Flaherty 質量差吃飯**
- **chain_break / heat_vs_babip 信號**：🔴 chain breaks at #2-3（OPS 落差 0.335，high）
  - Contreras → Abreu → Story 中 Story OPS .543 是 chain killer，#3 棒成黑洞 → 1-2 棒上壘後 Story 解決掉，難形成大局；BOS 攻擊 chain 高度依賴 1-2 棒長打而非串聯

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.88 / 10 / 3 名 🔴🔴 極高 | 3.4 / 8 / 2 名 🔴 高 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：DET ERA 3.88 看似中上，**但 core IL 達 3 人（Brieske / Melton + 1 人 IL60d）= 牛棚崩盤等級**。Flaherty 撐不過第 5-6 局時，DET 後段必須丟次級 RP；BOS 1-2 棒在熱（Contreras / Abreu last7 OPS > .950）→ DET 牛棚是本場最大壓力點，第 7 局後 BOS 得分機會明顯放大
- AWAY 牛棚：BOS ERA 3.40 較好，**但 core IL 達 2 人（Coulombe / Slaten 都 IL15d）= 明顯吃緊**。Gray 36 歲老將通常 5-6 局，丟給後段時 DET RHB 三人段會吃到 BOS B-tier RP；不像 DET 那麼崩，但 7-8 局有空隙

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🟠 HOME single-pitch dependent：主球種使用率 45.8%（≥45.0%）
- 🟠 HOME TTO3 penalty：OPS Δ +0.027（TTO1 0.762 → TTO3 0.789），第三輪明顯衰退；K% 從 28.0% 掉到 23.4%（Δ -4.6pp）（career fallback）
- 🔴 AWAY reverse platoon Δ +0.438（vs RHB OPS 0.974 > vs LHB OPS 0.536）— RHP 對非預期手別反而吃虧
- ℹ️ AWAY balanced 4+ pitches：最高球種僅 22.4%（<25.0%）
- 🔴 AWAY TTO3 penalty：OPS Δ +0.189（TTO1 0.600 → TTO3 0.789），第三輪明顯衰退；K% 從 27.5% 掉到 22.3%（Δ -5.2pp）（career fallback）
- 🟠 HOME chain breaks at #8-9：OPS 落差 0.186
- 🔴 AWAY chain breaks at #2-3：OPS 落差 0.335
- 🔴 ⏳ HOME 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
- 🔴 ⏳ AWAY 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - 雙隊牛棚同時吃緊（DET 3 名 / BOS 2 名）→ 本場高機率變「先發誰先 fade，後段失分多」的局面。Flaherty K-BB% 5.0 + WHIP 1.79 比 Gray 撐到 5+ 局更難 → DET 牛棚先暴露，總分判讀偏多。⏳ 標記提醒短半衰期：對手 7 天可調 mix，但 IL 名單沒法立刻變

## 條件修正

- Park Factor: 106.0 → +0.30 run（Comerica Park HR +5%；大外野利打深安打但壓 HR — 對 BOS Abreu / Anthony 這類 fly-ball 打者影響稍小）
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：兩位 RHP 都已退化（Flaherty 30 初期 / Gray 36 明顯）；非 doubleheader

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 5.5 | +0.5（reverse_platoon high +0.4 → DET RHB 三人段吃 Gray；BOS bullpen IL ×2 +0.3 → 7-8 局 DET 加碼；DET chain_break #8-9 -0.2；interaction cap） | 6.0 |
| AWAY | 6.5 | +0.4（Flaherty single-pitch +0.2 + TTO3 +0.1 互動取 single max；DET bullpen IL ×3 +0.5 → 後段大放開；BOS chain_break #2-3 -0.3；cap 後 +0.4） | 6.9 |
| Total | 12.0 | +0.9 | 12.9 |

## 整體判斷

- **方向（基本面）**：AWAY (BOS) 微優；勝負空間大但傾 BOS 約 +0.9 run 差
- **總分（基本面）**：12.9（base 12.0 + 信號 +0.9，落在 over 8.5 / over 9 / over 10 / over 11 區間都成立；考慮投手都 below-mid + 雙牛棚崩，over 偏多）
- **方向信心**：62%（lean BOS 但非強）
  - 為何不到 70%：Gray 的 xERA 5.66 + 36 歲 + reverse_platoon = Gray 隨時可能爆掉；Flaherty 雖然紙面更差，但他在主場 + Comerica 大球場略保護
  - 為何 > 50%：BOS 1-2 棒熱 + Flaherty K-BB% 5.0 + DET 牛棚崩盤級（3 人 core IL）→ 中後段攻擊優勢清晰
- **風險**：
  1. **Gray 反向 platoon 樣本（vs RHB 47 BF）** 不夠大，可能是噪音；若 Gray 今天找到對 RHB 的解法 → DET 中段預期得分快速塌陷
  2. **DET 牛棚 IL ×3 是 ⏳ 短半衰期信號**，dossier 顯示 Brieske / Melton IL60d 不會今晚回來，但 DET 可能用先發 piggyback 或長中繼 — 真實牛棚壓力或許沒崩盤級那麼嚴重
  3. **打線都是 🟡 projected**（未公布），Story / Duran 若被換掉，BOS chain_break 嚴重度會變
  4. **xERA-ERA 雙向偏移**：Gray ERA 4.30 / xERA 5.66 + Flaherty ERA 5.90 / xERA 5.34（兩人 xERA 都 ≥ 5.34）→ 場面可能比 ERA 預期更難看，total 走 over 機率比 12.9 點估值還高

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
