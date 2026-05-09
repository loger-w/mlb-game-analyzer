## 投手對決

### Payton Tolle (HOME, LHP, 23 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = —（GS 3 / 17.7 IP，樣本不足以算 v2 score），dossier tier_script 🔴 Elite Ace
  - 表面數據（ERA 2.04 = xERA 2.04，K-BB% 25.3，whiff% 13.1，FIP 2.25）的確是 elite 級；ERA-xERA 完美吻合（無 BABIP 運氣偏差），球質 = 結果。但 GS 僅 3 場、IP 17.7，**正式評級保守降為 🟠 Strong Ace 偏 🔴**，本場敘事仍按 elite 對待但留小樣本浮動 buffer。
- **Reverse platoon 信號**：未 fire（vs LHB BF 18 < 30 觸發門檻）。隱性風險：vs LHB OPS .749 (.235/.278/.471) 反而比 vs RHB OPS .277 (.070/.184/.093) 高 0.47，左投對左打反向偏弱
  - 本場 TB 預計打序 5 人中有 3 LHB（Aranda、Mullins、Mullins 已列；Caminero/Díaz 為 RHB）。但 18 BF 樣本太薄，正常情境下 LHP-LHB 仍會壓制；列為次階風險而非主要敘事
- **對手打線威脅**：🟢 LOW。TB 打線 vs LHP top5 中 4 人（Caminero/Aranda/Simpson/Mullins）比 season OPS 低（典型 LHP 退讓型左打側）；Tolle 對 RHB 的 .093 SLG 是壓倒性，TB 主要 RHB 威脅（Caminero/Díaz）會被壓死；加上 TB 自身 chain breaks 在 #4-5，5 棒後幾乎無威脅 → Tolle 撐 5-6 IP 失 1-2 分為 likely 區間

### Nick Martinez (AWAY, RHP, 35 📉📉 明顯退化)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p40, K-BB% p54），gap vs ERA-only = -43.2 → ERA 嚴重高估真實水平
  - **偏向運氣 + 防守加持，非結構性突破**：xFIP 4.28 / K-BB% 10.7 / whiff% 7.3 / 球速 86.7（35 歲退化軌跡上）這些 underlying 全指 🟡 Solid 等級；ERA 1.71 由 BABIP / LOB% / HR/FB 運氣值撐起。**依 Flag 8 紀律不自動下修 base formula**，但敘事將 Martinez 視為 🟡 而非 🟠/🔴，BOS 預期得分上限按 Solid Starter 對位給
- **Reverse platoon 信號**：未 fire（vs LHB .602 / vs RHB .497 兩側都壓制，符合 RHP 預期）
- **對手打線威脅**：🟡 MEDIUM-HIGH。BOS 前段熱（Duran last7 .833 / Abreu 1.068 / Rafaela .923），Abreu 對 RHP .851 是高品質威脅；中段 Yoshida/Story 平庸偏弱；chain break #5-6（Story .514 → Gasper .000）讓 5 棒後幾乎無延續火力。Martinez 進入第三輪 K% 從 19.2 → 8.9（drop -10.3pp，TTO3 fire）→ BOS 第二、三輪打擊機會增加，加上 35 歲 + 5-6 IP 上限 → BOS 中段集中得分壓力大

## 打線評級

### HOME — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak — 與 season 一致；前段（Duran/Contreras/Abreu）last7 偏熱，中後段（Story/Gasper/Mayer/Durbin）持續偏弱。對 Martinez 這種 finesse-RHP 屬於略佔優：球速 86.7、whiff% 7.3，BOS 前段對 contact-pitcher 的擊球品質（EV95% 40+）能轉成 BABIP 進攻。整體**同意 Weak 評級但加 last7 hot 修正 → 本場略上修**
- **chain_break / heat_vs_babip 信號**：
  - 🔴 chain_break #5-6（Story .514 → Gasper .000，落差 .514）— 5 棒後幾乎無延續火力，Martinez 即使被前 4 棒打到也容易在第 5-6 棒切斷局面，**壓制 BOS 一輪超過 3 分的可能**；此 break 屬 high (≥0.300)，magnitude 取下界 -0.3 但已在 base formula 內
  - heat_vs_babip 未 fire

### AWAY — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak — TB top5 中 4 人（Caminero/Aranda/Simpson/Mullins）vs LHP OPS 都比 season 低 0.05-0.15，唯 Díaz vs LHP OPS .866 與 season 持平。**與 season tier 一致甚至略下修**：本場面對 LHP（且為 elite 球質）會放大這個 platoon 劣勢
- **chain_break / heat_vs_babip 信號**：
  - 🔴 chain_break #4-5（Díaz .868 → Mullins .430，落差 .438）— TB 5 棒後火力銳減，Tolle 中後段壓制變容易，**壓制 TB 一輪超過 2 分的可能**
  - Aranda last7 BABIP .600（OPS 1.033）→ 顯著 lucky-hot，雖未 surface 為 signal（top5 整體未達 fire 條件），但 Aranda 個人這個 BABIP 不可持續，本場期望值應回歸 season .824 OPS

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.42 / 7 / 2 (Coulombe + Slaten 皆 IL15d) | 4.02 / 7 / 3 (Uceta + M. Rodríguez + 1 — 皆 IL60d 長傷) |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：🔴 高（core IL ×2）。ERA 3.42 整體尚可，但 Coulombe + Slaten 兩名 setup/HL RP 缺陣（IL15d 短期）→ Whitlock / Houck / Chapman 等可用性正常，後段並未崩盤，僅是「核心薄」。Tolle 若按預期 5-6 IP 退場，BOS 牛棚需吃 3-4 局；G1 (5/8) BOS 投出 shutout 2-0，牛棚在系列中仍有彈性。對 TB 末段（5-9 棒火力薄）影響較小 → 本場對 TB 額外得分加成估 +0.2 ~ +0.3 run
- AWAY 牛棚：🔴🔴 極高（core IL ×3）。ERA 4.02 偏差 + Uceta 與 M. Rodríguez 兩名核心 IL60d 長傷（無短期回歸希望）→ TB 後段 high-leverage 嚴重缺人。Martinez 若 TTO3 penalty fire 提早退場（5-6 IP）→ TB 牛棚需吃 4 局，撞上 BOS 前段火力（Duran/Contreras/Abreu last7 OPS .833/.764/1.068）+ 中段 Yoshida → BOS 末段（7-9 局）得分機會放大顯著。對 BOS 額外得分加成估 +0.4 ~ +0.6 run（崩盤級的下界保守取，因 BOS 中後段 chain 自身斷掉）

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-2.20):
  - **判讀：偏向運氣 + 防守加持，非結構性突破**。Martinez 35 歲在退化軌跡（球速 86.7 平均、max 94.4），underlying（xFIP 4.28 / K-BB% 10.7 / whiff% 7.3 / barrel% 6.9）全指 🟡 Solid Starter；ERA 1.71 是由低 BABIP（防守 + 球場運） + 高 LOB% + 低 HR/FB 撐起。本場 Fenway PF 104 + 順風 Out To LF 不利 Martinez 的 GB-style approach（SI 31% + CH 27%），加上對手 BOS 前段 last7 火力（Duran/Abreu/Rafaela 都 hot）→ ERA 有可能在這場開始向 xERA 靠攏
  - **依紀律不自動下修 base formula** — 但敘事將真實壓制力按 🟡 Solid Starter 對位給；AWAY 投手實質 ceiling 是 5-6 IP 失 2-3 分，而非 ERA 顯示的 5-6 IP 失 1 分

### 額外信號
- 🟠 HOME single-pitch dependent：主球種使用率 46.3%（≥45.0%）
- 🟠 AWAY TTO3 penalty：OPS Δ +-0.049（TTO1 0.830 → TTO3 0.781），第三輪明顯衰退；K% 從 19.2% 掉到 8.9%（Δ -10.3pp）
- 🔴 HOME chain breaks at #5-6：OPS 落差 0.514
- 🔴 AWAY chain breaks at #4-5：OPS 落差 0.438
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
- 🔴 ⏳ AWAY 牛棚 core IL ×3：🔴🔴 極高（牛棚崩盤級）
  - **本場高度受影響**：Flag 8（Martinez ERA 不可信、真實 5-6 IP 失 2-3 分）+ TTO3 penalty（K% drop -10.3pp，Martinez 第三輪可能提早退場）+ core IL ×3（牛棚崩盤）→ **三重同向壓力**疊加，BOS 7-9 局得分機會顯著放大。但 BOS 自身 chain break #5-6 限制了 big inning 上限 → 對 BOS 影響估 +0.4 run（取 Table B core_il high 區間 +0.4~0.8 的下界，因 chain 自斷抵銷）

## 條件修正

- Park Factor: 104.0 → +0.20 run（Fenway 略偏打者，但 HR PF -15% 抵銷飛球轉本壘打）
- 天氣：Cloudy, 59°F, wind 10 mph, Out To LF
  - 影響判讀：**整體中性**。59°F 屬 50-60 區間「輕度利投」，球的飛行距離略受抑制；wind 10 mph Out To LF 是輕度順風（往 Green Monster 方向，但 Fenway HR -15% 結構抑制本就壓 HR）→ 兩股力量大致抵銷，無明顯傾向。風速 10 mph 屬 8-15 mph 噪音段，**summary 不另加 ±run**
- 先發 tier / doubleheader：非 doubleheader（系列 G2，獨立排程）。先發 tier 落差是本場最大條件因子：Tolle 真實 🔴/🟠 vs Martinez 真實 🟡，HOME 投手淨優勢明顯（已透過 base formula 反映）

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.7 | +0.4（core_il_count AWAY ×3: +0.4 / tto3_penalty: +0.2 / chain_break HOME #5-6: -0.2；interaction max+0.1） | 4.1 |
| AWAY | 2.4 | +0.0（core_il_count HOME ×2: +0.2 / pitch_mix_concentration HOME single-pitch: +0.1 / chain_break AWAY #4-5: -0.3） | 2.4 |
| Total | 6.1 | +0.4 | 6.5 |

## 整體判斷

- **方向（基本面）**：HOME（Boston Red Sox）。Tolle 真實壓制力（🔴/🟠 elite 球質）對 Martinez 真實水平（🟡 Solid，ERA 1.71 不可信）= 投手對位淨差 ≥ 1 tier；TB 牛棚 core IL ×3 vs BOS ×2 = 牛棚深度差距明顯；TB 打線 vs LHP 全面退讓（top5 中 4 人 OPS 比 season 低）+ Tolle 對 RHB 的 .093 SLG 壓死 TB 主要 RHB 威脅。BOS 主場 + 系列 G1 已勝 → 動能延續
- **總分（基本面）**：6.5 run（HOME 4.1 / AWAY 2.4）。base 6.1 + 牛棚崩盤 + TTO3 penalty 修正後略偏 over 6.0，但因兩邊 chain 都斷在中後段、Tolle 質量壓 TB 上限，難破 7.0
- **方向信心**：65%。HOME 勝出方向偏強但非極端 — Tolle GS 3 小樣本是最大不確定（17.7 IP 不足以驗證 elite 持續性），且 TB 近 10 場 8-2 / RA 1.50 的防守動能極熱（如 Martinez 運氣再延續一場，HOME 進攻可能受壓）。對於信心 ≤ 50% 的場景才寫持平，本場結構偏差明確 → 65%
- **風險**：
  1. **Tolle 小樣本陷阱**：GS 僅 3 / IP 17.7，elite 數據能否持續 sustainability 待驗；vs LHB 18 BF OPS .749 是隱性 reverse-platoon 苗頭，若樣本擴大且 TB 主要 LHB（Aranda/Mullins）狀態好，Tolle 可能比 dossier 顯示更脆
  2. **Martinez ERA 運氣延續**：xERA-ERA gap 2.20 雖大，但近 3 場 ER/IP 4/16.7（ERA 2.16）顯示運氣短期仍延續中；單場層面 BOS 進攻可能繼續被低 BABIP 壓制（Flag 8 紀律：不下修 base，但實際 over 機率比 +0.3 更低）
  3. **TB 整體形勢極熱**：近 30 天 22-8 / RS-RA +25，近 10 場 RA 1.50 — 結構性數據偏 TB 站不住的地方少，比賽級的隨機波動可能讓 TB 偷一場
  4. **兩邊牛棚都吃緊**：HOME 2 / AWAY 3，若進入延長賽或 close game 後段，AWAY 牛棚崩盤級 IL 反而是 TB 更大風險，但 close-game variance 高 → 信心不宜過 70%

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組