## 投手對決

### Eric Lauer (HOME, LHP, 30 yo)
- **Tier 覆寫**：沿用腳本 ⚪ Below Average（xERA 4.56 也只到聯盟平均偏下水準）
- 真實水平判斷：
  - 表面 ERA 6.75 略受 BABIP/HR 拖累（xERA 4.56），但底層體質仍差：K-BB% 6.7%（聯盟均約 13%）、whiff% 6.9%（極低）、平均球速 86.2 mph（即便 LHP 也偏慢）。
  - FIP 6.45 / xFIP 5.08 都告訴我們長線水平大約是 ERA 4.8–5.3 區間的 back-end / below-average 先發。
  - 30 歲，📉 初期退化，球速無法靠回升救濟。
- 對手打線威脅：
  - BOS vs LHP 主力：Contreras（.815 vs LHP）、Abreu（.794 vs LHP）為主要威脅。
  - Story（.525 vs LHP）、Roman Anthony（.678 vs LHP）為中等。
  - BOS 整體 K% 23.4 偏高，但 Lauer 三振能力差，K 對沖不大。
  - 預期 BOS 對 Lauer 5 局內掉 3–4 分屬合理區間。

### Brayan Bello (AWAY, RHP, 26 yo)
- **Tier 覆寫**：沿用腳本 ⚪ Below Average，但實際更接近「最底層 below average」。
- 真實水平判斷：
  - ERA 9.00 / xERA 8.24（兩者一致，沒有運氣補貼），表面與底層皆崩。
  - K-BB% 1.7%（接近完全沒有控球差距）、WHIP 2.27、hard_hit% 30.4、barrel% 16.1。
  - vs LHB 慘到不像 MLB：.420/.483/.900（59 BF，**樣本夠大不算雜訊**）。vs RHB 也 .327/.449。
  - FIP 8.24 vs xFIP 4.45：xFIP 用聯盟均 HR/FB rate，目前 HR 比例異常，但即便 HR 回歸，K-BB% 1.7% 也支撐不住穩定先發。
  - 26 歲 ⚡ 巔峰期應該是身體狀態最好的時候，這種數據強烈暗示**指令性、變化球品質或健康問題**。本季 5 GS 14.7 IP 樣本仍顯著小，但已經足以判斷現階段狀態極差。
- 對手打線威脅：
  - TOR vs Bello（RHP）：Vlad Jr（.888 OPS，vs RHP .809，EV95% 48.4 / barrel 13.2）為核心威脅；Okamoto（vs RHP .648 但 EV95% 52.2 / barrel 13.4）大棒尚未發揮但底層強。
  - 關鍵：Bello vs LHB 嚴重崩盤 → TOR 的 LHB 群（Varsho、Giménez、Daulton）將成為主要 run-producer。
  - 預期 TOR 對 Bello 5 局內掉 4–6 分為基線。

## 打線評級

### HOME (TOR) — 🟢 Weak / ⚖️ Normal
- **Tier 覆寫**：沿用腳本。xwOBA 0.299 / OPS .669 確實偏低，但 last7 BABIP 0.247 壓低了表面成績；底層 EV（Vlad 48.4、Okamoto 52.2）保留爆發潛力。對 Bello 這種「vs LHB 大爆炸」的投手，TOR 的左打結構可放大優勢 → 本場威脅力高於 baseline。

### AWAY (BOS) — 🟢 Weak / ⚖️ Normal
- **Tier 覆寫**：沿用腳本。xwOBA 0.305 / OPS .653，K% 23.4 偏高，串聯能力（chain OBP top3 .320 / SLG mid .280）偏弱。但對 Lauer（LHP）的對位有 Contreras / Abreu 兩名 vs-LHP 強打，預期可拿到 baseline 得分。

## 牛棚

| | HOME (TOR) | AWAY (BOS) |
|---|---|---|
| ERA / IL 數 / 核心 IL 估計 | 4.25 / 7 / 多為 IL60d，**無確定核心**（Bowden Francis、Cody Ponce 等多為長期傷停） | 3.49 / 6 / 多為 IL60d / IL15d，**Justin Slaten** 屬高槓桿但僅 IL15d |

### 牛棚雙向修正值
- HOME (TOR) 牛棚：對手 +0.0 run | TOR ML 0%（dossier IL 名單未列出明確核心 IL，無法套累計效應；ERA 4.25 算中段）
- AWAY (BOS) 牛棚：對手 +0.0 run | BOS ML 0%（ERA 3.49 較佳但無核心 IL → 無修正）
- **戰術影響**：兩位先發都極可能 5 局內退場，牛棚使用量會非常大 → BOS 牛棚（3.49）品質明顯優於 TOR（4.25），這對 BOS 是隱性 + 約 0.3–0.4 run 的後段優勢，但模型沒有明確信號表項，僅做敘事提示。

## 風險提示

- ⚠️ HOME 投手 Flag 13 (era_xera_delta=2.19)：
  - **判讀：部分運氣，但 base 仍差**。Lauer xERA 4.56 ≈ 聯盟均，xFIP 5.08 / K-BB% 6.7%。表面 ERA 6.75 偏離 xERA，主因為 BABIP 與 HR/FB 偏高，**但 xERA 已經不算好**。即使 ERA 朝 xERA 方向回歸，對 BOS 這種 OPS .653 的弱打，仍可能掉 3–4 分。**不下修總分預測**。
- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.247)：
  - **判讀：偏向回歸**。TOR 主力 EV95%（Vlad 48.4、Okamoto 52.2）顯著高於聯盟均（~38-40），擊球品質本身強，0.247 的 BABIP 在這種 EV 結構下是雜訊。預期回到 .280–.290 區間，**對本場有利於 TOR 攻勢**。**不自動 ±run value**，但敘事方向：TOR 進攻不要被表面冷打混淆。

## 條件修正

- Park Factor: 99.0 → -0.05 run（接近中性）
- 先發 tier：兩位都 ⚪ Below Average，**沒有 strong ace+ 或 solid+ 的下修折扣**（信號表的 -1.0 / -0.5 條件未滿足）
- 雙方先發投手品質落差大（Lauer xERA 4.56 vs Bello xERA 8.24 → Lauer 約優 3.7 ERA）→ 比賽方向訊號強
- 無 doubleheader / 天氣為室內球場（Rogers Centre 可關屋頂）→ 無風天氣修正
- BOS 牛棚較佳（3.49 vs 4.25），長中繼局數會放大此優勢，但無明確核心 IL 信號表項可量化 → 敘事提示

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (TOR) | 8.3 | -1.0（Bello xERA 校正 + Lauer 對 BOS 弱打不會被打太慘）；+0.0（無核心牛棚 IL） | 7.3 |
| AWAY (BOS) | 6.6 | -1.0（Lauer xERA 仍可頂住 BOS 弱攻 + Bello 真實 ERA 已被 base 充分反映）；-0.3（BOS 牛棚 vs TOR 牛棚 5+ 局負擔差距，但反向加在 TOR 那邊） | 5.6 |
| Total | 14.9 | -2.0 | **12.9** |

> **Sanity check**：base 公式 14.9 偏高（因雙方 ERA 都極端），但即便用 xERA 重算（Lauer 4.56、Bello 8.24），加總後仍應落在 12–14 區間，遠高於 8.5 的 O/U 線。

## 整體判斷

- **方向（基本面）**：TOR 主場小幅優勢（先發體質 Lauer >> Bello 為核心訊號；TOR 打線結構放大 Bello vs LHB 的弱點；BABIP 預期回歸有利 TOR 攻勢）。BOS 唯一的賣點是牛棚較好，但需要先發撐到中後段才能發揮。
- **總分（基本面）**：強烈 OVER 傾向。雙方先發底層皆 below average，xERA 加總已達 12.8（Lauer 4.56 + Bello 8.24），即使打線 Weak 也難以壓到 8.5 以下。修正後總分約 12.9 vs O/U 8.5 → 差距 +4.4 run。
- **信心**：MEDIUM-HIGH（OVER 訊號強）/ MEDIUM（ML 訊號中等，TOR 雖體質好但 BOS 仍有 vs-LHP 強打）
- **風險**：
  1. Bello 9.00 ERA 樣本仍小（5 GS / 14.7 IP），單場可能突然「正常化」掉 3 分內，但 K-BB% 1.7% 與 vs LHB .420 樣本足以支持 base 仍差。
  2. Lauer ERA 6.75 受運氣拖累，xERA 4.56 暗示能比 base 預期表現好。
  3. 兩隊打線都 🟢 Weak + last7 偏冷 → 雙方上限受限。
  4. Rogers Centre HR +4%（屋頂可開合，本場若關閉則風的因素消失），HR 環境略加分但 Runs PF 中性。
