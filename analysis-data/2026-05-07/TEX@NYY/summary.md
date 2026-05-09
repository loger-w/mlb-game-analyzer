## 投手對決

### Paul Blackburn (HOME, RHP, 32 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = —（season IP=14、GS=0，sample 不足無法 score），gap vs ERA-only = —
  - ERA 3.21 / xERA 3.57 表面 🟡 Solid，但 IP 僅 14（**Flag 8 小樣本**）；近 3 場累計 ER/IP = 1/4.0，明顯被當 opener / piggyback 使用而非常規先發。不引用 ERA 推斷今日表現上限，視為「2-3 IP 上限的銜接型」處理，後段牛棚 4-5 IP 是主軸。
- **Reverse platoon 信號**：未 fired（vs LHB 28 BF < 30 樣本門檻）
  - vs LHB .555 / vs RHB .711 OPS 雖反向，但 LHB 端只有 28 BF，**樣本太薄不下結論**。職業生涯 sinker/cutter 對 LHB 偏好不明顯。
- **對手打線威脅**：🟠 Strong。TEX 對 RHP 球種以 sinker tail-away 處理難度中等，但 1-4 棒（Nimmo .840 / Duran .910 / Seager .843 / Jung .956 vs RHP）皆 > .800 OPS，**TEX 上輪 6 分破門就是這一段做出**；Blackburn whiff% 7.2 偏低、靠弱接觸吃飯，**Yankee Stadium HR +12% + 9 mph Out To RF 順風**對 LHH 有利（Nimmo / Seager / Pederson / Carter 皆 LHH，可拉打短牆）。風險集中在前 3 局。

### MacKenzie Gore (AWAY, LHP, 27 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p87），gap vs ERA-only = +50.2（**Flag 8 / tier_mismatch high**）
  - 同意 score-derived tier 大方向：xFIP 3.26 / K-BB% 17.9 / FF max 97.9 mph 結構性指標都是真材實料；ERA 4.67 主要由 barrel% 12.6 偏高 + sequencing 噪音推高（xERA 4.28 與 ERA 差 0.39，並未支持「全是運氣」說法 → 結構偏 ace、近期被打硬球）。近 3 場 5 ER / 16.3 IP = ERA 2.76，已回到 ace 軌跡。**不自動下修對 TEX 失分預期**。
- **Reverse platoon 信號**：未 fired
- **對手打線威脅**：🟠 → 局部 🔴。NYY 1-3 棒對 LHP 是聯盟最毒組合：**Goldschmidt .714 / Judge 1.133 / Bellinger .949（last7 1.516）vs LHP**。Gore vs RHB 116 BF .712 OPS 並未壓制右打，barrel% 12.6 + Yankee 短牆 + 順風 → 至少 1 支 HR 預期合理。Gore 本場「結構強但對位差」，預期 5.1-5.2 IP / 2-3 ER，TTO3 career Δ +0.073（未到 0.100 門檻不 fire，但接近）→ 第三輪後仍可能被換。

## 打線評級

### HOME — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🟠 Strong（與 season 同 tier）
  - 同意維持 🟠 並小幅上修：top 3（Goldschmidt R / Judge R / Bellinger L）vs LHP 對位是 platoon-favorable（Bellinger 對 LHP 反而打得更好，last7 1.516 是 hot streak）。Judge 對 LHP 1.133 OPS + 27.9% Barrel% → 單支 HR 機率比平日高。
- **chain_break / heat_vs_babip 信號**：🟠 chain breaks #4-5（OPS 落差 0.194，medium）
  - 影響本場 cleanup 串聯：Rosario .830 → Chisholm .636 季 OPS，對 LHP 落差更大（Rosario .781 / Chisholm .481 = Δ 0.300）。**4-5 棒清壘能力略受限**，但 1-3 棒火力已足以製造分數，整體影響中等偏小（-0.1 run）。

### AWAY — season tier 🟢 Weak / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak（與 season 同 tier）
  - 同意維持 🟢，但**有局部上修**：1-4 棒（Nimmo / Duran / Seager / Jung）對 RHP OPS 全部 > .800，是 TEX 唯一可靠攻擊段；Duran last7 1.130 hot streak。**Blackburn 是 quasi-opener**，這 4 棒第一輪就能對位到他 → 前 3 局是 TEX 主要破門窗口；之後對 NYY 牛棚（ERA 3.22）難度上升。
- **chain_break / heat_vs_babip 信號**：🔴 chain breaks #8-9（OPS 落差 0.637，high）
  - 影響本場攻擊 chain：Jansen .637 → Foscue .000，**Foscue 幾乎是 auto-out**（無 MLB 樣本），9 棒之後再回 Nimmo 有 2 棒空轉。high severity → -0.2 run（取下界）。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.22 / 4 / 0 | 2.76 / 6 / 0 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 3.22 中段水準，core IL 0 → 可用性無顯著扣分。**但今日 Blackburn 預期僅 2-3 IP**，NYY 牛棚需吃 6+ IP，使用量遠超平日 → 後段（7-9 局）可能被迫派次優選擇，TEX 1-4 棒第二/三次面對他們有機會擊穿。可用性 normal、消耗風險 ↑。
- AWAY 牛棚：ERA 2.76 **聯盟前段**，core IL 0 → 是 TEX 本場最大隱性武器。Gore 預期 5-6 IP 後牛棚僅需 3-4 IP，pitchers IL 6 多為先發 / 長傷，**核心未受影響**。NYY 中後段對 sub-3 ERA 牛棚難取分 → 壓制 NYY 第 7 局後得分能力。

## 風險提示

- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.238):
  - 結構性 vs 運氣噪音：TEX 整季 xwOBA 0.290 確實偏低（結構偏弱），但 last7 BABIP 0.238 低於 .260 已到「可能反彈」門檻。輔證：昨日（5/6）TEX 客場剛打出 6 分破門 → 顯示火力並非完全冰封，今日 1-4 棒對位 Blackburn 也有利。**敘事判讀：傾向「冷而非死」，今日有 3-4 分潛力，但結構性 weak tier 仍壓總分上限**。**不自動 ±run value**。

### 額外信號
- 🟠 HOME chain breaks at #4-5：OPS 落差 0.194 → 已於「打線評級」段處理（-0.1）
- 🔴 AWAY chain breaks at #8-9：OPS 落差 0.637
  - 本場主要受影響的是 TEX 第二輪後的攻擊延續性。Foscue 9 棒幾乎是「免費出局」，與 Flag 3 的 BABIP 偏低同向疊加 → TEX 的「破門靠前 4 棒、難拉長 inning」模式被進一步強化。**單側 cap 紀律下，已併入 -0.2 不另加**。

## 條件修正

- Park Factor: 96.0 → -0.20 run（已含於 base）
- 天氣：Partly Cloudy, 60°F, wind 9 mph, Out To RF
  - 影響判讀：60°F 在 50-60°F 邊界，**輕度利投**（球易死、肌肉熱身慢）→ 微壓得分；但 9 mph Out To RF **輕度利 HR**，特別對 LHH 拉打方向（NYY 的 Bellinger / Chisholm Jr / Domínguez / Grisham，TEX 的 Nimmo / Pederson / Carter）。Yankee 短牆與順風疊加 → **HR 變異上升，但總分淨影響近中性、略偏 HR-driven**（即低分多由 HR 製造，非鏈式）。
- 先發 tier / doubleheader：Blackburn 為 quasi-opener / 14 IP 小樣本（**已於投手對決段下調為 2-3 IP 銜接型**），非常規先發；Gore tier_mismatch +50.2 不自動下修預期。非 doubleheader。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.6 | -0.1 | 4.5 |
| AWAY | 2.5 | -0.2 | 2.3 |
| Total | 7.1 | -0.3 | 6.8 |

## 整體判斷

- **方向（基本面）**：HOME (NYY) 持有顯著優勢
- **總分（基本面）**：6.8（base 7.1 - chain break 修正 0.3；HR-driven，下限拉得很低、但 Yankee 短牆 + 順風使單支 HR 後 5-6 分區間打開）
- **方向信心**：~63%
  - 支撐：Gore 雖結構強但 NYY top 3（Goldschmidt / Judge / Bellinger）對 LHP 是夢魘對位、Judge last7 + 27.9% Barrel% 對 Gore 12.6% Barrel%-allowed 等於正面對撞；NYY 主場、近 30 25-12、近 10 RS 5.80 火力延續中；Blackburn 雖小樣本但 NYY 牛棚 ERA 3.22 + core 健康，後援不致崩。
  - 壓力：TEX 牛棚 2.76 ERA 是隱性武器、Gore tier_mismatch 是真結構、TEX 昨日剛打出 6 分破門 + Duran last7 1.130 hot 不容忽視 → 67% 過於激進。
- **風險**（4 點）：
  1. **Blackburn 角色不確定**：14 IP / 4 IP-3 GS 顯示 NYY 用他像 opener / piggyback。若 NYY 計畫是「Blackburn 2 IP + bulk 投手」而 bulk 是品質佳的長中繼，NYY 失分上限被壓低；若 bulk 是低品質 → TEX 1-4 棒能在中局擴大領先。**在公布前是最大未知數**。
  2. **Gore 的 tier_mismatch 是真結構**：xFIP 3.26 / K-BB% 17.9 + 速球 max 97.9 mph 不是運氣 — 若他今日進入 ace mode，NYY 預期 4.5 可能下修到 3.0-3.5，總分破 6 都困難。**Flag 8 紀律下不自動扣，但這是基本面對總分最大下行風險**。
  3. **TEX 牛棚 ERA 2.76 壓制 NYY 末段**：Gore 5-6 IP 後 TEX 牛棚僅需吃 3-4 IP；NYY 第 7-9 局對位 sub-3 ERA 牛棚難加分，**抑制總分 over 7.5+ 機率**。
  4. **HR 變異**：Yankee Stadium HR +12% + 9 mph Out To RF + Judge / Bellinger / Goldschmidt 對 LHP barrel% 高 → 1-2 支 HR 機率高於均值；同時 Pederson / Nimmo 對 RHP 也可拉短牆。**單支 HR 可能是分差關鍵，總分高低分歧主因**。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
