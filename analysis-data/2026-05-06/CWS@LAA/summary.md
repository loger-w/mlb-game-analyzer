## 投手對決

### Walbert Ureña (HOME, RHP, 22 📈 成長期)
- **Tier 驗證**：腳本 tier = 🟡 Solid Starter
  - AI 判讀：偏弱端 Solid。ERA 3.86 / xERA 3.40 gap 0.46（不觸 Flag 8），但 K-BB% 5.0 + WHIP 1.84 結構性偏 Back-end，速球品質（avg 94.6 / max 100.9 / FF RV +1.4）+ 球種 balanced（SI 32 / CH 28 / FF 25）撐到 Solid 下緣。**不下修**，但不可當 Solid 中位數看。
- **Reverse platoon 信號**：🔴 fired Δ +0.305（vs RHB .850 > vs LHB .545，sample BF 45 / 34）
  - AI 判讀：CWS 線陣中 RHB 並非主力（Antonacci, Vargas, Adell-style 多為左打混雜），但 #1 Antonacci .951 vs RHP / #2 Murakami .964 vs RHP 都壓制 RHP，整體效應仍指向 CWS 多得分 — Reverse platoon 訊號與 vs RHP 對位優勢同向疊加（不重複加 run，取單側上界）。
- **對手打線威脅**：高。CWS season tier 🔴 Elite + 🔥 Hot vs RHP，前 4 棒 vs RHP OPS .951 / .964 / .645 / .770，#9 Romo vs RHP 1.969（24 BF）有限樣本但極熱。Ureña 速球頂級但控球（K-BB% 5.0）與壓制（WHIP 1.84）跟不上 → 預期被掛大量上壘 + 中段轟擊。

### Noah Schultz (AWAY, LHP, 22 📈 成長期)
- **Tier 驗證**：腳本 tier = 🟠 Strong Ace
  - AI 判讀：略樂觀，實際接近 Solid+ / Strong Ace 邊緣。ERA 2.53 vs xERA 3.38 gap 0.85（未到 Flag 8 1.5 門檻但偏大）→ 走 Flag 8 紀律敘事判讀：「ERA 領先 xERA 0.85 暗示有運氣成份，後續可能向 mid-3 區回歸」，**不自動下修預測**。結構性數據支持：whiff 8.9 / hard_hit 21.7 / barrel 3.8 / FF RV +4.4 / K-BB% 9.4（穩） → 真實水平 ≈ Solid+。
- **Reverse platoon 信號**：未 fired（vs LHB 15 BF < 30 不足樣本，vs RHB 70 BF .575 OPS 屬正常 LHP 對 RHB 表現）
- **對手打線威脅**：中偏低。LAA 打線 season tier 🟡 Average vs LHP matchup tier 🟢 Weak → **下修**。前 3 棒 Neto / Trout / Adell vs LHP OPS .700 / 1.033 / .957（Trout & Adell 是真威脅），但 #4-9 落差大（Schanuel vs LHP .488 / Soler .699 / Ward .——）+ chain break #2-3 high (Δ .342) → 即便 Trout 上壘也缺接續。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布；最終打序公布前評估保留 ±0.2 run noise）
- **Matchup tier (vs LHP)**：🟢 Weak
  - 落差 Average → Weak（一檔下修）。AI 判讀：Schultz 為 LHP，雖然 Trout vs LHP 1.033 / Adell vs LHP .957 是真威脅（前段集中），但 #4-9 多為對 LHP 偏弱（Schanuel .488、Soler .699）。**整體下修**，但不極端 — 因為 Trout / Adell 兩棒可獨自破局。
- **chain_break / heat_vs_babip 信號**：🔴 chain_break #2-3 OPS Δ 0.342 high（Trout .999 → Adell .657 落差，但 vs LHP Adell .957 修復一部分）
  - 影響：1-2 棒 Neto-Trout 上壘後若 Adell 卡住 → 4-5 棒 Schanuel/Soler 也是中段，導致 Trout 上壘後缺有效跟手 → 壓抑大局得分上限
  - heat_vs_babip 未 fired（last7 BABIP .322 中性）

### AWAY — season tier 🔴 Elite / heat 🔥 Hot
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🔴 Elite
  - 落差：Elite = Elite（無落差，**維持**）。前 4 棒 vs RHP OPS .951 / .964 / .645 / .770 真強，#9 Romo 1.969（小樣本噪音先打折）。Ureña reverse platoon 同向放大（不重複加 run）。
- **chain_break / heat_vs_babip 信號**：🟠 chain_break #2-3 OPS Δ 0.163 medium（Murakami .961 → Vargas .798）
  - 影響：核心 1-3 段尚 OK，Murakami 後傳 Vargas 雖落 .163 但 Vargas .798 仍中段水準，不致壓制 → medium 下緣修正，影響有限
  - heat_vs_babip 未 fired（last7 BABIP .321 中性）— 🔥 Hot 不含明顯運氣成份，可信度高

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 5.35 / 6 / 0 名核心 | 4.61 / 6 / 1 名核心（Vasil） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（LAA）：ERA 5.35 屬聯盟下游（bottom 5 級），core IL 0 → core 完整但整體品質差。對 CWS 末段威脅低 — 即使 Schultz 退場，CWS 中後段（Benintendi / Kelenic / Peters 等左打）仍可能對 LAA 牛棚連續上壘，**LAA 末段防線是本場主要漏點**。
- AWAY 牛棚（CWS）：ERA 4.61 中等，core IL 1（Vasil）屬 🟠 中高（後段防守變薄）→ 🟠 對 LAA 末段壓力可能下修；但 LAA 線陣 #4-9 對 LHP 弱，即便由 RHP 接替，Trout 與 Adell 之外的威脅仍有限。**對總分判讀微偏 LAA +0.1 run**。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🔴 HOME reverse platoon Δ +0.305（vs RHB OPS 0.850 > vs LHB OPS 0.545）— RHP 對非預期手別反而吃虧
- 🔴 HOME chain breaks at #2-3：OPS 落差 0.342
- 🟠 AWAY chain breaks at #2-3：OPS 落差 0.163
- 🟠 ⏳ AWAY 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - 半衰期 ⏳ short：last7 異動，留意對手 last-minute call-up。本場影響有限 — LAA 線陣 #4-9 對 LHP 偏弱，即使 CWS 末段交給次級 RP，LAA 中後段火力本身不足以放大此漏洞。**Flag 3/8 無雙重壓力**（兩邊 BABIP 中性、Schultz ERA-xERA 0.85 但未到 Flag 8 1.5 門檻）→ 純結構訊號，影響範圍 +0.1 run / 對 LAA。

## 條件修正

- Park Factor: 101.0 → +0.05 run（中性偏微利攻；HR PF +5%）
- 天氣：Sunny, 71°F, wind 8 mph, Out To LF
  - 影響判讀：71°F 中性溫度（無修正）；wind 8 mph Out To LF 屬「輕度順風往左外野」門檻邊緣 — LHB 拉打方向利好（CWS 線陣 Murakami / Benintendi / Kelenic 為左打、LAA 的 Trout 也是左打）。整體 +0.05 ~ +0.1 run total，HR 機率輕度上修，未到 summary 風險段必提門檻（>20mph）。
- 先發 tier / doubleheader：非 doubleheader（單場系列 G2）。Schultz GS 4 / Ureña GS 3 — 兩位都在初登先發前 5 場區間，**TTO3 樣本不足**（無 tto3_penalty fire），AI 不額外修正。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.8 | −0.3（chain_break high）+0.1（CWS core_il_count）+0.05（wind LF）= −0.15 | 3.6 |
| AWAY | 5.3 | +0.3（reverse_platoon high，與 vs RHP 對位優勢同向，取單側上界）−0.1（chain_break medium 下緣）+0.05（wind LF）= +0.25 | 5.5 |
| Total | 9.1 | 信號淨 +0.1 | 9.2 |

> 紀律備註：
> - Schultz ERA-xERA gap 0.85 與 Ureña tier 偏弱端皆未入 ±run（Flag 8 紀律：敘事處理）→ 反映於信心降階，不入此欄
> - reverse_platoon 與 vs RHP 對位優勢同向 → 取單側 max +0.3 而非區間相加（Table B 累積規則）
> - 單側修正皆在 ±0.8 cap 內

## 整體判斷

- **方向（基本面）**：AWAY (CWS) 偏好（5.5 vs 3.6，差距約 1.9 run）
- **總分（基本面）**：9.2（base 9.1 + 信號淨 +0.1，wind LF 微利攻）
- **方向信心**：60%（中等偏高）— 不到 75% 因 Schultz ERA-xERA gap 0.85 暗示真實水平偏 Solid+ 而非 Strong Ace（敘事降階），且 LAA 前 3 棒 Trout / Adell vs LHP 仍能單棒破局
- **風險**：
  1. **Schultz 回歸風險**：ERA 2.53 vs xERA 3.38 (gap 0.85) → 本季前 3 場可能含運氣成份；若退化到 mid-3 ERA 區，AWAY base 可能下修 0.3-0.5 run，將 fundamental gap 從 1.9 收斂到 1.4-1.6 run
  2. **Trout / Adell 單棒破局風險**：vs LHP OPS 1.033 / .957，即便 LAA 整體對 LHP weak，前段仍可能對 Schultz 製造 2-3 run
  3. **LAA 牛棚雷區 + wind**：BP ERA 5.35（聯盟下游）+ 8mph Out To LF → CWS 中後段（Benintendi / Kelenic 左打）+ Murakami 拉打方向利好，HR 機率輕度上修
  4. **CWS 連敗 streak −1 / 系列 0-1 落後**：本場若再丟，CWS 連敗壓力升級；short-half-life 心理因素在 model 之外，AI 不量化但提示

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組