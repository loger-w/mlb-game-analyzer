## 投手對決

### Parker Messick (HOME, LHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +5.1
  - 同意 Elite Ace。ERA 2.40 / xERA 2.85 / FIP 2.95 / xFIP 2.80 各項一致頂級，gap 僅 +5.1（< 15 門檻）→ 結構性 elite，非運氣偏差。velo 88.8 算極低（左投 sub-90），靠 deception + command + balanced mix（FF 32% / CH 24% / SI 12%）撐起來。
- **Reverse platoon 信號**：dossier 訊號摘要未 fire（vs LHB OPS .500 / vs RHB OPS .577，Δ 僅 0.077 < 0.080 門檻，且 LHB BF 48 略低於 60 不算強樣本）
  - 未 fire → 不需擔心反向風險。對 LHB / RHB 都壓制（LHB .250/.250、RHB .198/.250/.327）。
- **對手打線威脅**：對 Twins 是 🔴 高威脅。Messick 近 3 場 ER/IP = 1/17.7（ERA ~0.51 火燙），對 Twins 核心 vs LHP 整體均勢（Buxton .575 / Wallner .545 / Lee .708 偏弱；只有 Bell .796 突出）。Twins 推進到第三輪才有機會（TTO3 penalty fire），但 K% TTO3 仍 26.7% → 教練很可能 5-6 IP 主動換掉，把問題交給 Clase。

### Connor Prielipp (AWAY, LHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter，gap vs ERA-only = —（樣本不足計算 gap）
  - 同意 Solid。ERA 3.86 / xERA 3.46 / FIP 3.89 / xFIP 3.89 一致，無 luck distortion；xERA 略低於 ERA → 結構稍好但僅 0.4 差距，仍歸 Solid。⚠️ GS 僅 3 場 → tier 估計信心 medium。barrel% 14.3% 明顯偏高（聯盟均 ~7-8%）→ 被擊中時擊球質量危險。
- **Reverse platoon 信號**：未 fire（樣本 LHB 11 BF 過小無法評估；vs RHB OPS .642 算一般）
  - 訊號摘要也沒列 → 不影響本場判讀。
- **對手打線威脅**：對 Guardians 是 🟡 中等威脅。Guardians vs LHP matchup tier 🟢 Weak（season Average 下修），核心打者 vs LHP 表現分裂：Ramírez 1.019 / DeLauter 1.101 / Rocchio .759 不錯，Kwan .459 / Martínez .643 弱。Prielipp 的 SL+FF 占 73% 偏 concentrated（max usage 39.7% 未觸發 single-pitch 門檻 45%），但 barrel% 14.3% 配上 Ramírez/DeLauter 任一發長打就會被改寫。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - 落差 = season Average → matchup Weak（**下修一檔**）。對 LHP 體質明顯較差，Kwan / Martínez 對左投是黑洞；但 Ramírez 反向（vs LHP 1.019 > season .735）是攻擊核心。本場 Guardians 攻擊集中度極高，靠 Ramírez / DeLauter 1-2 棒效率。
- **chain_break / heat_vs_babip 信號**：HOME chain_break #8-9（Δ 0.281 medium）fire；heat_vs_babip 未 fire（last7 BABIP .267 接近低端但未觸發 ≤ 0.260 門檻）
  - chain_break 影響後段（projected 7-8-9 棒）→ 壓制持續推進能力，第三輪後不易組織得分串聯，與 Messick TTO3 penalty 「正向窗口」相互抵銷一部分。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟡 Average
  - 同意 Average，**無上下修**。Twins 對 LHP 沒明顯偏好或弱點；個別來看 Bell .796 / Buxton .575（vs LHP 偏弱但 last7 1.047 火）→ 個別熱度與 splits 反向，半衰期短不穩定。
- **chain_break / heat_vs_babip 信號**：AWAY chain_break #8-9（Δ 0.301 **high**）fire；heat_vs_babip 未 fire（last7 BABIP .283）
  - chain_break high 比 HOME 還嚴重 → Twins 後段（projected 7-8-9 棒）落差更大，1-3 棒沒打到就熄火。對 Messick 這種第三輪才衰退的 starter，這是嚴重結構性問題。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.12 / 2 / **1**（Shawn Armstrong, Setup） | 5.81 / 6 / **1**（Cole Sands, High-leverage RP） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 4.12 中規中矩，**core IL ×1（Armstrong, Setup）→ 🟠 中高影響**。但 Clase（Closer，未列 IL）+ Cantillo / Hentges / Smith 仍可組成 9 局後段，深度尚可。Messick 若 5-6 IP 退場，Guardians 牛棚有 3-4 局負擔可吸收。對 Twins 末段威脅仍維持。
- AWAY 牛棚：ERA 5.81 **明顯偏弱**（聯盟均 ~3.85），且 **core IL ×1（Sands）→ 🟠 中高影響**。整體弱 + 核心 IL 是本場最大結構弱點。Prielipp 樣本小（GS 3）通常 4-5 IP 退場，Twins 中後段（5-9 局）要靠 5.81 ERA 牛棚撐 4-5 局 → 對 Cleveland 攻擊端是顯著正向 leverage（Guardians 1-3 棒 + 牛棚弱 → 後段串聯機會）。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🔴 HOME TTO3 penalty：OPS Δ +0.170（TTO1 0.451 → TTO3 0.621），第三輪明顯衰退；K% 從 33.3% 掉到 26.7%（Δ -6.6pp）
- 🟠 HOME chain breaks at #8-9：OPS 落差 0.281
- 🔴 AWAY chain breaks at #8-9：OPS 落差 0.301
- 🟠 ⏳ HOME 牛棚 core IL ×1：🟠 中高（後段防守變薄）
- 🟠 ⏳ AWAY 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - 雙重壓力判讀：**AWAY 端最危險**——Twins chain_break high (#8-9) + 牛棚 ERA 5.81 + core IL 三重結構問題疊加。Messick TTO3 penalty 高給 Twins 5-6 局窗口，但 Cleveland 教練很可能在 .621 OPS 區間前換 Clase，把窗口關掉。HOME 端 chain_break + core IL 是中度抵減，由 Ramírez/DeLauter 個別威脅補償。

## 條件修正

- Park Factor: 101.0 → +0.05 run（Progressive Field 中性偏微利攻；HR -9% 壓制全壘打）
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：先發 tier 落差顯著（Messick Elite vs Prielipp Solid）→ Messick 結構優勢已含於 base formula，無額外 ±run。無 doubleheader。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.3 | -0.2（HOME chain_break medium）+0.15（AWAY core_il + 牛棚 ERA 5.81 但已部分含於 base，取保守）= **-0.05** | **4.25** |
| AWAY | 3.3 | -0.3（AWAY chain_break high）+max(0.2, 0.1) interaction（HOME tto3_penalty + HOME core_il 同向取單側 max + 0.1）= **0** | **3.30** |
| Total | 7.6 | **-0.05** | **7.55** |

> 累積規則 check：
> - HOME 同側：負向 -0.2（chain_break）+ 正向 +0.15（牛棚反差）→ 反向獨立可加，淨 -0.05 ✅
> - AWAY 同側：負向 -0.3（chain_break）+ 正向 +0.3（tto3 + core_il interaction，取單側 max + 0.1）→ 反向獨立可加，淨 0 ✅
> - 單側 cap ±0.8 ✅

## 整體判斷

- **方向（基本面）**：**HOME（Guardians）微優**
- **總分（基本面）**：~7.6（formula 微下修至 7.55，量級上仍同 7-8 區間）
- **方向信心**：**60-65%**（Guardians 微優）
  - 依據：Messick Elite Ace 結構性壓制 Twins（K-BB% 21.8 / xFIP 2.80 / 近 3 場 ER 1）；Twins 牛棚 ERA 5.81 + core IL → 中後段對 Cleveland 是攻擊窗口；Cleveland vs LHP 雖 🟢 Weak 但 Ramírez/DeLauter 反向打 LHP（season OPS 1.019 / 1.101）構成個別爆發點
- **風險**：
  1. **Prielipp 樣本小 + barrel% 14.3% 偏高** → GS 3 場數據不穩，Ramírez/DeLauter 任一發長打可重寫局面
  2. **Messick TTO3 penalty high (Δ+0.170)** → 教練若讓其撐滿 6 IP，第三輪 Twins Buxton/Bell 在 .621 OPS 區間有反擊機會
  3. **⏳ Twins 牛棚 5.81 ERA 是結構弱點，但若 Duran/Pressly 等核心可用** → 後段不會崩，下修攻擊預期才合理
  4. **⏳ 雙隊 last7 BABIP 偏低**（HOME .267 / AWAY .283）→ 短期內任一隊 BABIP 反彈都可能讓總分偏 Over

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
