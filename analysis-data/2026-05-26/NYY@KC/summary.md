## 投手對決

### Bailey Falter (HOME, LHP, 29 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = ⚪ Below Average（GS 1，本季僅 41 BF），gap vs ERA-only 因樣本太小無法計算
  - 同意 ⚪ Below Average 落點，但 ERA 9.82 vs xERA 4.80（Δ +5.02，Flag 8）運氣偏差成分大：13 BF vs LHB slash .636/.692/.727 是極端小樣本噪音。結構性問題在 K-BB% 0.0% 與 FF 60.7% 單一球種依賴；面對 Yankees 🔴 Elite (vs LHP) 打線，連 xERA 4.80 都偏樂觀，不自動下修但風險偏 ERA 那一側。
- **Reverse platoon 信號**：未 fired（vs LHB / vs RHB 樣本太小，平台分裂不可靠）
  - 不適用；惟需注意 LHP 對 Volpe vs LHP OPS 1.171、Goldschmidt vs LHP 1.183、Bellinger vs LHP .896 是切實的左投剋星組合。
- **對手打線威脅**：極高。Yankees vs LHP 整體 🔴 Elite（chain OBP top3 .357、SLG mid .554），1-5 棒 Judge .984 / Bellinger .896 / Goldschmidt 1.183 / Rice .914 全數 OPS ≥ .896 vs LHP；single-pitch FF 60.7% 在 EV95% 47-55 強打前極易被鎖定。

### Cam Schlittler (AWAY, RHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +1.1
  - 同意 🔴 Elite Ace。ERA 1.50 vs xERA 2.59（Δ -1.09，Flag 8 未觸發 1.5 門檻）雖有輕度好運成分，但 K-BB% 24.5%、velo 95.6 / max 101.3、whiff 14.1%、WHIP 0.86 是全方位 Elite 證據；FF/FC/SI 三球種 RV/100 全正且均衡（44/28/18 使用率），結構性扎實。tier_v2 +1.1 為輕度上修，不下修預測。
- **Reverse platoon 信號**：未 fired（vs LHB .211/.258/.289 與 vs RHB .146/.180/.188 差異在預期方向內）
  - 不適用；正常 RHP 平台優勢仍在，Royals 9 人陣容中僅 Pasquantino（1B）為 LHB，平台壓制效果不被稀釋。
- **對手打線威脅**：低。Royals 整體 🟢 Weak (vs RHP)，xwOBA 0.316、chain OBP top3 .323；威脅集中在 Witt vs RHP .804、Caglianone vs RHP .765 兩點，但 last7 OPS 分別 .655 / .539 皆在冷期。Schlittler 球速 + 三球種均衡，預期能壓制至 2-3 分區間。

## 打線評級

### HOME — season tier 🟡 Average / heat 🥶 Cold
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak
  - 下修。Matchup vs RHP（🟢 Weak）比 season（🟡 Average）再差一檔，1-9 棒 vs RHP OPS 多數 < .770，再遇到 Schlittler Elite RHP，本場攻擊期望下修。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #2-3（相鄰 OPS 落差 0.219）切斷 Witt 上壘後的清壘串聯，壓縮 big-inning 機率（-0.1 ~ -0.3 run，pick -0.2）。heat_vs_babip ⏳ unlucky-cold（last7 BABIP 0.259）→ 冷期有反彈可能，但對手 Elite RHP 壓制力使反彈窗口窄；Flag 3 紀律不自動 ±run，敘事抵銷 chain_break 部分壓制 → 最終取 -0.2。

### AWAY — season tier 🟠 Strong / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs LHP)**：🔴 Elite
  - 上修。Matchup vs LHP（🔴 Elite）比 season（🟠 Strong）再升一檔，1-5 棒 vs LHP 全部 OPS ≥ .896，遇到 Below Average LHP Falter（single-pitch FF 60.7%），本場攻擊期望顯著上修。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - chain_break #5-6（Rice .914 → Rosario .768，OPS 落差 0.223）切在 cleanup 之後，影響相對輕，因為 1-5 棒已完成最重得分產出（OBP top3 .357、SLG mid .554）；對總分壓制有限，pick -0.1 run。heat_vs_babip 未 fired（last7 BABIP 0.270 正常範圍）。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.67 / 6 / 2（Estévez Closer + Strahm Setup, 🔴 高） | 3.5 / 3 / 0（🟢 完整） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：ERA 4.67 已偏弱（聯盟均 ~4.0），加上 Closer（Estévez）+ Setup（Strahm）雙核心 IL15d → §牛棚累計效應「2 名核心 = 🔴 高（牛棚明顯吃緊）」。Falter 預計 4-5 IP 後就需早期接力，等於把第 5-6 局起的高槓桿時段交給 mop-up / 低槓桿替補去面對 Yankees Elite vs LHP 打線（注意換投後接的可能是 RHP，Yankees vs RHP 仍 OPS 高位）。對 AWAY 末段得分威脅 +0.4 run（§量級錨點 Table B：core_il_count 2 名 → +0.2 ~ +0.5，取偏上界）。
- AWAY 牛棚：ERA 3.50 屬聯盟前段，0 核心 IL 完整可用。Schlittler 是 Elite Ace 預期能撐 6-7 IP（HR 偏少 + WHIP 0.86），牛棚負擔輕；末段對 Royals 🟡 Average / 🥶 Cold 打線威脅充足。AWAY 牛棚為本場 AWAY 側「跑分護城河」。

## 風險提示

- ⚠️ HOME 投手 Flag 8 (era_xera_delta=+5.02):
  - 運氣偏差為主要成因（GS 1 共 41 BF 是極端小樣本，BABIP 必偏高；13 BF vs LHB slash .636/.692/.727 不具代表性）。但結構性弱點同時存在：K-BB% 0.0%、FF 60.7% 單一球種、velo 89.0 偏低。本場面對 Yankees vs LHP 🔴 Elite 打線（1-5 棒 OPS vs LHP 多 ≥ .900）即使 xERA 4.80 仍偏樂觀，預期 5 IP 內失 4-5 分屬合理範圍。Flag 8 紀律不自動下修，預測值維持腳本 base 2.1（HOME 攻擊不調整）+ AWAY 攻擊側信號加成。
- ⚠️ HOME 打線 Flag 3 (last7 BABIP=0.259):
  - 偏低 BABIP 反映冷期可能含運氣成分，理論上有回歸動能；但本場面對 Elite RHP Schlittler（被打 BABIP 對手難以拉高），反彈窗口受限。Flag 3 紀律不自動 ±run value，敘事處理為「冷期可能持續到本場」，HOME 打線預測不上修。

### 額外信號
- 🟠 HOME single-pitch dependent：主球種使用率 60.7%（≥45.0%）
- 🔴 AWAY TTO3 penalty：OPS Δ +0.222（TTO1 0.443 → TTO3 0.665），第三輪明顯衰退；K% 從 32.4% 掉到 14.5%（Δ -17.9pp）
- 🟠 HOME chain breaks at #2-3：OPS 落差 0.219
- 🟠 AWAY chain breaks at #5-6：OPS 落差 0.223
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - 本場直接受影響：Falter 預估 4-5 IP 後牛棚要扛 4-5 局，但 Closer + Setup 都缺，剩下的低槓桿手須去面對 Yankees 1-5 棒。與 Flag 8（先發弱）形成「先發塌→牛棚也撐不住」雙重失血，是 AWAY 跑分能拉開的核心結構性原因。額外信號中還疊加 single-pitch FF 60.7%（pitch_mix_concentration）放大平台優勢，HOME 守備側壓力極大。

## 條件修正

- Park Factor: 106.0 → +0.30 run
- 天氣：Partly Cloudy, 83°F, wind 8 mph, In From RF
  - 影響判讀：83°F 屬「> 85°F 球易飛」邊緣，輕度利攻；風 8 mph In From RF 屬「< 8-15 mph 輕度逆風」，對右打 Judge / Goldschmidt / Rice 拉打到 LF 區無壓制（風 In From RF 主壓左打的反向拉打到 RF，影響有限）。Kauffman 本身 HR -9% 已壓制 HR，但 Runs PF 106 補回安打與三壘打增加。淨效果：HR 微壓、Runs 微利，與 PF 106 倍率一致，不重複加 ±run。
- 先發 tier / doubleheader：先發 tier 嚴重落差（Elite Ace vs Below Average），非 doubleheader（系列 G2）。Schlittler 預期 6+ IP 高效，Falter 預期 4-5 IP 早洗；對 inning workload 落差具長期意義，已在牛棚段反映。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 2.1 | 0.0 (TTO3 +0.2 − chain_break -0.2) | 2.1 |
| AWAY | 7.2 | +0.4 (max(pitch_mix +0.3, core_il +0.4) + 0.1 interaction − chain_break -0.1) | 7.6 |
| Total | 9.3 | +0.4 | 9.7 |

## 整體判斷

- **方向（基本面）**：AWAY（Yankees）— adjusted 7.6 vs 2.1（gap 5.5 run，遠超 0.5 持平門檻）
- **總分（基本面）**：9.7（adjusted Total）
- **方向信心**：70% — 結構性 Elite vs Below Average 投手落差 + AWAY 牛棚 0 IL vs HOME 牛棚 2 核心 IL + AWAY 打線 vs LHP 平台優勢 三重共振；未達 75% 是因為 Falter 樣本小（GS 1）與 Royals 主場 PF 106 + Kauffman 利安打的條件殘留壓縮空間。
- **風險**：
  1. **Falter 樣本噪音**：GS 1（41 BF）的 ERA 9.82 帶極端 BABIP 失真，xERA 4.80 才是 reasonable 基準；若實際 IP 拉到 5-6 局且控制住長打，HOME 失分可壓到 3-4 而非 5+。
  2. **HOME 打線冷期反彈**：HOME last7 BABIP 0.259 在 Kauffman（HR -9% 但利安打 +6%）的環境，若 Witt / Pasquantino 任一棒突破 Schlittler，HOME 攻擊有 base 之上 +0.5-1 run 的非線性可能。
  3. **Schlittler TTO3 衰退**：dossier TTO3 OPS 0.665（K% 從 32.4% 掉到 14.5%）若教練第三輪硬留，HOME 7-9 局有機會吃分；但 Royals 末段打線（Loftin / Isbel OPS .693-.705）轉換得分能力有限，攻擊兌現率受打線深度拖累。
  4. **風（In From RF）+ Yankees HR 依賴**：Yankees vs LHP 多靠 Judge / Goldschmidt / Rice 長打，逆風進入 RF 對 Bellinger（LHB 拉打）有輕度壓制；若 Yankees 攻擊轉小球串聯而 chain_break #5-6 fired，AWAY adjusted 7.6 可能跌到 6.5-7.0 區間。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組