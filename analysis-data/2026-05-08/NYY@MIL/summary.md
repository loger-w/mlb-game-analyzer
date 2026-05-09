## 投手對決

### Jacob Misiorowski (HOME, RHP, 24 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = 🔴 Elite Ace（xFIP p95, K-BB% p95），gap vs ERA-only = +13.4
  - 同意 Elite Ace tier。gap 落在 +13.4（< 15 門檻、未觸 tier_mismatch high），但結構性指標一致偏強：xFIP 2.14 / K-BB% 28.4 / barrel 僅 3.9% / velo 96.3 avg、max **103 mph**。ERA 2.84 看似偏高是被 BABIP / 序列性失分推高，並非真實水平劣化。年僅 24 屬成長期，stuff 是這場最大不確定性。
- **Reverse platoon 信號**：未 fire（vs LHB .147/.278/.253 vs vs RHB .203/.277/.305，OPS Δ ≈ +0.052 < 門檻 .080）
  - 投手手別預期方向正常（RHP 略壓 RHB），無放大效應。但 vs LHB BF=90 vs vs RHB BF=65，樣本顯示他 reverse-ish 還是夠壓 RHB，NYY 多名右打主力（Judge / Caballero）並沒 platoon 漏洞可吃。
- **對手打線威脅**：NYY top 5 vs RHP 整體強（Judge 1.039 / Bellinger .885 vs RHP），但 Misiorowski stuff 是 elite 等級，single-pitch FF 61.2% 對重砲打線屬雙面刃 — 第一輪壓制機率高，TTO3 K% 從 43.1% 掉到 25.8% 是真風險點（Δ -17.3pp 樣本 31 BF），第三輪後 NYY 火力可能集中。

### Max Fried (AWAY, LHP, 32 📉 初期退化)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p76, K-BB% p73），gap vs ERA-only = -15.1
  - **|gap| ≥ 15，tier_mismatch (high) fired**。ERA 2.39 / FIP 2.57 vs **xFIP 3.75** 結構性落差大、K-BB% 僅 13.9（對 ace 偏低）、velo 88 avg 也是控球派常見的 stuff 衰退指標。但 barrel 2.1% / hard_hit 22.1% 極壓制 → 結構性風險（K-BB%）+ 表象運氣（少 hard contact）並存。**敘事面下修，但不自動扣 run（Flag 8 紀律）**。32 歲 LHP 初期退化，近 3 場 3 ER / 20 IP 反而是巔峰狀態，今晚會是分歧點。
- **Reverse platoon 信號**：未 fire（vs LHB .156/.188/.200 (48 BF) vs vs RHB .182/.263/.241 (153 BF)，OPS Δ vs 預期方向一致）
  - LHP 對 LHB 顯著壓制，符合 platoon 預期。MIL 打線中 Turang / Frelick / Bauers 為左打，正面被吃對位。
- **對手打線威脅**：MIL season 🟡 Average / vs LHP 🟢 Weak — top 5 中 Turang (.956 → .700) / Bauers (.778 → .379) / Frelick (.636 → .343) 三人 vs LHP 顯著縮水，Mitchell vs LHP .921 是少數反例。Fried 球種 balanced (SI/FC/FF 三球種 17.7-22.6%) 難對位、近期狀態頂級 → 本場 MIL 攻擊很難起串聯。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟢 Weak
  - matchup tier 比 season tier 又**降一檔**（Average → Weak）。落差來自 Turang / Bauers / Frelick 三名核心 vs LHP 大幅縮水（OPS 各掉 .250+），對 Fried 這種 balanced 4 球種 LHP 形成正面下修方向。
- **chain_break #1-2 (medium, OPS 落差 0.192)**
  - 影響打序開頭：Turang vs LHP 縮到 .700、Contreras 維持 .715，第一輪 setup → mid 串聯被切斷一次。MIL 全場上壘大概率倚賴 Mitchell（vs LHP .921）的單棒爆發、而非串聯，總 RS 上限受壓制。

### AWAY — season tier 🟠 Strong / heat 🔥 Hot
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟠 Strong
  - matchup tier 與 season tier **同檔（Strong）**，貼合 season 表現，無需上下修方向。但要注意 last7 BABIP 0.340（偏高邊緣，未觸 ≥ 0.350 fire 線），Bellinger last7 OPS 1.748 / BABIP .522、Judge last7 OPS 1.216 / BABIP .462 個人層面屬極端運氣熱（單人不觸 lineup-level signal，但敘事需提）。
- **chain_break #6-7 (high, OPS 落差 0.568)**
  - 影響後段：1-5 棒（Judge / Bellinger / Chisholm / Grisham / Caballero）強，#6-9 大幅落差，Misiorowski 若能撐到 6-7 局，下半段打線本身就會自動降檔。但若 NYY 提前打進 Misiorowski（TTO3 K% 大跌風險）、把比賽推進 MIL 牛棚（core IL ×2）→ chain_break 不再是壓制因素。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.67 / 5 / **2** (Closer Zerpa + Setup Koenig 都 IL15d) | 3.16 / 4 / **0** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（MIL）：ERA 3.67 中段水準，但 **core IL ×2 (Closer Zerpa + Setup Koenig 都 IL15d)** 直接觸發 🔴 高（牛棚明顯吃緊）。剩餘可用人選 leverage tier 整體下滑，第 7 局後對 NYY hot lineup（Judge / Bellinger 火力期）守不住的風險明顯放大。Misiorowski TTO3 K% 大跌的隱性訊號 + 牛棚薄 → 同向疊加，是本場 MIL 防守端最大裂縫。
- AWAY 牛棚（NYY）：ERA 3.16 偏好、core IL = 0，可用性正常。MIL 攻擊本來就 vs LHP 弱、chain 被 #1-2 切斷，後段對上 NYY 健康牛棚很難翻盤。

## 風險提示

Flag 3/8 無觸發；額外信號如下：

### 額外信號
- 🟠 HOME single-pitch dependent：主球種使用率 61.2%（≥45.0%）
- 🟠 HOME TTO3 penalty：OPS Δ +0.046（TTO1 0.520 → TTO3 0.566），第三輪明顯衰退；K% 從 43.1% 掉到 25.8%（Δ -17.3pp）
- ℹ️ AWAY balanced 4+ pitches：最高球種僅 22.6%（<25.0%）
- 🟠 AWAY TTO3 penalty：OPS Δ +-0.079（TTO1 0.514 → TTO3 0.435），第三輪明顯衰退；K% 從 33.3% 掉到 13.5%（Δ -19.8pp）
- 🟠 HOME chain breaks at #1-2：OPS 落差 0.192
- 🔴 AWAY chain breaks at #6-7：OPS 落差 0.568
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - 本場直接受此影響：MIL 失去 Closer + Setup 雙核，遇到 NYY 火力期打線（Judge / Bellinger last7 OPS 雙超 1.200）的機率高 → 後段失分風險 ↑。同時 Fried tier_mismatch (high, gap -15.1) 是 Flag 8 結構性下修方向、Misiorowski TTO3 K% 大跌 fire 同向 → 三因素疊加都把總分往多的方向推。⏳ 短半衰期，open 後若 lineup 公布 Mitchell 高棒、Bauers / Frelick 替換為強 vs LHP 反例則部分緩解。

## 條件修正

- Park Factor: 97.0 → -0.15 run（American Family Field 微壓 runs，但 HR +11% 對重砲 NYY 仍有利）
- 天氣：未公布（跳過天氣分析）— American Family Field 為可關閉式屋頂球場，若關頂則 wind/temp 失效。
- 先發 tier：HOME 🔴 Elite Ace（Misiorowski） vs AWAY 🟠 Strong Ace（Fried）— **HOME 投手帳面占優**，但 Fried 結構性紅燈 + 32 歲 LHP 退化邊緣，與 Misiorowski 的成長期 elite stuff 形成 tier 名次與真實壓制力之間的張力。非 doubleheader。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 2.8 | **-0.2**（chain_break #1-2 medium，#1-2 串聯被切） | **2.6** |
| AWAY | 3.0 | **+0.3**（核心：tto3+core_il 同向取 max 區間 +0.4 +0.1 interaction = +0.5；+ single-pitch +0.1；- chain_break #6-7 high -0.3） | **3.3** |
| Total | 5.8 | **+0.1** | **5.9** |

> 敘事側不入此欄但對方向有壓力的因素：
> - **Fried tier_mismatch (high, gap -15.1)** ⛔ Table A — xFIP 3.75 vs FIP 2.57 結構性紅燈、K-BB% 13.9 偏低；對 MIL 攻擊偏正向（Fried 真實水平比帳面差），但 Flag 8 紀律不自動 +run。
> - **Fried tto3_penalty (signal fire 但 OPS Δ 反而 -0.079)** — K% 從 33.3% 掉到 13.5% 是真，但 OPS 不升反降說明 contact 後品質沒提升；屬於異常觸發，不入 +run。
> - **Bellinger / Judge last7 BABIP 極端 (.522 / .462)** — 個人層級不觸 lineup-level signal，但 NYY +0.3 的可信度上限受此壓制（若回歸發生，AWAY adjusted 可能停在 +0.1 ~ +0.2）。

## 整體判斷

- **方向（基本面）**：**AWAY (NYY)**
- **總分（基本面）**：**~5.9**（HOME 2.6 / AWAY 3.3）— 略低於聯盟均，反映雙方先發都屬 ace 等級，總分上限被投手對決壓住。
- **方向信心**：**60-65%**。NYY 占優依據：戰績雙方差距明顯（NYY 26-12 .684 vs MIL 19-18 .514）、近 10 場 RA 雙方都低（NYY 2.90 / MIL 2.20）但 NYY RS 6.30 火力更頂、對位上 NYY vs RHP 強 + MIL vs LHP 弱、MIL 牛棚 core IL ×2 是結構性裂縫。下不到 70%+ 的原因：Misiorowski elite stuff (103 mph max) + Fried xFIP 結構紅燈，雙向變數還沒 collapse 成單一方向。
- **風險**：
  1. **Misiorowski stuff 是 X-factor**：24 歲成長期 + max 103 mph + xFIP 2.14 / barrel 3.9% 結構性極壓，NYY 第一輪容易被壓制 → 若 Misiorowski 撐到 6 局以上、MIL 牛棚薄不需上場，NYY 打線優勢窗口會大幅縮小。
  2. **NYY 火力的 BABIP 回歸風險**：Bellinger last7 OPS 1.748 / BABIP .522、Judge last7 OPS 1.216 / BABIP .462 都是極端運氣值，本場若同步回歸 → AWAY adjusted 可能掉到 3.0 ~ 3.1，總分壓向 5.6 ~ 5.7。
  3. **Fried 結構性下行 vs 近期巔峰的拉扯**：xFIP 3.75 / K-BB% 13.9 警告本季 ERA 2.39 含運氣，但近 3 場 3 ER / 20 IP 是頂級狀態。今晚他若狀態繼續 → MIL 攻擊根本起不來、總分壓到 5.5 以下；狀態突然回歸 xFIP → MIL 把比賽拖進中後段、總分破 6.5+ 機率上升。
  4. **MIL 牛棚 core IL ×2 (⏳ 短半衰期)**：是本場「總分偏多」的主要敘事支撐點。若 open 後球隊 transactions 補回任一核心、或啟用替補 RP 撐出意外效率 → 此風險點失效，AWAY 的 +0.4 牛棚加分要拉回。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組