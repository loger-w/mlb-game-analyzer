## 投手對決

### Nick Lodolo (HOME, LHP, 28 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = —（**No 2026 pitching stats found**，無法計算 score）。dossier `Tier (script) = Unknown` 為已知資料缺口。
  - **判讀**：本季 GS 為 0（IL 復出首戰可能性高）；avg velo 88.6（career 92-93 mph）+ Statcast 樣本極小（whiff% 0.0、hard_hit% 100.0 等於單球噪音）→ 所有腳本派生欄位都是 **小樣本不可信**。**單一信號 single-pitch dependent (SI 50%) 與 TTO3 penalty 都標 career fallback / heuristic，引用時要降信心一級**。本場以 career baseline 評估：career ERA ~4.30 / xFIP ~3.80 健康時為 🟡 Solid Starter，但傷後復出首戰 → 降一檔 → 🟢 Back-end 至 ⚪ Below Avg 區間，實際視首局 stuff 與 control 而定。
- **Reverse platoon 信號**：未 fired（vs 兩側 BF 都不足，無法判讀）。但 LHP 的 platoon 結構意味 Yordan / Friedl / Cam Smith 等 LHB 應有自然優勢。
- **對手打線威脅**：Astros 整體 vs LHP 為 🟡 Average，但呈雙峰結構：(1) **頂端極度危險** — Yordan vs LHP OPS 1.025 + Walker .859 + Paredes .775 都優於季均；(2) **中後段空** — Cam Smith vs LHP 僅 .541，Altuve last7 .441（BABIP .238 偏低，可能反彈）。chain_break #2-3（Walker→Altuve OPS Δ 0.247）放大此問題。Lodolo 若控球失準或球速沒回到 92+ mph，Yordan-Walker-Paredes 三 RHB（Yordan 為 LHB）會直接咬重砲；若 stuff 還在，靠中段斷鏈仍可控。

### Mike Burrows (AWAY, RHP, 26 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟠 Strong Ace（xFIP p84, K-BB% p71），gap vs ERA-only = +67.6（觸發 Flag 8）。**部分同意**：xFIP 3.57 + K-BB% 13.6 + barrel% 6.8 + hard_hit% 23.1 都顯示底層投球 ≥ Solid Starter；但 ERA 5.97 與 FIP 4.98 仍偏高，FF（fastball）RV/100 −3.2 是核心弱點（usage 26%、xwOBA .449、hard-hit% 52%），SI −2.5 同樣被打。**真實水平介於 🟡 Solid Starter ~ 🟠 Strong Ace 之間**（接近 Solid Starter 偏上），不採 v2 上界。Flag 8 解讀：以運氣偏差為主（xFIP 3.57 / xwOBA .311 vs xBA .244 都遠優於 ERA），預期 ERA 將收斂；但 FF/SI 容易被打硬球是結構性，CH（25% usage、RV +0.5、whiff 37%、xwOBA .231）才是真正的 putaway 武器。
- **Reverse platoon 信號**：未 fired（vs LHB OPS 1.017 比 vs RHB OPS .763 高 0.254 是 **正向放大** 而非反向，符合 RHP 預期手別劣勢）。但 0.254 magnitude 屬極端 — vs LHB SLG .610、9 BB / 91 BF。
- **對手打線威脅**：Reds vs RHP 為 🟢 Weak（season tier），但需細看手別：Reds 主力含 **Elly De La Cruz（switch hitter，多以 LHB 對 RHP）+ TJ Friedl（LHB）+ Spencer Steer（RHB）+ Sal Stewart（RHB）**。LHB 側對 Burrows 是放大鏡，Elly vs RHP OPS .784 + Friedl vs RHP .587 都不算頂級但 Burrows 的 vs LHB .610 SLG 結構讓 LHB 即使不熱也能咬出長打。Reds 整體 last7 全線冷凍（Stewart .368、McLain .490、Steer 熱但 BABIP .368 lucky）→ 短期對沖 Burrows 弱點。chain_break #2-3（Stewart→McLain OPS Δ 0.207）削弱串聯。**淨判**：Burrows 真實壓制力 > Reds 當前手感，但 vs LHB 結構弱點 + GABP HR PF 129 → 若被 Elly / Friedl 一發打中就單局崩。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs RHP)**：🟢 Weak
  - **下修**：matchup tier vs season tier 已差一檔，且 Reds 最近 10 場 RS 3.80 / 連敗 7 場 + last7 Stewart .368 / McLain .490 / TJ .709 等核心都偏冷，雖 BABIP 多數偏低（Stewart .091 / McLain .133）含可能反彈成份，但短期對 Burrows 的壓制力較難轉換為 multi-run innings。**唯一上修點**：Burrows vs LHB 致命弱點（OPS 1.017）— 若 Reds 排出 Elly + Friedl 連線並使球在投打主場 GABP（HR PF 129）打到死角，可能單發逆轉趨勢。
- **chain_break / heat_vs_babip 信號**：chain_break #2-3 OPS Δ 0.207（Stewart .830 → McLain .623）→ 1-3 棒攻擊串聯被中段斷掉，Elly 上壘後若無人接力 SLG，long ball 變成 solo HR；多得分 inning 機率受壓。heat_vs_babip 未 fired（heat 未達極端 hot/cold 門檻）但個別打者 last7 BABIP 偏低（Stewart .091、McLain .133）有反彈空間，AI 不直接 ±run。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟡 projected（PA 排序近似 — 打線尚未公布）
- **Matchup tier (vs LHP)**：🟡 Average
  - **同意原評**：matchup 與 season tier 一致。但本場特殊性：對手 Lodolo 是「No 2026 stats」未知投手，使 vs LHP 既有評估失去基準；以 Lodolo career 健康時 ~ Solid Starter 對應，Astros vs LHP 三 RHB 主力（Walker / Altuve / Paredes）有名義 platoon 優勢。**結構性**：Yordan vs LHP OPS 1.025 是兩隊唯一可稱「elite vs handedness」的打者，他單人就能拉高得分上限。
- **chain_break / heat_vs_babip 信號**：chain_break #2-3 OPS Δ 0.247（Walker .950 → Altuve .703）→ 4-5 棒清壘段 Altuve 較弱，limit 多得分 inning。heat_vs_babip 未 fired，但 Walker last7 OPS 1.063 BABIP .444 / Paredes last7 .899 BABIP .444 都是 **lucky-hot**（明顯回歸風險，AI 不下修預測但需在風險段標）。Yordan 反向 last7 .633 BABIP .278（cold + 可能反彈）→ 三人組合 net 接近持平。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 4.39 / 5 / **2（Caleb Ferguson + Emilio Pagán，皆 IL15d，🔴 高）** | 6.29 / 8 / **1（Josh Hader IL60d，🟠 中高）** |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- **HOME 牛棚（Reds, ERA 4.39）**：絕對品質中等偏上但結構吃緊 — Ferguson + Pagán 雙缺直接打掉 setup + closer 兩端可用性，🔴 對應 §牛棚傷兵累計效應「明顯吃緊」。Reds 連敗 7 場意味近期被迫使用後段（無 save situation 但也吃 mop-up），可動人手疲勞風險中度。對 Astros 而言：若 Burrows 撐到 6-7 局正常下莊，Reds 牛棚需面對 Yordan-Walker-Paredes 第三 / 四輪 → 後段失分機率明顯偏高。
- **AWAY 牛棚（Astros, ERA 6.29）**：**ERA 6.29 是聯盟災難級，比 IL 影響更嚴重**。Hader IL60d 已長期缺席，球隊應已調整角色，但替補無人 step up 才使整體 ERA 站不住。對 Reds 而言：即使 Reds 打線當前偏冷，Astros 牛棚每局期望失分都顯著偏高，late-inning 翻盤窗口大。Burrows 若被 Reds vs LHB 的 weakness 早退（< 5 IP），Astros 牛棚要吃多局 → **本場 AWAY 後段失分風險為單一最大變數**。

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=+2.12):
  - **以運氣偏差為主，但帶結構性副作用**。xFIP 3.57 / xwOBA .311 / barrel% 6.8 / hard_hit% 23.1 全線顯示底層投球品質遠優於 ERA 5.97，預期 ERA 將朝 4.0~4.5 區間收斂。但 FF（fastball）RV/100 −3.2、xwOBA .449、hard-hit% 52% 是 **真正的結構性弱點** — Burrows 在 platoon-disadvantage 情境（vs LHB）特別吃虧（OPS 1.017）。本場 Reds 主力 LHB 群（Elly switch / Friedl / 偶爾 Stewart 等）整體 last7 偏冷可短期掩蓋此弱點，但若被打中一發（GABP HR PF 129）就破盤。**判斷不下修整體預測**，但 LHB 端風險獨立記。

### 額外信號
- 🟠 HOME single-pitch dependent：主球種使用率 50.0%（≥45.0%）
- 🟠 HOME TTO3 penalty：OPS Δ +-0.002（TTO1 0.643 → TTO3 0.641），第三輪明顯衰退；K% 從 31.6% 掉到 19.6%（Δ -12.0pp）（career fallback）
- 🟠 HOME chain breaks at #2-3：OPS 落差 0.207
- 🟠 AWAY chain breaks at #2-3：OPS 落差 0.247
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
- 🟠 ⏳ AWAY 牛棚 core IL ×1：🟠 中高（後段防守變薄）
  - **解讀**：Lodolo 兩個信號（single-pitch + TTO3 penalty）都帶 career fallback / 小樣本標籤，引用時降信心一級 — 不單獨支持「Astros 中後段必爆 Lodolo」的結論，僅作為「若 stuff 退化就快垮」的保險層。HOME 核心 IL ×2（🔴）是 **最強的 single signal**，疊加 Astros 打線 vs LHP 雙峰結構（Yordan-Walker-Paredes 強 / Smith-Altuve 弱），預測 Astros 後段（7-9 局面對 Reds B-team 牛棚）能進球的機率高。AWAY 核心 IL ×1（Hader IL60d）已長期 priced in，但 Astros 牛棚 ERA 6.29 是更大的隱憂。**Flag 8 + chain_break + core_il 沒有形成同向疊壓**：Flag 8 影響 Burrows 自身、chain_break 各自壓抑兩邊串聯、core_il 偏 AWAY 受惠 → 總體不互相放大。

## 條件修正

- Park Factor: 104.0 → +0.20 run
- 天氣：未公布（跳過天氣分析）
- 先發 tier / doubleheader：**非 doubleheader**。先發 tier 修正：Lodolo Unknown（career baseline ~ Solid Starter，但傷後復出 → 降一檔至 Back-end）→ 對 Astros 預期得分 **+0.1 ~ +0.2 run**（保守，因 Astros vs LHP tier 仍 Average）。Burrows 真實水平 Solid Starter+（v2 不取上界）→ 對 Reds 預期得分 **−0.1 run**（Reds vs RHP 已是 Weak tier，重複 cap）。**GABP HR PF 129** + Burrows FF 弱 + 雙方 power 打者（Yordan / Walker / Elly / Steer）→ HR 戰風險中度偏高，總分上限拉抬。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (Reds) | 6.0 | **−0.2（chain_break #2-3 中段）+0.1（AWAY core IL ×1，Hader 已 priced in 取低端）= −0.1** | **5.9** |
| AWAY (Astros) | 5.1 | **−0.2（chain_break #2-3 中段）+0.4（HOME core IL ×2 取高端，🔴 明顯吃緊）+0.0（TTO3 / single-pitch 都 career fallback、低信心 → 折入 IL 影響不再加）= +0.2** | **5.3** |
| Total | 11.1 | **+0.1** | **11.2** |

> 累積規則檢核：AWAY 同向多 signal（core_il + tto3）→ 取單側 max（core_il +0.4）+ 0.0（tto3 career fallback 不再加成）= +0.4，未超 cap ±0.8。Table A（Flag 8 / heat_vs_babip / strong_park）依紀律不入此欄，內容已放於風險提示與條件修正。

## 整體判斷

- **方向（基本面）**：**HOME（Reds）略佔上風**。Reds adjusted 5.9 > Astros adjusted 5.3，差距僅 0.6 run。主因：(1) Burrows ERA 高估真實水平（運氣偏差為主）但 vs LHB OPS 1.017 結構弱點 + GABP HR PF 129 → 在主場有單發風險；(2) Astros 牛棚 ERA 6.29 + Hader IL60d → Reds 後段翻分窗口大；(3) Lodolo 雖未知但 career 健康時優於 Burrows ERA-only 表現。**反向因子**：Reds 整體 last7 大冷凍（連敗 7 場、streak −7）+ vs RHP 為 🟢 Weak matchup tier，短期動能完全壓在 Astros 一側。
- **總分（基本面）**：**11.2（formula 11.1 + 信號 +0.1）**。GABP HR PF 129 + 雙方 power 打者 + 兩名先發（Burrows FF 弱、Lodolo 可能球速沒回） → 實際分數的 variance 偏大，11.2 為中位點，68% 區間估 9–13。
- **方向信心**：**52-55%**。差距僅 0.6 run 屬「持平偏 HOME」區間；Lodolo 完全黑盒（無 2026 stats）使 Reds 上修空間被資料缺口削弱，無法 confidently 推上 60%+。
- **風險**：
  1. **Lodolo 復出首戰風險**（最大）— 無 2026 MLB stats、Statcast 樣本極小、avg velo 88.6 vs career 92-93 mph。若球速 / 控球未到位，Astros 三 RHB 主力（Walker / Paredes / Yordan-LHB）可單局打爆；若 stuff 在線，本場優勢翻轉至 HOME。**頭兩局數據是關鍵 indicator**。
  2. **Burrows vs LHB 結構性弱點**（OPS 1.017 / SLG .610）疊加 GABP HR PF 129 — 即使 Reds 短期偏冷，Elly + Friedl 任一發大號 long ball 可吞掉 Reds 預期得分大半。
  3. **Astros 牛棚 ERA 6.29 是聯盟災難級** — 若 Burrows 因 vs LHB 弱點早退（< 5 IP），Astros 牛棚多吃局數的失分期望明顯偏高，本場 AWAY 後段失分風險為單一最大變數。
  4. **lucky-hot 回歸風險**（Walker last7 BABIP .444 / Paredes .444 / Steer .368）— 三名 RHB 中，若任兩人 BABIP 回歸至季均，Astros 預期得分再 −0.3 ~ −0.5。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組