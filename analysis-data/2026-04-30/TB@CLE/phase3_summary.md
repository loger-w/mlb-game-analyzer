## 投手對決

### Gavin Williams (HOME, RHP, 26 歲 ⚡ 巔峰期)
- **Tier 覆寫**：沿用腳本 🟡 Solid Starter（偏 Solid 下緣、有風險）
- 真實水平判斷：ERA 3.28 但 xERA 4.09 / FIP 4.42 / xFIP 3.21 — ERA 受運氣加持。`barrel% 16.5`（高）+ `hard_hit% 28.0` 顯示被擊球品質偏差，但 `K-BB% 17.6` + `WHIP 1.09` 維持中上水準。近 3 場 4 ER / 17.7 IP（2.03 ERA）走勢佳。FF 90.7 平均（max 99.5 — 有頂級球速但平均壓低，可能配球理由），ST + CU 形成常規組合。
- 對手打線威脅：vs LHB `.153/.247/.375`（壓制 BA 但 SLG 偏高 = 偶發長打風險）、vs RHB `.176/.311/.333`（OBP 高 = 對 RHB 易給保送）。TB 主力 Yandy / Aranda（LHB） / Caminero 都在熱手期，適合打 Williams 的 SLG 漏洞。

### Drew Rasmussen (AWAY, RHP, 30 歲 📉 初期退化)
- **Tier 覆寫**：沿用腳本 🔴 Elite Ace（保留級別，但加註 FIP 警示）
- 真實水平判斷：ERA 2.45 / xERA 2.64（一致）、`WHIP 0.74`（極端低，限制上壘是 elite 核心）、`K-BB% 22.4` Elite 區間。但 `FIP 4.07` vs ERA 2.45 落差 1.62（HR 抑制可能含運氣，`xFIP 2.76` 校正回真實水準）。Statcast `whiff% 9.8 / hard_hit% 23.0 / barrel% 10.3` 全屬中低風險區間，配球以 FC 35.2 / FF 27.0 / SI 20.5（cutter 主力）切換手感。近 3 場 2 ER / 16.0 IP（1.13 ERA）狀態高峰。
- 對手打線威脅：vs LHB `.203/.254/.424`（小漏洞 — SLG .424 偏高，少數長打風險）、vs RHB `.086/.086/.200`（35 BF 小樣本但壓制力恐怖，連保送都不給）。CLE 主力 José Ramírez (S)、Kwan (L)、DeLauter (L) 都從左打席切，可吃到 Rasmussen 對 LHB 的偏差，但 .424 SLG 仍只是「相對」漏洞。

## 打線評級

### HOME (Cleveland Guardians) — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用 🟡 Average，但「上重下輕」加註。xwOBA .328 / OPS .697 偏均值；近 10 RS 3.70（冷）；連敗 -4。Top 5：J. Ramírez 季 OPS .777 但 last7 .531（冷），Kwan .590 全季偏弱，DeLauter .810 / last7 .810（穩），Rocchio .735 / last7 .400（冷），Á. Martínez .827 / last7 1.080（極熱）。整體核心多人冷期，僅 DeLauter 與 Á. Martínez 有狀態，再面對 Rasmussen 的 vs RHB 0.86 OPS 壓制 — 預期得分壓得很低。

### AWAY (Tampa Bay Rays) — 🟢 Weak / ⚖️ Normal
- **Tier 覆寫**：腳本 🟢 Weak 略保守，季 xwOBA .304 平庸但 Top 3 集體在熱期。Yandy .896 / last7 .854（穩熱）、Aranda .811 / last7 1.065（極熱）、Caminero .827 / last7 .948（極熱）— 三個核心 OPS 全 800+ 且 last7 上行；Simpson / Mullins 偏冷拖累。連勝 +6 / RS 4.40 last 10。對 Williams 的 vs LHB SLG .375 / vs RHB OBP .311 漏洞有錯位優勢。實質「上強下弱」，較接近 🟡 Average（適度上修）。

## 牛棚

| | HOME (CLE) | AWAY (TB) |
|---|---|---|
| ERA / IL 數 / 核心 IL 估計 | 4.40 / 2 / 0–1 核心（Walters 中槓桿、Armstrong 深度，未必算 high-leverage） | 5.09 / 8 / **2 核心**（Cleavinger 高槓桿 LHP IL15d、Uceta IL60d 結構性流失，加上 6 名其他 IL = 深度耗竭） |

### 牛棚雙向修正值
- **HOME (CLE) 牛棚**：對 TB +0.3 run（保守 1 核心 IL；4.40 ERA 中庸）| TB ML −2%
- **AWAY (TB) 牛棚**：對 CLE **+0.5–0.7 run**（2 核心 IL + 整體 ERA 5.09 + 8 IL 結構性深度耗竭）| TB ML −3~4%
- 雙向反映：TB pen 災難等於 Rasmussen 一旦下場 → CLE 後 3–4 局得分加成顯著；CLE pen 中庸但相對完整 → TB 後段得分輕度加成。

## 風險提示

- ⚠️ **AWAY 打線 Flag 3** (last7 BABIP=0.237)：
  - 判讀「**團隊冷數據被下半段拖累、上半段反而熱**」。Yandy / Aranda / Caminero 個別 last7 OPS 均 .854+，Simpson / Mullins 個別 BABIP .067~.217 顯示整隊 BABIP 是底層拖累。即使打線整體 BABIP 微回歸，核心 Top 3 的熱度由個人 hard hit / EV95% 數據支持，不視為全隊冷期。**不自動 ±run value**；不下修 TB 預期得分。
- ⚠️ **Rasmussen FIP-ERA 1.62 落差**：FIP 4.07 vs ERA 2.45。未觸發 Flag 13 門檻（< 1.5），但仍提示 HR 抑制有運氣成分。`xERA 2.64 / xFIP 2.76` 雙重校驗顯示底層仍是 Elite，不下修 Tier，僅敘事註記「不出現意外 HR 即穩」。
- ⚠️ **Williams ERA-xERA 0.81 落差**（未到 Flag 13 門檻）：Solid 數字 3.28 受運氣加持；遇到 TB 熱手 Top 3 + barrel 高漏洞，回歸風險具體。

## 條件修正

- Park Factor: 101.0 → +0.05 run（neutral）
- Progressive Field HR -9% → 微幅壓制大局，但 Rasmussen / Williams 都不是被 HR 主導的投手
- 雙方先發皆 🟡 Solid+：-0.5 run（總分壓縮信號）
- 先發 tier 不對等：Rasmussen Elite >> Williams Solid → 不對稱修正，AWAY 端再 -0.3 run（CLE 方期望得分被進一步壓低）
- Doubleheader：無
- 天氣：4 月底克利夫蘭日場（12:10 ET），預期氣溫低於季均，輕微壓打（不量化，留給 predict.py 環境補充）
- TB +6 連勝 / CLE -4 連敗：動能類軟性因素（依紀律不影響方向，僅納入信心）

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (CLE) | ~3.0–3.5（Rasmussen Elite 強壓） | + TB pen IL +0.5 / 兩 SP Solid+ −0.25 | ~3.5–3.8 |
| AWAY (TB) | ~3.5–4.0（Williams Solid + TB 上重打線） | + CLE pen IL +0.3 / 兩 SP Solid+ −0.25 | ~4.0–4.3 |
| Total | ~6.5–7.5 | 信號 +0.3 / Park ~0 | **~7.5–8.1** |

## 整體判斷

- **方向（基本面）**：**Rays 微幅優勢**。投手對決 Rasmussen Elite >> Williams Solid（且 Williams 數字含運氣），打線 TB Top 3 熱手 + 對 Williams 球種錯位，動能 +6 連勝。**唯一抵銷因素**：TB 牛棚深度災難（2 核心 IL + 8 IL 名單），中後段守不住領先風險顯著。實質：Rasmussen 在場時 TB 大幅領先機率高，但能否守住整 9 局是分歧點。
- **總分（基本面）**：修正後總分 ~7.5–8.1 vs O/U line **7**。差距 +0.5–1.1，**處於 D2 / D5 「< 1.5 不推薦」邊界**。Rasmussen Elite + Cleveland 4 月日場壓打因素都讓 OVER 邊際縮小。**邊緣 OVER 7，但 PASS 風險高**。
- **信心**：**LOW–MEDIUM**。投手 Tier 落差是真，但（1）TB 牛棚災難（2）Williams ERA 過去 3 場走勢佳（3）市場 ML 反而把 CLE 當微幅 favorite，三重訊號顯示市場已部分定價 Rasmussen 優勢。
- **風險**：
  1. TB 牛棚 8 IL — Rasmussen 下場後 4 局守備災難，CLE 後段反撲可能
  2. CLE 主場 + 場館 HR -9% 壓 TB 長打火力
  3. last7 BABIP .237 雖判定為下半段拖累，整隊回歸方向不明
  4. Williams 近 3 場 4 ER / 17.7 IP 顯示有狀態，可能繼續壓制 TB 中後段棒次

⛔ MUST NOT contain：星級、明確盤口推薦
