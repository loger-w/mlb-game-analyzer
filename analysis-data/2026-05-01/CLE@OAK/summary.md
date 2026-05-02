## 投手對決

### J.T. Ginn (HOME, RHP, 26 ⚡ 巔峰期)
- **Tier 覆寫**：沿用腳本 🟡 Solid Starter
- 真實水平判斷：地面球派 sinkerballer（SI 38.1% / FC 16.7% / CH 16.2%），ERA 3.24 / xERA 3.34 高度一致 → 結果反映實力。FIP 4.74 vs xFIP 3.86 落差顯示 HR 運氣偏差，但 hard hit% 23.1 / barrel% 7.0 對擊球品質壓制良好。velo 91.6 與 K-BB% 10.0 偏低限制上限，靠誘打弱接觸吃飯，符合 Solid Starter 中段定位。
- 對手打線威脅：CLE 整體 OPS .695 偏弱，Ramírez vs RHP 僅 .674 且近 7 天 .500 OPS 進入低潮，Kwan 季季 OPS .602 + last7 OPS .688 低迷；DeLauter (.778) / Martínez (.890) 是右打段中的反向亮點。Ginn vs RHB .220/.267/.341（45 BF）對右打強勢 → 整體對位 Ginn 略佔優。

### Joey Cantillo (AWAY, LHP, 26 ⚡ 巔峰期)
- **Tier 覆寫**：降至 🟡 Solid Starter（腳本 🟠 Strong Ace 偏樂觀）
- 真實水平判斷：ERA 2.97 vs xERA 3.88 落差 0.91（未觸 Flag 8 但偏軟），FIP 3.86 提示 HR 運氣同樣偏差。velo 85.6 全聯盟極低（即使 LHP 也偏低），主以 FF 44.9% + CH 25% 配球，K-BB% 16.8 中上但非頂尖。vs RHB .275/.376/.463（93 BF 大樣本）顯示對右打明顯吃力（OBP .376 警訊），vs LHB 樣本僅 32 BF 過小。整體實力介於 Strong Ace 與 Solid Starter，xERA 落於 Solid Starter 區間。
- 對手打線威脅：OAK 🔥 Hot + xwOBA .335 / chain OBP .368 串聯佳，且大多數核心打者為右打 — 正撞 Cantillo 弱點。Langeliers vs LHP 1.119（小樣本但火爆）+ Kurtz EV95% 60.7 / Barrel% 21.3 頂級擊球 + Wilson last7 OPS .831 是三層威脅。Soderstrom vs LHP .286 是唯一明顯避站；McNeil 偏低但 last7 OPS .749 持平。整體威脅度 **高**。

## 打線評級

### HOME — 🟡 Average / 🔥 Hot
- **Tier 覆寫**：沿用 🟡 Average，但近 10 場 RS 4.40 / OPS .751 + chain OBP .368 + 5 名先發級球員 Hot 趨勢 → 實戰上接近 🟠 Strong 下限。Last7 BABIP .306 屬正常區間（未觸 Flag 3），Hot 偏向結構性而非 BABIP 噪音 → 持續性可信度中高。

### AWAY — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：腳本 🟡 偏樂觀，本季 OPS .695 / xwOBA .325 / chain OBP .332 接近 🟢 Weak 上緣。Ramírez 近期低迷拖累中軸，Kwan / Rocchio 上壘能力縮水（season OPS 0.602 / 0.764）。整體串聯偏弱，比 Tier 標籤再低半檔。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.98 / 1 / 0 | 4.34 / 2 / 2 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。
> HOME 唯一 IL Gunnar Hoglund 為先發 / 60-day → 不計入核心牛棚。
> AWAY Andrew Walters（CLE high-leverage 7-8 局）+ Shawn Armstrong（setup）皆屬核心 → **2 名核心 IL，影響度 🔴 高**。

### 牛棚影響判讀
- HOME 牛棚：整體 ERA 3.98 中段水準，無核心 IL，可正常執行 Closer / Setup 配置。對應對手 CLE 偏弱中軸末段壓制力足。
- AWAY 牛棚：ERA 4.34 偏高 + 核心缺 2 名 → 末段（7-9 局）防守明顯吃緊。對應對手 OAK 的 🔥 Hot 中軸 + 串聯佳的打線，末段被攻陷風險高。**雙向影響**：對 OAK 末段得分 +、對 CLE 自身勝率 −。

## 風險提示

無風險提示（dossier 未標 ⚠️；Cantillo ERA-xERA gap 0.91 < 1.5 Flag 8 門檻，敘事已於投手段落判讀為「運氣偏差 + 上限被 velo 限制」，不自動下修。）

## 條件修正

- Park Factor: 109.0 → +0.45 run
- Sutter Health Park HR +6%（HR 環境輕微加成，貼合 OAK 拉打型中軸）
- 先發 tier / doubleheader / 天氣：無 doubleheader；Cantillo 與 Ginn 均 26 歲巔峰期，無年齡退化修正；天氣資料未抓（戶外球場 5 月 Sacramento 一般偏溫和，預設無修正）。

## 修正後預期得分

> 「+ 信號」欄僅納入規範允許的條件修正：Park Factor、牛棚累計效應（核心 IL ≥ 2 名）、主力打者傷兵。
> ⛔ BABIP 極端值 / ERA-xERA gap **不入此欄**（規範禁止 auto ±run value，見 reference/flags-checklist.md §3, §8）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 4.8 | +0.45 (PF) +0.30 (CLE 牛棚核心 IL 2 名 → OAK 末段得分機會 ↑) | 5.55 |
| AWAY | 5.7 | +0.45 (PF) | 6.15 |
| Total | 10.5 | +1.20 | 11.70 |

## 整體判斷

- **方向（基本面）**：略偏 OAK（HOME）。三層信號一致：(1) Cantillo 對右打弱 + Tier 高估，正撞 OAK 右打為主的 🔥 Hot 中軸；(2) CLE 核心牛棚 IL 2 名 → 末段防守變薄；(3) Ginn 對右打優勢 + CLE 中軸近期低迷壓制 CLE 攻擊上限。OAK 主場 Hot lineup + Park 109 加成構成方向護城河。
- **總分（基本面）**：11.5-12.0 區間，**OVER 方向**。base 已偏高（兩位投手 FIP 都 ≥ 3.86），加上 PF 雙邊 +0.45 與 CLE 牛棚信號 → 11.7 為中位估計。
- **信心**：MEDIUM。
- **風險**：
  1. Cantillo vs LHB 僅 32 BF，OAK 兩位主要左打（McNeil / Soderstrom）對位樣本噪音大；若 Soderstrom 突破 .286 vs LHP 紀錄 → AWAY 信號弱化。
  2. Ginn FIP 4.74 vs xFIP 3.86 提示 HR 運氣負債未必回正；若 OAK Hot 中軸 Barrel% 21.3 (Kurtz) 兌現一發 → HOME 上修但 AWAY 不變。
  3. CLE 牛棚 IL 2 名替補品質未知，若有 7+ 名強力深度（CLE 牛棚向來深）→ 影響度需降級至中。
  4. Sutter Health Park 樣本期短（臨時主場），PF 109 含估計噪音，3 年加權代表性弱於其他球場。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
