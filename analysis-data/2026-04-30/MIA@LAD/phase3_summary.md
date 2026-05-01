## 投手對決

### Tyler Glasnow (HOME, RHP, 32 歲)
- **Tier 覆寫**：沿用腳本 🔴 Elite Ace
- 真實水平判斷：本季 ERA 2.45 / xERA 2.16 / FIP 2.22 / K-BB% 25.9 / WHIP 0.70 全方位頂尖；whiff% 13.5、hard hit% 24.0 反映 swing-and-miss + 弱接觸雙修兼備。近 3 場 8 ER / 18 IP (4.00 ERA) 略遜，但 18 IP 樣本不足、結構數據未崩。32 歲 RHP 進入初期退化區，但球速 avg 90.1 / max 98.3、xERA 仍領先 — 退化尚未顯現。
- 對手打線威脅：MIA 打線 🟢 Weak (xwOBA .298, OPS .728)。Glasnow vs LHB .155/.218/.268 (78 BF) 是對 LHB 滅頂級壓制；MIA 主要左打 Marsee / Edwards 雖 last7 火熱（OPS .868/.859）但 BABIP .368/.381 屬於不可持續區間，且 EV95%/Barrel% 偏低（Marsee 38.8/2.4）→ 接觸品質撐不起表面數字。Glasnow vs RHB .122/.143/.244 (42 BF) 同樣壓制。整體威脅低。

### Sandy Alcantara (AWAY, RHP, 30 歲)
- **Tier 覆寫**：沿用腳本 🟠 Strong Ace（保留變異警戒）
- 真實水平判斷：ERA 3.05 / xERA 3.10 一致，但 FIP 4.00 與 K-BB% 僅 8.5 是隱憂 — 表現靠弱接觸（hard hit% 23.6、barrel% 4.0）+ 防守支撐，三振保送結構偏中後段。近 3 場 2 ER / 24.3 IP (0.74 ERA) 火熱。30 歲已進入初期退化但球速 avg 92.2/max 99.6 維持，TJ 後第三年起算為「完全恢復」階段。
- 對手打線威脅：LAD 打線 🟠 Strong (xwOBA .357, OPS .780)，Top 5 (Ohtani / Tucker / Freeman / Pages / Muncy) 整體 vs RHP 強健 (Muncy .946 / Freeman .822)。Alcantara vs LHB .236/.313/.375 (80 BF) 是合理但非滅頂壓制 — Ohtani、Freeman、Muncy 三大左打有產出空間。Alcantara 火熱期 vs LAD 火力 — 屬於「中等威脅、有單局崩盤可能」。

## 打線評級

### HOME — 🟠 Strong / ⚖️ Normal
- **Tier 覆寫**：沿用腳本。xwOBA .357 / OPS .780 / K% 22.5 / BB% 10.2 結構完整。last7 BABIP .274 略低於聯盟平均（.300），有小幅回升動能。但本場面對 Glasnow 🔴 Elite 對 LHB 滅頂壓制 — 預期難充分發揮 strong 火力。

### AWAY — 🟢 Weak / ⚖️ Normal
- **Tier 覆寫**：沿用腳本。xwOBA .298 / OPS .728，1-3 棒 chain OBP .369 還能上壘，但 mid SLG .372 清壘能力薄弱。Marsee / Edwards last7 OPS 高但 BABIP 高、EV95% 低 → 不可持續，回歸風險明顯。對手 Glasnow 是最差選擇之一。

## 牛棚

| | HOME (LAD) | AWAY (MIA) |
|---|---|---|
| ERA / IL 數 / 核心 IL 估計 | 3.93 / 10 / 1 名核心 (Casparius) | 3.68 / 3 / 1 名核心 (Fairbanks) |

LAD IL 10 人多為先發 (Snell IL15)、長期傷兵；高槓桿後援 Casparius 缺陣為主要影響。MIA IL 3 人含 Pete Fairbanks（closer/setup tier）+ Mazur（先發 IL60）。雙方各損失約 1 名核心，量級接近，淨效果接近互抵但略偏 MIA 較深。

### 牛棚雙向修正值
- HOME (LAD) 牛棚：對手 +0.3 run | HOME ML -2%
- AWAY (MIA) 牛棚：對手 +0.3 run | AWAY ML -2%

## 風險提示

無腳本標記之 BABIP / 連戰 / 球場類風險。但 AI 額外觀察：
- Marsee/Edwards last7 BABIP .368/.381 + 低 EV95% → 表面熱度不可持續（vs Glasnow vs LHB 結構性壓制下尤甚）
- Alcantara K-BB% 8.5 + FIP 4.00 → 弱接觸表現有單場 BABIP 反向風險，但 24 IP 火熱期短期不易反轉
- LAD IL 名單 10 人雖看似多，主要為先發輪值，對本場 ML 影響有限

## 條件修正

- Park Factor: 98.0 → -0.10 run（已內建於 signal_table）
- 雙方先發皆 🟠 Strong Ace+ → **-1.0 run**（reference: matchup-factors.md 總分下修信號）
- Doubleheader / 天氣 / Umpire / 多/少休息日：無
- TJ / 角色轉換：Alcantara TJ 第三年起 = 完全恢復、不額外修正

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (LAD) | 4.8 | 雙投 -0.5 / Park -0.05 / MIA bp IL +0.3 | **4.55** |
| AWAY (MIA) | 2.2 | 雙投 -0.5 / Park -0.05 / LAD bp IL +0.3 | **1.95** |
| Total | 7.0 | -1.0 - 0.1 + 0.6 = -0.5 | **6.5** |

## 整體判斷

- **方向（基本面）**：LAD 略佔優勢 — 投手對決 Glasnow > Alcantara 一個檔次，打線 Strong > Weak，主場優勢；但 LAD ML 1.452 (隱含 67-69%) vs formula log5 56.3%，市場已過度定價 LAD。
- **總分（基本面）**：修正後 6.5 vs O/U 8.5，差距 **2.0 run** → 強烈偏 UNDER。雙投皆有 sub-3.20 ERA、MIA 打線弱 + Glasnow 滅頂 LHB、LAD 打線面對熱手 Alcantara — 三因子一致指向低分。
- **信心**：MEDIUM — UNDER 結構性支撐強，但 Alcantara K-BB% 8.5 + LAD 強打有單局崩盤可能性，總分變異略高。
- **風險**：
  1. Alcantara 弱接觸路線單場 BABIP 反向 → LAD Top 5 多支安打串聯
  2. Glasnow 近 3 場 4.00 ERA — 若延續手感不在，MIA 1-3 棒可上壘但 mid 清壘弱仍難爆分
  3. 兩隊牛棚都缺核心，後段比賽若進入 6+ 局可能擴大失分
  4. 主場 Dodger Stadium HR PF 121 — 單支 HR 可大幅破壞 UNDER（Ohtani / Muncy 是主要威脅源）
