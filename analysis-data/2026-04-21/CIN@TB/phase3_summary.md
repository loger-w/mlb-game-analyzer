# Phase 3 Summary — CIN @ TB · 2026-04-21 (ET 18:40)

場館：Tropicana Field（圓頂，PF 96，-4%）

## 1. 先發對決

### Chase Burns (CIN, 23y, R) — 🟠 Strong Ace (script tier)
| 季 | ERA | xERA | FIP | xFIP | K% | BB% | GB% | IP |
|---|---|---|---|---|---|---|---|---|
| 2026 | 2.42 | 3.86 | 4.09 | 3.68 | 25.0 | 10.2 | 45.2 | 22.1 |
| 2025 (prior) | 4.57 | — | 2.62 | 2.20 | 35.6 | 8.5 | 43.8 | 43.1 |

- **YoY 閘門觸發（IP<30 且 ERA 比 prior low 2.15）** — 參照 prior year：K% 爆跌 -10.6pp、FIP +1.47、xFIP +1.48。ERA 2.42 是 BABIP-luck 掩蓋的殼；xERA 3.86 與 FIP 4.09 才是體感。
- **Platoon（關鍵）**：vs-R .132/.154/.237（宰割），vs-L .244/.367/.390 + **BB% 16.3**（大命中風險）。
- 上一場 vs LAA：5.1 IP 5 ER 4 BB（明顯 command slip）。
- 3 場最近 IP：5 / 6 / 5.1 → 負擔約 5-6 IP，不會吃很深。

### Steven Matz (TB, 34y, L) — 🟡 Solid Starter
| 季 | ERA | xERA | FIP | xFIP | K% | BB% | GB% | IP |
|---|---|---|---|---|---|---|---|---|
| 2026 | 3.80 | 4.42 | 3.19 | 3.62 | 24.7 | 7.1 | 34.1 | 21.1 |
| 2025 (prior) | 3.05 | — | 3.35 | 3.38 | 19.1 | 3.6 | 50.0 | 76.2 |

- **結構性警訊**：GB% 50→34（飛球化，easy 長打風險），xERA 4.42 > ERA 3.80（BABIP 運氣撐著）。
- **Platoon 關鍵反差**：vs-L .160/.154/.160（統治），vs-R .231/.322/.423（正常被打）。
- CIN 本場為 **右打為主** 陣容（Elly switch, Stewart R, McLain R, Suárez R, Steer R, Stephenson R, Hayes R），Matz 的最大優勢被中和。

## 2. 打線 × 對投手

### CIN vs Matz (LHP) — 🟡 Average lineup (OPS .628 / xwOBA .315)
- **BABIP .249 偏低（正回歸壓力）**，xwOBA .315 接近平均 → 季初低迷中有反彈結構。
- vs-LHP 威脅點：Sal Stewart 1.346 (23PA)、Elly 1.099 (28PA)、Steer .932 (20PA)、Suárez .762、McLain .705。
- 冷點回歸候選：Hayes 季 BABIP **.080**（極端負向，逼近回歸），Friedl last-7 BABIP .182。
- BvP：全員 PA 3-9，**樣本不足 → 不採用**。

### TB vs Burns (RHP) — 🟢 Weak lineup (OPS .697 / xwOBA .297)
- Burns vs-R .132 OPS-against → 核心右打（Díaz .928, Caminero .766, Fortes 1.022 vs RHP）會被壓制。
- 但 TB 有 **4 LHH**（Aranda, Simpson, Mullins, Fraley）＋ Walls(S) 能吃 Burns 的 BB% 16.3 vs-L 問題 → 會上壘但長打力受限。
- BvP 均 null。

## 3. 牛棚（雙向閘門）

| 側 | Bullpen ERA | 評價 |
|---|---|---|
| TB (home) | **5.28** | 嚴重偏弱 |
| CIN (away) | **2.23** | 菁英級 |

雙向反映：
- 支持 CIN ML（late game CIN 壓得住，TB 追分能力差）
- **Over 推力**：TB 牛棚失分傾向高
- **Under 推力**：CIN 牛棚壓制 TB 下半場
- 淨效應：ML 強傾 CIN，O/U 方向由 TB 牛棚（偏 Over）稍微主導，但 CIN 菁英牛棚拉回部分。

## 4. 條件修正

- **球場 Tropicana (PF 96)**：-4% 總分 → -0.3 run
- **天氣**：圓頂，無風/雨影響
- **近期**：CIN 4 連勝（昨日客場 6-1 大勝 TB）；TB 2 連敗，昨日 1-6 被羞辱。連勝動能不可逆方向但可降低 TB 信心。
- **季內累計**：CIN RS 3.65 / RA 3.78（RA 很漂亮），TB RS 4.73 / RA 5.23（RA 偏高，後段失血）。
- **先發賽季樣本**：兩隊 22/23 場 → **未達 30 場，D1.5 INSUFFICIENT_SAMPLE + D4 受讓偏見防護皆觸發**。

## 5. 比分估算（手動對照用）

- CIN 預期得分：Matz 飛球化 + CIN 右打威脅 + TB 牛棚 5.28 → **4.4-5.0 runs**
- TB 預期得分：Burns 對右打宰割但對左打命中 + TB lineup 偏弱 + CIN 牛棚 2.23 → **3.3-3.8 runs**
- 合計區間 ~7.7-8.8 → 接近 O/U 7.5 上方，Over 輕微傾向（但需 predict.py 模型確認）

## 6. 整體方向性（非盤口）

- 基本面偏 **CIN**：先發對位（Matz 飛球化 vs CIN 右打）＋ 牛棚差距（2.23 vs 5.28）＋ 連勝動能。
- Burns 本身有 regression 風險，但今天面對的 TB 陣容偏右打、整體弱 → regression 風險被稍微中和。
- 市場把本場定為近 pick'em（CIN 51.5% vs TB 50.5%）— 我的基本面傾向認為 CIN 被市場低估。
- O/U 方向不強，需 predict.py 最終比分確認。

（盤口/星級推薦 single source of truth → Phase 4 `prediction.json`，此檔不下結論）
