# Phase 3 綜合分析：TOR @ LAA (2026-04-22, ET 15:07, Angel Stadium)

## 1. 先發投手對決

| 指標 | Lauer (TOR, L) | Soriano (LAA, R) |
|---|---|---|
| Tier | ⚪ Below Average | 🔴 Elite Ace |
| IP / GS | 17.2 / 3 | 32.2 / 5 |
| ERA | 7.13 | **0.28** |
| FIP | 5.93 | **2.30** |
| xFIP | 4.90 | 2.78 |
| WHIP | 1.47 | **0.73** |
| K-BB% | 7.6 | **21.7** |
| HR/9 | 2.04 | 0.28 |
| GB% | 27.8 | 61.4 |
| avg velo | 86.4 | 93.6 |
| CSW% | 23.6 | 32.5 |
| Whiff% | 7.6 | 13.5 |
| Hard-Hit% | 19.0 | 22.9 |
| Barrel% | 9.6 | 10.3 |

**Step 2 驗證（ERA vs xERA/xFIP 落差 ≥ 1.5）：**
- Lauer：|7.13 − 4.90| = 2.23 → YoY：2025 ERA 3.18 / FIP 3.76 / xFIP 3.65（104.2 IP），真實水準推估 FIP 4.5~5.5（今年開局樣本小、速球 86.4 偏軟）。
- Soriano：|0.28 − 2.78| = 2.50 → YoY：2025 ERA 4.26 / FIP 3.61 / GB 73.6%（169 IP），今年 K% 32.5 / GB 61.4 / CSW 32.5 為 **真實的跳級**。ERA 雖受 BABIP 與低 HR/9 吃紅利，真實水準仍推 ERA 2.5–3.0 區間 → 本季至今頂尖先發。

**Platoon：**
- Lauer vs LHB .333/.500/.667（12 BF 小樣本，但左對左明顯無優勢），vs RHB .220/.303/.441。LAA 上場 Trout/Adell/Neto/Ward 均 R 手，**Lauer 左投 vs LAA 右打陣線沒有平台優勢**。
- Soriano vs LHB .133/.220/.222（51 BF），vs RHB .082/.188/.082（69 BF）— **雙向壓制**。TOR 核心 Guerrero Jr./Clement/Okamoto 均 R 手，正面撞上 Soriano 最強的 vs RHB 面。

## 2. 打線

| | TOR | LAA |
|---|---|---|
| Tier | 🟡 Average | 🟡 Average |
| avg OPS | .694 | .727 |
| avg xwOBA | .313 | .323 |
| avg BABIP | .292 | .267 |
| K% | 17.7 | 25.4 |
| BB% | 7.2 | 11.6 |
| last7 BABIP | .351 | .268 |
| recent_heat | ⚖️ Normal | ⚖️ Normal |
| OBP top3 | .346 | .363 |
| SLG mid | .435 | .396 |

**BABIP 回歸閘門：** TOR last7 .351（< .370 極端上緣）、LAA last7 .268（> .260 極端下緣）— 兩側都 **不觸發** 強制回歸修正。

**核心打者：** TOR Guerrero Jr. 1.057 OPS / 0.394 BABIP（熱度偏高、有回歸風險，但 xwOBA 撐得住）。LAA Trout 0.938 OPS / .232 BABIP（運氣低、真實水準更高）、Neto .805 OPS、Ward/Adell 中段爆發力。

## 3. 牛棚

- TOR bullpen ERA 4.60
- LAA bullpen ERA 4.65

近乎持平，**無雙向閘門差距**（兩邊皆未極端）。預期中段交棒後各增 ~1–1.5 分，不影響先發對決主導。

## 4. 環境 / 近況

- **球場**：Angel Stadium，Park Factor 100.0（中性）。
- **天氣**：Anaheim 4/22 日間賽，無極端風或雨預期（無異常修正）。
- **近況**：
  - LAA 最近 10 場 4-6、連敗 4 場；但 RS/G 5.5、RA/G 4.2（攻擊面仍佳）。
  - TOR 最近 10 場 RS/G 4.1、RA/G 4.6（攻擊面偏冷）。
- **今日是 doubleheader game 2**：game 1 (4/22 早) LAA 2-4 TOR 輸；兩隊均打過上一場（牛棚可能輕微疲勞，但中繼未吃太深）。

## 5. 不確定性 / 風險提示

- Lauer 本季 17.2 IP 樣本非常小，但速球 86.4、barrel 9.6、whiff 7.6 三項硬指標都支持「不如 2025」。
- Soriano ERA 0.28 本身不可持續，但 FIP/xFIP 雙低 + GB 61.4% 代表真實水準仍為本場 clear edge。
- Doubleheader game 2 可能影響兩隊牛棚使用深度。

（本檔為 Phase 3 分析，**不含** ML / O/U / Run Line 星級，盤口建議由 Phase 4 `prediction.json` 提供。）
