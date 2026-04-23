# Phase 3 Summary — CHW @ ARI（2026-04-22, Chase Field）

先發：Anthony Kay (CHW, LHP) vs Eduardo Rodriguez (ARI, LHP)

---

## YoY 對比結論

### ARI — Eduardo Rodriguez（33 歲 LHP）

| 指標 | 2025 | 2026 | Δ | 判定 |
|------|------|------|---|------|
| avg_velo | 89.4 | 87.9 | -1.5 mph | ⛔ 退化 |
| max_velo | 95.9 | 94.0 | -1.9 mph | ⛔ 退化 |
| whiff_pct | 9.6 | 7.0 | -2.6 | ⛔ 退化 |
| K% | 20.6 | 14.6 | -6.0 | ⛔ 大幅退化 |
| hard_hit_pct | 23.5 | 25.4 | +1.9 | 略差 |
| xERA | 4.51 | 4.19 | -0.32 | 持平 |
| pitch mix | FF 47% / CH 20% | FF 36% / CH 35% | 四縫線 -11 / 變速球 +15 | 策略改配球 |

**結論**：2026 ERA 1.96 完全是 BABIP / HR luck 驅動（FIP 4.19 / xFIP 4.70）。Statcast 五項指標中 4 項退化、球種策略大改（降低四縫線、增加變速球），典型 33 歲左投**初期退化 + 投球風格轉守**。真實水平在 xERA 4.19 附近。**實際分級：🟢 Back-end**（系統 tier 標 Strong Ace 是 ERA 假象）。

### CHW — Anthony Kay（31 歲 LHP）

| 指標 | 2025 | 2026 | 判定 |
|------|------|------|------|
| MLB 數據 | 無（未於 2025 登板 MLB） | 17.33 IP / 4G / 2GS | 樣本極小 |
| ERA vs xERA | — | 2.60 vs **6.91** | ⛔ 落差 4.31 run，極端 luck |
| hard_hit_pct | — | 33.3% | 高 |
| ev95 pct | — | 50.0% | 高 |
| vs RHP platoon | — | .265/.379/.449 (58 BF) | 失能 |

**結論**：2025 無 MLB 數據作為 YoY 對比基準；2026 season 17 IP 樣本過小 + xERA 6.91 顯示真實水平 ≈ Below Average。4G/2GS role-mix 特徵（swingman），按 workflow 先發評估需降一檔。**實際分級：⚪ Below Average**（系統 tier 標 Strong Ace 是 ERA 假象）。

---

## 雙方先發對決

| 面向 | E.Rodriguez (ARI) | A.Kay (CHW) |
|------|-------------------|-------------|
| 真實分級 | 🟢 Back-end (xERA 4.19) | ⚪ Below Average (xERA 6.91) |
| FIP | 4.19 | 4.77 |
| K-BB% | 5.2 | 3.9 |
| avg_velo | 87.9 | 91.0 |
| 球種主力 | FF 36% / CH 35% / FC 14% | FF 35% / SL 20% / ST 17% / SI 14% / CH 13% |
| vs LHP 打者 | .200/.304/.200 (23 BF) | .000/.150/.000 (20 BF, SSS) |
| vs RHP 打者 | .224/.288/.358 (73 BF) | .265/.379/.449 (58 BF, **失能**) |

**投手對決結論**：E.Rod 真實水平優於 Kay 約 1.5-2 檔。Kay 對右打嚴重失能（.449 SLG / .379 OBP），ARI 主力打者多 R/SHB（Arenado R / Marte S / Perdomo S / Vargas R），右打區段對 Kay 極有利。

---

## 打線評級與熱度

| 面向 | ARI（vs LHP Kay） | CHW（vs LHP E.Rod） |
|------|------------------|---------------------|
| Tier | 🟢 Weak | 🟡 Average |
| avg_ops | 0.711 | 0.696 |
| avg_xwoba | 0.308 | 0.311 |
| L7 BABIP | 0.308（正常） | 0.290（正常） |
| recent_heat | ⚖️ Normal | 🔥 Hot |
| over_under_lean | 0 | -1 (K% 25.9) |
| OBP top3 / SLG mid | 0.339 / 0.353 | 0.353 / 0.429 |
| 對 LHP 明星棒次 | Carroll vs LHP 1.164 OPS / Vargas 1.130 / Barrosa 0.992 | Murakami 0.880 (power) / Montgomery 1.210 / Vargas 1.101 / Meidroth 0.875 / Pereira 1.300 |

**BABIP 回歸檢查**：雙方全隊 L7 BABIP 均在 .290-.308 正常區間，免觸發 B10 強制回歸修正。個別熱打者（Perdomo L7 .409 / Meidroth .409 / Murakami .417）屬小樣本團內雜訊，不足以撼動整體判定。

**打線結論**：CHW 打線略勝一籌（Chain mid 段 SLG 0.429 vs ARI 0.353），且近 7 天多人熱打；但 CHW 整體高 K%（25.9）在 xERA 4.19 的 E.Rod 面前會打不少空棒（E.Rod K% 只 14.6 但 hard_hit 低）。ARI 打線整體偏弱但**對 LHP 有數個爆發點**（Carroll/Vargas/Barrosa vs L 都 OPS 0.9+）。

---

## 牛棚雙向修正值

| 球隊 | 全隊 ERA | 核心 IL | 修正 |
|------|---------|--------|------|
| ARI (home) | 4.40 | Puk / Martinez / Saalfrank 三名 60-Day（含 Closer + Setup + high-leverage）+ 先發 Burnes | ⛔ **3+ 核心 IL** |
| CHW (away) | 5.73 | Murphy / Thorpe / Cannon / Bush / Vasil / Berroa 六人 IL | 牛棚整體品質差 |

**ARI 牛棚 3+ 核心 IL 修正**（workflow §B9）：
- O/U：對手 CHW **+1.0 run** 加到客隊得分
- ML：ARI ML **-5%**
- 信號強度：**+2**

**CHW 牛棚品質補註**：5.73 ERA 顯示後段不可靠，ARI 打線若進入 6 局後有機會得分（但非 IL-based 修正，計入 formula base 會被 starter ERA 項主導）。

---

## 條件修正摘要

| 信號 | 狀態 | Run Value |
|------|------|----------|
| 雙方先發皆 🟠+ Strong Ace | ❌ 實際分級 🟢/⚪ | 0 |
| 雙方先發皆 🟡+ Solid | ❌ | 0 |
| 雙方打線近 7 天 Hot | ❌ CHW Hot / ARI Normal（非雙方） | 0 |
| Platoon 劣勢（全打線 vs 同手） | ❌ 雙方都派左投，但雙方右打/SHB 核心足夠 | 0 |
| 牛棚前日重操（5+ IP） | 未偵測（G2 ARI 11-5 大敗，可能 ARI 牛棚多用；但官方 IP 檔未查） | 保留 |
| 牛棚核心 2+ IL (ARI) | ✅ 3+ 名核心 IL | **+1.0 到 CHW 得分** |
| Park Factor | 99 | (99-100)×0.05 = -0.05（可忽略） |
| 投手休息日 | 雙方預設 5 天 | 0 |

---

## 修正後預期得分（公式粗估）

基礎 formula（聯盟均 4.5 RS/G, xwOBA 0.318, ERA 4.30, PF 99）：

- **E[R_ARI] 基礎** ≈ 4.5 × (0.308/0.318) × (Kay blended/4.30) × 0.99
  - Kay blended ERA（ERA 2.60 / FIP 4.77 / xERA 6.91 → 中位 ~4.77，考慮樣本小可略降至 ~4.60）
  - ≈ 4.5 × 0.969 × (4.60/4.30) × 0.99 ≈ **4.61**
- **E[R_CHW] 基礎** ≈ 4.5 × (0.311/0.318) × (E.Rod blended/4.30) × 0.99
  - E.Rod blended（ERA 1.96 / FIP 4.19 / xERA 4.19 → ~3.45，但 ERA luck driven 應偏向 xERA 4.19）
  - ≈ 4.5 × 0.978 × (3.80/4.30) × 0.99 ≈ **3.85**
- **信號修正**：ARI bullpen IL → CHW +1.0
  - E[R_ARI_adj] ≈ 4.61
  - E[R_CHW_adj] ≈ 3.85 + 1.0 = **4.85**
- **修正後總分** ≈ 9.46

> 以上為手算粗估。**真相來源**是 Phase 4 `predict.py --save` 輸出的 `formula_prediction` + `ml_prediction`。

---

## 整體判斷（方向傾向）

1. **投手面**：雙方都是左投但真實水平 E.Rod 優於 Kay 約 1.5-2 檔（xERA 4.19 vs 6.91）；E.Rod 一路 3 場沒失分是運氣+對手適應中。
2. **打線面**：CHW 打線略勝一籌但對 LHP E.Rod 會吃癟在高 K%（25.9）；ARI 打線弱但對 LHP 有爆發點（Carroll/Vargas）。
3. **牛棚面**：ARI 失三名核心（60-Day IL） vs CHW 牛棚整體 5.73 ERA。CHW 後援更差但未集中 IL，ARI 是「精銳缺陣」。
4. **近況**：CHW 連勝 3 + 系列賽剛 11-5 碾壓 ARI（但那場是對 ARI 先發 Kelly/Nelson 等），單場反覆性高。
5. **方向**：基本面**偏 ARI**（投手 E.Rod 真實水平較佳、Kay 對右打失能）。但牛棚 IL 將劣勢部分扳平，而且 CHW 近況確有熱度。**方向傾向 ARI 小贏，總分 9-10 附近中性區**。
6. **關鍵不確定性**：Kay 的 xERA 6.91 樣本僅 17.33 IP，回歸幅度未知；E.Rod 雖然數據退化但 ERA 假象仍可能延續一場。

> 具體盤口推薦（ML/OU/RL 星級）依 Phase 4 `predict.py` 模型輸出為準。
