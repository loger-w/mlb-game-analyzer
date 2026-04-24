# Phase 3 綜合分析 — LAA @ KC 2026-04-24

## 對戰基本面

- 客隊 Los Angeles Angels（12-14 季, 近10 場 4-6, RS 5.3 / RA 3.9）
- 主隊 Kansas City Royals（8-18 季, 近10 場 1-9, RS 3.9 / RA 6.4, 深度下滑）
- 先發：Yusei Kikuchi (L) vs Noah Cameron (L)
- 球場：Kauffman Stadium（Park Factor 99.0，中性）

---

## 3.1 投打對決

### Kikuchi（LAA 先發）
- Tier：⚪ Below Average（ERA 5.63）但 **FIP 3.85 / xFIP 3.47** → 帳面 ERA 被 BABIP/LOB 拖累，實際能力接近 🟢 Back-end 邊緣
- Statcast：avg_velo 89.0 / whiff 11.5% / hard_hit 28.6% / barrel 8.3%（接觸品質尚可）
- 34 歲左投，球種：FF 28.3% / FS 21.9% / SL 20.8% / FC 18.3% / CU 9.7%
- Platoon（小樣本警告）：vs L OPS .840（26 BF）/ vs R OPS .820（85 BF）→ 左右差異不顯著

### KC 打線 vs Kikuchi
- Tier：🟡 Average（OPS .666 偏弱、xwOBA .311）
- vs LHP 關鍵：Witt 1.028（23 PA）、Garcia 1.054（23 PA）前段砲火，但 Pasquantino .321（32 PA）對左投**極弱**
- Recent heat：Normal，last7 BABIP .312（正常區間，不觸發 B10）
- BvP 最大 PA = Garcia 10，全員 <15 → **不引用 BvP**

### Cameron（KC 先發）
- Tier：⚪ Below Average（ERA 5.40），但 **xERA 6.98** → 差 1.58 run 觸發 YoY 閘門
- Statcast：avg_velo 86.7 / whiff 10.2% / hard_hit 32.1% / barrel 15.2% / ev95 48.5%（球速偏慢、揮空率低、接觸品質差）
- 26 歲左投，球種：FF 31.2% / FC 23.1% / CH 20.7% / CU 16.4% / SL 8.6%
- Platoon：vs L OPS 1.281（21 BF 小樣本）/ vs R OPS .722（68 BF）

### LAA 打線 vs Cameron
- Tier：🟡 Average（OPS .736、xwOBA .323）
- vs LHP 關鍵：**Trout 1.124**（25 PA）、Adell .972（37 PA）、Neto .903（29 PA）→ 右打核心三人組對左投極具威脅
- Recent heat：Normal，last7 BABIP .273（正常區間，不觸發 B10）
- BvP：Trout 3 PA、Neto 3 PA、Adell 3 PA 皆 <15 → **不引用 BvP**

### 投打結論
- 投手差距：Kikuchi（FIP 3.85, xFIP 3.47）明顯優於 Cameron（xERA 6.98, FIP 5.55），**差距約 1-2 檔**
- 打線差距：同為 🟡 Average；LAA OPS .736 > KC .666，但主力對 LHP 威脅更集中（Trout/Adell/Neto 三人都極強）

---

## 牛棚雙向修正值

| | KC（主） | LAA（客） |
|---|---|---|
| 整體牛棚 ERA | **6.29（極差）** | 4.74（中等偏差） |
| 核心 IL | Estévez（Closer） | Joyce（高槓桿）、Yates（高槓桿） |
| 核心 IL 人數 | 1 | 2 |

**B9 雙向修正值**：
- KC 牛棚核心 1 人 IL + 整體 ERA 極差：O/U 對 LAA **+0.3 run** / ML KC **-2%**
- LAA 牛棚核心 2 人 IL：O/U 對 KC **+0.5 run** / ML LAA **-3%**
- 腳本自動 signal：「主隊牛棚 ERA 6.29 ≥ 5.0」= total +0.5（歸在 LAA 側，避免與 B9 KC 牛棚 +0.3 重複，合併取 +0.5）

---

## YoY 對比結論

### Noah Cameron 2026 vs 2025 Statcast

| 指標 | 2025（基準） | 2026（本季 4 GS） | 變化 | 判定 |
|---|---|---|---|---|
| ERA | 2.99 | 5.40 | +2.41 | ↓ 惡化 |
| xERA | 4.07 | 6.98 | **+2.91** | ↓↓ 嚴重惡化 |
| FIP | 4.08 | 5.55 | +1.47 | ↓ 惡化 |
| HR/9 | 1.17 | 2.25 | **+1.08（近 2 倍）** | ↓↓ |
| hard_hit% | 26.3 | 32.1 | **+5.8** | ↓ 惡化 |
| barrel% | 6.3 | 15.2 | **+8.9（>2 倍）** | ↓↓ |
| ev95% | 37.4 | 48.5 | **+11.1** | ↓↓ |
| GB% | 47.0 | 37.2 | -9.8 | ↓ 滾地球減少 |
| avg_velo | 86.0 | 86.7 | +0.7 | → 持平 |
| whiff% | 11.5 | 10.2 | -1.3 | → 略降 |
| 球種 SL% | 14.1 | 8.6 | -5.5 | ← 策略轉換 |

**結論**：球速未降，但 **被擊球品質三項一致大幅惡化**（hard_hit / barrel / ev95），HR/9 接近翻倍，球種配置 SL 使用減半。按 YoY Statcast 驗證方法規則「**一致退化 → ERA 低是假象，真實水平已退步**」—— Cameron 本季 ERA 5.40 被 xERA 6.98 更準確反映。

**Run Value**：對 LAA 得分 +0.3 run（補充基礎公式用 ERA 5.40 時對 xERA 6.98 的低估）。

### Kikuchi 未觸發 YoY
- 本季 ERA 5.63 比 prior 3.99 **高** 1.64，不符合「本季 ERA 比 prior 低 ≥ 1.0」條件
- FIP 3.85 / xFIP 3.47 反映真實能力接近去年，KC 得分估計應**下修 0.3**（公式用 ERA 5.63 估得太高）

---

## 3.4 近期狀態 & BABIP 回歸

- KC 近 10 場 1-9（RS 3.9 / RA 6.4, diff -25），深度下滑；本季崩盤
- LAA 近 10 場 4-6（RS 5.3 / RA 3.9, diff +14），趨勢上升
- H2H：無系列前場（series_prev null，首戰）
- BABIP 回歸閘門：KC last7 .312 / LAA last7 .273 → 皆在 .260-.370 正常區間，**不觸發** B10

---

## 修正後預期得分

基礎公式（由 predict.py formula）：LAA 6.0 / KC 4.0 / total 10.0

**Run Value 修正**：
| 修正項 | 方向 | Run Value |
|---|---|---|
| 自動 signal：KC 牛棚 ERA 6.29 ≥ 5.0 | +LAA | +0.5 |
| 手動 B9：LAA 牛棚 2 核心 IL | +KC | +0.5 |
| Cameron YoY 惡化（xERA 6.98） | +LAA | +0.3 |
| Kikuchi FIP<<ERA，KC 得分下修 | -KC | -0.3 |

**最終預期**：
- LAA ≈ 6.0 + 0.5 + 0.3 = **6.8**
- KC ≈ 4.0 + 0.5 - 0.3 = **4.2**
- 總分 ≈ **11.0**

---

## 整體判斷

- 方向傾向：基本面明顯偏 **LAA（客隊）**
  - 先發端 Kikuchi 優於 Cameron（至少 1 檔，Cameron xERA 嚴重退化）
  - 牛棚端 LAA 整體品質較好（ERA 4.74 vs 6.29），雖主力傷兵更多但後半局 KC 更脆弱
  - 打線端 LAA OPS 較高且主力 vs LHP 火力更集中（Trout/Adell/Neto）
  - KC 近 10 場 1-9 深度低潮，Cameron 在 home 得不到足夠火力支援
- 信心程度：**中高**（投手差距 YoY 證實，牛棚差距客觀）
- 風險：
  1. Kikuchi 本季 5 場 ERA 5.63 樣本不穩，單場亂流風險
  2. Pasquantino vs LHP 極差（.321 OPS, 32 PA）是 KC 打線弱鏈，但 Witt/Garcia 前段可能爆發
  3. Cameron 若 SL 回歸（本季從 14.1% 減到 8.6% 可能策略調整），風險修正可能改善
  4. 總分預測 11.0 偏高；單場隨機性 SD ≈ 4.5 run
  5. 無 ML XGBoost 模型可供 cross_validation，僅 formula 單一來源 → confidence 自動降 LOW 或 MEDIUM
