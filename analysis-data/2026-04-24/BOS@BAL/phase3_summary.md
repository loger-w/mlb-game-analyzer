# Phase 3 Summary — BOS @ BAL 2026-04-24 19:05 ET

> 基本面快照；**不含盤口推薦/星級**（single source of truth = Phase 4 prediction.json）

---

## 1. 先發投手對決

### Brandon Young (BAL, RHP, 27歲)
- **2026 本季**：ERA 0.00 / FIP 3.50 / xFIP 5.06 / xERA 3.01 / K% 11.1 / BB% 11.1（**1 GS / 5 IP**，樣本極小）
- **2025 prior**：ERA 6.24 / FIP 5.32 / xFIP **3.90** / K% 18.4 / BB% 8.6 / GB% 52.5（57.7 IP / 12 GS）
- tier：⚪ Below Average（本季 tier 標記 🟠 Strong Ace 僅因 ERA 0.00 樣本偏差）

### YoY 對比結論（觸發：|ERA−xERA|=3.01 ≥1.5 + IP<30 + 本季 ERA 比 2025 低 ≥1.0）

| Statcast 指標 | 2025 | 2026 | 變化 | 判定 |
|--------------|------|------|------|------|
| avg_velo | 87.9 | **89.4** | +1.5 mph ↑ | 實質提升 |
| max_velo | 96.7 | 95.3 | -1.4 ↓ | - |
| whiff% | 9.6 | 7.6 | -2.0 pp ↓ | 揮空能力退步 |
| csw% | 26.4 | 22.7 | -3.7 pp ↓ | 邊角率下滑 |
| hard_hit% | 26.4 | **31.8** | +5.4 pp ↑ | 被擊球質量變差 |
| barrel% | 10.2 | 7.1 | -3.1 ↓ | 好 |
| xERA | 4.27 | 3.01 | -1.26 | 5 IP 不可靠 |
| pitch mix | FF 43.7 / FS 17.8 / CU 15.7 / FC 9.1 / SL 8.9 | FF 45.5 / FS 19.7 / SL 18.2 / **SI 12.1** / CU 4.5 | 新增 **Sinker 12.1%**、SL ↑9.3pp、CU ↓11.2pp、FC 消失 | 配球重組 |

**判定**：球速 +1.5 mph + 新增 Sinker = 結構性改變（new-version signal），但 whiff/csw 同步下滑 + hard-hit 上升 = 球質真實改善疑問。保守定位 **🟢 Back-End Starter**（以 2025 xFIP 3.9 + sinker 加成為基線；ERA 0.00 是 5 IP 運氣）。

### Brayan Bello (BOS, RHP, 26歲)
- **2026 本季**：ERA 6.75 / FIP 5.72 / xFIP 4.54 / **xERA 6.85**（ERA ≈ xERA **非運氣**）/ K% 14.1 / **BB% 13.0**（崩盤）/ GB% 69.8（18.7 IP / 4 GS）
- **2025 prior**：ERA 3.35 / FIP 3.92 / xFIP 3.91 / K% 17.7 / BB% 8.4 / GB% 55.7（166.7 IP）
- **Platoon**：vs LHH .975 OPS（被左打殺爆）/ vs RHH .792
- tier：⚪ **Below Average**（真實退步，不是樣本偏差）

**對決結論**：Bello 本季 xERA 6.85 vs Young 2025 xFIP 3.9 + 新版 sinker → **Young 基線略優一檔**。但 Young 樣本僅 5 IP，若回歸 2025 ERA 6.24，BOS 打線可能翻倍得分。

---

## 2. 打線評級

### BAL（vs RHP Bello）
- tier 🟡 Average | avg OPS **.693** | avg xwoBA .330 | avg K% 24.7 / BB% 10.7
- recent_heat ⚖️ Normal | **last7 BABIP 0.214**（⛔ B10 觸發）

## BABIP 回歸判定
BAL 近7天 BABIP 0.214 遠低於 .260 → 回歸 ~.300 後，近7天 OPS 被運氣壓低約 30-50 點。真實熱度比「Normal」更好；Henderson last7 BABIP .118 / Mayo .091 / Beavers .263 → 即將回彈。**Hot 判定維持不升（未達 Hot 門檻），但扣 Cold run value 不適用**；run value 修正 +0.3 run 加在 BAL 得分。

BOS last7 BABIP 0.262 剛好不觸發（>.260），Cold 判定維持，不做回歸調整。

- 關鍵打者 vs RHP：**Ward** OPS .810 / **Taveras** OPS .996 / Jackson .717 / Alonso .748
- 弱點 vs RHP：Henderson .660（slump 回歸中）/ Basallo .653 / Mayo .545 / Beavers .588
- **IL 衝擊**：Jackson Holliday (2B) / Jordan Westburg (3B, 60D) / Ryan Mountcastle (1B, 60D) / O'Neill / Kjerstad — 4+ 核心缺陣，替補 Alonso / Mayo / Jackson / Beavers 填補

### BOS（vs RHP Young）
- tier 🟢 **Weak** | avg OPS **.633** | avg xwoBA .302 | avg K% 24.1 / BB% 8.3
- recent_heat 🥶 **Cold** | last7 BABIP 0.262（剛好不觸發 B10）
- 近3場 vs NYY 0/1/2 分 → 打線崩盤驗證

- 關鍵打者 vs RHP：**Abreu** OPS .844（唯一威脅）/ Anthony .666 / Mayer .586
- 弱點 vs RHP：Story .515 / Contreras .787 / Durbin .523 / Rafaela .590 / Duran .514
- **IL 衝擊**：Triston Casas (1B, 60D) — 少一個長打點

**打線對決結論**：**BAL 明顯強**（.693 vs .633 OPS + .330 vs .302 xwoBA）；BAL Normal（含運氣回歸）vs BOS Cold 基線本就弱 → 雙重優勢 BAL。

---

## 牛棚雙向修正值

（B9 觸發 — 雙方皆有核心 IL）

### BAL 牛棚（IL 狀況）
- **核心 IL**：Félix Bautista (Closer, 60D) + Andrew Kittredge (Setup, 15D) = **2 名核心**
- 其他 IL：Kremer / Eflin（輪值）/ Enns / Akin / Selby / Hiraldo
- 牛棚 ERA 3.58（含自動抓取值 — 尚可）

### BOS 牛棚（IL 狀況）
- **核心 IL**：Justin Slaten (Setup, 15D) = **1 名核心**
- 其他 IL：Houck / Crawford / Sandoval / Gray / Oviedo（多為輪值或 depth）
- 牛棚 ERA 3.72

### 修正值

| 修正項 | BAL 核心 2 缺 | BOS 核心 1 缺 | 淨效應 |
|--------|-------------|-------------|--------|
| **ML 修正** | BAL -3.5% | BOS -2.0% | 淨 BAL -1.5% |
| **OU 修正** | BOS +0.5 run | BAL +0.3 run | **+0.8 run 偏 OVER** |

（OU 雙向都 +run；ML 一側扣自家，最終淨效應 BAL 略承壓但 BOS 同受影響）

---

## 4. 條件修正

| 信號 | 觸發 | 方向 |
|------|------|------|
| 主場優勢 BAL | 是（Camden Yards） | BAL +25% HFA（predict.py 內建） |
| 動能 BAL +1 streak vs BOS -3 streak | 是 | BAL 微 +1% ML |
| BOS 連敗（-3 from NYY） | 是 | BOS -1 ML 信號 |
| Park Factor | Camden 101（近中性） | 微偏打（+0.1 run） |
| BAL 打線 BABIP 回歸 | 是（.214 → .300） | BAL +0.3-0.5 run 預期 |
| BOS 打線 Cold | 是（BABIP 不觸發回歸） | BOS -0.3 run |
| Young 小樣本風險 | 是（5 IP） | 不確定性 +/-1.5 |
| Bello 控球崩盤 | 是（BB% 13%） | BOS -0.5 run（主動性喪失） |

---

## 5. 修正後預期得分

### BAL 得分（vs Bello）
- 基準：BAL 近10 RS/G **5.1** · 近30 RS/G 4.44
- Bello xERA 6.85 遠高於聯盟 4.20 → **+1.3 run** 上修
- BOS bullpen (Slaten IL) → +0.3 run
- BABIP 回歸 → +0.3 run
- 打線 Normal → 中性
- **預期：5.5-6.0 runs**（中位數 **5.7**）

### BOS 得分（vs Young）
- 基準：BOS 近10 RS/G **3.0** · 近30 RS/G 3.68
- Young 以 2025 xFIP 3.9 基線（略優聯盟）→ **-0.3 run** 下修
- Young 新增 sinker 提升 GB → -0.2 run
- BAL bullpen (Bautista+Kittredge IL) → +0.5 run
- 打線 Cold → -0.3 run
- **預期：3.5-4.0 runs**（中位數 **3.8**）

### 總分預期
- 中位：**9.5 runs**
- 區間：9.0 - 10.0
- OU 線 9.15 → **微偏 OVER**（但幅度 ~0.3-0.4 非強信號）

---

## 6. 整體判斷（方向性，不含盤口）

**ML 方向**：明顯偏 **BAL**
- ✅ 主場 + 略優投手（Young > Bello by ~1 檔）+ 強打線（OPS .693 vs .633）+ 動能（+1 vs -3）
- ⚠️ 投手差不足 2 檔，比分差信心不到強讓分門檻

**比分差**：BAL 預估勝 ~1.5-2.0 runs

**O/U 方向**：微偏 OVER
- 雙方牛棚都 +run（+0.8 total）；Bello xERA 6.85 易爆
- 但 BOS Cold + Weak 可能壓低得分

**主要風險**：
1. **Young 5 IP 樣本極小**（最大 X-factor）— 若回歸 2025 ERA 6.24，BOS 得分可能 +1.5，總分破 10；若新版有效，BAL 大勝但 BOS 被壓
2. **盤口異常**：BOS -1.75 讓分反映市場按 Bello prior year 定價，但本季證據（xERA 6.85）否定這定位 → 盤口在 BOS 側可能高估
3. **BAL 打線 IL 深度**（Holliday/Westburg/Mountcastle 皆缺）— 若替補表現差，打線優勢縮水
