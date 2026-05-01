## 投手對決

### Erick Fedde (HOME, RHP, 33 yo 📉)
- **Tier 覆寫**：腳本給 🟡 Solid Starter（ERA 3.42 / ERA+ ~125），但 K-BB% 僅 4.6（嚴重偏低）、FIP 4.62、whiff 7.9% — peripheral 全是 🟢 Back-end 等級。**真實水平：🟡 / 🟢 之間，偏向 🟢 Back-end**。ERA-FIP 落差 1.20 顯示得分壓制有「BABIP 與殘壘運氣」成分，後續可能上修。
- 真實水平判斷：吃局數的 mid-rotation 軟投（FB 87.5 mph），靠 sweeper / cutter 弱接觸生存；近 3 場 6 ER/16.0 IP（3.38 ERA）狀態尚可。但 xERA 3.74 與 FIP 4.62 中間值更接近真實水平 ≈ ERA 4.0~4.2。
- 對手打線威脅：LAA 對 RHP xwOBA .324 / OPS .756（聯盟平均之上）；Trout last7 OPS 1.083（vs RHP 季 .961）+ Soler vs RHP .821 + Schanuel last7 .810 形成 3-4 棒長打鏈。Fedde 缺三振 + LAA 進階打擊指標佳 → contact 機會可觀。

### Yusei Kikuchi (AWAY, LHP, 34 yo 📉📉)
- **Tier 覆寫**：腳本給 ⚪ Below Average（ERA 6.21），但 FIP 3.58 / xFIP 3.63 / xERA 4.91 / K-BB% 14.3 — peripheral 屬 🟡 Solid。**真實水平：🟢 Back-end ~ 🟡 Solid 之間（落差來自 BABIP 與殘壘運氣）**。⚠️ ERA-xERA 落差 1.30（緊貼 1.5 閾值未觸發 Flag 13），但 ERA-FIP 落差 2.63 為極端值 → 屬「結構性問題（控球差 WHIP 1.59）+ 運氣差」混合。
- 真實水平判斷：FB 89.4 / Splitter 22.4% / Slider 18.4% 武器庫齊全；K-BB% 14.3 在 LHP 中屬中上。但近 3 場 11 ER/14.7 IP（6.74 ERA）顯示尚未回穩；單場仍可能突然炸裂 5+ ER。
- 對手打線威脅：⚠️ **Reverse platoon**：vs LHB .308/.400/.500（30 BF）顯著差於 vs RHB .278/.340/.444。CWS LHB 主力 Vargas vs LHP OPS 1.263（小樣本噪音大但近期手感真實）/ Montgomery vs LHP 1.056 / Murakami vs LHP .852 — 三人都被排在打線上半。RHB 端 Benintendi vs LHP .166 是反向劣勢但被 LHB 群覆蓋。CWS 對 Kikuchi 結構性占優。

## 打線評級

### HOME (CWS) — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用腳本（xwOBA .316 / OPS .693 — 略低於聯盟平均，但 last 10 RS 6.00 顯著高於季 RS 4.06，攻↑趨勢真實，BABIP .304 屬於健康範圍 — 不是 BABIP 泡沫）。對手左投 + reverse platoon → 本場期望值高於整體 tier。

### AWAY (LAA) — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用腳本（xwOBA .324 / OPS .756 — 進階指標略優於 CWS）。但 last 10 RS 僅 3.50（季 4.77）+ 6 連敗，攻↓趨勢明顯；last7 BABIP .346（高於 .300，未到 .370 Flag 3 閾值，但近期偏熱有運氣成分）。Trout last7 OPS 1.083 是 outlier，整體打線冷化是真實。

## 牛棚

| | HOME (CWS) | AWAY (LAA) |
|---|---|---|
| ERA / IL 數 / 核心 IL 估計 | 4.98 / 6 / **2 名核心**（Prelander Berroa 60d 高壓力 RP / Chris Murphy 後段 LHP；Thorpe / Cannon / Bush / Vasil 多為先發背景） | 5.79 / 6 / **3 名核心**（Ben Joyce 高壓力 closer 100+ mph / Kirby Yates 老牌 closer/setup / Robert Stephenson 60d setup；Manoah & G. Rodriguez 為 IL'd 先發不計入） |

### 牛棚雙向修正值
- HOME (CWS) 牛棚：對手 +0.5 run | HOME (CWS) ML −3% （核心 IL 2 名 → 🔴 高）
- AWAY (LAA) 牛棚：對手 +1.0 run | AWAY (LAA) ML −5% （核心 IL 3 名 → 🔴🔴 極高，Joyce + Yates 是封閉局數的關鍵）

→ 牛棚淨向：CWS 端 +0.5 vs LAA 端 +1.0，**LAA 牛棚問題更嚴重**，後段局數 CWS 占優。

## 風險提示

腳本未自動標 Flag。AI 加註：
- ⚠️ Kikuchi ERA-FIP 落差 2.63（極端），但 Flag 13 閾值是 ERA-xERA ≥ 1.5，他 1.30 剛好低於。實務上 Kikuchi 屬「結構性 + 運氣差」混合 → 不對 CWS 得分自動下修；若硬下修則違反 D2 信號紀律（沒有自動修正規則覆蓋此情境）。
- ⚠️ LAA last7 BABIP .346 雖未到 Flag 3 閾值（.370）但偏熱，Trout 1.083 OPS 含運氣成分，**不對 LAA 得分自動上修**。
- 兩位先發年齡 33-34 + 📉/📉📉 退化已反映在本季數據，不額外修正。

## 條件修正

- Park Factor: 97.0 → −0.15 run（已內含於 base formula）
- 先發 tier：Fedde 🟡 vs Kikuchi ⚪ — 不滿足「雙方先發皆 🟡 Solid+」-0.5 信號（Kikuchi ERA tier ⚪），不應用「投手對決壓低總分」修正
- doubleheader / 天氣：本場為 12:10 pm ET 日場，無 doubleheader；天氣資料未取得（4 月芝加哥常見涼風 + 西南風中等對 LHB OUT 略增），不另加修正

## 修正後預期得分

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME (CWS) | 3.7 | +1.0（LAA 核心 IL 3 名） | 4.7 |
| AWAY (LAA) | 4.9 | +0.5（CWS 核心 IL 2 名） | 5.4 |
| Total | 8.6 | +1.5 | **10.1** |

差距 vs O/U 8.5 = **+1.6 run（OVER 側）**

修正後勝差 LAA −0.7 run（CWS 4.7 vs LAA 5.4）— 比 base 1.2 縮小，仍偏向 LAA。

## 整體判斷

- **方向（基本面）**：LAA 險勝（formula log5 偏 LAA ~60%，加入 CWS 牛棚較淺 IL 影響後縮小 → 估真實勝率 ~57-60%）。CWS 主場 + 連勝動能 + reverse platoon 部分抵銷 LAA 紙面數據優勢，但模型方向（D1）為 LAA。
- **總分（基本面）**：偏 OVER（修正後 10.1 vs 8.5，差 +1.6 run）。雙方先發 K-BB 數值差距大但 Fedde 4.6 太低、Kikuchi 控球差，加上兩牛棚 ERA 4.98 / 5.79 都屬聯盟下游 + 共 5 名核心 IL → 後段失分概率高。
- **信心**：**MEDIUM**。
  - 偏低因素：Kikuchi ERA-FIP 極端落差（單場結果二極化）、LAA 連敗動能、樣本噪音（Vargas/Murakami vs LHP OPS 為小樣本）。
  - 偏高因素：模型方向 + 市場一致（LAA 1.869 隱含 53.5% / 模型 ~58-60%）、牛棚雙向 IL 修正方向同向（OVER）、reverse platoon 結構性訊號清晰。
- **風險**：
  1. Kikuchi 若 FIP/xFIP 兌現（3.58 / 3.63）→ CWS 5+ 局只取 1-2 分，OVER 押注危險。
  2. LAA 6 連敗心理面 + 進攻冷化，Fedde 控制比賽 6 IP 1-2 ER 場景並非小機率。
  3. CWS 先發 Fedde 雖 Solid Starter，但 K-BB% 4.6 + FIP 4.62 顯示 contact-heavy，遇到 Trout 一發即可破壞 base prediction。
  4. 兩位先發都可能早退（Kikuchi 控球 / Fedde 接觸品質），交給 ERA 4.98 / 5.79 牛棚 → variance 放大。

⛔ MUST NOT contain：星級、明確盤口推薦
