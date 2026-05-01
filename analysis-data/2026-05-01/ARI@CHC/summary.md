## 投手對決

### Colin Rea (HOME, RHP, 35 📉📉 明顯退化)
- **Tier 覆寫**：沿用腳本 🟢 Back-end Starter，但實質傾向 🟢↗（接近 Solid 邊緣）
- 真實水平判斷：ERA 4.61 看似平庸但 xERA 4.27 / FIP 3.76 / xFIP 3.52 / K-BB% 12.8 都比 ERA 樂觀，意味受到些許運氣懲罰；球速 89.6 偏慢但 whiff% 8.9、hard_hit% 27.7 控制住擊球品質。最大特徵是 **極端反向 Platoon split**：vs RHB 才 .179/.256/.256（43 BF），vs LHB 卻被打到 .288/.342/.424（74 BF，主要傷害來源）。
- 對手打線威脅：D-backs 先發中 Carroll（LHB）是唯一純左打，是 Rea 最大隱患；Marte / Perdomo / Vargas 是 switch hitter（會以 LHB 站上來打 Rea）→ 換算下來 Rea 今天大概要面對 **5 個 LHB 視角**，這是真正的 stress point。Arenado / Gurriel Jr. 是 RHB，正好命中 Rea 的優勢區。

### Zac Gallen (AWAY, RHP, 30 📉 初期退化)
- **Tier 覆寫**：覆寫 🟠 Strong Ace → 🟡 Solid Starter（lean low），實質可能更接近 🟢
- 真實水平判斷：**ERA 3.14 是嚴重 fool's gold**。xERA 4.95（gap −1.81）/ xwOBA .348 / xBA .296 / hard_hit% 30.2 / barrel% 9.4 / EV95% 45.8 — Statcast 全面亮紅燈，contact quality 已是後段先發水準。K-BB% 僅 8.4（Strong Ace 應 ≥ 18），球速 89.3 mph（生涯巔峰 92-93）。FIP 3.55 / xFIP 4.19 同樣高於 ERA。**這不是純運氣，是結構性退化 + 樣本運氣加成**（詳見風險提示段）。
- 對手打線威脅：Cubs 主力 vs RHP 表現分歧——Happ vs RHP .837 / Bregman vs RHP .706 是真威脅；PCA EV95 46.8 / Bregman EV95 45.2 contact quality 強。但 Hoerner、Busch、PCA 的 last7 OPS 落差大（PCA last7 .811 hot，Hoerner .527 cold）。Cubs 整體 .331 xwOBA 對上一個真實水準 4.5+ ERA 的投手，**有實質吃分能力**。

## 打線評級

### HOME — 🟡 Average / ⚖️ Normal
- **Tier 覆寫**：沿用腳本。xwOBA .331 / OPS .788 / chain OBP .357（Hoerner-Bregman-Happ 是聯盟中上的開路三人組）；EV95 / Barrel% 在 PCA / Bregman / Happ 三人身上偏強，contact quality 略優於 .331 xwOBA 帳面值；不下修因為近 7 天 BABIP .285 落在正常區、Heat 也是 ⚖️ Normal 沒虛胖。

### AWAY — 🟡 Average / 🔥 Hot
- **Tier 覆寫**：沿用 🟡 Average，但 🔥 Hot 需要折價。聚合 last7 BABIP .316 不極端，但 **個別主力的 last7 BABIP 集體飆高**：Carroll .438 / Perdomo .471 / Arenado .471 / Vargas .391——四個主力都顯著高於 .370 回歸警戒線。這是聚合層沒觸發但個別層觸發的「隱形 Flag 3」。Heat 真實，但持續性存疑；vs RHP 的長期能力（Carroll .750 / Arenado .791 / Perdomo .748）才是可信基線。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.77 / 10 / 4 名（Palencia、Thielbar、Harvey、Hodge） | 5.03 / 6 / 3 名（Puk、Saalfrank、J. Martinez） |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚（CHC）：4 名核心 IL → 🔴🔴 極高分級。但聚合 ERA 仍守在 3.77 是因為剩餘的 Webb / Maton / Milner / Ríos 撐住基本面。問題在 leverage——傳統高槓桿三人（Palencia / Harvey / Thielbar）全傷後，7-8 局接力深度被掏空，對左打應對能力（Thielbar、Wicks 都缺）特別差。對手 D-backs 中後段碰到左打輪轉時是 CHC 的破口。
- AWAY 牛棚（ARI）：3 名核心 IL（Puk / Saalfrank / J. Martinez 都是 setup 以上等級）+ 整體 ERA 5.03 → 🔴🔴 極高 + 品質崩盤的雙重打擊。Ginkel / Sewald 撐 closer，但中繼層只剩 Loáisiga / Thompson / Clarke / Morillo，沒有可信賴的 7-8 局橋樑。**ARI 一旦先發 Gallen 5 IP 後下場，6-9 局是 CHC 主場進攻最大窗口**。

## 風險提示

- ⚠️ AWAY 投手 Flag 8 (era_xera_delta=-1.81):
  - 判讀：**結構性退化為主、運氣加成為輔，並非純樣本噪音**。三組獨立指標一致指向真實水平下滑：(a) 球速 89.3 mph 為生涯新低（從 92-93 降）；(b) K-BB% 8.4 已不是 ace 等級；(c) Statcast 三件套（hard_hit% 30.2 / barrel% 9.4 / EV95% 45.8）顯示對手實際擊球品質很強，xERA 4.95 / xwOBA .348 與之吻合。ERA 3.14 是被低 BABIP / 高 LOB 暫時掩蓋。本場敘事上不應把 Gallen 視為 Strong Ace；但依規範 **不自動 ±run value**——D-backs 先發深度 28.2 IP 樣本仍可能再壓低 ERA 一場後爆走，方向偏 over 但時點不確定。

## 條件修正

- Park Factor: 92.0 → -0.40 run（HOME / AWAY 各 −0.20）
- 先發 tier / doubleheader / 天氣：非 doubleheader；4 月底 Wrigley 風向高度不可預測，無公開資料故不修正；雙方先發都 RHP，無左右手特殊修正。

## 修正後預期得分

> 「+ 信號」欄僅納入規範允許的條件修正：Park Factor、牛棚累計效應（核心 IL ≥ 2 名）、主力打者傷兵。
> ⛔ BABIP 極端值 / ERA-xERA gap **不入此欄**（規範禁止 auto ±run value，見 reference/flags-checklist.md §3, §8）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 3.7 | -0.20（PF）+0.30（ARI 牛棚 3 核心 IL + 5.03 ERA）= +0.10 | 3.8 |
| AWAY | 3.7 | -0.20（PF）+0.20（CHC 牛棚 4 核心 IL，但替補 ERA 仍 OK 故折半）= 0.00 | 3.7 |
| Total | 7.4 | +0.10 | 7.5 |

## 整體判斷

- **方向（基本面）**：CHC 略佔優。三條獨立邊：(1) 主場 Wrigley 抑投但 wind-dependent；(2) Gallen 真實水平 < ERA 帳面，Cubs 主力 vs RHP 有真實傷害力；(3) ARI 牛棚崩盤等級（5.03 ERA + 3 核心 IL）。反向因子是 D-backs 打線正炎（但 BABIP 不可持續）+ Rea vs LHB 漏（D-backs 5 個 LHB 視角是真實壓力）。淨結果：CHC 略優但非碾壓。
- **總分（基本面）**：~7.5。兩邊牛棚都殘是雙刀劍——上半段（Gallen vs Rea）會被 Wrigley PF 92 + 雙方先發控球壓低；6 局之後兩隊牛棚都不可靠 → 中後段傾向爆分。整場 trend 偏 **「先低後高」**，最終總得分偏正常偏高。
- **信心**：MEDIUM。Gallen Flag 8 / D-backs 個別 BABIP 過熱 / Wrigley 風向都是高方差因子。
- **風險**：
  1. **Gallen 終於回歸日**：xERA / contact quality 都警告今天可能就是爆掉的一場，如果 PCA / Bregman barrel 起來 → CHC 大局且 over 直奔。
  2. **ARI 牛棚 6-9 局崩**：機率高，CHC 主場最後攻擊的低槓桿橋樑投手會是破口。
  3. **D-backs 主力 BABIP 回歸**：Carroll / Arenado / Perdomo last7 BABIP .438-.471 不可持續；如果今天集體冷卻，ARI 攻擊面會比 .765 OPS 看起來更弱。
  4. **Wrigley 風向**：4 月底風向高度不確定，風吹出 → over 顯著，風吹進 → Wrigley 變極端投手園。風向是這場最大的單一不確定變數。

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
