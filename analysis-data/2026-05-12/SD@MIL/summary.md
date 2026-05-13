## 投手對決

### Brandon Sproat (HOME, RHP, 25 ⚡ 巔峰期)
- **Tier 驗證**：腳本 tier_v2 = 🟡 Solid Starter（xFIP p46, K-BB% p44），gap vs ERA-only = +39.6
  - 不同意 Solid Starter。ERA 5.87 / xERA 5.07 / FIP 5.87 / K-BB% 9.2 / vs LHB **.268/.414/.536**(!) 被左打狠打 — 結構是 Below Avg。tier_v2 受 xFIP 4.18 拉抬但 contact 證據壓倒；實質 ⚪ Below Average。
- **Reverse platoon 信號**：未 fired（vs LHB SLG .536 > vs RHB SLG .463，但 RHP 對 LHB 強是反向；應該 reverse 但未 fire — 可能因樣本或 wOBA 差距未過門檻）。
  - 雖未 fire，敘事上 vs LHB SLG .536 顯示對左打吃虧；SD top 5 多右打（Merrill/Tatis/Machado/Andujar/Bogaerts）— Sproat 反 platoon-disadvantage 對 SD 影響小。
- **對手打線威脅**：高。SD vs RHP top 5 偏冷（Tatis .578 / Machado .564 / Andujar .775 / Bogaerts .780 / Merrill .668）+ Cold last7 BABIP 0.187（Flag 3 反彈面）但 Sproat ERA 5.87 + FIP 5.87 結構性弱，SD 應能反彈得分。

### Bradgley Rodriguez (AWAY, RHP, 22 📈 成長期)
- **Tier 驗證**：腳本 tier_v2 = —（樣本 <30 BF，無打分），gap vs ERA-only = —
  - 1 GS 樣本無法 tier_v2。原始：ERA **1.83** / xERA 1.91 / FIP 2.44 / K-BB% 11.5 / WHIP 1.02 / velo 92.9 / 100.6 max / vs LHB .139/.179/.194(!)（極壓）vs RHB .286/.342/.400. **數字極漂亮但 1 GS 樣本不可信賴**。實質按 🟡 Solid Starter 處理（年輕成長股）。
- **Reverse platoon 信號**：見 dossier `## 🎯 訊號摘要`（若 fired）
  - fired Δ +0.369（vs RHB OPS .742 > vs LHB OPS .373）— 巨型 reverse！Rodriguez 對右打吃虧明顯。MIL 多右打（Turang/Chourio/Yelich(L)/Contreras/Bauers(L)/Mitchell/Frelick/Hamilton/Ortiz）— 多數右打 vs Rodriguez 是優勢，但 1 GS 樣本可信度有限。
- **對手打線威脅**：高。MIL Top 5 vs RHP 都火燙（Turang 1.061 vs RHP / Chourio 1.096 / Yelich .866 / Contreras .810 / Bauers .874）+ Rodriguez reverse platoon → Turang/Chourio/Contreras 右打殺手 zone。

## 打線評級

### HOME — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟠 Strong
  - 上修同意。Turang/Chourio/Yelich/Contreras/Bauers 整支 vs RHP .800+，對 Rodriguez 1 GS reverse platoon 有 edge — 評估維持 🟠 Strong，可能單場上修至 Elite 對位。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - 未 fired chain_break；heat_vs_babip last7 BABIP 0.282 normal。

### AWAY — season tier 🟡 Average / heat ⚖️ Normal
- 打線來源：🟢 official
- **Matchup tier (vs RHP)**：🟢 Weak
  - 上修同意 Weak（last7 BABIP 0.187 確認 Cold）。但 Sproat vs LHB SLG .536 → Bogaerts (S) / Andujar (R, 從 LHP 角度) / Castellanos (R) — SD 仍有局部機會。Flag 3 BABIP 0.187 反彈面拉高 distribution。
- **chain_break / heat_vs_babip 信號**：見 dossier `## 🎯 訊號摘要`
  - AWAY chain_break #8-9 gap 0.198 — Laureano .751 → Fermin .401 vs RHP，chain 尾影響小，−0.1 run。

## 牛棚

| | HOME | AWAY |
|---|---|---|
| ERA / 投手 IL / 核心 IL 估計 | 3.42 / 5 / 2 | 3.5 / 5 / 0 |

> 「投手 IL」含先發 / 60-day 長傷；「核心 IL」（Closer + Setup + High-leverage）由 AI 從 dossier 名單估計，對應 `matchup-factors.md` §牛棚傷兵累計效應 的 1 / 2 / 3+ 名分級。

### 牛棚影響判讀
- HOME 牛棚：MIL 3.42 ERA + 2 core IL（Zerpa LHP + Koenig）= 🔴 高。左側壓制變薄，對 SD Bogaerts(S)/Laureano(R) 後段壓制力降低。
- AWAY 牛棚：SD 3.50 ERA + 0 core IL — 完整深度。Rodriguez 5 局後完整火力銜接。

## 風險提示

- ⚠️ AWAY 打線 Flag 3 (last7 BABIP=0.187):
  - 短期極可能反彈（0.187 是 outlier 低點），SD top 5 個別 EV95 都 .380+ 顯示真實擊球品質仍在；反彈方向偏 SD 進攻 +。本場 distribution 向 SD 6+ 分傾斜。不自動 ±run value。

### 額外信號
- 🔴 AWAY reverse platoon Δ +0.369（vs RHB OPS 0.742 > vs LHB OPS 0.373）— RHP 對非預期手別反而吃虧
- 🟠 AWAY chain breaks at #8-9：OPS 落差 0.198
- 🔴 ⏳ HOME 牛棚 core IL ×2：🔴 高（牛棚明顯吃緊）
  - **本場主訊號 = Rodriguez 1 GS 樣本不確定 + MIL 多右打 vs Rodriguez reverse platoon + MIL 2 core IL 後段薄**。SD 對 Sproat ERA 5.87 結構性弱有 edge，但需要 BABIP 反彈才能 cash in。

## 條件修正

- Park Factor: 97.0 → -0.15 run
- 天氣：室內（Roof Closed，不適用）
- 先發 tier / doubleheader：Rodriguez SSS 但結構性 Strong > Sproat Below Avg；MIL 牛棚 2 IL vs SD 0 IL — SD 後段優勢。

## 修正後預期得分

> 「+ 信號」欄：依 `reference/matchup-factors.md §量級錨點` 區間挑值（單側 cap ±0.8 run / 場）。
> ⛔ **不入此欄**：BABIP 極端值（Flag 3）/ ERA-xERA gap（Flag 8）/ strong_park（已含於 PF 倍率）。

| | base (formula) | + 信號 | adjusted |
|---|---|---|---|
| HOME | 2.7 | +0.4（AWAY reverse platoon +0.3 + AWAY chain_break #8-9 +0.0 互動 +0.1） | 3.1 |
| AWAY | 6.1 | +0.2（HOME core IL ×2 +0.3 互動 max+0.1 −0.1 chain AWAY） | 6.3 |
| Total | 8.8 | +0.6 | 9.4 |

## 整體判斷

- **方向（基本面）**：AWAY (SD)
- **總分（基本面）**：9.4
- **方向信心**：62% — Rodriguez 1 GS 樣本給 SD 強投手 advantage 但有可信度折扣；MIL 多右打 + Rodriguez reverse platoon 是 hedge；SD top 5 雖冷但 Sproat 結構性弱 + Flag 3 反彈面支持 SD 進攻。
- **風險**：
  1. Rodriguez 1 GS 樣本 — 任何結果都可能，distribution 極寬
  2. MIL Chourio (.917) + Turang (.933) 真實強，Rodriguez reverse platoon → 1 球翻盤面
  3. SD Cold BABIP 0.187 — 極端 outlier，反彈可能性高但需要實際 hit
  4. MIL 牛棚 2 core IL 後段薄 — SD 若取得領先可擴大

⛔ MUST NOT contain：星級、盤口推薦（ML / O/U / RL）— 盤口屬 odds/ 模組
