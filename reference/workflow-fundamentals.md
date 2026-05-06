# Workflow: Fundamentals（步驟 1+2）

---

## 步驟 1：資料收集

```bash
$PYTHON scripts/prepare_game.py --date {ET-YYYY-MM-DD} --away {AWAY} --home {HOME}
# Doubleheader：加 --game-suffix G1 / G2
```

失敗 exit code 1/2/3/4/5/7（`1` = 子腳本找不到；其餘見 `prepare_game.py --help`）。

**後續動作**：
1. Read `$GAME_DIR/dossier.md`
2. Read `$GAME_DIR/summary.md` 與 `reference/matchup-factors.md`
3. 進入步驟 2：在 summary.md 上補完所有 `<!-- AI 補 -->` placeholder

ℹ️ **drill-down 只在以下情境 Read**（dossier 已涵蓋核心欄位，預設不必看）：
- 要查 GB% / xBA / csw% / EV95% / 完整 pitch mix / Pitch Arsenal RV/100 → `<side>_pitcher_summary.md`
- 要看 9 人完整 table / Last 7 per-player / Platoon per-player / BvP table → `<side>_lineup_summary.md`
- 要驗 IL list 細節（status / position） → `<side>_roster_summary.md`

---

## 步驟 2：綜合分析

### 2.1-2.4 順序執行

| 子步驟 | 分析內容 | 參考 |
|------|---------|------|
| 2.1 投打對決 | 投手 Tier + 打線評級 + Platoon + 球種 | `matchup-factors.md` |
| 2.2 牛棚 | 品質 + 可用性 + 近 3 天消耗 + 傷兵影響度 | `matchup-factors.md` |
| 2.3 條件修正 | 傷病/TJ/角色轉換/年齡/球場/**天氣** | `matchup-factors.md` §天氣修正 |
| 2.4 風險提示 | dossier 已標的 ⚠️（Flag 8 / Flag 3）AI 敘事判讀 | `flags-checklist.md` |

⛔ BvP 樣本 PA ≥ 15 才可引用（`flags-checklist.md` Flag 2）

### 2.5 完工條件

`$GAME_DIR/summary.md` 內所有 `<!-- AI 補 -->` placeholder 都已補完即為最終輸出。

**MUST contain**：投手 Tier 判斷、打線評級、牛棚影響判讀、風險提示判讀、條件修正、修正後預期得分、整體判斷（方向 / 總分 / 信心 (%) / 風險 1-4 點）。
