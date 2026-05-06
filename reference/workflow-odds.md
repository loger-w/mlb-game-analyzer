# Workflow: Odds（步驟 3）

---

## 步驟 3：盤口分析

### 3.1 找出本場條目

`odds/reports/{date}.md` 結構：tier 分組（🔥 Major ≥ 5pp / 🟡 Significant ≥ 3pp / 🔵 Watch ≥ 1pp / ⚪ Quiet < 1pp）→ Anchor Notes → 解讀說明。

用 Grep tool 搜 pattern `{Away} @ {Home}` on `odds/reports/{date}.md`，讀取：
- `direction_label`（→ TEAM ±Xpp，no-vig latest vs anchor 差）
- 時間軸 table（snapshots × ML / RL / Over / Under）
- Flags（位移 + 薄盤 + key number 跨越）

### 3.2 解讀架構

| 維度 | 判讀 |
|------|------|
| Tier | 🔥/🟡 = strong move 必看；🔵 watch；⚪ noise（不必引用） |
| 方向 | direction_label 顯示 market 偏向；對照 anchor 看 sharp 還是穩定 |
| 薄盤 | latest 距開球 < 4h → 訊號可能被晚場閉盤動作污染，可信度降一檔 |
| Key number | Total 跨 7 / 9 / 11 標 ⚠️ — 1.0 run 跨 key 比 0.5 跨非 key 重要 |
| 雙邊 vs 單邊 | no-vig pp-delta 區分莊家整體 vig 調整（雙邊同向）vs 真 sharp money（單邊位移） |

### 3.3 Paired analysis（only if `basic_state = complete`）

- 比對 summary.md 的 direction（基本面）vs odds report 的 direction（market）
- 同向 + market move 強 → confluence（雙重支持）
- 反向 → fundamental disagreement，AI 必須解釋 gap
- 計算 fundamental fair vs market price gap（如果 summary 有 adjusted runs，可粗算 implied ML 對照）

### 3.4 完工條件 / 紀律

✅ MUST contain：tier 引用、direction、薄盤 flag（若有）、paired lean（若 basic complete）
⛔ MUST NOT contain：
- 自行補資料 / 無中生有 fair odds
- 「下哪邊」明確指令（給 lean + 信心，user 自行決策）
- 無錨數字硬推（如「市場 +EV 6.4%」）
