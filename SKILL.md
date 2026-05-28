---
name: mlb-game-analyzer
description: Use when the user asks for MLB single-game or full-day (ET) RL / O-U prediction — team-level expected-runs model vs current Pinnacle line (edge in pp). Outputs RL + O/U only, not money line.
---

# MLB Game Predictor — RL / O-U 預測

## Step 0：建立 ET_NOW（必跑、1 次 tool call）

```bash
PYTHON=$(python3 --version >/dev/null 2>&1 && echo python3 || echo python)
$PYTHON -c "from datetime import datetime; from zoneinfo import ZoneInfo; n=datetime.now(ZoneInfo('America/New_York')); print(n.strftime('%Y-%m-%d %H:%M %Z'))"
```

把輸出記為 `ET_NOW`。相對日期(今天/今晚/明天/昨天)一律以 `ET_NOW.date()` 解析,**不信 system 注入的 currentDate**。

## Step 1：解析 intent

- 單場 還是 當日(ET)全部?
- 使用者是否要「先抓盤口 / 最新 odds」?→ 先跑一次 `fetch_odds` 存當下 snapshot。

## Step 2：跑指令

```bash
# (可選)先抓當下盤口並存 snapshot:
python odds/fetch_odds.py

# 單場:
$PYTHON scripts/predict_game.py --date {ET-YYYY-MM-DD} --matchup {AWAY}@{HOME}

# 當日全部例行賽:
$PYTHON scripts/predict_game.py --date {ET-YYYY-MM-DD} --all
```

輸出寫到 `analysis-data/{date}/{AWAY}@{HOME}/`(`features.json` + `prediction.md`)。隊名一律用英文縮寫(KC / LAA / NYY)。

## Step 3：給使用者

讀該資料夾的 `prediction.md` 念給使用者:RL + O/U 的 model 機率、市場 no-vig、edge(pp)。
**AI 只敘事**:指出哪邊有正 edge、量級、需注意什麼。

⛔ 不喊「下哪邊」、不硬掰 EV%、**不提 Money line**、不改任何數字(數字全由 `scripts/predict_game.py` 確定性算出)。

## 注意

⚠️ 模型係數(σ、權重等,集中於 `scripts/config.py`)**尚未經回測重新擬合**前,edge 數字僅供觀察、不可當下注依據。回測:`python scripts/backtest.py run --month {YYYY-MM}`(需先有賽果 `fetch_results.py` 與 v2 features.json)。
