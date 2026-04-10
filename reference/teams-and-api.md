# 隊名對照表 & API 端點

## 隊名比對（中文 / 簡稱 → 英文全名）

| 中文 | 英文全名 | 縮寫 | teamId |
|-----|---------|------|--------|
| 洋基 | New York Yankees | NYY | 147 |
| 大都會 | New York Mets | NYM | 121 |
| 紅襪 | Boston Red Sox | BOS | 111 |
| 道奇 | Los Angeles Dodgers | LAD | 119 |
| 天使 | Los Angeles Angels | LAA | 108 |
| 太空人 | Houston Astros | HOU | 117 |
| 勇士 | Atlanta Braves | ATL | 144 |
| 費城人 | Philadelphia Phillies | PHI | 143 |
| 教士 | San Diego Padres | SD | 135 |
| 巨人 | San Francisco Giants | SF | 137 |
| 小熊 | Chicago Cubs | CHC | 112 |
| 白襪 | Chicago White Sox | CWS | 145 |
| 紅人 | Cincinnati Reds | CIN | 113 |
| 紅雀 | St. Louis Cardinals | STL | 138 |
| 釀酒人 | Milwaukee Brewers | MIL | 158 |
| 海盜 | Pittsburgh Pirates | PIT | 134 |
| 響尾蛇 | Arizona Diamondbacks | ARI | 109 |
| 落磯 | Colorado Rockies | COL | 115 |
| 金鶯 | Baltimore Orioles | BAL | 110 |
| 光芒 | Tampa Bay Rays | TB | 139 |
| 藍鳥 | Toronto Blue Jays | TOR | 141 |
| 雙城 | Minnesota Twins | MIN | 142 |
| 皇家 | Kansas City Royals | KC | 118 |
| 老虎 | Detroit Tigers | DET | 116 |
| 守護者 | Cleveland Guardians | CLE | 114 |
| 水手 | Seattle Mariners | SEA | 136 |
| 運動家 | Athletics | OAK | 133 |
| 遊騎兵 | Texas Rangers | TEX | 140 |
| 馬林魚 | Miami Marlins | MIA | 146 |
| 國民 | Washington Nationals | WSH | 120 |

## API 端點

### 請求 A — 當日賽程與先發投手
```
https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={YYYY-MM-DD}&hydrate=probablePitcher(note)
```
> 必須加 `hydrate=probablePitcher(note)` 才能取得先發投手。

提取欄位：

| 欄位 | API 路徑 |
|------|---------|
| 比賽 ID | `games[].gamePk` |
| 比賽時間 | `games[].gameDate`（UTC → 當地時間） |
| 比賽狀態 | `games[].status.abstractGameState`（Preview / Live / Final） |
| 主隊 / 客隊 | `games[].teams.home.team.name` / `away.team.name` |
| 先發投手 | `games[].teams.{home/away}.probablePitcher.fullName` |
| 球場 | `games[].venue.name` |

### 請求 B & C — 雙方近 10 場戰績
```
https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId={球隊ID}&startDate={比賽日期-20天}&endDate={比賽前一天}&hydrate=linescore
```

提取：gameDate、home/away、score、isWinner
> 僅計算 `Final` 狀態 + `gameType = "R"`（例行賽），排除春訓。

### 前場比分驗證（系列賽第 2+ 場必須執行）
```
https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={前一場日期}&teamId={球隊ID}&hydrate=linescore
```
> 嚴禁依賴 WebSearch 摘要判斷前場比分，必須用 API。

### 資料來源優先順序
API > 官網公告 > ESPN/CBS/FanGraphs > 網頁抓取。切勿因第三方資料推翻 API 結果。

## Pythagorean Win% 公式（Pythagenport）

```
exponent = 1.50 × log10[(RS + RA) / G] + 0.45
Pythagorean Win% = RS^exp / (RS^exp + RA^exp)
```

> RS = 近 10 場總得分，RA = 近 10 場總失分，G = 10
> Pythagenport RMSE = 3.991 勝（優於固定指數 1.83 的 4.126）

回歸信號：|實際勝率 - Pythagorean Win%| > 10% → 標註「回歸風險」
