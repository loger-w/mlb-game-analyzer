"""一次性:把 relief_index 快取從 built_through 增量補到 target，避免整季重抓。
只抓 (built_through, target] 區間的 Final 例行賽,沿用 bullpen 的 fetch/解析函式。
歷史 Final 不會變,故與整季重建結果一致。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import requests
from bullpen import MLB_API_BASE, _fetch_boxscore, relief_er_ip, _cache_path, DEFAULT_CACHE_DIR

YEAR = 2026
# = needed_through (as_of) of today's games; 可由 argv[1] 覆寫,如 python _warm_relief_cache.py 2026-06-17
TARGET = sys.argv[1] if len(sys.argv) > 1 else "2026-06-14"

p = _cache_path(YEAR, DEFAULT_CACHE_DIR)
cached = json.loads(p.read_text(encoding="utf-8"))
built_through = cached["built_through"]
index = cached["index"]
print(f"cache built_through={built_through}, target={TARGET}")

# 抓 (built_through, TARGET] 的 Final 例行賽
params = {"sportId": 1, "startDate": built_through, "endDate": TARGET, "gameType": "R"}
r = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=30)
r.raise_for_status()
new_games = []
for d in r.json().get("dates", []):
    for g in d.get("games", []):
        date = g["gameDate"][:10]
        if date <= built_through:            # 已在快取中,跳過
            continue
        if g.get("status", {}).get("abstractGameState") != "Final" or g.get("gameType") != "R":
            continue
        new_games.append({"game_pk": g["gamePk"], "date": date,
                          "home_id": g["teams"]["home"]["team"]["id"],
                          "away_id": g["teams"]["away"]["team"]["id"]})

print(f"new Final games to append: {len(new_games)}")
for i, g in enumerate(new_games, 1):
    box = _fetch_boxscore(g["game_pk"]).get("teams", {})
    for side_key, tid in (("home", g["home_id"]), ("away", g["away_id"])):
        er, ip = relief_er_ip(box.get(side_key, {}))
        index.setdefault(str(tid), []).append({"date": g["date"], "er": er, "ip": ip})
    print(f"  [{i}/{len(new_games)}] {g['date']} pk={g['game_pk']} done")

p.write_text(json.dumps({"built_through": TARGET, "index": index}, ensure_ascii=False),
             encoding="utf-8")
print(f"OK wrote {p} built_through={TARGET}, teams={len(index)}")
