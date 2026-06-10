"""Point-in-time 牛棚 relief ERA:從每場 boxscore 重建後援(該場 gamesStarted==0)ER/IP,
依日期截止累計查詢。線上模型用 relief-only(sitCodes=rp);MLB API 無乾淨的 point-in-time
relief 端點(byDateRange+rp 數值錯亂、statSplits+日期被忽略),故由 boxscore 重建。"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
DEFAULT_CACHE_DIR = SKILL_ROOT / "analysis-data" / "backtest" / "cache"


def _parse_ip(ip_str) -> float:
    """MLB inningsPitched 字串 → float(.1=1/3, .2=2/3)。自帶以避免與 fetch_inputs 互相 import。"""
    whole, _, frac = str(ip_str).partition(".")
    thirds = {"1": 1/3, "2": 2/3}.get(frac, 0.0)
    return int(whole or 0) + thirds


def relief_er_ip(side: dict) -> tuple[float, float]:
    """單場單隊:加總後援(該場 gamesStarted==0)的 ER 與 IP。side = boxscore.teams[home|away]。"""
    players = side.get("players", {})
    er = 0.0
    ip = 0.0
    for pid in side.get("pitchers", []):
        ps = players.get(f"ID{pid}", {}).get("stats", {}).get("pitching", {})
        if not ps:
            continue
        if int(ps.get("gamesStarted", 0) or 0) != 0:
            continue
        er += int(ps.get("earnedRuns", 0) or 0)
        ip += _parse_ip(ps.get("inningsPitched", "0"))
    return er, ip


def relief_era_as_of(per_game: list[dict], as_of: str) -> float | None:
    """per_game=[{date,er,ip}]。加總 date < as_of 的場次 → ERA=9*ER/IP。IP=0 → None。"""
    tot_er = 0.0
    tot_ip = 0.0
    for g in per_game:
        if g["date"] < as_of:
            tot_er += g["er"]
            tot_ip += g["ip"]
    if tot_ip == 0:
        return None
    return round(9.0 * tot_er / tot_ip, 2)


def _fetch_season_final_games(year: int, through: str) -> list[dict]:
    """賽季 Final 例行賽:回 [{game_pk,date,home_id,away_id}]。一次 range 查詢全聯盟。"""
    params = {"sportId": 1, "startDate": f"{year}-03-01", "endDate": through, "gameType": "R"}
    r = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=30)
    r.raise_for_status()
    out = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final" or g.get("gameType") != "R":
                continue
            out.append({"game_pk": g["gamePk"], "date": g["gameDate"][:10],
                        "home_id": g["teams"]["home"]["team"]["id"],
                        "away_id": g["teams"]["away"]["team"]["id"]})
    return out


def _fetch_boxscore(game_pk: int) -> dict:
    import time
    last_err = None
    for attempt in range(4):
        try:
            r = requests.get(f"{MLB_API_BASE}/game/{game_pk}/boxscore", timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise last_err


def build_relief_index(year: int, through: str,
                       games_fetcher=None, boxscore_fetcher=None) -> dict:
    """一次掃全季 Final 賽,回 {team_id(str): [{date,er,ip}]}。fetchers 預設真連線,可注入供測試。"""
    games_fetcher = games_fetcher or _fetch_season_final_games
    boxscore_fetcher = boxscore_fetcher or _fetch_boxscore
    index: dict[str, list] = {}
    for g in games_fetcher(year, through):
        box = boxscore_fetcher(g["game_pk"])
        teams = box.get("teams", {})
        for side_key, tid in (("home", g["home_id"]), ("away", g["away_id"])):
            er, ip = relief_er_ip(teams.get(side_key, {}))
            index.setdefault(str(tid), []).append({"date": g["date"], "er": er, "ip": ip})
    return index


def _cache_path(year: int, cache_dir) -> Path:
    return Path(cache_dir) / f"relief_index_{year}.json"


def load_or_build_index(year: int, needed_through: str,
                        cache_dir=DEFAULT_CACHE_DIR, refresh: bool = False) -> dict:
    """載入快取(覆蓋足夠才用)否則重建。covers iff built_through >= needed_through。"""
    p = _cache_path(year, cache_dir)
    if p.exists() and not refresh:
        cached = json.loads(p.read_text(encoding="utf-8"))
        if cached.get("built_through", "") >= needed_through:
            return cached["index"]
    index = build_relief_index(year, needed_through)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"built_through": needed_through, "index": index}, ensure_ascii=False),
                 encoding="utf-8")
    return index


def relief_era(team_id: int, year: int, as_of: str,
               cache_dir=DEFAULT_CACHE_DIR, fallback: float = 4.00) -> float:
    """team_id 在 as_of(不含當日)之前的 point-in-time relief ERA。無資料 → fallback。"""
    index = load_or_build_index(year, needed_through=as_of, cache_dir=cache_dir)
    val = relief_era_as_of(index.get(str(team_id), []), as_of)
    return val if val is not None else fallback


def relief_ip_last_k(team_id, year, as_of, k=2, cache_dir=DEFAULT_CACHE_DIR, index=None) -> float:
    """team_id 在 as_of 前 k 天(不含 as_of 當日)的後援投球局數總和。index 可注入避免重載。"""
    if index is None:
        index = load_or_build_index(year, needed_through=as_of, cache_dir=cache_dir)
    a = datetime.strptime(as_of, "%Y-%m-%d")
    lo = a - timedelta(days=k)
    tot = 0.0
    for e in index.get(str(team_id), []):
        d = datetime.strptime(e["date"], "%Y-%m-%d")
        if lo <= d < a:
            tot += e["ip"]
    return round(tot, 4)
