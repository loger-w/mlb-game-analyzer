"""抓單場模型輸入 + 凍結用打線快照。純計算與 I/O 分離,純計算可單測。"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config
from _team_resolver import resolve_team_id, team_abbr
from park_factors_lib import runs_pf

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def parse_ip(ip_str) -> float:
    """MLB inningsPitched 字串 → float(.1=1/3, .2=2/3)。"""
    whole, _, frac = str(ip_str).partition(".")
    thirds = {"1": 1/3, "2": 2/3}.get(frac, 0.0)
    return int(whole or 0) + thirds


def calc_fip(*, hr: int, bb: int, hbp: int, k: int, ip: float) -> float | None:
    """標準 FIP(含 HBP)。IP < config.MIN_IP → None(樣本太小)。"""
    if ip < config.MIN_IP:
        return None
    return round((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + config.FIP_CONSTANT, 2)


def rs_blend(recent: float, season: float) -> float:
    """RS 近期/整季加權混合。"""
    return config.RECENT_W * recent + (1 - config.RECENT_W) * season


def fetch_schedule_game(date: str, home_id: int, away_id: int) -> dict | None:
    """抓當日賽程,回傳該 matchup 的 game dict(含 probablePitcher)。"""
    params = {"sportId": 1, "date": date, "hydrate": "probablePitcher(note)"}
    r = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=15)
    r.raise_for_status()
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            if g.get("gameType") != "R":
                continue
            h = g["teams"]["home"]["team"]["id"]
            a = g["teams"]["away"]["team"]["id"]
            if h == home_id and a == away_id:
                return g
    return None


def _team_rs_ra(team_id: int, before_date: str) -> dict:
    """近 RECENT_N 場 + 整季的 RS/RA per game。沿用 schedule+linescore 模式。"""
    def _games(start_days_back: int | None):
        end = datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=1)
        start = (datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=start_days_back)
                 if start_days_back else datetime(end.year, 3, 20))
        params = {"sportId": 1, "teamId": team_id, "startDate": start.strftime("%Y-%m-%d"),
                  "endDate": end.strftime("%Y-%m-%d"), "hydrate": "linescore"}
        r = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=15)
        r.raise_for_status()
        rows = []
        for d in r.json().get("dates", []):
            for g in d.get("games", []):
                if g["status"]["abstractGameState"] != "Final" or g.get("gameType", "R") != "R":
                    continue
                h = g["teams"]["home"]; a = g["teams"]["away"]
                is_home = h["team"]["id"] == team_id
                me, opp = (h, a) if is_home else (a, h)
                rows.append((me.get("score") or 0, opp.get("score") or 0, g["gameDate"][:10]))
        rows.sort(key=lambda x: x[2], reverse=True)
        return rows

    season = _games(None)
    recent = season[:config.RECENT_N]

    def _per_game(rows, idx):
        return round(sum(r[idx] for r in rows) / len(rows), 2) if rows else config.LEAGUE_RG

    return {
        "rs_recent": _per_game(recent, 0), "ra_recent": _per_game(recent, 1),
        "rs_season": _per_game(season, 0), "ra_season": _per_game(season, 1),
    }


def _stat_from_byrange_splits(splits: list) -> dict | None:
    """byDateRange 的 splits 可能重複;取第一筆彙總(IP 已等於整段總和)。空 → None。"""
    if not splits:
        return None
    return splits[0].get("stat", {})


def fetch_starter(mlbam_id: int | None, name: str, year: int, end_date: str) -> dict:
    """先發 point-in-time 成績(賽季起 → end_date,含)並算 FIP。id 缺 / 無成績 → fip=None。"""
    base = {"name": name, "id": mlbam_id, "fip": None,
            "ip": None, "k": None, "bb": None, "hbp": None, "hr": None}
    if not mlbam_id:
        return base
    try:
        r = requests.get(f"{MLB_API_BASE}/people/{mlbam_id}/stats",
                         params={"stats": "byDateRange", "group": "pitching", "season": year,
                                 "startDate": f"{year}-03-01", "endDate": end_date},
                         timeout=10)
        r.raise_for_status()
        splits = (r.json().get("stats") or [{}])[0].get("splits") or []
        s = _stat_from_byrange_splits(splits)
        if not s:
            return base
        ip = parse_ip(s.get("inningsPitched", "0"))
        k = int(s.get("strikeOuts", 0)); bb = int(s.get("baseOnBalls", 0))
        hbp = int(s.get("hitByPitch", 0)); hr = int(s.get("homeRuns", 0))
        base.update(ip=ip, k=k, bb=bb, hbp=hbp, hr=hr,
                    fip=calc_fip(hr=hr, bb=bb, hbp=hbp, k=k, ip=ip))
        return base
    except Exception as e:
        print(f"[fetch_inputs] starter {mlbam_id} 失敗:{e}", file=sys.stderr)
        return base


def fetch_bullpen_era(team_id: int, year: int, as_of: str) -> float:
    """牛棚 relief ERA(point-in-time,不含 as_of 當日)。委派 bullpen.relief_era;無資料 → 4.00。"""
    import bullpen
    return bullpen.relief_era(team_id, year, as_of)


def fetch_lineup_light(team_id: int, game_pk: int, year: int) -> list[dict]:
    """凍結用輕量打線:有官方先發打線就抓 9 人(order/name/id),否則回 []。
    只取名單與打序;進階攻擊值留空(v1 不進模型,凍結為日後 ablation 保留欄位)。"""
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", timeout=10)
        r.raise_for_status()
        box = r.json().get("liveData", {}).get("boxscore", {}).get("teams", {})
        side = "home" if box.get("home", {}).get("team", {}).get("id") == team_id else "away"
        order = box.get(side, {}).get("battingOrder", []) or []
        players = box.get(side, {}).get("players", {})
        out = []
        for i, pid in enumerate(order[:9]):
            p = players.get(f"ID{pid}", {})
            out.append({"order": i + 1, "name": p.get("person", {}).get("fullName"),
                        "id": pid, "ops": None, "woba": None})
        return out
    except Exception:
        return []


def assemble_inputs(raw: dict) -> dict:
    """純組裝:raw(各抓取結果) → run_model 可吃的扁平 dict + 透傳 raw 供凍結。"""
    return {
        "home_rs_blend": round(rs_blend(raw["home_rs_recent"], raw["home_rs_season"]), 3),
        "away_rs_blend": round(rs_blend(raw["away_rs_recent"], raw["away_rs_season"]), 3),
        "home_starter_fip": raw["home_starter"]["fip"] if raw["home_starter"]["fip"] is not None else config.LEAGUE_RG,
        "away_starter_fip": raw["away_starter"]["fip"] if raw["away_starter"]["fip"] is not None else config.LEAGUE_RG,
        "home_bullpen_era": raw["home_bullpen_era"],
        "away_bullpen_era": raw["away_bullpen_era"],
        "park_factor": raw["park_factor"],
        "raw": raw,
    }


def fetch_inputs(date: str, away: str, home: str) -> dict:
    """主入口:抓齊一場的輸入。回傳 assemble_inputs(raw) 結果(含 raw 供凍結)。"""
    home_id = resolve_team_id(home)
    away_id = resolve_team_id(away)
    year = int(date[:4])
    cutoff = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    game = fetch_schedule_game(date, home_id, away_id)
    if game is None:
        raise ValueError(f"找不到 {away}@{home} 於 {date} 的例行賽")

    gi_home = game["teams"]["home"]; gi_away = game["teams"]["away"]
    game_pk = game["gamePk"]
    venue = game["venue"]["name"]
    home_pp = gi_home.get("probablePitcher", {})
    away_pp = gi_away.get("probablePitcher", {})

    home_form = _team_rs_ra(home_id, date)
    away_form = _team_rs_ra(away_id, date)
    home_starter = fetch_starter(home_pp.get("id"), home_pp.get("fullName", "TBD"), year, cutoff)
    away_starter = fetch_starter(away_pp.get("id"), away_pp.get("fullName", "TBD"), year, cutoff)

    raw = {
        "game": {"date": date, "game_pk": game_pk, "venue": venue,
                 "home": {"team": gi_home["team"]["name"], "team_id": home_id,
                          "probable_pitcher": home_starter["name"], "probable_pitcher_id": home_pp.get("id")},
                 "away": {"team": gi_away["team"]["name"], "team_id": away_id,
                          "probable_pitcher": away_starter["name"], "probable_pitcher_id": away_pp.get("id")}},
        "home_rs_recent": home_form["rs_recent"], "home_rs_season": home_form["rs_season"],
        "away_rs_recent": away_form["rs_recent"], "away_rs_season": away_form["rs_season"],
        "home_ra_recent": home_form["ra_recent"], "home_ra_season": home_form["ra_season"],
        "away_ra_recent": away_form["ra_recent"], "away_ra_season": away_form["ra_season"],
        "home_starter": home_starter, "away_starter": away_starter,
        "home_bullpen_era": fetch_bullpen_era(home_id, year, date),
        "away_bullpen_era": fetch_bullpen_era(away_id, year, date),
        "park_factor": runs_pf(venue),
        "lineup_frozen": {"source": "official",
                          "home": fetch_lineup_light(home_id, game_pk, year),
                          "away": fetch_lineup_light(away_id, game_pk, year)},
    }
    if not raw["lineup_frozen"]["home"] and not raw["lineup_frozen"]["away"]:
        raw["lineup_frozen"]["source"] = "projected"
    return assemble_inputs(raw)
