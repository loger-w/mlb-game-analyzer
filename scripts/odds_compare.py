"""找預測當下的最新 Pinnacle snapshot,抽 RL+總分 no-vig,算 vs model 的 edge。"""
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
SNAPSHOTS_DIR = SKILL_ROOT / "odds" / "odds_snapshots"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.closing_line import (
    _parse_iso_utc, extract_pinnacle_no_vig, extract_pinnacle_rl_no_vig,
)


def find_latest_snapshot_for_game(date: str, home_team: str, away_team: str,
                                  snapshots_dir: Path = SNAPSHOTS_DIR) -> tuple[dict | None, str | None]:
    """掃 odds_snapshots,挑「snapshot_time 最新且 < 開球」且含此 matchup 的那筆。"""
    best = None  # (snap_ts, game_dict, filename)
    for f in sorted(Path(snapshots_dir).glob(f"{date}_*-ET.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        snap_ts = _parse_iso_utc(data.get("snapshot_time_utc", ""))
        if snap_ts is None:
            continue
        for g in data.get("games", []):
            if g.get("home_team") != home_team or g.get("away_team") != away_team:
                continue
            commence = _parse_iso_utc(g.get("commence_utc", ""))
            if commence is None or snap_ts >= commence:
                continue
            if best is None or snap_ts > best[0]:
                best = (snap_ts, g, f.name)
    if best is None:
        return None, None
    return best[1], best[2]


def market_from_snapshot(game: dict) -> dict | None:
    """從 snapshot game 抽 RL + 總分 no-vig。任一缺 → None。"""
    ml_total = extract_pinnacle_no_vig(game)   # 含 total_line / over_no_vig / under_no_vig
    rl = extract_pinnacle_rl_no_vig(game)
    if ml_total is None or rl is None:
        return None
    return {
        "rl": rl,
        "total": {"line": ml_total["total_line"],
                  "over_no_vig": ml_total["over_no_vig"],
                  "under_no_vig": ml_total["under_no_vig"]},
    }


def compute_edges(model: dict, market: dict | None) -> dict:
    """edge(pp) = (model 機率 − 市場 no-vig) × 100。market None → 全 None。"""
    if not market:
        return {"home_rl_pp": None, "over_pp": None}
    home_rl_pp = None
    if model.get("p_home_cover_rl") is not None:
        home_rl_pp = round((model["p_home_cover_rl"] - market["rl"]["home_no_vig"]) * 100, 1)
    over_pp = None
    if model.get("p_over") is not None:
        over_pp = round((model["p_over"] - market["total"]["over_no_vig"]) * 100, 1)
    return {"home_rl_pp": home_rl_pp, "over_pp": over_pp}
