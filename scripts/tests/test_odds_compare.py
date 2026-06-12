import sys, json
from pathlib import Path
SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

from lib.closing_line import extract_pinnacle_rl_no_vig
import odds_compare as oc


def _sample_game_with_rl():
    return {
        "home_team": "Colorado Rockies", "away_team": "Arizona Diamondbacks",
        "bookmakers": {"pinnacle": {"rl": {
            "Colorado Rockies": {"point": -1.5, "no_vig_pct": 38.0},
            "Arizona Diamondbacks": {"point": 1.5, "no_vig_pct": 62.0},
        }}},
    }


def test_extract_rl_no_vig():
    out = extract_pinnacle_rl_no_vig(_sample_game_with_rl())
    assert out["home_point"] == -1.5
    assert round(out["home_no_vig"], 3) == 0.380
    assert out["away_point"] == 1.5
    assert round(out["away_no_vig"], 3) == 0.620


def test_extract_rl_missing_returns_none():
    assert extract_pinnacle_rl_no_vig({"home_team": "X", "away_team": "Y",
                                        "bookmakers": {}}) is None


def test_compute_edges():
    model = {"p_home_cover_rl": 0.355, "p_over": 0.479}
    market = {"rl": {"home_point": -1.5, "home_no_vig": 0.41,
                     "away_point": 1.5, "away_no_vig": 0.59},
              "total": {"line": 8.5, "over_no_vig": 0.52, "under_no_vig": 0.48}}
    edges = oc.compute_edges(model, market)
    assert round(edges["home_rl_pp"], 1) == -5.5   # (0.355-0.41)*100
    assert round(edges["over_pp"], 1) == -4.1       # (0.479-0.52)*100


def test_compute_edges_preserves_small_edges():
    """模型機率存 4 位小數,edge 取 1 位會把 <0.05pp 的真實 edge 捨成 0.0(下游
    metrics 用 >0 / !=0 判斷會整個漏掉)。edge 必須保留 2 位小數。"""
    model = {"p_home_cover_rl": 0.5004, "p_over": 0.5003}
    market = {"rl": {"home_point": -1.5, "home_no_vig": 0.5, "away_point": 1.5, "away_no_vig": 0.5},
              "total": {"line": 8.5, "over_no_vig": 0.5, "under_no_vig": 0.5}}
    edges = oc.compute_edges(model, market)
    assert edges["home_rl_pp"] == 0.04
    assert edges["over_pp"] == 0.03


def test_compute_edges_none_market():
    assert oc.compute_edges({"p_home_cover_rl": 0.5, "p_over": 0.5}, None) == {
        "home_rl_pp": None, "over_pp": None}


def test_find_latest_snapshot_returns_copy_not_cache_alias(tmp_path):
    """回傳的 game dict 不得是 _snapshots_for_date 快取的別名 —
    呼叫端就地改動不可污染同日後續查詢(predict --all 同日 15 場共用快取)。"""
    snap = {
        "snapshot_time_utc": "2026-05-02T16:00:00+00:00",
        "snapshot_time_et": "2026-05-02 12:00 ET",
        "games": [{
            "home_team": "H", "away_team": "A", "game_date_et": "2026-05-02",
            "commence_utc": "2026-05-02T22:00:00Z",
            "bookmakers": {"pinnacle": {"ml": {"H": {"no_vig_pct": 60.0}}}},
        }],
    }
    (tmp_path / "2026-05-02_12-00-ET.json").write_text(json.dumps(snap), encoding="utf-8")
    g1, _ = oc.find_latest_snapshot_for_game("2026-05-02", "H", "A", snapshots_dir=tmp_path)
    g1["home_team"] = "POLLUTED"
    g2, _ = oc.find_latest_snapshot_for_game("2026-05-02", "H", "A", snapshots_dir=tmp_path)
    assert g2 is not None and g2["home_team"] == "H"


def test_market_from_snapshot_roundtrip():
    g = {
        "home_team": "Colorado Rockies", "away_team": "Arizona Diamondbacks",
        "bookmakers": {"pinnacle": {
            "ml": {"Colorado Rockies": {"no_vig_pct": 45.0},
                   "Arizona Diamondbacks": {"no_vig_pct": 55.0}},
            "ou": {"Over": {"point": 11.5, "no_vig_pct": 50.0},
                   "Under": {"point": 11.5, "no_vig_pct": 50.0}},
            "rl": {"Colorado Rockies": {"point": -1.5, "no_vig_pct": 38.0},
                   "Arizona Diamondbacks": {"point": 1.5, "no_vig_pct": 62.0}},
        }},
    }
    m = oc.market_from_snapshot(g)
    assert m["rl"]["home_point"] == -1.5
    assert m["total"]["line"] == 11.5
    assert round(m["total"]["over_no_vig"], 3) == 0.500
