"""凍結 features.json(回測/ablation) + 產 prediction.md(AI 敘事素材)。"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config


def build_features(b: dict) -> dict:
    """組 features.json(schema v2)。b = orchestrator 收集的 bundle。"""
    inp = b["inputs"]; raw = inp["raw"]; g = raw["game"]
    market = b.get("market")
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "game": {"date": g["date"], "game_pk": g["game_pk"],
                 "home": g["home"]["team"], "away": g["away"]["team"], "venue": g["venue"]},
        "inputs": {
            "home_rs_recent": raw["home_rs_recent"], "home_rs_season": raw["home_rs_season"],
            "away_rs_recent": raw["away_rs_recent"], "away_rs_season": raw["away_rs_season"],
            "home_ra_recent": raw["home_ra_recent"], "home_ra_season": raw["home_ra_season"],
            "away_ra_recent": raw["away_ra_recent"], "away_ra_season": raw["away_ra_season"],
            "home_starter": raw["home_starter"], "away_starter": raw["away_starter"],
            "home_bullpen_era": inp["home_bullpen_era"], "away_bullpen_era": inp["away_bullpen_era"],
            "park_factor": inp["park_factor"], "league_rg_used": config.LEAGUE_RG,
        },
        "lineup_frozen": raw["lineup_frozen"],
        "model": {**b["model"], "constants_used": config.constants_snapshot()},
        "odds": ({"snapshot_file": b.get("snapshot_file"),
                  "rl": market["rl"], "total": market["total"]} if market else None),
        "edges": b["edges"],
    }


def _pct(x) -> str:
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "—"


def render_prediction_md(b: dict) -> str:
    g = b["inputs"]["raw"]["game"]; m = b["model"]; mk = b.get("market"); e = b["edges"]
    home = g["home"]["team"]; away = g["away"]["team"]
    lines = [
        f"## {away} @ {home} — {g['date']}",
        f"- 期望得分:HOME {m['mu_home']} / AWAY {m['mu_away']}(total {m['mu_total']})",
        "",
        "| 市場 | 線 | model 機率 | 市場 no-vig | edge(pp) |",
        "|------|----|-----------|-------------|----------|",
    ]
    if mk:
        rl = mk["rl"]; tot = mk["total"]
        p_home = m["p_home_cover_rl"]; p_away = (1 - p_home) if p_home is not None else None
        p_over = m["p_over"]; p_under = (1 - p_over) if p_over is not None else None
        e_rl = e["home_rl_pp"]; e_ov = e["over_pp"]
        e_rl_a = (-e_rl) if isinstance(e_rl, (int, float)) else None
        e_ov_u = (-e_ov) if isinstance(e_ov, (int, float)) else None
        lines += [
            f"| RL HOME | {rl['home_point']:+} | {_pct(p_home)} | {_pct(rl['home_no_vig'])} | {e_rl:+} |",
            f"| RL AWAY | {rl['away_point']:+} | {_pct(p_away)} | {_pct(rl['away_no_vig'])} | {e_rl_a:+} |",
            f"| Over | {tot['line']} | {_pct(p_over)} | {_pct(tot['over_no_vig'])} | {e_ov:+} |",
            f"| Under | {tot['line']} | {_pct(p_under)} | {_pct(tot['under_no_vig'])} | {e_ov_u:+} |",
            "",
            f"- 所用盤口 snapshot:{b.get('snapshot_file', '—')}",
        ]
    else:
        lines += [
            f"| RL HOME | — | {_pct(m['p_home_cover_rl'])} | — | — |",
            f"| Over | — | {_pct(m['p_over'])} | — | — |",
            "",
            "- ⚠️ 無盤口可比(snapshot 缺或未開盤),只輸出 model 機率。",
        ]
    lines += [
        "",
        "<!-- AI 敘事:哪邊有正 edge、量級、需注意什麼。"
        "不喊「下哪邊」、不硬掰 EV%、只談 RL 與 O/U(不輸出勝負盤)。 -->",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(b: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = build_features(b)
    (out_dir / "features.json").write_text(
        json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "prediction.md").write_text(render_prediction_md(b), encoding="utf-8")
    return {"features": out_dir / "features.json", "prediction": out_dir / "prediction.md"}
