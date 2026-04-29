"""Tests for dossier_renderer."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_select_top5_pa_filter():
    """PA ≥ 30、IL'd 排除、最多 5 人，按 PA 降序"""
    from dossier_renderer import select_top5_vs_pitcher
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "season_ops": 0.800, "vs_rhp_ops": 0.750,
             "last7_ops": 0.700, "last7_babip": 0.300, "ev95_pct": 50.0, "barrel_pct": 10.0},
            {"name": "B", "pa": 80,  "season_ops": 0.750, "vs_rhp_ops": 0.700,
             "last7_ops": 0.650, "last7_babip": 0.280, "ev95_pct": 45.0, "barrel_pct": 8.0},
            {"name": "C", "pa": 25,  "season_ops": 0.900, "vs_rhp_ops": 0.850,
             "last7_ops": 0.800, "last7_babip": 0.330, "ev95_pct": 55.0, "barrel_pct": 12.0},  # PA < 30
            {"name": "D", "pa": 60,  "season_ops": 0.700, "vs_rhp_ops": 0.650,
             "last7_ops": 0.600, "last7_babip": 0.260, "ev95_pct": 40.0, "barrel_pct": 7.0},
            {"name": "E", "pa": 45,  "season_ops": 0.680, "vs_rhp_ops": 0.620,
             "last7_ops": 0.580, "last7_babip": 0.240, "ev95_pct": 38.0, "barrel_pct": 6.0},
            {"name": "F", "pa": 35,  "season_ops": 0.660, "vs_rhp_ops": 0.600,
             "last7_ops": 0.560, "last7_babip": 0.220, "ev95_pct": 36.0, "barrel_pct": 5.0},
            {"name": "G", "pa": 32,  "season_ops": 0.640, "vs_rhp_ops": 0.580,
             "last7_ops": 0.540, "last7_babip": 0.200, "ev95_pct": 34.0, "barrel_pct": 4.0},
        ]
    }
    il_names = set()
    result = select_top5_vs_pitcher(lineup, il_names)
    names = [p["name"] for p in result]
    assert names == ["A", "B", "D", "E", "F"]  # G 被擠出（取 top 5）；C 被 PA 過濾


def test_select_top5_excludes_il():
    """IL 名單上的球員直接濾掉"""
    from dossier_renderer import select_top5_vs_pitcher
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "season_ops": 0.8},
            {"name": "B", "pa": 80,  "season_ops": 0.75},
            {"name": "C", "pa": 60,  "season_ops": 0.7},
        ]
    }
    il_names = {"B"}
    result = select_top5_vs_pitcher(lineup, il_names)
    names = [p["name"] for p in result]
    assert names == ["A", "C"]


def test_select_top5_fewer_than_5_returns_what_exists():
    """候選池 < 5 → 返回所有合格者"""
    from dossier_renderer import select_top5_vs_pitcher
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "season_ops": 0.8},
            {"name": "B", "pa": 50,  "season_ops": 0.75},
        ]
    }
    result = select_top5_vs_pitcher(lineup, set())
    assert len(result) == 2


def test_select_top5_last7_top1_outside_pa_top5():
    """last7 OPS top1 不在 PA top5 內 → annotate"""
    from dossier_renderer import find_last7_top1_outside_pa_top5
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "last7_ops": 0.700},
            {"name": "B", "pa": 90,  "last7_ops": 0.650},
            {"name": "C", "pa": 80,  "last7_ops": 0.600},
            {"name": "D", "pa": 70,  "last7_ops": 0.550},
            {"name": "E", "pa": 60,  "last7_ops": 0.500},
            {"name": "Schneemann", "pa": 35, "last7_ops": 1.164},  # 不在 PA top5（被 E 擠掉）但 last7 OPS top1
        ]
    }
    pa_top5 = ["A", "B", "C", "D", "E"]
    annotation = find_last7_top1_outside_pa_top5(lineup, pa_top5, set())
    assert annotation is not None
    assert annotation["name"] == "Schneemann"
    assert annotation["last7_ops"] == 1.164


def test_select_top5_last7_top1_already_in_pa_top5_returns_none():
    """last7 OPS top1 已在 PA top5 內 → 不需 annotate"""
    from dossier_renderer import find_last7_top1_outside_pa_top5
    lineup = {
        "lineup": [
            {"name": "A", "pa": 100, "last7_ops": 1.000},  # PA top1 + last7 top1
            {"name": "B", "pa": 90,  "last7_ops": 0.500},
        ]
    }
    pa_top5 = ["A", "B"]
    assert find_last7_top1_outside_pa_top5(lineup, pa_top5, set()) is None
