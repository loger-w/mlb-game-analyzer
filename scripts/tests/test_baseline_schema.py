"""Schema tests for data/league_pitcher_baseline.json.

The baseline JSON is the single source of truth for pitcher tier_v2 percentile
lookups (added in PR-2). This test only validates structure — the underlying
percentile values are refreshed yearly via scripts/refresh_baselines.py and
are not asserted against live league data here.
"""
import json
import os

BASELINE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "league_pitcher_baseline.json"
)

REQUIRED_METRICS = ("xfip", "k_bb_pct", "avg_velo", "stuff_plus", "pitching_plus")
PERCENTILE_KEYS = ("p10", "p25", "p50", "p75", "p90")
DIRECTION_VALUES = {"lower_is_better", "higher_is_better"}


def _load():
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_baseline_file_exists_and_parses():
    assert os.path.exists(BASELINE_PATH), f"baseline missing: {BASELINE_PATH}"
    data = _load()
    assert isinstance(data, dict)


def test_baseline_top_level_keys():
    data = _load()
    assert "year" in data and isinstance(data["year"], int)
    assert "qualifier_min_ip" in data and isinstance(data["qualifier_min_ip"], int)
    assert "metrics" in data and isinstance(data["metrics"], dict)
    assert "metadata" in data and isinstance(data["metadata"], dict)


def test_baseline_has_required_metrics():
    data = _load()
    metrics = data["metrics"]
    for m in REQUIRED_METRICS:
        assert m in metrics, f"missing metric: {m}"


def test_baseline_each_metric_has_5_percentiles_and_direction():
    data = _load()
    for m in REQUIRED_METRICS:
        block = data["metrics"][m]
        assert "direction" in block
        assert block["direction"] in DIRECTION_VALUES, (
            f"{m}.direction must be one of {DIRECTION_VALUES}"
        )
        for pk in PERCENTILE_KEYS:
            assert pk in block, f"{m} missing {pk}"
            assert isinstance(block[pk], (int, float))


def test_baseline_percentile_ordering_lower_is_better():
    """For lower_is_better metrics (xfip), p10 (top decile = best) should be
    smaller than p90 (bottom decile = worst)."""
    data = _load()
    for m, block in data["metrics"].items():
        if block["direction"] != "lower_is_better":
            continue
        # p10 is the best (smallest value), p90 is the worst (largest value)
        assert block["p10"] < block["p25"] < block["p50"] < block["p75"] < block["p90"], (
            f"{m}: lower_is_better expects p10 < p25 < p50 < p75 < p90, got "
            f"{[block[k] for k in PERCENTILE_KEYS]}"
        )


def test_baseline_percentile_ordering_higher_is_better():
    """For higher_is_better metrics (k_bb_pct, avg_velo), p10 (top decile = best)
    should be larger than p90 (bottom decile = worst)."""
    data = _load()
    for m, block in data["metrics"].items():
        if block["direction"] != "higher_is_better":
            continue
        # p10 is the best (largest value), p90 is the worst (smallest value)
        assert block["p10"] > block["p25"] > block["p50"] > block["p75"] > block["p90"], (
            f"{m}: higher_is_better expects p10 > p25 > p50 > p75 > p90, got "
            f"{[block[k] for k in PERCENTILE_KEYS]}"
        )
