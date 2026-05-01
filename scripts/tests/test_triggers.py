"""Tests for Flag 8 (pitcher ERA-xERA gap) and Flag 3 (lineup BABIP) trigger detection."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Flag 8 — pitcher_stats.detect_triggers
# ---------------------------------------------------------------------------

def _pitcher_data(era=None, xera=None, ip=None, prior_era=None):
    return {
        "name": "Test Pitcher",
        "season": {"era": era, "ip": ip} if era is not None or ip is not None else {},
        "expected": {"xera": xera} if xera is not None else {},
        "prior_year": {"era": prior_era} if prior_era is not None else {},
    }


def test_flag8_gap_below_threshold_no_trigger():
    """gap = 1.49 (< 1.5) → 不觸發。"""
    from pitcher_stats import detect_triggers
    data = _pitcher_data(era=3.00, xera=1.51, ip=40)
    triggers = detect_triggers(data)
    assert triggers == [] or all(t.get("name") != "ERA-xERA gap" for t in triggers)


def test_flag8_gap_at_threshold_triggers():
    """gap = 1.50 (= 1.5) → 觸發（>= 閾值）。"""
    from pitcher_stats import detect_triggers
    data = _pitcher_data(era=4.00, xera=2.50, ip=40)
    triggers = detect_triggers(data)
    flag8 = [t for t in triggers if t.get("name") == "ERA-xERA gap"]
    assert len(flag8) == 1
    assert flag8[0]["flag"] == 8


def test_flag8_gap_above_threshold_triggers():
    """gap = 2.81 (>> 1.5) → 觸發（Lugo 真實案例）。"""
    from pitcher_stats import detect_triggers
    data = _pitcher_data(era=1.15, xera=3.96, ip=31.3, prior_era=4.15)
    triggers = detect_triggers(data)
    flag8 = [t for t in triggers if t.get("name") == "ERA-xERA gap"]
    assert len(flag8) == 1
    assert abs(flag8[0]["value"] - (-2.81)) < 0.01
    # 解讀應顯示「ERA 顯著低於 xERA」
    assert "低於" in flag8[0]["interpretation"] or "回升" in flag8[0]["interpretation"]


def test_flag8_negative_gap_inverts_interpretation():
    """ERA > xERA（正向 gap）解讀應為「壓制力被掩蓋」。"""
    from pitcher_stats import detect_triggers
    data = _pitcher_data(era=4.50, xera=2.50, ip=40)
    triggers = detect_triggers(data)
    flag8 = [t for t in triggers if t.get("name") == "ERA-xERA gap"]
    assert len(flag8) == 1
    assert flag8[0]["value"] > 0
    assert "高於" in flag8[0]["interpretation"] or "反彈" in flag8[0]["interpretation"]


def test_flag8_small_sample_regression():
    """IP < 30 且 prior_year ERA - current ERA >= 1.0 → 觸發 small-sample。"""
    from pitcher_stats import detect_triggers
    # gap 不觸發（避免重複），但 small-sample 條件成立
    data = _pitcher_data(era=2.00, xera=2.50, ip=20, prior_era=4.50)
    triggers = detect_triggers(data)
    small_sample = [t for t in triggers if t.get("name") == "Small-sample regression risk"]
    assert len(small_sample) == 1


def test_flag8_small_sample_not_triggered_when_ip_ge_30():
    from pitcher_stats import detect_triggers
    data = _pitcher_data(era=2.00, xera=2.50, ip=35, prior_era=4.50)
    triggers = detect_triggers(data)
    small_sample = [t for t in triggers if t.get("name") == "Small-sample regression risk"]
    assert len(small_sample) == 0


def test_flag8_small_sample_below_delta_no_trigger():
    from pitcher_stats import detect_triggers
    data = _pitcher_data(era=3.00, xera=2.50, ip=20, prior_era=3.50)  # delta = 0.5
    triggers = detect_triggers(data)
    small_sample = [t for t in triggers if t.get("name") == "Small-sample regression risk"]
    assert len(small_sample) == 0


def test_flag8_no_data_no_trigger():
    from pitcher_stats import detect_triggers
    assert detect_triggers({}) == []
    assert detect_triggers({"season": {}, "expected": {}}) == []


def test_flag8_error_section_no_trigger():
    """season/expected 含 error → 視為無資料，不觸發。"""
    from pitcher_stats import detect_triggers
    data = {
        "season": {"error": "lookup failed"},
        "expected": {"error": "no data"},
    }
    assert detect_triggers(data) == []


# ---------------------------------------------------------------------------
# Flag 3 — lineup_analyzer.detect_triggers (last7_babip)
# ---------------------------------------------------------------------------

def test_flag3_no_last7_babip_no_trigger():
    from lineup_analyzer import detect_triggers
    assert detect_triggers({}) == []
    assert detect_triggers({"last7_babip": None}) == []


def test_flag3_normal_value_no_trigger():
    from lineup_analyzer import detect_triggers
    assert detect_triggers({"last7_babip": 0.300}) == []
    # KC 案例：.326 在區間內
    assert detect_triggers({"last7_babip": 0.326}) == []


def test_flag3_below_low_boundary_triggers():
    from lineup_analyzer import detect_triggers
    triggers = detect_triggers({"last7_babip": 0.260})
    assert len(triggers) == 1
    assert triggers[0]["flag"] == 3
    assert "低" in triggers[0]["name"]


def test_flag3_below_low_boundary_strict_triggers():
    from lineup_analyzer import detect_triggers
    triggers = detect_triggers({"last7_babip": 0.250})
    assert len(triggers) == 1


def test_flag3_just_above_low_boundary_no_trigger():
    """0.261 不觸發（閾值是 ≤ 0.260）。"""
    from lineup_analyzer import detect_triggers
    assert detect_triggers({"last7_babip": 0.261}) == []


def test_flag3_at_high_boundary_triggers():
    from lineup_analyzer import detect_triggers
    triggers = detect_triggers({"last7_babip": 0.370})
    assert len(triggers) == 1
    assert triggers[0]["flag"] == 3
    assert "高" in triggers[0]["name"]


def test_flag3_above_high_boundary_triggers():
    from lineup_analyzer import detect_triggers
    triggers = detect_triggers({"last7_babip": 0.400})
    assert len(triggers) == 1


def test_flag3_just_below_high_boundary_no_trigger():
    """0.369 不觸發（閾值是 ≥ 0.370）。"""
    from lineup_analyzer import detect_triggers
    assert detect_triggers({"last7_babip": 0.369}) == []


def test_flag3_string_value_accepted():
    """last7_babip 為 string 數值也應處理。"""
    from lineup_analyzer import detect_triggers
    triggers = detect_triggers({"last7_babip": "0.250"})
    assert len(triggers) == 1


def test_flag3_invalid_value_no_crash():
    """非數值 last7_babip 應被忽略，不 crash。"""
    from lineup_analyzer import detect_triggers
    assert detect_triggers({"last7_babip": "N/A"}) == []
    assert detect_triggers({"last7_babip": "—"}) == []


# ---------------------------------------------------------------------------
# Roster Flag — roster_checker.detect_triggers (expected_starter)
# ---------------------------------------------------------------------------

def test_roster_starter_in_active_no_trigger():
    from roster_checker import detect_triggers
    data = {
        "active_roster": {"pitchers": ["Seth Lugo", "Cole Ragans"], "position_players": []},
        "injured_list": [],
    }
    assert detect_triggers(data, expected_starter="Seth Lugo") == []


def test_roster_starter_not_active_triggers():
    from roster_checker import detect_triggers
    data = {
        "active_roster": {"pitchers": ["Cole Ragans"], "position_players": []},
        "injured_list": [],
    }
    triggers = detect_triggers(data, expected_starter="Seth Lugo")
    assert len(triggers) == 1
    assert triggers[0]["flag"] == "STARTER_NOT_ACTIVE"
    assert triggers[0]["on_il"] is False


def test_roster_starter_on_il_triggers_with_il_marker():
    from roster_checker import detect_triggers
    data = {
        "active_roster": {"pitchers": [], "position_players": []},
        "injured_list": [{"name": "Seth Lugo", "status": "Injured 15-Day", "position": "Pitcher"}],
    }
    triggers = detect_triggers(data, expected_starter="Seth Lugo")
    assert len(triggers) == 1
    assert triggers[0]["on_il"] is True


def test_roster_no_expected_starter_no_trigger():
    from roster_checker import detect_triggers
    data = {"active_roster": {"pitchers": [], "position_players": []}, "injured_list": []}
    assert detect_triggers(data, expected_starter=None) == []
    assert detect_triggers(data) == []
