"""Tests for pitcher_stats.lookup_pitcher_id (diacritic fallback)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_lookup_stub(strict_df, fuzzy_df):
    """生成 monkeypatch 用的 playerid_lookup stub。

    呼叫順序：第 1 次（strict） → strict_df；第 2 次（fuzzy=True） → fuzzy_df。
    """
    calls = {"n": 0}

    def stub(last, first, fuzzy=False):
        calls["n"] += 1
        return fuzzy_df if fuzzy else strict_df

    return stub, calls


def test_lookup_strict_match_returns_id_no_fallback(monkeypatch):
    """strict match 成功 → 直接 return，不觸發 fuzzy"""
    import pitcher_stats
    strict = pd.DataFrame([{"key_mlbam": 676440, "mlb_played_last": 2026}])
    fuzzy = pd.DataFrame()  # 不該被讀
    stub, calls = _make_lookup_stub(strict, fuzzy)
    monkeypatch.setattr(pitcher_stats, "_import_pybaseball",
                        lambda: (stub, None, None, None))
    result = pitcher_stats.lookup_pitcher_id("Tanner Bibee")
    assert result == 676440
    assert calls["n"] == 1  # fuzzy 未呼叫


def test_lookup_diacritic_fallback_succeeds(monkeypatch, capsys):
    """ASCII 名字 strict 失敗 → fuzzy 成功 → 回傳 ID + stderr warning"""
    import pitcher_stats
    strict = pd.DataFrame()  # empty
    fuzzy = pd.DataFrame([{
        "key_mlbam": 607259,
        "name_first": "Nick",
        "name_last": "Martínez",
        "mlb_played_last": 2026,
    }])
    stub, calls = _make_lookup_stub(strict, fuzzy)
    monkeypatch.setattr(pitcher_stats, "_import_pybaseball",
                        lambda: (stub, None, None, None))
    result = pitcher_stats.lookup_pitcher_id("Nick Martinez")
    assert result == 607259
    assert calls["n"] == 2  # 兩次呼叫（strict 後 fuzzy）
    err = capsys.readouterr().err
    assert "fuzzy" in err.lower()
    assert "Martínez" in err


def test_lookup_fuzzy_year_filter_rejects_old_player(monkeypatch):
    """fuzzy 結果 mlb_played_last 早於 current_year - 1 → 拒絕，return None"""
    import pitcher_stats
    strict = pd.DataFrame()
    fuzzy = pd.DataFrame([{
        "key_mlbam": 100000,
        "name_first": "Old",
        "name_last": "Player",
        "mlb_played_last": 2010,  # 太舊
    }])
    stub, _ = _make_lookup_stub(strict, fuzzy)
    monkeypatch.setattr(pitcher_stats, "_import_pybaseball",
                        lambda: (stub, None, None, None))
    result = pitcher_stats.lookup_pitcher_id("Old Player")
    assert result is None


def test_lookup_fuzzy_multiple_results_picks_highest_year(monkeypatch):
    """fuzzy 多筆結果 → 取 mlb_played_last 最大者"""
    import pitcher_stats
    strict = pd.DataFrame()
    fuzzy = pd.DataFrame([
        {"key_mlbam": 111, "mlb_played_last": 2024},
        {"key_mlbam": 222, "mlb_played_last": 2026},  # 最新
        {"key_mlbam": 333, "mlb_played_last": 2025},
    ])
    stub, _ = _make_lookup_stub(strict, fuzzy)
    monkeypatch.setattr(pitcher_stats, "_import_pybaseball",
                        lambda: (stub, None, None, None))
    result = pitcher_stats.lookup_pitcher_id("Multi Match")
    assert result == 222


def test_lookup_both_empty_returns_none(monkeypatch):
    """strict + fuzzy 都 empty → return None"""
    import pitcher_stats
    stub, _ = _make_lookup_stub(pd.DataFrame(), pd.DataFrame())
    monkeypatch.setattr(pitcher_stats, "_import_pybaseball",
                        lambda: (stub, None, None, None))
    assert pitcher_stats.lookup_pitcher_id("Nonexistent Player") is None


def test_lookup_single_word_name_returns_none(monkeypatch):
    """單字名（無姓） → return None（既有行為保留）"""
    import pitcher_stats
    assert pitcher_stats.lookup_pitcher_id("Cher") is None


def test_lookup_strict_year_filtered_falls_to_fuzzy(monkeypatch):
    """strict match 找到但年份過舊（被 _resolve year filter 過濾） → 落入 fuzzy round 並回傳 fuzzy 結果"""
    import pitcher_stats
    # Round 1 strict 命中但 mlb_played_last=2010 → 過濾後視為失敗
    strict = pd.DataFrame([{
        "key_mlbam": 999,
        "name_first": "Old",
        "name_last": "Match",
        "mlb_played_last": 2010,
    }])
    # Round 2 fuzzy 命中現役球員
    fuzzy = pd.DataFrame([{
        "key_mlbam": 607259,
        "name_first": "Nick",
        "name_last": "Martínez",
        "mlb_played_last": 2026,
    }])
    stub, calls = _make_lookup_stub(strict, fuzzy)
    monkeypatch.setattr(pitcher_stats, "_import_pybaseball",
                        lambda: (stub, None, None, None))
    result = pitcher_stats.lookup_pitcher_id("Nick Martinez")
    assert result == 607259
    assert calls["n"] == 2  # 兩次都呼叫（strict 被年份濾掉，繼續 fuzzy）


# ---------------------------------------------------------------------------
# TTO splits helpers (Plan B — Statcast pitch-by-pitch aggregation)
# ---------------------------------------------------------------------------
import pandas as _pd_mod


def _pa_df(events: list[str]):
    """Build a tiny PA-level DataFrame for tests."""
    return _pd_mod.DataFrame({"events": events})


def test_pa_outcome_aggregates_all_strikeouts():
    """5 strikeouts → OPS=0、K%=100、BB%=0、BF=5."""
    from pitcher_stats import _pa_outcome_aggregates
    out = _pa_outcome_aggregates(_pa_df(["strikeout"] * 5))
    assert out["bf"] == 5
    assert out["k_pct"] == 100.0
    assert out["bb_pct"] == 0.0
    assert out["ops"] == 0.0


def test_pa_outcome_aggregates_basic_mix():
    """1 single + 1 walk + 1 K + 1 field_out + 1 home_run.

    AB = 5 - 1(BB) - 0(HBP) - 0(SF) - 0(SH) = 4
    H = 2 (single + HR), TB = 1 + 4 = 5
    OBP = (2 + 1) / (4 + 1) = 0.600
    SLG = 5 / 4 = 1.250
    OPS = 1.850
    K% = 1/5 = 20.0, BB% = 1/5 = 20.0
    """
    from pitcher_stats import _pa_outcome_aggregates
    out = _pa_outcome_aggregates(_pa_df([
        "single", "walk", "strikeout", "field_out", "home_run",
    ]))
    assert out["bf"] == 5
    assert out["k_pct"] == 20.0
    assert out["bb_pct"] == 20.0
    assert abs(out["ops"] - 1.850) < 0.005


def test_pa_outcome_aggregates_handles_sf_and_hbp():
    """SF doesn't count AB; HBP counts in OBP not AB.

    PAs: 2 single + 1 HBP + 1 SF + 1 K + 1 field_out → 6 PAs
    AB = 6 - 0(BB) - 1(HBP) - 1(SF) - 0(SH) = 4
    H = 2, TB = 2
    OBP = (2 + 0 + 1) / (4 + 0 + 1 + 1) = 3/6 = 0.500
    SLG = 2/4 = 0.500
    OPS = 1.000
    """
    from pitcher_stats import _pa_outcome_aggregates
    out = _pa_outcome_aggregates(_pa_df([
        "single", "single", "hit_by_pitch", "sac_fly", "strikeout", "field_out",
    ]))
    assert out["bf"] == 6
    assert abs(out["ops"] - 1.000) < 0.005


def test_pa_outcome_aggregates_empty_returns_zero_bf():
    from pitcher_stats import _pa_outcome_aggregates
    out = _pa_outcome_aggregates(_pa_df([]))
    assert out["bf"] == 0
    assert out["ops"] is None
