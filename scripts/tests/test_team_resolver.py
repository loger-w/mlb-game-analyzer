"""Tests for _team_resolver shared module."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _team_resolver import (  # noqa: E402
    FULL_NAMES,
    TEAM_ID_TO_ABBR,
    TEAM_MAP,
    resolve_team_id,
    team_abbr,
)


def test_resolve_abbreviation():
    assert resolve_team_id("KC") == 118
    assert resolve_team_id("LAA") == 108
    assert resolve_team_id("NYY") == 147


def test_resolve_lowercase_abbreviation():
    assert resolve_team_id("kc") == 118
    assert resolve_team_id("nyy") == 147


def test_resolve_chinese_name():
    assert resolve_team_id("皇家") == 118
    assert resolve_team_id("天使") == 108
    assert resolve_team_id("洋基") == 147


def test_resolve_full_english_name():
    assert resolve_team_id("Kansas City Royals") == 118
    assert resolve_team_id("Los Angeles Angels") == 108


def test_resolve_fuzzy_partial_match():
    """部分英文名也應 fuzzy match。"""
    assert resolve_team_id("royals") == 118


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown team"):
        resolve_team_id("Imaginary Team")


def test_resolve_numeric_id_rejected():
    """純數字 ID 必須以明確訊息拒絕（CLI 標準化要求）。"""
    with pytest.raises(ValueError, match="abbreviation"):
        resolve_team_id("118")
    with pytest.raises(ValueError, match="abbreviation"):
        resolve_team_id("147")


def test_resolve_empty_raises():
    with pytest.raises(ValueError):
        resolve_team_id("")


def test_resolve_none_raises():
    with pytest.raises(ValueError):
        resolve_team_id(None)


def test_team_abbr_from_id():
    assert team_abbr(118, "Kansas City Royals") == "KC"
    assert team_abbr(108, "Los Angeles Angels") == "LAA"


def test_team_abbr_id_only():
    assert team_abbr(147) == "NYY"


def test_team_abbr_fallback_from_full_name():
    """team_id 為 None 時，用 team_name 透過 FULL_NAMES 反查。"""
    assert team_abbr(None, "Kansas City Royals") == "KC"


def test_team_abbr_fallback_first_3_chars():
    """team_id 為 None 且 team_name 不在 FULL_NAMES → 取前 3 字大寫。"""
    assert team_abbr(None, "unknown stadium") == "UNK"


def test_team_id_to_abbr_excludes_chinese():
    """TEAM_ID_TO_ABBR 應只含 ASCII 大寫縮寫，無中文。"""
    for v in TEAM_ID_TO_ABBR.values():
        assert v.isascii()
        assert v.isupper()


def test_30_teams_covered():
    """30 個 MLB 球隊都有英文縮寫對應。"""
    ascii_abbrs = {k for k in TEAM_MAP if k.isascii() and k.isupper()}
    assert len(ascii_abbrs) == 30


def test_full_names_lowercase_keys():
    """FULL_NAMES 的 key 必為小寫，方便不分大小寫的 lookup。"""
    for k in FULL_NAMES:
        assert k == k.lower()
