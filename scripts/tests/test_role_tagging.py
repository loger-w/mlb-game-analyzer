"""Tests for lib_role_tagging — pitcher role inference from season stats.

Pure-function library:
  tag_role(pitcher_stats, team_total_games=None) -> dict
  detect_committee_closer(roles: list[dict]) -> list[dict]   (mutates)

First-match-wins rules:
  GS ≥ 5 and GS ≥ 0.6 × G   → Starter (or Opener if avg IP/GS < 3.0)
  SV ≥ 8                    → Closer
  HLD ≥ 8                   → Setup
  HLD ≥ 3 or SV ≥ 2         → High-leverage RP
  IP/G ≥ 2.0 and G ≥ 5      → Long RP
  G ≥ 10                    → Middle RP
  else                      → Unknown
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _stats(saves=0, holds=0, g=0, gs=0, ip=0.0):
    return {"saves": saves, "holds": holds, "g": g, "gs": gs, "ip": ip}


# ---------------------------------------------------------------------------
# tag_role — single-pitcher classification
# ---------------------------------------------------------------------------

def test_tag_role_starter_data_confidence():
    from lib_role_tagging import tag_role
    result = tag_role(_stats(g=10, gs=10, ip=60.0))
    assert result["core_role"] == "Starter"
    assert result["core_role_confidence"] == "data"
    assert result["core_role_small_sample"] is False
    assert result["core_role_evidence"]["gs"] == 10


def test_tag_role_opener_when_short_starts():
    """GS=6, total IP=12 → avg IP/start = 2.0 → Opener (not Starter)."""
    from lib_role_tagging import tag_role
    result = tag_role(_stats(g=6, gs=6, ip=12.0))
    assert result["core_role"] == "Opener"
    assert result["core_role_confidence"] == "heuristic"


def test_tag_role_closer_via_saves():
    from lib_role_tagging import tag_role
    result = tag_role(_stats(saves=14, holds=0, g=22, gs=0, ip=22.0))
    assert result["core_role"] == "Closer"
    assert result["core_role_confidence"] == "data"


def test_tag_role_setup_via_holds():
    from lib_role_tagging import tag_role
    result = tag_role(_stats(saves=2, holds=12, g=20, gs=0, ip=20.0))
    assert result["core_role"] == "Setup"
    assert result["core_role_confidence"] == "data"


def test_tag_role_high_leverage_rp_via_holds():
    """HLD=4 (≥3 but <8), no save dominance → High-leverage RP heuristic."""
    from lib_role_tagging import tag_role
    result = tag_role(_stats(saves=0, holds=4, g=15, gs=0, ip=15.0))
    assert result["core_role"] == "High-leverage RP"
    assert result["core_role_confidence"] == "heuristic"


def test_tag_role_high_leverage_rp_via_saves_2_to_7():
    """Save count 2-7 (not enough for Closer) → High-leverage RP."""
    from lib_role_tagging import tag_role
    result = tag_role(_stats(saves=3, holds=1, g=15, gs=0, ip=15.0))
    assert result["core_role"] == "High-leverage RP"


def test_tag_role_long_rp():
    """G=8, IP=20 → avg 2.5 IP/G → Long RP heuristic (not enough for Starter)."""
    from lib_role_tagging import tag_role
    result = tag_role(_stats(saves=0, holds=0, g=8, gs=0, ip=20.0))
    assert result["core_role"] == "Long RP"
    assert result["core_role_confidence"] == "heuristic"


def test_tag_role_middle_rp_default_for_active_relievers():
    """G=15, IP=15 (avg 1.0/G), HLD=1, SV=0 → Middle RP."""
    from lib_role_tagging import tag_role
    result = tag_role(_stats(saves=0, holds=1, g=15, gs=0, ip=15.0))
    assert result["core_role"] == "Middle RP"


def test_tag_role_unknown_when_too_few_appearances():
    """G=2, IP=2 → no rule fires → Unknown insufficient."""
    from lib_role_tagging import tag_role
    result = tag_role(_stats(saves=0, holds=0, g=2, gs=0, ip=2.0))
    assert result["core_role"] == "Unknown"
    assert result["core_role_confidence"] == "insufficient"


def test_tag_role_april_small_sample_flag():
    """team_total_games < 30 → core_role_small_sample True regardless of role."""
    from lib_role_tagging import tag_role
    result = tag_role(_stats(saves=8, g=12, ip=12.0), team_total_games=15)
    assert result["core_role"] == "Closer"
    assert result["core_role_small_sample"] is True


def test_tag_role_april_small_sample_false_when_team_has_30_games():
    from lib_role_tagging import tag_role
    result = tag_role(_stats(saves=8, g=12, ip=12.0), team_total_games=30)
    assert result["core_role_small_sample"] is False


def test_tag_role_handles_missing_stats_gracefully():
    """All stats missing → Unknown insufficient (no crash)."""
    from lib_role_tagging import tag_role
    result = tag_role({})
    assert result["core_role"] == "Unknown"


# ---------------------------------------------------------------------------
# detect_committee_closer — relabel two high-leverage RPs as Co-Closer
# ---------------------------------------------------------------------------

def test_detect_committee_closer_two_high_leverage_with_saves():
    """Two pitchers each tagged High-leverage RP with SV≥4 → both Co-Closer."""
    from lib_role_tagging import tag_role, detect_committee_closer
    roles = [
        tag_role(_stats(saves=5, holds=2, g=18, ip=18.0)),  # High-leverage RP
        tag_role(_stats(saves=4, holds=3, g=15, ip=15.0)),  # High-leverage RP
        tag_role(_stats(saves=0, holds=12, g=20, ip=20.0)), # Setup
    ]
    # Sanity: first two pre-detect should be High-leverage RP
    assert roles[0]["core_role"] == "High-leverage RP"
    assert roles[1]["core_role"] == "High-leverage RP"

    detect_committee_closer(roles)
    assert roles[0]["core_role"] == "Co-Closer"
    assert roles[1]["core_role"] == "Co-Closer"
    assert roles[2]["core_role"] == "Setup"  # untouched


def test_detect_committee_closer_negative_when_real_closer_exists():
    """One pitcher SV≥8 (real Closer) + one HL with SV=4 → no committee."""
    from lib_role_tagging import tag_role, detect_committee_closer
    roles = [
        tag_role(_stats(saves=14, g=20, ip=20.0)),  # Closer (SV>=8)
        tag_role(_stats(saves=4, holds=3, g=15, ip=15.0)),  # High-leverage RP
    ]
    detect_committee_closer(roles)
    assert roles[0]["core_role"] == "Closer"  # still real closer
    assert roles[1]["core_role"] == "High-leverage RP"  # not promoted


def test_detect_committee_closer_negative_when_only_one_qualifies():
    """Only one high-leverage RP has SV≥4 → no committee (single high-leverage closer)."""
    from lib_role_tagging import tag_role, detect_committee_closer
    roles = [
        tag_role(_stats(saves=5, holds=2, g=18, ip=18.0)),  # HL with saves
        tag_role(_stats(saves=1, holds=4, g=15, ip=15.0)),  # HL via holds, SV<4
    ]
    detect_committee_closer(roles)
    assert roles[0]["core_role"] == "High-leverage RP"
    assert roles[1]["core_role"] == "High-leverage RP"


# ---------------------------------------------------------------------------
# roster_checker.enrich_roster_with_roles — integration with parsed roster
# ---------------------------------------------------------------------------

def _parsed_roster_fixture():
    """Minimal parse_roster output shape (active + 40man combined output)."""
    return {
        "active_roster": {
            "pitchers": ["Devin Williams", "Edwin Diaz", "Carlos Estévez"],
            "pitcher_ids": [642207, 621242, 595014],
            "position_players": ["Bobby Witt Jr."],
        },
        "injured_list": [
            {"name": "Felix Bautista", "status": "Injured 15-Day",
             "position": "Pitcher", "player_id": 671737},
            {"name": "Jonathan India", "status": "Injured 10-Day",
             "position": "Second Base", "player_id": 663697},
        ],
        "not_active_40man": [],
        "summary": {"total_active": 4, "total_active_pitchers": 3,
                    "total_active_position": 1, "total_il": 2,
                    "total_40man_not_active": 0},
    }


def _stats_map_fixture():
    return {
        # Devin Williams — Closer profile
        642207: {"saves": 14, "holds": 0, "g": 22, "gs": 0, "ip": 22.0},
        # Edwin Diaz — also Closer (will become Co-Closer if pattern detected)
        621242: {"saves": 12, "holds": 1, "g": 20, "gs": 0, "ip": 20.0},
        # Carlos Estévez — Setup
        595014: {"saves": 1, "holds": 12, "g": 18, "gs": 0, "ip": 18.0},
        # Felix Bautista (IL) — was Closer pre-injury
        671737: {"saves": 8, "holds": 1, "g": 12, "gs": 0, "ip": 12.0},
    }


def test_enrich_with_roles_decorates_active_pitchers():
    from roster_checker import enrich_roster_with_roles
    parsed = _parsed_roster_fixture()
    stats = _stats_map_fixture()
    enriched = enrich_roster_with_roles(parsed, stats)

    pitcher_roles = enriched["active_roster"].get("pitcher_roles")
    assert pitcher_roles is not None
    assert len(pitcher_roles) == 3
    # Each entry has the role schema
    for entry in pitcher_roles:
        assert "name" in entry
        assert "player_id" in entry
        assert "core_role" in entry
        assert "core_role_confidence" in entry

    # Williams + Diaz: SV>=8 each → both Closer (data confidence)
    by_name = {p["name"]: p for p in pitcher_roles}
    assert by_name["Devin Williams"]["core_role"] == "Closer"
    assert by_name["Edwin Diaz"]["core_role"] == "Closer"
    # Estevez: HLD=12 → Setup
    assert by_name["Carlos Estévez"]["core_role"] == "Setup"


def test_enrich_with_roles_decorates_il_pitchers_with_role():
    from roster_checker import enrich_roster_with_roles
    parsed = _parsed_roster_fixture()
    stats = _stats_map_fixture()
    enriched = enrich_roster_with_roles(parsed, stats)

    bautista = next(p for p in enriched["injured_list"] if p["name"] == "Felix Bautista")
    assert bautista["core_role"] == "Closer"  # SV=8 → Closer
    assert bautista["core_role_confidence"] == "data"

    # Non-pitcher IL (Jonathan India) NOT decorated with role
    india = next(p for p in enriched["injured_list"] if p["name"] == "Jonathan India")
    assert "core_role" not in india


def test_enrich_with_roles_missing_pid_falls_back_to_unknown():
    from roster_checker import enrich_roster_with_roles
    parsed = _parsed_roster_fixture()
    stats = {}  # no stats fetched
    enriched = enrich_roster_with_roles(parsed, stats)

    pitcher_roles = enriched["active_roster"]["pitcher_roles"]
    for entry in pitcher_roles:
        assert entry["core_role"] == "Unknown"
        assert entry["core_role_confidence"] == "insufficient"


def test_enrich_with_roles_preserves_existing_pitchers_list_strings():
    """Backward-compat: active_roster.pitchers stays list[str], unchanged."""
    from roster_checker import enrich_roster_with_roles
    parsed = _parsed_roster_fixture()
    stats = _stats_map_fixture()
    enriched = enrich_roster_with_roles(parsed, stats)
    assert enriched["active_roster"]["pitchers"] == [
        "Devin Williams", "Edwin Diaz", "Carlos Estévez"
    ]
    assert all(isinstance(p, str) for p in enriched["active_roster"]["pitchers"])


def test_parse_roster_includes_pitcher_ids_parallel_to_pitchers():
    """Verify parse_roster grew the new pitcher_ids field (parallel order with pitchers)."""
    from roster_checker import parse_roster
    raw = {
        "roster": [
            {"person": {"id": 642207, "fullName": "Devin Williams"},
             "position": {"name": "Pitcher", "abbreviation": "P"},
             "status": {"code": "A", "description": "Active"}},
            {"person": {"id": 621242, "fullName": "Edwin Diaz"},
             "position": {"name": "Pitcher", "abbreviation": "P"},
             "status": {"code": "A", "description": "Active"}},
        ],
    }
    parsed = parse_roster(raw)
    # pitcher_ids must exist and have same order as pitchers
    assert "pitcher_ids" in parsed
    assert len(parsed["pitcher_ids"]) == len(parsed["pitchers"])
    assert isinstance(parsed["pitcher_ids"][0], int)


def test_parse_roster_il_pitcher_has_player_id():
    """IL entries must include player_id so enrich_with_roles can join with stats map."""
    from roster_checker import parse_roster
    raw = {
        "roster": [
            {"person": {"id": 671737, "fullName": "Felix Bautista"},
             "position": {"name": "Pitcher", "abbreviation": "P"},
             "status": {"code": "D15", "description": "Injured 15-Day"}},
        ],
    }
    parsed = parse_roster(raw)
    il = parsed["injured_list"]
    assert len(il) == 1
    assert il[0]["player_id"] == 671737


# ---------------------------------------------------------------------------
# Backlog #3 — tag_role honors `from_prior_year` flag (Bug 3 fallback support)
# ---------------------------------------------------------------------------


def test_tag_role_from_prior_year_appends_suffix_to_confidence():
    """When stats has from_prior_year=True (Bug 3 fallback), confidence label
    becomes "<base>, prior_year" so dossier / signal layers can mark the role
    inference as based on last-year data, not current."""
    from lib_role_tagging import tag_role
    stats = {"saves": 30, "holds": 0, "g": 60, "gs": 0, "ip": 60.0, "from_prior_year": True}
    result = tag_role(stats, team_total_games=None)
    assert result["core_role"] == "Closer"
    assert result["core_role_confidence"] == "data, prior_year"
    # Evidence carries the flag for full transparency at render time
    assert result["core_role_evidence"].get("from_prior_year") is True


def test_tag_role_from_prior_year_overrides_small_sample():
    """Prior-year stats represent a full season — small_sample (April-noise warning)
    must NOT fire even when team_total_games < 30, because the data itself is robust."""
    from lib_role_tagging import tag_role
    stats = {"saves": 30, "holds": 0, "g": 60, "gs": 0, "ip": 60.0, "from_prior_year": True}
    result = tag_role(stats, team_total_games=12)  # April team
    assert result["core_role"] == "Closer"
    assert result["core_role_small_sample"] is False
    # Sanity: without the flag, the same team_total_games would set small_sample True
    stats_no_flag = {"saves": 30, "holds": 0, "g": 60, "gs": 0, "ip": 60.0}
    result_no_flag = tag_role(stats_no_flag, team_total_games=12)
    assert result_no_flag["core_role_small_sample"] is True
