"""Team abbreviation/full-name → MLBAM team_id resolver.

Shared by fetch_game_data.py, lineup_analyzer.py, roster_checker.py.
Single source of truth for TEAM_MAP and FULL_NAMES.
"""

TEAM_MAP = {
    # English abbreviations
    "NYY": 147, "NYM": 121, "BOS": 111, "LAD": 119, "LAA": 108,
    "HOU": 117, "ATL": 144, "PHI": 143, "SD": 135, "SF": 137,
    "CHC": 112, "CWS": 145, "CIN": 113, "STL": 138, "MIL": 158,
    "PIT": 134, "ARI": 109, "COL": 115, "BAL": 110, "TB": 139,
    "TOR": 141, "MIN": 142, "KC": 118, "DET": 116, "CLE": 114,
    "SEA": 136, "OAK": 133, "TEX": 140, "MIA": 146, "WSH": 120,
    # Chinese names
    "洋基": 147, "大都會": 121, "紅襪": 111, "道奇": 119, "天使": 108,
    "太空人": 117, "勇士": 144, "費城人": 143, "教士": 135, "巨人": 137,
    "小熊": 112, "白襪": 145, "紅人": 113, "紅雀": 138, "釀酒人": 158,
    "海盜": 134, "響尾蛇": 109, "落磯": 115, "金鶯": 110, "光芒": 139,
    "藍鳥": 141, "雙城": 142, "皇家": 118, "老虎": 116, "守護者": 114,
    "水手": 136, "運動家": 133, "遊騎兵": 140, "馬林魚": 146, "國民": 120,
}

FULL_NAMES = {
    "new york yankees": 147, "new york mets": 121, "boston red sox": 111,
    "los angeles dodgers": 119, "los angeles angels": 108, "houston astros": 117,
    "atlanta braves": 144, "philadelphia phillies": 143, "san diego padres": 135,
    "san francisco giants": 137, "chicago cubs": 112, "chicago white sox": 145,
    "cincinnati reds": 113, "st. louis cardinals": 138, "milwaukee brewers": 158,
    "pittsburgh pirates": 134, "arizona diamondbacks": 109, "colorado rockies": 115,
    "baltimore orioles": 110, "tampa bay rays": 139, "toronto blue jays": 141,
    "minnesota twins": 142, "kansas city royals": 118, "detroit tigers": 116,
    "cleveland guardians": 114, "seattle mariners": 136, "athletics": 133,
    "texas rangers": 140, "miami marlins": 146, "washington nationals": 120,
}

# Reverse lookup: team_id → English abbreviation (uppercase ASCII keys only)
TEAM_ID_TO_ABBR = {tid: abbr for abbr, tid in TEAM_MAP.items() if abbr.isascii() and abbr.isupper()}


def resolve_team_id(team_input: str) -> int:
    """將隊名（縮寫 / 中文 / 全英文）轉為 MLBAM team_id。

    Raises ValueError when input cannot be resolved or is purely numeric
    (numeric IDs are no longer accepted — use abbreviations instead).
    """
    if team_input is None:
        raise ValueError("team_input cannot be None")
    s = str(team_input).strip()
    if not s:
        raise ValueError("team_input cannot be empty")
    # Reject numeric IDs explicitly (Plan: CLI 標準化 §C)
    if s.isdigit():
        raise ValueError(
            f'--team must be a team abbreviation (e.g., KC, LAA), got "{s}". '
            "Numeric team IDs are no longer accepted in this version."
        )
    upper = s.upper()
    if upper in TEAM_MAP:
        return TEAM_MAP[upper]
    if s in TEAM_MAP:
        return TEAM_MAP[s]
    lower = s.lower()
    if lower in FULL_NAMES:
        return FULL_NAMES[lower]
    for name, tid in FULL_NAMES.items():
        if lower in name:
            return tid
    raise ValueError(f"Unknown team: {team_input}")


def team_abbr(team_id: int | None, team_name: str = "") -> str:
    """team_id 優先反查 TEAM_ID_TO_ABBR；team_id 為 None 時用 team_name 透過 FULL_NAMES
    反查；都失敗 fallback 用 team_name 前 3 字大寫。"""
    if team_id is not None and team_id in TEAM_ID_TO_ABBR:
        return TEAM_ID_TO_ABBR[team_id]
    name_lower = (team_name or "").lower()
    if name_lower in FULL_NAMES:
        tid = FULL_NAMES[name_lower]
        if tid in TEAM_ID_TO_ABBR:
            return TEAM_ID_TO_ABBR[tid]
    return (team_name or "")[:3].upper()
