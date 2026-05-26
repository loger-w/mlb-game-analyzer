"""Parse mlb-game-analyzer summary.md into structured prediction fields.

Schema (per spec §2 / §7):
  {direction: "HOME"|"AWAY"|"持平"|None,
   total: float|None,
   confidence: "LOW"|"MEDIUM"|"HIGH"|None,
   park_factor: float|None,
   has_reverse_platoon: bool,
   has_chain_break_300: bool,
   has_bullpen_il_2plus: bool,
   parse_failed: bool}
"""

import re
from pathlib import Path
from typing import Optional

# Team last-name (English) lookup. Built from common usage; not exhaustive but
# covers all 30 MLB teams' single-word last name as appears in finished summaries.
TEAM_LAST_NAME_TO_ABBR = {
    "yankees": "NYY", "mets": "NYM", "red sox": "BOS", "dodgers": "LAD",
    "angels": "LAA", "astros": "HOU", "braves": "ATL", "phillies": "PHI",
    "padres": "SD", "giants": "SF", "cubs": "CHC", "white sox": "CWS",
    "reds": "CIN", "cardinals": "STL", "brewers": "MIL", "pirates": "PIT",
    "diamondbacks": "ARI", "rockies": "COL", "orioles": "BAL", "rays": "TB",
    "blue jays": "TOR", "twins": "MIN", "royals": "KC", "tigers": "DET",
    "guardians": "CLE", "mariners": "SEA", "athletics": "ATH", "rangers": "TEX",
    "marlins": "MIA", "nationals": "WSH",
}

PINGPAN_KEYWORDS = ("持平", "勢均", "無明顯方向", "中性", "難以判定")


def _resolve_direction(phrase: str, home_abbr: str, away_abbr: str) -> Optional[str]:
    """Resolve direction from phrasing in '方向（基本面）' line.

    Strategy chain (first match wins):
      1. Explicit HOME / AWAY marker → use that
      2. 持平/勢均/中性 keyword → "持平"
      3. Team abbreviation (NYY, BAL, ...) → map to HOME/AWAY
      4. Team last name (Yankees, Orioles, ...) → map to abbr → HOME/AWAY
      5. Return None (couldn't resolve)
    """
    p = phrase.lower()

    # 1. Explicit marker
    if "home" in p and "away" not in p:
        return "HOME"
    if "away" in p and "home" not in p:
        return "AWAY"
    # Both present → fall through (rare; usually phrasing like "AWAY (ATL) vs HOME")
    # In that case first occurrence wins:
    if "home" in p and "away" in p:
        return "HOME" if p.index("home") < p.index("away") else "AWAY"

    # 2. 持平 keyword
    for kw in PINGPAN_KEYWORDS:
        if kw in phrase:
            return "持平"

    # 3. Team abbreviation
    for token in re.findall(r"\b([A-Z]{2,4})\b", phrase):
        if token == home_abbr:
            return "HOME"
        if token == away_abbr:
            return "AWAY"

    # 4. Team last name
    for name, abbr in TEAM_LAST_NAME_TO_ABBR.items():
        if name in p:
            if abbr == home_abbr:
                return "HOME"
            if abbr == away_abbr:
                return "AWAY"

    return None


_DIR_LINE_RE = re.compile(r"^-\s+\*\*方向（基本面）\*\*[:：]\s*(.+?)$", re.MULTILINE)
_TOTAL_LINE_RE = re.compile(
    r"^-\s+\*\*總分（基本面）\*\*[:：].*?adjusted\s+([0-9]+(?:\.[0-9]+)?)",
    re.MULTILINE | re.IGNORECASE,
)
_TOTAL_LINE_FALLBACK_RE = re.compile(
    r"^-\s+\*\*總分（基本面）\*\*[:：]\s*\*?\*?\s*([0-9]+(?:\.[0-9]+)?)",
    re.MULTILINE,
)
_CONFIDENCE_RE = re.compile(
    r"^-\s+\*\*信心\*\*[:：]\s*\*?\*?\s*(LOW|MEDIUM|MED|HIGH)",
    re.MULTILINE | re.IGNORECASE,
)
_PARK_FACTOR_RE = re.compile(
    r"Park Factor[:：]\s*([0-9]+(?:\.[0-9]+)?)",
)


def _is_template_line(line: str) -> bool:
    return "<!-- AI" in line


def parse_summary(path: Path, home_team_abbr: str, away_team_abbr: str) -> dict:
    """Parse summary.md → prediction dict (see module docstring for schema).

    If '方向（基本面）' line is template, returns dict with parse_failed=True
    and all fields None / False.
    """
    text = Path(path).read_text(encoding="utf-8")

    result: dict = {
        "direction": None,
        "total": None,
        "confidence": None,
        "park_factor": None,
        "has_reverse_platoon": False,
        "has_chain_break_300": False,
        "has_bullpen_il_2plus": False,
        "parse_failed": True,
    }

    # Direction
    dir_match = _DIR_LINE_RE.search(text)
    if not dir_match:
        return result
    dir_phrase = dir_match.group(1).strip()
    if _is_template_line(dir_phrase):
        return result
    direction = _resolve_direction(dir_phrase, home_team_abbr, away_team_abbr)
    if direction is None:
        return result
    result["direction"] = direction

    # Total
    tot_match = _TOTAL_LINE_RE.search(text)
    if not tot_match:
        tot_match = _TOTAL_LINE_FALLBACK_RE.search(text)
    if tot_match:
        try:
            result["total"] = float(tot_match.group(1))
        except ValueError:
            pass

    # Confidence
    conf_match = _CONFIDENCE_RE.search(text)
    if conf_match:
        c = conf_match.group(1).upper()
        result["confidence"] = "MEDIUM" if c == "MED" else c

    # Park Factor
    pf_match = _PARK_FACTOR_RE.search(text)
    if pf_match:
        try:
            result["park_factor"] = float(pf_match.group(1))
        except ValueError:
            pass

    # Flags
    result["has_reverse_platoon"] = bool(re.search(r"reverse platoon", text, re.IGNORECASE))
    chain_break_match = re.search(r"chain breaks? at #.*OPS 落差\s+([0-9]+\.[0-9]+)", text)
    if chain_break_match:
        try:
            result["has_chain_break_300"] = float(chain_break_match.group(1)) >= 0.300
        except ValueError:
            pass
    result["has_bullpen_il_2plus"] = bool(re.search(r"牛棚 core IL [×x](?:2|3)", text))

    # parse_failed only if direction couldn't be resolved (other fields are best-effort)
    result["parse_failed"] = False
    return result
