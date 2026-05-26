"""Parse mlb-game-analyzer summary.md into structured prediction fields.

Schema (per spec §2 / §7):
  {direction: "HOME"|"AWAY"|"持平"|None,
   total: float|None,
   confidence: "LOW"|"MEDIUM"|"HIGH"|None,
   confidence_pct: float|None,
   park_factor: float|None,
   has_reverse_platoon: bool,
   has_chain_break_300: bool,
   has_bullpen_il_2plus: bool,
   parse_failed: bool}

Notes:
  - confidence: bucket label from older (pre-5/4) format: LOW/MEDIUM/HIGH
  - confidence_pct: percentage as decimal (e.g. 0.62) from newer (5/4+) format.
    Range values (e.g. 62-65%) are stored as midpoint (0.635).
    Downstream consumers should prefer confidence_pct when available,
    falling back to bucket mapping for older-format rows where only
    confidence is populated.
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
    r"^-\s+\*\*總分（基本面）\*\*[:：]\s*\*?\*?~?\s*([0-9]+(?:\.[0-9]+)?)",
    re.MULTILINE,
)
# Match BOTH '**信心**' AND '**方向信心**' labels.
# Bucket value: LOW / MEDIUM / MED / HIGH
_CONFIDENCE_BUCKET_RE = re.compile(
    r"^-\s+\*\*(?:方向)?信心\*\*[:：]\s*\*?\*?\s*(LOW|MEDIUM|MED|HIGH)",
    re.MULTILINE | re.IGNORECASE,
)
# Percentage value: 'X%', 'X-Y%' (range → midpoint), '~X%', '約 X%', '**約 X%**', etc.
# Two alternations:
#   A) range without % on first number: X-Y%   → groups (X, Y)
#   B) single value (possibly with ~ or 約):    → groups (X, None)
_CONFIDENCE_PCT_RE = re.compile(
    r"^-\s+\*\*(?:方向)?信心\*\*[:：]\s*\*?\*?約?\s*~?\s*"
    r"(?:([0-9]+(?:\.[0-9]+)?)\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)\s*%"
    r"|([0-9]+(?:\.[0-9]+)?)\s*%)",
    re.MULTILINE,
)
_PARK_FACTOR_RE = re.compile(
    r"\*?\*?Park Factor\*?\*?[:：]\s*(?:[^0-9(（\n]*?)([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
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
        "confidence_pct": None,
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

    # Confidence — try bucket first (older format), then percentage (newer format)
    conf_match = _CONFIDENCE_BUCKET_RE.search(text)
    if conf_match:
        c = conf_match.group(1).upper()
        result["confidence"] = "MEDIUM" if c == "MED" else c

    pct_match = _CONFIDENCE_PCT_RE.search(text)
    if pct_match:
        try:
            if pct_match.group(1) is not None:
                # Range alternation: groups 1 and 2 (e.g. 62-65%)
                n1 = float(pct_match.group(1))
                n2 = float(pct_match.group(2))
            else:
                # Single-value alternation: group 3 (e.g. 62%, ~70%)
                n1 = float(pct_match.group(3))
                n2 = n1
            # Midpoint for range; single number otherwise; divide by 200 to convert % to decimal and average
            result["confidence_pct"] = round((n1 + n2) / 200.0, 4)
        except (ValueError, TypeError):
            pass

    # Park Factor
    pf_match = _PARK_FACTOR_RE.search(text)
    if pf_match:
        try:
            result["park_factor"] = float(pf_match.group(1))
        except ValueError:
            pass

    # Flags
    result["has_reverse_platoon"] = bool(re.search(r"reverse platoon", text, re.IGNORECASE))
    chain_breaks = re.findall(r"chain breaks? at #.*OPS 落差\s+([0-9]+\.[0-9]+)", text)
    try:
        result["has_chain_break_300"] = any(float(v) >= 0.300 for v in chain_breaks)
    except ValueError:
        pass
    result["has_bullpen_il_2plus"] = bool(re.search(r"牛棚 core IL [×x](?:2|3)", text))

    # parse_failed only if direction couldn't be resolved (other fields are best-effort)
    result["parse_failed"] = False
    return result
