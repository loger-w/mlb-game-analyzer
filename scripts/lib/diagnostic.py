"""Select failure cases for the report's diagnostic section."""

import re
from pathlib import Path

import pandas as pd

from lib.metrics import _effective_confidence_bucket


_DIR_LINE_RE = re.compile(r"^-\s+\*\*方向（基本面）\*\*[:：]\s*(.+?)$", re.MULTILINE)
_MD_STRIP = re.compile(r"\*+")


def extract_main_signal(summary_text: str, max_chars: int = 50) -> str:
    """Extract first ~50 chars of the '方向（基本面）' content, markdown stripped.

    Used in failure case table to give human reader the skill's stated rationale.
    """
    m = _DIR_LINE_RE.search(summary_text)
    if not m:
        return ""
    phrase = m.group(1).strip()
    phrase = _MD_STRIP.sub("", phrase)
    # Trim at first sentence terminator (。/，/—) if any within max_chars
    for term in ["。", "，", "—"]:
        idx = phrase.find(term)
        if 0 < idx <= max_chars:
            return phrase[:idx].strip()
    return phrase[:max_chars].strip()


def select_failure_cases(df: pd.DataFrame, top_total_miss: int = 10) -> pd.DataFrame:
    """Two rankings merged:
       1. direction_miss at effective confidence >= MEDIUM (all)
       2. top N rows by |skill_total - actual_total|
       Merged + deduped by matchup. Flags `dual_failure` for rows in both lists.
    """
    valid = df[
        (~df["closing_missing"]) & (~df["result_missing"])
    ].copy()

    # Derive effective confidence bucket (handles pct-only rows)
    valid["_eff_conf"] = valid.apply(_effective_confidence_bucket, axis=1)

    # 1. Direction misses (MED/HIGH only, using effective bucket)
    dir_miss = valid[
        (valid["skill_direction"].isin(["HOME", "AWAY"]))
        & (valid["skill_direction"] != valid["actual_winner"])
        & (valid["_eff_conf"].isin(["MEDIUM", "HIGH"]))
    ].copy()
    dir_miss["is_direction_miss"] = True

    # 2. Top total misses (exclude exact hits: abs_error must be > 0)
    with_err = valid[valid["skill_total"].notna() & valid["actual_total"].notna()].copy()
    # Coerce to float — when valid is empty, subtraction gives object dtype and nlargest fails
    with_err["total_abs_error"] = pd.to_numeric(
        (with_err["skill_total"] - with_err["actual_total"]).abs(), errors="coerce"
    )
    with_err = with_err[with_err["total_abs_error"] > 0]
    top_total = with_err.nlargest(top_total_miss, "total_abs_error").copy()
    top_total["is_top_total_miss"] = True

    # Merge — ensure flag columns exist in combined before fill
    dir_miss["is_top_total_miss"] = False
    top_total["is_direction_miss"] = top_total.get("is_direction_miss", False)
    if "is_direction_miss" not in top_total.columns:
        top_total["is_direction_miss"] = False
    combined = pd.concat([dir_miss, top_total], ignore_index=False)
    combined["is_direction_miss"] = combined["is_direction_miss"].astype(bool)
    combined["is_top_total_miss"] = combined["is_top_total_miss"].astype(bool)

    # Dedupe by matchup+date (keep flags from both)
    agg_spec = {
        **{c: "first" for c in combined.columns if c not in ("is_direction_miss", "is_top_total_miss", "date", "matchup")},
        "is_direction_miss": "max",
        "is_top_total_miss": "max",
    }
    combined = combined.groupby(["date", "matchup"], as_index=False).agg(agg_spec)
    combined["dual_failure"] = combined["is_direction_miss"] & combined["is_top_total_miss"]

    # Drop internal helper column
    combined = combined.drop(columns=["_eff_conf"], errors="ignore")

    # Sort: direction miss first by date, then total miss by abs error
    combined = combined.sort_values(
        by=["is_direction_miss", "date"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return combined
