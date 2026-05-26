"""Compute backtest metrics from per-game DataFrame.

All functions operate on the DataFrame produced by lib.load.build_dataframe_for_month.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd


def _valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for indicator computation (excludes parse/closing/result failures)."""
    return df[
        (~df["parse_failed"])
        & (~df["closing_missing"])
        & (~df["result_missing"])
    ].copy()


def derive_direction_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Add `direction_hit` column to df. Operates on a copy."""
    out = df.copy()
    out["direction_hit"] = out["skill_direction"] == out["actual_winner"]
    return out


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _effective_confidence_bucket(row) -> Optional[str]:
    """Derive an effective LOW / MEDIUM / HIGH bucket.

    Order of preference:
      1. `skill_confidence` if set (old-format LOW/MED/HIGH)
      2. `skill_confidence_pct` mapped to bucket: <0.58 LOW, 0.58-0.67 MEDIUM, ≥0.67 HIGH
      3. None
    """
    if row.get("skill_confidence") in ("LOW", "MEDIUM", "HIGH"):
        return row["skill_confidence"]
    pct = row.get("skill_confidence_pct")
    if pct is not None and not pd.isna(pct):
        if pct < 0.58:
            return "LOW"
        if pct < 0.67:
            return "MEDIUM"
        return "HIGH"
    return None


def compute_direction_metrics(df: pd.DataFrame) -> dict:
    valid = _valid_rows(df)
    # Exclude '持平' skill picks from direction analysis
    direction_valid = valid[valid["skill_direction"].isin(["HOME", "AWAY"])]

    n = len(direction_valid)
    if n == 0:
        return {"skill_n": 0, "skill_hit_rate": None, "market_n": 0, "market_hit_rate": None,
                "edge_pp": None, "skill_aligned_n": 0, "skill_aligned_hit_rate": None,
                "skill_against_n": 0, "skill_against_hit_rate": None,
                "skill_ci": None, "market_ci": None}

    skill_hits = (direction_valid["skill_direction"] == direction_valid["actual_winner"]).sum()
    skill_rate = skill_hits / n
    skill_ci = _wilson_ci(skill_rate, n)

    market_hits = (direction_valid["market_favorite"] == direction_valid["actual_winner"]).sum()
    market_rate = market_hits / n
    market_ci = _wilson_ci(market_rate, n)

    # Aligned / against-market splits
    aligned = direction_valid[direction_valid["skill_direction"] == direction_valid["market_favorite"]]
    against = direction_valid[direction_valid["skill_direction"] != direction_valid["market_favorite"]]
    aligned_rate = (aligned["skill_direction"] == aligned["actual_winner"]).mean() if len(aligned) else None
    against_rate = (against["skill_direction"] == against["actual_winner"]).mean() if len(against) else None

    return {
        "skill_n": n,
        "skill_hit_rate": skill_rate,
        "skill_ci": skill_ci,
        "market_n": n,
        "market_hit_rate": market_rate,
        "market_ci": market_ci,
        "edge_pp": skill_rate - market_rate,
        "skill_aligned_n": len(aligned),
        "skill_aligned_hit_rate": aligned_rate,
        "skill_against_n": len(against),
        "skill_against_hit_rate": against_rate,
    }


def compute_total_metrics(df: pd.DataFrame) -> dict:
    valid = _valid_rows(df)
    valid = valid[
        valid["skill_total"].notna()
        & valid["actual_total"].notna()
        & valid["market_total_line"].notna()
    ]

    if len(valid) == 0:
        return {"n": 0, "total_mae": None, "total_bias": None, "ou_n": 0, "ou_hit_rate": None}

    abs_err = (valid["skill_total"] - valid["actual_total"]).abs()
    signed_err = valid["skill_total"] - valid["actual_total"]

    # Exclude pushes for OU hit-rate
    line = valid["market_total_line"]
    no_push = valid[
        (valid["actual_total"] != line)
        & (valid["skill_total"] != line)
    ]
    if len(no_push) > 0:
        skill_side = np.sign(no_push["skill_total"] - no_push["market_total_line"])
        actual_side = np.sign(no_push["actual_total"] - no_push["market_total_line"])
        ou_hit_rate = (skill_side == actual_side).mean()
    else:
        ou_hit_rate = None

    return {
        "n": len(valid),
        "total_mae": float(abs_err.mean()),
        "total_bias": float(signed_err.mean()),
        "ou_n": len(no_push),
        "ou_hit_rate": float(ou_hit_rate) if ou_hit_rate is not None else None,
    }


def compute_calibration(df: pd.DataFrame) -> dict:
    """Build reliability table grouped by effective_bucket (derived from skill_confidence
    OR skill_confidence_pct), plus skill/market Brier and log-loss."""
    valid = _valid_rows(df)
    valid = valid[valid["skill_direction"].isin(["HOME", "AWAY"])]
    valid = valid[valid["skill_prob_mapped"].notna()]

    if len(valid) == 0:
        return {"reliability_table": pd.DataFrame(), "brier_score": None, "log_loss": None,
                "brier_baseline_market": None, "log_loss_baseline_market": None}

    valid = valid.copy()
    valid["outcome"] = (valid["skill_direction"] == valid["actual_winner"]).astype(float)
    valid["effective_bucket"] = valid.apply(_effective_confidence_bucket, axis=1)

    # Reliability table grouped by effective_bucket
    rows = []
    for conf in ["LOW", "MEDIUM", "HIGH"]:
        sub = valid[valid["effective_bucket"] == conf]
        n_sub = len(sub)
        hit = float(sub["outcome"].mean()) if n_sub > 0 else None
        # Use median of skill_prob_mapped within bucket (mix of pct + bucket rows)
        mapped = float(sub["skill_prob_mapped"].median()) if n_sub > 0 else None
        ci = _wilson_ci(hit, n_sub) if n_sub > 0 and hit is not None else (None, None)
        rows.append({
            "confidence": conf,
            "n": n_sub,
            "mapped_p": mapped,
            "actual_hit_rate": hit,
            "delta": (hit - mapped) if (hit is not None and mapped is not None) else None,
            "ci_low": ci[0],
            "ci_high": ci[1],
        })
    reliability = pd.DataFrame(rows)

    # Brier / log-loss with skill mapping (uses skill_prob_mapped directly — already correct
    # whether sourced from pct or bucket)
    eps = 1e-12
    p = valid["skill_prob_mapped"].astype(float)
    y = valid["outcome"]
    brier = float(((p - y) ** 2).mean())
    log_loss = float(-(y * np.log(p.clip(eps, 1 - eps)) + (1 - y) * np.log((1 - p).clip(eps, 1 - eps))).mean())

    # Baseline: market no-vig prob on the side skill picked
    market_p_on_skill_side = np.where(
        valid["skill_direction"] == "HOME",
        valid["market_home_winprob_no_vig"],
        1 - valid["market_home_winprob_no_vig"],
    )
    valid["_market_p_skill_side"] = market_p_on_skill_side
    valid_market = valid[valid["_market_p_skill_side"].notna()]
    if len(valid_market) > 0:
        mp = valid_market["_market_p_skill_side"].astype(float)
        my = valid_market["outcome"]
        brier_mkt = float(((mp - my) ** 2).mean())
        log_loss_mkt = float(
            -(my * np.log(mp.clip(eps, 1 - eps)) + (1 - my) * np.log((1 - mp).clip(eps, 1 - eps))).mean()
        )
    else:
        brier_mkt = None
        log_loss_mkt = None

    return {
        "reliability_table": reliability,
        "brier_score": brier,
        "log_loss": log_loss,
        "brier_baseline_market": brier_mkt,
        "log_loss_baseline_market": log_loss_mkt,
    }


def compute_slice_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-slice metrics: dir_hit, ou_hit, mae, bias, n."""
    valid = _valid_rows(df)

    def _slice_stats(sub: pd.DataFrame) -> dict:
        if len(sub) == 0:
            return {"n": 0, "dir_n": 0, "dir_hit": None, "ou_hit": None, "mae": None, "bias": None}
        dir_sub = sub[sub["skill_direction"].isin(["HOME", "AWAY"])]
        dir_hit = (dir_sub["skill_direction"] == dir_sub["actual_winner"]).mean() if len(dir_sub) else None
        # OU
        tot_sub = sub[sub["skill_total"].notna() & sub["actual_total"].notna() & sub["market_total_line"].notna()]
        no_push = tot_sub[(tot_sub["actual_total"] != tot_sub["market_total_line"])
                         & (tot_sub["skill_total"] != tot_sub["market_total_line"])]
        if len(no_push):
            ss = np.sign(no_push["skill_total"] - no_push["market_total_line"])
            sa = np.sign(no_push["actual_total"] - no_push["market_total_line"])
            ou_hit = (ss == sa).mean()
        else:
            ou_hit = None
        mae = float((tot_sub["skill_total"] - tot_sub["actual_total"]).abs().mean()) if len(tot_sub) else None
        bias = float((tot_sub["skill_total"] - tot_sub["actual_total"]).mean()) if len(tot_sub) else None
        return {"n": len(sub), "dir_n": len(dir_sub), "dir_hit": dir_hit, "ou_hit": ou_hit, "mae": mae, "bias": bias}

    slices: dict[str, dict] = {}

    # Direction
    slices["direction_HOME"] = _slice_stats(valid[valid["skill_direction"] == "HOME"])
    slices["direction_AWAY"] = _slice_stats(valid[valid["skill_direction"] == "AWAY"])

    # Confidence (effective bucket)
    valid_with_eff = valid.copy()
    valid_with_eff["effective_bucket"] = valid_with_eff.apply(_effective_confidence_bucket, axis=1)
    for c in ["LOW", "MEDIUM", "HIGH"]:
        slices[f"confidence_{c}"] = _slice_stats(valid_with_eff[valid_with_eff["effective_bucket"] == c])

    # Park factor
    slices["park_factor_high"] = _slice_stats(valid[valid["park_factor"] > 102])
    slices["park_factor_mid"] = _slice_stats(
        valid[(valid["park_factor"] >= 98) & (valid["park_factor"] <= 102)]
    )
    slices["park_factor_low"] = _slice_stats(valid[valid["park_factor"] < 98])

    # Flag-based
    slices["has_reverse_platoon"] = _slice_stats(valid[valid["has_reverse_platoon"] == True])
    slices["no_reverse_platoon"] = _slice_stats(valid[valid["has_reverse_platoon"] == False])
    slices["has_chain_break_300"] = _slice_stats(valid[valid["has_chain_break_300"] == True])
    slices["has_bullpen_il_2plus"] = _slice_stats(valid[valid["has_bullpen_il_2plus"] == True])

    return pd.DataFrame.from_dict(slices, orient="index")
