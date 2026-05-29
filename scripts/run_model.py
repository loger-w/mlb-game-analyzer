"""確定性預測:team-level 期望得分 → RL / O/U / ML 機率。純函數,零 I/O。"""
from statistics import NormalDist

import config

_N = NormalDist()


def pitch_today(starter_fip: float, bullpen_era: float) -> float:
    """今日防守力(每9局)= SP_W×先發FIP + BP_W×牛棚ERA。"""
    return config.SP_W * starter_fip + config.BP_W * bullpen_era


def expected_runs(home_rs: float, away_rs: float,
                  home_pitch: float, away_pitch: float,
                  pf: float, league_rg: float | None = None) -> tuple[float, float]:
    """期望得分。home_pitch/away_pitch 為各隊今日防守力(pitch_today 輸出)。
    league_rg 省略時用 config.LEAGUE_RG(fitting 時可覆寫)。

    μ_home = 主隊RS × 對方(away)防守力 / 聯盟 × PF/100
    """
    lg = config.LEAGUE_RG if league_rg is None else league_rg
    pf_mult = pf / 100.0
    mu_home = home_rs * away_pitch / lg * pf_mult
    mu_away = away_rs * home_pitch / lg * pf_mult
    return mu_home, mu_away


def cover_prob_home(mu_margin: float, rl_point_home: float, sigma: float | None = None) -> float:
    """P(主隊過 RL)。主隊 cover 條件:margin > −rl_point_home。
    主 −1.5 → P(margin>1.5);主 +1.5 → P(margin>−1.5)。sigma 省略時用 config.SIGMA。"""
    s = config.SIGMA if sigma is None else sigma
    z = (-rl_point_home - mu_margin) / s
    return 1.0 - _N.cdf(z)


def over_prob(mu_total: float, total_line: float, sigma: float | None = None) -> float:
    """P(Over):P(total > 線)。sigma 省略時用 config.SIGMA。"""
    s = config.SIGMA if sigma is None else sigma
    z = (total_line - mu_total) / s
    return 1.0 - _N.cdf(z)


def home_ml_prob(mu_margin: float, sigma: float | None = None) -> float:
    """P(主隊勝)= P(margin > 0)。內部用,不輸出給使用者。sigma 省略時用 config.SIGMA。"""
    s = config.SIGMA if sigma is None else sigma
    return _N.cdf(mu_margin / s)


def predict(*, home_rs: float, away_rs: float,
            home_starter_fip: float, away_starter_fip: float,
            home_bullpen_era: float, away_bullpen_era: float,
            pf: float, rl_point_home: float | None, total_line: float | None) -> dict:
    """完整模型輸出。rl_point_home / total_line 可為 None(無盤口時機率仍算 ML)。

    機率由「2 位小數的 μ」推導,使顯示的 μ 與機率內部一致、可重現。
    """
    home_pitch = pitch_today(home_starter_fip, home_bullpen_era)
    away_pitch = pitch_today(away_starter_fip, away_bullpen_era)
    mu_home_raw, mu_away_raw = expected_runs(home_rs, away_rs, home_pitch, away_pitch, pf)
    mu_home = round(mu_home_raw, 2)
    mu_away = round(mu_away_raw, 2)
    mu_margin = round(mu_home - mu_away, 2)
    mu_total = round(mu_home + mu_away, 2)
    return {
        "mu_home": mu_home,
        "mu_away": mu_away,
        "mu_margin": mu_margin,
        "mu_total": mu_total,
        "p_home_ml": round(home_ml_prob(mu_margin), 4),
        "p_home_cover_rl": (round(cover_prob_home(mu_margin, rl_point_home), 4)
                            if rl_point_home is not None else None),
        "p_over": (round(over_prob(mu_total, total_line), 4)
                   if total_line is not None else None),
    }
