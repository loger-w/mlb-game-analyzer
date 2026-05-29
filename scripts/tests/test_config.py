import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def test_constants_present_and_typed():
    # held priors(不參與 refit,維持固定)
    assert config.RECENT_W == 0.35
    assert config.SP_W == 0.6 and config.BP_W == 0.4
    assert config.FIP_CONSTANT == 3.10
    assert config.RECENT_N == 10
    assert config.MIN_IP == 10
    assert math.isclose(config.SP_W + config.BP_W, 1.0)
    # fitted(refit 後會變;只查型別與合理範圍)
    assert isinstance(config.LEAGUE_RG, float) and 3.0 < config.LEAGUE_RG < 6.0
    assert isinstance(config.SIGMA_TEAM, float) and 2.0 < config.SIGMA_TEAM < 6.0


def test_sigma_derived():
    # 關係式不綁特定數值(SIGMA 由 SIGMA_TEAM 導出)
    assert math.isclose(config.SIGMA, config.SIGMA_TEAM * math.sqrt(2), rel_tol=1e-9)
