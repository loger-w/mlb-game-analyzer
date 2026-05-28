import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def test_constants_present_and_typed():
    assert config.LEAGUE_RG == 4.4
    assert config.RECENT_W == 0.35
    assert config.SP_W == 0.6 and config.BP_W == 0.4
    assert config.SIGMA_TEAM == 3.0
    assert config.FIP_CONSTANT == 3.10
    assert config.RECENT_N == 10
    assert config.MIN_IP == 10
    # weights coherent
    assert math.isclose(config.SP_W + config.BP_W, 1.0)


def test_sigma_derived():
    assert math.isclose(config.SIGMA, 3.0 * math.sqrt(2), rel_tol=1e-9)
