"""所有預測先驗係數集中於此。回測後重新擬合只改這裡。"""
import math

LEAGUE_RG = 4.196        # 聯盟每場均分
RECENT_W = 0.35        # RS blend:近期權重(季 = 1 - RECENT_W)
SP_W = 0.6             # 期望得分 — 先發權重(約 6/9 局)
BP_W = 0.4             # 期望得分 — 牛棚權重(約 3/9 局)
SIGMA_TEAM = 3.462       # 單隊單場得分 SD(歷史先驗)
SIGMA = SIGMA_TEAM * math.sqrt(2)   # margin / total SD ≈ 4.24
FIP_CONSTANT = 3.10    # FIP 聯盟正規化常數
RECENT_N = 10          # 近期窗口場數
MIN_IP = 10            # 先發 IP 低於此 → FIP 不穩,用聯盟替代


def constants_snapshot() -> dict:
    """凍結進 features.json 的當下係數值(重現用)。"""
    return {
        "league_rg": LEAGUE_RG, "recent_w": RECENT_W,
        "sp_w": SP_W, "bp_w": BP_W, "sigma_team": SIGMA_TEAM,
        "fip_constant": FIP_CONSTANT, "recent_n": RECENT_N, "min_ip": MIN_IP,
    }
