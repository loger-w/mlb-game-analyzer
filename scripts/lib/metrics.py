"""回測指標:RL 過盤命中、O/U 命中、edge 校準。讀 lib.load 產的 DataFrame。"""
import pandas as pd


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[(~df["result_missing"]) & (~df["odds_missing"])].copy()


def compute_rl_metrics(df: pd.DataFrame) -> dict:
    """model 預測『主過盤』(p>0.5) 是否命中(實際 margin > −point)。"""
    v = _valid(df)
    v = v[v["p_home_cover_rl"].notna() & v["rl_home_point"].notna() & v["actual_margin"].notna()]
    if len(v) == 0:
        return {"n": 0, "rl_hit_rate": None}
    pred_home_cover = v["p_home_cover_rl"] > 0.5
    actual_home_cover = v["actual_margin"] > (-v["rl_home_point"])
    hit = (pred_home_cover == actual_home_cover).mean()
    return {"n": int(len(v)), "rl_hit_rate": float(hit)}


def compute_ou_metrics(df: pd.DataFrame) -> dict:
    """model 預測 Over(p>0.5) 是否命中。排除 push(actual == line)。"""
    v = _valid(df)
    v = v[v["p_over"].notna() & v["total_line"].notna() & v["actual_total"].notna()]
    v = v[v["actual_total"] != v["total_line"]]
    if len(v) == 0:
        return {"n": 0, "ou_hit_rate": None}
    pred_over = v["p_over"] > 0.5
    actual_over = v["actual_total"] > v["total_line"]
    hit = (pred_over == actual_over).mean()
    return {"n": int(len(v)), "ou_hit_rate": float(hit)}


def compute_edge_calibration(df: pd.DataFrame) -> dict:
    """正 edge 那側是否真的較常贏(edge 有沒有預測力)。"""
    v = _valid(df)
    out = {}
    # RL:home_rl_pp>0 → 看好主過盤
    rl = v[v["home_rl_pp"].notna() & v["actual_margin"].notna() & v["rl_home_point"].notna()]
    rl_pos = rl[rl["home_rl_pp"] > 0]
    if len(rl_pos):
        actual = rl_pos["actual_margin"] > (-rl_pos["rl_home_point"])
        out["rl_pos_edge_n"] = int(len(rl_pos))
        out["rl_pos_edge_hit"] = float(actual.mean())
    else:
        out["rl_pos_edge_n"] = 0
        out["rl_pos_edge_hit"] = None
    # O/U:over_pp>0 → 看好 Over
    ou = v[v["over_pp"].notna() & v["actual_total"].notna() & v["total_line"].notna()]
    ou = ou[ou["actual_total"] != ou["total_line"]]
    ou_pos = ou[ou["over_pp"] > 0]
    if len(ou_pos):
        actual = ou_pos["actual_total"] > ou_pos["total_line"]
        out["ou_pos_edge_n"] = int(len(ou_pos))
        out["ou_pos_edge_hit"] = float(actual.mean())
    else:
        out["ou_pos_edge_n"] = 0
        out["ou_pos_edge_hit"] = None
    return out
