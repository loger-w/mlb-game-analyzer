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


def compute_threshold_sweep(df: pd.DataFrame, thresholds) -> dict:
    """雙向下注:|edge|≥t 下 model pick 側(sign(edge)),逐門檻命中率。
    RL 用 home_rl_pp(pick home/away)、O/U 用 over_pp(pick over/under,排 push)。"""
    v = _valid(df)

    rl = v[v["home_rl_pp"].notna() & (v["home_rl_pp"] != 0)
           & v["actual_margin"].notna() & v["rl_home_point"].notna()].copy()
    rl_home_cover = rl["actual_margin"] > (-rl["rl_home_point"])
    rl_hit = (rl["home_rl_pp"] > 0) == rl_home_cover   # pick home & cover, or pick away & no-cover

    ou = v[v["over_pp"].notna() & (v["over_pp"] != 0)
           & v["actual_total"].notna() & v["total_line"].notna()].copy()
    ou = ou[ou["actual_total"] != ou["total_line"]]    # exclude push
    ou_over = ou["actual_total"] > ou["total_line"]
    ou_hit = (ou["over_pp"] > 0) == ou_over

    def _sweep(frame, edge_col, hit):
        rows = []
        for t in thresholds:
            mask = frame[edge_col].abs() >= t
            n = int(mask.sum())
            rows.append({"t": t, "n_bets": n,
                         "hit_rate": float(hit[mask].mean()) if n else None})
        return rows

    return {"thresholds": list(thresholds),
            "rl": _sweep(rl, "home_rl_pp", rl_hit),
            "ou": _sweep(ou, "over_pp", ou_hit)}
