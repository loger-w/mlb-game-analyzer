# 牛棚短休疲勞 ablation — 2026

## Path A:μ 懲罰(w_fat × 近2天後援IP 加進牛棚ERA)
_train Mar–Apr=468｜test May(有盤口)=292_
- w_fat* = 0.36
- pooled log-loss:baseline 0.6979 → candidate 0.6923(改善 0.0056 ± 0.0045 SE)
- **判決:ACCEPT**(接受條件:OOS 改善 > 1 SE)

## Path B:尾巴過濾器(任一隊近2天後援IP ≥ 12.0)— 正 edge 注,valid=292
- 尾巴(tail):n=34｜命中率 35.3%｜CLV mean 0.229pp(n_clv=31)
- 非尾巴:n=259｜命中率 49.0%｜CLV mean -0.227pp(n_clv=230)

> 尾巴 n 小屬正常;命中率與 CLV 要同向且離噪音才算數,否則 inconclusive。
> 對線上模型唯讀;此判決僅決定是否值得進一步,baking 是另一個決定。
