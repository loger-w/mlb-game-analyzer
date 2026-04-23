# Phase 3 Summary — SDP @ LAA (2026-04-19)

**Matchup**: San Diego Padres @ Los Angeles Angels
**Venue**: Angel Stadium (Park Factor 100, neutral)
**First Pitch**: 2026-04-19 20:07 UTC (13:07 PT day game)
**Starters**: Mike King (SDP, R) vs Reid Detmers (LAA, L)

## Pitching

### Mike King (SDP, away)
- Line: 2.78 ERA / 3.23 FIP / 3.92 xFIP / 1.15 WHIP, 22.2 IP (4 GS)
- **xERA 4.57 vs ERA 2.78 → gap 1.79 ≥ 1.5** → regression-warning flag; early-season ERA overstated by luck
- Prior year: 3.44 ERA / 3.56 xFIP (73.1 IP as SP) — role unchanged; 2026 line is luck-inflated vs a true ~mid-3s pitcher
- K-BB 10.7% (modest), GB 48.8% (GB-lean), hard-hit 21.8% (solid contact suppression)
- Platoon: vs L .167/.268/.229 (CH-heavy approach works); **vs R .267/.389/.433** → exploitable by RHH-heavy lineups
- Tier: Strong Ace (age 30, early decline indicator); 11-day rest since 2026-04-08 (not on IL)

### Reid Detmers (LAA, home)
- Line: 3.57 ERA / **2.17 FIP / 2.92 xFIP** / 1.06 WHIP, 22.2 IP (4 GS)
- **xERA 2.53 vs ERA 3.57 → gap 1.04** (under 1.5 threshold, but ERA understates quality)
- K-BB 21.3% elite, hard-hit 19.2%, barrel 6.7%, whiff 12.7%, CSW 29.3%
- **🚨 role_change flag**: prior year 61 G / 0 GS (pure reliever) → 2026 4 GS; limited SP workload track record; pitch counts 92-104
- Platoon: vs L .172/.250/.310, vs R .232/.290/.339 — effective both sides, slightly better vs LHH
- Tier: Solid Starter at peak age 26; 11-day rest since 2026-04-08 (not on IL) — likely workload management

## Bullpens (🚨 critical IL impact)

| Team | Bullpen ERA | Key IL losses |
|------|-------------|---------------|
| SDP  | **2.94** (strong) | Estrada, Matsui, Canning, Hoeing, Musgrove, Pivetta, Brito (60-day) |
| LAA  | **4.74** (weak)   | **Joyce (closer), Yates, Stephenson (60-day), G.Rodriguez, Manoah, R.Johnson, Ben Joyce** |

- **Bullpen-gate**: both sides hit; LAA loses **3 high-leverage arms (closer + 2 setup)** → bilateral adjustment: **LAA ML ↓, LAA-game O/U ↑**; SDP losses shallower — SDP ML mostly intact.
- Absolute gap 1.80 ERA favors SDP late in close games.

## Offense

| Metric | LAA (home) | SDP (away) |
|--------|-----------|-----------|
| Season RS / RA | 5.18 / 4.64 | 4.48 / 3.71 |
| L7 RS / RA | 6.5 / 4.7 | 5.6 / 3.2 |
| xwOBA | 0.327 | **0.342** |
| OPS | 0.754 | 0.685 |
| K% | 24.8 | 20.3 |
| BABIP | 0.264 | 0.280 |
| Tier | Average | Strong |
| Heat (L7) | **Hot** | Normal |
| OU lean | +1 | 0 |

- LAA **Hot L7** (RS 6.5 vs season 5.18) with BABIP .264 (below .280 threshold) → heat is *legit*, not BABIP-luck driven; positive regression still possible.
- SDP OPS .685 vs xwOBA .342 → underlying offense better than output suggests; BABIP .280 normal — mild positive regression plausible.
- Chain: LAA top-3 OBP .370 (above avg) / mid SLG .420; SDP top-3 OBP .324 / mid SLG .469 (mid-order power).

## Matchup-specific

- **King vs LAA RHH-heavy lineup** (Trout, Neto, Adell, Ward typical): King's vs-R split .267/.389/.433 is his weakness → LAA's right-handed core exploits this.
- **Detmers vs SDP lineup** (Tatis R, Merrill L, Machado R, Cronenworth L): balanced; Detmers slightly better vs L (.172 avg) — neutral to slight pitcher edge.
- **King pitch mix**: SI 35% / CH 27% / ST 18% / FF 17% — GB-sinker + CH approach; LAA's 24.8 K% suggests they'll whiff enough for King to navigate, but barrel 11.9% is concerning against Trout-Neto.
- Mike Trout BABIP .220 with xwOBA .485 → **extreme BABIP-unluck**, positive regression candidate vs any right-hander today.

## Environmental

- Angel Stadium PF 100 (neutral, slight pitcher-friendly historically)
- Day game (13:07 PT) — no known weather/wind issue at retractable-roof-less park; check closer to pitch time
- Series: SDP won previous game 4-1 on 2026-04-18 (King didn't pitch; Detmers didn't pitch)

## Signal summary (inputs to Phase 4 predict.py; no ratings here)

- SDP edge: bullpen quality gap (1.80 ERA), Detmers role_change risk, SDP xwOBA higher
- LAA edge: Hot L7 offense (legit), King ERA-regression risk, King vs-RHH weakness, Trout BABIP regression
- Total pressure: mixed — elite Detmers FIP suppresses SDP; King xERA + LAA bullpen inflate LAA runs; net slight lean pending model output
- Role-change and 11-day rest are **risk flags**, not exploitable predictions — handle via confidence discount not point estimate
