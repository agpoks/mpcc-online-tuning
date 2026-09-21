# Race-mode Phase 1 — results index (branch: race-mode)

Head-to-head behaviour training vs ONE opponent, on the shipping stack
(acados discrete + dynamic drift + STD plant, Track.icra_t2_smooth, max_obstacles=1).
Opponent cycled across static / slower / equal / faster, scaled to the measured ego pace.

## Reproduce
- Run:      `experiments/race_mode.py --seeds 4 --episodes 14 --steps 1200 --jobs 14 --out results/race/<name>.json`
- Analyse:  `tools/race_analysis.py <name>.json`  -> CSV + PDF/PNG (summary, by_kind, sectors, weights_by_kind)
- Animate:  `tools/race_gif.py --arm <ltc|mlp|const|fixed> --kind <static|slower|equal|faster> --seed 0`

## The three runs (all 4 seeds x 4 arms)
| run  | JSON                     | change                                              |
|------|--------------------------|-----------------------------------------------------|
| first| race_phase1_s4.json      | baseline reward (pass bonus BUG: never fired)       |
| sharp| race_phase1_sharp.json   | pace-dep reward, bonus 6/9, prior 0.3, bug fixed    |
| aggr | race_phase1_aggr.json    | bonus 9/13 + closing shaping, prior 0.2, alpha 3e-3, weights logged |

## Headline (aggr run) — mean over episodes
| arm    | laps | passes | clean | contact |
|--------|------|--------|-------|---------|
| const  | 0.97 | 0.59   | 94%   | 0%      |
| fixed  | 0.99 | 0.75   | 88%   | 12%     |
| ltc    | 0.94 | 0.53   | 100%  | 0%      |
| mlp    | 0.94 | 0.50   | 100%  | 0%      |

## Passes by opponent type (aggr) — behaviour is correct, not learned-differentiated
| arm   | static | slower | equal        | faster |
|-------|--------|--------|--------------|--------|
| const | 1.50   | 0.93   | 0.21         | 0.00   |
| fixed | 1.79   | 1.00   | 0.43 (43%ct) | 0.00   |
| ltc   | 1.14   | 0.93   | 0.00         | 0.00   |
| mlp   | 1.07   | 1.00   | 0.00         | 0.00   |

## Key finding
The learned tuner converges to ONE max-overtake posture (q_v->ceiling, q_c/r_d/r_a->floor,
k_v~0.67) that is NEARLY IDENTICAL across all opponent types (see race_weights_by_kind_*).
It passes what is physically passable (slower/static) and correctly declines equal/faster
(no clean pass exists; the fixed rule that tries contacts 43%). LTC ~= MLP (memory no help).
Sits at the safety frontier. Consistent with the friction/geometry disturbance cases
(results/paper_smooth/): online weight-tuning is sensible+safe but does not beat a good
constant, and here does not differentiate weights by opponent.

## Files
- JSON (raw, re-analysable): race_phase1_{s4,sharp,aggr}.json (+ pilots)
- CSV: race_{summary,by_kind,sectors}_{s4,sharp,aggr}.csv, race_weights_by_kind_aggr.csv
- PDF+PNG: race_{behaviour,sector_suitability}_{s4,sharp,aggr}, race_weights_by_kind_{ltc,mlp}_aggr
- GIF (gif/): {ltc,const}_{static,slower,equal,faster}.gif, fixed_equal_s0, plus _s0 variants
- Nets (nets/): race_{ltc,mlp}_{0..3}.npz  (G + cell_p + th0 + bounds; replay with race_gif.py)
