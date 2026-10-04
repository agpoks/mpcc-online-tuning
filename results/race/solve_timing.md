# acados MPCC solve-time analysis (dynamic + discrete + two-layer corridor, STD plant)

Pure `m.value()` wall time, single process (acados built WITHOUT OpenMP), 5 horizons, 250 solves each along a drive. Real-time budget = **50 ms (20 Hz)**. The cost is dominated by `hessian_approx=EXACT`, required because the cost is `EXTERNAL` (keeps progress linear) and the online weight-tuning gradient needs the solver's exact Hessian.

| horizon N | lookahead | mean [ms] | p50 | p95 | max | vs 20 Hz |
|---:|---:|---:|---:|---:|---:|:--|
| 15 | 0.75 s | 36.5 | 34.0 | 37.9 | 586.6 | OK |
| 20 | 1.00 s | 33.4 | 33.3 | 35.6 | 40.9 | OK |
| 25 | 1.25 s | 34.8 | 34.4 | 38.0 | 86.1 | OK |
| 30 | 1.50 s | 34.8 | 34.5 | 37.8 | 45.9 | OK |
| 40 | 2.00 s | 35.7 | 34.0 | 49.4 | 60.9 | OK |
