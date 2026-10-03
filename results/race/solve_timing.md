# acados MPCC solve-time analysis (shipping stack: dynamic + discrete + two-layer corridor, STD plant)

Pure `m.value()` wall time, single process, acados built **WITHOUT OpenMP** (single-threaded).
Real-time budget = **50 ms (20 Hz)**. Reproduce on an **idle** machine with
`python3 tools/paper_solve_timing.py` → this table + `results/race/paper/fig_solve_timing.{pdf,png}`
(horizon scaling). Numbers below are for the **deployed horizon N=25**.

## Clean (idle machine) — 300 solves at N=25
| metric | value | vs 20 Hz (50 ms) |
|---|---:|:--|
| mean | **82.4 ms** | OVER (~12 Hz) |
| p50  | 78.3 ms | OVER |
| p95  | 108.6 ms | OVER |
| max  | 179.6 ms | OVER |

## Under concurrent CPU load (a 6-worker job running)
~170–185 ms/solve and **flat across N=15…40** — this is CPU contention, not the isolated solve cost
(shown only to illustrate how shared-CPU inflates it; do not use for the real-time claim).

## Why it's ~82 ms (not single-digit)
`hessian_approx=EXACT`, which is **required**: the cost is `cost_type=EXTERNAL` (keeps the progress
reward LINEAR instead of a squared penalty), and the online weight-tuning gradient
(`dJ*/dθ = dL/dθ`, envelope theorem) needs the solver's **exact** Hessian. Gauss-Newton would be much
faster but silently breaks the tuning gradient.

## Paths to real-time (future)
1. Multi-threaded acados build (compile with OpenMP) + `OMP_NUM_THREADS>1` → HPIPM parallelism.
2. Shorter horizon (trades preview for speed).
3. Gauss-Newton reformulation (fast, but requires re-deriving the online-tuning gradient).
