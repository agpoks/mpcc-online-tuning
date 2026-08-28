# TODO

Ordered by what unblocks the paper. Everything already measured is in
`docs/source/` — this is only what is *not* done.

## 1. Make the tuner work on the fitted-tyre plant — the blocker

The controller runs there (`--plant scuderia`) and the feasible region exists:
an open-loop sweep shows **every run with `r_d=10` completes 400 steps and
every run with `r_d=1.0` crashes**. The tuner overshoots it and lands on "do
not drive".

This is a **reward-design** problem, not a research one. Distance covered minus
a single terminal −5 makes standing still a local optimum worth ≈0 against a
crash worth −5, and from an initialisation that crashes in under a second it is
the nearest one.

- [ ] Add a per-step survival bonus, or initialise inside the feasible region
      the sweep identifies.
- [ ] Re-run `examples/tune_online.py --plant scuderia` and report.

**Without this the paper cannot claim online tuning works on a realistic
vehicle.**

## 1b. Per-segment weights — the best result so far, and it is cheap

`experiments/per_segment_weights.py`. One θ per curvature segment, the segment
read from the path ahead. On the oval:

| mode | first off-track | last 8 episodes |
|---|---|---|
| global θ | ep 5 | 7.6 m, 8/8 off |
| **per-segment θ** | **never** | **78.0 m, 0/8 off** |

Against everything else tried on this problem: baseline 7.6 m (8/8 off),
event-triggered discard 38.5 m (6/8), behind a safety filter 68.1 m (0/8),
**per-segment 78.0 m (0/8)**. It beats the safety filter and needs nothing
bolted on.

- [ ] Reproduce over several seeds — this is one seed.
- [ ] Combine with the safety filter and with item 1, and check they compose.
- [ ] Try a continuous schedule θ(κ) rather than three bins.
- [ ] The learned weights are not intuitive (progress weighted *higher* in the
      tightest segment). Do not build a story on one seed.

## 2. Trust region on θ — partly answered by event-triggering

`experiments/event_triggered_tuning.py` tests an alternative: instead of
bounding *where* θ may go, bound *when* it moves. An integrate-and-fire
accumulator on measurable badness (lateral error, tyre-limit saturation) gates
the update.

| mode | collapses at | last 8 episodes |
|---|---|---|
| every tick | ep 5 | 7.6 m, 8/8 off |
| defer (accumulate, apply on event) | ep 3 | 7.4 m, 8/8 off |
| **discard (drop the move unless in trouble)** | **ep 18** | **38.5 m, 6/8 off** |

Deferring is *worse* than not gating at all: the total movement of θ is
unchanged, only the granularity, and coarser steps are less stable. Discarding
delays the collapse by 13 episodes and gives 5× the distance — but does not
prevent it.

- [ ] Trigger on the **safety filter's intervention** instead of on lateral
      error. Behind the filter, `q_c` falls 2.52 → 0.136 with performance
      drifting down, because the filter absorbs the consequence and it never
      reaches the return. The intervention *is* the missing error signal.
- [ ] Then a proper trust region, and compare. Discarding is biased (it learns
      only from trouble); a trust region is not.

## 2b. Trust region on θ

Most of the damage happens in episode 0: `q_v` moves by a factor of **222**
before a single episode boundary. The α=2e-4 and α=2e-3 runs give the same
shape ten times apart, which is a divergence signature, not a tuning artefact.

- [ ] Bound where θ may go (trust region, or a feasible set), rather than
      lowering the step size.

## 3. acados backend

`mpcc_tuning/rti.py` reaches 1.9 ms mean / 3.4 ms worst at N=12 with CasADi's
`qrqp`. `MPCC_planner_acados` already runs `SQP_RTI`.

- [ ] Port the OCP to acados as an alternative backend; check the envelope
      gradient there (acados returns the multipliers).
- [ ] Confirm the direction-agreement result holds for a real acados RTI step.

## 4. Filters on a state *estimate*

Every filter reads the true state. On the car it reads an estimate, and the
guarantee is conditional on it.

- [ ] Run the filters on odometry rather than ground truth; add the estimator's
      covariance to the margin.

## Known odd, undiagnosed

`benchmarks/filters.py`: the `worst-case` tube leaves the track in 60% of runs
at true grip 1.0. A filter assuming *less* grip than the plant has should be
conservative. Do not cite that row until it is explained.
