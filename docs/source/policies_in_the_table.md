# The policies in the result table, one by one

The summary figure (`paper/figures/cold_summary.png`) compares several ways of
choosing the MPCC's eight cost weights on ICRA T2. Every row is driven **cold**
— a freshly built solver, learning off, exploration off — from three physically
different starts, because that is what the car sees when it is switched on. This
page says what each row *is*.

Two things are constant across all rows: the **controller** is always the same
acados MPCC solving the same optimal-control problem every 50 ms, and the
**car** is the dynamic drift model on the scuderia plant. What changes between
rows is only how the eight weights
`(q_c, q_l, q_v, r_d, r_a, r_dv, d_obs, k_v)` are set.

The weights, in one line each: `q_c` how expensive it is to leave the reference
line; `q_l` lag along the path; `q_v` how much progress is rewarded; `r_d`,
`r_a`, `r_dv` how much steering, acceleration and their rates are damped;
`d_obs` the berth around an opponent; `k_v` how much grip the plan claims.

## START — the safe baseline (a constant)

One fixed weight vector, the same at every point of the track, chosen to be
stable and *not* aggressive: it drives cleanly but leaves obvious lap time on
the table (`k_v = 0.40`, a cautious grip claim). This is where a real team
starts — a good, conservative set — and it is the thing every other row is
measured against. It is **not** meant to be the best; it is meant to be the
honest starting point. In the code: `mpcc_tuning.baselines.START`.

**Cold: 1.87 / 2.01 / 2.00 laps.**

## Best constant by search — context, not a target (a constant)

The best *single* weight vector, found by driving an exhaustive grid of vectors
and keeping the one with the highest mean over the whole track (`k_v = 0.50`).
It is drawn as a reference line because it is an **upper bound on what any
constant can do**: a policy that beats it has done something no fixed set of
weights can. Nobody has this vector at the start of a race — it is the answer to
a search, shown for scale. In the code: `mpcc_tuning.baselines.BEST`.

**Cold: 2.32 / 2.33 / 2.40 laps.**

## Grid-fitted network — the deliverable (a situation-dependent policy)

This is the point of the project. Instead of one constant, a small recurrent
network (an LTC cell plus a linear readout) emits the weights **afresh every
tick** from what the car can see: the sector ahead, its speed, the curvature
preview. The network is trained, once and offline, to reproduce the *situation
grid* — a table that drove every `(q_v, k_v, q_c)` combination in every
`(sector, entry-speed)` cell and kept the winner of each. So the network learns
"on a straight entered slowly, want this; in a hairpin, want that", with the
grip claim capped at the level shown safe over full laps (`k_v = 0.50`).

The result is a policy that lowers `q_c` on the straights (letting the car leave
the centre line for a faster line) and raises it in the corners, and moves `q_v`
the opposite way — a light, smooth switching by sector, learned from data, with
no hand-set thresholds. It is produced with **no critic and no exploration
noise**: a supervised fit plus one safety cap. In the code:
`scripts/fit_policy_to_grid.py`; the network is
`results/fitted_policy_icra_t2_raceline_kv0.50.npz`.

**Cold: 2.48 / 2.49 / 2.76 laps — the best result, above the best constant, and
reproducible.** The gain over the constant is exactly the sector-dependence,
since the grip claim is the same.

## Online, RETURN critic — adaptation that holds (situation-dependent + online)

Starts from the grid-fitted network and keeps learning *while driving*, with
TD(λ). Its **critic** — the learner's estimate of how good the situation is — is
a small separate model fitted to the actual reward (lap time), which never sees
the weights. The actor is nudged by small random perturbations of the emitted
weights, reinforced when they help (the score-function estimator). It **holds**
the fitted network's performance but does not improve on it here.

**Cold: 2.42 / 2.33 / 2.74 laps — reproduces, roughly the fit.**

## Online, MPCC critic — a cautionary row (situation-dependent + online)

Same idea, but the critic is the controller's *own* optimal cost, `V = −J*`,
which is free to compute but points the learner in a direction fixed by units
rather than by driving (see {doc}`formulation`). It drifts to aggressive
weights. In the experiment it *reported* the highest laps — but that evaluation
ran on a solver already warmed by the learning run, and those weights only stay
feasible from a warm start. Driven **cold**, from a standing start, the network
covers about a lap and leaves the track.

**Cold: 2.31 / 1.06 / 1.01 laps, all off track.** It is kept in the table as a
worked example of why every result must be re-driven cold, and of why a critic
in the controller's cost units is the wrong learning signal.

## Why "cold" is the honest column

A policy is only useful if it works from the state the car is actually in. The
online rows were first evaluated on a solver that had been running for the whole
learning session, which flatters an aggressive policy. Re-driving cold —
`scripts/drive_policy.py` for a standing start, `scripts/warm_cold_check.py` to
see both — is what separates a real policy from one that rides a warm solver.
The grid-fitted network is the same cold and warm; the MPCC-critic network is
not.
