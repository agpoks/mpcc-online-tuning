# TODO — mpcc-online-tuning

Ordered by what unblocks a result. Everything already measured lives in
`docs/source/`; this file is only what is **not** done.

## MUST HAVE — acados, because this has to run on the car

**The target is a real-time MPCC on the physical car, via acados code
generation.** This is a requirement, not a preference, and it constrains
choices being made now rather than being a port to do at the end.

IPOPT is the development solver. It is not what ships: `nlpsol` re-solves from
scratch every tick and there is no C to flash. acados gives an SQP-RTI solver
generated to C, which is the only version of this that closes a loop at 20-50 Hz
on the vehicle.

- [ ] **Generate the OCP through acados**, copying the pattern from
      `MPCC_planner_acados/scripts/generate_acados_solver.py` per the standing
      rule below. `tools/`/`mpcc_tuning/rti.py` already hand-rolls an RTI; that
      was always a stand-in for this.
- [ ] **Keep every model expression acados-generatable.** Practical
      consequences for what is being written *today*:
      - `DynamicBicycle.f_sym` must stay pure CasADi `SX`/`MX` with no Python
        branching on values. It currently does — keep it that way.
      - The sub-stepped Euler integrator is fine (acados has ERK/IRK too), but
        the *choice* must stay explicit rather than implied by a Python loop
        that cannot be exported.
      - Anything non-smooth (`fabs`, `fmin`/`fmax` in the friction-ellipse and
        slip penalties, the smoothed `abs` in the barrier) is acceptable to
        IPOPT and **must be checked against acados' Hessian approximation**,
        which is Gauss-Newton by default. A cost acados cannot linearise is a
        cost that does not ship.
- [ ] **Soft constraints must map onto acados' own slack machinery**
      (`Z`/`z`/`Zl`/`zl`) rather than this repo's hand-added slack variables.
      The grip, CBF and obstacle rows all currently carry explicit slacks in
      `w`; acados expects them declared as soft constraints instead.
- [ ] **The envelope gradient has to survive the port.** `grad_theta` reads
      `lam_g` out of the IPOPT solution; acados exposes multipliers differently.
      Until that is written and checked against finite differences on the same
      problem, none of the online tuning runs on the car.
- [ ] **Decide where the tuner lives** — on-vehicle beside the solver, or
      off-board. TD(lambda) with an LTC is small, but it has to be honest about
      the tick budget it takes from the solver.
- [ ] Verify timing: report solve time per tick at the horizon actually used
      (12 on the synthetic tracks, 40-50 on ICRA), not a best case.

**Standing rule for this repo: no dependency on another of my repos.** Where an
existing repo has a good pattern (`MPCC_planner_acados` for the acados OCP,
`MPCC_controller_ipopt` for the dynamic tyre model), **copy it in and adapt**,
with a comment saying where it came from. The one exception is
`scuderia_gym_jax`, which is a *plant* and stays an optional extra — the bridge
is `mpcc_tuning/plant_scuderia.py` and it already works.

### The gauge fix is harmful — measured 2026-09-02

On the parameterisation that PASSES the oval gate (`q_c=0.3, q_l=200, r_d=0.1`,
7.72 laps, 100% solve), both tuner configurations, three seeds:

    feature group      gauge fixed + no decay      default tuner
    sector                    0.0%                  4.1% +-1.1%
    opponent class            0.0%                  3.7% +-0.7%
    curvature preview         0.0%                  0.9% +-0.1%
    gap                       0.0%                  2.7% +-1.5%
    corridor width            0.0%                  0.9% +-0.3%

Absolute spread of `q_v/q_c` across sectors: **1.19 with the default tuner
against 0.0013 with the gauge fix** — a factor of about a thousand.

**The gauge fix suppresses the variation it was meant to enable.** Pinning the
mean of the six log cost weights INSIDE the existing per-weight box collapses
the reachable ratio range: raising `q_v` requires lowering `q_c` in lockstep and
the per-component bounds cut that short, leaving a reachable `q_v/q_c` span of
about ×2. The gauge freedom is real — `V/c` is constant to four decimals — but
this remedy removes the unidentifiable direction *and most of the useful one*.

- [ ] **Do not ship `gauge_fix=True` as it stands.** It is worse than the
      default on every feature group.
- [ ] Redesign it if it is wanted: project the gauge out in a basis whose
      coordinates are the RATIOS, and bound those, rather than pinning the mean
      and clipping into a box drawn for the original coordinates.
- [ ] Note that neither arm passes the 5% bar. 4.1% +-1.1% is still decorative,
      which is consistent with the direct search: the whole prize for perfect
      per-situation tuning is +3.5% to +8.9%, so the advantage signal for
      deviating sits near the noise floor and a constant is close to correct.

### On a tyre model the tuner is 94% worse, and it is not exploration — 2026-09-02

Steps 1 and 2 of the stated order, run on `scuderia_gym_jax`'s STD drift model
(Pacejka tyres fitted to RC-car recordings) instead of the kinematic bicycle.

**Step 1 passes.** `q_c=0.3, q_l=50, q_v=1.0, r_d=5` completes 5.97 laps of the
oval at 100% solve. It needs `r_d = 5` where the bicycle wanted `0.1`: fifty
times the steering-rate damping, because the car can now slide. The bicycle
grid never searched that region, which is why it looked like no baseline
existed.

**Step 2 fails, badly.**

    track      fixed   tuner
    oval        5.29    0.33, 0.22
    circuit     1.04    0.56, 0.31

**It is not the actuator exploration.** With `explore=0` the tuner is just as
bad -- 0.24 and 0.18 laps against the baseline's 4.08:

    mode     explore   laps    q_v/q_c emitted
    fixed          -   4.08    3.33 +-0.00
    tuner       0.05   0.30    3.54 +-2.04
    tuner       0.05   0.19    4.32 +-1.27
    tuner       0.00   0.24    3.35 +-1.33
    tuner       0.00   0.18   10.16 +-7.62

The cause is **the weights it emits**. The baseline sits at `q_v/q_c = 3.33`
and the tuner drifts up to 3.5, 4.3, and in one seed 10.2 +- 7.6. On a
kinematic bicycle a higher ratio means "go faster"; on tyres it means asking
for grip that is not there. This is the proxy-optimised-past-validity failure
with real physics under it for the first time.

- [ ] **The baseline does not transfer between tracks.** 5.97 laps on the oval,
      1.04 on the circuit -- below the 2-lap gate. A per-track baseline is
      needed before any cross-track claim.
- [ ] The tuner needs a **grip-aware** cost or a constraint the learner cannot
      exceed, not a smaller step size. Its objective is progress and progress
      is exactly what spins the car.
- [ ] μ = 0.6 on the tightest corner STILL does not bind: fixed dry 5.44 laps,
      fixed wet 5.42. The surviving baseline is too slow to ask for the grip it
      lost, and the configuration fast enough to notice spins. **That gap is
      the result**: on this plant there is no fixed weight vector that is both
      fast enough for the wet corner to matter and slow enough to survive.

### Softening the grip constraint broke the envelope gradient — 2026-09-02

`k_v` is `theta[7]` and it sits in the grip row, so **theta is in `g`**. That was
already known and commented at `mpcc.py:182`: the analytic gradient goes wrong
when that row is ACTIVE, with an xfail recorded against it.

What the note did not anticipate is that **softening the row made it active
almost everywhere**. Hard, the row was either strictly satisfied (inactive,
lambda = 0, gradient fine) or infeasible (no solve at all). Soft, it is active
whenever grip binds even slightly, because that is exactly when `Sg > 0`. The
fix for the 34% infeasibility converted "no solve" into "a solve whose gradient
is wrong", which is the more dangerous failure of the two.

Measured on `Track.circuit()`, cosine between analytic and finite-difference
`dQ/dtheta`, all 16 perturbed solves converging in every row:

    grip slacks   CBF     cosine
    soft          off     +1.0000
    soft          ON      +0.0811     <-- broken
    hard          off     +1.0000
    hard          ON      +1.0000

Neither alone does it; it is the interaction. The CBF holds the car nearer the
boundary and slower, which is precisely the regime where the grip row binds.

- [ ] **This does not explain the tuner regression.** `cbf` defaults to False
      and no tuning experiment turns it on, so those runs are in row 1. Do not
      reach for this as the cause without measuring first.
- [ ] **Measure how often the grip row is active during a tuning run** once
      step 1 gives a baseline that drives. If `Sg > 0` on a large fraction of
      ticks, the gradient is wrong that often and this becomes the leading
      explanation for step 2 -- ahead of the discount horizon, which was tested
      and did nothing (gamma 0.98 -> 0.19 laps, 0.995 -> 0.21).
- [ ] Either take `k_v` out of theta, or add the `lambda^T dg/dtheta` term the
      envelope formula needs when theta is genuinely in `g`. The second is
      correct and is not hard; the current `grad_theta` returns `dJ/dtheta`
      alone.
- [ ] The paper's CBF section claims "gradient unaffected -- theta stays out of
      `g`". That is **false as written** and must be corrected before this
      constraint appears in any result.

### acados solver options, measured — 2026-09-02

Suggested settings from a working acados setup, all tested on the dynamic model
at N=12, r_a=0.05, with the RTI shift and true-projection s:

    configuration                          laps   solve   ms/tick (worst)
    nonlinear corridor, soft, SQP it=8     3.40    77%     6.7  (45.2)   <- best
    linear half-space corridor, it=8       1.12    91%     5.0  (21.4)
    linear corridor, soft, it=16           1.20    83%     5.8  (51.4)
    hpipm ROBUST + qp_warm_start=2         0.04     5%     2.2  ( 5.3)
    ROBUST + hard corridor                 0.28    13%     2.4  ( 5.4)
    linear corridor + HARD, it=8/16        0.27    25%     2.3  ( 7.2)
    SQP max_iter 1 / 2 / 3 (any options)   0.04-0.06  0-5%

- [ ] **`SQP_WITH_FEASIBLE_QP` is not in acados v0.1.9** -- the installed build
      accepts only `('SQP', 'SQP_RTI')`. It is exactly the feature this problem
      wants (a QP that stays feasible when a nonlinear constraint is violated
      at the linearisation point), and its absence is why the hard corridor
      cannot work here. **Upgrading acados is the highest-value next step**,
      ahead of any further option tuning.
- [x] **`hpipm_mode = ROBUST` and `qp_solver_warm_start = 2` are HARMFUL here**,
      not neutral: 3.40 -> 0.04 laps. Do not carry them over from another
      setup without measuring.
- [x] **The linear half-space corridor does what it should to the QP** --
      solve 77% -> 91%, worst tick 45 -> 21 ms -- and still loses laps. The
      geometry is re-linearised once per tick, so during the 8 SQP iterations
      the half-spaces are fixed at the previous tick's prediction. Worth
      revisiting if acados ever exposes per-iteration re-linearisation.
- [x] The hard corridor fails whether the row is nonlinear OR linear (25%
      solve, car does not move), so its failure is not about linearisation.
      `timeout_max_time` / `timeout_heuristic` / `nlp_solver_warm_start_first_qp`
      are also absent from v0.1.9.

### The ICRA circuits work: 2-3 laps, clean, in real time — 2026-09-02

    track   originally   now            what did it
    T1        0.18       2.05 clean     horizon 40 -> 25, q_v 1.0 -> 0.2
    T2        0.37       3.08 clean     r_d -> 0.02, corridor widened, N=25

9-11 ms/tick, 99% usable steps. Both are STEP-LIMITED, not crashing.

Four things had to be right, and three of my own assumptions were backwards:

- **r_d wanted to go DOWN, not up.** Every sweep went 0.5 -> 3 -> 10, which is
  backwards for a track that needs MORE steering. At r_d = 0.02, T2 went 1.15
  -> 1.93 laps. Twenty-five times lower than anything previously tried.
- **A SHORTER horizon fixed T1.** N=60 gave 0.47, N=40 gave 0.59, N=25 gives
  2.05 -- and costs 10.8 ms against 18.1. The standing note in this repo said
  T1 needed the horizon RAISED to 40 because "0.6 s of lookahead cannot see
  through a 0.7 m hairpin". That was measured on the kinematic controller and
  is now wrong: more lookahead on an un-turnable corner does not help, it just
  makes the NLP harder.
- **Less progress weight, not more.** q_v 1.0 -> 0.2 on T1.
- **r_a is the single biggest lever** and wants to be LARGE: 0.05 -> 6.0 took
  T1 from 0.18 to 1.20 before any of the above.

Geometry, measured:

    track   min corner radius   half-width      un-turnable fraction
    oval          2.46 m        0.75 const           0.0%
    T1            0.69 m        0.55-2.18            1.2%
    T2            0.71 m        0.30-1.88            1.3%

The car's minimum turn radius is wheelbase/tan(STEER_MAX) = 0.78 m, so ~1.2%
of both ICRA laps is TIGHTER THAN THE CAR CAN TURN. Raising STEER_MAX to 0.50
takes that to 0% and helped T2 (1.93 laps at r_d=0.02); T1 was insensitive.

- [x] **T2's corridor widened 1.35x.** T1 has an occupancy grid and T2 does
      not, so the map-vs-raceline ratio was MEASURED on T1 (1.58x at the
      tightest point, 1.17x median) and applied to T2 by proportion.
- [ ] **That widening is an assumption, not a measurement.** It presumes the
      optimiser was equally conservative on both tracks. If a T2 occupancy grid
      exists anywhere in the archive, raycast it instead -- the 3.08-lap figure
      moves if the real corridor is narrower.
- [ ] `k_v` is a DEAD WEIGHT for dynamic models: the grip row auto-disables and
      q_vref defaults to 0, so it has no path to the cost. A tuner given eight
      weights will waste capacity on it. Same failure as the q_v/q_l dead span.

### RESOLVED: globalization was the gap. The controller drives, and overtakes — 2026-09-02

**acados defaults to `FIXED_STEP`: the full SQP step, no line search, no
acceptance test.** IPOPT runs a filter line search and rejects steps that do
not improve. That was the whole difference. Measured on the oval, dynamic
model, everything else held fixed (3 repeats, perturbed starts):

    configuration        FIXED_STEP   MERIT_BACKTRACKING   FUNNEL_L1PEN
    soft SQP                2.27          4.79                5.52
    feasible-QP soft        2.17          5.46                7.12   <-- best
    feasible-QP hard        0.97          1.53                2.95
    feasible-QP hard+lin    1.94          1.90                3.22

**7.12 laps beats IPOPT's 5.12, at 7.6 ms/tick against 126 ms.**

The negative control is what makes this readable: raising iterations 8 -> 50 ->
300 gave 3.16 / 3.06 / 2.43 laps. More iterations of an uncritically accepted
direction cannot help, which is why that sweep was flat.

**Overtaking works** -- first time on a controller that can drive. acados
`fqp_soft_funnel`, opponents scaled to the ego's MEASURED solo pace (2.20 m/s):

    opponent   laps   passes   min gap   p99 ms
    0.55x      4.79     3       0.320     25.3
    0.85x      3.11     1       0.313     27.5   <-- left the track
    1.10x      4.70     1       0.254     24.1

Every pass cleared the 0.24 m keep-out, so none was a disguised collision. The
1.10x pass is genuine: the opponent holds CONSTANT speed while the ego varies
around its mean, so it is quicker on the straights.

- [ ] **0.85x is the weak case** -- one pass then off the track. That is the
      "commit or don't" condition and the one worth understanding.
- [ ] **Three of five tracks still fail.** oval 8.00 and circuit 2.22 clear the
      two-lap bar; ICRA T1 0.18, T2 0.37, 2025 0.06. Usable steps fall to
      46-67% and p99 reaches 85-190 ms against a 50 ms budget. The signature is
      horizon cost and oval-tuned weights, NOT the car losing control --
      sideslip stays low. **This is a weight-tuning problem, which is what the
      online tuner is for.**
- [x] Figures: paper/figures/acados_globalization.png, acados_tracks.png,
      overtaking_results.png, overtaking.gif.

Harness safeguards added after two full runs were silently voided:

- [x] `sanitize_name` in `build_ocp` -- CasADi rejects consecutive underscores
      and non-letter leading characters, and the failure message says only
      "SXFunction", never "naming".
- [x] A sweep where nothing ran now **exits non-zero** and says so. Twice a run
      skipped every variant, exited 0, and read as success.
- [x] Repeats must perturb something PHYSICAL. Seeding the plant RNG changes
      nothing here, so "3 repeats" was one run three times and every std was
      0.00.

### RESOLVED: acados needed iterations, not fixes — 2026-09-02

The dynamic MPCC drives. IPOPT: **5.12 laps**, 99% solve, mean |e| = 0.024 m,
peak sideslip 3.3 deg. acados with a converging solver: **4.77 laps**
(kinematic) and 2.16 (dynamic).

**The whole acados difference was SQP_RTI's single iteration.** Matched
model, constraints, cost, weights, horizon and integration step:

    iterations      dynamic laps    ms/tick (worst)
    1 (RTI)             0.31          2.8  (32)
    2                   0.27          5.0  (38)
    4                   1.42          6.0  (79)
    8                   2.09          6.2  (51)
    50                  2.16         10.9 (175)

**Ship max_iter = 8.** 2.09 laps dynamic at 6.2 ms/tick, 3.98 laps kinematic at
4.1 ms mean and 29.4 ms worst -- inside the 50 ms budget with the car driving.

- [x] Verified like-for-like before concluding: same cost (identical to 1e-6),
      same trajectory when seeded (1e-4), same weights, same horizon. Earlier
      acados runs used `r_a = 0.01` and `N = 20` against IPOPT's `0.05` and
      `12` -- and 0.01 is the value that SPINS in IPOPT too, so those runs were
      testing the spin configuration and blaming the solver.
- [x] **Hard corridor is settled: it is wrong here.** Retested under the
      working configuration: 17% and 9% solve, peak speed 1.14 m/s, the car
      does not move. Soft is correct, and the CasADi backend's hard row is
      tolerable only because a full solve absorbs it.
- [ ] Worst case at max_iter = 8 is 51.3 ms for the dynamic model, marginally
      over budget. Bound it properly before the car.

### acados is real-time but does not drive — 2026-09-02

**The real-time goal is met.** Per control tick on the oval, against 50 ms:

    backend            kinematic          dynamic
    IPOPT           126 ms (worst 903)  1840 ms (worst 29056)
    acados          2-5 ms (worst 35)   3.6 ms (worst 15)

That is 64x and 459x, and both fit the budget worst-case. This is the acados
MUST HAVE demonstrated rather than asserted.

**And the two backends solve the SAME problem.** Verified directly rather than
by inspection:

* stage cost at the same (x, u, theta): CasADi -0.108000, acados -0.108000,
  difference 0.
* seeded with the CasADi solution, acados converges to it: dv = 0.0014,
  ds = 0.0063 at k = 20, status 0.

So the dynamics, integrator, grip row, reference coupling and cost are all
equivalent. **The remaining fault is in the closed loop, not the formulation.**

Best acados result so far: 0.45 laps (dynamic), 0.31 (kinematic), against IPOPT
kinematic's 7.16. Not a working controller.

Fixed along the way, each real and none sufficient on its own:

- [x] `WEIGHT_NAMES` grew to 8 and `build_ocp` still unpacked 6 -- the acados
      backend had not built at all since. Now unpacked by name.
- [x] `spline_mode="parameter"` decouples `s` from the path. The plant's `s`
      runs 1.3 m ahead of the car after 39 ticks, so the frozen reference sat
      at the wrong place. `spline` mode builds fine, contrary to the docstring's
      hint, and drops `np` from 11 to 8.
- [x] Feeding the TRUE track projection instead of the drifting virtual `s`:
      dynamic solve 60% -> 83%.
- [x] The grip row was in the CasADi OCP and NOT here. Porting a cost without
      its constraints is why the "same" controller drove differently.
- [x] **The RTI warm start was never shifted.** IPOPT re-solves to convergence
      so a stale guess costs iterations; RTI takes ONE step from the guess it
      is handed. Shifting: dynamic 0.18 -> 0.45 laps, QP NaNs 94 -> 17.
- [x] `n_dyn` read off the model instead of a hardcoded 7, so the servo state
      does not silently desynchronise the two backends.

Measured and REJECTED, so nobody retries them:

- **Hard corridor -- and the earlier claim about it was wrong twice.**
  First, the constraints were never different: for a track with
  `variable_width=False` (the oval) `mpcc.py` takes its ELSE branch,
  `lbg = -margin, ubg = +margin`, which is the same two-sided hard bound acados
  has, with the same +-0.630. The "hard in CasADi, soft in acados" reading came
  from the variable-width branch, which this track does not use.
  Second, "hard is worse" was a measurement artefact. Aggregate solve rate over
  a whole run measures WHEN THE CAR LEFT, not whether the constraint works: a
  hard corridor is permanently infeasible once the car is outside, so every
  post-departure tick counts against it. Read tick by tick, the hard corridor
  tracks BETTER early (|e_c| = 0.028 against 0.078) and does not fail at the
  tick the soft one does.
  **The corridor is a red herring either way.** The soft-corridor QP dies at
  tick 8 with |e_c| = 0.078 against a margin of 0.630 -- the car is
  comfortably inside the track when the solver fails.
- **Full SQP instead of SQP_RTI.** 0.13 / 0.02 laps. RTI's single iteration was
  never the problem by itself.
- **acados integrator steps 2 -> 4.** 147 -> 146 NaNs.
- **Blend width** 0.15 -> 0.50 at midpoint 0.70: worse.
- **V_BIAS on the slip-angle denominators** (0.001 / 0.3 / 0.5 / 0.8): all
  0.05-0.21 laps. Kept in the model anyway -- the previous 1e-3 guard did
  nothing at 0.79 m/s, and capping the Jacobian at 1/V_BIAS is right in
  principle.

Best acados configuration measured: soft corridor + true-projection s + RTI
shift, giving 0.45 laps (dynamic) and 0.31 (kinematic) at 83-87% solve. Adding
the hard corridor to that makes it worse (0.35 / 0.28 at 23-27%).

- [ ] **Stop permuting solver settings.** Six configurations, none drives. The
      two backends provably solve the same problem (cost identical to 1e-6,
      trajectories to 1e-4 when seeded), so the fault is in how the closed loop
      drives the solver and not in any single option.
- [ ] **Do this instead: run both backends from the SAME initial condition and
      diff the trajectories tick by tick.** Where the applied controls first
      diverge is where the closed loops differ. That took two minutes for the
      cost comparison and settled a question six experiments could not.
- [ ] The QP dies at tick 8 with the car well inside the corridor, at a benign
      state (vx = 1.26, e_c = 0.078, on a straight, kinematic model). Nothing
      about that state explains a NaN, which is the thing to chase.

### acados: the OCP is fine, the dynamic model at low speed is not — 2026-09-02

Isolated properly rather than guessed at. Solving 30 times from a seeded state
at vx = 2.0:

    kinematic                 nx=5  30/30
    dynamic, as configured    nx=8  30/30
    dynamic, no penalties     nx=8  30/30
    dynamic, NO_REGULARIZE    nx=8  30/30
    dynamic, MIRROR           nx=8  30/30

**The acados OCP is correct.** The EXTERNAL cost, the soft-constraint slacks,
the parameter packing, the EXACT Hessian and CONVEXIFY all work. So do the
friction-ellipse and slip penalties. Every one of those was a suspect and every
one is exonerated.

The failure is in CLOSED LOOP and only for the dynamic model:

    acados kinematic, 60 ticks   no QP failure at all
    acados dynamic               QP status 3 (NaN) at tick 2

- [x] **Cause found, and it was a change I made.** The blend midpoint had been
      moved from 1.40 to the plant's own 0.70, on the reasoning that matching
      the plant is more accurate. Measured, that is an eighteenfold regression:
      first QP failure at tick 2 (0.70) against tick 36 (1.40) and tick 27
      (2.00). Reverted to 1.40/0.30, with the reasoning written into model.py
      so it does not get "corrected" back.
- [x] The controller's blend is NOT the plant's blend. The plant blends because
      the dynamic limb is meaningless at vx -> 0 and integrates at 2 ms. The
      controller must additionally LINEARISE for a QP at 12.5 ms, and the slip
      angles have Jacobian entries of order 1/vx.
- [ ] **Still fails at tick 36.** Better by 18x and still not a working
      controller. Starting above the blend (v0 = 2.5) reaches tick 17 with the
      old constants, so speed alone is not the whole story either.

Rejected by measurement, so nobody retries them:

- **acados integrator steps** 2 -> 4 (h = 25 -> 12.5 ms): 147 -> 146 NaNs. No
  effect, despite the same fault being the cause of three bugs on the CasADi
  side. Kept anyway because h = 25 ms genuinely cannot resolve a 17.8 ms mode.
- **Widening the blend** 0.15 -> 0.30 -> 0.50 at midpoint 0.70: WORSE, tick 1.
  It is the midpoint that matters, not the width.

### IPOPT cannot run the dynamic MPCC — 2026-09-02

Measured per control tick on the oval, against the 50 ms budget at dt = 0.05:

    model        mean ms/tick    worst ms    x budget
    kinematic          126.0        902.9        2.5x
    dynamic           1839.9      29055.6       36.8x

A single tick taking **29 seconds**. This is not a tuning problem and no cost
weight fixes it. Two consequences:

- [ ] **"The kinematic MPCC works" needs qualifying.** It drives 7.16 laps
      because simulation does not care how long a tick takes. At 126 ms mean
      and 903 ms worst it misses every deadline on a real car. Every lap count
      in this repo is an OFFLINE result.
- [ ] **acados stops being a deployment step and becomes the blocker.** IPOPT
      re-solves a full NLP each tick; SQP-RTI does one QP from the previous
      solution. That is the difference between 100+ ms and single digits, and
      it is the only way the dynamic model runs on the vehicle at all. See the
      MUST HAVE section at the top.

Also measured, and worth keeping because it repeats a pattern:

- **More iterations makes it worse.** `max_iter` 300 -> 1000 took solve success
  from 93% to 87% at 10 s/tick. Identical to the hard-grip-constraint finding
  earlier the same day: a small budget stops at a usable half-solution.
- **Integration fidelity and drivability trade against each other.** The
  accurate integrators (`sub=4` RK4, `sub=8` Euler) drive at 1.08-1.09 laps
  and then spin; the inaccurate ones (`sub=2` RK4, `sub=4` Euler) solve at
  94-99% and barely move (0.06-0.07 laps). Neither end produces a car that
  drives, so the missing ingredient is not integration accuracy.
- **The spin is not a compute problem.** `sub=8` Euler at 794 ms/tick and 94%
  solve still spins. Slow solves and the spin are separate faults.

### The prediction model was wrong in four places — 2026-09-02

Validated the only way that settles it: put the plant in a state, apply a fixed
control sequence, roll plant and model forward over one horizon, compare. None
of the faults were in the equations -- those cross-check clean against
On-Track-SysID and CommonRoad's own STD. All four were setup or numerics.

    case                        error before   after
    straight, accelerating vx   0.833 m/s      0.008
    steady cornering       r      -37%         -1.2%
    corner entry braking   r      -39%           -4%
    low speed              vx     0.287        0.002

1. **The plant started every episode in a skid.** `ScuderiaPlant.reset` set
   body speed to 1.0 m/s and left `omega_f = omega_r = 0`. Locked wheels under
   a moving car: tyres saturated in longitudinal slip, combined slip took the
   lateral grip with it. **Every STD result recorded before this is affected**,
   including the "kinematic controller reaches 8.31 laps" reference -- that one
   was driving a skidding car and should be re-measured.
2. **`DRAG = 0.15` is fictional for this plant.** It coasts 3.000 -> 3.000 m/s
   at zero throttle. In `ctrl_mode="accl"` there is no speed-proportional loss.
3. **The yaw mode was under-resolved.** Its time constant is 17.8 ms
   (`r_dot = 77.5` from rest toward 1.378) and the NLP integrated it with 25 ms
   steps. `sub = 2 -> 4`; sub=4 and sub=8 agree to four decimals. The plant
   resolves it at 2 ms.
4. **Wheel inertia was missing.** Each wheel adds `I_y_w/R_w^2 = 1.04 kg`, so
   the car accelerates as though it weighed 6.33 kg, not 4.25. That is a gain
   of 0.671 on the acceleration command, not a drag term -- commanding
   2.0 m/s^2 moves the plant at 1.328, predicted 1.343.

Blend constants also now match the plant's own (`v_s = 0.70, v_b = 0.15`; they
were 1.4/0.30, which held the model kinematic to far too high a speed).

- [ ] **The steering servo is still not modelled.** First tick after a steering
      step: plant `r = 0.938`, model `r = 1.945`. The plant carries delta as a
      state behind a rate limit (`sv_max = 4 rad/s`) and a transport delay; the
      controller applies it instantly. They agree by k=3, but the MPCC acts on
      k=1. `MPCC_controller_ipopt` models this properly -- delta is a state and
      the input is its RATE. That is the pattern to copy.
- [ ] **Re-measure everything on STD.** Points 1-4 all changed the plant or the
      prediction, so no lap count in this file predating 2026-09-02 carries.

### The controller had no tyres either — 2026-09-02

Swapping the *plant* to STD was only half the swap. **Every `MPCC(...)` in this
repo passed `model=KinematicBicycle`**, including in the experiments that were
supposedly measuring tyre behaviour. The controller predicted a car with no
sideslip while driving a car that drifts.

That explains the baseline. `r_delta = 5.0` is fifty times the bicycle's 0.1,
and it was never a tuning result — it is the damping needed to stop a blind
controller exciting dynamics it has no state for. A baseline reached that way is
not "stable but not racing ready", it is a car held still enough not to fall
over, which is why grip (item below) could not be made to bind and why the
tuner had nothing to improve.

- [x] `DynamicBicycle` in `mpcc_tuning/model.py` — single-track planar with
      simplified Pacejka, adapted from
      `MPCC_controller_ipopt/MPCC_controller.cpp::build_dynamics()`. Input
      changed from duty/steering *rates* to this repo's angle/acceleration, and
      parameters taken from `scuderia_gym_jax`'s `rc10_default.yaml` rather
      than that source's 1:43 Liniger car.
- [x] MPCC state generalised to `NS = 5 + model.n_dyn`, laid out
      `[x, y, psi, vx, s, vy, r]`. **`s` stays at index 4**, so every index the
      kinematic layout established keeps its meaning and no existing experiment
      changes. A short `state5` is zero-padded in `MPCC._pad`.
- [x] Sub-stepped Euler in the NLP, not RK4 — the source repo's own documented
      choice. With RK4 the cold solves returned `Maximum_Iterations_Exceeded`
      and `Restoration_Failed`; four Pacejka curves per shooting node is a
      Hessian IPOPT cannot get through.
- [x] `ScuderiaPlant.state_dyn()` feeds the real sideslip back: their STD state
      carries `beta`, so `vx = v cos(beta)`, `vy = v sin(beta)`.
- [ ] **Re-run step 1 with the dynamic controller.** The baseline to look for
      is stable-but-not-fast *for the drift car*, and it should not need
      `r_delta = 5`. Every step-2/3/4 number recorded before this is a
      kinematic controller's and does not carry over.

## 0. The design this project is supposed to follow — and does not

Stated repeatedly and never written down here, which is why it keeps being
lost:

> **Start from a stable working parameterisation.** It need not be perfect or
> the fastest. **Then adapt from there** — by track, sector, opponent, surface,
> grip.

The *shape* of that is implemented: the policy anchors at θ₀ and emits
deviations, and the squash spans are measured from θ₀. **The baseline is
wrong.** θ₀ is `q_c=1.0, q_l=200, q_v=2.0, r_d=1.0`, used in 7 places across
the experiments, and it does not drive the competition tracks — measured at
0.5–0.6 m covered on ICRA T1. What works there is `q_c=0.1, q_l=50, r_d=0.1`
at horizon 40, which reaches 125.3 m.

So the learner has been asked to adapt **around an operating point that
crashes on the tracks we care about**, with its output range measured from that
same point. Every adaptation result is a deviation from a bad baseline, and
that is a more likely explanation for weak adaptation than any of the five
mechanisms investigated on 2026-08-31/09-01.

- [ ] **Establish a stable θ₀ per track, verified to complete laps**, before any
      further learner work. Not the fastest — the one that finishes.
      Candidates measured so far: circuit `q_c=1.0` horizon 12 (92 m, clean);
      ICRA T1 `q_c=0.1, q_l=50, r_d=0.1` horizon 40 (125 m, 1.6 laps);
      ICRA 2025 same weights horizon 50 (148 m, clean).
- [ ] **Re-anchor the policy on it**, and re-derive `THETA_LO`/`THETA_HI` around
      the new θ₀ so the spans mean something. The current box was drawn around
      the old anchor and put two weights on their own ceilings (§ dead span).
- [ ] **Only then** re-run the adaptation experiments. Everything measured
      before this is a deviation from a baseline that does not drive.
- [ ] The horizon belongs in the baseline too: 12 on the synthetic tracks, 40–50
      on the competition ones. It is a structural parameter of the OCP, not
      something the weight policy can adapt, and 0.6 s of lookahead cannot see
      through a 0.7 m-radius hairpin (2.2 m of arc against 1.8 m of plan).

### Two figures assert what their own data denies — 2026-09-01

`adaptation.png` and `ltc_gate.png` were regenerated with the current
configuration and are WORSE, not stale. Neither is synced into `paper/figures/`
and neither should be until item 0 is done.

* **`adaptation.png`** is titled "the ratio crosses the behaviour boundary
  where it passes". In the figure `q_v` is flat at 3.0 and `q_c` runs
  1.9 → 2.4, so the ratio goes 1.58 → 1.25 and **never crosses 1.0**. The
  episode panel shows 0–3 m covered: the car barely moves.
* **`ltc_gate.png`** is titled "the hand-written schedule wins" and **does not
  plot the schedule**. The three arms it does plot crash 47–55% of episodes
  with error bars that overlap entirely.

Both run on the oval with the θ₀ anchor that item 0 shows crashes on the
competition tracks. The titles are inherited from an older result and the data
underneath has moved.

- [ ] Regenerate both **after** a stable θ₀ exists, on a track that drives.
- [ ] Rewrite both titles from the figure that is actually produced. A title
      that states a conclusion the axes do not show is worse than no figure.
- [ ] If the LTC gate still fails to separate at that point, cut the figure:
      the paper already calls those numbers void, and a plot of three
      indistinguishable arms is not evidence of anything.

## Where things stand

Oval, 26 episodes, same seed, last 8 episodes:

| | covered | off-track |
|---|---|---|
| baseline, one global θ | 7.6 m | 8/8 |
| event-triggered updates, *deferred* | 7.4 m | 8/8 |
| event-triggered updates, *discard* | 38.5 m | 6/8 |
| behind a predictive safety filter | 68.1 m | 0/8 |
| **per-segment θ (curvature)** | **78.0 m** | **0/8** |

Tracks: `oval` (straights + two 180s), `mixed` (harmonic, no 90s),
**`circuit`** (all four named sectors, 47.2 m, min radius 2.59 m).

Per-segment θ is now **confirmed over six seeds**
(`experiments/per_segment_seeds.py`): global 8.9 m (sd 0.9), collapsing on
**6/6** seeds; per-segment 77.8 m (sd 0.2), collapsing on **0/6**. The
distributions do not overlap.

Real-time is done: `mpcc_tuning/rti.py`, 1.9 ms mean / 3.4 ms worst at N=12
against 50 ms at 20 Hz. **It is CasADi `qrqp`, not acados** — see item 3.

---

## Status board — done, and whether the paper says so

Two different questions, and a thing can pass the first and fail the second.
**Done** = measured and committed. **In the paper** = a section or table carries
the result, with a figure that is `\ref`'d in the text.

| item | done | in paper | note |
|---|---|---|---|
| 1. per-segment, six seeds | yes | yes | §V, `reversal.png` |
| 1b. named sectors | yes | yes | §VI, `tracks.png` |
| — sector4 vs curvature3 | yes | yes | null, 0.8 SE |
| — circuit reverses item 1 | yes | yes | §V-A |
| 2a. obstacle keep-out | yes | yes | §IV-C |
| — **elliptical** keep-out | yes | **NO** | needs a subsection |
| 2b. safe region | partial | yes | ceiling bracketed 2–10, not pinned |
| 2c. physics features | yes | partial | 18 features listed, no result yet |
| 2d. LTC gate | yes | yes | §IX, NOT CLAIMED |
| — online adaptation | yes | yes | §X, `adaptation.png` |
| — **RFLO vs exact RTRL** | yes | **NO** | cosine 0.9997; belongs in §X |
| — trust region + prior | yes | partial | argued in §X, no table |
| 2e. personas | yes | yes | §VII, `behaviour.png` |
| — static vs dynamic | yes | yes | §VII-B, the livelock |
| 3. **acados** | yes | **NO** | paper 1 still says "hand-rolled, not acados" |
| 4. tuner on fitted tyres | no | n/a | blocks 4b |
| 4b. multi-car scuderia | no | n/a | blocked on 4 |
| 5. bound where θ goes | partial | partial | trust region exists |
| 5b. ICRA track | partial | **NO** | 100% in corridor, seam kink open |
| 6. filters on an estimate | no | n/a | |
| 6b. ROS export | no | n/a | |
| animations | yes | yes | 5 GIFs on the animations page |

**Why this table exists.** A result gets measured, committed, and never reaches
the paper. It has happened five times: `gradient_check.png` absent from paper 1;
two figures orphaned in paper 2; five figures placed but never `\ref`'d; both
new GIFs committed but referenced nowhere; and the three rows marked **NO**
above. Check this table before calling anything finished.

- [ ] **Ellipse keep-out into the paper.** 0.635 m along against 0.267 m
      across, where the circle used 0.350 both ways — so every closest-approach
      number reported so far was measured against a keep-out that forbids the
      manoeuvre it was measuring.
- [ ] **RFLO vs exact RTRL into §X.** Cosine 0.9997 is what rules the gradient
      *out* as the cause of the runaway; without it §X asserts the objective is
      at fault instead of showing it.
- [ ] **Paper 1 says the RTI is hand-rolled and not acados. That is now
      false.** `mpcc_tuning/acados_ocp.py` compiles and solves at 1.63 ms mean.
      Either re-run the influence measurement against a real `SQP_RTI` step —
      the check that paper's headline owes — or say plainly that acados exists
      and the check has not been redone.

## Order of work

Sequenced by what unblocks what, not by interest.

1. ~~Per-sector tuning on `Track.circuit()`~~ — **done, and it reversed the
   item 1 result.** See item 1b. The next experiment is instead a task where
   scheduling *can* win, because the circuit turned out to be ceiling-limited.
2. **The `q_v` ceiling** (item 2b). Cheap, and nothing can bound a policy's
   output until it exists.
3. **acados** (item 3). Feasible but not a copy-paste — two structural problems
   documented there. Also settles the open caveat on paper 1.
4. **Personas** (item 2e). Mostly a characterisation sweep once 2b is done.
5. **A network, if anything** (item 2d). Last, gated, and the prior is negative.

## 0. Papers — decided: two

`paper/main.tex` is **paper 1**, *"Differentiating a Real-Time MPC Is Safe"*:
warm starts as memory, the influence recursion, the `max_iter=1` trap. It is a
different story from the weights work and stays that way.

**Paper 2 — situation-dependent cost weights and behaviour parameterisation.**
`paper/weights.tex` — drafted, structure and Method complete, results sections
are `\todo{}` awaiting the runs. It cites paper 1 for the gradient rather than
restating it. What is measured and ready to go in it:

- [x] per-segment θ over six seeds — 77.8 m (sd 0.2), 0/6 collapsed, against a
      global θ's 8.9 m (sd 0.9), 6/6 collapsed. Distributions do not overlap.
- [x] the mechanism: global reaches 77.6–79.1 m on *every* seed and destroys it
      on *every* seed; per-segment reaches the same peak and keeps it.
- [x] the keep-out, with the envelope gradient re-verified under slacks
      (cosine > 0.999 to finite differences with an obstacle active).
- [x] overtake vs follow as two orthogonal axes: behaviour is `q_v/q_c`
      (15/15 cells), safety is `q_v` magnitude.
- [x] the named-sector taxonomy and why pointwise curvature cannot express it.

Not ready, and **must not be claimed until measured**: per-sector tuning results,
the personas as anything learned, the `q_v` ceiling, anything with a network in
it.

**The LTC / behaviour-policy method is written but marked `\unmeasured`**
(paper 2, Sec. VI). It is specified in full — features, the cell, the
`tau_eff = tau/(1 + tau f)` hold-duration argument, the output bounding, and the
gate — because the specification *is* what the next experiment tests. Nothing in
it may be claimed until it is run, and the section states a prediction against
itself: the fixed schedule should be strong because the behaviour boundary is
linear in θ.

**Method sections are now the thing to keep honest.** Paper 1 had none at all
until this session; it now has controller parameters, the TD(λ) algorithm as a
listing, a hyperparameter table, the replay protocol as an enumerated
three-way comparison, and an explicit statement of what is randomised. Paper 2's
Method was written the same way from the start.

- [ ] Paper 1's safety-filter result is **single seed** and is now labelled as
      such in the caption. Re-run it over six seeds to the standard the
      per-segment result now sets, or leave it labelled.
- [ ] Neither paper has been compiled — there is no LaTeX toolchain on this
      machine, and `algorithm`/`algpseudocode` are newly added to paper 1.
      Overleaf will be the first compile.
- [ ] `\todo{}` markers remain in both papers' Introduction and Related work.

**Framing point paper 2 has to make itself, before a reviewer does:** the
six-seed result is a **three-bin curvature schedule, not a sector schedule.**
`Track.segment` bins pointwise |κ| by quantile, and pointwise curvature provably
cannot separate a 90-degree corner from a 180-degree one (item 1b). Calling it
"per-sector" would overclaim.

### CBF as a constraint of the OCP — added 2026-08-31, `MPCC(cbf=True)`

Why in the solver when a filter already exists: **when the cost weights are what
is being learned, anything expressed as a cost term is negotiable.** The policy
can learn a small weight for a safety term and trade it away — item 5's failure
pointed at safety. A constraint cannot be learned away. It also removes a
mismatch: with an external filter the learner differentiates the *unfiltered*
problem while the plant executes the *filtered* action.

Same barrier as `filters/cbf_qp.py` (`h_kind="braking"`, α=0.35, margin 0.18,
lookahead 0.45), so the in-solver and post-hoc versions are the same criterion
and any difference is about *where* it is enforced. `θ` is deliberately absent
from it — measured: `cos 1.000000` unchanged.

200 steps on the oval, no external filter:

    cbf    weights   covered   off track   solve ok
    off    good        28.7m   YES              74%
    on     good        28.7m   no               72%
    off    bad         12.9m   YES              56%
    on     bad         10.9m   YES              46%

With sensible weights it is **free**: same distance, no crash. Note the
baseline — at SPEED_MAX 8 the default weights leave the track over 200 steps
even with the grip and terminal constraints, so the barrier is doing real work.

**The barrier must be smoothed.** `|e_c|` and `|v sin e_psi|` are absolute
values on DECISION VARIABLES, unlike the grip row's `|kappa|`, which is a track
property. With hard `fabs`, solves succeeded **16%** of the time against
pathological weights — the barrier was not making the car safe, it was making
the problem intractable, and an infeasible solve is no safety at all. With
`sqrt(z^2 + eps^2)`, eps = 1 cm: 46%, and the progress cost with good weights
disappears (27.7 → 28.7 m). Gradient cost of the smoothing is small:
`cos 0.999985`, rel 0.0055.

- [x] Gradient unaffected — θ stays out of `g`.
- [ ] **Do not make the barrier margin learnable** until the active-constraint
      gradient question is settled. `k_v` enters the grip row and its analytic
      gradient collapses to zero once that row is active while finite
      differences read −4.5. A learnable margin adds another such term in
      exactly that regime.
- [ ] **A constraint cannot rescue pathological weights** — bad weights still
      leave the track, the constraint merely makes the OCP harder. So this
      complements the external filter, it does not replace it, and the filter
      is still what acts when the solve does not. State this in the paper
      rather than claiming safety-by-construction.
- [ ] Off by default. Measure its effect on *learning* (does the policy explore
      more freely when it cannot crash?) before turning it on anywhere.

### Safety filters — two results from 2026-08-31, one of them bad

**The pointwise filters are fine, and their zero is the MPCC improving.**
`test_pointwise_filters_are_measurably_more_conservative` asserted a few
percent of overrides against the default weights and got 0.0%. Measured over
120 steps on the oval, toggling the grip-limited and terminal-speed
constraints:

    filter    grip   weights   intervene   off track
    cbf       on     good          0.0%    no
    cbf       on     bad          50.0%    no
    cbf       off    good         17.5%    no
    cbf       off    bad          80.8%    no

`clf_cbf` identical. The barrier is not permissive — it overrides half of a bad
controller's commands and keeps the car on the track. The zero is the MPCC's
own terminal-speed and grip constraints leaving nothing to correct. The old
thresholds were calibrated at SPEED_MAX 4 and are stale at *both* ends: at 8
with the constraints off, the default weights override 17.5%, past the 15%
ceiling the test used to assert. Test rewritten to check permissiveness against
weights that need correcting.

**The viability kernel is not converged in its own discretisation.** It grids
speed as `linspace(0, SPEED_MAX, n_v)` with `n_v=21`, so raising SPEED_MAX 4→8
halved the resolution without touching the filter, and it stopped saving a
controller that crashes unfiltered. "Coarser is worse" is **wrong** — 200 steps
on the oval, bad weights:

    n_v    dv (m/s)   intervene   off track
     21      0.400        8.1%    YES
     41      0.200       66.0%    no
     81      0.100       10.0%    YES

Non-monotonic, and dv=0.200 is exactly the spacing from before SPEED_MAX
changed. The only grid that works is the one it was implicitly validated on,
and a *finer* grid fails too — certifying more states safe (10% intervention)
and leaving the track. **The filter's guarantee is currently a property of
`n_v`, not of the dynamics.**

- [ ] **Check the kernel for convergence**: refine until the safe set stops
      moving, and report the resolution at which it does. Picking a better
      `n_v` is not a fix.
- [ ] Tie the grid to a resolution in m/s, not a point count that silently
      rescales whenever `SPEED_MAX` does.
- [ ] **Leading candidate for the undiagnosed `worst-case` row** (leaves the
      track 60% at grip 1.0, which should be impossible). The signature
      matches: plausible intervention rates while wrong about which states are
      recoverable. Not established — check it before believing it, and the row
      stays uncited either way.
- [ ] Both filter suites' stored benchmarks predate SPEED_MAX 4→8, the grip
      constraints, and the 7th/8th weights. Regenerate before citing any of
      them.

### Overleaf

`https://git.overleaf.com/6a91eb483816017e248781d9` is a **full mirror of this
repo**, not just `paper/`.

**Its history is UNRELATED to this repo's** — corrected 2026-08-31. An earlier
push came from a scratch clone, so Overleaf's `main` shares no merge base with
our branch: `git diff HEAD...FETCH_HEAD` fails with "no merge base", the same
commit *messages* appear there under different SHAs, and an ordinary push is
rejected. The previous note here said "push is a clean fast-forward"; it is not.

Do **not** force-push to fix that. Overleaf held one file that existed nowhere
else — `paper/figures/architecture.png`, generated into `docs/` and pushed from
the scratch clone but never committed here, so the paper referenced a figure
this repo did not contain. A force push would have deleted it silently.

The working procedure, used for `54d08a5`:

1. `git fetch <overleaf-url> main`, then `comm -13` the two `ls-tree` listings
   to find anything that exists only on Overleaf, and commit it here first.
2. Diff `paper/` both ways and read the Overleaf-only lines — they are usually
   just the older text, but that is a check, not an assumption.
3. On a side branch, `git merge -s ours --allow-unrelated-histories FETCH_HEAD`.
   This keeps our tree byte-for-byte while recording their ancestry, so the
   push becomes a fast-forward and GitHub's history stays clean.
4. Push `HEAD:main` (not `master` — Overleaf rejects new branches).
5. **Verify by fetching back** and diffing, not by trusting the push output.
6. Pass the token inline in the URL so it is never written to `.git/config`;
   confirm with `grep -c olp_ .git/config`.

- [x] Pushed and verified 2026-08-31: Overleaf tree identical to
      `situation-dependent-weights`.
- [ ] **Rotate the Overleaf token.** It has now been pasted in plaintext into
      chat transcripts twice and used again on 2026-08-31.

---

## 1. Situation-dependent weights — best result, needs confirming

`experiments/per_segment_weights.py`. One θ per curvature segment, segment read
from the path *ahead*, so no detector is needed.

- [x] **Reproduce over ≥5 seeds.** Done, six seeds,
      `experiments/per_segment_seeds.py`. Global 8.9 m (sd 0.9), 6/6 seeds
      collapsed, first off-track at episode 3–5 every time; per-segment 77.8 m
      (sd 0.2), 0/6 collapsed, never off-track. Seed 0 returns the published
      7.6 / 78.0 pair exactly.
      **The mechanism is not a richer parameterisation.** Global reaches a
      77.6–79.1 m controller on *every* seed and destroys it on *every* seed;
      per-segment reaches the same 78.0–78.3 m peak and keeps it. What changes
      is that a bad update in a corner no longer moves the straight's weights.
- [ ] Continuous schedule θ(κ) instead of three bins — a linear map from a
      curvature-preview feature vector, with the envelope gradient chain-ruled
      through it. **Item 2's overtake grid says the map should be linear in θ**:
      the behaviour boundary there is a *ratio* of weights, i.e. a difference of
      log-weight components.
- [ ] Check it composes with the safety filter and with item 4.
- [x] The learned weights are counter-intuitive (progress weighted *higher* in
      the tightest segment). **Explained, and it was an artefact.** `q_v` lands
      between 19.7 and 71.2 across all seeds and segments — entirely inside the
      dead zone above ≈2 — so those are all the same behaviour and the ordering
      is noise in a flat region. The 71.2 figure is a seed-0 outlier; the other
      five are 24.7–30.8. The weight that carries the schedule is **`q_c`**:
      0.42 straight against 1.95 / 1.22 in the curves, straight lowest on 6/6
      seeds. Hold the line in the corners, run wide on the straight — which is
      the intuitive schedule, and was invisible while `q_v` was being read.

## 1b. Named sectors — **taxonomy done, tuning not started**

`Track.corners()`, `Track.sector()`, `Track.circuit()`, `tests/test_sectors.py`.

**Pointwise curvature cannot express the sector names, and no retuning of
`segment`'s quantiles will fix it.** `kappa = 1/R` for an arc of radius `R`
*however far it sweeps*, so a 90-degree corner and a 180-degree hairpin of the
same radius are identical to anything reading curvature at a point. Measured on
a circuit carrying both at R = 2.6 m: same peak |κ| to within 2%, same
`segment()` bin, different `sector()`.

What separates them is the **integral** of curvature through the corner — the
total heading change — so a corner has to be detected as an *extended object*
first and classified second. That is what `corners()` does, and the label is a
property of the corner, so it cannot flicker part-way through one.

- [x] Four classes: straight, long curve, 90-degree, 180-degree.
- [x] `Track.circuit()` — a 47.2 m lap containing all four, min radius 2.59 m
      (above the oval's 2.46, so a *scheduling* result is not confounded by the
      initialisation failure that kills `mixed`).
- [x] **Per-sector tuning on the circuit, six seeds.** Done,
      `experiments/per_sector_weights.py`, `benchmarks/results/per_sector.json`.
- [x] Check a four-way schedule against the three-bin curvature one.
      **It does not beat it** — −0.07 m at 0.8 SE, not separated. Four labels
      cost twice the parameters of three and return nothing measurable.

      | | θ | mean | sd | collapsed |
      |---|---|---|---|---|
      | global | 1 | **79.84 m** | 0.07 | 0/6 |
      | curvature3 | 3 | 78.99 m | 0.19 | 0/6 |
      | sector4 | 4 | 78.92 m | 0.07 | 0/6 |

      global − curvature3 = +0.85 m (10.4 SE), global − sector4 = +0.92 m
      (22.5 SE). **The unscheduled controller wins on this track**, and
      performance decreases monotonically in parameter count.

### Five silent failures building the circuit — all produced a plausible track

Recorded because each one looked fine and none announced itself, and because the
third is a general lesson.

1. **Turns summing to 480 degrees, not 360.** A closed lap's heading returns to
   where it started. The constructor bridged the gap with a chord and the chord
   read as two corners that were never designed.
2. **Corner threshold at 25% of peak |κ|.** One tight hairpin sets the peak and
   every gentler corner falls below the threshold; the detected ones have their
   entry ramps clipped, so a designed −90 measured as −59. 0.10 works.
3. **Unsigned radius in the closure formula.** The arc displacement is written
   in terms of `R = 1/kappa`, which carries the turn's sign; the unsigned `r`
   mirrors every right-hand corner. **The solver reported a residual of 2e-15
   while the real geometry missed by 2.17 m** — a solve reporting success on the
   wrong model, which is the same class of error as the `max_iter=1` trap in
   paper 1.
4. **A balanced turn set sums to zero** (+180,−180,+90,−90,+60,−60) and closes
   as a self-intersecting figure-eight. A valid closed curve and a useless
   racetrack; the two hairpins have to turn the same way.
5. **Same-sign corners separated by < 1.2 m of straight merge into one**,
   because `Track.curvature`'s 0.6 m stencil never lets κ fall back to zero. A
   180 and a 60 were reported as a single +239.5-degree corner. Opposite-sign
   neighbours separate for free.

Hand-picking a layout failed twice. The final one came from searching all 720
orderings against closure, positive straights, same-sign separation and
self-intersection *simultaneously*.

### The result that reframes item 1

**The collapse is a property of the track, not of the parameterisation.** Global
θ collapses 6/6 on the oval and **0/6 on the circuit**, where it reaches 79.84 m
with sd 0.07. So item 1 shows per-segment weights fix *the oval's* failure — not
that they are a better parameterisation, and it must not be read as that.

Mechanism the two tracks suggest: the oval is mostly straight, so the tuner is
rewarded for raising `q_v` over most of the lap and then meets a 180-degree
corner carrying straight-tuned weights. The circuit is 82% non-straight and that
feedback arrives constantly. **The variable is how much of the lap rewards the
wrong thing**, not how many θ are held.

The defensible claim, narrower and more useful than "scheduling helps":

> Situation-dependent weights pay when part of the lap rewards the wrong thing,
> and cost a little when it does not.

- [ ] **The caveat is not small and is not yet addressed.** All three arms sit
      within 1 m of each other at the 4.0 m/s speed cap, with the unmodelled
      grip limit active in the corners. This is a **ceiling-limited task**, so
      "scheduling does not help here" is measured and "scheduling does not help"
      is not. Build a task where scheduling has something to win — conflicting
      demands between sectors, or a speed range the cap does not truncate — and
      re-run. Until then the null result does not generalise.
- [ ] Two tracks is not many, and the conclusion **reversed** between them.
      Treat any third track as capable of reversing it again.

## 2z. The policy degenerates to a constant — the blocking result

`experiments/feature_sensitivity.py`. Train the policy, freeze it, sweep one
feature group, record the emitted θ.

    feature group        spread of q_v/q_c   relative
    named sector                    0.46        1.9%
    opponent class                  0.34        1.4%
    curvature preview               0.21        0.9%
    gap                             0.40        1.7%
    corridor width                  0.05        0.2%

**The trained policy emits q_v/q_c ≈ 19.8 whatever it is shown.** All eighteen
features are decorative. Every LTC gate number reported (31.8, 29.3, 24.0,
32.5 m) was a comparison between two constants and is void.

**Three output parameterisations, same endpoint**: hard clip, tanh centred on
the box, tanh anchored at the reference with asymmetric span. The *untrained*
policy is responsive (ratio 0.70–0.95); training destroys it. TD(λ) drives θ
monotonically, θ hits whatever bound exists, the squash saturates, the output
stops responding. Each bound moved *where* it saturates, never *whether*.

This is item 5's failure — "optimised a proxy past the point where the proxy
was valid, with nothing to stop it" — inherited whole.

- [ ] **A stopping criterion is a prerequisite for item 2, not an improvement
      to it.** No output bound can supply one; the deficiency is that nothing
      makes stopping preferable to continuing. Candidates, none tried: a
      terminal value that stops the critic under-estimating at saturation; a
      decaying step size; keep-best-and-revert.
- [ ] **Do not run another policy comparison until it exists.** Four gate runs
      have now measured a constant.
- [ ] The lesson for the method, and it is cheap: a unit test that a feature is
      *present* passes happily while the policy ignores it. Sensitivity of a
      *trained* policy is the only test that separates them.

### The benchmark cannot reward adaptation — 2026-09-01, and this outranks the rest

`experiments/situation_demands.py`. Instead of asking what the learner emits,
drive a grid of fixed weight vectors in each cell of (sector × opponent ×
corridor) and keep the winner. 144 runs, 400 steps, circuit, no crashes:

    sector      opponent                    best q_v/q_c   covered
    straight    none/slower/equal/faster           16.67   79.0-79.9m
    long curve  none/slower/equal/faster           16.67   74.4-78.9m
    90-deg      none/slower/equal/faster           16.67   74.6-79.0m
    180-deg     none/slower/equal/faster           16.67   76.9-78.9m

    best single constant:       q_v/q_c = 16.67, 77.9 m mean
    best weight PER SITUATION:                    77.9 m mean
    headroom for an adaptive policy:              +0.0%

**Every cell wants the same weights, and they are the largest ratio in the
grid.** Per-situation tuning wins exactly nothing.

Two reasons, both properties of the experiment rather than of the learner:

1. **The metric is pure progress.** `q_v` weights progress and the score IS
   progress, so more `q_v` is always better and the search runs to the top of
   the grid everywhere. Nothing is traded against anything. This is the same
   degenerate optimum as the "runaway" and the "collapse to a constant" — the
   policy was not broken, it was answering correctly a question with one answer.
2. **The opponent never blocks.** Distance is 74-80 m whether the opponent is
   absent, slower, equal or faster. A faster one drives away; a slower one is
   passed on a corridor wide enough that passing is free. That is why
   `opponent class` reads decorative in every sensitivity table we have run.

- [ ] **Do not run another policy experiment until the benchmark can
      discriminate.** Every negative result about the learner is confounded by
      this. Five mechanisms were investigated (output parameterisation, dead
      span, entropy, meta-RL feedback, gauge freedom + readout decay) against a
      task where a constant is provably optimal.
- [ ] **A racing metric.** Price position and contact, not only metres, or
      "attack always" is optimal by construction. Lap time under a
      contact/off-track penalty, or position against a defending opponent.
- [ ] **An opponent that defends** — matched speed, holding the racing line, so
      a pass needs commitment and carries risk.
- [ ] **Narrow corridors where a pass does not fit.** ICRA T1 vs T2 is the
      instrument: the SAME geometry at two widths (curvature cross-correlation
      0.874; median half-width 0.72 m vs 0.66 m, ranges 0.35-1.56 and
      0.30-1.88).
- [ ] Re-run `situation_demands.py` after each change. It is the cheapest
      possible check on whether an experiment can show what we want to claim,
      and it should have existed before any of the learner work.

### Resolved mechanism, unresolved result — 2026-09-01

Two defects were masking each other, and neither alone explains anything.

1. **The critic has a gauge freedom.** `V = -J*` is linear in the six cost
   weights, so scaling all six by `c` scales `V` by exactly `c` while the plan
   changes by micrometres (`V/c` constant to four decimals). TD(λ) climbs it
   until θ hits a bound.
2. **The readout is decayed to zero.** `theta_prior=0.5` applies
   `G *= (1 - alpha*prior)` every step, which exists to contain (1) and does so
   by parking the policy at θ₀.

Remove one → runaway. Remove the other → constant. Both look identical in the
sensitivity table, which is why three output parameterisations, the dead-span
repair, anti-saturation, meta-RL feedback and the gauge fix alone all failed
the same way — every one ran with `prior=0.5`.

**Fixing both restores responsiveness and costs half the distance:**

    condition                 sector spread   covered
    default (prior 0.5)        1.9% -> decor.  46.3 +-1.3
    gauge fixed, no decay     20.5% -> USED    24.3 +-4.6
    gauge fixed, decay .05    18.6% -> USED    28.7 +-8.5
    no gauge, no decay        14.2% (runaway)  24.9 +-9.9

4.6 SE apart on distance. Four of five feature groups become USED, and the car
goes half as far.

**The finding that matters more.** The default's collapsed constant covers the
most ground of anything tested. `q_v` weights progress and the metric *is*
progress, so the runaway is aligned with the objective: a constant is a correct
answer to "go far on a known track". Part of what we have been calling a
degenerate policy is a correct policy for a task that does not require
situation-dependence.

- [ ] **Do not "fix" this by trading distance for spread.** That is the trade
      the entropy term lost, made twice in one day.
- [ ] **Build a task a constant cannot win** before any further work on the
      learner: several opponents demanding different behaviour per sector, or
      varying grip. Then ask whether the responsive configuration beats the
      constant. Until such a task exists, sector spread is not evidence of
      anything and neither is its absence.
- [ ] Gauge fixing restricts the reachable *ratios*: pinned mean, bounded box,
      so `q_v/q_c` tops out near 6 against 21 for the default. Uniform scaling
      cannot change behaviour, but the pin plus the box can. Re-centre or widen
      the box so the same ratios stay reachable with the gauge fixed.
- [ ] Everything above is n=3.

### A structural cause, found 2026-08-31 — supersedes the hypotheses above

θ0 is the offline-tuned weight vector, and the box was drawn with two of its
entries **exactly on the ceiling**: q_l at 200 and q_v at 2.0. The anchored
asymmetric squash — the third parameterisation listed above — computes

    span = where(t >= 0, hi - θ0, θ0 - lo);   dθ/dz = (1 - tanh²z) · span

so with `hi == θ0` the span is **zero** and the policy gradient through those
two weights is identically zero whenever the pre-activation is non-negative.
Half the time, structurally, q_l and q_v had no learning signal and could only
ever be revised *downward*.

q_v carries behaviour (the ratio q_v/q_c) and safety (its magnitude), so the
most consequential weight in the policy was the half-dead one. This explains
the signature that all four hypotheses shared and none accounted for — the
output always pinned at *a bound* (0.006, 0.100, 19.8, 39.97).

Fixed in 05b3e9d: ceilings raised so every anchor is strictly interior, and
`WeightPolicy` now refuses to build with an anchor on a bound.

- [ ] **Re-measure `feature_sensitivity.py` against the fixed box.** The table
      at the top of this section, the four gate runs, and the entropy result
      (0.006 → 1.733, spread 15.6%, distance halved 4.9 → 2.3) were all
      measured on a policy whose behaviour weight could move one way only.
      Provisional until re-run, in both directions — the collapse may simply
      have been this.
- [ ] Only after that: whether meta-RL feedback (`experiments/meta_rl.py`,
      previous θ + reward + TD error as inputs, adapted from RTRRL) adds
      anything the fixed box does not already supply. Running the comparison
      before the box was fixed would have credited the fix to the feedback.

## 2. Weights as a behaviour policy — the direction

θ is a behaviour parameterisation, not just a tuning vector: `q_v` sets
aggression, `q_c` sets how strictly the line is followed. A policy mapping
*situation → θ* subsumes item 1, with curvature as one feature among opponent
gap, closing rate, and distance to the boundaries. It should be able to express
**overtake vs stay behind**, and **use every corner vs follow the global
racing line**.

### 2a. Precondition: the MPCC cannot see opponents — **done**

- [x] Circular keep-out per opponent, soft, with explicit slacks. Formulation
      copied from `MPCC_planner_acados/scripts/generate_acados_solver.py`,
      including both of its quirks: `r_eff = r_raw + obs_margin` with inactive
      slots passed as `r_raw = -obs_margin` (so `r_eff` is *exactly* zero and no
      `max()` enters the NLP), and the slack in units of **squared** distance.
      acados' `idxsh`/`Zl`/`zl` become explicit slack variables at the end of
      `w`, so `_nx` and the `u0` slice keep their meaning. `max_obstacles=0` is
      the default and is bit-for-bit the old problem. See
      `docs/source/obstacles.md`.
- [x] Opponents in the plant, moving at their own speed —
      `mpcc_tuning/opponents.py`, `Plant(opponents=...)`. Collision is terminal
      at the same −5 as leaving the track, with the cause in `plant.failure`.
- [x] **The envelope gradient survives the slacks**, checked rather than
      argued: cosine > 0.999 to finite differences with a keep-out active,
      relative error < 5e-2 — the same thresholds as the obstacle-free check.
      `tests/test_obstacles.py`, 11 tests.
- [x] Fixed on the way: `rti.py` packed the parameter vector itself and would
      have silently mispacked it once obstacles existed.

### 2b. Establish the safe weight region — before any learning

`experiments/weights_as_behaviour.py` measured: `q_v` is a genuine dial (mean
speed 0.40 → 3.92 m/s, monotone, **saturating above ≈2**), and `q_c` is **not**
— ×10 barely changes the line, ×100 **drives off the track**.

So the usual safety argument for learning weights rather than steering — "the
MPCC enforces the constraints, so any θ is feasible" — is **false as stated**.

**`experiments/overtake_or_follow.py` now says the region is not a box.** A 3×5
grid over (`q_v`, `q_c`) against an opponent 3 m ahead at 1.0 m/s, 15 runs:

* **behaviour is the ratio.** Every cell with `q_v/q_c ≤ 1` follows; every cell
  with `q_v/q_c > 1` passes. 15 of 15, over two decades in each weight. The
  follow is a settled behaviour, not indecision — 0.73 m back, 7 mm of lateral
  wander, held for the whole run.
* **safety is `q_v` alone,** and it is a *different* axis. The ratio does not
  predict survival: ratio 5.0 completes the lap, ratio 3.33 leaves the track.
  Sorted by `q_v`: every pass at `q_v = 10` goes off, one of two at `q_v = 2`,
  none at `q_v = 0.5`.
* **the keep-out never failed** — zero collisions, minimum clearance +0.123 m.
  All five failures are `off_track`: running wide *while* passing.

- [ ] Bound the policy's output. **Not a box in θ, and not a bound on the
      ratio** — a ceiling on `q_v` with the ratio left free. Establish the
      ceiling properly; the grid only brackets it between 2 and 10.
- [ ] Re-run the grid over opponent geometries (gap, closing speed, offset). One
      geometry, deterministic runs — each cell is exact, but the *geometry* is
      n=1.
- [ ] **The `q_v` dead zone is not benign.** On an empty track everything above
      ≈2 was the same behaviour. With an opponent, above ≈2 the extra progress
      weight stops buying speed and starts costing track — at `q_v = 10` every
      attempted pass ends off-track, at every `q_c` tried. The tuner drives
      `q_v` to 30+, i.e. straight into that region, and item 1's per-segment
      run leaves *every* segment there (19.7–71.2).

### 2c. Physics-informed features, not raw positions

A behaviour policy keyed on coordinates will not transfer between tracks or
speeds. Key it on quantities with units that mean something:

- [ ] time-to-collision with the opponent ahead;
- [ ] gap measured against **braking distance** $v^2/2a_{\max}$, not metres;
- [ ] lateral acceleration required to take the gap, against
      $a_{\text{lat,max}}\mu$ — i.e. *is this pass physically available*;
- [ ] curvature preview at several distances;
- [ ] margin to the boundary in units of the tube width from the filter.

### 2d. Only then a network — and this is where spiking earns its place

Two things are true here that were **not** true of event-triggered weight
updates (where deferring actively hurt):

- a behaviour decision is made **once**, not 20 times a second, and re-deciding
  every tick invites chattering between "overtake" and "follow";
- a **closing gap is a temporal pattern**. A threshold on instantaneous gap
  cannot tell "catching them" from "being caught", so there is finally
  something a recurrent or spiking net can represent that a threshold cannot.

**An LTC head was considered and the evidence is against it** — written up in
`docs/source/behaviour_policy.md`. `rtrrl-playground`'s `overtake` task is this
problem in a different action space, and it has already been run: `ltc` came
**6th of 7**, made the **fewest passes of any cell** (1.29) and scored below the
memoryless MLP (267.5 vs 374.2). Nothing on that page is separated (sd 137–296),
so LTC is not *shown* worse — but it is not shown better, which for a component
one is adding is the same answer. `liquid_gru`'s own docstring: *"this cell does
not win."*

- [ ] Event-triggered behaviour selection: re-decide θ on events, hold between.
- [ ] Compare against (i) a fixed schedule, (ii) a per-tick MLP. **Only claim
      the network if it beats both** — otherwise it is decoration. Prediction
      now attached: **the fixed schedule should be strong**, because the
      overtake grid's behaviour boundary is `q_v/q_c ≷ 1`, i.e. *linear* in θ.
      The network can only win on the temporal axis.
- [ ] **Decide explicitly whether the policy sees the opponent's velocity.**
      Today the MPCC gets the true state and opponents are exact `(x,y,r)`. Hand
      the policy the velocity too and there is no hidden state, no temporal
      pattern, and no role for recurrence at all — the gate above would
      correctly kill any net. The temporal argument *requires* withholding it,
      which is a design choice to defend, not to inherit by accident.
- [ ] If built: recurrence confined to the **closing rate** (a derivative of a
      range, genuinely unavailable from one frame), feeding the measured linear
      map, output clamped by 2b's `q_v` ceiling. Not a network choosing the
      behaviour. Copy `ltc.py` in (117 self-contained NumPy lines with analytic
      derivatives), do not import.
- [ ] Watch for the failure the caution predicts: LTC is the cautious end of
      that table, and here *following* is a stable low-progress attractor
      (12.0 m against a pass's 36.8 m). A head that never passes still scores
      respectably, because progress alone pays.

### 2e. Behaviour personas — **measured on a real circuit**

`mpcc_tuning/ltc.py` (`BEHAVIOURS`, `AGGRESSION`, `POSTURES`),
`experiments/behaviour_modes.py`, `Track.spielberg()`.

- [x] Named behaviours over the weights: follow / overtake, crossed with
      cautious / neutral / aggressive, and three postures — stay behind,
      overtake when safe, always try.
- [x] Measured on the Red Bull Ring at F1TENTH scaling (public data, copied in
      with provenance). **34 m following against 74.6 m overtaking, 2.2×, 1.00
      passes, zero crashes in 27 runs.** The behaviours are expressible.
- [x] Use a real track, because the synthetic `circuit` is ceiling-limited —
      its tightest corner admits 3.9 m/s against a 4.0 cap, so every weight
      setting lands within 1 m. Spielberg's admits 2.5 m/s and discriminates.
- [ ] **The postures do not separate, and that is the experiment's fault.**
      `overtake_when_safe` and `always_try` differ in switch count but are
      identical in distance and passes. One slower car on a wide circuit makes
      a pass almost always available. **So we have shown a weight policy
      expresses overtaking, NOT that it expresses *safe* overtaking** — which
      is the claim the safety argument actually needs. Build a case where the
      pass is genuinely unavailable: a narrow section, a blind corner, a faster
      opponent, or two cars abreast.
- [ ] `cautious` never completes a pass in any posture. Arguably correct for a
      cautious driver; check it is not a speed floor artefact.

### Three defects found by measurement, not review

Worth keeping because the first two were fixed first and the table did not move.

1. **Aggression crossed the behaviour boundary** — `q_c/sqrt(g)` with `q_v*g`
   put "cautious overtake" at ratio 0.71, below 1, so it followed. Crossing the
   boundary is a change of *behaviour*, not of intensity.
2. **The ceiling and the dial collided** — with `q_v` clipped at the measured
   ceiling, aggressive and neutral overtaking were the same weights.
3. **The engagement test was denominated in braking distance**, which vanishes
   at low speed: at 1.4 m/s it demanded a 0.54 m gap while the keep-out holds
   the car at 1.04 m, so it asked for a proximity the safety constraint
   forbids. It fired on 0 of 400 ticks. **This was the only one that mattered**,
   and it made the LTC/MLP arms learn overtaking from a feature that never
   changed — so `benchmarks/results/ltc.json` had to be re-run.

   Same shape as the filter lesson in `docs/source/filters.md`: a wrong
   safety-relevant component *intervenes less*, so it reads as working.

### The old 2e note — two axes, one of which collapses

Wanted: *overtaking*, *following*, *safe driver*, *aggressive driver*. The
overtake grid already gives the axes, and they are orthogonal:

* **`q_v/q_c`** selects overtake vs follow (15/15 cells, boundary at 1);
* **`q_v` magnitude** selects aggressive vs safe, and decides survival.

So the 2x2 is the right shape, but **one cell is not reachable**: while
following, aggression is invisible — 12.0, 12.0, 12.1 m covered at `q_v` = 0.5,
2 and 10, because the car is pace-limited by the opponent ahead and the
aggression weight has nothing to act on. There are **three** behaviours, not
four:

| | follow | overtake |
|---|---|---|
| safe (`q_v` 0.5) | 12.0 m | 22.3 m, pass at step 99 |
| aggressive (`q_v` 2.0) | 12.0 m *(same)* | 36.6 m, pass at step 33 |
| (`q_v` 10) | 12.1 m *(same)* | **all crash** |

- [ ] Re-measure over several opponent geometries (gap, closing speed, offset).
      The runs are deterministic, so each cell is exact — but the *geometry* is
      n=1, and the personas should not be defined off one.
- [ ] Fix the three reachable personas as named weight sets once the `q_v`
      ceiling from 2b is known, and check each does what its name says.
- [ ] Cross them with the sectors from 1b: a persona is a *global* choice, a
      sector schedule is a *local* one, and the natural policy is the product.
      Whether the product needs more than one θ per (persona, sector) pair is an
      open question and probably the answer is no.

## 3. acados backend for real time

`mpcc_tuning/rti.py` already meets 20 Hz with CasADi's `qrqp` (1.9 ms mean,
3.4 ms worst at N=12). acados would be faster and is what the car will run.

**Feasible: acados is built and usable** at `~/acados` — `libacados.so`,
`blasfeo`, `hpipm`, `qpOASES_e`, and `bin/t_renderer` for codegen.
`acados_template` imports. Nothing needs building first.

**But it is not a copy-paste, and two problems are structural.** Both were found
by reading `MPCC_planner_acados/scripts/generate_acados_solver.py` properly and
both change the formulation rather than the plumbing:

* **The path.** This repo's OCP contains the centreline as a CasADi B-spline of
  the progress variable `s`, evaluated *inside* the NLP — that is what lets the
  solver choose its own reference point, which is the whole of MPCC. The
  template instead passes `ref_x`, `ref_y`, `t_angle` as **stage parameters**,
  sampled outside the solver. That breaks `s`'s differentiable coupling to the
  path within a solve, and that coupling is the mechanism the envelope gradient
  runs through. Porting naively would silently change what `dJ*/dtheta` means.
* **The progress reward is not a least-squares residual.** `-q_v v_s dt` is
  *linear*. The template encodes it as `-sqrt(w_progress) * p_prog` inside a
  `NONLINEAR_LS` cost, and with `yref = 0` and `W = I` that squares to
  `+w_progress * p_prog**2` — a quadratic **penalty** on progress, not a linear
  reward. (It only behaves as intended if `yref` is set to a large negative
  value at runtime, which the generator does not do.) The template's own comment
  claims otherwise. **Do not copy this one; fix it.**

- [ ] Vendor an acados OCP builder into `mpcc_tuning/acados_ocp.py`. Template:
      `MPCC_planner_acados/scripts/generate_acados_solver.py` — **copy and
      adapt, do not import**. Keep weights as **runtime stage parameters**
      (it already does) so the envelope gradient still applies.
- [ ] Use `cost_type = "EXTERNAL"` rather than `NONLINEAR_LS`, so the exact
      objective — linear progress term included — survives, and with it the
      exact envelope gradient. Consequence to plan for: `EXTERNAL` rules out
      `GAUSS_NEWTON`, so it needs an exact or custom Hessian, and the template's
      solver options assume Gauss-Newton.
- [ ] Decide what to do about the spline: either code-generate the interpolant
      into the acados model, or accept the stage-parameter reference and
      **measure** how much `dJ*/dtheta` moves. The second is cheaper and is a
      result either way.
- [ ] Port the obstacle keep-out *back*: `mpcc_tuning/mpcc.py` now has it as
      explicit slacks, which in acados is `idxsh`/`Zl`/`zl` and is where the
      formulation came from. That direction is a straight swap.
- [ ] Note the two-place `ctrl_mode` gotcha: the config override lands in the
      packed vehicle array and the `ModelSpec` copy is what the kernels
      dispatch on; `validate_against` checks they agree.
- [ ] Confirm the direction-agreement result (cosine 1.0000) holds for a real
      acados `SQP_RTI` step — the current measurement used a hand-rolled SQP.
      **This is the open caveat on paper 1's headline claim**, and it is the
      first question a reviewer asks. Paper 1 is careful to say the scheme is
      "as in acados's SQP_RTI" rather than that acados was used, which is
      honest, but the check is still owed.
- [ ] Nonsmooth penalties must become soft constraints: `max(0, |v_y| - v_soft)²`
      is not an NLS residual. Template for the slack form:
      `MPCC_planner_acados` (`idxsh`, `Zl`, `zl`).

## 4. Make the tuner survive the fitted-tyre plant

`--plant scuderia` runs, and the tuner does not. The feasible region exists —
**every open-loop run with `r_d=10` completes 400 steps, every run with
`r_d=1.0` crashes** — and the tuner overshoots it into "do not drive".

This is **reward design**, not research: progress minus a single terminal −5
makes standing still a local optimum worth ≈0 against a crash worth −5.

- [ ] Per-step survival bonus, or initialise inside the feasible region.
- [ ] Then re-run and report. **Without this the paper cannot claim online
      tuning works on a realistic vehicle.**
- [ ] Consider vendoring the dynamic bicycle + simplified Pacejka model from
      `MPCC_controller_ipopt` as a *controller* model, so the controller is not
      permanently a kinematic bicycle. Copy it in.

## 4b. Multiple cars on the fitted-tyre plant — yes, and one blocker has cleared

`scuderia_gym_jax` has the multi-agent side already: `envs/multi_agent_env.py`
(`num_agents` cars sharing one `State`, one `step` integrating all of them),
`envs/collision_models.py` (SAT on the cars' actual **rectangles**, with a
GJK parity test), and `examples/overtake.py` (ego plus traffic, `--cars`,
`--spacing`, and dumb traffic by the same deliberate choice made here). It is
the one permitted optional dependency under the standing rule, because it is a
*plant*. Our bridge currently builds it with `num_agents=1` and
`collision_on=False`.

**`docs/source/plant.md` listed two blockers and one is now gone.** It said
overtaking there "would need the obstacle constraint the MPCC does not
currently have" — item 2a built it. The centreline blocker is also partly
cleared by `tools/centerline_from_map.py` (item 5b).

### What it would add that the current setup cannot

- [ ] **Collision geometry that is not a circle.** Our keep-out is
      `dist2 - r_eff**2 >= 0`, and a car is a rectangle. The two disagree most
      exactly when passing — side by side, where a circular keep-out is
      conservative along the flanks and optimistic at the corners. Every
      "closest approach +0.056 m" number here is measured against a circle, and
      that margin is the one a rectangle would eat.
- [ ] **Opponents with real dynamics.** `mpcc_tuning/opponents.py` drives the
      centreline at a constant speed and cannot be pushed off line, cannot
      brake, cannot lose grip. On the multi-agent plant the opponents are the
      same ST/STD vehicle model as the ego, so a pass changes *their* state too.
- [ ] **A contact that means something.** There, a car that is hit freezes, so
      a failed pass is unmistakable rather than a silent interpenetration.

### The blocker that has not cleared, and it is decisive

- [ ] **Item 4 first.** The tuner does not survive that plant at all with a
      single car: 120 episodes, 100% off-track, best −2.20 m, and the default
      weights leave the track in 12–22 steps. Adding opponents to a plant where
      the controller cannot complete a lap would measure nothing —
      "overtaking fails" would be indistinguishable from "driving fails". It is
      a reward-design problem (progress minus a terminal −5 makes standing
      still a local optimum) and it is the prerequisite.

So: **worth doing, and not yet.** The order is item 4, then this, and doing
them in the other order produces a number that cannot be interpreted.

- [ ] When it happens, the honest comparison is our circular keep-out against
      the SAT rectangle on the *same* manoeuvre, because that difference is a
      result in itself and is cheap to measure once both exist.

## 5. Bound where θ goes

Most of the damage happens in episode 0: `q_v` moves ×222 before a single
episode boundary. α = 2e-4 and 2e-3 give the same shape ten times apart, which
is a divergence signature rather than a tuning artefact.

- [ ] A trust region on θ, and compare against item 1 and against discarding
      updates (38.5 m, partial). **Item 1 appears to fix this failure already:**
      over six seeds the global run goes off-track first at episode 3–5 on 6/6
      seeds and the per-segment run never does, on 6/6. The drift-during-good-
      driving that a trust region targets does not occur under per-segment
      weights on any seed. Two caveats before calling it redundant — per-segment
      θ *isolates* damage rather than *bounding* it, so it does nothing where
      there is only one segment; and the mixed-track and `scuderia` failures are
      initialisation failures, which neither mechanism addresses.
- [ ] Trigger on the **safety filter's intervention** rather than lateral
      error: behind the filter `q_c` falls to 0.136 with performance drifting
      down, because the filter absorbs the consequence and it never reaches the
      return. The intervention *is* the missing error signal.

## 5b. A real competition track — 80% done, and the last 20% is a decision

`tools/centerline_from_map.py`, and the ICRA 2026 maps
(`ICRA_T1_..._gimped.pgm/.yaml`, `ICRA_T2_..._gimped_ev13.pgm/.yaml`, 0.05 m/px).
This is the centreline extraction `docs/source/plant.md` has listed as missing
since the `scuderia_gym_jax` bridge was written.

    map   lap      half-width (median)   fraction inside the corridor
    T1    53.0 m   0.95 m                81.8%
    T2    41.3 m   0.45 m                78.7%

Plausible laps and widths, most of each contour tracking the corridor — and the
remaining fifth cutting straight across the infield. **Do not use the output as
a track until that is fixed.**

- [ ] **Both tracks branch**: an outer ring plus an inner section. Free space is
      not an annulus, so there is no unique centreline — "the" centreline is a
      choice of *route*, not a property of the geometry. Every purely geometric
      method fails here for that reason. Fix it by choosing the route, not by
      more geometry.
- [ ] Route from a driven path is the cheap fix: project a racing rosbag's
      trajectory onto the corridor and take the branch it uses. **The bag we
      have does not supply one** — `ICRA26_..._TRACK_2_0955` is a 96 m SLAM
      mapping run at 0.29 m/s whose start and end are 39 m apart. A bag of
      actual laps would settle it immediately.
- [ ] Or prune the branch explicitly: keep the cycle enclosing the largest
      infield, discard spurs. That is a graph decision on the medial axis.
- [ ] `Track` carries a **constant** half-width; T2 varies 0.45 m median against
      a 1.79 m maximum. Either take the minimum (safe, wastes track) or teach
      `Track` a per-station width.

Three methods were tried and the failures are recorded in the tool's docstring:
skeletonization loops around every cone; marching outward from the infield
outline fails because the infield is strongly concave (compactness 0.11 on T1);
the equidistance contour is the closest and is what the tool implements.

## 6. Filters on a state estimate

Every filter reads the true state. On the car it reads an estimate, and every
guarantee is conditional on that.

- [ ] Run the filters on odometry rather than ground truth.
- [ ] Add the estimator covariance to the margin, or state the guarantee as
      conditional.

## 6b. ROS export, for the real RC car

The point of all of this is a car. Nothing here has ever run outside Python, and
the export is not a packaging job -- three of the assumptions the results rest
on are false on hardware, and each is a separate piece of work.

**Templates to copy from, per the standing rule --- copy in, do not depend:**
`MPCC_controller_cpp` (6 `package.xml`, the closest thing to a working
controller node), `race_stack` (30 packages; the full stack layout),
`ekf_state_estimator_node` (state estimation node + config),
`datmo` (detection and tracking of moving objects -- this is what feeds the
keep-out), `f1tenth_gym_ros` (the sim-side bridge and its Docker setup).

### What actually has to be exported

- [ ] **The solver, as generated C.** This is item 3, not a separate task:
      acados' code export *is* the ROS artefact, and `mpcc_tuning/rti.py`'s
      CasADi `qrqp` path is not something to ship. Do item 3 first.
- [ ] **theta as a runtime parameter, not a rebuild.** Already true in
      `mpcc_tuning/mpcc.py` and in the acados template; keep it true, because
      the whole point is that the weights change while the car drives.
- [ ] **Decide where the tuner runs.** On-car in the control loop, or off-car
      updating theta between laps. The envelope gradient is cheap enough for
      on-car, but the *exploration* solve is a second NLP on the ticks where it
      fires (see `docs/source/results.md`), and that has to fit the budget in
      the worst case, not the mean.
- [ ] **Topics and message types**, once the above is settled: odometry in,
      `AckermannDriveStamped` out, keep-outs in from `datmo`, theta in/out plus
      the TD error and intervention rate as diagnostics -- the tuner is
      unobservable without them and this repo's whole history says the
      diagnostics are what catch the bugs.

### The three assumptions that break on hardware

1. **Every filter reads the true state** -- item 6, immediately below. On the
   car it reads an estimate, and every guarantee is conditional on that. This
   is the one most likely to produce a crash rather than a disappointment.
2. **The track is an analytic centreline.** `Track` is a periodic B-spline
   fitted to samples; the real car has a map. `scuderia_gym_jax`'s occupancy
   maps have the same problem and it is why they are still not connected
   (see `docs/source/plant.md`). A centreline extractor is a prerequisite,
   and `race_stack` has one.
3. **Opponents are exact circles at known positions.** `MPCC.set_obstacles`
   takes ground truth. From `datmo` it gets a tracked estimate with latency and
   dropouts, and the keep-out's `obs_margin` is currently 0.15 m of pure model
   error with no allowance for perception error at all.

- [ ] **Timing on the target, measured the way `benchmarks/solve_time.py`
      measures it: worst case, not mean.** 1.9 ms mean / 3.4 ms worst is on a
      workstation. The car's compute is not this machine, and the existing
      result explicitly makes the point that a controller whose mean fits and
      whose tail does not is a controller that misses deadlines.
- [ ] **The safety filter has to be in the loop before the tuner is**, and it
      inherits its guarantee from its model -- a kinematic bicycle that does not
      know about slip angles. On the real car that is the same wrongness
      `docs/source/plant.md` documents against the fitted-tyre plant.

## Known odd, undiagnosed

- `benchmarks/filters.py`: the `worst-case` tube leaves the track in 60% of runs
  at true grip 1.0. A filter assuming *less* grip than the plant has should be
  conservative and should not fail. **Do not cite that row.**
- On the `mixed` track both global and per-segment θ leave the track in episode
  0. That is not a scheduling failure — 1.76 m minimum radius against the
  oval's 2.46 m, and the default weights do not survive a lap however they are
  scheduled. Same failure as item 4.
