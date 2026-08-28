# TODO — mpcc-online-tuning

Ordered by what unblocks a result. Everything already measured lives in
`docs/source/`; this file is only what is **not** done.

**Standing rule for this repo: no dependency on another of my repos.** Where an
existing repo has a good pattern (`MPCC_planner_acados` for the acados OCP,
`MPCC_controller_ipopt` for the dynamic tyre model), **copy it in and adapt**,
with a comment saying where it came from. The one exception is
`scuderia_gym_jax`, which is a *plant* and stays an optional extra — the bridge
is `mpcc_tuning/plant_scuderia.py` and it already works.

## Where things stand

Oval, 26 episodes, same seed, last 8 episodes:

| | covered | off-track |
|---|---|---|
| baseline, one global θ | 7.6 m | 8/8 |
| event-triggered updates, *deferred* | 7.4 m | 8/8 |
| event-triggered updates, *discard* | 38.5 m | 6/8 |
| behind a predictive safety filter | 68.1 m | 0/8 |
| **per-segment θ (curvature)** | **78.0 m** | **0/8** |

Real-time is done: `mpcc_tuning/rti.py`, 1.9 ms mean / 3.4 ms worst at N=12
against 50 ms at 20 Hz.

---

## 1. Situation-dependent weights — best result, needs confirming

`experiments/per_segment_weights.py`. One θ per curvature segment, segment read
from the path *ahead*, so no detector is needed.

- [ ] **Reproduce over ≥5 seeds.** The 78.0 m result is one seed.
- [ ] Continuous schedule θ(κ) instead of three bins — a linear map from a
      curvature-preview feature vector, with the envelope gradient chain-ruled
      through it.
- [ ] Check it composes with the safety filter and with item 4.
- [ ] The learned weights are counter-intuitive (progress weighted *higher* in
      the tightest segment). Explain or stop repeating it.

## 2. Weights as a behaviour policy — the direction

θ is a behaviour parameterisation, not just a tuning vector: `q_v` sets
aggression, `q_c` sets how strictly the line is followed. A policy mapping
*situation → θ* subsumes item 1, with curvature as one feature among opponent
gap, closing rate, and distance to the boundaries. It should be able to express
**overtake vs stay behind**, and **use every corner vs follow the global
racing line**.

### 2a. Precondition: the MPCC cannot see opponents

**There is no obstacle constraint in `mpcc_tuning/mpcc.py` at all.** Overtaking
cannot be expressed today. Nothing else in item 2 can start before this.

- [ ] Add a circular keep-out per opponent, as a soft path constraint with
      slacks. Template: `MPCC_planner_acados/scripts/generate_acados_solver.py`
      builds exactly this (`max_obstacles`, `dist2 - r_eff**2 >= 0`) — **copy
      the formulation in, do not import it.**
- [ ] Opponents in the plant too, moving at their own speed.

### 2b. Establish the safe weight box — before any learning

`experiments/weights_as_behaviour.py` measured: `q_v` is a genuine dial (mean
speed 0.40 → 3.92 m/s, monotone, **saturating above ≈2**), and `q_c` is **not**
— ×10 barely changes the line, ×100 **drives off the track**.

So the usual safety argument for learning weights rather than steering — "the
MPCC enforces the constraints, so any θ is feasible" — is **false as stated**.

- [ ] Sweep every weight to failure and record the box where θ is a behaviour
      knob rather than a conditioning knob.
- [ ] Bound the policy's output to that box. Note the tuner currently drives
      `q_v` to 30+, deep inside the dead zone above 2.

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

- [ ] Event-triggered behaviour selection: re-decide θ on events, hold between.
- [ ] Compare against (i) a fixed schedule, (ii) a per-tick MLP. **Only claim
      the network if it beats both** — otherwise it is decoration.

## 3. acados backend for real time

`mpcc_tuning/rti.py` already meets 20 Hz with CasADi's `qrqp` (1.9 ms mean,
3.4 ms worst at N=12). acados would be faster and is what the car will run.

- [ ] Vendor an acados OCP builder into `mpcc_tuning/acados_ocp.py`. Template:
      `MPCC_planner_acados/scripts/generate_acados_solver.py` — **copy and
      adapt, do not import**. Keep weights as **runtime stage parameters**
      (it already does) so the envelope gradient still applies.
- [ ] Note the two-place `ctrl_mode` gotcha: the config override lands in the
      packed vehicle array and the `ModelSpec` copy is what the kernels
      dispatch on; `validate_against` checks they agree.
- [ ] Confirm the direction-agreement result (cosine 1.0000) holds for a real
      acados `SQP_RTI` step — the current measurement used a hand-rolled SQP.
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

## 5. Bound where θ goes

Most of the damage happens in episode 0: `q_v` moves ×222 before a single
episode boundary. α = 2e-4 and 2e-3 give the same shape ten times apart, which
is a divergence signature rather than a tuning artefact.

- [ ] A trust region on θ, and compare against item 1 (which may already fix it)
      and against discarding updates (38.5 m, partial).
- [ ] Trigger on the **safety filter's intervention** rather than lateral
      error: behind the filter `q_c` falls to 0.136 with performance drifting
      down, because the filter absorbs the consequence and it never reaches the
      return. The intervention *is* the missing error signal.

## 6. Filters on a state estimate

Every filter reads the true state. On the car it reads an estimate, and every
guarantee is conditional on that.

- [ ] Run the filters on odometry rather than ground truth.
- [ ] Add the estimator covariance to the margin, or state the guarantee as
      conditional.

## Known odd, undiagnosed

- `benchmarks/filters.py`: the `worst-case` tube leaves the track in 60% of runs
  at true grip 1.0. A filter assuming *less* grip than the plant has should be
  conservative and should not fail. **Do not cite that row.**
- On the `mixed` track both global and per-segment θ leave the track in episode
  0. That is not a scheduling failure — 1.76 m minimum radius against the
  oval's 2.46 m, and the default weights do not survive a lap however they are
  scheduled. Same failure as item 4.
