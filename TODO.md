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

Tracks: `oval` (straights + two 180s), `mixed` (harmonic, no 90s),
**`circuit`** (all four named sectors, 47.2 m, min radius 2.59 m).

Per-segment θ is now **confirmed over six seeds**
(`experiments/per_segment_seeds.py`): global 8.9 m (sd 0.9), collapsing on
**6/6** seeds; per-segment 77.8 m (sd 0.2), collapsing on **0/6**. The
distributions do not overlap.

Real-time is done: `mpcc_tuning/rti.py`, 1.9 ms mean / 3.4 ms worst at N=12
against 50 ms at 20 Hz. **It is CasADi `qrqp`, not acados** — see item 3.

---

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

### Overleaf

`https://git.overleaf.com/6a91eb483816017e248781d9` is a **full mirror of this
repo**, not just `paper/`. One commit, "Update on Overleaf.", identical to
`aa3f2ba`. Push is a clean fast-forward.

- [ ] Nothing has been pushed yet. The session's work is uncommitted locally.
- [ ] Rotate the Overleaf token — it was pasted in plaintext into a chat
      transcript and is embedded in a scratch clone's `.git/config`.

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
