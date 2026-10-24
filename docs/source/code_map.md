# How the code works — a map from the method to the functions

This page is the bridge between the method and the source. It follows **one
control tick** all the way around the loop, gives the equation each stage
computes, and names the exact file and function that does it. If you want the
gentle build-up first, read {doc}`tutorial/learning_algorithm`; if you want the
*why* behind each design choice, {doc}`formulation`, {doc}`safety` and
{doc}`behaviour_policy` have it. This page is the "where is it, and what does it
compute" reference.

Everything below lives in the `mpcc_tuning/` package; the runnable loop that
wires it together is `experiments/online_from_baseline.py`.

## The loop in one picture

Every 50 ms the car's situation becomes an observation, the policy turns that
into nine cost weights, the MPCC solves one optimal-control problem with those
weights and emits an actuator command, the plant advances, and the learner
updates the policy from the reward that came back. Once per episode a keep-best
rule decides whether the new policy is kept.

```text
          ┌───────────────────────────────────────────────────────────────────┐
          │  CLOSED LOOP — once per 50 ms control tick                          │
          └───────────────────────────────────────────────────────────────────┘

  (1) track.py              (2) ltc.py                 (3) ltc.py + baselines.py
  track geometry ─────────▶ features()      ─────────▶ WeightPolicy.step()
  curvature / sector /      18-dim obs  x              θ = θ₀ + span·tanh(G·u)
  width / lateral                                      + class residual + bands
                                                               │  θ  (9 log-weights)
                                                               ▼
  (6) ltc.py                (5) plant_scuderia.py      (4) acados_ocp.py + acados_mpcc.py
  PolicyTuner.learn() ◀──── ScuderiaPlant.step(u) ◀─── AcadosMPCC.value()
  TD(λ) critic + actor      dynamic drift plant         solve  min J(θ)  →  u₀
  updates G, cell.p, w      returns state s′, reward r
        │
        └─ per episode ─▶ PolicyTuner.end_episode() → run frozen lap →
                          PolicyTuner.confirm_candidate()   (keep-best + validate)
```

The numbers (1)–(6) are the sections below. Two facts hold across the whole
loop: the **MPCC solves the control problem and drives the car on every tick**,
learning or frozen; and the **weights never touch the hard constraints** — they
are a cost parameter, so the track, grip and actuator limits hold whatever the
policy emits.

## The nine weights θ

The policy's output is `θ`, a vector of nine numbers in **log space**
(`θ = log(weight)`), so a weight is `exp(θᵢ)` and can never go non-positive. The
names and order are fixed in `WEIGHT_NAMES` (`mpcc_tuning/mpcc.py`):

| idx | θ name  | weight it sets                                   | enters the OCP as |
|-----|---------|--------------------------------------------------|-------------------|
| 0   | `q_c`   | contouring — cost of leaving the reference line  | **cost** (linear) |
| 1   | `q_l`   | lag — cost of being ahead/behind along the path  | **cost** (linear) |
| 2   | `q_v`   | progress reward — how much speed is paid for      | **cost** (linear) |
| 3   | `r_d`   | steering damping                                  | **cost** (linear) |
| 4   | `r_a`   | acceleration damping                              | **cost** (linear) |
| 5   | `r_dv`  | progress-rate damping                             | **cost** (linear) |
| 6   | `d_obs` | berth kept around an opponent                     | **constraint** (keep-out) |
| 7   | `k_v`   | fraction of the grip-limited corner speed aimed for | **cost** (speed-ref term) |
| 8   | `d_bound` | extra margin kept from the track edge           | **constraint** (corridor) |

Indices 0–5 are `WeightPolicy.COST_IDX`: they multiply cost terms **linearly**,
so scaling all six by a constant scales the optimal cost `J*` but leaves the
plan unchanged (this is the "gauge" the optional `gauge_fix` removes). `k_v`
enters a *squared* speed-tracking term, and `d_obs`/`d_bound` enter the
constraints — so they are handled separately (bands, not the gauge).

---

## (1) The track — geometry

**Where:** `mpcc_tuning/track.py` (`Track`).

The track supplies everything geometric as functions of arc length `s`:
`curvature(s)`, `sector(s)` (straight / long-curve / 90° / hairpin, by total
heading change), `half_width`, `lateral(x, y)` (signed distance from the
reference line), `wrap(s)` and `length`. A raceline-referenced track also carries
its optimiser's speed profile `v_ref(s)`. The reference point `p(s)` that the
contouring cost is measured against is the spline built into the OCP (next
sections). See {doc}`plant` for how a track is mapped and smoothed.

---

## (2) The observation — `features()`

**Where:** `features()` in `mpcc_tuning/ltc.py`; opponent labelling in
`classify_opponent()`.

Turns the raw state `s5 = (x, y, ψ, v, s)` and the visible opponents into an
**18-dimensional, dimensionless** vector `x` (constant `N_FEATURES = 18`). Every
entry is a ratio of physical quantities, so a policy keyed on it transfers across
tracks and speeds:

- **curvature preview** (3) — `tanh(3·curvature)` at ~0.5/1.5/3 m ahead;
- **speed fraction** `v/v_max`, **lateral margin fraction**;
- **gap features** (4) — gap in braking distances, time-to-collision, pass
  availability, gap in car lengths (saturated, not zeroed, when no opponent — "no
  car" is a *state*, not a missing value);
- **sector one-hot** (4) — a *soft* membership that ramps up over ~2 m of
  approach rather than switching at a line;
- **track width** in car widths (1);
- **opponent-class one-hot** (4) — static / slower / equal / faster, from the
  *estimated* opponent speed via `classify_opponent()`.

The class is graded by **relative** speed on purpose: catching and being caught
are the same instantaneous gap with opposite closing rates, which is why the
policy needs memory (next section). Details and the reasoning: {doc}`behaviour_policy`.

---

## (3) The policy — `WeightPolicy` over an `LTCCell`

**Where:** `LTCCell` and `WeightPolicy` in `mpcc_tuning/ltc.py`; the box from
`baselines.adaptation_box()`; the anchor `θ₀` from `baselines.start()`.

### The recurrent cell

`LTCCell.step(x)` advances one liquid-time-constant layer. With gate
`f = σ(W ξ)` on the stacked input `ξ = [x; h; 1]`, the hidden state updates as

$$
h' \;=\; \frac{h + \Delta t\, f \odot A}{1 + \Delta t\,(1/\tau + f)} .
$$

`A` is a per-neuron reversal potential, `τ` a per-neuron time constant; both are
clipped in `LTCCell.clip()` so a long online run cannot run the state to
infinity. The same `step()` also returns the pieces the learner needs: `imm`
(the immediate parameter sensitivity), `leak = 1/den`, and optionally the full
recurrent Jacobian `D` (for exact RTRL vs the cheaper RFLO approximation).
`MLPCell` is the memoryless drop-in used for the ablation — same interface, no
state.

### Situation → weights

`WeightPolicy.step(feat)` emits `θ`. It reads a **direct path**
`u = [h; feat; 1]` (the hidden state alone barely varies with the input, so the
features reach `θ` directly and the recurrence adds only what needs memory), then
squashes into the measured box anchored at `θ₀`:

$$
z = G\,u, \qquad
\theta \;=\; \theta_0 \;+\; \mathrm{span}\odot\tanh(z), \qquad
\mathrm{span}_i =
\begin{cases} \mathrm{hi}_i-\theta_{0,i}, & \tanh z_i \ge 0\\[2pt]
\theta_{0,i}-\mathrm{lo}_i, & \tanh z_i < 0. \end{cases}
$$

Anchoring at `θ₀` puts the working controller at `tanh(0)` — the *steepest*,
most responsive point — and the **asymmetric span** lets a weight that sits on a
bound still move inward. The box `[lo, hi]` comes from
`baselines.adaptation_box(track, factor=2)`: at most ×2 / ÷2 around `θ₀`,
intersected with the global `THETA_LO/THETA_HI`. The squash (a `tanh`, not a hard
clip) keeps a gradient everywhere inside the box — a hard clip would switch off
learning on any axis that reached its bound.

### The safety layer

After the squash, `WeightPolicy.step()` applies, in order:

- a **class-conditioned residual** `θ += delta_log·tanh(D[class])` (a learned
  `(n_classes × 9)` table, indexed by the episode's opponent class via
  `set_class()`), clipped back to the box — so different classes can hold
  different weights even when `G u` is identical;
- **class-conditioned bands** that clamp the safety-carrying weights: a `k_v`
  floor *and* ceiling (the grip band), an `r_a` floor, a `q_c` floor — each one
  removing a specific measured failure;
- a **crowd override** (`set_crowd()`): once a full field is *remembered*
  (counting opponents seen at least once, so it persists through occlusion),
  `q_c` is opened to a looser weaving band.

Each band masks the gradient on its axis exactly where it bites (like a clip), so
the learner never fights a bound it cannot cross. The full rationale, and the one
failure each band fixes, is in {doc}`safety`.

`WeightPolicy.grads(dQ_dθ)` chains a downstream `∂·/∂θ` back into the policy
parameters — `dG = (dQ_dθ ⊙ sq) ⊗ u` exactly, and the cell gradient
`(g·G)·P` as RFLO — and stashes the residual-head gradient for the tuner. This
is the only coupling point between the controller's gradient and the network.

---

## (4) The controller — the MPCC OCP

**Where:** the problem is built once in `build_ocp()` (`mpcc_tuning/acados_ocp.py`)
and solved each tick through `AcadosMPCC` (`mpcc_tuning/acados_mpcc.py`).

The contouring and lag errors are the projections of the car onto the frame of
the reference point `p(s) = (ref_x, ref_y)` chosen by the solver:

$$
e_c = \sin\psi\,(p_x-\mathrm{ref}_x) - \cos\psi\,(p_y-\mathrm{ref}_y), \qquad
e_l = -\cos\psi\,(p_x-\mathrm{ref}_x) - \sin\psi\,(p_y-\mathrm{ref}_y).
$$

With `wᵢ = exp(θᵢ)` the per-stage cost (assembled in `build_ocp()`, kept as an
`EXTERNAL` cost so the *linear* progress term survives) is

$$
J_k = q_c\,e_c^2 + q_l\,e_l^2 \;-\; q_v\,v_s\,\Delta t
\;+\; r_\delta\,\delta^2 + r_a\,a^2 + r_{\dot s}\,(v_s-v)^2
\;+\; q_{\mathrm{vref}}\,(v - k_v\,v_{\mathrm{ref}}(s))^2 ,
$$

with a terminal `q_c e_c² + q_l e_l²`. The `−q_v v_s Δt` term is what makes it
*contouring* control: progress `v_s` is rewarded directly rather than tracking a
time-parameterised trajectory. `k_v` reaches the cost only through the
`q_vref` speed-reference term. The **hard constraints** — the corridor
half-spaces (with the `d_bound` margin), the actuator bounds, and the opponent
keep-out row (radius set by `d_obs`) — are built alongside the cost and are
**independent of θ's cost weights**.

`AcadosMPCC.value(state, θ)` is the per-tick entry point: it pushes `θ` and the
obstacle slots into the solver (`_set_params()` / `_p()`), solves, and returns
`{u0, value=J*, ok, status}`. A **solver-failure guard** (opt-in; the reference
for the deployed control loop) never acts on a failed solve — it holds the last
feasible steering and brakes gently during the cold-start phase. `action_value()`
gives `Q(s,a)`, equal to `V(s)` when `a = π(s)`. `grad_theta()` returns the
envelope gradient `dJ*/dθ` — used only by the **shelved** MPCC critic (below).

---

## (5) The plant — `ScuderiaPlant`

**Where:** `ScuderiaPlant.step(u)` in `mpcc_tuning/plant_scuderia.py`.

The evaluation plant is the **dynamic single-track drift model** (slip, load
transfer, a rate-limited servo) from `scuderia_gym_jax`. `step(u)` advances the
true state and returns `(s5, reward, off_track, terminated)`, with the plant
reward being progress minus a penalty for leaving the track,
`r = Δs − c_off·1[off]`. The controller predicts with a *simpler* model and does
**not** share the plant's parameters — that mismatch is the point of the project:
absorbing it online is what the learning is for. See {doc}`plant`.

---

## (6) The learner — `PolicyTuner`

**Where:** `PolicyTuner` in `mpcc_tuning/ltc.py`: `act()`, `learn()`,
`end_episode()`, `confirm_candidate()`.

`PolicyTuner` runs TD(λ) on the **policy** (not on `θ` directly). `act(feat,
state)` emits `θ` (adding exploration noise `ε ~ N(0, σ²)` in log space), solves
through the MPCC and returns `(θ, action)`. `learn(reward, …)` does one update.

### The clock

With `clock="progress"` the update is **semi-Markov**: it fires once per
`ds_ref` metres of real progress, not once per tick, so the learning rate per
metre is speed-independent. The discount and trace decay are raised to
`n = ds/ds_ref`. The reward is then `−Δt` per tick, so the return is minus the
lap time — the racing objective. (`experiments/online_from_baseline.py`
recovers the progress from the plant reward and charges time instead.)

### The critic — the return critic (deployed)

A small **linear** critic `V_w(x) = w·x` on the policy's own features, trained by
TD(λ) on the **actual reward**, with `θ` nowhere in it:

$$
\delta = r + \gamma^{\,n} V_w(x') - V_w(x), \qquad
e_w \leftarrow (\gamma\lambda)^{\,n} e_w + x, \qquad
w \leftarrow w + \alpha_c\,\delta\,e_w .
$$

### The actor — score-function gradient

`θ` enters the return only through the policy, so the direction comes from the
realized perturbation (the **score-function** estimator), chained through the
policy by `WeightPolicy.grads()` and accumulated in eligibility traces:

$$
g = \varepsilon/\sigma^2, \qquad
(dG, dc) = \texttt{pol.grads}(g), \qquad
e_G \leftarrow (\gamma\lambda)^{\,n} e_G + \widehat{dG}, \qquad
G \leftarrow G + \alpha\,\delta\,e_G
$$

(and likewise `cell.p` and, when present, the class-residual table `D`, which has
its own learning rate). `∂θ/∂φ` is carried forward by **real-time recurrent
learning** — no backprop through time, so the same single acados solve serves
learning and driving. An optional **trust region** caps the step norm; the traces
are RMS-normalised per block (`_norm()`).

### Keep-best and validate

- `end_episode(score, crashed)` — if the episode beat the banked score it
  returns `"validate"`; if it was worse by more than a tolerance (or crashed) it
  **reverts** to the banked network. The policy snapshot banked is the LTC cell
  parameters, the readout `G`, and the residual table.
- The experiment then drives that candidate **frozen** (no learning, no
  exploration) for one lap and calls `confirm_candidate(frozen_score)`, which
  banks it only if the *frozen* score beats the incumbent. This is the stopping
  criterion the gradient lacks, and the on-vehicle safety layer.

### The shelved alternative

`critic="mpcc"` uses `V = −J*` and the envelope gradient `grad_theta()` instead.
It is implemented and kept for comparison but **not** used for any result: it
points the same way in every situation and drifts into over-claiming grip. The
story is in {doc}`spike_results` and {doc}`safety_filter_for_the_tuner`.

---

## File-by-file map

| file | what it is | key functions / classes |
|------|------------|--------------------------|
| `mpcc_tuning/track.py` | track geometry, mapping, sectors, raceline | `Track`: `curvature`, `sector`, `lateral`, `half_width`, `v_ref` |
| `mpcc_tuning/plant_scuderia.py` | the dynamic drift **plant** (sim stand-in for the car) | `ScuderiaPlant.step`, `.reset`, `.state_dyn` |
| `mpcc_tuning/acados_ocp.py` | **builds** the acados MPCC problem (cost + constraints) | `build_ocp`, `pack_params` |
| `mpcc_tuning/acados_mpcc.py` | **solves** it each tick; the controller interface | `AcadosMPCC.value`, `.action_value`, `.grad_theta`, `.set_obstacles` |
| `mpcc_tuning/model.py` | the prediction models (`DynamicBicycle`) and limits | `DynamicBicycle`, `tyre_sym`, `ACCEL_MAX`, `STEER_MAX` |
| `mpcc_tuning/ltc.py` | **observation, policy, learner** — the heart of this repo | `features`, `classify_opponent`, `LTCCell`, `WeightPolicy`, `PolicyTuner` |
| `mpcc_tuning/baselines.py` | the anchor `θ₀` and the measured box | `start`, `best`, `adaptation_box`, `Setting` |
| `mpcc_tuning/opponents.py` | opponent motion and the noisy speed tracker | `ObstacleTracker`, opponent models |
| `mpcc_tuning/speed.py` | the grip-limited reference speed profile `v_ref(s)` | `track_speed_profile` |
| `mpcc_tuning/acados_grad.py` | the envelope gradient `dJ*/dθ` (shelved critic only) | `AcadosEnvelopeGradient` |
| `mpcc_tuning/viz.py` | the animations and plots | `animate_run` |
| `experiments/online_from_baseline.py` | the runnable loop that wires it all together | `one`, `run_frozen`, `main` |

## Reading order

1. **Run it** — {doc}`getting_started` has the acados environment and the exact
   commands.
2. **Understand the learning rule on a toy** — {doc}`tutorial/learning_algorithm`.
3. **The controller and the safe-learning construction** — {doc}`formulation`,
   {doc}`safety`, {doc}`behaviour_policy`, {doc}`obstacles`.
4. **This page**, with the source open, to connect each equation to its function.
