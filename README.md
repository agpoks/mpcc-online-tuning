<p align="center">
  <img src="docs/source/_static/logo-banner.svg" alt="mpcc-online-tuning" width="560">
</p>

# mpcc-online-tuning

**Tuning an MPCC's cost weights online, every control tick, from one scalar.**

The controller stays an MPCC — same solver, same constraints, same guarantees.
What changes is that its cost weights are treated as the parameters of a
reinforcement-learning policy and updated *while the car drives*, from a single
TD error per tick.

This is a **spike**: a small, deliberately throwaway prototype built to answer
one question — is this real-time feasible, and does the gradient it needs
actually exist in closed form? Both answers are yes, and both are checked here
rather than asserted. It is not a finished tool.

Docs: see [`docs/`](docs) (built on Read the Docs).

## The idea in one equation

The reason this is affordable is the **envelope theorem**. At the solution of
the MPCC's NLP, the derivative of the optimal value with respect to a cost
weight is the *partial* derivative of the Lagrangian, with the primal and dual
variables held fixed:

```
d/dθ  J*(s; θ)  =  ∂/∂θ [ f(w*, θ) + λ*ᵀ g(w*, θ) ]
```

No implicit function theorem. No differentiating through the solver. No adjoint
sweep. It falls out of a solve that was happening anyway, which is the whole
argument for why an MPC's parameters can be tuned at control rate when a neural
policy's cannot.

Verified against finite differences across 15 states and three weight settings
(`python examples/gradient_check.py`): **cosine 1.00000**, max relative error
3.7e-4 — and the gradient costs **0.079% of the solve it comes from**. That
last number is the entire argument for why this can run at control rate.

## The algorithm

Gros & Zanon's MPC-as-function-approximator, plus eligibility traces. Per tick:

```
a       = argmin of the MPCC at s                 # this is π_θ(s)
Q(s,a)  = the MPCC's optimal value with u₀ pinned to a
δ       = r + γ V(s') − Q(s,a)                    # one scalar
e      ← γλ e + ∇_θ Q
θ      ← θ + α δ e
```

`r` is the **real** objective — metres of track covered, with leaving it
penalised — and deliberately not the MPCC's internal cost. If the two were the
same quantity there would be nothing to learn.

One NLP is built, and solved twice per tick: once unconstrained (that is `V`,
and its first control is what gets applied) and once with `u₀` pinned (that is
`Q`). The second is not a second problem, just tighter bounds on two decision
variables — and when the applied action *is* the argmin, `Q(s,π(s)) = V(s)` and
the second solve is skipped entirely.

## Run it

```bash
pip install -e .
python examples/gradient_check.py                # check the premise first
python examples/tune_online.py --episodes 30 --plot runs/tuning.png
python examples/tune_online.py --frozen          # the control: no tuning
```

Both examples have a notebook version (`examples/*.ipynb`), regenerated with
`python scripts/make_notebooks.py`.

The MPCC starts with deliberately bad weights (far too much lag penalty, almost
no reward for progress, so it crawls) and has to find better ones from driving.

## What it does, measured

Starting from deliberately bad weights (`q_l = 200`, `q_v = 0.05` — far too
much lag penalty, almost no reward for progress, so the MPCC crawls), 200 ticks
per episode, `alpha = 2e-4`:

| episode | metres covered | `q_l` | `q_v` |
|---|---|---|---|
| 0 | 19.3 | 195 | 0.05 |
| 3 | 23.7 | 183 | 0.06 |
| 6 | 30.9 | 144 | 0.10 |
| 8 | 37.1 | 101 | 0.48 |
| 10 | **37.2** | 66 | 3.5 |
| 12 | 37.1 | 43 | 27 |
| **13** | **20.1  OFF-TRACK** | 41 | 36 |
| 20 | 5.2  OFF-TRACK | 37 | 37 |

**It works.** Distance covered nearly doubles in a dozen episodes, with no
crashes on the way up, and 37.2 m in 200 ticks is 3.7 m/s average against a
grip-limited corner speed of 3.9 — it has essentially found the optimum. The
weights move the way you would move them by hand, and nobody moved them.

**Then it destroys itself, and the mechanism is worth more than the success.**
Once performance saturates the TD error stays slightly positive, so `q_v` keeps
climbing and `q_c` keeps falling long after either helps. By episode 13 the
MPCC is willing to ride the constraint boundary, the tyre limit it does not
model puts it off the track, and with `q_c ~ 0.2` it has no way back. The tuner
optimised a proxy past the point where the proxy was valid, with nothing to
stop it.

That is the honest result of the spike: **the gradient is exact and cheap, the
loop works, and it has no stopping criterion.** Candidate fixes, none tried
here: a decaying step size; keeping the best `theta` and reverting on
regression; a trust region on `theta`; or -- the interesting one -- putting a
predictive safety filter around the whole thing, which is exactly the failure
mode it exists for. That construction is in
[`rtrrl-playground`](https://github.com/agpoks/rtrrl-playground)'s `safety.py`,
and on a comparable task it took an unfiltered learner from crashing in 61% of
training episodes to 0%.

## The idea this is actually a stepping stone to

[`docs/source/influence_through_a_solver.md`](docs/source/influence_through_a_solver.md)
works out what happens when you stop assuming the controller is memoryless.
Under real-time iteration it is not: warm-starting makes the solver a
*dynamical system*, `w_{t+1} = Phi(w_t, s_t, theta)`, which is structurally an
RNN whose update happens to be a Newton step — so the influence obeys the same
RTRL recursion, and the Newton contraction matrix plays the role of the
recurrent Jacobian.

The observation that makes it worth writing up: **the solver's contraction rate
is the influence trace's decay rate**, and unlike an RNN's you do not have to
guess it. It is a property of the QP you already solved, so the truncation
horizon is *derived* rather than tuned.

The note also lists the experiment that would kill the idea in an afternoon,
and puts it first.

## Why a separate repo, and not a branch of `scuderia_gym_jax`

Asked and answered deliberately, because the alternative was tempting:

* **`scuderia_gym_jax` is a validated simulator.** It has parity tests against
  the numba original at every level and its own published docs. Its value is
  that you can trust its numbers. Adding CasADi, IPOPT and an RL loop to it
  widens its dependency surface and blurs what it is for.
* **The dependency arrow points the wrong way for a branch.** This repo needs
  a simulator; the simulator does not need a tuner. A branch inverts that, and
  branches that invert a dependency never merge.
* **`scuderia_gym_jax` is pure JAX on purpose.** CasADi/IPOPT is a different
  numerical stack with a different build story. Keeping them in separate
  environments is what lets either one be installed without the other.

So: separate repo, and the simulator is an optional dependency. The plant here
is a plain kinematic bicycle so the spike runs with nothing installed; swapping
in `scuderia_gym_jax`'s ST/STD models is the obvious next step and is a change
to one class.

## What is deliberately wrong with the controller

The plant has a **tyre grip limit** — a cap on yaw rate at `A_LAT_MAX·grip/v` —
and the MPCC does not model it. That mismatch is the point. A limit the
controller does not know about shows up as cost weights that are wrong for the
real vehicle, and compensating for it is exactly what an online tuner should be
able to do. A shared model between plant and controller would make the question
unaskable.

## Layout

```
mpcc-online-tuning/
├── mpcc_tuning/     mpcc.py (the NLP + the gradient), learner.py (TD(lambda)),
│                    track.py (the path as a CasADi spline), model.py
├── examples/        gradient_check.py, tune_online.py -- .py and .ipynb
├── notes/           formulation.md: the algebra, and the three traps in it
├── tests/           the gradient identity, checked against finite differences
└── docs/            Sphinx / Read the Docs source
```

There is **no dataset and nothing to download**. The plant is a kinematic
bicycle in `model.py` and the track is generated; every number here is
reproducible from a seed.

## Status, honestly

- [x] Parametrised MPCC in CasADi, contouring/lag/progress/effort weights
- [x] `V(s)`, `Q(s,a)` and `π(s)` from one NLP
- [x] Envelope-theorem gradient, validated against finite differences
- [x] TD(λ) Q-learning outer loop with eligibility traces
- [x] Plant/controller model mismatch as a first-class knob
- [ ] **Solve time is ~150 ms per NLP with IPOPT** — fine for a spike, far too
      slow for 20 Hz on a car. The fix is acados with an RTI scheme (one SQP
      iteration per tick), which is what `MPCC_planner_acados` already uses; the
      envelope gradient is available there too.
- [ ] **A stopping criterion.** See "What it does, measured" -- the loop is
      stable while it improves and then over-optimises. This is the next thing
      to fix, not a footnote.
- [ ] `scuderia_gym_jax` as the plant
- [ ] Tuning the *model* parameters, not only the cost weights (the Lagrangian
      term in the gradient is already implemented for it)
- [ ] Deterministic policy gradient as an alternative to Q-learning — needs
      `du*/dθ` from the KKT system, which is a real but larger piece of machinery

**No convergence guarantee.** Q-learning with a nonlinear function
approximator — and an MPC certainly is one — has none, and eligibility traces
do not add one. This is an empirical procedure with a good structural prior.

## References

* Gros & Zanon, *Data-driven Economic NMPC using Reinforcement Learning*, IEEE
  TAC 2020 — the MPC-as-function-approximator framework this implements
* Zanon & Gros, *Safe Reinforcement Learning Using Robust MPC*, IEEE TAC 2021
* Gros & Zanon, *Reinforcement Learning for MPC: Fundamentals and Current
  Challenges*, IFAC 2023 — the survey to read first
* [`mpcrl`](https://github.com/FilippoAiraldi/mpc-reinforcement-learning) —
  a fuller CasADi implementation of the same framework
* [MPC4RL](https://arxiv.org/html/2501.15897) — the acados-based counterpart
* Nguyen, Nguyen, Amine, Vo-Duy, Mangharam, Nghiem, *AD-MPCC: Adaptive
  Differentiable Model Predictive Contouring Control for Autonomous Racing*,
  2026 — [arXiv:2607.00141](https://arxiv.org/abs/2607.00141); the closest
  neighbour, adapting MPCC weights per tick by differentiating the solver
* Liniger, Domahidi & Morari, *Optimization-based autonomous racing of 1:43
  scale RC cars*, 2015 — the MPCC formulation itself

Eligibility traces and the TD(λ) machinery are shared in spirit with
[`rtrrl-playground`](https://github.com/agpoks/rtrrl-playground), where the same
outer loop drives a recurrent network instead of a solver.

## License

MIT, see [`LICENSE`](LICENSE).
