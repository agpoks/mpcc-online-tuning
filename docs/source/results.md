# Results, including the failure

Everything below is from `examples/tune_online.py` on a 26.7 m oval with 2.5 m
corners, 200 control ticks per episode at 20 Hz, plant grip 1.0, `alpha = 2e-4`.

## The gradient is exact

The claim the whole approach rests on: at the solution of the MPCC's NLP,

$$\frac{\mathrm{d}J^*}{\mathrm{d}\theta} \;=\; \frac{\partial \mathcal{L}}{\partial \theta}\Big|_{w^*,\lambda^*}$$

Checked against central finite differences on the optimal value:

| | |
|---|---|
| cosine to finite differences | **1.0000** |
| relative error | **< 1e-3** |

`tests/test_gradient.py` asserts this, and `examples/gradient_check.py` shows
it across states and weight settings. It costs one evaluation of a function
built once, on quantities the solver already returned.

## It tunes itself into a good controller

Starting from `q_l = 200`, `q_v = 0.05` — far too much lag penalty, almost no
reward for progress, so the MPCC crawls:

| episode | metres covered | `q_l` | `q_v` |
|---|---|---|---|
| 0 | 19.3 | 195 | 0.05 |
| 3 | 23.7 | 183 | 0.06 |
| 6 | 30.9 | 144 | 0.10 |
| 8 | 37.1 | 101 | 0.48 |
| 10 | **37.2** | 66 | 3.5 |
| 12 | 37.1 | 43 | 27 |

Distance covered nearly doubles in a dozen episodes, no crashes on the way up.
37.2 m in 200 ticks is 3.7 m/s average against a grip-limited corner speed of
3.9 — it has essentially found the optimum, and nobody touched the weights.

## Then it destroys itself

| episode | metres covered | `q_c` | `q_v` |
|---|---|---|---|
| 12 | 37.1 | 0.36 | 27 |
| **13** | **20.1  OFF-TRACK** | 0.30 | 36 |
| 20 | 5.2  OFF-TRACK | 0.26 | 37 |
| 23 | 5.1  OFF-TRACK | 0.22 | 45 |

The mechanism is worth more than the success. Once performance saturates the TD
error stays slightly positive — the critic under-estimates a policy that is
already at the limit — so `q_v` keeps climbing and `q_c` keeps falling long
after either helps. By episode 13 the MPCC is willing to ride the constraint
boundary, the tyre limit it does not model puts it off the track, and with
`q_c ≈ 0.2` it has no way back.

**The tuner optimised a proxy past the point where the proxy was valid, with
nothing to stop it.** That is the honest result of the spike: the gradient is
exact and cheap, the loop works, and it has no stopping criterion.

Candidate fixes, none tried here:

* a **decaying step size** — the cheapest thing that would work;
* **keep the best `theta` and revert on regression** — cheap and crude, and it
  needs a notion of "the same conditions" to compare across;
* a **trust region on `theta`** — principled, and the right long-term answer;
* a **predictive safety filter around the whole thing** — the interesting one,
  because this failure is exactly what such a filter exists for. That
  construction is in
  [`rtrrl-playground`](https://github.com/agpoks/rtrrl-playground)'s
  `safety.py`; on a comparable task it took an unfiltered learner from crashing
  in 61% of training episodes to 0%, with no cost in final performance.

## What is not real-time yet

| | |
|---|---|
| solve time, IPOPT, N=12 | ~150 ms |
| budget at 20 Hz | < 50 ms for the whole tick |

Fine for a spike, far too slow for a car. The answer is acados with a real-time
iteration scheme — one SQP iteration per tick, warm-started from the last —
which is what `MPCC_planner_acados` already does. The envelope gradient is
available there too: it needs the objective's partial derivative and the
multipliers, and acados returns both.

Note also that the second solve is usually free. `Q(s, π(s)) = V(s)` by
definition, so the action-value solve is skipped entirely unless the applied
action was perturbed for exploration.
