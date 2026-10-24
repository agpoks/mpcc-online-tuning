# Keeping the learning safe

On a car that must not crash, the open problem is not finding better weights but
**moving them safely**. An unconstrained tuner walks a weight to a bound,
saturates, and the policy collapses to a constant that claims more grip than the
tyres provide (a cold-start departure) or drives harder than the corridor allows
(running wide). Three mechanisms, stacked, make an unsafe controller impossible
to construct. None is a cost term the learner can negotiate away.

```{admonition} Why not a safety filter?
:class: note
The spike's answer was a predictive filter around the tuner — a braking backup
that corrects unsafe commands. It works, and it is documented in
{doc}`safety_filter_for_the_tuner` and {doc}`filters`. But a filter absorbs a
bad weight *after* it reaches the OCP: nothing then discourages the tuner from
pushing $q_c$ toward zero, so the filter converted a crash into a slow
degradation. The guarantee has to live in the *weights*, before the solve. That
is what this page is.
```

## 1. Anchored, and bounded to a measured box

The policy does not output the weights freely. It is **anchored** at a stable,
hand-verified baseline $\vect\theta_0$ and squashed into a box
$[\vect\theta_{\mathrm{lo}},\vect\theta_{\mathrm{hi}}]$:

$$\vect\theta(\vect f)=\vect\theta_0+\mathrm{span}\odot\tanh(G\vect u),\qquad
  \mathrm{span}=\begin{cases}\vect\theta_{\mathrm{hi}}-\vect\theta_0 & \tanh\ge 0\\
                             \vect\theta_0-\vect\theta_{\mathrm{lo}} & \tanh<0.\end{cases}$$

So $\vect\theta_0$ sits at $\tanh(0)$: with the readout $G\approx 0$ the policy
emits the baseline, and online learning bends $\vect\theta(\vect f)$ *inside* the
box but can never leave it. **The box is measured, not guessed** — each edge is
set from a sweep of where that weight stops improving a clean lap, so the
admissible set is observed rather than assumed. A global search box spans four
orders of magnitude; a controller meant to *adapt* around a working anchor is
given a factor of two (`baselines.adaptation_box(track, factor=2)`). This is the
first mechanism: a parameterisation that is unsafe to construct does not exist in
the policy's range.

## 2. Class-conditioned bands

The box is necessary but not sufficient, because the weights that carry safety
have a **situation-dependent** safe range. These are clamped per opponent class
and crowd, *after* the squash:

- a band $k_v\in[\underline{k_v},\overline{k_v}]$ on the engaged classes — a high
  $k_v$ over-claims grip, a low one stalls an overtake;
- a **floor on $r_a$** — an under-penalised acceleration over-drives the corner
  and runs wide;
- a **floor on $q_c$** for the recurrent arm, opened to a weaving band only once
  a full crowd has been *remembered* through occlusion.

Each band was introduced to remove **one measured failure**, and each removes it
directly: the grip band on $k_v$ cuts grip-over-claim departures from 100% to 12%
of runs; the floor on $r_a$ eliminates the over-driven wall-out. The result is
clean laps across every opponent class. These are the mechanism that lets the
policy keep learning on a car that must not crash — not a tuning convenience.

```{admonition} The band is a floor/ceiling, not a target
:class: tip
Inside its band a safety weight still adapts to the situation; the band only
forbids the edge that is unsafe. In practice the reward optimum often sits *at*
the band edge and the tuner pins the weight there (flat, in the weight traces) —
which is exactly the behaviour the band exists to make safe.
```

## 3. Validated keep-best

How an updated policy is **accepted** supplies the stopping criterion the
gradient lacks. The rule banks a new incumbent $\phi^\star$ only when a lap
driven **frozen** — no learning, no exploration noise — improves on the
incumbent, and reverts to $\phi^\star$ on any slower lap or track departure:

```
after each learning episode with score s:
    if s > best_validated:   candidate = snapshot(policy)
                             s_frozen  = drive(candidate, frozen=True)
                             if s_frozen > best_validated: bank(candidate, s_frozen)
    elif s < best_validated - tol or crashed:
                             restore(best policy)
```

The frozen validation matters, and it was measured the hard way: an episode's
score is produced by the policy *as it changes through the episode* plus
exploration noise, not by the snapshot at its end. Networks credited with
2.32 / 2.34 / 2.47 laps drove, frozen, 2.11 / 2.06-then-crash / 1.83. Banking at
the frozen score is what keeps the deployed net from capturing a transient fluke.

Run continuously, the same rule is the **on-vehicle adaptation-and-safety
layer**: it reverts each excursion rather than compounding it. On the deployment
map it drove 14.3 laps through 38 track-departure ticks, every one recovered by a
revert and none terminal.

## Where this lives

| mechanism | where |
|---|---|
| anchor + squash into the box | `mpcc_tuning/ltc.py::WeightPolicy` |
| measured adaptation box | `mpcc_tuning/baselines.py::adaptation_box` |
| class-conditioned bands | `WeightPolicy` (the $k_v$/$r_a$/$q_c$ clamps) |
| keep-best with frozen validation | `PolicyTuner.end_episode / confirm_candidate` |
| the loop that runs all of it | `experiments/online_from_baseline.py` |

The learning rule these wrap — the return-fitted critic and the score-function
actor — is in {doc}`tutorial/learning_algorithm`.
