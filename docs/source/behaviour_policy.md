# The weight policy

A small recurrent network maps the *situation* to the MPCC weights, afresh every
tick. This page is what that network is, why it is recurrent, and the honest
scope of the recurrence — which is a justified design choice, not the paper's
contribution.

## Situation → weights

Let $\vect f$ be a feature vector describing the situation ahead: curvature
preview at several distances, speed as a fraction of the cap, the margin to the
boundary, the time-to-collision and gap to the car ahead, the lateral
acceleration a pass would need given the grip left after cornering, a soft
membership of the sector ahead (straight, long curve, 90°, hairpin), the corridor
width in car widths, and a one-hot opponent class (static, slower, equal, faster)
graded by *relative* speed. A policy $\pi_\phi$ maps $\vect f$ to the weight
vector $\vect\theta$.

The policy does not emit the weights freely: it is anchored at a verified
baseline and squashed into a measured box, then its safety weights are clamped by
class-conditioned bands. That construction is {doc}`safety`; here the policy is
the thing those bounds wrap.

```{admonition} The geometric half of the decision is linear — and measured
:class: tip
Overtake-vs-follow is set by the ratio $q_v/q_c$: every weight setting with
$q_v/q_c \le 1$ follows, every setting with $q_v/q_c > 1$ passes — 15 of 15
across two decades of each weight ({doc}`obstacles`). Since $\vect\theta$ is the
*log* weights, that boundary is a **difference of two components of
$\vect\theta$** — linear. So a strong fixed or feed-forward policy already
expresses the behaviour *choice*. The recurrence is not for the choice.
```

## Why recurrent: tracking through occlusion

The one quantity a single frame cannot give is what a *closing* or *occluded*
opponent is doing. Traffic is observed through a realistic $270^\circ$ sensor
with car-height occlusion; on the twisty circuit an opponent is unseen 80–90% of
the time. A rival sitting in the ego car's blind arc has to be **remembered**:
its position over time, not its position now, is what says whether a pass is on.
That is a temporal pattern, and it is the role the recurrent state fills.

The state is produced by a **liquid-time-constant (LTC) cell**. Its effective
time constant

$$\tau_{\mathrm{eff}} = \frac{\tau}{1 + \tau\,f(\vect x, \vect h)}$$

is a **run-time hold duration set by what is happening** — long when the gate is
quiet, short when the input drives it. A behaviour decision against traffic
should be taken once and *held*, not re-decided at 20 Hz, and the LTC makes that
hold duration a learned, input-dependent quantity rather than a guessed cap. The
same denominator keeps the online influence estimate from diverging without an
arbitrary clip (the sensitivity leak is structurally below one) — see
{doc}`influence_through_a_solver` for where that $\tau$-as-leak idea came from.

## The honest scope of the claim

```{admonition} Memory is a design choice justified by ablation, not the contribution
:class: important
The paper rests the recurrence claim on **pass completion**, the quantity stable
from learning to deployment. On the equal-pace occluded opponent the LTC finishes
0.89 of passes once frozen against the MLP's 0.67; it keeps tracking the car
through the blind stretches and clears it where the memoryless net loses it and
stalls. On static, faster and slower traffic the two arms pass at the same rate,
so the advantage is *equal-pace specific*. The memoryless MLP is competitive as a
single net and slightly cleaner at three cars — so the policy class is a choice,
reported with its ablation, not the claim.
```

This is a deliberately narrower statement than "a recurrent network chooses the
behaviour," and it is the one the evidence supports. It also reverses an earlier
prediction on this page.

```{admonition} What an earlier draft predicted, and why it was wrong
:class: note
An earlier version of this page, reading a benchmark where an LTC head came sixth
of seven cells, concluded the honest prior was that recurrence *would not* beat a
memoryless MLP here. That prediction was made against an **instantaneous**
observation, where it was correct — the LTC tied the MLP. What changed the
outcome was enriching the observation with the quantities occlusion makes
temporal (slip, closing rate, opponent speed, per-sector map): with those, the
LTC beats the MLP on the equal-pace occluded opponent, and a memory-driven crowd
rule (open the line once the full field has been counted through occlusion)
recovers the three-car case a memoryless policy cannot express. The gate — *only
claim the network if it beats a fixed schedule and a per-tick MLP* — was the
right gate; the observation was what had to change for it to pass.
```

## Where this lives

| idea | where |
|---|---|
| the LTC cell, forward step and analytic leak | `mpcc_tuning/ltc.py` |
| squash, anchor, class residual | `mpcc_tuning/ltc.py::WeightPolicy` |
| the features $\vect f$ | the observation builder in the race/qualify experiments |
| the memoryless ablation (MLP on identical features) | the `--arm mlp` path |

The learning rule that updates the policy online — the return-fitted critic and
the score-function actor carried by RTRL — is {doc}`tutorial/learning_algorithm`.
