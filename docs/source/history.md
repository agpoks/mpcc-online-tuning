# How we got here — the exploration log

The pages in this section document the **earlier spike** and the dead ends that
shaped the shipped method. They are kept, not deleted, because every one of them
is a measured result, and because the current design is best understood as the
answer to what they found. Read them as history: where a page here contradicts
{doc}`the method <index>`, the method is current and the page records what we
tried on the way.

Three things changed between the spike and the shipped method, and each has a
page below that explains the change rather than hiding it:

| the spike did | the method does | why |
|---|---|---|
| critic $V=-J^*$ with the **envelope-theorem** weight gradient (free, exact) | a **return-fitted** critic + **score-function** actor | $J^*$ is linear in the weights, so $\partial J^*/\partial\theta$ points the same way in every situation — it collapses the opponent classes onto one behaviour and drifts into over-claiming grip. See {doc}`spike_results` and {doc}`spike_formulation`. |
| a **predictive safety filter** around the tuner (braking backup) | an **anchored box + class-conditioned bands + validated keep-best** | the filter converted a crash into a slow degradation — it absorbed the off-track event so nothing discouraged the tuner from pushing $q_c$ to zero. The fix had to constrain the *weights*, before the OCP. See {doc}`safety_filter_for_the_tuner` and {doc}`filters`. |
| **IPOPT** to convergence on a **kinematic bicycle** | **acados** SQP-RTI C code on a **dynamic drift** plant | real-time on the car, and a model mismatch worth studying rather than one hand-written line. |

One page here is not a dead end but an un-taken road: {doc}`influence_through_a_solver`
asks whether the solver's warm-start memory needs its own influence term, and
answers — with the measurement — that for a real-time iteration it does not. It
is kept because the *measurement* is the contribution.

```{admonition} The load-bearing lesson, repeated across these pages
:class: important
A wrong safety mechanism **intervenes less, not more.** Every filter bug in this
log showed up as a reassuringly low intervention rate; the grip-over-claim in
the shelved critic showed up as a confident, fast policy that then left the
track. Reading a cost metric as a safety metric inverts the sign — which is why
the shipped method measures safety by what it *prevents*, not by what it costs.
```
