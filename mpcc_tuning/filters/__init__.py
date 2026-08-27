"""Safety filters: seven implementations, one interface, one comparison.

Every filter here answers the same question -- *may I apply this input?* -- and
they differ only in what they check. Because the interface is identical, the
comparison in ``docs/source/filters.md`` is like for like: same plant, same
controller, same track, same reward.

=========================  ===================  =========================  ==========
class                      certifies by         cost per tick              exact?
=========================  ===================  =========================  ==========
:class:`ASIF`              a backup rollout     ~30 model steps            no (sampled)
:class:`TubeASIF`          rollout per model    ~30 x n_samples            no (sampled)
:class:`AdaptiveTubeASIF`  rollout, tube learnt  ~30 steps                 no (sampled)
:class:`CBFQP`             one inequality       1 model step + 4 for a grad  yes
:class:`CLFCBFQP`          two inequalities     as above + a line search   yes (safety)
:class:`ViabilityFilter`   table lookup         one index                  yes (on grid)
:class:`MPCCSafetyFilter`  an NLP solve         a full NLP                 yes
=========================  ===================  =========================  ==========

Choosing between them
---------------------
* You have a credible backup manoeuvre -> :class:`ASIF`. Cheapest sound option.
* You do not trust your model -> :class:`TubeASIF`, or
  :class:`AdaptiveTubeASIF` if the uncertain parameter is observable.
* You need a hard real-time bound and can write down ``h`` ->
  :class:`CBFQP`. One model step, no horizon.
* The track is fixed and you can afford an offline computation ->
  :class:`ViabilityFilter`. Exact, and the online cost is an array index.
* You already have a tuned MPC and want no second model ->
  :class:`MPCCSafetyFilter`.

Not implemented, and why
------------------------
**Shielding** (Alshiekh, Bloem, Ehlers, Könighofer, Niekum & Topcu, *"Safe
Reinforcement Learning via Shielding"*, AAAI 2018) synthesises a filter from a
temporal-logic specification over a *discrete abstraction* of the system. It is
the right tool when the specification is logical -- orderings, liveness, "never
two of these at once" -- and the state is naturally finite. Here the
specification is "stay between two lines", which is a geometric constraint that
the continuous methods above express directly and exactly, and building a sound
discrete abstraction of a car would introduce more conservatism than the
constraint itself contains. It is listed because it is a real branch of the
field, not because it was tried and rejected on this problem.

**A full HJ reachability solve** (a level-set PDE via ``helperOC`` or
``optimized_dp``). :class:`ViabilityFilter` is a discrete dynamic-programming
stand-in: same fixed point, no numerical Hamiltonian, accuracy set by the grid
rather than by a PDE scheme.
"""

from mpcc_tuning.filters.adaptive import AdaptiveTubeASIF
from mpcc_tuning.filters.asif import ASIF, TubeASIF
from mpcc_tuning.filters.base import SafetyFilter
from mpcc_tuning.filters.cbf_qp import CBFQP, CLFCBFQP
from mpcc_tuning.filters.mpcc_terminal import MPCCSafetyFilter
from mpcc_tuning.filters.reachability import ViabilityFilter

#: Name -> class, for the benchmark and the docs tables.
FILTERS = {
    "asif": ASIF,
    "tube": TubeASIF,
    "adaptive": AdaptiveTubeASIF,
    "cbf": CBFQP,
    "clf_cbf": CLFCBFQP,
    "viability": ViabilityFilter,
    "mpcc_terminal": MPCCSafetyFilter,
}

__all__ = ["SafetyFilter", "ASIF", "TubeASIF", "AdaptiveTubeASIF", "CBFQP",
           "CLFCBFQP", "ViabilityFilter", "MPCCSafetyFilter", "FILTERS"]
