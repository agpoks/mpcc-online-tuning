"""What acados actually reports, and what counts as a usable control step.

Treating ``status != 0`` as a failed MPC step is wrong, and it made several
measurements in this repo meaningless. acados returns:

    0  SUCCESS          converged to tolerance
    1  NAN_DETECTED     the iterate is not a number
    2  MAX_ITER         hit the iteration cap
    3  MIN_STEP         line search could not make progress
    4  QP_FAILURE       the QP solver failed
    5  READY            solver created, not yet solved

**MAX_ITER is the normal outcome for a bounded-iteration controller.** An RTI
step takes one Newton step and stops; it reports 2 every tick and its control
is perfectly usable -- that is the entire premise of real-time iteration. A
variant configured with ``nlp_solver_max_iter = 1`` was being scored 0% solve
while driving 1.74 laps, which is not a measurement of anything.

What actually matters for a controller is narrower and more physical:

* did a **finite, in-bounds control** come back, and
* is the predicted trajectory **inside the corridor**.

A solve that hit its iteration cap and returned a sane control did its job. A
solve that returned SUCCESS with the plan leaving the track did not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: acados return codes, by number.
STATUS = {0: "SUCCESS", 1: "NAN_DETECTED", 2: "MAX_ITER", 3: "MIN_STEP",
          4: "QP_FAILURE", 5: "READY"}

#: Statuses where the returned iterate cannot be trusted at all. MAX_ITER and
#: MIN_STEP are deliberately NOT here: both return the best iterate found.
FATAL = (1, 4)


@dataclass
class StepLog:
    """One control tick, recorded in enough detail to diagnose it later."""

    status: int
    qp_status: int
    sqp_iters: int
    qp_iters: int
    u: np.ndarray
    solve_ms: float
    #: Worst corridor violation over the predicted horizon, in metres. Zero
    #: when the whole plan stays inside.
    corridor_violation: float = 0.0
    #: Accepted step length. 1.0 means the full SQP step was taken;
    #: anything less means globalization rejected it and backtracked.
    alpha: float = 1.0
    #: True when a finite, in-bounds control came back from a non-fatal solve.
    usable: bool = False


@dataclass
class RunLog:
    """Every tick of one closed-loop run, plus the summaries that matter."""

    steps: list = field(default_factory=list)

    def add(self, step: StepLog) -> None:
        self.steps.append(step)

    # -- the summaries ---------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.steps)

    @property
    def usable_pct(self) -> float:
        """Ticks that produced a control the car could act on."""
        return 100.0 * np.mean([s.usable for s in self.steps]) if self.n else 0.0

    @property
    def converged_pct(self) -> float:
        """Ticks that reached SUCCESS. Informative, not a pass/fail."""
        return 100.0 * np.mean([s.status == 0 for s in self.steps]) if self.n else 0.0

    @property
    def nlp_failures(self) -> int:
        return int(sum(s.status in FATAL for s in self.steps))

    @property
    def qp_failures(self) -> int:
        return int(sum(s.qp_status != 0 for s in self.steps))

    @property
    def max_iter_pct(self) -> float:
        return 100.0 * np.mean([s.status == 2 for s in self.steps]) if self.n else 0.0

    @property
    def corridor_violation_max(self) -> float:
        return float(max((s.corridor_violation for s in self.steps), default=0.0))

    @property
    def corridor_violation_mean(self) -> float:
        return float(np.mean([s.corridor_violation for s in self.steps])) if self.n else 0.0

    @property
    def alpha_mean(self) -> float:
        """Mean accepted step length. Below 1.0 means globalization is
        rejecting full steps, which is the whole point of enabling it."""
        return float(np.mean([s.alpha for s in self.steps])) if self.n else 1.0

    @property
    def rejected_pct(self) -> float:
        """Ticks where a full step was NOT accepted."""
        return (100.0 * np.mean([s.alpha < 0.999 for s in self.steps])
                if self.n else 0.0)

    def timing(self) -> dict:
        """Mean and tail of the per-tick solve time. The tail is what a real
        controller misses its deadline on, so p95/p99/max are reported and not
        summarised away."""
        t = np.array([s.solve_ms for s in self.steps]) if self.n else np.zeros(1)
        return dict(mean=float(t.mean()), p95=float(np.percentile(t, 95)),
                    p99=float(np.percentile(t, 99)), max=float(t.max()))

    def status_counts(self) -> dict:
        out = {}
        for s in self.steps:
            out[STATUS.get(s.status, str(s.status))] = \
                out.get(STATUS.get(s.status, str(s.status)), 0) + 1
        return out


def read_stats(solver) -> tuple[int, int]:
    """``(qp_status, sqp_iters)`` from an acados solver, tolerantly.

    Field names and shapes differ across acados versions, so every read is
    guarded: a missing statistic must not take down a run whose numbers are
    otherwise fine.
    """
    def one(field, default=0, cast=int):
        try:
            v = np.asarray(solver.get_stats(field)).ravel()
            return cast(v[-1]) if v.size else default
        except Exception:
            return default
    return one("qp_stat"), one("sqp_iter")


def read_alpha(solver) -> float:
    """Accepted step length, 1.0 when the full step was taken."""
    try:
        v = np.asarray(solver.get_stats("alpha")).ravel()
        return float(v[-1]) if v.size else 1.0
    except Exception:
        return 1.0


def evaluate(solver, status: int, u, solve_ms: float, u_lb=None, u_ub=None,
             corridor=None) -> StepLog:
    """Turn one solve into a :class:`StepLog`, deciding whether it was usable.

    ``corridor`` is ``(e_c_over_horizon, margin)`` when the caller can supply
    it; the violation is reported in metres beyond the boundary.
    """
    qp_status, sqp_iters = read_stats(solver)
    u = np.asarray(u, float).ravel()
    finite = bool(np.isfinite(u).all())
    in_bounds = True
    if finite and u_lb is not None and u_ub is not None:
        in_bounds = bool((u >= np.asarray(u_lb) - 1e-6).all()
                         and (u <= np.asarray(u_ub) + 1e-6).all())
    viol = 0.0
    if corridor is not None:
        e_c, margin = corridor
        e_c = np.asarray(e_c, float)
        if e_c.size:
            viol = float(max(0.0, np.max(np.abs(e_c)) - margin))
    return StepLog(status=int(status), qp_status=qp_status,
                   sqp_iters=sqp_iters, qp_iters=0, u=u, solve_ms=solve_ms,
                   corridor_violation=viol, alpha=read_alpha(solver),
                   usable=finite and in_bounds and int(status) not in FATAL)
