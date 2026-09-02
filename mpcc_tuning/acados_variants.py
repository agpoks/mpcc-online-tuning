"""Solver variants for the acados backend, as named reproducible configurations.

Every acados setting that has been *measured* to matter lives here as a named
variant rather than as a scattered keyword argument, so a comparison is a list
of names and a result table rather than a set of edits nobody can reproduce.

The hypothesis this exists to test:

    a newer acados with feasible-QP restoration should allow a HARD physical
    corridor -- no permanent slack on the track boundary -- and the local
    half-space representation may additionally improve real-time performance.

## Why the corridor keeps coming up

`mpcc_tuning/mpcc.py` holds the corridor hard and drives 5.12 laps of the oval.
Every attempt to hold it hard in acados v0.1.9 collapses the solve rate to
9-25% and the car stops moving, whether the row is written nonlinearly (from
the spline, at the state) or linearised into half-spaces. The difference is
that IPOPT restores feasibility when a nonlinear row is violated at the
linearisation point and acados v0.1.9 has no equivalent -- its
``nlp_solver_type`` accepts only ``SQP`` and ``SQP_RTI``.

``SQP_WITH_FEASIBLE_QP`` arrived in acados v0.4.5 and is exactly that missing
piece. Testing it needs a newer build, which is why :data:`ACADOS_V053` exists.

## What has already been measured, on the dynamic model

Do not re-run these hoping for a different answer; re-run them to check a
change has not regressed something.

    variant                             laps   solve   ms/tick (worst)
    baseline (soft, nonlinear, it=8)    3.40    77%     6.7  (45.2)
    linear half-space corridor          1.12    91%     5.0  (21.4)
    hpipm ROBUST + qp_warm_start=2      0.04     5%     2.2  ( 5.3)
    hard corridor (any of the above)    0.27    25%     2.3  ( 7.2)
    max_iter 1 / 2 / 3                  0.04-0.06  0-5%

``ROBUST`` and ``qp_solver_warm_start=2`` are carried in this file only as a
NAMED, KNOWN-BAD variant, because they came from a working setup elsewhere and
the surprise is worth recording: they cost 3.40 laps and return 0.04.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

#: Where an acados new enough for ``SQP_WITH_FEASIBLE_QP`` is installed. The
#: v0.1.9 tree the rest of this machine uses is deliberately left alone: this
#: is a separate worktree and a separate install prefix, selected purely
#: through the environment, so nothing that works today stops working.
ACADOS_V060 = dict(
    ACADOS_SOURCE_DIR=os.path.expanduser("~/acados-v060"),
    LD_LIBRARY_PATH=os.path.expanduser("~/acados-v060/lib"),
    PYTHONPATH=os.path.expanduser("~/acados-v060/interfaces/acados_template"),
)

ACADOS_V053 = dict(
    ACADOS_SOURCE_DIR=os.path.expanduser("~/acados-v053"),
    LD_LIBRARY_PATH=os.path.expanduser("~/acados-v053/lib"),
    PYTHONPATH=os.path.expanduser("~/acados-v053/interfaces/acados_template"),
)


#: CasADi rejects a function name that is not ``[A-Za-z][A-Za-z0-9_]*`` with
#: NO consecutive underscores, and also rejects ``null``, ``jac`` and ``hess``.
#: The variant name reaches CasADi through the generated model functions
#: (``<name>_expl_ode_fun`` and friends), so an invalid name fails every build
#: in a sweep with a message about SXFunction that says nothing about naming.
#: This has voided two full runs already -- once from "," and "(" in trial
#: names, once from a "__" separator -- so sanitising is done here rather than
#: relied upon at each call site.
_RESERVED = ("null", "jac", "hess")


def sanitize_name(name: str, fallback: str = "ocp") -> str:
    """Coerce ``name`` into a valid CasADi/acados function identifier."""
    import re
    s = re.sub(r"[^0-9A-Za-z_]", "_", str(name))
    s = re.sub(r"_+", "_", s).strip("_")          # no consecutive underscores
    if not s or not s[0].isalpha():
        s = fallback + ("_" + s if s else "")
    s = re.sub(r"_+", "_", s).strip("_")
    return s + "_x" if s.lower() in _RESERVED else s


@dataclass(frozen=True)
class Variant:
    """One reproducible solver configuration.

    ``needs`` records the minimum acados version, so a variant that cannot run
    on the installed build is reported as skipped rather than silently falling
    back to something else and being compared as though it were the same thing.
    """

    name: str
    nlp_solver_type: str = "SQP"
    nlp_solver_max_iter: int = 8
    hpipm_mode: str = "BALANCE"
    #: FIXED_STEP takes the full SQP step with no line search.
    #: MERIT_BACKTRACKING and FUNNEL_L1PEN_LINESEARCH reject and
    #: shorten steps that do not improve a merit/funnel measure --
    #: which is what IPOPT does and acados by default does not.
    globalization: str = "FIXED_STEP"
    qp_solver_warm_start: int = 1
    qp_solver_iter_max: int = 200
    #: False = the track boundary is a HARD constraint, as in mpcc.py.
    soft_corridor: bool = True
    #: True = corridor as linear half-spaces from per-stage parameters.
    lin_corridor: bool = False
    #: Newer builds only; ignored where unsupported.
    timeout_max_time: float | None = None
    timeout_heuristic: str | None = None
    warm_start_first_qp: bool | None = None
    needs: str = "0.1.9"
    note: str = ""

    def apply(self, ocp) -> list[str]:
        """Set what this build supports; return what it could not."""
        so = ocp.solver_options
        so.nlp_solver_type = self.nlp_solver_type
        if self.nlp_solver_type != "SQP_RTI":
            so.nlp_solver_max_iter = self.nlp_solver_max_iter
        so.hpipm_mode = self.hpipm_mode
        so.qp_solver_warm_start = self.qp_solver_warm_start
        so.qp_solver_iter_max = self.qp_solver_iter_max
        missing = []
        if hasattr(so, "globalization"):
            try:
                so.globalization = self.globalization
            except Exception:
                missing.append(f"globalization={self.globalization}")
        elif self.globalization != "FIXED_STEP":
            missing.append("globalization")
        for attr, val in (("timeout_max_time", self.timeout_max_time),
                          ("timeout_heuristic", self.timeout_heuristic),
                          ("nlp_solver_warm_start_first_qp",
                           self.warm_start_first_qp)):
            if val is None:
                continue
            if hasattr(so, attr):
                setattr(so, attr, val)
            else:
                missing.append(attr)
        return missing

    def build_kwargs(self) -> dict:
        """The parts that belong to :func:`mpcc_tuning.acados_ocp.build_ocp`."""
        return dict(soft_corridor=self.soft_corridor,
                    lin_corridor=self.lin_corridor)

    def as_row(self) -> dict:
        return asdict(self)


#: The measured best on acados v0.1.9, and the reference every variant below is
#: compared against. 3.40 laps, 77% solve, 6.7 ms/tick.
BASELINE = Variant(
    name="baseline",
    note="soft nonlinear corridor, SQP(8). Best on v0.1.9.",
)

VARIANTS: tuple[Variant, ...] = (
    BASELINE,

    # -- the hypothesis ----------------------------------------------------
    Variant(name="feasible_qp_hard", nlp_solver_type="SQP_WITH_FEASIBLE_QP",
            soft_corridor=False, needs="0.4.5",
            note="THE TEST: hard physical corridor, no permanent slack, with "
                 "feasible-QP restoration to survive violations at the "
                 "linearisation point."),
    Variant(name="feasible_qp_hard_lin", nlp_solver_type="SQP_WITH_FEASIBLE_QP",
            soft_corridor=False, lin_corridor=True, needs="0.4.5",
            note="the same, with the corridor as linear half-spaces -- "
                 "exactly representable in the QP."),
    Variant(name="feasible_qp_soft", nlp_solver_type="SQP_WITH_FEASIBLE_QP",
            soft_corridor=True, needs="0.4.5",
            note="control: is the gain the feasible QP, or just hardness?"),
    Variant(name="feasible_qp_rti", nlp_solver_type="SQP_WITH_FEASIBLE_QP",
            nlp_solver_max_iter=1, soft_corridor=False, needs="0.4.5",
            note="RTI-like: one iteration, hard corridor, real-time budget."),
    Variant(name="feasible_qp_timeout", nlp_solver_type="SQP_WITH_FEASIBLE_QP",
            nlp_solver_max_iter=2, soft_corridor=False,
            timeout_max_time=0.010, timeout_heuristic="LAST", needs="0.4.5",
            note="bounded wall-clock per tick, which is what the car needs."),

    # -- already measured on v0.1.9, kept so a regression is visible -------
    Variant(name="linear_corridor", lin_corridor=True,
            note="MEASURED: solve 77 -> 91%, worst tick 45 -> 21 ms, but "
                 "1.12 laps against 3.40. Re-linearised once per tick."),
    Variant(name="hard_corridor_v019", soft_corridor=False,
            note="MEASURED BAD on v0.1.9: 0.27 laps, 25% solve, car does not "
                 "move. This is the failure the hypothesis expects to fix."),
    Variant(name="robust_warmstart", hpipm_mode="ROBUST",
            qp_solver_warm_start=2, qp_solver_iter_max=50,
            note="MEASURED BAD: 0.04 laps, 5% solve. Carried from a working "
                 "setup elsewhere; does not transfer."),
    Variant(name="rti", nlp_solver_type="SQP_RTI",
            note="MEASURED BAD: one Newton step cannot track this problem."),
)

#: The globalization test. Four configurations x three globalization
#: strategies, everything else held fixed.
#:
#: The question: is the remaining gap to IPOPT (5.12 laps against ~2.2) caused
#: by acados ACCEPTING POOR FULL SQP STEPS on a hard-constrained NLP, rather
#: than by too few iterations? Iterations are already ruled out -- 8 gives 3.16
#: laps, 50 gives 3.06, 300 gives 2.43, so more solving is not the answer.
#:
#: acados defaults to FIXED_STEP: it takes the full Newton step with no line
#: search and no acceptance test. IPOPT does not -- it runs a filter line
#: search and will reject a step that does not improve. On a problem with a
#: hard corridor, a full step that violates it is exactly the step that should
#: be rejected.
_GLOBALIZATIONS = ("FIXED_STEP", "MERIT_BACKTRACKING", "FUNNEL_L1PEN_LINESEARCH")

_GLOB_BASE = (
    ("soft_sqp", dict(nlp_solver_type="SQP", soft_corridor=True)),
    ("fqp_soft", dict(nlp_solver_type="SQP_WITH_FEASIBLE_QP", soft_corridor=True,
                      needs="0.4.5")),
    ("fqp_hard", dict(nlp_solver_type="SQP_WITH_FEASIBLE_QP", soft_corridor=False,
                      needs="0.4.5")),
    ("fqp_hard_lin", dict(nlp_solver_type="SQP_WITH_FEASIBLE_QP",
                          soft_corridor=False, lin_corridor=True, needs="0.4.5")),
)

#: Single underscore only: CasADi function names may not contain CONSECUTIVE
#: underscores, and the variant name reaches CasADi through the generated
#: model function names. A "__" separator silently broke all twelve builds.
GLOBALIZATION_VARIANTS: tuple[Variant, ...] = tuple(
    Variant(name=f"{base}_{g.split('_')[0].lower()}", globalization=g, **kw)
    for base, kw in _GLOB_BASE for g in _GLOBALIZATIONS
)

VARIANTS = VARIANTS + GLOBALIZATION_VARIANTS

BY_NAME = {v.name: v for v in VARIANTS}


def capabilities() -> dict:
    """What the INSTALLED acados can actually do.

    Asked of the build rather than parsed from a version string: v0.1.9 does
    not set ``acados_template.__version__`` at all, so a version comparison
    silently reports 0.0.0 and marks everything unsupported. Feature detection
    cannot drift out of date the way a version table can.
    """
    caps = {"solver_types": (), "timeout": False, "warm_start_first_qp": False,
            "version": "unknown", "source": ""}
    try:
        import acados_template
        from acados_template import AcadosOcp
        caps["version"] = getattr(acados_template, "__version__", "unknown")
        caps["source"] = getattr(acados_template, "__file__", "")
        o = AcadosOcp()
        found = []
        for t in ("SQP", "SQP_RTI", "DDP", "SQP_WITH_FEASIBLE_QP"):
            try:
                o.solver_options.nlp_solver_type = t
                found.append(t)
            except Exception:
                pass
        caps["solver_types"] = tuple(found)
        caps["timeout"] = hasattr(o.solver_options, "timeout_max_time")
        caps["warm_start_first_qp"] = hasattr(
            o.solver_options, "nlp_solver_warm_start_first_qp")
    except Exception:
        pass
    return caps


def supported(variant: Variant, caps: dict | None = None) -> bool:
    """Can ``variant`` run on the installed build?"""
    caps = capabilities() if caps is None else caps
    return variant.nlp_solver_type in caps["solver_types"]
