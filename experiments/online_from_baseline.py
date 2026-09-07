"""Does the online tuner improve a baseline that already drives?

    PYTHONPATH=/path/to/scuderia_gym_jax python3 \
        experiments/online_from_baseline.py --episodes 10 --seeds 3

This is the project's stated order, asked for the first time with every
precondition actually met:

1. a fixed parameterisation that simply drives -- :mod:`mpcc_tuning.baselines`
   ``START``, clean on the acados backend with the dynamic drift model;
2. **switch the tuner on from there and see whether it goes faster**, with
   ``BEST`` recorded up front as the hand-tuned target.

Every earlier version of this experiment failed one of those. ``MPCC`` with
``KinematicBicycle`` is not the controller that ships; the old anchor left the
track within half a lap of ICRA T1; and the target sat behind ``q_vref``, a
build constant the policy cannot emit, so most of the "headroom" was
unreachable by construction.

## What is compared

Three lines per track, same seeds, same plant, same OCP:

``fixed``
    theta held at ``START`` for the whole episode, no exploration noise.
``fixed_noise``
    theta held at ``START``, but with the SAME actuator exploration noise the
    tuner applies. This control exists because the first version of this
    experiment did not have it and could not answer the obvious question: the
    tuner perturbs steering and acceleration by 5% of their limits every tick,
    and on a car driving at the grip limit that alone can put it off the
    track. Without this row, "the tuner made it worse" cannot be separated
    from "the exploration noise made it worse", and those call for opposite
    fixes.
``tuner``
    TD(lambda) on the LTC policy, anchored at ``START``.
``BEST``
    a horizontal reference line, not a run: the hand-tuned ceiling.

## Seeds differ physically, not just in an RNG

The plant is deterministic and consults no random stream, so reseeding it
perturbs nothing -- a spread of exactly zero across "repeats" was twice
reported here as agreement between runs. Each seed therefore starts at a
different point on the track and a different entry speed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

from mpcc_tuning import baselines as B  # noqa: E402
from mpcc_tuning.mpcc import WEIGHT_NAMES  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

OUT = ROOT / "results"


def run_frozen(m, pol, t, s0, v0, steps, features, kv_launch=0.0,
               kv_launch_laps=1.0):
    """Drive the CURRENT network with no learning and no noise; one episode.

    ``kv_launch`` (>0) is a **launch schedule on the grip claim**: cap the
    emitted ``k_v`` to this value until the car has covered ``kv_launch_laps``
    laps, then release it to whatever the network asks for. Measured 2026-09-06
    (TODO 2h): the online MPCC-critic net over-claims grip (k_v drifted 0.50 ->
    0.69) and from a COLD standstill plans ~4 m/s into the hairpin, the QP goes
    infeasible, it crashes (1/3 clean). Capping k_v to 0.50 for the launch lap
    then releasing it -> 3/3 clean, mean 2.57, because once moving it is in the
    already-clean flying regime. Unlike a control-loop brake this shapes the
    PLAN (a valid, gentler grip claim), so it does not distort the launch the
    way braking did. Default 0.0 = off, so every existing result is unchanged.
    """
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.mpcc import WEIGHT_NAMES
    ik = WEIGHT_NAMES.index("k_v")
    cap = float(np.log(kv_launch)) if kv_launch and kv_launch > 0 else None
    P = ScuderiaPlant(t, model="std", dt=0.05); P.max_steps = steps
    s5 = P.reset(s0=s0, v0=v0); m.reset(); pol.reset()
    base = float(s5[4]); off = tr = False

    def emit():
        th = np.asarray(pol.step(features(t, s5)), float)
        if cap is not None and (float(s5[4]) - base) < kv_launch_laps * t.length:
            th[ik] = min(th[ik], cap)          # launch: cap the grip claim
        return th

    th = emit()
    for _ in range(steps):
        u = m.value(P.state_dyn(), th)["u0"]
        s5, r, off, tr = P.step(u)
        th = emit()
        if off or tr:
            break
    return (float(s5[4]) - base) / t.length, bool(off)


def one(job):
    """One (track, seed, condition) run. Builds its own solver."""
    (track_name, seed, cond, episodes, steps, alpha, grad, box, factor,
     clock, keep_best, kb_tol, eval_eps, critic, theta_explore, explore,
     validate, init_policy) = job
    learn = cond == "tuner"
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    from mpcc_tuning.ltc import (LTCCell, N_FEATURES, THETA_HI, THETA_LO,
                                 PolicyTuner, WeightPolicy, features)
    from mpcc_tuning.plant_scuderia import ScuderiaPlant

    t = getattr(Track, track_name)()
    st = B.start(track_name)
    th0 = np.asarray(st.theta(), float)
    _use_vref = bool(getattr(t, "use_optimiser_vref", False))
    m = AcadosMPCC(t, horizon=st.horizon, dt=0.05, vehicle="dynamic",
                   q_vref=st.q_vref, theta_global=(grad == "native"),
                   discrete=True, use_track_vref=_use_vref,
                   name=f"onl_{track_name}_{seed}_{int(learn)}")

    from mpcc_tuning.model import ACCEL_MAX, STEER_MAX
    lim = np.array([STEER_MAX, ACCEL_MAX])
    rng = np.random.default_rng(seed)

    tu = None
    if learn:
        # "global" is the search box ltc.py was drawn with -- q_l spans 0.05
        # to 400 -- and the policy walks it to the corners inside one episode.
        # "adapt" is baselines.adaptation_box: at most a factor of `factor`
        # either way, the design the project actually describes.
        if box == "adapt":
            lo, hi = B.adaptation_box(track_name, factor)
        else:
            lo, hi = THETA_LO, THETA_HI
        if _use_vref:
            from mpcc_tuning.ltc import KV_BOUNDS
            from mpcc_tuning.mpcc import WEIGHT_NAMES as _WN
            _ikv = _WN.index("k_v")
            lo = np.array(lo, float).copy(); hi = np.array(hi, float).copy()
            lo[_ikv] = np.log(KV_BOUNDS[0]); hi[_ikv] = np.log(KV_BOUNDS[1])
        init = np.load(init_policy) if init_policy else None
        if init is not None:
            # idea 4: start from the network fitted to the situation grid,
            # inside the data-driven box it was fitted in. theta_0 stays
            # START -- the anchor of the squash -- the network's parameters
            # are what carries the situation-dependence in.
            lo, hi = init["lo"], init["hi"]
        pol = WeightPolicy(LTCCell(N_FEATURES, 12, seed=seed), th0,
                           lo, hi, seed=seed)
        if init is not None:
            pol.G[...] = init["G"]; pol.cell.p[...] = init["cell_p"]
            # The bank must not start empty when we start from a network that
            # already drives: otherwise the first candidate is accepted
            # whatever its score, and a regression from the start gets
            # "banked" (measured: return critic banked 2.30 and 2.34 from a
            # start that drives 2.49 frozen). Seed the incumbent with the
            # init network at its own frozen score, so only an improvement
            # on it can ever be banked and every revert goes back to it.
            init_seeded = True
        else:
            init_seeded = False
        tu = PolicyTuner(m, pol, alpha=alpha, explore=explore, delta_clip=1.0,
                         seed=seed, trust_region=0.01, clock=clock,
                         critic=critic, theta_explore=theta_explore)

    # physical perturbation per seed -- see the module docstring
    s0 = (seed % 4) * (t.length / 4.0)
    v0 = 1.0 + 0.1 * (seed % 3)
    if learn and keep_best and init_seeded:
        with tu.frozen():
            l0, o0 = run_frozen(m, pol, t, s0, v0, steps, features)
        tu._best = (-1.0 if o0 else float(l0), pol.G.copy(), pol.cell.p.copy())
        pol.reset(); m.reset()

    per_ep, wtrace = [], []
    for ep in range(episodes):
        P = ScuderiaPlant(t, model="std", dt=0.05)
        P.max_steps = steps
        s5 = P.reset(s0=s0, v0=v0)
        m.reset()
        base = float(s5[4])
        off = tr = False
        th = th0
        if learn:
            tu.reset()
            th, u = tu.act(features(t, s5), P.state_dyn())
        ep_th = []
        for k in range(steps):
            if not learn:
                u = np.asarray(m.value(P.state_dyn(), th)["u0"], float).copy()
                if cond == "fixed_noise":
                    # identical perturbation to PolicyTuner._explore
                    u[:2] = np.clip(u[:2] + rng.normal(0.0, 0.05, 2) * lim,
                                    -lim, lim)
            s5n, r, off, tr = P.step(u)
            if k % 5 == 0:
                # position and sector alongside theta: the same trace then
                # answers "did it get faster" and "did it use DIFFERENT
                # weights in different parts of the track", which is the
                # situation-dependence claim, and drives the GIF.
                ep_th.append([float(s5n[0]), float(s5n[1]), float(s5n[3]),
                              int(t.sector(t.wrap(float(s5n[4]))))]
                             + np.exp(th).tolist())
            if learn:
                if clock == "progress":
                    # the plant's reward is progress - 5*off; recover the
                    # progress, and charge TIME instead so the return is
                    # minus the lap time (see PolicyTuner on why the reward
                    # must change with the clock)
                    progress = float(r) + (5.0 if off else 0.0)
                    r_learn = -0.05 - (5.0 if off else 0.0)
                    out = tu.learn(r_learn, P.state_dyn(), features(t, s5n),
                                   off, ds=progress)
                else:
                    out = tu.learn(r, P.state_dyn(), features(t, s5n), off)
                if out[0] is None:
                    break
                th, u = out
            s5 = s5n
            if off or tr:
                break
        laps = (float(s5[4]) - base) / t.length
        action, val = "none", None
        if learn and keep_best:
            action = tu.end_episode(laps, crashed=bool(off), tol=kb_tol,
                                    validate=validate)
            if action == "validate":
                # drive the candidate network frozen; bank only if THAT
                # score beats the validated incumbent
                with tu.frozen():
                    val_laps, val_off = run_frozen(m, pol, t, s0, v0, steps,
                                                   features)
                val = dict(laps=val_laps, off=val_off)
                banked = tu.confirm_candidate(-1.0 if val_off else val_laps)
                action = "banked" if banked else "rejected"
        per_ep.append(dict(ep=ep, laps=laps, off=bool(off),
                           theta=np.exp(th).tolist(),
                           reverted=(action == "reverted"), action=action,
                           validation=val,
                           best=(tu.best_score if (learn and keep_best)
                                 else None)))
        wtrace.append(ep_th)
    # "Fix the network after we find a good policy network." Restore the
    # banked best, switch off learning AND exploration, and drive it for
    # eval_eps more episodes. This is the deliverable the user described: not
    # a weight vector, a policy network that is then held. If it does not hold
    # here, keep-best found a fluke rather than a policy.
    evals = []
    if learn and keep_best and eval_eps > 0 and tu.best_score is not None:
        _, G, cp = tu._best
        pol.G[...] = G; pol.cell.p[...] = cp
        with tu.frozen():
            for k in range(eval_eps):
                l_, o_ = run_frozen(m, pol, t, s0, v0, steps, features)
                evals.append(dict(laps=l_, off=o_))
        # keep the network itself, so it can be reloaded and driven again
        # the critic and the box are part of the identity of a banked network:
        # two runs that differed only in the critic overwrote each other here
        # EVERYTHING needed to drive this network again must be in the file:
        # the readout and cell parameters are useless without the box they
        # were trained in (the squash spans depend on it) and the LTC seed
        # (the cell's fixed structure depends on it). Measured without them:
        # the 2.99-lap network re-driven with the default box and seed 0 gave
        # 1.31x / 2.89 / 0.81x -- on its own start it left the track.
        np.savez(str(OUT / f"best_policy_{track_name}_{seed}_{clock}_{critic}"
                     f"{'_val' if validate else ''}.npz"),
                 G=G, cell_p=cp, best_laps=tu.best_score, critic=critic,
                 clock=clock, box=box, factor=factor, validated=validate,
                 lo=np.asarray(pol.lo, float), hi=np.asarray(pol.hi, float),
                 theta0=th0, seed=seed, hidden=12,
                 init_policy=str(init_policy or ""))
    return (track_name, seed, cond), per_ep, wtrace, evals


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="*", default=list(B.TRACKS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=10)
    # 0 = take each track's own verified step budget from baselines,
    # rather than one number that is too short for one track and wasteful
    # for the other
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=2e-3)
    ap.add_argument("--grad", choices=("envelope", "native"),
                    default="envelope")
    ap.add_argument("--box", choices=("global", "adapt"), default="global",
                    help="policy output box: the global search box, or a "
                         "small adaptation box around theta_0")
    ap.add_argument("--factor", type=float, default=2.0,
                    help="for --box adapt: max multiplicative move per weight")
    ap.add_argument("--clock", choices=("time", "progress"), default="time",
                    help="what advances the learner: control ticks, or metres "
                         "of real progress (semi-Markov TD, reward = -time)")
    ap.add_argument("--keep-best", action="store_true",
                    help="bank the policy network at the best episode and "
                         "revert to it when an episode is worse by more than "
                         "--keep-best-tol laps, or crashes")
    ap.add_argument("--critic", choices=("mpcc", "return", "fitted"), default="mpcc",
                    help="where the learner's value estimate comes from. The MPCC "
                         "drives the car in BOTH cases. mpcc: V = -J*, the "
                         "controller's own optimal cost, envelope gradient. "
                         "return: a linear critic on the policy's features fitted "
                         "to the measured return, actor by theta-exploration "
                         "('fitted' is an alias)")
    ap.add_argument("--theta-explore", type=float, default=0.0,
                    help="sigma of Gaussian noise on theta, log space; the "
                         "fitted critic needs it > 0 (0.1 ~ 10%% jitter)")
    ap.add_argument("--explore", type=float, default=0.05,
                    help="actuator exploration as a fraction of the input "
                         "limits; measured to cost 0.28 laps on T2")
    ap.add_argument("--validate", action="store_true",
                    help="keep-best banks a candidate only after a FROZEN "
                         "validation episode beats the incumbent")
    ap.add_argument("--init-policy", default=None,
                    help="path to a fitted_policy_*.npz from "
                         "scripts/fit_policy_to_grid.py: start the learner "
                         "from that network and its box (idea 4)")
    ap.add_argument("--eval-episodes", type=int, default=0,
                    help="after learning, drive the banked best network "
                         "FROZEN (no learning, no exploration) for this many "
                         "episodes -- the 'fix the network' step")
    ap.add_argument("--keep-best-tol", type=float, default=0.15,
                    help="drop in laps that counts as worse (the fixed "
                         "controller's seed spread is about 0.12)")
    ap.add_argument("--conditions", nargs="*",
                    default=["fixed", "fixed_noise", "tuner"],
                    choices=("fixed", "fixed_noise", "tuner"))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default="online_from_baseline.json",
                    help="filename under results/; use a distinct name when "
                         "running a subset of conditions to be merged later")
    a = ap.parse_args(argv)

    ap_conds = a.conditions
    jobs = [(t, s, c, a.episodes, a.steps or B.start(t).steps, a.alpha, a.grad,
             a.box, a.factor, a.clock, a.keep_best, a.keep_best_tol,
             a.eval_episodes, a.critic, a.theta_explore, a.explore,
             a.validate, a.init_policy)
            for t in a.tracks for s in range(a.seeds) for c in ap_conds]
    res, traces, eval_res = {}, {}, {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            key, per_ep, wt, evals = f.result()
            res[key] = per_ep
            traces["|".join(map(str, key))] = wt
            if evals:
                eval_res["|".join(map(str, key))] = evals
            last = per_ep[-1]
            nrev = sum(1 for e in per_ep if e.get("reverted"))
            nval = sum(1 for e in per_ep if e.get("validation"))
            nbank = sum(1 for e in per_ep if e.get("action") == "banked")
            ev = ("  FROZEN best: " + " ".join(
                f"{e['laps']:.2f}{'x' if e['off'] else ''}" for e in evals)
                  if evals else "")
            print("  [%2d/%2d] %-18s seed %d %-11s  last ep %.2f laps%s%s%s"
                  % (i + 1, len(futs), key[0], key[1], key[2], last["laps"],
                     " OFF" if last["off"] else "",
                     f"  best {last['best']:.2f}, {nrev} reverts, "
                     f"{nbank}/{nval} validations banked"
                     if last.get("best") is not None else "", ev), flush=True)

    print()
    print("  Laps, mean of the last three episodes. START and BEST are the")
    print("  recorded hand-tuned anchors; the tuner has to beat FIXED, which")
    print("  is START held constant on the same seeds.")
    hdr = "  %-18s %8s" + " %13s" * len(ap_conds) + " %8s"
    print(hdr % (("track", "START") + tuple(ap_conds) + ("BEST",)))
    summary = {}
    for t in a.tracks:
        def tail(c):
            v = [np.mean([e["laps"] for e in res[(t, s, c)][-3:]])
                 for s in range(a.seeds)]
            return float(np.mean(v)), float(np.std(v))
        got = {c: tail(c) for c in ap_conds}
        st, bs = B.start(t).laps, B.best(t).laps
        summary[t] = dict(start=st, best=bs,
                          **{c: got[c][0] for c in ap_conds},
                          **{c + "_sd": got[c][1] for c in ap_conds})
        row = "  %-18s %8.2f" % (t, st)
        for c in ap_conds:
            row += " %8.2f+-%.2f" % got[c]
        row += " %8.2f" % bs
        print(row)
    print()
    print("  fixed_noise isolates the exploration noise from the learning:")
    print("  if fixed_noise ~ tuner, the noise did it, not the policy.")
    print("  A difference smaller than the spread across seeds is not a result.")

    OUT.mkdir(exist_ok=True)
    p = OUT / a.out
    p.write_text(json.dumps(dict(
        summary=summary, weight_names=list(WEIGHT_NAMES),
        episodes={"|".join(map(str, k)): v for k, v in res.items()},
        traces=traces, evals=eval_res, config=vars(a)), indent=1))
    if eval_res:
        print()
        print("  Banked best network, driven FROZEN (no learning, no noise):")
        for k, ev in sorted(eval_res.items()):
            v = [e["laps"] for e in ev]
            print("    %-30s %s  -> %.2f +- %.2f%s"
                  % (k, " ".join(f"{e['laps']:.2f}{'x' if e['off'] else ' '}" for e in ev),
                     np.mean(v), np.std(v),
                     "" if not any(e["off"] for e in ev) else "  (crashes)"))
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
