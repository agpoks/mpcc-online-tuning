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


def one(job):
    """One (track, seed, condition) run. Builds its own solver."""
    (track_name, seed, cond, episodes, steps, alpha, grad, box, factor,
     clock) = job
    learn = cond == "tuner"
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    from mpcc_tuning.ltc import (LTCCell, N_FEATURES, THETA_HI, THETA_LO,
                                 PolicyTuner, WeightPolicy, features)
    from mpcc_tuning.plant_scuderia import ScuderiaPlant

    t = getattr(Track, track_name)()
    st = B.start(track_name)
    th0 = np.asarray(st.theta(), float)
    m = AcadosMPCC(t, horizon=st.horizon, dt=0.05, vehicle="dynamic",
                   q_vref=st.q_vref, theta_global=(grad == "native"),
                   discrete=True, name=f"onl_{track_name}_{seed}_{int(learn)}")

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
        pol = WeightPolicy(LTCCell(N_FEATURES, 12, seed=seed), th0,
                           lo, hi, seed=seed)
        tu = PolicyTuner(m, pol, alpha=alpha, explore=0.05, delta_clip=1.0,
                         seed=seed, trust_region=0.01, clock=clock)

    # physical perturbation per seed -- see the module docstring
    s0 = (seed % 4) * (t.length / 4.0)
    v0 = 1.0 + 0.1 * (seed % 3)

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
        per_ep.append(dict(ep=ep, laps=laps, off=bool(off),
                           theta=np.exp(th).tolist()))
        wtrace.append(ep_th)
    return (track_name, seed, cond), per_ep, wtrace


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
             a.box, a.factor, a.clock)
            for t in a.tracks for s in range(a.seeds) for c in ap_conds]
    res, traces = {}, {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            key, per_ep, wt = f.result()
            res[key] = per_ep
            traces["|".join(map(str, key))] = wt
            last = per_ep[-1]
            print("  [%2d/%2d] %-18s seed %d %-11s  last ep %.2f laps%s"
                  % (i + 1, len(futs), key[0], key[1], key[2], last["laps"],
                     " OFF" if last["off"] else ""), flush=True)

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
        traces=traces, config=vars(a)), indent=1))
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
