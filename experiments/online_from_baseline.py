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
    theta held at ``START`` for the whole episode. The control. Any difference
    the tuner shows has to beat this, not beat nothing.
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
    track_name, seed, learn, episodes, steps, alpha, grad = job
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

    tu = None
    if learn:
        pol = WeightPolicy(LTCCell(N_FEATURES, 12, seed=seed), th0,
                           THETA_LO, THETA_HI, seed=seed)
        tu = PolicyTuner(m, pol, alpha=alpha, explore=0.05, delta_clip=1.0,
                         seed=seed, trust_region=0.01)

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
                u = m.value(P.state_dyn(), th)["u0"]
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
    return (track_name, seed, bool(learn)), per_ep, wtrace


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
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)

    jobs = [(t, s, learn, a.episodes, a.steps or B.start(t).steps, a.alpha,
             a.grad)
            for t in a.tracks for s in range(a.seeds) for learn in (False, True)]
    res, traces = {}, {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            key, per_ep, wt = f.result()
            res[key] = per_ep
            traces["|".join(map(str, key))] = wt
            last = per_ep[-1]
            print("  [%2d/%2d] %-18s seed %d %-5s  last ep %.2f laps%s"
                  % (i + 1, len(futs), key[0], key[1],
                     "tuner" if key[2] else "fixed", last["laps"],
                     " OFF" if last["off"] else ""), flush=True)

    print()
    print("  Laps, mean of the last three episodes. START and BEST are the")
    print("  recorded hand-tuned anchors; the tuner has to beat FIXED, which")
    print("  is START held constant on the same seeds.")
    print("  %-18s %8s %8s %8s %8s %10s"
          % ("track", "START", "fixed", "tuner", "BEST", "closed"))
    summary = {}
    for t in a.tracks:
        def tail(learn):
            v = [np.mean([e["laps"] for e in res[(t, s, learn)][-3:]])
                 for s in range(a.seeds)]
            return float(np.mean(v)), float(np.std(v))
        fm, fs = tail(False)
        tm, ts = tail(True)
        st, bs = B.start(t).laps, B.best(t).laps
        gap = bs - fm
        closed = (tm - fm) / gap * 100 if abs(gap) > 1e-9 else float("nan")
        summary[t] = dict(fixed=fm, fixed_sd=fs, tuner=tm, tuner_sd=ts,
                          start=st, best=bs, closed_pct=closed)
        print("  %-18s %8.2f %8.2f %8.2f %8.2f %9.0f%%"
              % (t, st, fm, tm, bs, closed))
    print()
    for t in a.tracks:
        d = summary[t]
        print("  %-18s fixed %.2f +-%.2f   tuner %.2f +-%.2f  (n=%d seeds)"
              % (t, d["fixed"], d["fixed_sd"], d["tuner"], d["tuner_sd"],
                 a.seeds))
    print("  A difference smaller than the spread across seeds is not a result.")

    OUT.mkdir(exist_ok=True)
    p = OUT / "online_from_baseline.json"
    p.write_text(json.dumps(dict(
        summary=summary, weight_names=list(WEIGHT_NAMES),
        episodes={"|".join(map(str, k)): v for k, v in res.items()},
        traces=traces, config=vars(a)), indent=1))
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
