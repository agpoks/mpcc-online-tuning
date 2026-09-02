"""Step 3: a task a constant CANNOT win -- part of the track is slippery.

    PYTHONPATH=/path/to/scuderia_gym_jax \
        python3 experiments/slippery_sector.py --plant std --jobs 4

Every experiment in this repo has been run on uniform grip, and the direct
search says the whole prize for perfect per-situation weights there is +3.5% to
+8.9% -- small enough that a constant is nearly optimal and TD(lambda) has
almost nothing to chase. That is a property of the *task*, not of the learner.

This builds the task that breaks it. One arc of the track is given mu = 0.6
while the rest keeps mu = 1.0, using ``scuderia_gym_jax``'s own friction map: a
greyscale PNG sampled bilinearly at the car's position, so the grip change is
real tyre physics and not a speed cap. A kinematic bicycle cannot express this
at all, which is why it could not be asked before.

The claim under test is simple. One fixed weight vector must either be too
timid on the dry part or too bold on the wet part; a policy that can see where
it is should beat it. If the fixed baseline still wins, adaptation is not
merely hard here, it is unnecessary, and that is worth knowing.
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

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

#: The baseline that passes the acceptance gate on the STD tyre model.
BASE = dict(q_c=1.0, q_l=50.0, q_v=1.0, r_d=5.0)


def wet_arc_on_a_corner(track, span_m=None):
    """Arc range covering ONE corner -- the tightest one.

    Grip only matters where it is being used. A slippery patch on a straight,
    where the car neither steers nor brakes hard, is invisible by construction:
    the first version of this placed the patch at 35-60% of arc length, which on
    the oval is mostly straight, and wet and dry returned identical distances.

    Centred on the point of maximum curvature and spanning that corner, so the
    car meets the low grip exactly where it is asking the tyres for lateral
    force.
    """
    s = np.linspace(0.0, track.length, 2000, endpoint=False)
    k = np.abs([float(track.curvature(track.wrap(v))) for v in s])
    i = int(np.argmax(k))
    span = span_m if span_m is not None else max(0.10 * track.length, 4.0)
    lo = (s[i] - span / 2.0) / track.length
    hi = (s[i] + span / 2.0) / track.length
    return float(lo), float(hi), float(1.0 / max(k[i], 1e-9))


def write_friction_map(track, out_dir, wet_from=None, wet_to=None,
                       mu_dry=1.0, mu_wet=0.60, res=0.05, pad=1.5):
    """A greyscale PNG where ONE CORNER of the track is wet, plus its map yaml.

    ``wet_from``/``wet_to`` are fractions of arc length; when omitted they are
    placed on the tightest corner by :func:`wet_arc_on_a_corner`. The PNG
    encodes mu linearly between ``mu_min`` and ``mu_max`` as scuderia reads it,
    so we write the byte that decodes to the mu we want rather than the mu
    itself.
    """
    if wet_from is None or wet_to is None:
        wet_from, wet_to, _r = wet_arc_on_a_corner(track)
    from PIL import Image

    xy = track.center
    x0, y0 = xy[:, 0].min() - pad, xy[:, 1].min() - pad
    x1, y1 = xy[:, 0].max() + pad, xy[:, 1].max() + pad
    W = int(np.ceil((x1 - x0) / res))
    H = int(np.ceil((y1 - y0) / res))

    MU_MIN, MU_MAX = 0.2, 2.0
    def byte_for(mu):
        v = (float(mu) - MU_MIN) / (MU_MAX - MU_MIN)
        return int(round(np.clip(v, 0.0, 1.0) * 255.0))

    img = np.full((H, W), byte_for(mu_dry), dtype=np.uint8)

    # Paint the wet arc as a thick stroke along the centreline, wide enough to
    # cover the whole corridor there.
    # The corner may straddle the start line -- on ICRA T1 the tightest one
    # lands at -5% to 5% -- so test membership on the WRAPPED arc length rather
    # than on the raw interval, which would paint nothing at all.
    s = np.linspace(0.0, track.length, 4000)
    lo, hi = wet_from * track.length, wet_to * track.length
    L = track.length

    def in_arc(v):
        a = (lo % L)
        b = (hi % L)
        w = v % L
        return (a <= w <= b) if a <= b else (w >= a or w <= b)
    half = max(float(np.max([float(track.width(v)[0]) for v in s[::40]])), 1.0)
    rad_px = int(np.ceil((half + 0.5) / res))
    wet = byte_for(mu_wet)
    for v in s:
        if not in_arc(v):
            continue
        p = np.asarray(track.pos(v)).ravel()
        c = int((p[0] - x0) / res)
        r = int((y1 - p[1]) / res)
        r0, r1 = max(0, r - rad_px), min(H, r + rad_px + 1)
        c0, c1 = max(0, c - rad_px), min(W, c + rad_px + 1)
        img[r0:r1, c0:c1] = wet

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "friction.png"
    yml = out_dir / "friction.yaml"
    Image.fromarray(img, mode="L").save(png)
    yml.write_text(f"resolution: {res}\norigin: [{x0}, {y0}, 0.0]\n")
    return str(png), str(yml), (mu_dry, mu_wet, wet_from, wet_to)


def one(job):
    track_name, horizon, seed, episodes, steps, learn, wet, png, yml = job
    from mpcc_tuning.ltc import (LTCCell, N_FEATURES, THETA_HI, THETA_LO,
                                 PolicyTuner, WeightPolicy, features)
    from mpcc_tuning.plant_scuderia import ScuderiaPlant

    t = getattr(Track, track_name)()
    th0 = MPCCWeights(**BASE).to_log()
    m = MPCC(t, model=KinematicBicycle(dt=0.05), horizon=horizon, dt=0.05,
             max_iter=300)

    fmap = None
    if wet:
        import scuderia_gym_jax as sgj
        from scuderia_gym_jax.envs.friction_map import FrictionMap
        fmap = FrictionMap(png, yml, mu_min=0.2, mu_max=2.0, mu_nominal=1.0)

    tu = None
    if learn:
        pol = WeightPolicy(LTCCell(N_FEATURES, 12, seed=seed), th0, THETA_LO,
                           THETA_HI, seed=seed)
        tu = PolicyTuner(m, pol, alpha=2e-3, explore=0.05, delta_clip=1.0,
                         seed=seed, trust_region=0.01)

    per_ep = []
    for ep in range(episodes):
        P = ScuderiaPlant(t, model="std", dt=0.05,
                          **({"friction_map": fmap} if fmap is not None else {}))
        P.max_steps = steps
        s5 = P.reset()
        m.reset()
        s0 = float(s5[4])
        off = False
        if learn:
            tu.reset()
            th, u = tu.act(features(t, s5), s5)
        else:
            th = th0
        for _ in range(steps):
            if not learn:
                u = m.value(s5, th)["u0"]
            s5n, r, off, tr = P.step(u)
            if learn:
                out = tu.learn(r, s5n, features(t, s5n), off)
                if out[0] is None:
                    break
                th, u = out
            s5 = s5n
            if off or tr:
                break
        per_ep.append(dict(ep=ep, laps=(float(s5[4]) - s0) / t.length,
                           off=bool(off)))
    return (track_name, seed, learn, wet), per_ep


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="oval")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--mu-wet", type=float, default=0.60,
                    help="whether this binds depends on how fast the car is "
                         "going, not on mu alone: at the oval's 2.46 m corner "
                         "mu=0.6 allows 3.81 m/s, so it constrains a car doing "
                         "4 and not one doing 3. The run prints both numbers.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = str(ROOT / "benchmarks" / "results"
                    / f"slippery_{a.track}.json")

    t = getattr(Track, a.track)()
    tmp = ROOT / "benchmarks" / "results" / f"frictionmap_{a.track}"
    png, yml, info = write_friction_map(t, tmp, mu_wet=a.mu_wet)
    ss = np.linspace(0, t.length, 600)
    kk = np.abs([float(t.curvature(t.wrap(x))) for x in ss])
    rmin = float(1.0 / max(kk.max(), 1e-9))
    print(f"  {a.track}: mu = {info[0]:.2f} dry, {info[1]:.2f} over arc "
          f"{info[2]:.0%}-{info[3]:.0%}  -> {png}", flush=True)
    print(f"  the patch is ON THE TIGHTEST CORNER, radius {rmin:.2f} m: dry "
          f"allows {np.sqrt(info[0] * 9.81 * rmin):.2f} m/s there, wet allows "
          f"{np.sqrt(info[1] * 9.81 * rmin):.2f} m/s.", flush=True)
    print(f"  Whether that binds depends on the speed the car actually "
          f"carries into it -- a slippery straight would not bind at any mu, "
          f"which is why the patch is placed by curvature and not by arc "
          f"fraction.", flush=True)

    jobs = [(a.track, 12, s, a.episodes, a.steps, learn, wet, png, yml)
            for s in range(a.seeds) for learn in (False, True)
            for wet in (False, True)]
    print(f"  {len(jobs)} runs over {a.jobs} workers", flush=True)

    res = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for key, per_ep in ex.map(one, jobs):
            res[key] = per_ep
            lp = float(np.mean([e["laps"] for e in per_ep[-3:]]))
            print(f"    seed {key[1]} {'tuner' if key[2] else 'fixed'} "
                  f"{'WET' if key[3] else 'dry'}  last-3 laps {lp:5.2f}",
                  flush=True)

    print()
    print("%-10s %16s %16s %10s" % ("grip", "fixed", "tuner", "change"))
    summary = {}
    for wet in (False, True):
        fx = [e["laps"] for k, v in res.items() if not k[2] and k[3] == wet
              for e in v[-3:]]
        tn = [e["laps"] for k, v in res.items() if k[2] and k[3] == wet
              for e in v[-3:]]
        if not fx or not tn:
            continue
        f_m, t_m = float(np.mean(fx)), float(np.mean(tn))
        f_s = float(np.std(fx) / max(np.sqrt(len(fx)), 1))
        t_s = float(np.std(tn) / max(np.sqrt(len(tn)), 1))
        lbl = "slippery" if wet else "uniform"
        summary[lbl] = dict(fixed=f_m, fixed_se=f_s, tuner=t_m, tuner_se=t_s,
                            change_pct=100.0 * (t_m - f_m) / max(f_m, 1e-9))
        print("%-10s %8.2f +-%5.2f %8.2f +-%5.2f %9.1f%%"
              % (lbl, f_m, f_s, t_m, t_s, summary[lbl]["change_pct"]))
    print()
    print("  If the tuner beats the fixed baseline on the SLIPPERY track and")
    print("  not on the uniform one, adaptation is being rewarded by the task")
    print("  rather than by the method, which is the claim worth making.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        dict(track=a.track, baseline=BASE, mu_wet=a.mu_wet,
             wet_arc=[info[2], info[3]], summary=summary,
             per_episode={f"{k[1]}|{'tuner' if k[2] else 'fixed'}|"
                          f"{'wet' if k[3] else 'dry'}": v
                          for k, v in res.items()}), indent=2) + "\n")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
