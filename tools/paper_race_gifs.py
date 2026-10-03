"""Per-opponent race GIFs for the paper, animated from the cached frozen re-drive state logs
(results/race/paper/states_ltc_<seed>_<kind>.npz) -- no acados re-drive. Reuses race_gif.animate
so the GIF matches the paper figures exactly (same trajectory, same overtake).

    PYTHONPATH=...:. python3 tools/paper_race_gifs.py [--seed 0] [--kinds slower equal faster]
"""
import sys, argparse
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from experiments.race_mode import KEEPOUT_R, CONTACT_R
import tools.race_gif as RG

OUT = ROOT / "results/race/gif"; OUT.mkdir(parents=True, exist_ok=True)
PAPER = ROOT / "results/race/paper"


def gif_from_cache(seed, kind, stride=6, fps=20, frozen=False):
    # online = the explore+learn method run (states_online_*); frozen = the deployed/banded 10-lap
    # eval (states_10lapfrozen_*, the all-clean band result). Longer race -> bigger stride.
    pre = "states_10lapfrozen" if frozen else "states_online"
    z = np.load(PAPER / f"{pre}_ltc_{seed}_{kind}.npz")
    EX, EY, OX, OY = z["EX"], z["EY"], z["OX"], z["OY"]
    cg = np.hypot(EX - OX, EY - OY) < CONTACT_R
    d = dict(EX=EX, EY=EY, EV=z["V"], OX=OX, OY=OY,
             GAP=list(z["GAP"]), PASS=list(z["PASS"]),
             CONTACT=list(np.maximum.accumulate(cg)), rad=KEEPOUT_R,
             track=Track.icra_t2_smooth())
    out = str(OUT / f"ltc_{kind}_{'frozen' if frozen else 'race'}_s{seed}.gif")
    RG.animate(d, "ltc", kind, out, stride=(20 if frozen else stride), fps=fps)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", nargs="+", default=["slower", "equal", "faster"])
    ap.add_argument("--frozen", action="store_true", help="animate the all-clean 10-lap frozen caches")
    a = ap.parse_args()
    for k in a.kinds:
        gif_from_cache(a.seed, k, frozen=a.frozen)
