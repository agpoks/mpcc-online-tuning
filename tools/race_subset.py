"""Reproducible seed-count VIEW of a race run: write the first N seeds of a race_phase1.json to
race_phase1_<N>seed.json. Because the run is deterministic and seeds are independent, the N-seed
experiment IS the full run restricted to seeds 0..N-1 -- no re-training, provably consistent with
the canonical run (seeds 0..N-1 are bit-identical).

    python3 tools/race_subset.py 5                 # -> results/race/race_phase1_5seed.json
    python3 tools/race_subset.py 5 --in results/race/race_phase1.json
"""
import sys, json, argparse
from pathlib import Path
ROOT = Path("/home/poxx/github/mpcc-online-tuning")


def subset(src, n):
    d = json.load(open(src))
    keep = set(range(n))
    out = dict(d)
    out["runs"] = [r for r in d["runs"] if r.get("seed") in keep]
    nseeds = len({r["seed"] for r in out["runs"]})
    dst = Path(str(src).replace(".json", f"_{nseeds}seed.json"))
    json.dump(out, open(dst, "w"))
    print(f"wrote {dst}  ({nseeds} seeds, {len(out['runs'])} runs)")
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, help="number of seeds (first 0..n-1)")
    ap.add_argument("--in", dest="src", default=str(ROOT / "results/race/race_phase1.json"))
    a = ap.parse_args()
    subset(a.src, a.n)
