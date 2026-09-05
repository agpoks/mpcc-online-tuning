"""Merge condition-subset runs of online_from_baseline into one JSON.

The three conditions were measured in two passes: `fixed` and `tuner`
together, then `fixed_noise` on its own once it became clear the comparison
needed it. Rerunning all twelve jobs to get one extra column would have cost
another twenty minutes of solves for data already in hand.

The first pass wrote boolean condition keys (`track|seed|True`), the second
writes names (`track|seed|fixed_noise`), so the boolean form is translated on
the way in. Anything unrecognised is left alone and reported rather than
silently dropped.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RENAME = {"True": "tuner", "False": "fixed"}


def _norm(key):
    t, s, c = key.rsplit("|", 2)
    return f"{t}|{s}|{RENAME.get(c, c)}"


def main(argv):
    if len(argv) < 3:
        raise SystemExit("usage: merge_online_results.py OUT.json IN.json ...")
    out = RES / argv[1]
    eps, traces, evals, names, cfg = {}, {}, {}, None, None
    for spec in argv[2:]:
        # "file.json" or "file.json:old=new" to rename a condition on the
        # way in -- the progress-clock run writes its learner as "tuner" like
        # every other run, and it has to sit beside the time-clock "tuner"
        name, _, ren = spec.partition(":")
        rename = dict([ren.split("=")]) if ren else {}
        p = RES / name
        if not p.exists():
            print(f"  skipping {p} -- not found")
            continue
        d = json.loads(p.read_text())
        names = names or d.get("weight_names")
        cfg = cfg or d.get("config")
        def _key(k):
            k = _norm(k)
            t, s_, c = k.rsplit("|", 2)
            return f"{t}|{s_}|{rename.get(c, c)}"
        for k, v in d["episodes"].items():
            eps[_key(k)] = v
        for k, v in d.get("traces", {}).items():
            traces[_key(k)] = v
        for k, v in d.get("evals", {}).items():
            evals[_key(k)] = v
        print(f"  {p.name}: {len(d['episodes'])} runs"
              + (f"  ({ren})" if ren else ""))

    conds = sorted({k.rsplit('|', 1)[1] for k in eps})
    tracks = sorted({k.split('|')[0] for k in eps})
    print(f"  merged {len(eps)} runs, conditions {conds}, tracks {tracks}")

    # recompute the summary over whatever conditions are actually present
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from mpcc_tuning import baselines as B
    summary = {}
    for t in tracks:
        seeds = sorted({int(k.split('|')[1]) for k in eps
                        if k.startswith(t + "|")})
        d = dict(start=B.start(t).laps, best=B.best(t).laps)
        for c in conds:
            vals = [np.mean([e["laps"] for e in eps[f"{t}|{s}|{c}"][-3:]])
                    for s in seeds if f"{t}|{s}|{c}" in eps]
            if vals:
                d[c] = float(np.mean(vals))
                d[c + "_sd"] = float(np.std(vals))
        summary[t] = d
    out.write_text(json.dumps(dict(summary=summary, weight_names=names,
                                   episodes=eps, traces=traces, evals=evals,
                                   config=cfg), indent=1))
    print(f"  wrote {out}")

    print()
    hdr = "  %-18s %8s" + " %14s" * len(conds) + " %8s"
    print(hdr % (("track", "START") + tuple(conds) + ("BEST",)))
    for t in tracks:
        d = summary[t]
        row = "  %-18s %8.2f" % (t, d["start"])
        for c in conds:
            row += (" %9.2f+-%.2f" % (d[c], d[c + "_sd"])) if c in d else " %14s" % "-"
        row += " %8.2f" % d["best"]
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
