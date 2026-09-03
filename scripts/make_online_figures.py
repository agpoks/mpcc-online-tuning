"""Figures for the online-tuning experiment.

Reads ``results/online_from_baseline.json`` and draws three things:

1. **learning curves** -- laps per episode, tuner against the fixed control,
   with START and BEST as reference lines. The control is the point: an
   improving curve on its own says nothing if the fixed baseline improves too
   (it cannot here, but the reader should be able to see that rather than
   take it on trust).
2. **weight trajectories** -- which of the eight weights actually move, drawn
   as position in the policy's own log box so weights of wildly different
   scale are comparable on one axis.
3. **weights by sector** -- the situation-dependence claim: does the policy
   emit different weights in corners than on straights?

Palette is the repo's existing categorical order, validated: all checks pass,
with orange<->green tritan dE 7.9 in the floor band, so those two are
direct-labelled dashed references rather than relying on hue alone.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUT, GRID = "#212529", "#868E96", "#DEE2E6"
C_FIXED, C_TUNER = "#868E96", "#4C6EF5"
C_BEST, C_START = "#0CA678", "#E8590C"
NICE = {"oval": "Oval", "icra_t2_raceline": "ICRA T2"}


def _load():
    name = sys.argv[1] if len(sys.argv) > 1 else "online_all.json"
    p = ROOT / "results" / name
    if not p.exists():
        raise SystemExit(f"no results at {p} -- run experiments/"
                         f"online_from_baseline.py first")
    return json.loads(p.read_text())


#: fixed and fixed_noise are the SAME thing without learning, so they share a
#: hue and separate by line style; only the tuner gets its own colour.
C_TUNER_P = "#AE3EC9"          # the fourth hue of the validated set
CONDS = (("fixed", C_FIXED, "-", "fixed (START held)"),
         ("fixed_noise", C_FIXED, (0, (4, 2)), "fixed + exploration noise"),
         ("tuner", C_TUNER, "-", "online tuner, per tick"),
         ("tuner_progress", C_TUNER_P, "-", "online tuner, per metre"))


def _curves(d, track, cond):
    """(episodes, seeds) laps array."""
    out = []
    for k, v in d["episodes"].items():
        t, s, lr = k.rsplit("|", 2)
        if t == track and lr == cond:
            out.append([e["laps"] for e in v])
    return np.asarray(out).T if out else np.zeros((0, 0))


def fig_learning(d):
    tracks = [t for t in NICE if any(k.startswith(t + "|")
                                     for k in d["episodes"])]
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.9 * len(tracks), 4.5),
                             squeeze=False)
    fig.patch.set_facecolor("white")
    for ax, t in zip(axes[0], tracks):
        for cond, c, ls, lab in CONDS:
            y = _curves(d, t, cond)
            if not y.size:
                continue
            x = np.arange(1, y.shape[0] + 1)
            # INDIVIDUAL seeds, not a mean and a band.
            #
            # The tuner is bimodal on T2 -- two seeds settle near 2.28 and one
            # crashes at 0.41 -- and mean +- sd draws a distribution that does
            # not exist, centred on a value no run ever took. The band was
            # also wider than the gap it was being compared against, which is
            # the shape of a summary that hides its own result.
            for j in range(y.shape[1]):
                ax.plot(x, y[:, j], color=c, lw=0.9, ls=ls, alpha=0.45,
                        zorder=2)
            ax.plot(x, y.mean(1), color=c, lw=2.2, ls=ls, zorder=3, label=lab,
                    marker="o", ms=4.5, mec="white", mew=1.0)
        s = d["summary"][t]
        for val, c, lab, dash in ((s["start"], C_START, "START", (0, (5, 3))),
                                  (s["best"], C_BEST, "BEST (hand-tuned)",
                                   (0, (1.5, 2)))):
            ax.axhline(val, color=c, lw=1.5, ls=dash, zorder=1)
            ax.text(0.985, val, f" {lab} {val:.2f}", transform=
                    ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=8.5, color=c, fontweight="bold")
        ax.set_title(NICE[t], fontsize=11.5, fontweight="bold", loc="left",
                     color=INK)
        ax.set_xlabel("episode", fontsize=9.5, color=MUT)
        ax.set_ylabel("laps completed", fontsize=9.5, color=MUT)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUT, labelsize=8.5)

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Online tuning from a verified baseline. Thick = mean, thin "
                 "= individual seeds: where runs are bimodal the mean "
                 "describes no actual run.",
                 fontsize=10.5, color=MUT, x=0.01, ha="left", y=1.02)
    fig.savefig(OUT / "online_learning.png", dpi=190, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  wrote", OUT / "online_learning.png")


def fig_weights(d, cond="tuner", suffix=""):
    """Which weights move, in policy-box coordinates so scales are comparable."""
    from mpcc_tuning.ltc import THETA_HI, THETA_LO
    names = d["weight_names"]
    tracks = [t for t in NICE if any(k.startswith(t + "|")
                                     for k in d["episodes"])]
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.9 * len(tracks), 4.5),
                             squeeze=False)
    fig.patch.set_facecolor("white")
    # sequential-by-index would imply an order the weights do not have; these
    # are identities, so a fixed categorical order it is
    cols = ["#4C6EF5", "#0CA678", "#E8590C", "#AE3EC9", "#1098AD", "#F59F00",
            "#E03131", "#495057"]
    for ax, t in zip(axes[0], tracks):
        per = [v for k, v in d["episodes"].items()
               if k.startswith(t + "|") and k.endswith("|" + cond)]
        if not per:
            continue
        th = np.asarray([[e["theta"] for e in run] for run in per])  # seed,ep,w
        # Position in the box the policy ACTUALLY had. With --box adapt that
        # is baselines.adaptation_box, a factor of `factor` around theta_0;
        # drawn against the global box instead, a weight sitting hard on the
        # adaptation ceiling would read as "barely moved".
        cfg = d.get("config") or {}
        if cfg.get("box") == "adapt":
            from mpcc_tuning import baselines as B
            lo, hi = B.adaptation_box(t, float(cfg.get("factor", 2.0)))
            box_lab = f"adaptation box (x/{cfg.get('factor', 2.0)} .. x{cfg.get('factor', 2.0)} of theta_0)"
        else:
            lo, hi = np.asarray(THETA_LO), np.asarray(THETA_HI)
            box_lab = "global search box"
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        frac = (np.log(th) - lo) / (hi - lo)
        # Prepend the TRUE anchor as episode 0.
        #
        # Ranking movement between episodes 1 and 10 hid the whole result:
        # most of the travel happens between theta_0 and the END of episode 1,
        # so q_v -- which is pinned at the ceiling by episode 1 and stays --
        # was drawn faint as though it had not moved.
        from mpcc_tuning import baselines as B
        th0 = (np.asarray(B.start(t).theta(), float) - lo) / (hi - lo)
        m = np.vstack([th0, frac.mean(0)])
        x = np.arange(0, m.shape[0])
        moved = np.argsort(-np.abs(m[-1] - m[0]))
        # Highlight a weight if it ENDS PINNED AT A BOUND, not merely if it
        # travelled far. Being stuck on the floor or ceiling is the failure
        # signature here -- the policy saturates and stops responding -- and
        # q_v reaches the oval's ceiling while ranking only fourth by distance.
        pinned = set(np.flatnonzero((m[-1] > 0.93) | (m[-1] < 0.07)).tolist())
        big_set = pinned | set(moved[:2].tolist())
        lab_pts = []
        for i in range(m.shape[1]):
            big = i in big_set
            ax.plot(x, m[:, i], color=cols[i], lw=2.0 if big else 1.0,
                    alpha=1.0 if big else 0.45, zorder=3 if big else 2)
            if big:
                tag = "%s %.2f->%.2f" % (names[i], m[0, i], m[-1, i])
                if m[-1, i] > 0.93:
                    tag += "  CEILING"
                elif m[-1, i] < 0.07:
                    tag += "  floor"
                lab_pts.append([m[-1, i], tag, cols[i]])
        # nudge labels apart so a cluster on the floor stays readable
        lab_pts.sort()
        for j in range(1, len(lab_pts)):
            if lab_pts[j][0] - lab_pts[j - 1][0] < 0.055:
                lab_pts[j][0] = lab_pts[j - 1][0] + 0.055
        for yv, tag, c in lab_pts:
            ax.annotate(tag, (x[-1], yv), xytext=(6, 0),
                        textcoords="offset points", va="center",
                        fontsize=8, color=c, fontweight="bold")
        ax.set_title(f"{NICE[t]} -- weights the policy actually moves",
                     fontsize=11.5, fontweight="bold", loc="left", color=INK)
        ax.axvline(0, color=MUT, lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.text(0, 1.03, " theta_0", fontsize=8, color=MUT, va="bottom")
        ax.set_xlabel("episode  (0 = the START anchor)", fontsize=9.5,
                      color=MUT)
        ax.set_ylabel(f"position in {box_lab}\n(0 = floor, 1 = ceiling)",
                      fontsize=9.5, color=MUT)
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlim(-0.4, m.shape[0] + 4.4)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUT, labelsize=8.5)
    fig.suptitle("Bold = weights that end PINNED at a bound (the failure "
                 "signature: the squash saturates and the output stops "
                 "responding), plus the two furthest travelled. Faint = the "
                 "rest, drawn so 'nothing moved' stays visible.",
                 fontsize=10.5, color=MUT, x=0.01, ha="left", y=1.02)
    fig.savefig(OUT / f"online_weights{suffix}.png", dpi=190,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", OUT / f"online_weights{suffix}.png")


def fig_sectors(d, cond="tuner", suffix=""):
    """Does the policy emit different weights in different sectors?"""
    from mpcc_tuning.ltc import THETA_HI, THETA_LO
    names = d["weight_names"]
    tracks = [t for t in NICE if any(k.startswith(t + "|")
                                     for k in d["episodes"])]
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.9 * len(tracks), 4.2),
                             squeeze=False)
    fig.patch.set_facecolor("white")
    lo, hi = np.asarray(THETA_LO), np.asarray(THETA_HI)
    for ax, t in zip(axes[0], tracks):
        rows = []
        for k, tr in d["traces"].items():
            if not (k.startswith(t + "|") and k.endswith("|" + cond)):
                continue
            for ep in tr[-3:]:                     # settled behaviour only
                rows.extend(ep)
        if not rows:
            continue
        a = np.asarray(rows, float)
        want = 4 + len(names)
        if a.shape[1] != want:
            raise SystemExit(
                f"{t}: trace rows are {a.shape[1]} wide, expected {want} "
                f"([x, y, v, sector] + {len(names)} weights). This JSON was "
                f"written before the trace carried position and sector -- "
                f"rerun experiments/online_from_baseline.py.")
        sec, th = a[:, 3].astype(int), a[:, 4:]
        frac = (np.log(np.maximum(th, 1e-12)) - lo) / (hi - lo)
        secs = sorted(set(sec.tolist()))
        # Four named sectors, four hues, never cycled: the old list wrapped at
        # three and painted sectors 0 and 3 the same blue, which makes two
        # categories indistinguishable. Validated (all checks pass; the
        # orange<->green tritan dE of 7.9 sits in the floor band, so the bars
        # also carry a white separator and a legend).
        SEC_COLS = ("#4C6EF5", "#0CA678", "#E8590C", "#AE3EC9")
        if len(secs) > len(SEC_COLS):
            raise SystemExit(
                f"{t}: {len(secs)} sectors but only {len(SEC_COLS)} hues. "
                f"Add hues and re-validate rather than cycling them.")
        w = 0.8 / max(len(secs), 1)
        xs = np.arange(len(names))
        for j, sname in enumerate(secs):
            mu = frac[sec == sname].mean(0)
            ax.bar(xs + (j - (len(secs) - 1) / 2) * w, mu, w * 0.92,
                   color=SEC_COLS[j],
                   label=f"sector {sname}", zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=8.5, color=INK)
        ax.set_title(f"{NICE[t]} -- weights by sector, last 3 episodes",
                     fontsize=11.5, fontweight="bold", loc="left", color=INK)
        ax.set_ylabel("mean position in policy box", fontsize=9.5, color=MUT)
        ax.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUT, labelsize=8.5)
        ax.legend(frameon=False, fontsize=8.5, ncol=len(secs),
                  loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Equal bars across sectors = the policy learned a CONSTANT, "
                 "not a function of situation. That is what these show.",
                 fontsize=10.5, color=MUT, x=0.01, ha="left", y=1.02)
    fig.savefig(OUT / f"online_sectors{suffix}.png", dpi=190,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", OUT / f"online_sectors{suffix}.png")


if __name__ == "__main__":
    d = _load()
    fig_learning(d)
    fig_weights(d)
    fig_sectors(d)
    if any(k.endswith("|tuner_progress") for k in d["episodes"]):
        fig_weights(d, "tuner_progress", "_progress")
        fig_sectors(d, "tuner_progress", "_progress")
