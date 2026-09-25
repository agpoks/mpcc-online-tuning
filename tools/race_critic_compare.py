"""MPCC-critic vs return-critic, judged on the RACING objective (pace + overtakes), not clean%."""
import sys, json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
L = float(Track.icra_t2_smooth().length); DT = 0.05
KINDS = ["static", "slower", "equal", "faster"]
EQ, FA, SOLO = 1.57, 1.93, 1.43           # opponent straight-line speeds + ego solo pace


def passes_by_kind(js, arm):
    rows = [r for x in js["runs"] if x["arm"] == arm for r in x["rows"]]
    return {k: np.mean([r["passes"] for r in rows if r["kind"] == k] or [0]) for k in KINDS}


def overall(js, arm):
    rows = [r for x in js["runs"] if x["arm"] == arm for r in x["rows"]]
    return dict(clean=np.mean([r["clean"] for r in rows]), off=np.mean([r["off"] for r in rows]),
                contact=np.mean([r["contact"] for r in rows]), laps=np.mean([r["laps"] for r in rows]))


def traj_pace(arm):
    vs = []
    for f in sorted(glob.glob(str(ROOT / f"results/race/traj/traj_{arm}_*.npz"))):
        z = np.load(f)
        for k in KINDS:
            key = f"{k}_EV"
            if key in z.files and len(z[key]):
                vs.append(float(np.mean(z[key])))
    return float(np.mean(vs)) if vs else np.nan


mpcc = json.load(open(ROOT / "results/race/race_phase1_fast.json"))   # MPCC critic, steps=5500
ret = json.load(open(ROOT / "results/race/race_phase1_return.json"))  # return critic, steps=4000
STEPS_MPCC = 5500

fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
x = np.arange(len(KINDS)); w = 0.2
cols = {"MPCC ltc": "tab:blue", "RET ltc": "tab:green", "MPCC mlp": "tab:cyan", "RET mlp": "tab:red"}
series = [("MPCC ltc", passes_by_kind(mpcc, "ltc")), ("RET ltc", passes_by_kind(ret, "ltc")),
          ("MPCC mlp", passes_by_kind(mpcc, "mlp")), ("RET mlp", passes_by_kind(ret, "mlp"))]
for i, (name, pk) in enumerate(series):
    ax[0].bar(x + (i - 1.5) * w, [pk[k] for k in KINDS], w, label=name, color=cols[name], edgecolor="k", lw=.4)
ax[0].set_xticks(x); ax[0].set_xticklabels(KINDS); ax[0].set_ylabel("overtakes per episode")
ax[0].set_title("Overtakes by opponent class\n(equal/faster = the race that matters)")
ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=.3)
ax[0].annotate("baseline never\nraced equal (~0)", (2 - 1.5 * w, 0.05), (1.2, 2.5),
               fontsize=8, arrowprops=dict(arrowstyle="->", color="0.4"))

# ONE consistent order for panels 2 & 3 (name, source json, arm, pace source)
srcs = [("MPCC ltc", mpcc, "ltc"), ("MPCC mlp", mpcc, "mlp"), ("RET ltc", ret, "ltc"), ("RET mlp", ret, "mlp")]
names = [s[0] for s in srcs]
# pace: MPCC from laps (few truncations), RETURN from trajectory EV (accurate under truncation)
pace = {}
for name, js, arm in srcs:
    pace[name] = traj_pace(arm) if name.startswith("RET") else overall(js, arm)["laps"] * L / (STEPS_MPCC * DT)
ax[1].bar(names, [pace[n] for n in names], color=[cols[n] for n in names], edgecolor="k")
for lvl, lab in [(SOLO, f"ego solo {SOLO}"), (EQ, f"equal opp {EQ}"), (FA, f"faster opp {FA}")]:
    ax[1].axhline(lvl, ls="--", color="0.4", lw=1); ax[1].text(3.5, lvl + .02, lab, fontsize=8, ha="right")
ax[1].set_ylabel("mean on-track speed [m/s]"); ax[1].set_title("Pace: does it drive on the limit?")
ax[1].tick_params(axis="x", rotation=20); ax[1].grid(axis="y", alpha=.3)

# safety cost: off-track + contact, SAME order as names
off = [overall(js, arm)["off"] for _, js, arm in srcs]
con = [overall(js, arm)["contact"] for _, js, arm in srcs]
ax[2].bar(np.arange(4) - .2, off, .4, label="off-track (ran wide)", color="tab:orange", edgecolor="k")
ax[2].bar(np.arange(4) + .2, con, .4, label="contact (crash)", color="tab:red", edgecolor="k")
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(names, rotation=20)
ax[2].set_ylabel("fraction of episodes"); ax[2].set_title("Safety cost\n(contact is the real crash; off-track = ran wide)")
ax[2].legend(fontsize=8); ax[2].grid(axis="y", alpha=.3)
fig.suptitle("Race strategy: MPCC critic (clean but passive) vs return critic + class residual (races equal opponents)", fontsize=13)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(ROOT / f"results/race/race_critic_compare.{e}", dpi=140)
print("pace:", {k: round(v, 2) for k, v in pace.items()})
print("equal passes  MPCC ltc %.2f -> RET ltc %.2f | MPCC mlp %.2f -> RET mlp %.2f" %
      (passes_by_kind(mpcc, "ltc")["equal"], passes_by_kind(ret, "ltc")["equal"],
       passes_by_kind(mpcc, "mlp")["equal"], passes_by_kind(ret, "mlp")["equal"]))
print("saved results/race/race_critic_compare.{pdf,png}")
