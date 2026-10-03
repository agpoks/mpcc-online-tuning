"""NEW paper figure: ego STATES over a full 10-LAP race (not the single-lap trace of
fig_race_states). One row per opponent class, showing the SUSTAINED racing behaviour under the
r_a-floor band -- speed and lateral corridor position across all ~10 laps, with any off-track
marked. This is where the faster-opponent fix shows: at the baseline faster ran wide ~lap 3.6;
with the band it holds the line for the whole race.

Reads the 10-lap state caches written by tools/race_eval_laps.py:
    states_10lapfrozen_ltc_<seed>_<kind>.npz   (frozen, deterministic; default)
    states_10lap_ltc_<seed>_<kind>.npz         (online; --online)
so it re-plots WITHOUT re-driving acados. Fully regeneratable:
    python3 tools/race_eval_laps.py --seed 0 --frozen      # produces the caches (band applied)
    python3 tools/paper_race_10lap_states.py --seed 0      # this figure

Writes results/race/paper/fig_10lap_states.{pdf,png}.
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning.mpcc import WEIGHT_NAMES

OUT = ROOT / "results/race/paper"
KINDS = ["static", "slower", "equal", "faster"]
L = Track.icra_t2_smooth().length


def _edge_pct(EX, EY, b):
    """Lateral position as % of the corridor half-width (100% = at the edge), via the ref centreline."""
    rx, ry, wl, wr = b["ref_x"], b["ref_y"], b["wl"], b["wr"]
    th = np.unwrap(np.arctan2(np.gradient(ry), np.gradient(rx)))
    out = np.empty(len(EX))
    for i, (x, y) in enumerate(zip(EX, EY)):
        j = int(np.argmin((rx - x) ** 2 + (ry - y) ** 2))
        nx, ny = -np.sin(th[j]), np.cos(th[j])
        off = (x - rx[j]) * nx + (y - ry[j]) * ny
        hw = wl[j] if off >= 0 else wr[j]
        out[i] = abs(off) / hw * 100.0
    return out


def _laps(S):
    S = np.asarray(S, float); dS = np.diff(S); dS[dS < -L / 2] += L; dS[dS > L / 2] -= L
    return np.concatenate([[0.0], np.cumsum(dS)]) / L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--online", action="store_true", help="use the online caches instead of frozen")
    a = ap.parse_args()
    tag = "_10lap" if a.online else "_10lapfrozen"
    b = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")
    rai, kvi = WEIGHT_NAMES.index("r_a"), WEIGHT_NAMES.index("k_v")

    fig, axs = plt.subplots(len(KINDS), 1, figsize=(9.2, 2.15 * len(KINDS)), sharex=False)
    found = 0
    for ax, kind in zip(axs, KINDS):
        f = OUT / f"states{tag}_ltc_{a.seed}_{kind}.npz"
        if not f.exists():
            ax.text(0.5, 0.5, f"(no cache: {f.name})", ha="center", va="center", fontsize=8, color="0.5")
            ax.set_axis_off(); continue
        found += 1
        z = np.load(f); V = z["V"]; off = bool(z["off"][0])
        lap = _laps(z["S"]); ep = _edge_pct(z["EX"], z["EY"], b)
        ra = np.exp(z["THETA"][:, rai]).mean(); kv = np.exp(z["THETA"][:, kvi]).mean()
        axb = ax.twinx()
        axb.axhspan(100, 130, color="#f0c0c0", alpha=.5, lw=0)          # past the corridor edge
        axb.plot(lap, ep, color="#8a3ffc", lw=1.0, alpha=.85, label="lateral pos [% corridor]")
        axb.axhline(100, color="crimson", ls="--", lw=1)
        ax.plot(lap, V, color="#1f4e8c", lw=1.3, label="speed")
        if off:
            ax.scatter([lap[-1]], [V[-1]], marker="X", s=90, color="crimson", zorder=6)
        ax.set_ylabel("speed [m/s]", color="#1f4e8c"); axb.set_ylabel("lat %", color="#8a3ffc")
        ax.set_ylim(0, 3.6); axb.set_ylim(0, 130); ax.set_xlim(0, max(lap.max(), 1))
        ax.set_xlabel("lap"); ax.grid(alpha=.2)
        out = f"OFF-TRACK at {lap[-1]:.1f} laps" if off else f"CLEAN {lap[-1]:.1f} laps"
        ax.set_title(f"{kind}  —  {out}   (r_a {ra:.1f}, k_v {kv:.2f})", fontsize=9.5, loc="left")
    axs[0].plot([], [], color="#1f4e8c", label="speed"); axs[0].plot([], [], color="#8a3ffc", label="lateral %")
    axs[0].legend(loc="upper right", fontsize=7.5, ncol=2)
    mode = "online" if a.online else "frozen"
    fig.suptitle(f"Ego states over a 10-lap race under the r_a-floor band (ltc {mode}, seed {a.seed})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_10lap_states.{e}", dpi=150)
    print(f"wrote {OUT}/fig_10lap_states.pdf (+.png) from {found}/{len(KINDS)} class caches ({mode})")


if __name__ == "__main__":
    main()
