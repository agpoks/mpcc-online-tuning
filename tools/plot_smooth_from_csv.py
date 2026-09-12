"""Plot the exported smooth CSVs over the real occupancy map, and overlay the
corridor the MPCC actually enforces, so the two can be compared by eye.

    python3 tools/plot_smooth_from_csv.py

Reads (nothing but) the CSV files -- proves they contain the right geometry:
    mpcc_tuning/tracks/icra_t2_smooth_raceline.csv    (s,x,y,v_ref,kappa)
    mpcc_tuning/tracks/icra_t2_smooth_boundaries.csv  (s,ref,left,right,widths)
Then overlays, as dashed lines, the edges rebuilt from Track.icra_t2_smooth's
OWN spline + track.width(s) -- the exact quantities build_ocp puts in the
corridor constraint. If the dashed lines sit on top of the CSV lines, the CSV
is a faithful copy of what the MPCC uses.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
TR = ROOT / "mpcc_tuning" / "tracks"

# --- read the CSVs (only) ---
rl = np.genfromtxt(TR / "icra_t2_smooth_raceline.csv", delimiter=",", names=True)
bd = np.genfromtxt(TR / "icra_t2_smooth_boundaries.csv", delimiter=",", names=True)

# --- real occupancy map, placed in world coords from the yaml ---
im = np.array(Image.open(TR / "icra2026_t2.pgm")); H, W = im.shape
res, ox, oy = 0.05, -2.8, -7.25                      # icra2026_t2.yaml
extent = [ox, ox + W * res, oy, oy + H * res]

# --- what the MPCC enforces, rebuilt from the track itself ---
from mpcc_tuning.track import Track
t = Track.icra_t2_smooth()
s = bd["s_m"]; mpx = []; mpLx = []; mpLy = []; mpRx = []; mpRy = []
for si in s:
    p = np.array(t.pos(float(si))).ravel(); a = float(t.tangent_angle(float(si)))
    nx, ny = -np.sin(a), np.cos(a); wl, wr = t.width(float(si))
    mpLx.append(p[0] + nx * float(wr)); mpLy.append(p[1] + ny * float(wr))   # +normal wall (dist wr)
    mpRx.append(p[0] - nx * float(wl)); mpRy.append(p[1] - ny * float(wl))   # -normal wall (dist wl)

fig, ax = plt.subplots(figsize=(11, 9.5))
ax.imshow(im, cmap="gray", extent=extent, origin="upper", zorder=0)
ax.plot(bd["left_x"], bd["left_y"], color="tab:blue", lw=2.2, label="CSV left boundary", zorder=3)
ax.plot(bd["right_x"], bd["right_y"], color="tab:cyan", lw=2.2, label="CSV right boundary", zorder=3)
ax.plot(rl["x_m"], rl["y_m"], color="red", lw=1.6, label="CSV raceline (MPCC ref)", zorder=4)
# overlay MPCC's own corridor, dashed
ax.plot(mpLx, mpLy, "--", color="black", lw=1.0, label="MPCC corridor (track.width)", zorder=5)
ax.plot(mpRx, mpRy, "--", color="black", lw=1.0, zorder=5)
ax.scatter([rl["x_m"][0]], [rl["y_m"][0]], c="lime", s=90, edgecolor="k", zorder=6, label="start")
ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
# residual: max distance between CSV edge and MPCC edge (should be ~0)
dL = np.hypot(bd["left_x"] - np.array(mpLx), bd["left_y"] - np.array(mpLy)).max()
dR = np.hypot(bd["right_x"] - np.array(mpRx), bd["right_y"] - np.array(mpRy)).max()
ax.set_title(f"CSV boundaries over the real map (dashed = what the MPCC enforces)\n"
             f"max CSV-vs-MPCC gap: left {dL*1000:.1f} mm, right {dR*1000:.1f} mm")
ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
fig.tight_layout(); out = ROOT / "paper/figures/t2_smooth_from_csv.png"
fig.savefig(str(out), dpi=135); print("saved", out)
print(f"CSV-vs-MPCC max gap: left {dL*1000:.2f} mm, right {dR*1000:.2f} mm")
