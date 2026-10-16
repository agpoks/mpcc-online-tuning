#!/usr/bin/env python3
"""Build an MPCC corridor npz for a real deploy map from a raceline_optimization WORLD csv.

The WORLD csv carries the optimized line, its speed profile and the per-side track widths:
    x;y;vx_mps;vy_mps;dpsi_rad;psi_rad;kappa_radpm;ax_mps2;s_m;w_left_m;w_right_m
where w_left_m / w_right_m are the perpendicular distances from the raceline to each track edge.

The corridor is built RACELINE-REFERENCED, exactly like Track.icra_t2_smooth / Track._smooth_from_csv -- the
representation every deployed net was trained on:
    centre = the raceline (lightly smoothed), per-side half-widths = w_left_m / w_right_m (minus a car berth).

WHY NOT RAY-CAST THE MAP WALLS (the previous approach): casting a ray from a recovered centre to the nearest
occupied pixel breaks in two places on a real SLAM map. (1) Where a wall has a HOLE the ray finds nothing and
saturates to the cap, so the "boundary" runs out into free space -- an OPEN END. (2) At a figure-8 CROSSOVER
the XY-nearest wall belongs to the WRONG loop, so the width is wrong. Both produced a tangled corridor with
open ends through the start and the crossover on lab_test_ev8 (centre drifted up to 0.85 m off the raceline),
and the trained net drove straight off it. The optimiser's own w_left/w_right are arc-length parameterised
(correct at a crossover, the raceline passes through it twice with its own width each time) and have no holes,
so the raceline-referenced corridor is clean by construction (verified: centre + both edges 100% on free map,
zero boundary jumps).

    python3 tools/build_corridor_from_world_csv.py <world.csv> <out_corridor.npz> \
        [--margin 0.05] [--map map.yaml  # optional: only to VERIFY edges land on free space]

Output npz schema (matches the raceline-ref corridors build_track() loads):
    cx, cy (centre = raceline) · wl, wr (per-side half-width about the centre) · raceline (N,2) ·
    vref (optimiser speed profile) · ds · length
"""
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world_csv")
    ap.add_argument("out_npz")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="car-berth inset subtracted from each side [m] (the MPCC corridor_safety adds the "
                         "rest; a few cm here keeps the tight apex sections off the raw wall)")
    ap.add_argument("--floor", type=float, default=0.08, help="minimum per-side half-width [m]")
    ap.add_argument("--map", default=None,
                    help="optional map .yaml: if given, VERIFY the centre and both edges land on free space "
                         "and print the fraction (does not change the corridor)")
    a = ap.parse_args()

    # --- parse the WORLD csv (';' separated; drop the '#:standard..' line, first real row is the header) ---
    rows = [ln for ln in open(a.world_csv) if not ln.startswith("#") and ln.strip()]
    hdr = rows[0].strip().split(";")
    col = {n: i for i, n in enumerate(hdr)}
    d = np.array([[float(v) for v in r.strip().split(";")] for r in rows[1:]])
    rx, ry = d[:, col["x"]], d[:, col["y"]]
    wl_r, wr_r = d[:, col["w_left_m"]], d[:, col["w_right_m"]]
    vref = d[:, col["vx_mps"]] if "vx_mps" in col else np.full(len(rx), 2.0)

    # --- RACELINE-REFERENCED corridor: centre = raceline, per-side widths = optimiser's own edge distances ---
    cx = gaussian_filter1d(rx, 1.0, mode="wrap")       # light smoothing kills CSV sampling jitter
    cy = gaussian_filter1d(ry, 1.0, mode="wrap")
    wl = np.maximum(gaussian_filter1d(wl_r, 1.5, mode="wrap") - a.margin, a.floor)
    wr = np.maximum(gaussian_filter1d(wr_r, 1.5, mode="wrap") - a.margin, a.floor)

    seg = np.hypot(np.diff(cx), np.diff(cy))
    length = float(seg.sum() + np.hypot(cx[0] - cx[-1], cy[0] - cy[-1]))
    np.savez(a.out_npz, cx=cx, cy=cy, wl=wl, wr=wr, ds=float(np.mean(seg)), length=length,
             raceline=np.column_stack([rx, ry]), vref=vref)
    print(f"wrote {a.out_npz}:  N={len(cx)}  length={length:.2f} m  (raceline-referenced)")
    print(f"  half-width wl {wl.min():.2f}..{wl.max():.2f}  wr {wr.min():.2f}..{wr.max():.2f} "
          f"(total corridor {np.min(wl + wr):.2f}..{np.max(wl + wr):.2f} m)")
    print(f"  raceline = optimized line (apexes) · vref {vref.min():.2f}..{vref.max():.2f} m/s")

    if a.map:
        import yaml
        from PIL import Image
        m = yaml.safe_load(open(a.map))
        base = a.map.rsplit("/", 1)[0] + "/" if "/" in a.map else ""
        img = np.array(Image.open(base + m["image"]).convert("L"))
        res = float(m["resolution"]); ox, oy = m["origin"][0], m["origin"][1]
        H, W = img.shape
        FREE = 255 * (1.0 - float(m.get("free_thresh", 0.2)))

        def free(wx, wy):
            px = int((wx - ox) / res); py = int(H - 1 - (wy - oy) / res)
            return 0 <= px < W and 0 <= py < H and img[py, px] > FREE

        C = np.column_stack([cx, cy])
        t = np.gradient(C, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        n = np.column_stack([-t[:, 1], t[:, 0]])
        left = C + wl[:, None] * n; right = C - wr[:, None] * n
        print(f"  MAP CHECK: centre {100 * np.mean([free(*p) for p in C]):.1f}%  "
              f"left {100 * np.mean([free(*p) for p in left]):.1f}%  "
              f"right {100 * np.mean([free(*p) for p in right]):.1f}% on free space")


if __name__ == "__main__":
    main()
