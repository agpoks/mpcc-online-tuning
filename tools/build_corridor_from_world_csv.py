#!/usr/bin/env python3
"""Build an MPCC corridor npz for a real deploy map from a raceline_optimization WORLD csv.

The WORLD csv carries the optimized line, its speed profile and the per-side track widths:
    x;y;vx_mps;vy_mps;dpsi_rad;psi_rad;kappa_radpm;ax_mps2;s_m;w_left_m;w_right_m
We keep the OPTIMIZED raceline + speed profile from the csv, recover the corridor CENTRE from the line and
its margins (same reconstruction as mpcc_tuning/track.py._raceline), and then derive the per-side
boundaries by RAY-CASTING the centre to the real map walls (capped) -- this keeps the boundaries exactly on
the walls and stops them tangling across a figure-8 crossover, where the csv's symmetric width overshoots.

    python3 tools/build_corridor_from_world_csv.py <world.csv> <map.yaml> <out_corridor.npz> \
        [--margin 0.15] [--cap 1.0]

Output npz schema (matches the geom-narrow corridors build_track() loads):
    cx, cy (centre) · wl, wr (per-side half-width about the centre) · raceline (N,2, the optimized line) ·
    vref (optimiser speed profile) · ds · length
"""
import argparse
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter1d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world_csv")
    ap.add_argument("map_yaml")
    ap.add_argument("out_npz")
    ap.add_argument("--margin", type=float, default=0.15, help="car half-width safety inset [m]")
    ap.add_argument("--cap", type=float, default=1.0, help="max half-width [m] (bounds cross-neck leak)")
    a = ap.parse_args()

    # --- parse the WORLD csv (';' separated; drop the '#:standard..' line, first real row is the header) ---
    rows = [ln for ln in open(a.world_csv) if not ln.startswith("#") and ln.strip()]
    hdr = rows[0].strip().split(";")
    col = {n: i for i, n in enumerate(hdr)}
    d = np.array([[float(v) for v in r.strip().split(";")] for r in rows[1:]])
    rx, ry = d[:, col["x"]], d[:, col["y"]]
    wl_r, wr_r = d[:, col["w_left_m"]], d[:, col["w_right_m"]]
    vref = d[:, col["vx_mps"]] if "vx_mps" in col else np.full(len(rx), 2.0)

    # --- recover the corridor centre from the raceline + its margins (track.py._raceline) ---
    tang = np.gradient(np.column_stack([rx, ry]), axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])                 # left normal
    cx = gaussian_filter1d(rx + 0.5 * (wl_r - wr_r) * nrm[:, 0], 1.5, mode="wrap")
    cy = gaussian_filter1d(ry + 0.5 * (wl_r - wr_r) * nrm[:, 1], 1.5, mode="wrap")

    # --- ray-cast the centre to the real map walls for accurate, capped per-side widths ---
    m = yaml.safe_load(open(a.map_yaml))
    base = a.map_yaml.rsplit("/", 1)[0] + "/" if "/" in a.map_yaml else ""
    img = np.array(Image.open(base + m["image"]).convert("L"))
    res = float(m["resolution"]); ox, oy = m["origin"][0], m["origin"][1]
    H, W = img.shape
    FREE = 255 * (1.0 - float(m.get("free_thresh", 0.2)))           # pixel brighter than this == free track

    def free(wx, wy):
        px = int((wx - ox) / res); py = int(H - 1 - (wy - oy) / res)
        return 0 <= px < W and 0 <= py < H and img[py, px] > FREE

    ct = np.gradient(np.column_stack([cx, cy]), axis=0)
    ct /= np.maximum(np.linalg.norm(ct, axis=1, keepdims=True), 1e-12)
    cn = np.column_stack([-ct[:, 1], ct[:, 0]])

    def cast(i, sign):
        s, dt = 0.0, res * 0.5
        while s < a.cap:
            if not free(cx[i] + sign * cn[i, 0] * (s + dt), cy[i] + sign * cn[i, 1] * (s + dt)):
                break
            s += dt
        return s

    wl = np.array([cast(i, +1) for i in range(len(cx))])
    wr = np.array([cast(i, -1) for i in range(len(cx))])
    wl = np.maximum(gaussian_filter1d(wl, 2.0, mode="wrap") - a.margin, 0.05)
    wr = np.maximum(gaussian_filter1d(wr, 2.0, mode="wrap") - a.margin, 0.05)

    seg = np.hypot(np.diff(cx), np.diff(cy))
    length = float(seg.sum() + np.hypot(cx[0] - cx[-1], cy[0] - cy[-1]))
    np.savez(a.out_npz, cx=cx, cy=cy, wl=wl, wr=wr, ds=float(np.mean(seg)), length=length,
             raceline=np.column_stack([rx, ry]), vref=vref)
    print(f"wrote {a.out_npz}:  N={len(cx)}  length={length:.2f} m")
    print(f"  half-width wl {wl.min():.2f}..{wl.max():.2f}  wr {wr.min():.2f}..{wr.max():.2f} "
          f"(min corridor {np.min(wl + wr):.2f} m)")
    print(f"  raceline = optimized line (apexes) · vref {vref.min():.2f}..{vref.max():.2f} m/s")


if __name__ == "__main__":
    main()
