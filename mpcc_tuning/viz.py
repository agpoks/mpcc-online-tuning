"""Animate the controller, and the tuner tuning it.

Two functions:

:func:`animate_run`
    one run at fixed weights, showing the MPCC's **predicted horizon** at every
    tick. A trajectory plot shows where the car went; this shows what the
    controller believed was about to happen, which is the thing that is wrong
    when the controller is wrong.
:func:`animate_tuning`
    a whole online-tuning session: the trajectory improving (or collapsing)
    while the six cost weights move underneath it. The results page has that as
    two tables; the point of the animation is that the collapse and the weight
    that causes it are visible in the same frame.

GIFs via Pillow -- no ffmpeg, and a GIF renders on Read the Docs with no
player. Frames are subsampled; see ``max_frames``.

    from mpcc_tuning.viz import animate_run
    animate_run(mpcc, plant, theta, "run.gif")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

INK, BLUE, RED, GREEN, AMBER = "#222222", "#1f4e9c", "#c1272d", "#2a9d5c", "#e8a33d"
WEIGHT_NAMES = ("q_c", "q_l", "q_v", "r_d", "r_a", "r_dv")


def _plt():
    if matplotlib.get_backend().lower().startswith("agg") or not matplotlib.is_interactive():
        try:
            matplotlib.use("Agg", force=False)
        except Exception:      # pragma: no cover
            pass
    import matplotlib.pyplot as plt
    return plt


def save_gif(anim, out, fps: int = 18, colors: int = 128) -> Path:
    """Write an animation as a GIF, quantised to a palette shared by the clip.

    The shared palette is built from frames sampled *across* the clip rather
    than from the first one: a colour that is not on screen at t=0 otherwise
    gets no entry and is silently remapped to whatever is nearest.
    """
    from matplotlib.animation import PillowWriter
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    if colors:
        from PIL import Image
        im = Image.open(out)
        n = im.n_frames
        picks = sorted({int(round(i)) for i in np.linspace(0, n - 1, min(n, 12))})
        strip = []
        for i in picks:
            im.seek(i)
            strip.append(im.convert("RGB"))
        w, h = strip[0].size
        sheet = Image.new("RGB", (w, h * len(strip)))
        for k, fr in enumerate(strip):
            sheet.paste(fr, (0, k * h))
        base = sheet.quantize(colors=colors, method=Image.MEDIANCUT)
        frames = []
        for i in range(n):
            im.seek(i)
            frames.append(im.convert("RGB").quantize(palette=base, dither=Image.NONE))
        before = out.stat().st_size
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
        if out.stat().st_size > before:       # never make it worse
            anim.save(out, writer=PillowWriter(fps=fps))
    return out


def _track_bg(ax, track):
    c, hw = track.center, track.half_width
    d = np.gradient(c, axis=0)
    n = np.stack([d[:, 1], -d[:, 0]], axis=1)
    n /= np.linalg.norm(n, axis=1)[:, None]
    for b in (c - hw * n, c + hw * n):
        ax.plot(*np.vstack([b, b[:1]]).T, color="0.25", lw=1.4)
    ax.plot(*np.vstack([c, c[:1]]).T, color="0.7", lw=0.8, ls="--")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def _horizon(mpcc, out):
    """The predicted state trajectory ``[x, y]`` from a solved NLP."""
    X = out["w"][:mpcc._nx].reshape(5, mpcc.N + 1, order="F")
    return X[0], X[1], X[4]


# ---------------------------------------------------------------------------
def animate_run(mpcc, plant, theta, out, steps: int = 260, fps: int = 18,
                max_frames: int = 180, title: str | None = None):
    """One run at fixed weights, drawing the horizon the solver committed to.

    The orange line is the MPCC's prediction for the next ``N`` steps and the
    orange dot is its reference point ``p(s)``. Watching those is how a
    controller's model error becomes visible: on a plant it models correctly
    the prediction lies on the path the car then follows, and on one it does
    not -- the ``scuderia`` plant, say -- the prediction sails through a corner
    the car cannot take, every tick, and is wrong again the next tick.
    """
    plt = _plt()
    from matplotlib.animation import FuncAnimation

    s5 = plant.reset()
    mpcc.reset()
    frames = []
    for _ in range(steps):
        o = mpcc.value(s5, theta)
        hx, hy, _hs = _horizon(mpcc, o)
        ref = mpcc.track.pos(float(o["w"][4]))
        frames.append(dict(x=s5[0], y=s5[1], v=s5[3], hx=hx.copy(), hy=hy.copy(),
                           ref=(float(ref[0]), float(ref[1])), ok=bool(o["ok"])))
        s5, _r, off, tr = plant.step(o["u0"])
        if off or tr:
            break
    crashed = bool(off)
    if max_frames and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    _track_bg(ax, mpcc.track)
    trail = ax.scatter([], [], s=8, c=[], cmap="viridis", vmin=0, vmax=4.0, zorder=3)
    horizon, = ax.plot([], [], "-", lw=2.0, color=AMBER, zorder=5)
    refdot, = ax.plot([], [], "o", ms=7, color=AMBER, mec="white", mew=1.0, zorder=6)
    car, = ax.plot([], [], "o", ms=10, color=BLUE, mec="white", mew=1.2, zorder=7)
    txt = ax.text(0.015, 0.97, "", transform=ax.transAxes, va="top", fontsize=9,
                  family="monospace",
                  bbox=dict(fc="white", ec="0.7", alpha=0.88, boxstyle="round,pad=0.3"))
    ax.plot([], [], "-", lw=2.0, color=AMBER, label="the MPCC's predicted horizon")
    ax.plot([], [], "o", ms=7, color=AMBER, label="reference point $p(s)$")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.colorbar(trail, ax=ax, label="speed [m/s]", shrink=0.85, pad=0.02)
    fig.suptitle(title or "MPCC: what the controller thinks is about to happen",
                 fontsize=10.5)

    xs, ys, vs = [], [], []

    def update(k):
        f = frames[k]
        xs.append(f["x"]); ys.append(f["y"]); vs.append(f["v"])
        trail.set_offsets(np.column_stack([xs, ys]))
        trail.set_array(np.asarray(vs))
        car.set_data([f["x"]], [f["y"]])
        horizon.set_data(f["hx"], f["hy"])
        refdot.set_data([f["ref"][0]], [f["ref"][1]])
        tail = "  OFF-TRACK" if (crashed and k == len(frames) - 1) else ""
        txt.set_text(f"t = {k:3d}\nv = {f['v']:4.1f} m/s{tail}")
        return [trail, car, horizon, refdot, txt]

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=1000 // fps)
    path = save_gif(anim, out, fps=fps)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def animate_tuning(episodes, out, track=None, fps: int = 4, title: str | None = None):
    """A tuning session: the trajectory and the six weights, episode by episode.

    ``episodes`` is a list of ``dict(traj=(M, 2) array, theta=(6,) log weights,
    covered=float, off=bool)`` -- whatever the caller recorded per episode. The
    weights are drawn on a log axis because that is the space they are learned
    in, and because ``q_l`` starts at 200 and ``r_a`` at 0.01.

    The reason to animate this rather than plot it: the results page reports
    that the tuner improves the controller and then destroys it, in two
    separate tables. Here the collapse and the weight that causes it are in the
    same frame.
    """
    plt = _plt()
    from matplotlib.animation import FuncAnimation

    fig, (ax, wx) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                                 gridspec_kw={"width_ratios": [1.5, 1.0]})
    if track is not None:
        _track_bg(ax, track)
    line, = ax.plot([], [], "-", lw=1.8, color=BLUE)
    crash, = ax.plot([], [], "x", ms=13, mew=3, color=RED)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    thetas = np.array([e["theta"] for e in episodes])
    w = np.exp(thetas)
    bars = wx.bar(WEIGHT_NAMES, w[0], color=[BLUE] * 3 + [GREEN] * 3)
    wx.set_yscale("log")
    wx.set_ylim(max(w.min() * 0.5, 1e-4), w.max() * 2)
    wx.set_ylabel("cost weight (log scale)")
    wx.tick_params(axis="x", labelsize=9)
    fig.suptitle(title or "online tuning: six weights, one reward, no replay buffer",
                 fontsize=10.5)

    def update(k):
        e = episodes[k]
        t = np.asarray(e["traj"])
        line.set_data(t[:, 0], t[:, 1])
        if e.get("off"):
            crash.set_data([t[-1, 0]], [t[-1, 1]])
        else:
            crash.set_data([], [])
        for b, v in zip(bars, w[k]):
            b.set_height(v)
        ax.set_title(f"episode {k:3d}   covered {e['covered']:6.1f} m"
                     f"{'   OFF-TRACK' if e.get('off') else ''}",
                     fontsize=10, color=RED if e.get("off") else INK)
        return [line, crash, *bars]

    anim = FuncAnimation(fig, update, frames=len(episodes), blit=False,
                         interval=1000 // fps)
    path = save_gif(anim, out, fps=fps)
    plt.close(fig)
    return path
