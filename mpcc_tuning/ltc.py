"""A liquid time-constant policy over the MPCC's cost weights.

The tuner in ``learner.py`` moves **one** theta by TD(lambda). This moves a
*policy* that emits theta from the situation, so the weights become a function
of what is happening rather than a single vector to be found.

## Why a liquid cell rather than any recurrent one

The behaviour a policy has to express here is *overtake or follow*, and two
things about it are awkward for a memoryless map:

* a **closing rate is a derivative of a range**, so it cannot be read from one
  frame at any resolution -- "catching them" and "being caught" look identical
  instantaneously;
* the decision should be taken **once and held**. Re-deciding at 20 Hz invites
  chattering between overtake and follow, which is a real failure and not a
  hypothetical (see ``scuderia_gym_jax``'s ``examples/overtake.py``, whose whole
  overtake logic exists to latch a side).

An LTC's effective time constant

    tau_eff = tau / (1 + tau f(x, h))

is **a hold duration set at run time by the input** -- long when the gate is
shut, short when the input drives it. That is the principled form of "decide on
an event, hold between", rather than a hand-set dwell time.

Cell and derivatives adapted from Hasani, Lechner, Amini, Rus & Grosu,
*Liquid Time-constant Networks*, AAAI 2021, and from the implementation in
``rtrrl-playground/rtrrl_playground/nets/ltc.py`` -- **copied in and adapted,
not imported**, per this repo's standing rule. The fused semi-implicit Euler
step is the paper's, and is stable at any step size because the state appears on
both sides.

## How it is trained, and why the recurrence is not free

The envelope theorem gives ``dQ/dtheta`` for free at the solution. The policy
adds ``dtheta/dphi`` for its own parameters ``phi``, and because ``h`` is
recurrent that factor is **not** a per-tick quantity: ``phi`` influenced ``h``
at every earlier tick too. That is exactly the influence recursion, and here it
is carried in the RFLO form

    P <- leak * P + immediate

with ``leak = 1/den < 1`` guaranteed by ``1/tau > 0``. Two things follow that
are worth stating. The bound is **structural, not a clipped hyperparameter** --
an LTC cannot learn to hold its state perfectly and so cannot make the influence
series diverge. And the decay is *input-dependent*: the trace forgets fast
exactly when the neuron is being driven hard, which is the property that makes
liquid cells and local online rules fit together.

Every parameter belongs to exactly one neuron, so the RFLO update is exact per
neuron and the influence array is the same shape as the parameters.
"""

from __future__ import annotations

import numpy as np

from mpcc_tuning.mpcc import WEIGHT_NAMES


def _sigmoid(z):
    return 0.5 * (np.tanh(0.5 * z) + 1.0)


class LTCCell:
    """One LTC layer with analytic derivatives, RFLO-ready.

    ``theta`` here is the *cell's* parameter block ``[W | A | tau]``, not the
    MPCC's cost weights; the two are kept apart by calling the latter ``theta``
    only outside this class.
    """

    def __init__(self, n_in: int, n: int, dt: float = 1.0, tau_init=(2.0, 30.0),
                 tau_min: float = 1.0, tau_max: float = 60.0, a_max: float = 2.0,
                 input_gain: float = 3.0, seed: int = 0):
        self.n_in, self.n, self.dt = int(n_in), int(n), float(dt)
        self.tau_min, self.tau_max, self.a_max = tau_min, tau_max, a_max
        self.n_xi = n_in + n + 1                     # [x ; h ; bias]
        rng = np.random.default_rng(seed)
        # An LTC needs a bigger input gain than a tanh cell: the pre-activation
        # goes through a sigmoid gate and then only *scales* the reversal
        # potential, so the same spread in W xi produces a fraction of the state
        # variation a CT-RNN would get.
        W = rng.normal(0.0, input_gain / np.sqrt(self.n_xi), (n, self.n_xi))
        # Gaussian A, not +/-1. The state settles at f A / (1/tau + f), so A is
        # the only thing setting a unit's amplitude -- with A = +/-1 the layer is
        # one signal and its negation, the hidden state is near rank one, and a
        # linear head on top has almost nothing to work with.
        A = rng.normal(0.0, 1.0, (n, 1))
        tau = np.exp(rng.uniform(*np.log(tau_init), (n, 1)))
        self.p = np.concatenate([W, A, tau], axis=1)
        self.SL = {"W": (0, self.n_xi), "A": (self.n_xi, self.n_xi + 1),
                   "tau": (self.n_xi + 1, self.n_xi + 2)}
        self.reset()

    def reset(self):
        self.h = np.zeros(self.n)

    def step(self, x):
        """Advance one tick. Returns ``(h, immediate, leak)`` for RFLO."""
        nx = self.n_xi
        xi = np.concatenate([np.asarray(x, float), self.h, [1.0]])
        W, A, tau = self.p[:, :nx], self.p[:, nx], self.p[:, nx + 1]
        f = _sigmoid(W @ xi)
        den = 1.0 + self.dt * (1.0 / tau + f)
        h_new = (self.h + self.dt * f * A) / den

        dz = self.dt * f * (1.0 - f) * (A - h_new) / den
        imm = np.empty_like(self.p)
        imm[:, :nx] = dz[:, None] * xi[None, :]
        imm[:, nx] = self.dt * f / den
        imm[:, nx + 1] = h_new * self.dt / (tau ** 2 * den)
        leak = 1.0 / den                 # < 1 for any gate value, since 1/tau > 0
        self.h = h_new
        return h_new, imm, leak

    def clip(self):
        lo, hi = self.SL["tau"]
        self.p[:, lo:hi] = np.clip(self.p[:, lo:hi], self.tau_min, self.tau_max)
        lo, hi = self.SL["A"]
        # A bounded reversal potential is a bounded state. Unbounded, it feeds
        # back through the recurrence and takes the cell to infinity, which over
        # a long online run with no batch to average over is what happens.
        self.p[:, lo:hi] = np.clip(self.p[:, lo:hi], -self.a_max, self.a_max)

    def tau_eff(self):
        """The hold duration each unit is currently choosing, in ticks."""
        nx = self.n_xi
        xi = np.concatenate([np.zeros(self.n_in), self.h, [1.0]])
        f = _sigmoid(self.p[:, :nx] @ xi)
        tau = self.p[:, nx + 1]
        return tau / (1.0 + tau * f)


class MLPCell:
    """The memoryless control for the same interface. One tanh layer, no state.

    Item 2d's gate requires beating this as well as a fixed schedule, so it has
    to exist and be trained by the identical rule -- otherwise a win is a win
    over an untuned baseline, which is not a win.
    """

    def __init__(self, n_in: int, n: int, seed: int = 0, **_):
        self.n_in, self.n, self.dt = int(n_in), int(n), 1.0
        self.n_xi = n_in + 1
        rng = np.random.default_rng(seed)
        self.p = rng.normal(0.0, 1.0 / np.sqrt(self.n_xi), (n, self.n_xi))
        self.SL = {"W": (0, self.n_xi)}
        self.reset()

    def reset(self):
        self.h = np.zeros(self.n)

    def step(self, x):
        xi = np.concatenate([np.asarray(x, float), [1.0]])
        z = self.p @ xi
        self.h = np.tanh(z)
        imm = ((1.0 - self.h ** 2)[:, None] * xi[None, :])
        return self.h, imm, np.zeros(self.n)      # leak 0: no influence carried

    def clip(self):
        pass

    def tau_eff(self):
        return np.zeros(self.n)


class WeightPolicy:
    """situation -> theta, with the output bounded to a measured box.

    ``theta = theta0 + G h``, clipped. **The clip is not cosmetic.** The usual
    safety argument for learning weights rather than steering -- "the MPCC
    enforces the constraints, so any theta is feasible" -- is false as stated:
    ``experiments/weights_as_behaviour.py`` found a q_c that drives off the
    track, and ``experiments/overtake_or_follow.py`` found that every attempted
    pass at q_v = 10 leaves it. The bound is what makes the argument
    recoverable, and it is a *ceiling on q_v* rather than a box, because
    behaviour is set by the ratio q_v/q_c and safety by q_v alone.
    """

    def __init__(self, cell, theta0, lo, hi, out_scale: float = 0.5, seed: int = 0):
        self.cell = cell
        self.theta0 = np.asarray(theta0, float)
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        rng = np.random.default_rng(seed + 7919)
        self.G = rng.normal(0.0, out_scale / np.sqrt(cell.n),
                            (len(self.theta0), cell.n))
        self.P = np.zeros_like(cell.p)      # RFLO influence, dh/d(cell params)
        self.reset()

    def reset(self):
        self.cell.reset()
        self.P[:] = 0.0

    def step(self, feat):
        """One tick: advance the cell, emit theta, update the influence trace."""
        h, imm, leak = self.cell.step(feat)
        self.P = leak[:, None] * self.P + imm
        theta = np.clip(self.theta0 + self.G @ h, self.lo, self.hi)
        self._h = h
        return theta

    def grads(self, dQ_dtheta):
        """Chain the envelope gradient through the policy.

        ``dQ/dG = dQ/dtheta (x) h`` exactly, and ``dQ/d(cell params)`` is
        ``(dQ/dtheta . G) * P`` -- the RFLO approximation, which keeps only each
        neuron's own influence and drops the coupling between neurons.
        """
        g = np.asarray(dQ_dtheta, float)
        dG = np.outer(g, self._h)
        dcell = (g @ self.G)[:, None] * self.P
        return dG, dcell


def features(track, s5, opponents=(), v_max: float = 4.0, a_max: float = 4.0,
             a_lat_max: float = 6.0, preview=(0.5, 1.5, 3.0)):
    """Physics-informed features, in units that mean something.

    A policy keyed on coordinates will not transfer between tracks or speeds, so
    every entry here is a ratio of two physical quantities and is dimensionless
    by construction. TODO item 2c.

    Ordered: curvature preview (3), speed fraction, lateral margin fraction,
    then the opponent block -- gap in braking distances, time-to-collision, and
    whether the pass is physically available.
    """
    x, y, psi, v, s = (float(q) for q in s5)
    f = [np.tanh(3.0 * track.curvature(track.wrap(s + d))) for d in preview]
    f.append(v / v_max)
    half = track.half_width - 0.12
    f.append(float(np.clip(track.lateral(x, y) / max(half, 1e-6), -1.5, 1.5)))

    # No opponent is a *state*, not a missing value: saturate the gap features
    # rather than pass zeros, which would read as "someone is right here".
    gap_brake, ttc, avail = 1.0, 1.0, 0.0
    if len(opponents):
        best = None
        for o in opponents:
            d = (o.s - s) % track.length
            if best is None or d < best[0]:
                best = (d, o)
        d, o = best
        if d < track.length / 2:
            brake = max(v * v / (2.0 * a_max), 1e-3)
            gap_brake = float(np.tanh(d / (3.0 * brake)))
            closing = v - o.speed
            ttc = float(np.tanh((d / closing) / 4.0)) if closing > 1e-3 else 1.0
            # Is the pass physically available: the lateral acceleration needed
            # to move one car-width across within the gap, against the limit.
            t_gap = d / max(closing, 1e-3)
            a_need = 2.0 * 0.30 / max(t_gap ** 2, 1e-6)
            avail = float(np.clip(1.0 - a_need / a_lat_max, -1.0, 1.0))
    f += [gap_brake, ttc, avail]
    return np.array(f, float)


N_FEATURES = 8
THETA_LO = np.log(np.array([0.05, 0.05, 0.02, 1e-3, 1e-3, 1e-3]))
#: Ceiling on q_v at 2.0 -- measured: it saturates above ~2 on an empty track
#: and every attempted pass above it leaves the track with an opponent present.
THETA_HI = np.log(np.array([20.0, 200.0, 2.0, 10.0, 10.0, 10.0]))
assert len(WEIGHT_NAMES) == len(THETA_LO) == len(THETA_HI)


class PolicyTuner:
    """TD(lambda) on a *policy* over theta, instead of on theta itself.

    Identical outer loop to :class:`~mpcc_tuning.learner.QLambdaTuner` -- same
    TD error, same trace, same clipping, same RMS normalisation -- with one
    factor inserted. Where that tuner uses ``dQ/dtheta`` directly, this chains
    it through the policy to ``dQ/dphi``. Keeping the rest identical is what
    makes the comparison in ``experiments/ltc_behaviour.py`` a comparison of
    *parameterisations* and not of learning rules.
    """

    def __init__(self, mpcc, policy, gamma=0.98, lam=0.9, alpha=2e-3,
                 clip=1.0, delta_clip=1.0, explore=0.05, seed=0):
        from mpcc_tuning.model import ACCEL_MAX, STEER_MAX
        self.mpcc, self.pol = mpcc, policy
        self.gamma, self.lam, self.alpha = gamma, lam, alpha
        self.clip, self.delta_clip, self.explore = clip, delta_clip, explore
        self._lim = np.array([STEER_MAX, ACCEL_MAX])
        self.rng = np.random.default_rng(seed)
        self._rms_G = self._rms_c = None
        self.reset()

    def reset(self):
        self.pol.reset()
        self.eG = np.zeros_like(self.pol.G)
        self.ec = np.zeros_like(self.pol.cell.p)
        self.prev = None
        self.stats = {}

    def _explore(self, u):
        if self.explore <= 0:
            return u
        u = np.asarray(u, float).copy()
        u[:2] += self.rng.normal(0.0, self.explore, 2) * self._lim
        u[:2] = np.clip(u[:2], -self._lim, self._lim)
        return u

    def _norm(self, g, which):
        a = np.abs(g)
        if which == "G":
            self._rms_G = a if self._rms_G is None else 0.99 * self._rms_G + 0.01 * a
            sc = np.maximum(self._rms_G, 1e-8)
        else:
            self._rms_c = a if self._rms_c is None else 0.99 * self._rms_c + 0.01 * a
            sc = np.maximum(self._rms_c, 1e-8)
        return np.clip(g / sc, -self.clip, self.clip)

    def act(self, feat, state5):
        """Emit theta for this tick, solve, and return ``(theta, action)``."""
        theta = self.pol.step(feat)
        out = self.mpcc.value(state5, theta)
        action = self._explore(out["u0"])
        q = self.mpcc.action_value(state5, theta, action, v_out=out)
        self._pending = (state5, theta, q)
        return theta, action

    def learn(self, reward, next_state5, next_feat, terminated):
        """One TD update, then emit the next tick's theta and action."""
        s, theta, q = self._pending
        gQ = self.mpcc.grad_theta(q, s, theta)
        if self.prev is not None:
            pg, pq = self.prev
            v_next = 0.0 if terminated else -q["value"]
            delta = float(np.clip(reward + self.gamma * v_next - (-pq),
                                  -self.delta_clip, self.delta_clip))
            dG, dc = pg
            self.eG = self.gamma * self.lam * self.eG + self._norm(-dG, "G")
            self.ec = self.gamma * self.lam * self.ec + self._norm(-dc, "c")
            self.pol.G += self.alpha * delta * self.eG
            self.pol.cell.p += self.alpha * delta * self.ec
            self.pol.cell.clip()
            self.stats = {"delta": delta}
        self.prev = (self.pol.grads(gQ), q["value"])
        if terminated:
            return None, None
        return self.act(next_feat, next_state5)


def fixed_schedule(feat, theta0):
    """The hand-written control the gate requires beating.

    A lookup, not a learner: if an opponent is close and the pass is physically
    available, use the measured *aggressive overtake* weights; otherwise the
    *follow* ones. Both come from ``experiments/overtake_or_follow.py``, where
    the behaviour boundary is the ratio q_v/q_c and the safe pass is
    (q_v, q_c) = (2.0, 1.0) against a follow at (0.5, 10.0).

    It commits on a single frame and has no idea how fast it is closing, which
    is precisely the defect a recurrent policy is supposed to fix.
    """
    gap_brake, avail = feat[6], feat[7]
    th = np.array(theta0, float)
    if gap_brake < 0.6 and avail > 0.0:
        th[0], th[2] = np.log(1.0), np.log(2.0)     # q_c, q_v -- overtake
    else:
        th[0], th[2] = np.log(10.0), np.log(0.5)    # follow
    return th
