"""Tune an MPCC's cost weights online, every control tick.

The controller stays an MPCC. What changes is that its cost weights -- and,
optionally, parameters of its internal model -- are treated as the parameters
of a reinforcement-learning policy and updated from one scalar TD error per
tick, while the car drives.

The formulation is Gros & Zanon's: the MPC *is* the function approximator, for
the policy and for the value function at once. The piece that makes it
real-time is the envelope theorem -- see :mod:`mpcc_tuning.learner`.
"""

from mpcc_tuning.mpcc import MPCC, MPCCWeights
from mpcc_tuning.track import Track
from mpcc_tuning.learner import QLambdaTuner
from mpcc_tuning.model import KinematicBicycle

__version__ = "0.1.0"
__all__ = ["MPCC", "MPCCWeights", "Track", "QLambdaTuner", "KinematicBicycle"]
