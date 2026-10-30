"""
Navigation: simpleNav with Gauss-Markov noise on position, velocity,
attitude, rate and sun direction.
"""

import numpy as np

from Basilisk.simulation import simpleNav

import config

_STATE_KEYS = ["pos_m", "vel_mps", "att_mrp", "rate_radps", "sun_rad", "dv_mps"]


def _expand(per_group):
    """Expand {group: value} into the 18-element simpleNav state vector."""
    return np.repeat([per_group[k] for k in _STATE_KEYS], 3)


def create_simple_nav(sim, task_name, sc, sun_msg):
    nav = simpleNav.SimpleNav()
    nav.ModelTag = "simpleNav"
    nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    nav.sunStateInMsg.subscribeTo(sun_msg)
    nav.PMatrix = np.diag(_expand(config.NAV_SIGMA)).tolist()
    nav.walkBounds = _expand(config.NAV_WALK_BOUNDS).tolist()
    nav.RNGSeed = config.RNG_SEED
    sim.AddModelToTask(task_name, nav, config.PRIORITY["simpleNav"])
    return nav
