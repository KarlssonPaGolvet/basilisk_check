"""
Vehicle: spacecraft hub, initial orbit/attitude and the reaction wheel cluster.

Based on scenarioAttitudeFeedbackRW.py / scenarioAttitudeFeedbackRWPower.py.
"""

import numpy as np

from Basilisk.architecture import messaging
from Basilisk.simulation import reactionWheelStateEffector, spacecraft
from Basilisk.utilities import macros, orbitalMotion, simHelpers, simIncludeRW

import config


def create_spacecraft(sim, task_name):
    sc = spacecraft.Spacecraft()
    sc.ModelTag = config.SC_NAME
    sc.hub.mHub = config.SC_MASS_KG
    sc.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    sc.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(config.SC_INERTIA)
    sim.AddModelToTask(task_name, sc, config.PRIORITY["spacecraft"])
    return sc


def set_initial_state(sc, mu):
    """Set the initial orbit and attitude. Returns the orbital period [s]."""
    oe = orbitalMotion.ClassicElements()
    oe.a = orbitalMotion.REQ_EARTH * 1e3 + config.ALTITUDE_M
    oe.e = config.ECCENTRICITY
    oe.i = config.INCLINATION_DEG * macros.D2R
    oe.Omega = config.RAAN_DEG * macros.D2R
    oe.omega = config.ARG_PERIAPSIS_DEG * macros.D2R
    oe.f = config.TRUE_ANOMALY_DEG * macros.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe)

    sc.hub.r_CN_NInit = rN
    sc.hub.v_CN_NInit = vN
    sc.hub.sigma_BNInit = config.SIGMA_BN_INIT
    sc.hub.omega_BN_BInit = config.OMEGA_BN_B_INIT

    return 2.0 * np.pi / np.sqrt(mu / oe.a**3)


class ReactionWheels:
    """Three orthogonal wheels with Coulomb friction enabled."""

    def __init__(self, sim, task_name, sc):
        self.factory = simIncludeRW.rwFactory()
        for axis, speed in zip(config.RW_AXES, config.RW_INIT_SPEED_RPM):
            self.factory.create(config.RW_TYPE, axis,
                                Omega=speed,  # RPM
                                RWModel=messaging.BalancedWheels,
                                useRWfriction=True,
                                useMaxTorque=True)
        self.num = self.factory.getNumOfDevices()

        self.effector = reactionWheelStateEffector.ReactionWheelStateEffector()
        self.effector.ModelTag = "rwEffector"
        self.factory.addToSpacecraft(sc.ModelTag, self.effector, sc)
        sim.AddModelToTask(task_name, self.effector, config.PRIORITY["rwEffector"])

        # FSW-side wheel configuration message
        self.fsw_config_msg = self.factory.getConfigMessage()

    @property
    def speed_msg(self):
        return self.effector.rwSpeedOutMsg

    @property
    def wheel_msgs(self):
        return self.effector.rwOutMsgs
