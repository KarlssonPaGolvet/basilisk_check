"""Basilisk sanity check.

Propagates a spacecraft in a point-mass Earth gravity field with Basilisk and
compares the final state to the analytic Keplerian solution.

Run:  bsk-env\\Scripts\\python.exe src\\two_body_orbit_check.py [--plot]
"""
import sys

import numpy as np

from Basilisk import __path__ as bsk_path
from Basilisk.simulation import spacecraft
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody


def run(show_plot=False):
    sim = SimulationBaseClass.SimBaseClass()
    process = sim.CreateNewProcess("dynProcess")
    step = macros.sec2nano(10.0)
    process.addTask(sim.CreateNewTask("dynTask", step))

    sc = spacecraft.Spacecraft()
    sc.ModelTag = "bsk-sat"
    sim.AddModelToTask("dynTask", sc)

    grav_factory = simIncludeGravBody.gravBodyFactory()
    earth = grav_factory.createEarth()
    earth.isCentralBody = True
    mu = earth.mu
    grav_factory.addBodiesTo(sc)

    # Initial orbit: ~LEO, slightly eccentric and inclined
    oe = orbitalMotion.ClassicElements()
    oe.a = 7000.0e3      # m
    oe.e = 0.01
    oe.i = 33.3 * macros.D2R
    oe.Omega = 48.2 * macros.D2R
    oe.omega = 347.8 * macros.D2R
    oe.f = 85.3 * macros.D2R
    r0, v0 = orbitalMotion.elem2rv(mu, oe)
    sc.hub.r_CN_NInit = r0
    sc.hub.v_CN_NInit = v0

    period = 2.0 * np.pi * np.sqrt(oe.a**3 / mu)
    sim_time = macros.sec2nano(0.75 * period)

    recorder = sc.scStateOutMsg.recorder(macros.sec2nano(30.0))
    sim.AddModelToTask("dynTask", recorder)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(sim_time)
    sim.ExecuteSimulation()

    pos = recorder.r_BN_N
    vel = recorder.v_BN_N
    times = recorder.times() * macros.NANO2SEC

    # Analytic Kepler propagation to the final recorded time
    M0 = orbitalMotion.E2M(orbitalMotion.f2E(oe.f, oe.e), oe.e)
    n = np.sqrt(mu / oe.a**3)
    oe_end = orbitalMotion.ClassicElements()
    oe_end.a, oe_end.e, oe_end.i = oe.a, oe.e, oe.i
    oe_end.Omega, oe_end.omega = oe.Omega, oe.omega
    oe_end.f = orbitalMotion.E2f(orbitalMotion.M2E(M0 + n * times[-1], oe.e), oe.e)
    r_true, v_true = orbitalMotion.elem2rv(mu, oe_end)

    pos_err = np.linalg.norm(pos[-1] - np.array(r_true))
    vel_err = np.linalg.norm(vel[-1] - np.array(v_true))

    # Energy should be conserved for two-body motion
    energy = 0.5 * np.sum(vel**2, axis=1) - mu / np.linalg.norm(pos, axis=1)
    energy_drift = np.max(np.abs(energy - energy[0]) / abs(energy[0]))

    print(f"Basilisk installed at: {bsk_path[0]}")
    print(f"Orbit period:          {period / 60:.2f} min, simulated {times[-1] / 60:.2f} min")
    print(f"Samples recorded:      {len(times)}")
    print(f"Final position error:  {pos_err:.3e} m")
    print(f"Final velocity error:  {vel_err:.3e} m/s")
    print(f"Max rel. energy drift: {energy_drift:.3e}")

    ok = pos_err < 1.0 and vel_err < 1e-3 and energy_drift < 1e-8
    print("RESULT:", "PASS - Basilisk works" if ok else "FAIL")

    if show_plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot(*(pos.T / 1e3), label="Basilisk trajectory")
        ax.scatter(0, 0, 0, color="tab:blue", s=80, label="Earth")
        ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]")
        ax.legend()
        plt.show()

    return ok


if __name__ == "__main__":
    sys.exit(0 if run(show_plot="--plot" in sys.argv) else 1)
