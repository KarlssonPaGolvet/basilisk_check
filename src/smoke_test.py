"""
Basilisk smoke test for the ASTRID feasibility check.

Simulates one orbit of a small satellite in a 500 km sun-synchronous-like LEO
(point-mass Earth gravity only) and checks the result against the analytic
Kepler solution:

  1. Position after one orbit vs. the analytic position  (should be ~metres or less)
  2. Specific orbital energy drift over the orbit          (should be ~0)

Run from the repo root with the virtual environment active:
    python src/smoke_test.py
"""

import os
import time
from copy import copy

import numpy as np

from Basilisk import __path__ as bsk_path
from Basilisk.simulation import spacecraft
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody

POS_TOL_M = 100.0        # max allowed position error after one orbit [m]
ENERGY_TOL = 1e-6        # max allowed relative energy drift [-]
STEP_S = 10.0            # integration time step [s]


def main():
    wall_start = time.perf_counter()

    # --- Simulation, process and task -----------------------------------
    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("dynProcess")
    proc.addTask(sim.CreateNewTask("dynTask", macros.sec2nano(STEP_S)))

    # --- Spacecraft module ------------------------------------------------
    sc = spacecraft.Spacecraft()
    sc.ModelTag = "astridSmokeSat"
    sim.AddModelToTask("dynTask", sc)

    # --- Earth gravity (point mass) --------------------------------------
    grav = simIncludeGravBody.gravBodyFactory()
    earth = grav.createEarth()
    earth.isCentralBody = True
    mu = earth.mu
    grav.addBodiesTo(sc)

    # --- Initial orbit: 500 km altitude, near-polar ----------------------
    oe = orbitalMotion.ClassicElements()
    oe.a = 6371.0e3 + 500.0e3
    oe.e = 0.001
    oe.i = 97.4 * macros.D2R
    oe.Omega = 0.0
    oe.omega = 0.0
    oe.f = 0.0
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    oe = orbitalMotion.rv2elem(mu, rN, vN)  # consistent elements
    sc.hub.r_CN_NInit = rN
    sc.hub.v_CN_NInit = vN

    n = np.sqrt(mu / oe.a**3)
    period_s = 2.0 * np.pi / n

    # --- Recorder: log the spacecraft state every step -------------------
    rec = sc.scStateOutMsg.recorder()
    sim.AddModelToTask("dynTask", rec)

    # --- Run one orbit ----------------------------------------------------
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(period_s))
    sim.ExecuteSimulation()
    wall_s = time.perf_counter() - wall_start

    # --- Results ----------------------------------------------------------
    t = rec.times() * macros.NANO2SEC
    r = np.array(rec.r_BN_N)
    v = np.array(rec.v_BN_N)

    # Analytic (Kepler) position at the last recorded time
    M0 = orbitalMotion.E2M(orbitalMotion.f2E(oe.f, oe.e), oe.e)
    oe_ref = copy(oe)
    oe_ref.f = orbitalMotion.E2f(orbitalMotion.M2E(M0 + n * t[-1], oe.e), oe.e)
    r_ref, _ = orbitalMotion.elem2rv(mu, oe_ref)
    pos_err = np.linalg.norm(r[-1] - np.array(r_ref))

    # Specific orbital energy drift
    energy = 0.5 * np.sum(v**2, axis=1) - mu / np.linalg.norm(r, axis=1)
    drift = abs((energy[-1] - energy[0]) / energy[0])

    alt_km = (np.linalg.norm(r, axis=1) - 6371.0e3) / 1e3

    print("=" * 60)
    print("Basilisk smoke test")
    print(f"  Basilisk location : {bsk_path[0]}")
    print(f"  Orbit period      : {period_s / 60:.1f} min")
    print(f"  Samples recorded  : {len(t)} (step {STEP_S:.0f} s)")
    print(f"  Altitude range    : {alt_km.min():.1f} - {alt_km.max():.1f} km")
    print(f"  Wall-clock time   : {wall_s:.2f} s "
          f"(~{period_s / wall_s:.0f}x faster than real time)")
    print(f"  Position error    : {pos_err:.3f} m   (limit {POS_TOL_M:.0f} m)")
    print(f"  Energy drift      : {drift:.2e}   (limit {ENERGY_TOL:.0e})")

    ok = pos_err < POS_TOL_M and drift < ENERGY_TOL
    print("  RESULT            :", "PASS" if ok else "FAIL")
    print("=" * 60)

    # --- Optional plot ----------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        os.makedirs("results", exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t / 60.0, alt_km)
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Altitude [km]")
        ax.set_title("Smoke test: altitude over one orbit")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join("results", "smoke_test_altitude.png")
        fig.savefig(out, dpi=120)
        print(f"Plot saved to {out}")
    except ImportError:
        print("matplotlib not installed - skipping plot")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())