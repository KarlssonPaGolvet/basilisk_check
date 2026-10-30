"""
Debris proximity check on the full mini prototype.

Adds a second object (debris: its own spacecraft module, same Earth/Sun/Moon gravity, same MSIS
atmosphere and cannonball drag setup) to the unchanged prototype. The debris orbit is chosen so it
crosses the satellite's orbit at 1 deg, 3 km above it, at a planned closest approach of 1.25 orbits:
the satellite's Kepler state at that time is offset 3 km radially, its velocity rotated 1 deg about
the radial direction, and the result propagated back to t = 0 with Kepler.

A Basilisk event raises a "proximity alert" when the distance drops below 10 km.

Checks:
    - closest approach (CA) found within the first 2 orbits, at a few km
    - CA distance and time match an independent minimal Basilisk sim (two bare spacecraft, same
      gravity and drag, no prototype code): within 1 m and 0.5 s
    - CA distance and time match a pure two-body Kepler propagation (orbitalMotion) within 1 km
      and 10 s (Kepler ignores drag and the Sun/Moon pull, which act over ~2 h)
    - the alert fires once, within one step of the first sample below 10 km
    - adding the debris leaves the satellite's recorded data bit-identical (vs a run without it)

Run from the repo root with bsk-env active:
    python src\\mini_prototype\\check_debris_proto.py          checks only
    python src\\mini_prototype\\check_debris_proto.py --viz    also write a Vizard file (satellite + debris)

Output: results\\proto_debris.png
"""

import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from Basilisk.simulation import dragDynamicEffector, msisAtmosphere, spacecraft  # noqa: E402
from Basilisk.architecture import messaging  # noqa: E402
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion, simHelpers,  # noqa: E402
                                simIncludeGravBody)

import config  # noqa: E402
import main  # noqa: E402
from recorders import INTERNAL, NAVIGATION  # noqa: E402

ALERT_KM = 10.0
CA_ORBITS = 1.25            # planned closest approach
CA_RADIAL_OFFSET_M = 3000.0
CROSSING_DEG = 1.0
REC_S = 1.0                 # relative-distance recorders run every step
SECOND_SIM_TOL = (1.0, 0.5)     # [m], [s]
KEPLER_TOL = (1000.0, 10.0)     # [m], [s]


# ---------------------------------------------------------------- Kepler helpers
def satellite_initial_oe():
    """The satellite's initial elements, exactly as vehicle.set_initial_state() sets them."""
    oe = orbitalMotion.ClassicElements()
    oe.a = orbitalMotion.REQ_EARTH * 1e3 + config.ALTITUDE_M
    oe.e = config.ECCENTRICITY
    oe.i = config.INCLINATION_DEG * macros.D2R
    oe.Omega = config.RAAN_DEG * macros.D2R
    oe.omega = config.ARG_PERIAPSIS_DEG * macros.D2R
    oe.f = config.TRUE_ANOMALY_DEG * macros.D2R
    return oe


def kepler_propagate(mu, r0, v0, times):
    """Two-body states at the given times [s] from (r0, v0) at t = 0."""
    oe = orbitalMotion.rv2elem(mu, r0, v0)
    n = np.sqrt(mu / oe.a ** 3)
    m0 = orbitalMotion.E2M(orbitalMotion.f2E(oe.f, oe.e), oe.e)
    rs, vs = [], []
    for t in np.atleast_1d(times):
        oe.f = orbitalMotion.E2f(orbitalMotion.M2E((m0 + n * t) % (2 * np.pi), oe.e), oe.e)
        r, v = orbitalMotion.elem2rv(mu, oe)
        rs.append(r)
        vs.append(v)
    return np.array(rs), np.array(vs)


def rotate(vec, axis, angle):
    """Rodrigues rotation of vec about the unit vector axis."""
    k = axis / np.linalg.norm(axis)
    return vec * np.cos(angle) + np.cross(k, vec) * np.sin(angle) + k * np.dot(k, vec) * (1 - np.cos(angle))


def debris_initial_state(mu, period):
    oe = satellite_initial_oe()
    r_s0, v_s0 = orbitalMotion.elem2rv(mu, oe)
    t_ca = float(round(CA_ORBITS * period))
    r_s, v_s = kepler_propagate(mu, np.array(r_s0), np.array(v_s0), [t_ca])
    r_hat = r_s[0] / np.linalg.norm(r_s[0])
    r_d = r_s[0] + CA_RADIAL_OFFSET_M * r_hat
    v_d = rotate(v_s[0], r_hat, np.radians(CROSSING_DEG))
    r_d0, v_d0 = kepler_propagate(mu, r_d, v_d, [-t_ca])
    return np.array(r_s0), np.array(v_s0), r_d0[0], v_d0[0], t_ca


def closest_approach(t, r1, v1, r2, v2):
    """CA from sampled states, refined between samples with linear relative motion."""
    dr, dv = r2 - r1, v2 - v1
    d = np.linalg.norm(dr, axis=1)
    k = int(np.argmin(d))
    dt = -np.dot(dr[k], dv[k]) / np.dot(dv[k], dv[k])
    dt = float(np.clip(dt, -REC_S, REC_S))
    return t[k] + dt, float(np.linalg.norm(dr[k] + dv[k] * dt)), d


# ---------------------------------------------------------------- simulations
def add_drag(sim, sc, atmo, idx, area, cd, com_offset):
    drag = dragDynamicEffector.DragDynamicEffector()
    drag.ModelTag = f"drag_{sc.ModelTag}"
    drag.coreParams.projectedArea = area
    drag.coreParams.dragCoeff = cd
    drag.coreParams.comOffset = com_offset
    drag.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[idx])
    sc.addDynamicEffector(drag)
    sim.AddModelToTask(main.TASK, drag, config.PRIORITY["atmosphere"])
    return drag


def run_prototype(with_debris, r_d0=None, v_d0=None, viz=False):
    """Full prototype for NUM_ORBITS; optionally with the debris object and the proximity event.

    viz: also write a Vizard file showing the satellite (with its wheels) and the debris.
    """
    sim = SimulationBaseClass.SimBaseClass()
    sim.CreateNewProcess(main.PROCESS).addTask(sim.CreateNewTask(main.TASK, config.dyn_step_ns()))
    parts = main.build(sim)
    out = {"parts": parts, "alerts": []}
    rec_ns = macros.sec2nano(REC_S)
    out["sat_log"] = parts.sc.scStateOutMsg.recorder(rec_ns)
    sim.AddModelToTask(main.TASK, out["sat_log"])

    if with_debris:
        deb = spacecraft.Spacecraft()
        deb.ModelTag = "debris"
        deb.hub.mHub = config.SC_MASS_KG
        deb.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(config.SC_INERTIA)
        deb.hub.r_CN_NInit = r_d0.tolist()
        deb.hub.v_CN_NInit = v_d0.tolist()
        parts.env.grav_factory.addBodiesTo(deb)                  # same Earth/Sun/Moon gravity
        sim.AddModelToTask(main.TASK, deb, config.PRIORITY["spacecraft"])
        parts.env.atmo.addSpacecraftToModel(deb.scStateOutMsg)  # same MSIS atmosphere
        deb_drag = add_drag(sim, deb, parts.env.atmo, 1, config.DRAG_AREA_M2, config.DRAG_COEFF,
                            config.DRAG_COM_OFFSET_B)
        out["deb_log"] = deb.scStateOutMsg.recorder(rec_ns)
        sim.AddModelToTask(main.TASK, out["deb_log"])
        out["deb"], out["deb_drag"] = deb, deb_drag

        def distance():
            r1 = np.array(parts.sc.scStateOutMsg.read().r_BN_N)
            r2 = np.array(deb.scStateOutMsg.read().r_BN_N)
            return float(np.linalg.norm(r2 - r1))

        def alert(s):
            t_now = s.TotalSim.CurrentNanos * 1e-9
            out["alerts"].append((t_now, distance()))
            print(f"  PROXIMITY ALERT at t = {t_now:.0f} s: debris at {distance() / 1e3:.2f} km")

        sim.createNewEvent("proximityAlert", config.dyn_step_ns(), True,
                           conditionFunction=lambda s: distance() < ALERT_KM * 1e3, actionFunction=alert)
        if viz:
            out["viz"], out["viz_file"] = main.enable_viz(sim, [parts.sc, deb], [parts.rws.effector, None],
                                                          "proto_debris")

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(config.NUM_ORBITS * parts.period_s))
    sim.ExecuteSimulation()
    parts.env.unload()
    out["frames"] = parts.recorder.datasets()
    return out


def run_second_sim(r_s0, v_s0, r_d0, v_d0, sat_mass, t_end):
    """Independent minimal sim: two bare spacecraft, same gravity and drag models, no prototype code."""
    sim = SimulationBaseClass.SimBaseClass()
    sim.CreateNewProcess("proc").addTask(sim.CreateNewTask(main.TASK, macros.sec2nano(config.DYN_STEP_S)))
    grav = simIncludeGravBody.gravBodyFactory()
    earth = grav.createEarth()
    earth.isCentralBody = True
    grav.createSun()
    grav.createMoon()
    spice = grav.createSpiceInterface(time=config.EPOCH, epochInMsg=True)
    spice.zeroBase = "Earth"
    sim.AddModelToTask(main.TASK, spice, 200)

    atmo = msisAtmosphere.MsisAtmosphere()
    atmo.ModelTag = "msis2"
    atmo.epochInMsg.subscribeTo(grav.epochMsg)
    atmo.planetPosInMsg.subscribeTo(spice.planetStateOutMsgs[list(grav.gravBodies.keys()).index("earth")])
    sw = [messaging.SwDataMsg().write(messaging.SwDataMsgPayload(dataValue=config.SW_AP if i < 21 else config.SW_F107))
          for i in range(23)]
    for i, m in enumerate(sw):
        atmo.swDataInMsgs[i].subscribeTo(m)
    sim.AddModelToTask(main.TASK, atmo, 130)

    keep, logs = [sw, grav, spice, atmo], []
    for idx, (name, mass, r0, v0) in enumerate([("sat2", sat_mass, r_s0, v_s0), ("deb2", config.SC_MASS_KG, r_d0, v_d0)]):
        sc = spacecraft.Spacecraft()
        sc.ModelTag = name
        sc.hub.mHub = mass
        sc.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(config.SC_INERTIA)
        sc.hub.r_CN_NInit = list(r0)
        sc.hub.v_CN_NInit = list(v0)
        grav.addBodiesTo(sc)
        sim.AddModelToTask(main.TASK, sc, 120)
        atmo.addSpacecraftToModel(sc.scStateOutMsg)
        keep += [sc, add_drag(sim, sc, atmo, idx, config.DRAG_AREA_M2, config.DRAG_COEFF, [0.0, 0.0, 0.0])]
        logs.append(sc.scStateOutMsg.recorder(macros.sec2nano(REC_S)))
        sim.AddModelToTask(main.TASK, logs[-1])
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(t_end))
    sim.ExecuteSimulation()
    grav.unloadSpiceKernels()
    t = logs[0].times() * macros.NANO2SEC
    return t, [(np.array(lg.r_BN_N), np.array(lg.v_BN_N)) for lg in logs]


def frames_identical(fa, fb):
    bad = []
    for name in (INTERNAL, NAVIGATION):
        for col in fa[name].columns:
            x = fa[name][col].to_numpy(np.float64)
            y = fb[name][col].to_numpy(np.float64)
            if not np.array_equal(x.view(np.uint64), y.view(np.uint64)):
                bad.append(f"{name}:{col}")
    return bad, sum(df.size for df in fa.values())


# ---------------------------------------------------------------- check
def main_check(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", action="store_true", help="write a Vizard file of the debris run to results/_VizFiles/")
    args = ap.parse_args(argv)
    wall0 = time.perf_counter()
    print("Run 1/3: prototype without debris (reference) ...")
    ref = run_prototype(False)
    period = ref["parts"].period_s
    mu = ref["parts"].env.mu
    r_s0, v_s0, r_d0, v_d0, t_ca_plan = debris_initial_state(mu, period)
    t_end = config.NUM_ORBITS * period

    print("Run 2/3: prototype with debris ...")
    run = run_prototype(True, r_d0, v_d0, viz=args.viz)
    t = run["sat_log"].times() * macros.NANO2SEC
    r1, v1 = np.array(run["sat_log"].r_BN_N), np.array(run["sat_log"].v_BN_N)
    r2, v2 = np.array(run["deb_log"].r_BN_N), np.array(run["deb_log"].v_BN_N)
    t_ca, d_ca, dist = closest_approach(t, r1, v1, r2, v2)

    # Total satellite mass (hub + reaction wheels) for the bare satellite in the second sim
    sat_mass = config.SC_MASS_KG + sum(rw.mass for rw in run["parts"].rws.factory.rwList.values())
    print("Run 3/3: independent minimal sim (two bare spacecraft) ...")
    t2, ((r1b, v1b), (r2b, v2b)) = run_second_sim(r_s0, v_s0, r_d0, v_d0, sat_mass, t_end)
    t_ca2, d_ca2, dist2 = closest_approach(t2, r1b, v1b, r2b, v2b)

    tk = np.arange(0.0, t_end, REC_S)
    rk1, vk1 = kepler_propagate(mu, r_s0, v_s0, tk)
    rk2, vk2 = kepler_propagate(mu, r_d0, v_d0, tk)
    t_cak, d_cak, distk = closest_approach(tk, rk1, vk1, rk2, vk2)
    wall = time.perf_counter() - wall0

    below = dist < ALERT_KM * 1e3
    t_first_below = t[np.argmax(below)] if below.any() else np.nan
    passes = int(np.count_nonzero(np.diff(below.astype(int)) == 1) + int(below[0]))
    alerts = run["alerts"]
    bad_cols, n_values = frames_identical(ref["frames"], run["frames"])
    sat_traj_same = np.array_equal(np.array(ref["sat_log"].r_BN_N).view(np.uint64), r1.view(np.uint64))

    checks = [
        ("Closest approach within 2 orbits, a few km", f"{d_ca / 1e3:.3f} km at t = {t_ca:.1f} s "
         f"({t_ca / period:.3f} orbits; planned {CA_RADIAL_OFFSET_M / 1e3:.0f} km at {t_ca_plan:.0f} s)",
         t_ca < 2 * period and 1e3 <= d_ca <= ALERT_KM * 1e3),
        ("CA matches independent minimal sim", f"{d_ca2 / 1e3:.3f} km at {t_ca2:.1f} s; diff {abs(d_ca - d_ca2):.3f} m, "
         f"{abs(t_ca - t_ca2):.3f} s (limit {SECOND_SIM_TOL[0]:g} m, {SECOND_SIM_TOL[1]:g} s)",
         abs(d_ca - d_ca2) < SECOND_SIM_TOL[0] and abs(t_ca - t_ca2) < SECOND_SIM_TOL[1]),
        ("CA matches two-body Kepler propagation", f"{d_cak / 1e3:.3f} km at {t_cak:.1f} s; diff {abs(d_ca - d_cak):.0f} m, "
         f"{abs(t_ca - t_cak):.2f} s (limit {KEPLER_TOL[0]:g} m, {KEPLER_TOL[1]:g} s)",
         abs(d_ca - d_cak) < KEPLER_TOL[0] and abs(t_ca - t_cak) < KEPLER_TOL[1]),
        ("Proximity alert fires once, on time", f"{len(alerts)} alert(s)"
         + (f", at t = {alerts[0][0]:.0f} s ({alerts[0][1] / 1e3:.2f} km); first sample < {ALERT_KM:g} km at "
            f"{t_first_below:.0f} s" if alerts else ""),
         len(alerts) == 1 and 0.0 <= alerts[0][0] - t_first_below <= config.DYN_STEP_S),
        ("Satellite data unchanged by the debris", f"{len(bad_cols)} of {sum(len(df.columns) for df in ref['frames'].values())} "
         f"columns differ ({n_values} values); 1 s trajectory bit-identical: {'yes' if sat_traj_same else 'no'}",
         not bad_cols and sat_traj_same),
    ]
    info = [
        ("Passes below 10 km in the run", f"{passes} (min distance outside the CA pass "
         f"{dist[np.abs(t - t_ca) > 600].min() / 1e3:.1f} km)"),
        ("Relative speed at CA", f"{np.linalg.norm(v2 - v1, axis=1)[np.argmin(dist)]:.1f} m/s"),
        ("Absolute deviation from Kepler at CA", f"satellite {np.linalg.norm(r1[np.argmin(dist)] - rk1[np.argmin(distk)]):.0f} m, "
         f"debris {np.linalg.norm(r2[np.argmin(dist)] - rk2[np.argmin(distk)]):.0f} m (drag + Sun/Moon; common to "
         f"both, so the miss distance still matches)"),
        ("Wall-clock time", f"{wall:.1f} s for 3 simulations + Kepler"),
    ]

    print("\n" + "=" * 118)
    print(f"Mini prototype debris proximity  |  debris crossing at {CROSSING_DEG:g} deg, alert below {ALERT_KM:g} km, "
          f"{config.NUM_ORBITS} orbits")
    print("-" * 118)
    for name, detail, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:44s} {detail}")
    for name, detail in info:
        print(f"[info] {name:44s} {detail}")
    print("-" * 118)
    print(f"{sum(ok for _, _, ok in checks)} of {len(checks)} checks passed")
    print("=" * 118)

    plot(t, dist, t2, dist2, tk, distk, t_ca, d_ca, alerts, period)
    if args.viz:
        print(f"Vizard file: {run['viz_file']}")
    return 0 if all(ok for _, _, ok in checks) else 1


def plot(t, dist, t2, dist2, tk, distk, t_ca, d_ca, alerts, period):
    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    ax[0].semilogy(t / 60, dist / 1e3, label="prototype + debris")
    ax[0].axhline(ALERT_KM, color="red", ls="--", lw=1, label=f"alert {ALERT_KM:g} km")
    for k in range(1, config.NUM_ORBITS + 1):
        ax[0].axvline(k * period / 60, color="grey", lw=0.6)
    ax[0].set_ylabel("distance [km]")
    ax[0].set_xlabel("time [min] (grey: orbit boundaries)")
    ax[0].set_title("Satellite - debris distance", fontsize=10)
    ax[0].legend(fontsize=8)

    w = (t > t_ca - 150) & (t < t_ca + 150)
    ax[1].plot(t[w] - t_ca, dist[w] / 1e3, lw=2, label="prototype + debris")
    w2 = (t2 > t_ca - 150) & (t2 < t_ca + 150)
    ax[1].plot(t2[w2] - t_ca, dist2[w2] / 1e3, "--", label="independent minimal sim")
    wk = (tk > t_ca - 150) & (tk < t_ca + 150)
    ax[1].plot(tk[wk] - t_ca, distk[wk] / 1e3, ":", label="two-body Kepler")
    ax[1].axhline(ALERT_KM, color="red", ls="--", lw=1)
    ax[1].plot([0], [d_ca / 1e3], "k*", ms=10, label=f"CA {d_ca / 1e3:.2f} km")
    for ta, _ in alerts:
        ax[1].axvline(ta - t_ca, color="orange", lw=1.5, label=f"alert at t = {ta:.0f} s")
    ax[1].set_xlabel(f"time from closest approach [s] (CA at t = {t_ca:.1f} s)")
    ax[1].set_ylabel("distance [km]")
    ax[1].set_title("Close-approach pass", fontsize=10)
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(config.REPO_ROOT, "results", "proto_debris.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    raise SystemExit(main_check())
