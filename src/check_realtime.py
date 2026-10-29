"""
Basilisk real-time pacing check (ASTRID feasibility check, step 4.2).

Runs a small spacecraft simulation slaved to wall-clock time with Basilisk's
ClockSynch module, at several step rates, and measures independently how well
simulated time keeps pace with real time.

Configurations tested (each rate x each setting):
  * step rates          : 1, 10, 50, 100, 200 Hz
  * ClockSynch accuracy : 10 ms (Basilisk default) and 1 ms (tightened)
  * Windows timer       : default (~15.6 ms) and 1 ms (timeBeginPeriod)  [Windows only]

Measured per run:
  * timing error  = (elapsed wall-clock time) - (elapsed simulated time)
                    positive = simulation is LATE, negative = AHEAD of real time
  * step jitter   = how far each step interval deviates from the nominal period
  * overruns      = steps where ClockSynch itself reports it fell behind

Budgets for reference: KR 3.3 recovery < 20 ms, KR 1.2 fallback < 30 ms.

Run from the repo root with the venv active (takes about 2 minutes):
    python src\\check_realtime.py            full matrix
    python src\\check_realtime.py --quick    default settings only (~30 s)

Outputs:
    results\\realtime_summary.csv
    results\\realtime_error.png
"""

import argparse
import csv
import ctypes
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")  # save plots to file, don't block
import matplotlib.pyplot as plt

from Basilisk.architecture import sysModel
from Basilisk.simulation import spacecraft, simpleNav, simSynch
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody

RATES_HZ = [1, 10, 50, 100, 200]
BUDGET_MS = 20.0  # KR 3.3


class WallClockProbe(sysModel.SysModel):
    """Python module that records wall-clock time at every simulation step."""

    def __init__(self):
        super(WallClockProbe, self).__init__()
        self.sim_ns = []
        self.wall_s = []

    def Reset(self, CurrentSimNanos):
        self.sim_ns = []
        self.wall_s = []

    def UpdateState(self, CurrentSimNanos):
        self.wall_s.append(time.perf_counter())
        self.sim_ns.append(CurrentSimNanos)


def run_case(rate_hz, accuracy_ms, duration_s):
    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("proc")
    proc.addTask(sim.CreateNewTask("task", macros.sec2nano(1.0 / rate_hz)))

    # Small but representative load: spacecraft, Earth gravity, navigation sensor
    sc = spacecraft.Spacecraft()
    sc.ModelTag = "rtSat"
    sc.hub.mHub = 20.0
    sim.AddModelToTask("task", sc, 300)

    grav = simIncludeGravBody.gravBodyFactory()
    earth = grav.createEarth()
    earth.isCentralBody = True
    grav.addBodiesTo(sc)

    oe = orbitalMotion.ClassicElements()
    oe.a, oe.e, oe.i = 6871.0e3, 0.001, 97.4 * macros.D2R
    oe.Omega, oe.omega, oe.f = 0.0, 0.0, 0.0
    rN, vN = orbitalMotion.elem2rv(earth.mu, oe)
    sc.hub.r_CN_NInit = rN
    sc.hub.v_CN_NInit = vN

    nav = simpleNav.SimpleNav()
    nav.ModelTag = "nav"
    nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    sim.AddModelToTask("task", nav, 250)

    # Real-time pacing
    clock = simSynch.ClockSynch()
    clock.ModelTag = "clock"
    clock.accelFactor = 1.0
    clock.accuracyNanos = int(accuracy_ms * 1e6)
    sim.AddModelToTask("task", clock, 200)

    # Independent measurement, runs right after the clock in every step
    probe = WallClockProbe()
    probe.ModelTag = "probe"
    sim.AddModelToTask("task", probe, 100)

    clockRec = clock.clockOutMsg.recorder()
    sim.AddModelToTask("task", clockRec, 50)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(duration_s))
    sim.ExecuteSimulation()

    sim_t = (np.array(probe.sim_ns) - probe.sim_ns[0]) * 1e-9
    wall_t = np.array(probe.wall_s) - probe.wall_s[0]
    err_ms = (wall_t - sim_t) * 1000.0
    dt_err_ms = (np.diff(wall_t) - 1.0 / rate_hz) * 1000.0
    overruns = int(np.array(clockRec.overrunCounter)[-1]) if len(clockRec.times()) else -1

    return {
        "rate_hz": rate_hz,
        "accuracy_ms": accuracy_ms,
        "steps": len(sim_t),
        "max_late_ms": float(err_ms.max()),
        "max_ahead_ms": float(-err_ms.min()),
        "p99_abs_err_ms": float(np.percentile(np.abs(err_ms), 99)),
        "max_step_jitter_ms": float(np.abs(dt_err_ms).max()) if len(dt_err_ms) else 0.0,
        "final_drift_ms": float(err_ms[-1]),
        "overruns": overruns,
        "keeps_pace": bool(err_ms.max() < BUDGET_MS),
        "_t": sim_t,
        "_err": err_ms,
    }


def set_windows_timer(ms):
    """Request a finer Windows timer resolution (no effect on other OSes)."""
    if sys.platform != "win32":
        return False
    ctypes.windll.winmm.timeBeginPeriod(ms)
    return True


def reset_windows_timer(ms):
    if sys.platform == "win32":
        ctypes.windll.winmm.timeEndPeriod(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="default settings only")
    args = ap.parse_args()

    configs = [("default timer", 10.0)]
    if not args.quick:
        configs += [("default timer", 1.0)]
        if sys.platform == "win32":
            configs += [("1 ms timer", 10.0), ("1 ms timer", 1.0)]

    est = len(configs) * (10 + 5 * (len(RATES_HZ) - 1))
    print(f"Platform: {sys.platform}. Running {len(configs) * len(RATES_HZ)} cases, about {est} s.")
    print("Avoid heavy use of the PC while it runs.\n")

    results = []
    for timer_label, acc_ms in configs:
        fine = timer_label == "1 ms timer" and set_windows_timer(1)
        try:
            for rate in RATES_HZ:
                dur = 10.0 if rate == 1 else 5.0
                r = run_case(rate, acc_ms, dur)
                r["timer"] = timer_label
                results.append(r)
                print(f"[{timer_label:13s} | accuracy {acc_ms:4.1f} ms | {rate:3d} Hz] "
                      f"late max {r['max_late_ms']:7.2f} ms, ahead max {r['max_ahead_ms']:6.2f} ms, "
                      f"step jitter max {r['max_step_jitter_ms']:7.2f} ms, overruns {r['overruns']:4d} "
                      f"-> {'OK' if r['keeps_pace'] else 'NOT OK'}")
        finally:
            if fine:
                reset_windows_timer(1)

    # ---- Summary table ----
    print("\n" + "=" * 100)
    print(f"{'Timer':14s}{'Accuracy':>10s}{'Rate':>7s}{'Late max':>11s}{'Ahead max':>11s}"
          f"{'p99 |err|':>11s}{'Jitter max':>12s}{'Overruns':>10s}{'  <20 ms?':>10s}")
    for r in results:
        print(f"{r['timer']:14s}{r['accuracy_ms']:8.1f}ms{r['rate_hz']:5d}Hz{r['max_late_ms']:9.2f}ms"
              f"{r['max_ahead_ms']:9.2f}ms{r['p99_abs_err_ms']:9.2f}ms{r['max_step_jitter_ms']:10.2f}ms"
              f"{r['overruns']:10d}{'   yes' if r['keeps_pace'] else '    NO'}")
    print("=" * 100)
    for label, acc in configs:
        ok = [r["rate_hz"] for r in results if r["timer"] == label and r["accuracy_ms"] == acc and r["keeps_pace"]]
        print(f"Highest rate within {BUDGET_MS:.0f} ms ({label}, accuracy {acc:.0f} ms): "
              f"{max(ok) if ok else 'none'} Hz")

    # ---- Save CSV ----
    os.makedirs("results", exist_ok=True)
    keys = [k for k in results[0] if not k.startswith("_")]
    with open(os.path.join("results", "realtime_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in keys})

    # ---- Plot: timing error over time, one panel per configuration ----
    fig, axes = plt.subplots(len(configs), 1, figsize=(10, 3.2 * len(configs)), sharex=False, squeeze=False)
    for ax, (label, acc) in zip(axes[:, 0], configs):
        for r in results:
            if r["timer"] == label and r["accuracy_ms"] == acc:
                ax.plot(r["_t"], r["_err"], label=f"{r['rate_hz']} Hz", lw=1)
        ax.axhline(BUDGET_MS, color="red", ls="--", lw=1, label="20 ms budget")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"{label}, ClockSynch accuracy {acc:.0f} ms", fontsize=10)
        ax.set_ylabel("late (+) / ahead (−) [ms]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=6, loc="upper right")
    axes[-1, 0].set_xlabel("simulated time [s]")
    fig.tight_layout()
    out = os.path.join("results", "realtime_error.png")
    fig.savefig(out, dpi=120)
    print(f"\nSaved results\\realtime_summary.csv and {out}")


if __name__ == "__main__":
    main()