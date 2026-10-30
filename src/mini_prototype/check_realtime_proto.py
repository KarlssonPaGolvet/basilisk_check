"""
Real-time pacing check on the full mini prototype (feasibility check 4.2, re-run with EcmBattery).

Builds the unchanged prototype (main.build) and slaves it to wall-clock time with Basilisk's
ClockSynch (accelFactor 1.0) for 60 s of simulated time at its normal task rate, with the
Windows timer set to 1 ms (timeBeginPeriod(1), restored afterwards). A probe module records
wall-clock time every step, the same method as src/check_realtime.py:
    timing error = elapsed wall-clock time - elapsed simulated time  (+ = late, - = ahead)
    step jitter  = deviation of each wall-clock step interval from the nominal step
    overruns     = steps where ClockSynch reports it fell behind
PASS if the maximum lateness stays within 20 ms (KR 3.3 recovery budget).

Also measures how fast the prototype runs WITHOUT ClockSynch (3 orbits, x real time).

Run from the repo root with bsk-env active (about 70 s):
    python src\\mini_prototype\\check_realtime_proto.py

Output: results\\proto_realtime.png
"""

import ctypes
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from Basilisk.architecture import sysModel  # noqa: E402
from Basilisk.simulation import simSynch  # noqa: E402
from Basilisk.utilities import SimulationBaseClass, macros  # noqa: E402

import config  # noqa: E402
import main  # noqa: E402

DURATION_S = 60.0
BUDGET_MS = 20.0              # KR 3.3
CLOCK_ACCURACY_MS = 1.0       # ClockSynch accuracy, matches the 1 ms timer
CLOCK_PRIORITY = 1000         # above every prototype module: wait first, then run the step
PROBE_PRIORITY = 999          # measure right after the wait


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


def build_sim():
    sim = SimulationBaseClass.SimBaseClass()
    sim.CreateNewProcess(main.PROCESS).addTask(sim.CreateNewTask(main.TASK, config.dyn_step_ns()))
    parts = main.build(sim)
    return sim, parts


def run_realtime():
    sim, parts = build_sim()
    clock = simSynch.ClockSynch()
    clock.ModelTag = "clock"
    clock.accelFactor = 1.0
    clock.accuracyNanos = int(CLOCK_ACCURACY_MS * 1e6)
    sim.AddModelToTask(main.TASK, clock, CLOCK_PRIORITY)

    probe = WallClockProbe()
    probe.ModelTag = "probe"
    sim.AddModelToTask(main.TASK, probe, PROBE_PRIORITY)

    clock_rec = clock.clockOutMsg.recorder()
    sim.AddModelToTask(main.TASK, clock_rec)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(DURATION_S))
    sim.ExecuteSimulation()
    parts.env.unload()

    sim_t = (np.array(probe.sim_ns) - probe.sim_ns[0]) * 1e-9
    wall_t = np.array(probe.wall_s) - probe.wall_s[0]
    err_ms = (wall_t - sim_t) * 1000.0
    dt_err_ms = (np.diff(wall_t) - config.DYN_STEP_S) * 1000.0
    overruns = int(np.array(clock_rec.overrunCounter)[-1]) if len(clock_rec.times()) else -1
    return {
        "steps": len(sim_t),
        "max_late_ms": float(err_ms.max()),
        "max_ahead_ms": max(0.0, float(-err_ms.min())),  # first sample is the reference (error 0)
        "p99_abs_err_ms": float(np.percentile(np.abs(err_ms), 99)),
        "max_step_jitter_ms": float(np.abs(dt_err_ms).max()),
        "final_drift_ms": float(err_ms[-1]),
        "overruns": overruns,
        "t": sim_t,
        "err": err_ms,
        "dt_err": dt_err_ms,
    }


def run_free():
    """Full 3-orbit prototype run without ClockSynch; times only ExecuteSimulation."""
    sim, parts = build_sim()
    sim_time_s = config.NUM_ORBITS * parts.period_s
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(sim_time_s))
    t0 = time.perf_counter()
    sim.ExecuteSimulation()
    wall = time.perf_counter() - t0
    parts.env.unload()
    return sim_time_s, wall


def main_check():
    if sys.platform != "win32":
        print("This check sets the Windows timer resolution; run it on Windows.")
        return 1

    print(f"Running the full prototype in real time for {DURATION_S:.0f} s "
          f"(task rate {1.0 / config.DYN_STEP_S:g} Hz). Avoid heavy use of the PC meanwhile.")
    ctypes.windll.winmm.timeBeginPeriod(1)
    try:
        r = run_realtime()
    finally:
        ctypes.windll.winmm.timeEndPeriod(1)

    sim_time_s, wall = run_free()
    speed = sim_time_s / wall
    ok = r["max_late_ms"] < BUDGET_MS

    rows = [
        ("Task rate / steps", f"{1.0 / config.DYN_STEP_S:g} Hz / {r['steps']} steps over {DURATION_S:.0f} s"),
        ("Windows timer / ClockSynch accuracy", f"1 ms (timeBeginPeriod) / {CLOCK_ACCURACY_MS:g} ms"),
        ("Max late", f"{r['max_late_ms']:.3f} ms"),
        ("Max ahead", f"{r['max_ahead_ms']:.3f} ms"),
        ("p99 |timing error|", f"{r['p99_abs_err_ms']:.3f} ms"),
        ("Max step jitter", f"{r['max_step_jitter_ms']:.3f} ms"),
        ("Final drift", f"{r['final_drift_ms']:+.3f} ms"),
        ("ClockSynch overruns", f"{r['overruns']}"),
        ("Without ClockSynch (3 orbits)", f"{sim_time_s:.0f} s simulated in {wall:.2f} s = {speed:,.0f}x real time"),
    ]
    print("\n" + "=" * 90)
    print("Mini prototype real-time pacing  |  EcmBattery, ClockSynch accelFactor 1.0")
    print("-" * 90)
    for name, value in rows:
        print(f"  {name:38s} {value}")
    print("-" * 90)
    print(f"[{'PASS' if ok else 'FAIL'}] Max late within {BUDGET_MS:.0f} ms: {r['max_late_ms']:.3f} ms")
    print("=" * 90)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    ax[0].plot(r["t"], r["err"], ".-", lw=1, ms=3, label="timing error (wall - sim)")
    ax[0].axhline(BUDGET_MS, color="red", ls="--", lw=1, label=f"{BUDGET_MS:.0f} ms budget")
    ax[0].axhline(0, color="grey", lw=0.6)
    ax[0].set_ylabel("late (+) / ahead (-) [ms]")
    ax[0].set_title(f"Mini prototype in real time: 1 ms timer, ClockSynch accuracy {CLOCK_ACCURACY_MS:g} ms, "
                    f"{1.0 / config.DYN_STEP_S:g} Hz  ({'PASS' if ok else 'FAIL'})", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].plot(r["t"][1:], r["dt_err"], ".-", lw=1, ms=3, color="C1")
    ax[1].axhline(0, color="grey", lw=0.6)
    ax[1].set_ylabel("step jitter [ms]")
    ax[1].set_xlabel("simulated time [s]")
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.join(config.REPO_ROOT, "results"), exist_ok=True)
    out = os.path.join(config.REPO_ROOT, "results", "proto_realtime.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main_check())
