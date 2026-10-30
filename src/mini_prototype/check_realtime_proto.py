"""
Real-time pacing check on the full mini prototype (feasibility check 4.2, re-run with EcmBattery).

Builds the unchanged prototype (main.build) and slaves it to wall-clock time with Basilisk's
ClockSynch (accelFactor 1.0), with the Windows timer set to 1 ms (timeBeginPeriod(1), restored
afterwards). A probe module records wall-clock time every step, the same method as
src/check_realtime.py:
    timing error = elapsed wall-clock time - elapsed simulated time  (+ = late, - = ahead)
    step jitter  = deviation of each wall-clock step interval from the nominal step
    overruns     = steps where ClockSynch reports it fell behind
    compute      = wall time from the start of a step (after the clock wait) to its end
PASS if the maximum lateness stays within 20 ms (KR 3.3 recovery budget).

Also measures how fast the prototype runs WITHOUT ClockSynch (3 orbits, x real time).

Task rate: the prototype normally runs its single task at 1 Hz (config.DYN_STEP_S = 1 s).
--rate-hz sets config.DYN_STEP_S before the build, so every module runs at that rate. At a rate
other than 1 Hz the physics after --duration seconds (battery SoC, pointing error) are also
compared with a 1 Hz run of the same length.

Run from the repo root with bsk-env active:
    python src\\mini_prototype\\check_realtime_proto.py                              1 Hz, 60 s (~70 s)
    python src\\mini_prototype\\check_realtime_proto.py --rate-hz 10 --duration 300  10 Hz, 300 s

Output: results\\proto_realtime.png (1 Hz) or results\\proto_realtime_<rate>hz.png
"""

import argparse
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
from recorders import INTERNAL, NAVIGATION  # noqa: E402

DURATION_S = 60.0
BUDGET_MS = 20.0              # KR 3.3
CLOCK_ACCURACY_MS = 1.0       # ClockSynch accuracy, matches the 1 ms timer
CLOCK_PRIORITY = 1000         # above every prototype module: wait first, then run the step
PROBE_PRIORITY = 999          # measure right after the wait
END_PROBE_PRIORITY = -10000   # after every module and recorder: end of the step's computation
SOC_TOL_PCT = 0.1             # physics vs 1 Hz: battery SoC within 0.1 percentage points
POINT_TOL = (0.25, 0.05)      # pointing error within 25 % or 0.05 deg of the 1 Hz value


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


def physics_at_end(frames):
    """Battery SoC [%] and sun-pointing error [deg] at the last logged sample."""
    soc = 100.0 * frames[INTERNAL]["battery_soc"].to_numpy()
    sig = frames[NAVIGATION][[f"att_err_sigma_BR_{i}" for i in (1, 2, 3)]].to_numpy()
    point = np.degrees(4.0 * np.arctan(np.linalg.norm(sig, axis=1)))
    return {"t": float(frames[INTERNAL]["time_s"].iloc[-1]), "soc": float(soc[-1]), "point": float(point[-1]),
            "point_mean_last100": float(point[frames[INTERNAL]["time_s"].to_numpy() >
                                              frames[INTERNAL]["time_s"].iloc[-1] - 100].mean())}


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
    end_probe = WallClockProbe()
    end_probe.ModelTag = "endProbe"
    sim.AddModelToTask(main.TASK, end_probe, END_PROBE_PRIORITY)

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
    compute_ms = (np.array(end_probe.wall_s) - np.array(probe.wall_s)) * 1000.0
    overruns = int(np.array(clock_rec.overrunCounter)[-1]) if len(clock_rec.times()) else -1
    return {
        "steps": len(sim_t),
        "max_late_ms": float(err_ms.max()),
        "max_ahead_ms": max(0.0, float(-err_ms.min())),  # first sample is the reference (error 0)
        "p99_abs_err_ms": float(np.percentile(np.abs(err_ms), 99)),
        "max_step_jitter_ms": float(np.abs(dt_err_ms).max()),
        "final_drift_ms": float(err_ms[-1]),
        "overruns": overruns,
        "compute_ms": compute_ms,
        "physics": physics_at_end(parts.recorder.datasets()),
        "t": sim_t,
        "err": err_ms,
        "dt_err": dt_err_ms,
    }


def run_free(duration_s=None):
    """Prototype run without ClockSynch (3 orbits by default); times only ExecuteSimulation."""
    sim, parts = build_sim()
    sim_time_s = config.NUM_ORBITS * parts.period_s if duration_s is None else duration_s
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(sim_time_s))
    t0 = time.perf_counter()
    sim.ExecuteSimulation()
    wall = time.perf_counter() - t0
    parts.env.unload()
    return sim_time_s, wall, parts.recorder.datasets()


def main_check(argv=None):
    global DURATION_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate-hz", type=float, default=1.0 / config.DYN_STEP_S, help="task rate [Hz] (default 1)")
    ap.add_argument("--duration", type=float, default=DURATION_S, help="real-time duration [s] (default 60)")
    args = ap.parse_args(argv)
    if sys.platform != "win32":
        print("This check sets the Windows timer resolution; run it on Windows.")
        return 1

    nominal_step = config.DYN_STEP_S
    DURATION_S = args.duration
    config.DYN_STEP_S = 1.0 / args.rate_hz
    rate = args.rate_hz
    try:
        print(f"Running the full prototype in real time for {DURATION_S:.0f} s "
              f"(task rate {rate:g} Hz). Avoid heavy use of the PC meanwhile.")
        ctypes.windll.winmm.timeBeginPeriod(1)
        try:
            r = run_realtime()
        finally:
            ctypes.windll.winmm.timeEndPeriod(1)
        sim_time_s, wall, _ = run_free()
        ref = None
        if abs(config.DYN_STEP_S - nominal_step) > 1e-12:
            config.DYN_STEP_S = nominal_step
            _, _, ref_frames = run_free(DURATION_S)
            ref = physics_at_end(ref_frames)
    finally:
        config.DYN_STEP_S = nominal_step
    speed = sim_time_s / wall
    c = r["compute_ms"]

    checks = [(f"Max late within {BUDGET_MS:.0f} ms", f"{r['max_late_ms']:.3f} ms", r["max_late_ms"] < BUDGET_MS)]
    if ref is not None:
        p = r["physics"]
        d_point = abs(p["point"] - ref["point"])
        checks += [
            ("Battery SoC after the run matches 1 Hz", f"{p['soc']:.4f} % vs {ref['soc']:.4f} % at {p['t']:.0f} s "
             f"(diff {abs(p['soc'] - ref['soc']):.4f} %-pt, limit {SOC_TOL_PCT:g})", abs(p["soc"] - ref["soc"]) < SOC_TOL_PCT),
            ("Pointing error after the run matches 1 Hz", f"{p['point']:.4f} deg vs {ref['point']:.4f} deg "
             f"(mean last 100 s {p['point_mean_last100']:.4f} vs {ref['point_mean_last100']:.4f}; limit "
             f"{POINT_TOL[0]:.0%} or {POINT_TOL[1]:g} deg)",
             d_point < max(POINT_TOL[0] * ref["point"], POINT_TOL[1])),
        ]
    ok = all(x[2] for x in checks)

    rows = [
        ("Task rate / steps", f"{rate:g} Hz / {r['steps']} steps over {DURATION_S:.0f} s"),
        ("Windows timer / ClockSynch accuracy", f"1 ms (timeBeginPeriod) / {CLOCK_ACCURACY_MS:g} ms"),
        ("Max late", f"{r['max_late_ms']:.3f} ms"),
        ("Max ahead", f"{r['max_ahead_ms']:.3f} ms"),
        ("p99 |timing error|", f"{r['p99_abs_err_ms']:.3f} ms"),
        ("Max step jitter", f"{r['max_step_jitter_ms']:.3f} ms"),
        ("Final drift", f"{r['final_drift_ms']:+.3f} ms"),
        ("ClockSynch overruns", f"{r['overruns']}"),
        ("Compute per step (mean / p99 / max)", f"{c.mean():.3f} / {np.percentile(c, 99):.3f} / {c.max():.3f} ms "
         f"({100 * c.mean() / (1000.0 / rate):.2f} % of the {1000.0 / rate:g} ms step)"),
        ("Without ClockSynch (3 orbits)", f"{sim_time_s:.0f} s simulated in {wall:.2f} s = {speed:,.0f}x real time"),
    ]
    print("\n" + "=" * 100)
    print(f"Mini prototype real-time pacing  |  EcmBattery, ClockSynch accelFactor 1.0, {rate:g} Hz")
    print("-" * 100)
    for name, value in rows:
        print(f"  {name:38s} {value}")
    print("-" * 100)
    for name, detail, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name:42s} {detail}")
    print("=" * 100)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    ax[0].plot(r["t"], r["err"], ".-", lw=1, ms=3 if r["steps"] < 1000 else 1, label="timing error (wall - sim)")
    ax[0].axhline(BUDGET_MS, color="red", ls="--", lw=1, label=f"{BUDGET_MS:.0f} ms budget")
    ax[0].axhline(0, color="grey", lw=0.6)
    ax[0].set_ylabel("late (+) / ahead (-) [ms]")
    ax[0].set_title(f"Mini prototype in real time: 1 ms timer, ClockSynch accuracy {CLOCK_ACCURACY_MS:g} ms, "
                    f"{rate:g} Hz  ({'PASS' if ok else 'FAIL'})", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].plot(r["t"][1:], r["dt_err"], ".-", lw=1, ms=3 if r["steps"] < 1000 else 1, color="C1")
    ax[1].axhline(0, color="grey", lw=0.6)
    ax[1].set_ylabel("step jitter [ms]")
    ax[2].plot(r["t"], c, ".", ms=2, color="C2")
    ax[2].set_ylabel("compute per step [ms]")
    ax[2].set_xlabel("simulated time [s]")
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.join(config.REPO_ROOT, "results"), exist_ok=True)
    name = "proto_realtime.png" if ref is None else f"proto_realtime_{rate:g}hz.png"
    out = os.path.join(config.REPO_ROOT, "results", name)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main_check())
