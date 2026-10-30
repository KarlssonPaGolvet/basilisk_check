"""
Monte Carlo check on the full mini prototype with Basilisk's Monte Carlo Controller.

20 runs of the unchanged prototype over 1 orbit, in parallel on all cores. Dispersed per run:
    initial attitude      uniform random Euler angles (0..2 pi) -> parts.sc.hub.sigma_BNInit
    initial body rate     normal, 0 +/- 0.002 rad/s per axis    -> parts.sc.hub.omega_BN_BInit
    battery initial SoC   uniform 0.5 .. 0.9                    -> parts.power.battery.initialSoc
    bus load              uniform -9 .. -4 W                    -> parts.power.bus_load.nodePowerOut
    sensor noise seeds    Controller.setShouldDisperseSeeds(True): a new RNGSeed for every task model

Checks:
    - all runs complete (no failures)
    - results archived: MonteCarlo.data plus run<i>.json (parameters, seeds) and run<i>.data for every run
    - dispersions applied: initial attitude, rate, SoC and bus load differ between runs
    - one chosen run re-run from its archived parameters is bit-identical to its archived data.
      Controller.reRunCases() drops the retention policies (it re-runs but saves nothing), so the
      re-run here re-executes the archived run<i>.json through the Controller's SimulationExecutor
      with retention kept, into a separate folder, and compares every retained array bit for bit.
Reports total time, time per run and an extrapolation to 500 runs.

Run from the repo root with bsk-env active:
    python src\\mini_prototype\\check_montecarlo_proto.py

Outputs: results\\proto_montecarlo.png, archive in results\\proto_MC\\ (git-ignored by *_MC/)
"""

import copy
import gzip
import json
import os
import pickle
import shutil
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from Basilisk.utilities import SimulationBaseClass, macros  # noqa: E402
from Basilisk.utilities.MonteCarlo.Controller import Controller, RetentionPolicy, SimulationExecutor  # noqa: E402
from Basilisk.utilities.MonteCarlo.Dispersions import (NormalVectorCartDispersion, UniformDispersion,  # noqa: E402
                                                       UniformEulerAngleMRPDispersion)

import config  # noqa: E402
import main  # noqa: E402

N_RUNS = 20
N_EXTRAPOLATE = 500
RERUN_CASE = 7
LOG_S = 10.0
ARCHIVE_DIR = os.path.join(config.REPO_ROOT, "results", "proto_MC")
RERUN_DIR = os.path.join(ARCHIVE_DIR, "rerun")
DISPERSED = ["parts.sc.hub.sigma_BNInit", "parts.sc.hub.omega_BN_BInit",
             "parts.power.battery.initialSoc", "parts.power.bus_load.nodePowerOut"]


# ---- functions the Controller calls in its worker processes (module level: picklable) ----
def create_scenario():
    t0 = time.perf_counter()
    sim = SimulationBaseClass.SimBaseClass()
    sim.CreateNewProcess(main.PROCESS).addTask(sim.CreateNewTask(main.TASK, config.dyn_step_ns()))
    sim.parts = main.build(sim)  # kept on the sim: dispersion paths and object lifetime
    rec = macros.sec2nano(LOG_S)
    sim.msgRecList = {"pack": sim.parts.power.battery.packOutMsg.recorder(rec),
                      "guid": sim.parts.fsw.att_guid_msg.recorder(rec),
                      "sc": sim.parts.sc.scStateOutMsg.recorder(rec)}
    for r in sim.msgRecList.values():
        sim.AddModelToTask(main.TASK, r)
    sim.build_s = time.perf_counter() - t0
    return sim


def execute_scenario(sim):
    t0 = time.perf_counter()
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(sim.parts.period_s))  # 1 orbit
    sim.ExecuteSimulation()
    sim.parts.env.unload()
    sim.exec_s = time.perf_counter() - t0


def run_timing(sim):
    return {"build_s": sim.build_s, "exec_s": sim.exec_s}


class _NullQueue:
    """Stand-in for the Controller's data-writer queue when re-executing one run directly."""

    def put(self, *_):
        pass


def load_run_data(path):
    with gzip.open(path) as f:
        return pickle.load(f)


def main_check():
    shutil.rmtree(ARCHIVE_DIR, ignore_errors=True)
    mc = Controller()
    mc.setSimulationFunction(create_scenario)
    mc.setExecutionFunction(execute_scenario)
    mc.setExecutionCount(N_RUNS)
    mc.setThreadCount(os.cpu_count())
    mc.setShouldDisperseSeeds(True)
    mc.setArchiveDir(ARCHIVE_DIR)
    mc.setShowProgressBar(False)
    mc.setVerbose(False)
    mc.addDispersion(UniformEulerAngleMRPDispersion(DISPERSED[0]))
    mc.addDispersion(NormalVectorCartDispersion(DISPERSED[1], [0.0, 0.0, 0.0], [0.002, 0.002, 0.002]))
    mc.addDispersion(UniformDispersion(DISPERSED[2], [0.5, 0.9]))
    mc.addDispersion(UniformDispersion(DISPERSED[3], [-9.0, -4.0]))
    policy = RetentionPolicy()
    policy.addMessageLog("pack", ["stateOfCharge", "packVoltage", "packCurrent"])
    policy.addMessageLog("guid", ["sigma_BR"])
    policy.addMessageLog("sc", ["sigma_BN", "omega_BN_B", "r_BN_N"])
    policy.addRetentionFunction(run_timing)
    mc.addRetentionPolicy(policy)

    print(f"Monte Carlo: {N_RUNS} runs x 1 orbit on {os.cpu_count()} processes ...")
    t0 = time.perf_counter()
    failures = mc.executeSimulations()
    total_s = time.perf_counter() - t0

    # ---- archive ----
    missing = [i for i in range(N_RUNS) for ext in (".json", ".data")
               if not os.path.exists(os.path.join(ARCHIVE_DIR, f"run{i}{ext}"))]
    has_controller = os.path.exists(os.path.join(ARCHIVE_DIR, "MonteCarlo.data"))
    loaded = Controller.load(ARCHIVE_DIR)
    data = [loaded.getRetainedData(i) for i in range(N_RUNS)]
    params = [loaded.getParameters(i) for i in range(N_RUNS)]
    n_seeds = sum(1 for k in params[0] if k.endswith(".RNGSeed"))
    distinct = {name: len({str(p[name]) for p in params}) for name in DISPERSED}
    seed_runs_distinct = len({tuple(v for k, v in sorted(p.items()) if k.endswith(".RNGSeed")) for p in params})
    sigma0 = np.array([d["messages"]["sc.sigma_BN"][0, 1:] for d in data])
    soc0 = np.array([d["messages"]["pack.stateOfCharge"][0, 1] for d in data])

    # ---- re-run one case from its archived parameters, retention kept ----
    os.makedirs(RERUN_DIR, exist_ok=True)
    rp = copy.deepcopy(loaded.simParams)
    rp.index = RERUN_CASE
    rp.shouldDisperseSeeds = False      # use the archived seeds
    rp.shouldArchiveParameters = False  # don't overwrite run<i>.json
    rp.icfilename = ""
    rp.filename = os.path.join(RERUN_DIR, f"run{RERUN_CASE}")
    with open(os.path.join(ARCHIVE_DIR, f"run{RERUN_CASE}.json")) as f:
        rp.modifications = json.load(f)
    t_re = time.perf_counter()
    ok_rerun, _ = SimulationExecutor()([rp, _NullQueue()])
    rerun_s = time.perf_counter() - t_re
    new = load_run_data(rp.filename + ".data") if ok_rerun else {"messages": {}}
    old = data[RERUN_CASE]
    diff_keys = [k for k, v in old["messages"].items()
                 if k not in new["messages"] or v.shape != new["messages"][k].shape
                 or not np.array_equal(np.ascontiguousarray(v).view(np.uint64),
                                       np.ascontiguousarray(new["messages"][k]).view(np.uint64))]
    n_vals = sum(v.size for v in old["messages"].values())

    # ---- timing ----
    build = np.array([d["custom"]["build_s"] for d in data])
    execs = np.array([d["custom"]["exec_s"] for d in data])
    per_run = total_s / N_RUNS
    extrap_s = per_run * N_EXTRAPOLATE

    checks = [
        ("All runs complete", f"{N_RUNS - len(failures)} of {N_RUNS} succeeded" + (f", failed: {failures}" if failures else ""),
         not failures),
        ("Results archived", f"MonteCarlo.data {'present' if has_controller else 'MISSING'}; run<i>.json + run<i>.data "
         f"for {N_RUNS - len({m for m in missing})} of {N_RUNS} runs ({len(missing)} files missing)",
         has_controller and not missing),
        ("Dispersions applied (all runs differ)", ", ".join(f"{n.split('.')[-1]} {distinct[n]}/{N_RUNS}" for n in DISPERSED)
         + f", seed sets {seed_runs_distinct}/{N_RUNS} ({n_seeds} seeds per run)",
         all(v == N_RUNS for v in distinct.values()) and seed_runs_distinct == N_RUNS),
        ("Dispersions reach the simulation", f"initial SoC {soc0.min() * 100:.1f}..{soc0.max() * 100:.1f} %, "
         f"{len({tuple(np.round(s, 12)) for s in sigma0})} distinct initial attitudes",
         len({tuple(np.round(s, 12)) for s in sigma0}) == N_RUNS and soc0.max() - soc0.min() > 0.1),
        (f"Run {RERUN_CASE} re-run bit-identical to archive", f"{len(old['messages'])} retained arrays, {n_vals} values, "
         f"{len(diff_keys)} differ" + (f" ({', '.join(diff_keys)})" if diff_keys else "") + f"; re-run took {rerun_s:.1f} s",
         bool(ok_rerun) and not diff_keys),
    ]
    rows = [
        ("Processes / runs", f"{os.cpu_count()} / {N_RUNS} x 1 orbit ({loaded_period(data):.0f} s simulated each)"),
        ("Total wall time", f"{total_s:.1f} s"),
        ("Per run (effective, total / runs)", f"{per_run:.2f} s"),
        ("Per run inside a worker", f"build {build.mean():.2f} s + execute {execs.mean():.2f} s "
         f"(+ ~1 s: the Controller's executor sleeps 1 s after saving each run)"),
        (f"Extrapolated to {N_EXTRAPOLATE} runs", f"{extrap_s:.0f} s = {extrap_s / 60:.1f} min on {os.cpu_count()} processes "
         f"(1 orbit each; scales ~linearly with orbits per run)"),
        ("Not dispersed", "the 6 coarse sun sensors' seeds: they are held by the CSS constellation, not task "
         "models, so setShouldDisperseSeeds() does not reach them"),
    ]

    print("\n" + "=" * 118)
    print(f"Mini prototype Monte Carlo  |  Basilisk Controller, {N_RUNS} runs x 1 orbit, {os.cpu_count()} processes")
    print("-" * 118)
    for name, value in rows:
        print(f"  {name:36s} {value}")
    print("-" * 118)
    for name, detail, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:42s} {detail}")
    print("-" * 118)
    print(f"{sum(ok for _, _, ok in checks)} of {len(checks)} checks passed")
    print("=" * 118)

    plot(data, params)
    return 0 if all(ok for _, _, ok in checks) else 1


def loaded_period(data):
    return float(data[0]["messages"]["pack.stateOfCharge"][-1, 0] * 1e-9)


def plot(data, params):
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i, d in enumerate(data):
        soc = d["messages"]["pack.stateOfCharge"]
        sig = d["messages"]["guid.sigma_BR"]
        t = soc[:, 0] * 1e-9 / 60.0
        ax[0].plot(t, 100 * soc[:, 1], lw=1)
        err = np.degrees(4.0 * np.arctan(np.linalg.norm(sig[:, 1:], axis=1)))
        ax[1].semilogy(sig[:, 0] * 1e-9 / 60.0, np.maximum(err, 1e-4), lw=1,
                       label=f"run {i}" if i < 10 else None)
    ax[0].set_ylabel("battery SoC [%]")
    ax[0].set_title(f"{len(data)} Monte Carlo runs: dispersed attitude, rate, SoC, bus load, noise seeds", fontsize=10)
    ax[1].set_ylabel("sun-pointing error [deg]")
    ax[1].set_xlabel("time [min]")
    ax[1].legend(fontsize=7, ncol=5)
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(config.REPO_ROOT, "results", "proto_montecarlo.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    raise SystemExit(main_check())
