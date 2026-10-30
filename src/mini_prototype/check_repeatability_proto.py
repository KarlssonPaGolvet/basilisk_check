"""
Repeatability check on the full mini prototype.

Runs the unchanged prototype (main.build, 3 orbits) several times:
    A1, A2  same seeds (config.RNG_SEED = 42), twice in the SAME process
    B       same seeds, in a separate process
    C       different seeds (config.RNG_SEED = 43), separate process
Every run's two data sets are stored as raw float64 (pickle, not CSV) and compared bit for bit.

Checks:
    - same seeds give bit-identical data, both across processes (A1 vs B) and in one process (A1 vs A2)
    - different seeds change every noisy sensor output
    - with different seeds nothing changes outside the expected column groups
      (attitude, wheels and power change through the closed loop: nav noise -> controller)
Reported as info, not pass/fail: with different seeds the orbit and environment differ only by
round-off (attitude-dependent body-frame drag), amplified by the steep eclipse penumbra.
Also lists every random seed in the prototype and whether the module actually draws random numbers.

Run from the repo root with bsk-env active:
    python src\\mini_prototype\\check_repeatability_proto.py
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from Basilisk.utilities import SimulationBaseClass, macros

import config
import main
from recorders import INTERNAL, NAVIGATION

SEED_SAME, SEED_OTHER = 42, 43

# Column classification for the different-seed comparison
NOISE_PREFIXES = ("nav_", "imu_", "st_q_", "mag_B_S_", "css", "meas_")        # sensor outputs with RNG noise
ENV_PREFIXES = ("time_s", "true_r_", "true_v_", "spice_", "sun_", "moon_", "earth_", "eclipse_",
                "atmo_", "mag_field_N_")                                      # orbit / environment
CLOSED_LOOP_PREFIXES = ("true_sigma_", "true_omega_", "att_err_", "rw", "battery_", "solar_panel_",
                        "bus_load_", "sensor_temp_", "drag_force_B_", "drag_torque_")  # via nav -> control (body frame)
# Modules that actually draw random numbers in this prototype
NOISY_CLASSES = {"SimpleNav", "ImuSensor", "StarTracker", "Magnetometer", "CoarseSunSensor", "TempMeasurement"}


def classify(col):
    for name, prefixes in (("noise", NOISE_PREFIXES), ("environment", ENV_PREFIXES),
                           ("closed loop", CLOSED_LOOP_PREFIXES)):
        if col.startswith(prefixes):
            return name
    return "unclassified"


def seed_table(sim, parts):
    """Every module with an RNGSeed: task models plus the CSS sensors held by the constellation."""
    rows = []
    for task in sim.TaskList:
        for model in task.TaskModels:
            if hasattr(model, "RNGSeed"):
                rows.append({"module": getattr(model, "ModelTag", "") or type(model).__name__,
                             "class": type(model).__name__, "seed": int(model.RNGSeed), "in_task": True})
    for css in parts.sensors.css_list:  # not task models: updated by the CSS constellation
        rows.append({"module": css.ModelTag, "class": type(css).__name__, "seed": int(css.RNGSeed),
                     "in_task": False})
    for r in rows:
        r["noisy"] = r["class"] in NOISY_CLASSES
    return rows


def worker(seed, n_runs, out_dir):
    """Run the prototype n_runs times in this process with config.RNG_SEED = seed."""
    config.RNG_SEED = seed
    for i in range(n_runs):
        sim = SimulationBaseClass.SimBaseClass()
        sim.CreateNewProcess(main.PROCESS).addTask(sim.CreateNewTask(main.TASK, config.dyn_step_ns()))
        parts = main.build(sim)
        if i == 0:
            with open(os.path.join(out_dir, f"seeds_{seed}.json"), "w") as f:
                json.dump(seed_table(sim, parts), f)
        sim.InitializeSimulation()
        sim.ConfigureStopTime(macros.sec2nano(config.NUM_ORBITS * parts.period_s))
        sim.ExecuteSimulation()
        parts.env.unload()
        pd.to_pickle(parts.recorder.datasets(), os.path.join(out_dir, f"run_{seed}_{i}.pkl"))


def spawn(seed, n_runs, out_dir, tag):
    subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", str(seed), str(n_runs), out_dir],
                   check=True, cwd=config.REPO_ROOT)
    return [pd.read_pickle(os.path.join(out_dir, f"run_{seed}_{i}.pkl")) for i in range(n_runs)]


def compare(fa, fb):
    """Per-column comparison of two runs. Returns {column: (bit_identical, max relative difference)}."""
    out = {}
    for name in (INTERNAL, NAVIGATION):
        a, b = fa[name], fb[name]
        assert list(a.columns) == list(b.columns) and len(a) == len(b), "data set layouts differ"
        for col in a.columns:
            x, y = a[col].to_numpy(dtype=np.float64), b[col].to_numpy(dtype=np.float64)
            same = np.array_equal(x.view(np.uint64), y.view(np.uint64))  # bit for bit
            scale = max(np.abs(x).max(), np.abs(y).max(), 1e-300)
            out[f"{name}:{col}"] = (same, float(np.abs(x - y).max() / scale))
    return out


def main_check():
    with tempfile.TemporaryDirectory() as tmp:
        print("Running the prototype 4 times (3 orbits each, 3 processes) ...")
        a1, a2 = spawn(SEED_SAME, 2, tmp, "A")
        os.rename(os.path.join(tmp, f"run_{SEED_SAME}_0.pkl"), os.path.join(tmp, "A1.pkl"))
        (b,) = spawn(SEED_SAME, 1, tmp, "B")
        (c,) = spawn(SEED_OTHER, 1, tmp, "C")
        with open(os.path.join(tmp, f"seeds_{SEED_SAME}.json")) as f:
            seeds_a = json.load(f)
        with open(os.path.join(tmp, f"seeds_{SEED_OTHER}.json")) as f:
            seeds_c = json.load(f)

    n_values = sum(df.size for df in a1.values())
    ab, aa, ac = compare(a1, b), compare(a1, a2), compare(a1, c)
    diff_ab = [k for k, (same, _) in ab.items() if not same]
    diff_aa = [k for k, (same, _) in aa.items() if not same]

    groups = {}
    for key, (same, rel) in ac.items():
        g = classify(key.split(":", 1)[1])
        st = groups.setdefault(g, {"cols": 0, "changed": 0, "max_rel": 0.0, "unchanged_cols": []})
        st["cols"] += 1
        st["changed"] += int(not same)
        st["max_rel"] = max(st["max_rel"], rel)
        if same:
            st["unchanged_cols"].append(key.split(":", 1)[1])
        else:
            st.setdefault("changed_cols", []).append((key.split(":", 1)[1], rel))
    noise, env = groups.get("noise"), groups.get("environment")
    uncl = groups.get("unclassified", {"cols": 0, "changed": 0, "max_rel": 0.0})

    checks = [
        ("Same seeds, separate processes: bit-identical", f"{len(ab)} columns, {n_values} values, "
         f"{len(diff_ab)} columns differ", not diff_ab),
        ("Same seeds, same process: bit-identical", f"{len(aa)} columns, {len(diff_aa)} columns differ", not diff_aa),
        ("Different seeds: every noisy sensor output changes", f"{noise['changed']} of {noise['cols']} noise columns "
         f"changed (max rel. diff {noise['max_rel']:.2e})"
         + (f"; unchanged: {', '.join(noise['unchanged_cols'])}" if noise["unchanged_cols"] else ""),
         noise["changed"] == noise["cols"]),
        ("Different seeds: no unexplained differences", f"{uncl['changed']} unclassified columns changed",
         uncl["changed"] == 0),
    ]
    # Not pass/fail: attitude-dependent drag (computed in the body frame) leaves round-off differences
    # in the orbit, and the steep penumbra transition in the eclipse model amplifies them.
    info = [
        ("Different seeds: orbit/environment", f"{env['changed']} of {env['cols']} columns not bit-identical, "
         f"max rel. diff {env['max_rel']:.2e} (round-off via closed loop, amplified by eclipse)"),
    ]

    print("\n" + "=" * 116)
    print("Mini prototype repeatability  |  3 orbits per run, all recorded data compared bit for bit")
    print("-" * 116)
    for name, detail, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:52s} {detail}")
    for name, detail in info:
        print(f"[info] {name:52s} {detail}")
    print("-" * 116)
    print(f"{sum(ok for _, _, ok in checks)} of {len(checks)} checks passed")
    print("=" * 116)

    print("\nDifferent seeds (A vs C), by column group:")
    print(f"  {'group':14s}{'columns':>9s}{'changed':>9s}{'max rel. diff':>15s}  meaning")
    meaning = {"noise": "sensor outputs that draw random numbers",
               "closed loop": "attitude/wheels/power/thermal: nav noise reaches the controller",
               "environment": "orbit, ephemerides, eclipse, atmosphere, magnetic field",
               "unclassified": "-"}
    for g in ("noise", "closed loop", "environment", "unclassified"):
        if g in groups:
            st = groups[g]
            print(f"  {g:14s}{st['cols']:9d}{st['changed']:9d}{st['max_rel']:15.2e}  {meaning[g]}")
    print("\nEnvironment columns that changed with the seeds (relative difference):")
    for col, rel in sorted(env.get("changed_cols", []), key=lambda x: -x[1]):
        print(f"  {col:32s}{rel:10.2e}")

    print("\nRandom seeds in the prototype (config.RNG_SEED = 42; C used 43):")
    print(f"  {'module':22s}{'class':26s}{'seed (A)':>12s}{'seed (C)':>12s}  {'draws random numbers':22s} in task list")
    is_logger = lambda r: r["module"].startswith("Rec:") or r["class"] == "PythonVariableLogger"
    for ra, rc in zip(seeds_a, seeds_c):
        if is_logger(ra):
            continue
        print(f"  {ra['module']:22s}{ra['class']:26s}{ra['seed']:12d}{rc['seed']:12d}  "
              f"{'yes' if ra['noisy'] else 'no (seed unused)':22s} {'yes' if ra['in_task'] else 'no (via constellation)'}")
    n_loggers = sum(is_logger(r) for r in seeds_a)
    print(f"  ({n_loggers} message recorders / variable loggers also carry the default seed {seeds_a[0]['seed']}; "
          f"they draw no random numbers)")
    noisy = [r for r in seeds_a if r["noisy"]]
    print(f"  -> {len(noisy)} modules draw random numbers, all seeded from config.RNG_SEED; the other "
          f"{len(seeds_a) - len(noisy)} keep the Basilisk default seed and are deterministic.")
    return 0 if all(ok for _, _, ok in checks) else 1


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        raise SystemExit(main_check())
