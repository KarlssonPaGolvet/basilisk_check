"""
Mini prototype for the ASTRID feasibility check.

A small LEO satellite (500 km SSO) with three reaction wheels, a battery,
a body-mounted solar panel, thermal models, an IMU, star tracker,
magnetometer and coarse sun sensors, noisy simpleNav, SPICE ephemerides
(Sun, Earth, Moon), eclipse, WMM magnetic field, MSIS atmosphere and drag.
The panel is kept Sun-pointing by an MRP feedback law on the wheels.
The run covers a few orbits and records two data sets:

  * Internal sensors - power, wheels, temperatures, attitude sensors
  * Navigation       - simpleNav solution, ephemerides, eclipse, atmosphere, drag, field

Outputs (in results/mini_prototype/):
    internal_sensors.csv, navigation.csv, inventory.md, summary.png

Run from the repo root with the virtual environment active:
    python src/mini_prototype/main.py
"""

import os
import time
from types import SimpleNamespace

from Basilisk import __path__ as bsk_path
from Basilisk.utilities import SimulationBaseClass, macros

import config
from control import SunPointingControl
from environment import Environment
from inventory import save_datasets, write_inventory
from navigation import create_simple_nav
from plots import plot_summary
from power import PowerSystem
from recorders import INTERNAL, NAVIGATION, DataRecorder
from sensors import AttitudeSensors
from thermal import Thermal
from vehicle import ReactionWheels, create_spacecraft, set_initial_state

TASK = "simTask"
PROCESS = "simProcess"


def build(sim):
    """Create and connect all modules.

    Returns every component: the Python wrappers own the C++ modules and
    stand-alone messages, so they must stay alive for the whole run.
    """
    p = SimpleNamespace()
    p.sc = create_spacecraft(sim, TASK)
    p.env = Environment(sim, TASK, p.sc)
    p.period_s = set_initial_state(p.sc, p.env.mu)
    p.rws = ReactionWheels(sim, TASK, p.sc)
    p.nav = create_simple_nav(sim, TASK, p.sc, p.env.sun_msg)
    p.fsw = SunPointingControl(sim, TASK, p.nav, p.env.sun_ephem_msg, p.rws)
    p.power = PowerSystem(sim, TASK, p.sc, p.env, p.rws)
    p.thermal = Thermal(sim, TASK, p.sc, p.env, p.rws)
    p.sensors = AttitudeSensors(sim, TASK, p.sc, p.env)

    p.recorder = DataRecorder(sim, TASK, config.log_step_ns())
    p.recorder.setup(p.sc, p.env, p.rws, p.power, p.nav, p.fsw, p.thermal, p.sensors)
    return p


def main():
    wall_start = time.perf_counter()

    sim = SimulationBaseClass.SimBaseClass()
    sim.CreateNewProcess(PROCESS).addTask(sim.CreateNewTask(TASK, config.dyn_step_ns()))
    parts = build(sim)
    period_s = parts.period_s

    sim_time_s = config.NUM_ORBITS * period_s
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(sim_time_s))
    sim.ExecuteSimulation()
    wall_s = time.perf_counter() - wall_start
    parts.env.unload()

    recorder = parts.recorder
    frames = recorder.datasets()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_paths = save_datasets(frames, config.RESULTS_DIR)
    inv_path = os.path.join(config.RESULTS_DIR, "inventory.md")
    png_path = os.path.join(config.RESULTS_DIR, "summary.png")

    isd, nav = frames[INTERNAL], frames[NAVIGATION]
    soc = 100.0 * isd["battery_stored_energy_J"] / isd["battery_capacity_J"]
    eclipse_frac = (nav["eclipse_shadow_factor"] < 0.5).mean()
    run_info = {
        "Basilisk": bsk_path[0],
        "Epoch": config.EPOCH,
        "Orbit": f"{config.ALTITUDE_M / 1e3:.0f} km, i = {config.INCLINATION_DEG} deg",
        "Orbital period": f"{period_s / 60:.2f} min",
        "Orbits simulated": config.NUM_ORBITS,
        "Simulated time": f"{sim_time_s:.0f} s",
        "Dynamics step / log step": f"{config.DYN_STEP_S:g} s / {config.LOG_STEP_S:g} s",
        "Samples per data set": len(isd),
        "Time in eclipse": f"{100 * eclipse_frac:.1f} %",
        "Battery SoC start / min / end": f"{soc.iloc[0]:.1f} / {soc.min():.1f} / {soc.iloc[-1]:.1f} %",
        "Wall-clock time": f"{wall_s:.1f} s ({sim_time_s / wall_s:.0f}x real time)",
    }
    write_inventory(inv_path, recorder, frames, run_info)
    plot_summary(frames, parts.rws.num, parts.sensors.num_css, png_path)

    print("=" * 64)
    print("Mini prototype run")
    for k, v in run_info.items():
        print(f"  {k:<30}: {v}")
    for name, df in frames.items():
        print(f"  {name + ' data set':<30}: {df.shape[0]} rows x {df.shape[1]} cols -> {csv_paths[name]}")
    print(f"  {'Inventory':<30}: {inv_path}")
    print(f"  {'Summary plot':<30}: {png_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
