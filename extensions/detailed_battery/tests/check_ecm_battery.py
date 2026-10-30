"""
Basilisk check of the detailed_battery EcmBattery extension (feasibility check 5.3).

Runs the C++ battery module inside normal Basilisk simulations, connected to
standard Basilisk power nodes, and injects battery faults with Basilisk events.

Scenario 1 - faults. Pack: 2S2P, 3.0 Ah cells (~43 Wh), start at 80 % SoC, 5 W constant load
(at 10 W the faults drain the pack before the overload test - itself a realistic result).
Timeline (1 s steps, 2.5 h):
   1800 s  internal resistance x2                 (ageing)
   3600 s  one parallel string fails (open)       (cell drop-off -> capacity loss)
   5400 s  one series cell per string shorts      (voltage drop-off)
   7200 s  extra 300 W load for 60 s              (overload, held at the BMS cutoff limit)
Scenario 2 - charging: +20 W source, start at 90 % SoC (CC phase, then CV at maxCellVoltage).
Scenario 3 - depletion: 5 W load, start at 1 % SoC.

Run from the repo root with bsk-env active, after installing the extension wheel:
    python extensions\\detailed_battery\\tests\\check_ecm_battery.py
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Basilisk.simulation import simplePowerSink
from Basilisk.utilities import SimulationBaseClass, macros

from detailed_battery import ecmBattery

LOAD_W = 5.0
T_RES, T_STRING, T_SHORT, T_OVER, T_END = 1800.0, 3600.0, 5400.0, 7200.0, 9000.0
OVERLOAD_W = 300.0
OVERLOAD_S = 60.0
CHARGE_W = 20.0
STEP_S = 1.0


def set_attr(obj, name, value):
    def action(_s):
        setattr(obj, name, value)
    return action


def run_scenario(node_power_w, initial_soc, t_end, events=()):
    """Run one sim: a constant power node (+ = source, - = load) and an optional extra load.

    events: (name, time [s], target, attribute, value), target is "battery" or "extra".
    Returns (battery, pack log dict, standard-message log dict).
    """
    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("proc")
    proc.addTask(sim.CreateNewTask("task", macros.sec2nano(STEP_S)))

    node = simplePowerSink.SimplePowerSink()
    node.ModelTag = "powerNode"
    node.nodePowerOut = node_power_w
    sim.AddModelToTask("task", node, 100)

    extra = simplePowerSink.SimplePowerSink()
    extra.ModelTag = "extraLoad"
    extra.nodePowerOut = 0.0
    sim.AddModelToTask("task", extra, 100)

    bat = ecmBattery.EcmBattery()
    bat.ModelTag = "detailedBattery"
    bat.nSeries, bat.nParallel = 2, 2
    bat.cellCapacityAh = 3.0
    bat.initialSoc = initial_soc
    bat.addPowerNodeToModel(node.nodePowerOutMsg)
    bat.addPowerNodeToModel(extra.nodePowerOutMsg)
    sim.AddModelToTask("task", bat, 50)

    rec = macros.sec2nano(STEP_S)
    packLog = bat.packOutMsg.recorder(rec)
    statLog = bat.batPowerOutMsg.recorder(rec)
    sim.AddModelToTask("task", packLog)
    sim.AddModelToTask("task", statLog)

    targets = {"battery": bat, "extra": extra}
    for name, t, target, attr, value in events:
        sim.createNewEvent(name, rec, True, conditionTime=macros.sec2nano(t),
                           actionFunction=set_attr(targets[target], attr, value))

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(t_end))
    sim.ExecuteSimulation()

    fields = ["packVoltage", "packCurrent", "stateOfCharge", "internalResistance", "capacityAh",
              "openCircuitVoltage", "polarizationVoltage", "cellVoltage", "powerDelivered", "heatLoss",
              "powerLimited", "underVoltage", "chargeLimited", "depleted", "activeStrings",
              "activeSeriesCells"]
    pack = {f: np.array(getattr(packLog, f)) for f in fields}
    pack["t"] = packLog.times() * macros.NANO2SEC
    stat = {"storageCapacity": np.array(statLog.storageCapacity)}
    return bat, pack, stat


def fault_checks():
    events = [
        ("resistance", T_RES, "battery", "resistanceFactor", 2.0),
        ("string", T_STRING, "battery", "failedStrings", 1),
        ("short", T_SHORT, "battery", "shortedCellsPerString", 1),
        ("overloadOn", T_OVER, "extra", "nodePowerOut", -OVERLOAD_W),
        ("overloadOff", T_OVER + OVERLOAD_S, "extra", "nodePowerOut", 0.0),
    ]
    bat, p, stat = run_scenario(-LOAD_W, 0.8, T_END, events)
    t, V, I, soc = p["t"], p["packVoltage"], p["packCurrent"], p["stateOfCharge"]
    R0, cap, ocv, v1 = p["internalResistance"], p["capacityAh"], p["openCircuitVoltage"], p["polarizationVoltage"]
    pdel, lim, uv, heat, vcell = p["powerDelivered"], p["powerLimited"], p["underVoltage"], p["heatLoss"], p["cellVoltage"]
    capWs = stat["storageCapacity"]

    idx = lambda tt: int(np.searchsorted(t, tt))
    before = lambda tt: idx(tt) - 2
    after = lambda tt: idx(tt) + 2

    ocv0 = 2 * bat.ocvFromSoc(0.8)

    # Coulomb counting: each sample is the start of a 1 s step, so the SoC drop from
    # sample a to sample b equals the sum of I over samples a..b-1 (times 1 s).
    i_a, i_b = idx(1.0), idx(T_RES)
    soc_drop = soc[i_a] - soc[i_b]
    soc_coulomb = float(np.sum(I[i_a:i_b]) * STEP_S / (3600.0 * cap[i_a]))
    coulomb_err = abs(soc_drop - soc_coulomb) / soc_coulomb

    sag = ocv - v1 - V  # ohmic part only
    i_res_b, i_res_a = before(T_RES), after(T_RES)
    i_str_b, i_str_a = before(T_STRING), after(T_STRING)
    i_sh_b, i_sh_a = before(T_SHORT), after(T_SHORT)
    w_over = (t >= T_OVER + 1) & (t < T_OVER + OVERLOAD_S)

    # RC recovery: first sample after the overload where the BMS limit is released; from there
    # V1 relaxes toward I*R1 with tau = R1*resistanceFactor*C1. Check the value 1 tau later.
    tau = bat.cellR1 * bat.resistanceFactor * bat.cellC1
    k0 = int(np.argmax((t > T_OVER + 1) & (lim == 0)))
    k1 = k0 + int(round(tau / STEP_S))
    nS, nP = p["activeSeriesCells"][k1], p["activeStrings"][k1]
    v1_ss = nS * bat.cellR1 * bat.resistanceFactor * I[k1] / nP
    v1_pred = v1_ss + (v1[k0] - v1_ss) * np.exp(-1.0)
    rc_err = abs(v1[k1] - v1_pred) / abs(v1[k0] - v1_ss)

    return p, [
        ("C++ module runs in a Basilisk simulation", f"{len(t)} samples recorded from packOutMsg", len(t) > 8000),
        ("Initial open-circuit voltage", f"{ocv[1]:.3f} V (expected {ocv0:.3f} V for 2S at 80 %)", abs(ocv[1] - ocv0) < 0.01),
        ("Coulomb counting (first 30 min)", f"SoC drop {soc_drop * 100:.4f} % vs integral(I dt)/Ah {soc_coulomb * 100:.4f} % "
         f"(error {coulomb_err * 100:.3f} %, limit 0.5 %)", coulomb_err < 0.005),
        ("Resistance x2: ohmic sag doubles", f"R0 {R0[i_res_b]:.3f} -> {R0[i_res_a]:.3f} Ohm, sag {sag[i_res_b] * 1e3:.1f} -> {sag[i_res_a] * 1e3:.1f} mV",
         1.9 < sag[i_res_a] / sag[i_res_b] < 2.1),
        ("Failed string: capacity 6 -> 3 Ah", f"{cap[i_str_b]:.1f} -> {cap[i_str_a]:.1f} Ah; current {I[i_str_b]:.2f} -> {I[i_str_a]:.2f} A (cell current ~2x)",
         abs(cap[i_str_a] - 3.0) < 1e-9),
        ("Failed string visible in standard telemetry", f"batPowerOutMsg capacity {capWs[i_str_b] / 3600:.1f} -> {capWs[i_str_a] / 3600:.1f} Wh",
         abs(capWs[i_str_a] / capWs[i_str_b] - 0.5) < 1e-9),
        ("Shorted cell: voltage drop-off", f"{V[i_sh_b]:.2f} -> {V[i_sh_a]:.2f} V; current {I[i_sh_b]:.2f} -> {I[i_sh_a]:.2f} A",
         0.45 < V[i_sh_a] / V[i_sh_b] < 0.55),
        ("Overload: power limited", f"{OVERLOAD_W + LOAD_W:.0f} W requested, max delivered {pdel[w_over].max():.1f} W; "
         f"powerLimited in {int(lim[w_over].sum())} of {int(w_over.sum())} s", bool(lim[w_over].all()) and pdel[w_over].max() < OVERLOAD_W),
        ("Overload: held at cutoff (underVoltage)", f"underVoltage in {int(uv[w_over].sum())} of {int(w_over.sum())} s", bool(uv[w_over].all())),
        ("Overload: cell voltage >= cutoff", f"min cell voltage {vcell[w_over].min():.4f} V (cutoff {bat.cutoffCellVoltage:.2f} V)",
         bool(np.all(vcell[w_over] >= bat.cutoffCellVoltage - 1e-6))),
        ("RC recovery after overload (1 tau)", f"tau {tau:.0f} s; V1 {v1[k0] * 1e3:.1f} -> {v1[k1] * 1e3:.1f} mV, expected {v1_pred * 1e3:.1f} mV "
         f"(steady state {v1_ss * 1e3:.1f} mV, error {rc_err * 100:.1f} % of the step, limit 3 %)", rc_err < 0.03),
        ("heatLoss >= 0 at all times", f"min {heat.min():.3e} W, max {heat.max():.2f} W", bool(np.all(heat >= 0.0))),
        ("No NaN / inf in telemetry", "none" if np.all(np.isfinite([V, I, soc, R0, cap])) else "FOUND",
         bool(np.all(np.isfinite([V, I, soc, R0, cap])))),
    ]


def charge_checks():
    bat, p, _ = run_scenario(+CHARGE_W, 0.90, 3600.0)
    t, I, soc, lim, vcell = p["t"], p["packCurrent"], p["stateOfCharge"], p["chargeLimited"], p["cellVoltage"]
    vmax = bat.maxCellVoltage
    # CC phase: before the CV limit first engages; CV phase: chargeLimited from then on
    k_cv = int(np.argmax(lim == 1)) if lim.any() else len(t)
    cc = (t >= 1.0) & (np.arange(len(t)) < k_cv)
    cv = np.arange(len(t)) >= k_cv
    i_cv0 = I[k_cv] if lim.any() else np.nan
    return [
        ("Charging: current < 0 in the CC phase", f"{int(cc.sum())} s CC, current {I[cc].min():.3f} .. {I[cc].max():.3f} A",
         cc.any() and bool(np.all(I[cc] < 0.0))),
        ("Charging: chargeLimited at CV start", f"at t = {t[k_cv]:.0f} s, SoC {soc[k_cv] * 100:.2f} %" if lim.any() else "never",
         bool(lim.any())),
        ("Charging: cell voltage <= maxCellVoltage", f"max {vcell.max():.6f} V (limit {vmax:.2f} V)",
         bool(np.all(vcell <= vmax + 1e-6))),
        ("Charging: held at maxCellVoltage in CV", f"max |V - {vmax:.2f}| in CV {np.abs(vcell[cv] - vmax).max():.2e} V"
         if lim.any() else "no CV phase", lim.any() and bool(np.all(np.abs(vcell[cv] - vmax) < 1e-6))),
        ("Charging: current tapers in CV", f"{i_cv0:.3f} A at CV start -> {I[-1]:.4f} A after {t[-1] - t[k_cv]:.0f} s"
         if lim.any() else "no CV phase", lim.any() and abs(I[-1]) < 0.1 * abs(i_cv0)),
        ("Charging: SoC never exceeds 1", f"max SoC {soc.max():.12f}", bool(np.all(soc <= 1.0))),
        ("Charging: heatLoss >= 0", f"min {p['heatLoss'].min():.3e} W", bool(np.all(p["heatLoss"] >= 0.0))),
    ]


def depletion_checks():
    _, p, _ = run_scenario(-LOAD_W, 0.01, 1800.0)
    t, I, soc, empty = p["t"], p["packCurrent"], p["stateOfCharge"], p["depleted"]
    return [
        ("Depletion: depleted becomes 1", f"at t = {t[int(np.argmax(empty == 1))]:.0f} s" if empty.any()
         else f"never (final SoC {soc[-1]:.3e}, underVoltage {int(p['underVoltage'][-1])})", bool(empty.any())),
        ("Depletion: current goes to 0", f"final current {I[-1]:.3e} A", abs(I[-1]) < 1e-6),
        ("Depletion: SoC stays >= 0", f"min SoC {soc.min():.3e}", bool(np.all(soc >= 0.0))),
        ("Depletion: heatLoss >= 0", f"min {p['heatLoss'].min():.3e} W", bool(np.all(p["heatLoss"] >= 0.0))),
    ]


def plot_faults(p, path):
    tm = p["t"] / 60.0
    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(tm, p["packVoltage"], label="terminal voltage")
    ax[0].plot(tm, p["openCircuitVoltage"], "--", lw=0.8, label="open-circuit voltage")
    ax[0].set_ylabel("pack voltage [V]")
    ax[0].legend(fontsize=8)
    ax[1].plot(tm, p["packCurrent"])
    ax[1].set_ylabel("current [A] (+ discharge)")
    ax[2].plot(tm, p["stateOfCharge"] * 100)
    ax[2].set_ylabel("state of charge [%]")
    ax[3].plot(tm, p["capacityAh"], label="usable capacity [Ah]")
    ax[3].plot(tm, p["internalResistance"] * 100, label="R0 [Ohm x 100]")
    ax[3].set_ylabel("capacity / resistance")
    ax[3].legend(fontsize=8)
    ax[3].set_xlabel("time [min]")
    for a in ax:
        a.grid(True, alpha=0.3)
        for tt in (T_RES, T_STRING, T_SHORT, T_OVER):
            a.axvline(tt / 60.0, color="red", ls="--", lw=0.8, alpha=0.6)
    fig.suptitle("EcmBattery: resistance x2 (30 min), failed string (60 min), shorted cell (90 min), overload (120 min)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p, checks = fault_checks()
    checks += charge_checks()
    checks += depletion_checks()

    print("\n" + "=" * 110)
    print("EcmBattery (C++ extension) check  |  2S2P 3.0 Ah: faults (5 W), charging (+20 W), depletion (5 W)")
    print("-" * 110)
    for name, detail, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:44s} {detail}")
    print("-" * 110)
    print(f"{sum(ok for _, _, ok in checks)} of {len(checks)} checks passed")
    print("=" * 110)

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", "ecm_battery_check.png")
    plot_faults(p, out)
    print(f"Saved {out}")
    return 0 if all(ok for _, _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
