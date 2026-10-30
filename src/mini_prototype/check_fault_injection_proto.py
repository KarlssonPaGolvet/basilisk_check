"""
Fault-injection check on the full mini prototype (feasibility check 4.1, re-run with EcmBattery).

Builds the unchanged prototype (main.build) and injects faults with Basilisk events
during a 3-orbit run:
    1.0 orbit   reaction wheel 1: max torque 4 -> 0.4 mN m (10 %), Coulomb friction x10
    1.5 orbits  battery: one parallel string fails (failedStrings = 1, capacity halves)
    2.0 orbits  battery: resistanceFactor = 2.0 (internal resistance doubles)
    2.5 orbits  drag area x10 (e.g. stuck deployable)

Checks that every fault takes effect at the right time and shows up in the recorded data,
that no recorded value is NaN/inf, and that the run completes.

Run from the repo root with bsk-env active:
    python src\\mini_prototype\\check_fault_injection_proto.py

Output: results\\proto_fault_injection.png
"""

import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from Basilisk.utilities import SimulationBaseClass, macros  # noqa: E402

import config  # noqa: E402
import main  # noqa: E402

FAULT_LOG_S = 1.0           # fault-signal recorders run every step
RW1_UMAX_FACTOR = 0.10      # max torque reduced to 10 %
RW1_FRICTION_FACTOR = 10.0  # Coulomb friction x10
DRAG_AREA_FACTOR = 10.0
TIMING_TOL_S = 1.0          # a fault must show up within one step of its event


def first_time(t, cond, t_from):
    """Time of the first sample at or after t_from where cond is true (nan if none)."""
    idx = np.flatnonzero((t >= t_from) & cond)
    return t[idx[0]] if idx.size else np.nan


def on_time(t_seen, t_fault):
    return bool(np.isfinite(t_seen) and t_fault <= t_seen <= t_fault + TIMING_TOL_S)


def main_check():
    wall0 = time.perf_counter()
    sim = SimulationBaseClass.SimBaseClass()
    sim.CreateNewProcess(main.PROCESS).addTask(sim.CreateNewTask(main.TASK, config.dyn_step_ns()))
    parts = main.build(sim)
    period = parts.period_s
    # Fault times on whole seconds so they line up with the 1 s task
    t_rw, t_str, t_res, t_drag = (float(round(k * period)) for k in (1.0, 1.5, 2.0, 2.5))
    t_end = config.NUM_ORBITS * period

    # ---- 1 s recorders on the signals each fault should change ----
    rec_ns = macros.sec2nano(FAULT_LOG_S)
    bat, eff = parts.power.battery, parts.rws.effector
    logs = {
        "rw1": eff.rwOutMsgs[0].recorder(rec_ns),
        "cmd": parts.fsw.motor_cmd_msg.recorder(rec_ns),
        "pack": bat.packOutMsg.recorder(rec_ns),
        "std": bat.batPowerOutMsg.recorder(rec_ns),
        "ecl": parts.env.eclipse_msg.recorder(rec_ns),
        "guid": parts.fsw.att_guid_msg.recorder(rec_ns),
        "drag": parts.env.drag.logger("forceExternal_B", rec_ns),
    }
    for lg in logs.values():
        sim.AddModelToTask(main.TASK, lg)

    # ---- the faults, as Basilisk events ----
    rw1_nominal = eff.ReactionWheelData[0]
    umax0, fric0 = rw1_nominal.u_max, rw1_nominal.fCoulomb
    area0 = parts.env.drag.coreParams.projectedArea
    injected = {}

    def wheel_fault(s):
        w = eff.ReactionWheelData[0]  # live configuration inside the effector
        w.u_max = umax0 * RW1_UMAX_FACTOR
        w.fCoulomb = fric0 * RW1_FRICTION_FACTOR
        injected["wheel"] = s.TotalSim.CurrentNanos * 1e-9

    def string_fault(s):
        bat.failedStrings = 1
        injected["string"] = s.TotalSim.CurrentNanos * 1e-9

    def resistance_fault(s):
        bat.resistanceFactor = 2.0
        injected["resistance"] = s.TotalSim.CurrentNanos * 1e-9

    def drag_fault(s):
        parts.env.drag.coreParams.projectedArea = area0 * DRAG_AREA_FACTOR
        injected["drag"] = s.TotalSim.CurrentNanos * 1e-9

    for name, t_f, fn in [("wheelFault", t_rw, wheel_fault), ("stringFault", t_str, string_fault),
                          ("resistanceFault", t_res, resistance_fault), ("dragFault", t_drag, drag_fault)]:
        sim.createNewEvent(name, config.dyn_step_ns(), True, conditionTime=macros.sec2nano(t_f),
                           actionFunction=fn)

    # ---- run ----
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(t_end))
    sim.ExecuteSimulation()
    wall = time.perf_counter() - wall0
    parts.env.unload()

    # ---- extract ----
    t = logs["rw1"].times() * macros.NANO2SEC
    u1 = np.array(logs["rw1"].u_current)
    fr1 = np.array(logs["rw1"].frictionTorque)
    umax_log = np.array(logs["rw1"].u_max)
    cmd1 = np.array(logs["cmd"].motorTorque)[:, 0]
    cap_ah = np.array(logs["pack"].capacityAh)
    cap_ws = np.array(logs["std"].storageCapacity)
    r0 = np.array(logs["pack"].internalResistance)
    v = np.array(logs["pack"].packVoltage)
    i_pack = np.array(logs["pack"].packCurrent)
    sag = np.array(logs["pack"].openCircuitVoltage) - np.array(logs["pack"].polarizationVoltage) - v
    cv = np.array(logs["pack"].chargeLimited)
    sunlit = np.array(logs["ecl"].shadowFactor) > 0.5
    sig_br = np.array(logs["guid"].sigma_BR)
    point_deg = np.degrees(4.0 * np.arctan(np.linalg.norm(sig_br, axis=1)))
    f_drag = np.linalg.norm(np.array(logs["drag"].forceExternal_B), axis=1)
    a_drag = f_drag / config.SC_MASS_KG
    frames = parts.recorder.datasets()
    k = lambda tt: int(np.searchsorted(t, tt))  # index of the sample at time tt

    # ---- 1. wheel fault ----
    umax_f = umax0 * RW1_UMAX_FACTOR
    t_umax = first_time(t, np.isclose(umax_log, umax_f), t_rw - 5)
    t_fric = first_time(t, np.abs(fr1) > 2.0 * fric0, t_rw - 5)
    after_rw = t > t_rw + TIMING_TOL_S
    saturated = after_rw & (np.abs(u1) >= umax_f - 1e-12) & (np.abs(cmd1) > umax_f)
    t_sat = first_time(t, saturated, t_rw)
    pre_rw = (t > 1200) & (t < t_rw)  # after the initial slew has settled

    # ---- 2. string fault ----
    kb, ka = k(t_str) - 1, k(t_str) + 1
    t_cap = first_time(t, cap_ah < cap_ah[kb] - 1e-9, t_str - 5)
    t_capws = first_time(t, cap_ws < cap_ws[kb] - 1e-9, t_str - 5)

    # ---- 3. resistance fault ----
    rb, ra = k(t_res) - 1, k(t_res) + 1
    t_r0 = first_time(t, r0 > r0[rb] * 1.5, t_res - 5)
    # Sag at matched load: discharge (eclipse) samples between the string and resistance
    # faults vs eclipse samples after the resistance fault (same strings, same bus load).
    ecl_before = (~sunlit) & (t > t_str + 60) & (t < t_res) & (i_pack > 0)
    ecl_after = (~sunlit) & (t > t_res + 60) & (i_pack > 0)
    sag_b, sag_a = sag[ecl_before].mean(), sag[ecl_after].mean()
    state_res = "CV charging" if cv[rb] else ("charging" if i_pack[rb] < 0 else "discharging")

    # ---- 4. drag fault ----
    db, da = k(t_drag) - 1, k(t_drag) + 1
    t_dragseen = first_time(t, f_drag > 3.0 * f_drag[db], t_drag - 5)

    # ---- 5. finite values everywhere ----
    fault_arrays = [u1, fr1, umax_log, cmd1, cap_ah, cap_ws, r0, v, i_pack, sag, point_deg, f_drag]
    bad_fault = sum(int(np.count_nonzero(~np.isfinite(a))) for a in fault_arrays)
    bad_proto = sum(int(np.count_nonzero(~np.isfinite(df.to_numpy(dtype=float)))) for df in frames.values())
    n_proto = sum(df.size for df in frames.values())

    checks = [
        ("Run completes (3 orbits)", f"simulated {t[-1]:.0f} of {t_end:.0f} s, wall {wall:.1f} s, "
         f"faults injected at {', '.join(f'{n} {v:.0f} s' for n, v in injected.items())}",
         t[-1] >= t_end - FAULT_LOG_S - 1e-6 and len(injected) == 4),
        ("RW1 max torque cut at 1 orbit", f"u_max {umax0 * 1e3:.2f} -> {umax_f * 1e3:.2f} mN m, seen at "
         f"t = {t_umax:.0f} s (fault {t_rw:.0f} s)", on_time(t_umax, t_rw)),
        ("RW1 friction increased at 1 orbit", f"|friction| {np.abs(fr1[pre_rw]).max() * 1e3:.3f} -> "
         f"{np.abs(fr1[after_rw]).max() * 1e3:.3f} mN m, first seen at t = {t_fric:.0f} s", on_time(t_fric, t_rw)),
        ("RW1 torque saturates after the fault", f"saturated (|u| = u_max, |cmd| > u_max) in {int(saturated.sum())} s, "
         f"first at t = {t_sat:.0f} s; max |u| after {np.abs(u1[after_rw]).max() * 1e3:.3f} mN m",
         bool(saturated.any()) and np.abs(u1[after_rw]).max() <= umax_f + 1e-12),
        ("Battery string fails at 1.5 orbits (packOutMsg)", f"capacity {cap_ah[kb]:.1f} -> {cap_ah[ka]:.1f} Ah, "
         f"seen at t = {t_cap:.0f} s (fault {t_str:.0f} s)",
         on_time(t_cap, t_str) and abs(cap_ah[ka] / cap_ah[kb] - 0.5) < 1e-9),
        ("Battery string fails at 1.5 orbits (batPowerOutMsg)", f"capacity {cap_ws[kb] / 3600:.2f} -> "
         f"{cap_ws[ka] / 3600:.2f} Wh, seen at t = {t_capws:.0f} s",
         on_time(t_capws, t_str) and abs(cap_ws[ka] / cap_ws[kb] - 0.5) < 1e-9),
        ("Resistance x2 at 2 orbits", f"R0 {r0[rb] * 1e3:.1f} -> {r0[ra] * 1e3:.1f} mOhm, seen at t = {t_r0:.0f} s "
         f"(fault {t_res:.0f} s)", on_time(t_r0, t_res) and abs(r0[ra] / r0[rb] - 2.0) < 1e-9),
        ("Voltage sag increases (eclipse, same load)", f"ohmic sag {sag_b * 1e3:.1f} -> {sag_a * 1e3:.1f} mV "
         f"(x{sag_a / sag_b:.2f}) at mean current {i_pack[ecl_before].mean():.3f} / {i_pack[ecl_after].mean():.3f} A",
         bool(ecl_before.any() and ecl_after.any()) and 1.8 < sag_a / sag_b < 2.2),
        ("Drag area x10 at 2.5 orbits", f"drag accel {a_drag[db] * 1e9:.1f} -> {a_drag[da] * 1e9:.1f} nm/s^2 "
         f"(x{f_drag[da] / f_drag[db]:.2f}), seen at t = {t_dragseen:.0f} s (fault {t_drag:.0f} s)",
         on_time(t_dragseen, t_drag) and 9.0 < f_drag[da] / f_drag[db] < 11.0),
        ("No NaN / inf anywhere", f"{bad_fault} bad in fault logs ({sum(a.size for a in fault_arrays)} values), "
         f"{bad_proto} bad in prototype data sets ({n_proto} values)", bad_fault == 0 and bad_proto == 0),
    ]
    info = [
        ("Sun-pointing error before / after wheel fault", f"mean {point_deg[pre_rw].mean():.3f} / "
         f"{point_deg[after_rw].mean():.3f} deg, max after {point_deg[after_rw].max():.2f} deg"),
        ("Battery at the resistance fault", f"{state_res}: current {i_pack[rb]:+.4f} -> {i_pack[ra]:+.4f} A, "
         f"sag {sag[rb] * 1e3:.2f} -> {sag[ra] * 1e3:.2f} mV (in CV the voltage is pinned, so current halves instead)"),
    ]

    print("\n" + "=" * 118)
    print("Mini prototype fault injection  |  EcmBattery, 3 orbits, faults at 1 / 1.5 / 2 / 2.5 orbits")
    print("-" * 118)
    for name, detail, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:50s} {detail}")
    for name, detail in info:
        print(f"[info] {name:50s} {detail}")
    print("-" * 118)
    print(f"{sum(ok for _, _, ok in checks)} of {len(checks)} checks passed")
    print("=" * 118)

    plot(t, period, (t_rw, t_str, t_res, t_drag), u1, cmd1, fr1, umax_log, cap_ah, cap_ws, sag, r0,
         i_pack, f_drag)
    return 0 if all(ok for _, _, ok in checks) else 1


def plot(t, period, faults, u1, cmd1, fr1, umax_log, cap_ah, cap_ws, sag, r0, i_pack, f_drag):
    tm = t / 60.0
    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    ax[0].plot(tm, cmd1 * 1e3, lw=0.8, alpha=0.6, label="commanded")
    ax[0].plot(tm, u1 * 1e3, lw=1, label="applied")
    ax[0].plot(tm, fr1 * 1e3, lw=1, label="friction")
    ax[0].plot(tm, umax_log * 1e3, "k--", lw=0.8, label="+u_max")
    ax[0].plot(tm, -umax_log * 1e3, "k--", lw=0.8)
    ax[0].set_ylim(-1.0, 1.0)
    ax[0].set_ylabel("RW1 torque [mN m]")
    ax[0].legend(fontsize=8, ncol=4, loc="upper left")

    ax[1].plot(tm, cap_ah, label="packOutMsg capacity [Ah]")
    ax1b = ax[1].twinx()
    ax1b.plot(tm, cap_ws / 3600.0, "C1--", label="batPowerOutMsg capacity [Wh]")
    ax[1].set_ylabel("capacity [Ah]")
    ax1b.set_ylabel("capacity [Wh]")
    ax[1].legend(handles=ax[1].get_lines() + ax1b.get_lines(), fontsize=8, loc="lower left")

    ax[2].plot(tm, sag * 1e3, label="ohmic sag (OCV - V1 - V)")
    ax2b = ax[2].twinx()
    ax2b.plot(tm, r0 * 1e3, "C3--", label="R0 [mOhm]")
    ax2b.plot(tm, i_pack * 100, "C2:", lw=0.8, label="current [A x 100]")
    ax[2].set_ylabel("sag [mV]")
    ax2b.set_ylabel("R0 [mOhm] / I [A x 100]")
    ax[2].legend(handles=ax[2].get_lines() + ax2b.get_lines(), fontsize=8, loc="upper left")

    ax[3].semilogy(tm, f_drag * 1e6)
    ax[3].set_ylabel("drag force [uN]")
    ax[3].set_xlabel("time [min]")

    labels = ["RW1 fault", "string fails", "R0 x2", "drag x10"]
    for a in ax:
        a.grid(True, alpha=0.3)
        for tf, lab in zip(faults, labels):
            a.axvline(tf / 60.0, color="red", ls="--", lw=0.8, alpha=0.7)
    for tf, lab in zip(faults, labels):
        ax[0].text(tf / 60.0, 0.92, f" {lab}", color="red", fontsize=8, va="top",
                   transform=ax[0].get_xaxis_transform())
    fig.suptitle(f"Mini prototype fault injection (orbit period {period / 60:.1f} min)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.join(config.REPO_ROOT, "results"), exist_ok=True)
    out = os.path.join(config.REPO_ROOT, "results", "proto_fault_injection.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    raise SystemExit(main_check())
