"""
Basilisk fault-injection check (ASTRID feasibility check, step 4.1).

One spacecraft with reaction-wheel attitude control, a power system (battery,
wheel power, payload load) and atmospheric drag. Faults are injected DURING the
run with Basilisk events, and the script checks that each fault took effect.

Timeline (seconds of simulated time, ~2 orbits at 300 km in total):
    600 s  slew A  - 34 deg turn about x, healthy wheels           (reference)
   3000 s  CONCURRENT FAULT:
             * reaction wheel 1 degraded: max torque 0.2 -> 0.02 N*m,
               Coulomb friction 0 -> 0.005 N*m
             * battery cell drop-off: usable capacity 100 % -> 50 %
   3600 s  slew B  - same 34 deg turn back, with degraded wheel 1
   1 orbit ORBIT DEGRADATION: drag area 2 -> 20 m^2 (e.g. stuck deployable)
   2 orbits end

A telemetry guard module checks every step for NaN/inf and implausible spin
rates (lesson from the thermal example).

Run from the repo root with the venv active (takes well under a minute):
    python src\\check_fault_injection.py

Outputs:
    results\\fault_injection.png
    results\\fault_injection_summary.csv
"""

import csv
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import attTrackingError, inertial3D, mrpFeedback, rwMotorTorque
from Basilisk.simulation import (ReactionWheelPower, dragDynamicEffector, msisAtmosphere,
                                 reactionWheelStateEffector, simpleBattery, simpleNav,
                                 simplePowerSink, spacecraft)
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion, simHelpers,
                                simIncludeGravBody, simIncludeRW)

# ---------------- scenario parameters ----------------
STEP_S = 0.2
ALT_KM = 300.0
SLEW_SIGMA = [0.15, 0.0, 0.0]          # MRP, ~34 deg about body x
T_SLEW_A, T_FAULT, T_SLEW_B = 600.0, 3000.0, 3600.0
RW1_UMAX_HEALTHY, RW1_UMAX_FAULT = 0.2, 0.02      # N*m
RW1_FRICTION_FAULT = 0.005                         # N*m Coulomb
BATTERY_RATIO_FAULT = 0.5
DRAG_AREA_HEALTHY, DRAG_AREA_FAULT = 2.0, 20.0    # m^2
SETTLE_THR = 0.005                                 # |sigma_BR| (~1.1 deg)


class TelemetryGuard(sysModel.SysModel):
    """Checks key telemetry every step for NaN/inf and implausible spin rates."""

    def __init__(self):
        super(TelemetryGuard, self).__init__()
        self.scIn = messaging.SCStatesMsgReader()
        self.rwIn = messaging.RWSpeedMsgReader()
        self.batIn = messaging.PowerStorageStatusMsgReader()
        self.numRW = 3
        self.maxRate = 1.0  # rad/s, implausible body rate for this spacecraft
        self.firstBad = None
        self.badCount = 0

    def Reset(self, CurrentSimNanos):
        self.firstBad = None
        self.badCount = 0

    def UpdateState(self, CurrentSimNanos):
        sc = self.scIn()
        rw = self.rwIn()
        bat = self.batIn()
        values = {
            "r_BN_N": np.array(sc.r_BN_N),
            "sigma_BN": np.array(sc.sigma_BN),
            "omega_BN_B": np.array(sc.omega_BN_B),
            "wheelSpeeds": np.array(rw.wheelSpeeds[:self.numRW]),
            "storageLevel": np.array([bat.storageLevel]),
        }
        problem = None
        for name, v in values.items():
            if not np.all(np.isfinite(v)):
                problem = f"{name} not finite"
                break
        if problem is None and np.linalg.norm(values["omega_BN_B"]) > self.maxRate:
            problem = "body rate above 1 rad/s"
        if problem:
            self.badCount += 1
            if self.firstBad is None:
                self.firstBad = (CurrentSimNanos * 1e-9, problem)


def main():
    t_wall0 = time.perf_counter()
    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("dynProcess")
    dt = macros.sec2nano(STEP_S)
    proc.addTask(sim.CreateNewTask("simTask", dt))
    proc.addTask(sim.CreateNewTask("dragTask", dt))

    # ---------- spacecraft ----------
    I = [900., 0., 0., 0., 800., 0., 0., 0., 600.]
    sc = spacecraft.Spacecraft()
    sc.ModelTag = "astridSat"
    sc.hub.mHub = 750.0
    sc.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    sc.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(I)
    sim.AddModelToTask("simTask", sc, 1)

    grav = simIncludeGravBody.gravBodyFactory()
    earth = grav.createEarth()
    earth.isCentralBody = True
    mu = earth.mu
    grav.addBodiesTo(sc)

    oe = orbitalMotion.ClassicElements()
    oe.a = earth.radEquator + ALT_KM * 1e3
    oe.e = 0.0005
    oe.i = 51.6 * macros.D2R
    oe.Omega, oe.omega, oe.f = 0.0, 0.0, 0.0
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    sc.hub.r_CN_NInit = rN
    sc.hub.v_CN_NInit = vN
    period = 2 * np.pi * np.sqrt(oe.a ** 3 / mu)
    T_DRAG = period
    T_END = 2 * period

    # ---------- reaction wheels ----------
    rwFactory = simIncludeRW.rwFactory()
    RW1 = rwFactory.create('Honeywell_HR16', [1, 0, 0], maxMomentum=50., Omega=100.,
                           RWModel=messaging.BalancedWheels)
    RW2 = rwFactory.create('Honeywell_HR16', [0, 1, 0], maxMomentum=50., Omega=200.,
                           RWModel=messaging.BalancedWheels)
    RW3 = rwFactory.create('Honeywell_HR16', [0, 0, 1], maxMomentum=50., Omega=300.,
                           RWModel=messaging.BalancedWheels)
    numRW = rwFactory.getNumOfDevices()
    rwEff = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwFactory.addToSpacecraft(sc.ModelTag, rwEff, sc)
    sim.AddModelToTask("simTask", rwEff, 2)

    nav = simpleNav.SimpleNav()
    nav.ModelTag = "nav"
    nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
    sim.AddModelToTask("simTask", nav)

    # ---------- attitude control (flight software) ----------
    vcMsg = messaging.VehicleConfigMsg().write(messaging.VehicleConfigMsgPayload(ISCPntB_B=I))
    fswRwMsg = rwFactory.getConfigMessage()

    ref = inertial3D.inertial3D()
    ref.ModelTag = "inertial3D"
    ref.sigma_R0N = [0., 0., 0.]
    sim.AddModelToTask("simTask", ref)

    attErr = attTrackingError.attTrackingError()
    attErr.ModelTag = "attErr"
    attErr.attRefInMsg.subscribeTo(ref.attRefOutMsg)
    attErr.attNavInMsg.subscribeTo(nav.attOutMsg)
    sim.AddModelToTask("simTask", attErr)

    ctrl = mrpFeedback.mrpFeedback()
    ctrl.ModelTag = "mrpFeedback"
    ctrl.guidInMsg.subscribeTo(attErr.attGuidOutMsg)
    ctrl.vehConfigInMsg.subscribeTo(vcMsg)
    ctrl.rwParamsInMsg.subscribeTo(fswRwMsg)
    ctrl.rwSpeedsInMsg.subscribeTo(rwEff.rwSpeedOutMsg)
    ctrl.K, ctrl.Ki, ctrl.P = 3.5, -1, 30.0
    ctrl.integralLimit = 2. / ctrl.Ki * 0.1
    sim.AddModelToTask("simTask", ctrl)

    rwTorque = rwMotorTorque.rwMotorTorque()
    rwTorque.ModelTag = "rwMotorTorque"
    rwTorque.vehControlInMsg.subscribeTo(ctrl.cmdTorqueOutMsg)
    rwTorque.rwParamsInMsg.subscribeTo(fswRwMsg)
    rwTorque.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    rwEff.rwMotorCmdInMsg.subscribeTo(rwTorque.rwMotorTorqueOutMsg)
    sim.AddModelToTask("simTask", rwTorque)

    # ---------- power system ----------
    rwPower = []
    for c in range(numRW):
        p = ReactionWheelPower.ReactionWheelPower()
        p.ModelTag = f"rwPower{c}"
        p.basePowerNeed = 5.0
        p.rwStateInMsg.subscribeTo(rwEff.rwOutMsgs[c])
        sim.AddModelToTask("simTask", p)
        rwPower.append(p)

    payload = simplePowerSink.SimplePowerSink()
    payload.ModelTag = "payload"
    payload.nodePowerOut = -20.0  # W
    sim.AddModelToTask("simTask", payload)

    battery = simpleBattery.SimpleBattery()
    battery.ModelTag = "battery"
    battery.storageCapacity = 2.0e6           # W*s (~556 Wh)
    battery.storedCharge_Init = 0.8 * battery.storageCapacity
    for p in rwPower:
        battery.addPowerNodeToModel(p.nodePowerOutMsg)
    battery.addPowerNodeToModel(payload.nodePowerOutMsg)
    batFaultMsg = messaging.PowerStorageFaultMsg().write(
        messaging.PowerStorageFaultMsgPayload(faultCapacityRatio=1.0))
    battery.batteryFaultInMsg.subscribeTo(batFaultMsg)
    sim.AddModelToTask("simTask", battery)

    # ---------- atmosphere and drag (MSIS, fixed space weather, offline) ----------
    atmo = msisAtmosphere.MsisAtmosphere()
    atmo.ModelTag = "msis"
    ap, f107 = 8, 110
    swKeys = ["ap_24_0", "ap_3_0", "ap_3_-3", "ap_3_-6", "ap_3_-9", "ap_3_-12", "ap_3_-15", "ap_3_-18",
              "ap_3_-21", "ap_3_-24", "ap_3_-27", "ap_3_-30", "ap_3_-33", "ap_3_-36", "ap_3_-39",
              "ap_3_-42", "ap_3_-45", "ap_3_-48", "ap_3_-51", "ap_3_-54", "ap_3_-57",
              "f107_1944_0", "f107_24_-24"]
    swMsgs = []
    for c, key in enumerate(swKeys):
        val = f107 if key.startswith("f107") else ap
        swMsgs.append(messaging.SwDataMsg().write(messaging.SwDataMsgPayload(dataValue=val)))
        atmo.swDataInMsgs[c].subscribeTo(swMsgs[-1])
    atmo.addSpacecraftToModel(sc.scStateOutMsg)
    sim.AddModelToTask("simTask", atmo)

    drag = dragDynamicEffector.DragDynamicEffector()
    drag.ModelTag = "drag"
    drag.coreParams.projectedArea = DRAG_AREA_HEALTHY
    drag.coreParams.dragCoeff = 2.2
    drag.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[0])
    sc.addDynamicEffector(drag)
    sim.AddModelToTask("dragTask", drag)

    # ---------- telemetry guard ----------
    guard = TelemetryGuard()
    guard.ModelTag = "guard"
    guard.scIn.subscribeTo(sc.scStateOutMsg)
    guard.rwIn.subscribeTo(rwEff.rwSpeedOutMsg)
    guard.batIn.subscribeTo(battery.batPowerOutMsg)
    sim.AddModelToTask("simTask", guard)

    # ---------- logging (1 s) ----------
    rec_dt = macros.sec2nano(1.0)
    logs = {
        "sc": sc.scStateOutMsg.recorder(rec_dt),
        "att": attErr.attGuidOutMsg.recorder(rec_dt),
        "rw1": rwEff.rwOutMsgs[0].recorder(rec_dt),
        "rwSpeed": rwEff.rwSpeedOutMsg.recorder(rec_dt),
        "bat": battery.batPowerOutMsg.recorder(rec_dt),
        "atmo": atmo.envOutMsgs[0].recorder(rec_dt),
    }
    for lg in logs.values():
        sim.AddModelToTask("simTask", lg)
    dragLog = drag.logger("forceExternal_B", rec_dt)
    sim.AddModelToTask("simTask", dragLog)

    # ---------- the faults, as Basilisk events ----------
    def live_rw(i):
        try:
            return rwEff.getReactionWheelAt(i)   # live configuration inside the effector
        except AttributeError:
            return [RW1, RW2, RW3][i]           # factory objects are shared with the effector

    injected = {}

    def slew_a(s):
        ref.sigma_R0N = SLEW_SIGMA
        injected["slew A"] = s.TotalSim.CurrentNanos * 1e-9

    def concurrent_fault(s):
        rw = live_rw(0)
        rw.u_max = RW1_UMAX_FAULT
        rw.fCoulomb = RW1_FRICTION_FAULT
        batFaultMsg.write(messaging.PowerStorageFaultMsgPayload(faultCapacityRatio=BATTERY_RATIO_FAULT))
        injected["wheel + battery fault"] = s.TotalSim.CurrentNanos * 1e-9

    def slew_b(s):
        ref.sigma_R0N = [0., 0., 0.]
        injected["slew B"] = s.TotalSim.CurrentNanos * 1e-9

    def drag_fault(s):
        drag.coreParams.projectedArea = DRAG_AREA_FAULT
        injected["drag fault"] = s.TotalSim.CurrentNanos * 1e-9

    for name, t, fn in [("slewA", T_SLEW_A, slew_a), ("fault", T_FAULT, concurrent_fault),
                        ("slewB", T_SLEW_B, slew_b), ("dragFault", T_DRAG, drag_fault)]:
        sim.createNewEvent(name, rec_dt, True, conditionTime=macros.sec2nano(t), actionFunction=fn)

    # ---------- run ----------
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(T_END))
    sim.ExecuteSimulation()
    wall = time.perf_counter() - t_wall0

    # ---------- evaluate ----------
    t = logs["sc"].times() * macros.NANO2SEC
    r = np.array(logs["sc"].r_BN_N)
    v = np.array(logs["sc"].v_BN_N)
    a = -mu / (2 * (0.5 * np.sum(v ** 2, axis=1) - mu / np.linalg.norm(r, axis=1)))
    sig = np.linalg.norm(np.array(logs["att"].sigma_BR), axis=1)
    u1 = np.array(logs["rw1"].u_current)
    umax1 = np.array(logs["rw1"].u_max)
    fric1 = np.array(logs["rw1"].frictionTorque)
    store = np.array(logs["bat"].storageLevel)
    cap = battery.storageCapacity
    fdrag = np.linalg.norm(np.array(dragLog.forceExternal_B), axis=1)
    idx = lambda tt: int(np.searchsorted(t, tt))

    def settle(t0, t1):
        w = (t >= t0) & (t < t1)
        tw, sw = t[w], sig[w]
        above = np.where(sw >= SETTLE_THR)[0]
        if len(above) == 0:
            return 0.0, float(sw.max())
        if above[-1] == len(sw) - 1:
            return float("nan"), float(sw.max())  # never settled in window
        return float(tw[above[-1] + 1] - t0), float(sw.max())

    setA, peakA = settle(T_SLEW_A, T_FAULT)
    setB, peakB = settle(T_SLEW_B, T_DRAG)
    w_a = (t >= T_SLEW_A) & (t < T_FAULT)
    w_b = (t >= T_SLEW_B) & (t < T_DRAG)
    torqueA, torqueB = float(np.abs(u1[w_a]).max()), float(np.abs(u1[w_b]).max())

    i_f = idx(T_FAULT)
    bat_before, bat_after = float(store[i_f - 2]), float(store[i_f + 2])

    i_d = idx(T_DRAG)
    decay1 = float(a[i_d] - a[0])
    decay2 = float(a[-1] - a[i_d])
    force_before = float(np.median(fdrag[max(i_d - 300, 0):i_d]))
    force_after = float(np.median(fdrag[i_d + 5:i_d + 305]))

    checks = [
        ("Wheel 1 max torque reduced",
         f"u_max {umax1[i_f - 2]:.3f} -> {umax1[i_f + 2]:.3f} N*m; peak |torque| slew A {torqueA:.3f}, slew B {torqueB:.3f} N*m",
         umax1[i_f + 2] <= RW1_UMAX_FAULT + 1e-9 and torqueB <= RW1_UMAX_FAULT + 1e-6),
        ("Wheel 1 friction added",
         f"|friction| before {abs(fric1[i_f - 2]):.4f}, after (max) {np.abs(fric1[i_f + 2:]).max():.4f} N*m",
         np.abs(fric1[i_f + 2:]).max() > np.abs(fric1[:i_f]).max()),
        ("Degraded slew is slower",
         f"settle time slew A {setA:.0f} s, slew B {setB:.0f} s (|sigma_BR| < {SETTLE_THR})",
         np.isfinite(setB) and setB > setA),
        ("Battery capacity fault",
         f"stored {bat_before / 3600:.1f} Wh -> {bat_after / 3600:.1f} Wh "
         f"(limit {BATTERY_RATIO_FAULT * cap / 3600:.1f} Wh); reported capacity still {logs['bat'].storageCapacity[-1] / 3600:.0f} Wh",
         bat_after <= BATTERY_RATIO_FAULT * cap + 1.0 and bat_after < bat_before),
        ("Concurrent faults in one step",
         f"wheel and battery faults injected at t = {injected.get('wheel + battery fault', float('nan')):.1f} s",
         "wheel + battery fault" in injected),
        ("Orbit degradation (drag x10)",
         f"drag force {force_before * 1e3:.2f} -> {force_after * 1e3:.2f} mN; semi-major axis change "
         f"orbit 1 {decay1:.1f} m, orbit 2 {decay2:.1f} m (ratio {decay2 / decay1 if decay1 else float('nan'):.1f})",
         decay2 < 3 * decay1 < 0),
        ("No NaN / implausible values",
         "none" if guard.firstBad is None else f"first at t = {guard.firstBad[0]:.1f} s: {guard.firstBad[1]} ({guard.badCount} steps)",
         guard.firstBad is None),
    ]

    print("\n" + "=" * 100)
    print(f"Basilisk fault-injection check  |  {T_END / 60:.0f} min simulated (2 orbits, {ALT_KM:.0f} km), "
          f"step {STEP_S} s, wall-clock {wall:.1f} s")
    print("Events fired at: " + ", ".join(f"{k} {v:.1f} s" for k, v in injected.items()))
    print("-" * 100)
    for name, detail, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:32s} {detail}")
    print("=" * 100)

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "fault_injection_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "result", "detail"])
        for name, detail, ok in checks:
            w.writerow([name, "PASS" if ok else "FAIL", detail])

    # ---------- plot ----------
    tm = t / 60.0
    marks = [(T_SLEW_A, "slew A"), (T_FAULT, "wheel+battery fault"), (T_SLEW_B, "slew B"), (T_DRAG, "drag fault")]
    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    ax[0].semilogy(tm, np.maximum(sig, 1e-6))
    ax[0].axhline(SETTLE_THR, color="grey", ls=":", lw=1)
    ax[0].set_ylabel("|attitude error σ_BR|")
    ax[1].plot(tm, u1, lw=0.8, label="wheel 1 torque")
    ax[1].plot(tm, umax1, "k--", lw=1, label="wheel 1 max torque")
    ax[1].plot(tm, -umax1, "k--", lw=1)
    ax[1].plot(tm, fric1, lw=0.8, label="wheel 1 friction")
    ax[1].set_ylabel("torque [N·m]")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[2].plot(tm, store / 3600.0, label="stored energy")
    ax[2].axhline(cap / 3600.0, color="grey", ls=":", lw=1, label="nominal capacity")
    ax[2].axhline(BATTERY_RATIO_FAULT * cap / 3600.0, color="red", ls=":", lw=1, label="capacity after fault")
    ax[2].set_ylabel("battery [Wh]")
    ax[2].legend(fontsize=8, loc="upper right")
    ax[3].plot(tm, a - a[0])
    ax[3].set_ylabel("Δ semi-major axis [m]")
    ax[3].set_xlabel("time [min]")
    for a_ in ax:
        a_.grid(True, alpha=0.3)
        for tt, lab in marks:
            a_.axvline(tt / 60.0, color="red", ls="--", lw=0.8, alpha=0.6)
    for tt, lab in marks:
        ax[0].text(tt / 60.0 + 0.5, ax[0].get_ylim()[1] * 0.5, lab, fontsize=8, color="red", rotation=90, va="top")
    fig.suptitle("Fault injection: slews, concurrent wheel + battery fault, drag fault", fontsize=11)
    fig.tight_layout()
    out = os.path.join("results", "fault_injection.png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out} and results\\fault_injection_summary.csv")


if __name__ == "__main__":
    main()