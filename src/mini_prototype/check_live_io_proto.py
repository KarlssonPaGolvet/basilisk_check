"""
Live data in/out check on the full mini prototype (standard library only: socket, json).

Adds two Python Basilisk modules (sysModel.SysModel) to the unchanged prototype:
    CommandInput     start of every step: non-blocking read of incoming UDP commands, applied at once
    TelemetryOutput  end of every step: one JSON packet over UDP (seq, sim time, wall time, attitude
                     error, battery voltage/current/SoC, wheel speeds, bus load, active strings)
A separate client process (live_io_client.py) receives the telemetry and, mid-run, sends two
commands: (a) bus load -6 -> -10 W at 40 s, (b) battery fault failedStrings = 1 at 80 s.

Runs in real time (ClockSynch accelFactor 1.0, 1 ms accuracy, 1 ms Windows timer) for 120 s.

Checks:
    - no lost, duplicated or out-of-order packets
    - telemetry latency (send -> receive) max and p99 within 20 ms
    - each command applied within 1 step of the client sending it, exactly once
    - each command visible in the recorded data from the step it was applied
    - the run still keeps real time (max late within 20 ms, no overruns)
    - extra compute cost per step of the I/O modules

Run from the repo root with bsk-env active (about 2.5 minutes):
    python src\\mini_prototype\\check_live_io_proto.py

Output: results\\proto_live_io.png
"""

import ctypes
import json
import os
import socket
import subprocess
import sys
import tempfile
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
from check_realtime_proto import WallClockProbe  # noqa: E402

DURATION_S = 120.0
BUDGET_MS = 20.0
CLOCK_PRIORITY, PROBE_PRIORITY = 1000, 999
COMMAND_PRIORITY = 998        # right after the clock wait: commands act in the same step
TELEMETRY_PRIORITY = -100     # after every model and recorder: telemetry of the finished step


class CommandInput(sysModel.SysModel):
    """Non-blocking UDP command reader. Only whitelisted targets can be changed."""

    def __init__(self, sock, targets):
        super(CommandInput, self).__init__()
        self.sock, self.targets = sock, targets
        self.applied, self.rejected, self.cost_s = [], [], []

    def UpdateState(self, CurrentSimNanos):
        t0 = time.perf_counter()
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except BlockingIOError:
                break
            except ConnectionResetError:  # Windows: ICMP "port unreachable" from an earlier send
                continue
            recv_ns = time.perf_counter_ns()
            try:
                cmd = json.loads(data)
                obj, attr, cast = self.targets[cmd["target"]]
                setattr(obj, attr, cast(cmd["value"]))
                self.applied.append({"id": cmd["id"], "target": cmd["target"], "value": cmd["value"],
                                     "sim_t": CurrentSimNanos * 1e-9, "recv_ns": recv_ns})
            except (ValueError, KeyError, TypeError) as e:
                self.rejected.append((CurrentSimNanos * 1e-9, repr(e)))
        self.cost_s.append(time.perf_counter() - t0)


class TelemetryOutput(sysModel.SysModel):
    """One JSON telemetry packet per step over UDP."""

    def __init__(self, sock, dest, parts):
        super(TelemetryOutput, self).__init__()
        self.sock, self.dest, self.p = sock, dest, parts
        self.seq, self.cost_s = 0, []

    def UpdateState(self, CurrentSimNanos):
        t0 = time.perf_counter()
        p = self.p
        sigma = np.array(p.fsw.att_guid_msg.read().sigma_BR)
        pack = p.power.battery.packOutMsg.read()
        speeds = list(p.rws.speed_msg.read().wheelSpeeds)[:p.rws.num]
        pkt = {
            "type": "tm", "seq": self.seq, "sim_t": CurrentSimNanos * 1e-9,
            "att_err_deg": float(np.degrees(4.0 * np.arctan(np.linalg.norm(sigma)))),
            "bat_V": pack.packVoltage, "bat_A": pack.packCurrent, "bat_soc": pack.stateOfCharge,
            "bat_strings": pack.activeStrings,
            "bus_load_W": p.power.bus_load.nodePowerOutMsg.read().netPower,
            "rw_rpm": [w / macros.RPM for w in speeds],
        }
        pkt["wall_t"] = time.time()
        pkt["send_ns"] = time.perf_counter_ns()
        self.sock.sendto(json.dumps(pkt).encode(), self.dest)
        self.seq += 1
        self.cost_s.append(time.perf_counter() - t0)


def start_client(result_path):
    client = subprocess.Popen([sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                   "live_io_client.py"), result_path],
                              stdout=subprocess.PIPE, text=True, cwd=config.REPO_ROOT)
    line = client.stdout.readline().strip()
    if not line.startswith("PORT "):
        client.kill()
        raise RuntimeError(f"client did not start: {line!r}")
    return client, int(line.split()[1])


def main_check():
    if sys.platform != "win32":
        print("This check sets the Windows timer resolution; run it on Windows.")
        return 1
    tmp = tempfile.mkdtemp()
    result_path = os.path.join(tmp, "client_result.json")
    client, client_port = start_client(result_path)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    if hasattr(socket, "SIO_UDP_CONNRESET"):
        sock.ioctl(socket.SIO_UDP_CONNRESET, False)  # don't turn ICMP unreachable into recv errors

    sim = SimulationBaseClass.SimBaseClass()
    sim.CreateNewProcess(main.PROCESS).addTask(sim.CreateNewTask(main.TASK, config.dyn_step_ns()))
    parts = main.build(sim)

    clock = simSynch.ClockSynch()
    clock.ModelTag = "clock"
    clock.accelFactor = 1.0
    clock.accuracyNanos = int(1e6)
    sim.AddModelToTask(main.TASK, clock, CLOCK_PRIORITY)
    probe = WallClockProbe()
    probe.ModelTag = "probe"
    sim.AddModelToTask(main.TASK, probe, PROBE_PRIORITY)
    targets = {"bus_load.nodePowerOut": (parts.power.bus_load, "nodePowerOut", float),
               "battery.failedStrings": (parts.power.battery, "failedStrings", int)}
    cmd_in = CommandInput(sock, targets)
    cmd_in.ModelTag = "commandInput"
    sim.AddModelToTask(main.TASK, cmd_in, COMMAND_PRIORITY)
    tm_out = TelemetryOutput(sock, ("127.0.0.1", client_port), parts)
    tm_out.ModelTag = "telemetryOutput"
    sim.AddModelToTask(main.TASK, tm_out, TELEMETRY_PRIORITY)

    rec_ns = config.dyn_step_ns()
    load_log = parts.power.bus_load.nodePowerOutMsg.recorder(rec_ns)
    pack_log = parts.power.battery.packOutMsg.recorder(rec_ns)
    clock_log = clock.clockOutMsg.recorder()
    for lg in (load_log, pack_log, clock_log):
        sim.AddModelToTask(main.TASK, lg)

    print(f"Live I/O run: {DURATION_S:.0f} s in real time, telemetry to UDP port {client_port} ...")
    ctypes.windll.winmm.timeBeginPeriod(1)
    try:
        sim.InitializeSimulation()
        sim.ConfigureStopTime(macros.sec2nano(DURATION_S))
        sim.ExecuteSimulation()
    finally:
        ctypes.windll.winmm.timeEndPeriod(1)
    parts.env.unload()
    sock.setblocking(True)
    sock.sendto(json.dumps({"type": "end", "sent": tm_out.seq}).encode(), ("127.0.0.1", client_port))
    out, _ = client.communicate(timeout=60)
    print(out.strip())
    with open(result_path) as f:
        res = json.load(f)
    sock.close()

    # ---- packets ----
    pk = res["packets"]
    seqs = np.array([p["seq"] for p in pk])
    sent = tm_out.seq
    in_order = bool(np.all(np.diff(seqs) == 1)) and len(seqs) and seqs[0] == 0
    lost = sent - len(set(seqs.tolist()))
    dup = len(seqs) - len(set(seqs.tolist()))
    lat_ms = np.array([(p["recv_ns"] - p["send_ns"]) * 1e-6 for p in pk])

    # ---- commands ----
    step = config.DYN_STEP_S
    t = load_log.times() * macros.NANO2SEC
    load = np.array(load_log.netPower)
    strings = np.array(pack_log.activeStrings)
    cap = np.array(pack_log.capacityAh)
    applied = {a["id"]: a for a in cmd_in.applied}
    cmd_rows = []
    for c in res["commands"]:
        a = applied.get(c["id"])
        n_applied = sum(1 for x in cmd_in.applied if x["id"] == c["id"])
        if a is None:
            cmd_rows.append((c, None, np.nan, False, False))
            continue
        delay = a["sim_t"] - c["last_seen_sim_t"]
        if c["target"] == "bus_load.nodePowerOut":
            seen_idx = np.flatnonzero(np.isclose(load, c["value"]))
            detail = f"bus load {load[0]:.1f} -> {c['value']:.1f} W"
        else:
            seen_idx = np.flatnonzero(strings == c["value"])
            detail = f"active strings 2 -> {c['value']}, capacity {cap[0]:.1f} -> {cap[-1]:.1f} Ah"
        t_seen = t[seen_idx[0]] if seen_idx.size else np.nan
        visible = bool(seen_idx.size) and abs(t_seen - a["sim_t"]) < 1e-9 and bool(np.all(np.diff(seen_idx) == 1)) \
            and seen_idx[-1] == len(t) - 1
        cmd_rows.append((c, a, t_seen, 0 < delay <= step + 1e-9 and n_applied == 1, visible, delay, detail,
                         (a["recv_ns"] - c["sent_ns"]) * 1e-6))

    # ---- real time ----
    sim_t = (np.array(probe.sim_ns) - probe.sim_ns[0]) * 1e-9
    wall_t = np.array(probe.wall_s) - probe.wall_s[0]
    err_ms = (wall_t - sim_t) * 1000.0
    overruns = int(np.array(clock_log.overrunCounter)[-1])
    cost_us = (np.array(cmd_in.cost_s) + np.array(tm_out.cost_s)) * 1e6

    checks = [
        ("Telemetry: no lost / duplicate / out-of-order", f"sent {sent}, received {len(pk)}, lost {lost}, "
         f"duplicates {dup}, in order: {'yes' if in_order else 'NO'}", lost == 0 and dup == 0 and in_order),
        ("Telemetry latency (send -> receive)", f"max {lat_ms.max():.3f} ms, p99 {np.percentile(lat_ms, 99):.3f} ms, "
         f"median {np.median(lat_ms):.3f} ms (limit {BUDGET_MS:g} ms)", lat_ms.max() < BUDGET_MS),
    ]
    for row in cmd_rows:
        c, a = row[0], row[1]
        label = "Command 1 (bus load)" if c["id"] == 1 else "Command 2 (battery fault)"
        if a is None:
            checks.append((f"{label}: applied within 1 step", "never applied", False))
            checks.append((f"{label}: visible in recorded data", "-", False))
            continue
        _, _, t_seen, on_time, visible, delay, detail, link_ms = row
        checks.append((f"{label}: applied within 1 step", f"sent after telemetry of t = {c['last_seen_sim_t']:.0f} s, "
                       f"applied at t = {a['sim_t']:.0f} s ({delay:.0f} step), once; send -> applied "
                       f"{link_ms:.1f} ms (waits for the next step)", on_time))
        checks.append((f"{label}: visible in recorded data", f"{detail}; first recorded at t = {t_seen:.0f} s", visible))
    if len(cmd_rows) < 2:
        checks.append(("Both commands sent by the client", f"{len(cmd_rows)} sent", False))
    checks += [
        ("Real time kept with live I/O", f"max late {err_ms.max():.3f} ms, p99 {np.percentile(np.abs(err_ms), 99):.3f} ms, "
         f"overruns {overruns} (limit {BUDGET_MS:g} ms)", err_ms.max() < BUDGET_MS and overruns == 0),
        ("I/O compute cost per step", f"mean {cost_us.mean():.0f} us, p99 {np.percentile(cost_us, 99):.0f} us, "
         f"max {cost_us.max():.0f} us = {cost_us.max() / (step * 1e4):.3f} % of the {step:g} s step", cost_us.max() < 5000),
        ("No rejected / malformed commands", f"{len(cmd_in.rejected)} rejected", not cmd_in.rejected),
    ]

    print("\n" + "=" * 118)
    print(f"Mini prototype live I/O  |  UDP JSON on localhost, {DURATION_S:.0f} s real time, ClockSynch + 1 ms timer")
    print("-" * 118)
    for name, detail, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:46s} {detail}")
    print("-" * 118)
    print(f"{sum(ok for _, _, ok in checks)} of {len(checks)} checks passed")
    print("=" * 118)

    plot(pk, lat_ms, t, load, strings, sim_t, err_ms, cost_us, cmd_in.applied)
    return 0 if all(ok for _, _, ok in checks) else 1


def plot(pk, lat_ms, t, load, strings, sim_t, err_ms, cost_us, applied):
    ts = np.array([p["sim_t"] for p in pk])
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    ax[0].plot(ts, [p["bat_V"] for p in pk], label="battery voltage [V] (client view)")
    ax0b = ax[0].twinx()
    ax0b.plot(ts, [p["bat_A"] for p in pk], "C1", label="battery current [A] (client view)")
    ax[0].set_ylabel("V")
    ax0b.set_ylabel("A")
    ax[0].legend(handles=ax[0].get_lines() + ax0b.get_lines(), fontsize=8, loc="lower left")
    ax[1].step(t, load, where="post", label="bus load [W] (recorded)")
    ax[1].step(t, strings, where="post", label="active strings (recorded)")
    ax[1].legend(fontsize=8)
    ax[2].plot(ts, lat_ms, ".-", ms=3, label="telemetry latency send->receive [ms]")
    ax[2].plot(sim_t, err_ms, ".-", ms=3, label="real-time error wall-sim [ms]")
    ax[2].axhline(BUDGET_MS, color="red", ls="--", lw=1, label=f"{BUDGET_MS:g} ms budget")
    ax[2].set_yscale("symlog", linthresh=1.0)
    ax[2].legend(fontsize=8)
    ax[3].plot(sim_t[:len(cost_us)], cost_us, ".-", ms=3, label="I/O modules compute per step [us]")
    ax[3].set_xlabel("simulated time [s]")
    ax[3].legend(fontsize=8)
    for a in ax:
        a.grid(True, alpha=0.3)
        for c in applied:
            a.axvline(c["sim_t"], color="red", ls="--", lw=0.8, alpha=0.7)
    fig.suptitle("Mini prototype live I/O over UDP (red: commands applied)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(config.REPO_ROOT, "results", "proto_live_io.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    raise SystemExit(main_check())
