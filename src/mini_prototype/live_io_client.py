"""
UDP client for check_live_io_proto.py (standard library only).

Receives the prototype's JSON telemetry on localhost and, mid-run, sends two commands back:
    1. at sim time >= CMD1_T: set the bus load to CMD1_LOAD_W
    2. at sim time >= CMD2_T: inject a battery fault (failedStrings = 1)
Every packet's receive time is taken with time.perf_counter_ns() right after recvfrom(); on
Windows this is the system-wide QueryPerformanceCounter, so it is comparable with the sender's.

Usage (started by check_live_io_proto.py):
    python live_io_client.py <result.json>
Prints "PORT <n>" once its telemetry socket is bound, then runs until it receives the "end"
packet (or IDLE_TIMEOUT_S of silence) and writes everything it saw to <result.json>.
"""

import json
import socket
import sys
import time

CMD1_T, CMD1_LOAD_W = 40.0, -10.0
CMD2_T = 80.0
IDLE_TIMEOUT_S = 30.0


def main(result_path):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(IDLE_TIMEOUT_S)
    print(f"PORT {sock.getsockname()[1]}", flush=True)

    packets, commands, end = [], [], None
    pending = [
        {"id": 1, "at": CMD1_T, "cmd": {"id": 1, "target": "bus_load.nodePowerOut", "value": CMD1_LOAD_W}},
        {"id": 2, "at": CMD2_T, "cmd": {"id": 2, "target": "battery.failedStrings", "value": 1}},
    ]
    while True:
        try:
            data, addr = sock.recvfrom(65536)
        except socket.timeout:
            break
        recv_ns = time.perf_counter_ns()
        msg = json.loads(data)
        if msg.get("type") == "end":
            end = msg
            break
        msg["recv_ns"] = recv_ns
        packets.append(msg)
        # Commands go back to the address the telemetry came from (the sim's socket)
        while pending and msg["sim_t"] >= pending[0]["at"]:
            p = pending.pop(0)
            payload = json.dumps(p["cmd"]).encode()
            sent_ns = time.perf_counter_ns()
            sock.sendto(payload, addr)
            commands.append({"id": p["id"], "target": p["cmd"]["target"], "value": p["cmd"]["value"],
                             "sent_ns": sent_ns, "last_seen_seq": msg["seq"], "last_seen_sim_t": msg["sim_t"]})

    with open(result_path, "w") as f:
        json.dump({"packets": packets, "commands": commands, "end": end}, f)
    print(f"client: {len(packets)} packets, {len(commands)} commands sent, end packet "
          f"{'received' if end else 'MISSING'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
