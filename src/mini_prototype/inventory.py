"""
Write the data sets and an inventory of what the run actually recorded.

The inventory is generated from the recorder channels and the recorded
values, so it always matches the script output.
"""

import os

import numpy as np

from Basilisk.utilities import RigidBodyKinematics as rbk

import config
from recorders import (G_BAT_ELEC, G_ENV, G_NAV, G_POWER, G_RW, G_SENSORS, G_TEMP, INTERNAL, NAVIGATION, XYZ)

# Data-set table from the feasibility check: (dataset, signals, Basilisk modules).
# simpleBattery is replaced by EcmBattery (detailed_battery C++ extension), which also
# provides the battery voltage/current row.
DATASET_TABLE = [
    (INTERNAL, G_POWER, ["EcmBattery", "simplePowerSink", "simpleSolarPanel", "ReactionWheelPower"]),
    (INTERNAL, G_BAT_ELEC, ["EcmBattery"]),
    (INTERNAL, G_RW, ["reactionWheels"]),
    (INTERNAL, G_TEMP, ["sensorThermal", "motorThermal", "tempMeasurement"]),
    (INTERNAL, G_SENSORS, ["imuSensor", "starTracker", "magnetometer", "coarseSunSensor"]),
    (NAVIGATION, G_NAV, ["simpleNav"]),
    (NAVIGATION, G_ENV, ["spiceInterface", "eclipse", "MsisAtmosphere", "dragEffector"]),
]

CSV_NAMES = {INTERNAL: "internal_sensors.csv", NAVIGATION: "navigation.csv"}


def save_datasets(frames, out_dir):
    paths = {}
    for name, df in frames.items():
        path = os.path.join(out_dir, CSV_NAMES[name])
        df.to_csv(path, index=False, float_format="%.9g")
        paths[name] = path
    return paths


def _fmt(x):
    if x == 0 or 1e-3 <= abs(x) < 1e6:
        return f"{x:.4g}"
    return f"{x:.3e}"


def nav_noise_check(nav_df):
    """Realised nav-minus-truth error statistics, per signal (all axes pooled)."""
    rows = []
    for label, nav_col, true_col, unit, key in [
        ("Position", "nav_r_BN_N", "true_r_BN_N", "m", "pos_m"),
        ("Velocity", "nav_v_BN_N", "true_v_BN_N", "m/s", "vel_mps"),
        ("Attitude (MRP)", "nav_sigma_BN", "true_sigma_BN", "-", "att_mrp"),
        ("Body rate", "nav_omega_BN_B", "true_omega_BN_B", "rad/s", "rate_radps"),
    ]:
        suffixes = (1, 2, 3) if "sigma" in nav_col else XYZ
        err = np.column_stack([nav_df[f"{nav_col}_{s}"] - nav_df[f"{true_col}_{s}"] for s in suffixes])
        if "sigma" in nav_col:
            # wrap MRP shadow-set switches so the difference stays small
            err = err[np.all(np.abs(err) < 0.5, axis=1)]
        rows.append((label, unit, config.NAV_SIGMA[key], config.NAV_WALK_BOUNDS[key],
                     err.mean(), err.std(), np.abs(err).max()))
    return rows


def _vec(df, prefix, suffixes=XYZ):
    return np.column_stack([df[f"{prefix}_{s}"] for s in suffixes])


def sensor_error_check(isd, nav_df):
    """Realised sensor-minus-truth error statistics (sensor frames are aligned with the body).

    The first sample is skipped: the IMU differences consecutive states, so its t=0 output is zero.
    """
    isd, nav_df = isd.iloc[1:], nav_df.iloc[1:]
    sigma_bn = _vec(nav_df, "true_sigma_BN", (1, 2, 3))
    rows = []

    gyro_err = _vec(isd, "imu_omega_P") - _vec(nav_df, "true_omega_BN_B")
    rows.append(("IMU gyro rate", "rad/s", f"{_fmt(config.IMU_GYRO_NOISE)} + bias", gyro_err))

    # Star tracker: rotation angle between measured and true attitude
    q_meas = _vec(isd, "st_q", range(4))
    q_true = np.array([rbk.MRP2EP(s) for s in sigma_bn])
    dot = np.clip(np.abs(np.sum(q_meas * q_true, axis=1)), 0.0, 1.0)
    rows.append(("Star tracker attitude", "rad", f"{_fmt(config.ST_NOISE_RAD)} per axis",
                 2.0 * np.arccos(dot)[:, None]))

    # Magnetometer vs WMM field rotated into the body frame
    b_true = np.array([np.array(rbk.MRP2C(s)) @ b for s, b in zip(sigma_bn, _vec(nav_df, "mag_field_N"))])
    rows.append(("Magnetometer field", "T", f"{_fmt(config.MAG_NOISE_T)} + bias",
                 _vec(isd, "mag_B_S") - b_true))

    temp_true = ["sensor_temp_C"] + [c for c in isd.columns if c.endswith("_motor_temp_C") and
                                     not c.startswith("meas_")]
    temp_err = np.column_stack([isd[f"meas_{c}"] - isd[c] for c in temp_true])
    rows.append(("Temperature readout", "degC", f"{_fmt(config.TEMP_MEAS_NOISE_C)} + bias", temp_err))

    return [(label, unit, spec, err.mean(), err.std(), np.abs(err).max()) for label, unit, spec, err in rows]


def battery_electrical_check(isd, nav_df):
    """Voltage/current checks on the EcmBattery telemetry.

    Returns rows of (check, result, criterion, status); status is PASS/FAIL, or "info"
    for values without a pass criterion.
    """
    V, I = isd["battery_voltage_V"], isd["battery_current_A"]
    v_cell, soc, cap = isd["battery_cell_voltage_V"], isd["battery_soc"], isd["battery_capacity_Ah"]
    demand, delivered = isd["battery_power_demand_W"], isd["battery_power_delivered_W"]
    limited = (isd["battery_power_limited"] + isd["battery_charge_limited"] + isd["battery_depleted"]) > 0
    nodes = isd["solar_panel_power_W"] + isd["bus_load_power_W"] + isd.filter(regex=r"^rw\d+_power_W$").sum(axis=1)
    sunlit = nav_df["eclipse_shadow_factor"] > 0.5
    dt = float(isd["time_s"].diff().median())
    cutoff = config.BATTERY_CUTOFF_CELL_V
    status = lambda ok: "PASS" if ok else "FAIL"

    # Coulomb counting over the run from the logged samples (start-of-step values,
    # so the SoC change between samples a and b is the sum of I over a..b-1)
    soc_change = soc.iloc[-1] - soc.iloc[1]
    soc_coulomb = -float((I.iloc[1:-1] * dt / (3600.0 * cap.iloc[1:-1])).sum())
    soc_std = isd["battery_stored_energy_J"] / isd["battery_capacity_J"]

    def flag_time(col):
        return f"{isd[col].sum() * dt:.0f} s ({100 * isd[col].mean():.1f} %)"

    def c_rate(amps):
        return f"{amps:.2f} A ({amps / cap.max():.2f} C)"

    rows = [
        ("Pack voltage range", f"{V.min():.3f} - {V.max():.3f} V", "-", "info"),
        ("Cell voltage at or above BMS cutoff", f"min {v_cell.min():.4f} V", f">= {cutoff:.2f} V",
         status(v_cell.min() >= cutoff - 1e-6)),
        ("Cell voltage at or below CV charge limit", f"max {v_cell.max():.4f} V",
         f"<= {config.BATTERY_MAX_CELL_V:.2f} V", status(v_cell.max() <= config.BATTERY_MAX_CELL_V + 1e-6)),
        ("Max discharge current", c_rate(I.max()), "-", "info"),
        ("Max charge current", c_rate(-I.min()), "-", "info"),
        ("Sunlit: mean voltage / current", f"{V[sunlit].mean():.3f} V / {I[sunlit].mean():+.3f} A", "-", "info"),
        ("Eclipse: mean voltage / current", f"{V[~sunlit].mean():.3f} V / {I[~sunlit].mean():+.3f} A", "-", "info"),
        ("Power demand = -(sum of power nodes)", f"max diff {np.abs(demand + nodes).max():.2e} W", "< 1e-6 W",
         status(np.abs(demand + nodes).max() < 1e-6)),
        ("Delivered = demanded when not limited", f"max diff {np.abs(delivered - demand)[~limited].max():.2e} W",
         "< 1e-6 W", status(np.abs(delivered - demand)[~limited].max() < 1e-6)),
        ("Delivered power = V x I", f"max diff {np.abs(delivered - V * I).max():.2e} W", "< 1e-9 W",
         status(np.abs(delivered - V * I).max() < 1e-9)),
        ("Coulomb counting over the run", f"SoC change {100 * soc_change:+.3f} %, integral(I dt)/Ah "
         f"{100 * soc_coulomb:+.3f} %", f"diff < 0.5 % SoC ({dt:g} s samples)",
         status(abs(soc_change - soc_coulomb) < 0.005)),
        ("SoC agrees between packOutMsg and batPowerOutMsg", f"max diff {100 * np.abs(soc - soc_std).max():.4f} % SoC",
         "< 0.1 % SoC (packOutMsg is start of step)", status(np.abs(soc - soc_std).max() < 0.001)),
        ("heatLoss >= 0", f"min {isd['battery_heat_loss_W'].min():.2e} W, mean {isd['battery_heat_loss_W'].mean():.3f} W",
         ">= 0", status(isd["battery_heat_loss_W"].min() >= 0.0)),
        ("Time charge-limited (CV phase / full)", flag_time("battery_charge_limited"), "-", "info"),
        ("Time power-limited / at cutoff", f"{flag_time('battery_power_limited')} / {flag_time('battery_under_voltage')}",
         "-", "info"),
        ("Time depleted", flag_time("battery_depleted"), "-", "info"),
    ]
    return rows


def write_inventory(path, recorder, frames, run_info):
    channels = recorder.channels
    lines = ["# Mini prototype - data inventory", "",
             "Generated by `src/mini_prototype/main.py` from the recorded output. Do not edit by hand.", "",
             "## Run", "",
             "| Item | Value |", "|---|---|"]
    for k, v in run_info.items():
        lines.append(f"| {k} | {v} |")

    # --- Coverage against the data-set table ----------------------------
    lines += ["", "## Coverage against the data-set table", "",
              "| Data set | Signals | Basilisk module(s) | In prototype | Recorded columns | Status |",
              "|---|---|---|---|---|---|"]
    for dataset, group, modules in DATASET_TABLE:
        in_group = [c for c in channels if c.dataset == dataset and c.group == group]
        present = [m for m in modules if any(c.module == m for c in in_group)]
        n_cols = sum(len(c.columns) for c in in_group)
        if not present:
            status = "Not modelled"
        elif len(present) == len(modules):
            status = "Recorded"
        else:
            status = "Partial"
        lines.append(f"| {dataset} | {group} | {', '.join(modules)} | "
                     f"{', '.join(present) or '-'} | {n_cols} | {status} |")

    # --- Channel list ---------------------------------------------------
    for dataset in (INTERNAL, NAVIGATION):
        df = frames[dataset]
        lines += ["", f"## {dataset} - `{CSV_NAMES[dataset]}`", "",
                  f"{len(df)} rows x {len(df.columns)} columns (incl. `time_s`), "
                  f"sampled every {config.LOG_STEP_S:g} s.", "",
                  "| Signal group | Module | Message.field | Column | Unit | Min | Max | Mean |",
                  "|---|---|---|---|---|---|---|---|"]
        for c in channels:
            if c.dataset != dataset:
                continue
            for col in c.columns:
                s = df[col]
                lines.append(f"| {c.group} | {c.module} | `{c.message}.{c.field}` | `{col}` | {c.unit} | "
                             f"{_fmt(s.min())} | {_fmt(s.max())} | {_fmt(s.mean())} |")

    # --- Nav noise ------------------------------------------------------
    lines += ["", "## simpleNav noise (nav minus truth, all axes pooled)", "",
              "The error is a bounded random walk, so it wanders out to the walk bound "
              "rather than staying within the 1-sigma step size.", "",
              "| Signal | Unit | Configured 1-sigma | Walk bound | Mean error | Std error | Max abs error |",
              "|---|---|---|---|---|---|---|"]
    for label, unit, sigma, bound, mean, std, mx in nav_noise_check(frames[NAVIGATION]):
        lines.append(f"| {label} | {unit} | {_fmt(sigma)} | {_fmt(bound)} | {_fmt(mean)} | "
                     f"{_fmt(std)} | {_fmt(mx)} |")

    # --- Sensor errors --------------------------------------------------
    lines += ["", "## Sensor errors (sensor minus truth, all axes pooled)", "",
              "| Sensor | Unit | Configured noise | Mean error | Std error | Max abs error |",
              "|---|---|---|---|---|---|"]
    for label, unit, spec, mean, std, mx in sensor_error_check(frames[INTERNAL], frames[NAVIGATION]):
        lines.append(f"| {label} | {unit} | {spec} | {_fmt(mean)} | {_fmt(std)} | {_fmt(mx)} |")

    # --- Battery voltage / current --------------------------------------
    lines += ["", "## Battery voltage and current (EcmBattery)", "",
              f"{config.BATTERY_N_SERIES}S{config.BATTERY_N_PARALLEL}P, {config.BATTERY_CELL_AH:g} Ah cells, "
              f"start at {100 * config.BATTERY_INIT_SOC:.0f} % SoC. Unregulated bus: the bus voltage is the "
              "pack voltage. Current + = discharge.", "",
              "| Check | Result | Criterion | Status |", "|---|---|---|---|"]
    for check, result, criterion, state in battery_electrical_check(frames[INTERNAL], frames[NAVIGATION]):
        lines.append(f"| {check} | {result} | {criterion} | {state} |")

    used = {c.module for c in channels}
    missing = [(d, g, [m for m in mods if m not in used]) for d, g, mods in DATASET_TABLE]
    missing = [(d, g, m) for d, g, m in missing if m]
    lines += ["", "## Not modelled in this prototype", ""]
    lines += [f"- **{d} / {g}:** {', '.join(m)}" for d, g, m in missing] or ["None - every module in the "
                                                                             "data-set table is recorded."]
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
