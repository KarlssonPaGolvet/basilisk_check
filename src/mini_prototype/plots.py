"""Summary figure of the two data sets."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from recorders import INTERNAL, NAVIGATION, XYZ  # noqa: E402


def plot_summary(frames, num_rw, num_css, path):
    isd, nav = frames[INTERNAL], frames[NAVIGATION]
    t = isd["time_s"] / 60.0

    fig, axs = plt.subplots(6, 2, figsize=(13, 19), sharex=True)

    ax = axs[0, 0]
    soc = 100.0 * isd["battery_stored_energy_J"] / isd["battery_capacity_J"]
    ax.plot(t, soc, label="Battery SoC [%]")
    ax.plot(t, 100.0 * nav["eclipse_shadow_factor"], "--", alpha=0.6, label="Sun visibility [%]")
    ax.set_ylabel("%")
    ax.set_title("Battery state of charge and eclipse")
    ax.legend(loc="lower left")

    ax = axs[0, 1]
    ax.plot(t, isd["solar_panel_power_W"], label="Solar panel")
    ax.plot(t, isd["bus_load_power_W"], label="Bus load")
    for i in range(num_rw):
        ax.plot(t, isd[f"rw{i + 1}_power_W"], label=f"RW{i + 1}")
    ax.plot(t, isd["battery_net_power_W"], "k", lw=1, label="Battery net")
    ax.set_ylabel("Power [W]")
    ax.set_title("Power nodes")
    ax.legend(loc="lower left", ncol=3, fontsize=8)

    ax = axs[1, 0]
    for i in range(num_rw):
        ax.plot(t, isd[f"rw{i + 1}_speed_rpm"], label=f"RW{i + 1}")
    ax.set_ylabel("Speed [rpm]")
    ax.set_title("Wheel speeds")
    ax.legend()

    ax = axs[1, 1]
    for i in range(num_rw):
        line, = ax.plot(t, 1e3 * isd[f"rw{i + 1}_motor_torque_Nm"], label=f"RW{i + 1} motor")
        ax.plot(t, 1e3 * isd[f"rw{i + 1}_friction_torque_Nm"], ":", color=line.get_color(),
                label=f"RW{i + 1} friction")
    ax.set_ylabel("Torque [mN m]")
    ax.set_title("Wheel motor and friction torque")
    ax.legend(ncol=2, fontsize=8)

    ax = axs[2, 0]
    sig = nav[[f"att_err_sigma_BR_{i}" for i in (1, 2, 3)]].to_numpy()
    err_deg = np.degrees(4.0 * np.arctan(np.linalg.norm(sig, axis=1)))
    ax.semilogy(t, np.maximum(err_deg, 1e-4))
    ax.set_ylabel("Error [deg]")
    ax.set_title("Sun-pointing error")

    ax = axs[2, 1]
    for a in XYZ:
        ax.plot(t, nav[f"nav_r_BN_N_{a}"] - nav[f"true_r_BN_N_{a}"], label=a)
    ax.set_ylabel("Error [m]")
    ax.set_title("simpleNav position error (nav - truth)")
    ax.legend()

    ax = axs[3, 0]
    line, = ax.plot(t, isd["sensor_temp_C"], label="Payload sensor")
    ax.plot(t, isd["meas_sensor_temp_C"], ".", ms=1, color=line.get_color(), alpha=0.4)
    for i in range(num_rw):
        ax.plot(t, isd[f"rw{i + 1}_motor_temp_C"], label=f"RW{i + 1} motor")
    ax.set_ylabel("Temperature [degC]")
    ax.set_title("Temperatures (lines: model, dots: tempMeasurement)")
    ax.legend(fontsize=8)

    ax = axs[3, 1]
    ax.semilogy(t, nav["atmo_density_kgm3"], label="MSIS density")
    ax.set_ylabel("Density [kg/m^3]")
    ax2 = ax.twinx()
    drag = np.linalg.norm(nav[[f"drag_force_B_{a}" for a in XYZ]].to_numpy(), axis=1)
    ax2.plot(t, 1e6 * drag, color="C1", label="Drag force")
    ax2.set_ylabel("Drag [uN]")
    ax.set_title("Atmosphere and drag")
    ax.legend(handles=ax.get_lines() + ax2.get_lines(), fontsize=8, loc="upper right")

    ax = axs[4, 0]
    for i in range(num_css):
        ax.plot(t, isd[f"css{i + 1}_cos"], label=f"CSS{i + 1}")
    ax.set_ylabel("Cosine output [-]")
    ax.set_title("Coarse sun sensors")
    ax.legend(ncol=3, fontsize=8)

    ax = axs[4, 1]
    for a in XYZ:
        line, = ax.plot(t, 1e6 * isd[f"mag_B_S_{a}"], label=f"TAM {a}")
    ax.plot(t, 1e6 * np.linalg.norm(nav[[f"mag_field_N_{a}" for a in XYZ]].to_numpy(), axis=1),
            "k--", lw=1, label="|B| WMM")
    ax.set_ylabel("Field [uT]")
    ax.set_title("Magnetometer (body) and WMM field magnitude")
    ax.legend(ncol=2, fontsize=8)

    ax = axs[5, 0]
    ax.plot(t, isd["battery_voltage_V"], label="Terminal (bus) voltage")
    ax.plot(t, isd["battery_ocv_V"], "--", lw=1, label="Open-circuit voltage")
    ax.axhline(config.BATTERY_N_SERIES * config.BATTERY_CUTOFF_CELL_V, color="r", ls=":", lw=1,
               label="BMS cutoff")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("Battery pack voltage")
    ax.set_xlabel("Time [min]")
    ax.legend(fontsize=8)

    ax = axs[5, 1]
    ax.plot(t, isd["battery_current_A"], label="Pack current (+ discharge)")
    full = isd["battery_charge_limited"] > 0
    ax.fill_between(t, 0, 1, where=full, transform=ax.get_xaxis_transform(), color="C2", alpha=0.15,
                    label="Charge-limited (CV / full)")
    ax.set_ylabel("Current [A]")
    ax.set_title("Battery pack current")
    ax.set_xlabel("Time [min]")
    ax.legend(fontsize=8)

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
