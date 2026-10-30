"""
Message recorders and the two output data sets.

Every recorded column is registered as a Channel, so the inventory can be
built from what was actually logged rather than from a hand-written list.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from Basilisk.utilities import macros

INTERNAL = "Internal sensors"
NAVIGATION = "Navigation"

# Signal groups, worded as in the feasibility data-set table
G_POWER = "Battery stored energy, capacity, net power"
G_BAT_ELEC = "Battery voltage, current, electrical state"
G_RW = "Wheel speed, motor torques, friction"
G_TEMP = "Temperatures"
G_SENSORS = "Rates, attitude, field and sun sensors"
G_NAV = "Position, velocity, attitude, with noise"
G_ENV = "Ephemerides, eclipse, atmosphere, drag"
G_TRUTH = "Truth state (reference for nav noise)"
G_GUID = "Pointing error (guidance)"

XYZ = ["x", "y", "z"]


@dataclass
class Channel:
    dataset: str
    group: str
    module: str
    message: str
    field: str
    unit: str
    columns: List[str]
    rec_key: str
    index: Optional[List[int]] = None
    scale: float = 1.0
    stats: dict = field(default_factory=dict)


class DataRecorder:

    def __init__(self, sim, task_name, sample_ns):
        self.sim = sim
        self.task_name = task_name
        self.sample_ns = sample_ns
        self.recorders = {}
        self.channels: List[Channel] = []

    def _record(self, key, make_recorder):
        if key not in self.recorders:
            rec = make_recorder()
            self.sim.AddModelToTask(self.task_name, rec)
            self.recorders[key] = rec

    def add(self, dataset, group, module, msg_name, msg, fld, unit, columns,
            index=None, scale=1.0):
        """Record field `fld` of message `msg`."""
        key = f"{module}.{msg_name}"
        self._record(key, lambda: msg.recorder(self.sample_ns))
        self.channels.append(Channel(dataset, group, module, msg_name, fld, unit,
                                     columns, key, index, scale))

    def add_var(self, dataset, group, module, model, var, unit, columns):
        """Record module variable `var` (for values not published in a message)."""
        key = f"{module}.{var}"
        self._record(key, lambda: model.logger(var, self.sample_ns))
        self.channels.append(Channel(dataset, group, module, "(module variable)", var, unit,
                                     columns, key))

    # ------------------------------------------------------------------
    def setup(self, sc, env, rws, power, nav, fsw, thermal, sensors):
        """Register every recorded signal of the prototype."""
        # --- Internal sensors: power -------------------------------------
        bat = power.battery.batPowerOutMsg
        self.add(INTERNAL, G_POWER, "EcmBattery", "batPowerOutMsg", bat,
                 "storageLevel", "J", ["battery_stored_energy_J"])
        self.add(INTERNAL, G_POWER, "EcmBattery", "batPowerOutMsg", bat,
                 "storageCapacity", "J", ["battery_capacity_J"])
        self.add(INTERNAL, G_POWER, "EcmBattery", "batPowerOutMsg", bat,
                 "currentNetPower", "W", ["battery_net_power_W"])
        self.add(INTERNAL, G_POWER, "simpleSolarPanel", "nodePowerOutMsg",
                 power.panel.nodePowerOutMsg, "netPower", "W", ["solar_panel_power_W"])
        self.add(INTERNAL, G_POWER, "simplePowerSink", "nodePowerOutMsg",
                 power.bus_load.nodePowerOutMsg, "netPower", "W", ["bus_load_power_W"])
        for i, p in enumerate(power.rw_power):
            self.add(INTERNAL, G_POWER, "ReactionWheelPower", f"nodePowerOutMsg[{i}]",
                     p.nodePowerOutMsg, "netPower", "W", [f"rw{i + 1}_power_W"])

        # --- Internal sensors: battery electrical state (EcmBattery packOutMsg)
        # Every field describes the start of the battery's 1 s step; current + = discharge.
        pack = power.battery.packOutMsg
        for fld, unit, col in [
            ("packVoltage", "V", "battery_voltage_V"),
            ("packCurrent", "A", "battery_current_A"),
            ("cellVoltage", "V", "battery_cell_voltage_V"),
            ("openCircuitVoltage", "V", "battery_ocv_V"),
            ("polarizationVoltage", "V", "battery_rc_voltage_V"),
            ("stateOfCharge", "-", "battery_soc"),
            ("internalResistance", "Ohm", "battery_r0_Ohm"),
            ("capacityAh", "Ah", "battery_capacity_Ah"),
            ("powerDemand", "W", "battery_power_demand_W"),
            ("powerDelivered", "W", "battery_power_delivered_W"),
            ("heatLoss", "W", "battery_heat_loss_W"),
            ("activeStrings", "-", "battery_active_strings"),
            ("activeSeriesCells", "-", "battery_active_series_cells"),
            ("powerLimited", "flag", "battery_power_limited"),
            ("underVoltage", "flag", "battery_under_voltage"),
            ("chargeLimited", "flag", "battery_charge_limited"),
            ("depleted", "flag", "battery_depleted"),
        ]:
            self.add(INTERNAL, G_BAT_ELEC, "EcmBattery", "packOutMsg", pack, fld, unit, [col])

        # --- Internal sensors: reaction wheels ---------------------------
        n = rws.num
        wheels = list(range(n))
        self.add(INTERNAL, G_RW, "reactionWheels", "rwSpeedOutMsg", rws.speed_msg,
                 "wheelSpeeds", "rpm", [f"rw{i + 1}_speed_rpm" for i in wheels],
                 index=wheels, scale=1.0 / macros.RPM)
        self.add(INTERNAL, G_RW, "rwMotorTorque (FSW)", "rwMotorTorqueOutMsg", fsw.motor_cmd_msg,
                 "motorTorque", "N m", [f"rw{i + 1}_cmd_torque_Nm" for i in wheels], index=wheels)
        for i in wheels:
            msg = rws.wheel_msgs[i]
            self.add(INTERNAL, G_RW, "reactionWheels", f"rwOutMsgs[{i}]", msg,
                     "u_current", "N m", [f"rw{i + 1}_motor_torque_Nm"])
            self.add(INTERNAL, G_RW, "reactionWheels", f"rwOutMsgs[{i}]", msg,
                     "frictionTorque", "N m", [f"rw{i + 1}_friction_torque_Nm"])

        # --- Internal sensors: temperatures ------------------------------
        self.add(INTERNAL, G_TEMP, "sensorThermal", "temperatureOutMsg", thermal.sensor.temperatureOutMsg,
                 "temperature", "degC", ["sensor_temp_C"])
        for i, mt in enumerate(thermal.motors):
            self.add(INTERNAL, G_TEMP, "motorThermal", f"temperatureOutMsg[{i}]", mt.temperatureOutMsg,
                     "temperature", "degC", [f"rw{i + 1}_motor_temp_C"])
        meas_names = ["sensor"] + [f"rw{i + 1}_motor" for i in range(len(thermal.motors))]
        for i, (tm, name) in enumerate(zip(thermal.measurements, meas_names)):
            self.add(INTERNAL, G_TEMP, "tempMeasurement", f"tempOutMsg[{i}]", tm.tempOutMsg,
                     "temperature", "degC", [f"meas_{name}_temp_C"])

        # --- Internal sensors: attitude sensors --------------------------
        imu = sensors.imu.sensorOutMsg
        self.add(INTERNAL, G_SENSORS, "imuSensor", "sensorOutMsg", imu,
                 "AngVelPlatform", "rad/s", [f"imu_omega_P_{a}" for a in XYZ])
        self.add(INTERNAL, G_SENSORS, "imuSensor", "sensorOutMsg", imu,
                 "AccelPlatform", "m/s^2", [f"imu_accel_P_{a}" for a in XYZ])
        self.add(INTERNAL, G_SENSORS, "imuSensor", "sensorOutMsg", imu,
                 "DVFramePlatform", "m/s", [f"imu_dv_P_{a}" for a in XYZ])
        self.add(INTERNAL, G_SENSORS, "imuSensor", "sensorOutMsg", imu,
                 "DRFramePlatform", "rad", [f"imu_dtheta_P_{a}" for a in XYZ])
        self.add(INTERNAL, G_SENSORS, "starTracker", "sensorOutMsg", sensors.star_tracker.sensorOutMsg,
                 "qInrtl2Case", "quaternion", [f"st_q_{i}" for i in range(4)])
        self.add(INTERNAL, G_SENSORS, "magnetometer", "tamDataOutMsg", sensors.magnetometer.tamDataOutMsg,
                 "tam_S", "T", [f"mag_B_S_{a}" for a in XYZ])
        css = list(range(sensors.num_css))
        self.add(INTERNAL, G_SENSORS, "coarseSunSensor", "constellationOutMsg",
                 sensors.css_array.constellationOutMsg, "CosValue", "-",
                 [f"css{i + 1}_cos" for i in css], index=css)

        # --- Navigation: simpleNav (noisy) -------------------------------
        self.add(NAVIGATION, G_NAV, "simpleNav", "transOutMsg", nav.transOutMsg,
                 "r_BN_N", "m", [f"nav_r_BN_N_{a}" for a in XYZ])
        self.add(NAVIGATION, G_NAV, "simpleNav", "transOutMsg", nav.transOutMsg,
                 "v_BN_N", "m/s", [f"nav_v_BN_N_{a}" for a in XYZ])
        self.add(NAVIGATION, G_NAV, "simpleNav", "attOutMsg", nav.attOutMsg,
                 "sigma_BN", "MRP", [f"nav_sigma_BN_{i}" for i in (1, 2, 3)])
        self.add(NAVIGATION, G_NAV, "simpleNav", "attOutMsg", nav.attOutMsg,
                 "omega_BN_B", "rad/s", [f"nav_omega_BN_B_{a}" for a in XYZ])
        self.add(NAVIGATION, G_NAV, "simpleNav", "attOutMsg", nav.attOutMsg,
                 "vehSunPntBdy", "-", [f"nav_sun_B_{a}" for a in XYZ])

        # --- Navigation: ephemerides --------------------------------------
        self.add(NAVIGATION, G_ENV, "spiceInterface", "spiceTimeOutMsg", env.time_msg,
                 "J2000Current", "s", ["spice_et_J2000_s"])
        self.add(NAVIGATION, G_ENV, "spiceInterface", "spiceTimeOutMsg", env.time_msg,
                 "JulianDateCurrent", "day", ["spice_julian_date"])
        for body in ("sun", "moon"):
            msg = env.planet_msgs[body]
            self.add(NAVIGATION, G_ENV, "spiceInterface", f"planetStateOutMsgs[{body}]", msg,
                     "PositionVector", "m", [f"{body}_r_N_{a}" for a in XYZ])
            self.add(NAVIGATION, G_ENV, "spiceInterface", f"planetStateOutMsgs[{body}]", msg,
                     "VelocityVector", "m/s", [f"{body}_v_N_{a}" for a in XYZ])
        earth = env.ephem_msgs["earth"]
        self.add(NAVIGATION, G_ENV, "ephemerisConverter", "ephemOutMsgs[earth]", earth,
                 "sigma_BN", "MRP", [f"earth_sigma_PN_{i}" for i in (1, 2, 3)])
        self.add(NAVIGATION, G_ENV, "ephemerisConverter", "ephemOutMsgs[earth]", earth,
                 "omega_BN_B", "rad/s", [f"earth_omega_PN_P_{a}" for a in XYZ])

        # --- Navigation: eclipse -----------------------------------------
        self.add(NAVIGATION, G_ENV, "eclipse", "eclipseOutMsgs[0]", env.eclipse_msg,
                 "shadowFactor", "-", ["eclipse_shadow_factor"])
        self.add(NAVIGATION, G_ENV, "eclipse", "eclipseOutMsgs[0]", env.eclipse_msg,
                 "illuminationFactor", "-", ["eclipse_illumination_factor"])

        # --- Navigation: atmosphere, drag, magnetic field ---------------
        self.add(NAVIGATION, G_ENV, "MsisAtmosphere", "envOutMsgs[0]", env.atmo_msg,
                 "neutralDensity", "kg/m^3", ["atmo_density_kgm3"])
        self.add(NAVIGATION, G_ENV, "MsisAtmosphere", "envOutMsgs[0]", env.atmo_msg,
                 "localTemp", "K", ["atmo_temp_K"])
        self.add_var(NAVIGATION, G_ENV, "dragEffector", env.drag,
                     "forceExternal_B", "N", [f"drag_force_B_{a}" for a in XYZ])
        self.add_var(NAVIGATION, G_ENV, "dragEffector", env.drag,
                     "torqueExternalPntB_B", "N m", [f"drag_torque_B_{a}" for a in XYZ])
        self.add(NAVIGATION, G_ENV, "magneticFieldWMM", "envOutMsgs[0]", env.mag_field_msg,
                 "magField_N", "T", [f"mag_field_N_{a}" for a in XYZ])

        # --- Navigation: truth + guidance (supporting columns) -----------
        st = sc.scStateOutMsg
        self.add(NAVIGATION, G_TRUTH, "spacecraft", "scStateOutMsg", st,
                 "r_BN_N", "m", [f"true_r_BN_N_{a}" for a in XYZ])
        self.add(NAVIGATION, G_TRUTH, "spacecraft", "scStateOutMsg", st,
                 "v_BN_N", "m/s", [f"true_v_BN_N_{a}" for a in XYZ])
        self.add(NAVIGATION, G_TRUTH, "spacecraft", "scStateOutMsg", st,
                 "sigma_BN", "MRP", [f"true_sigma_BN_{i}" for i in (1, 2, 3)])
        self.add(NAVIGATION, G_TRUTH, "spacecraft", "scStateOutMsg", st,
                 "omega_BN_B", "rad/s", [f"true_omega_BN_B_{a}" for a in XYZ])
        self.add(NAVIGATION, G_GUID, "locationPointing (FSW)", "attGuidOutMsg", fsw.att_guid_msg,
                 "sigma_BR", "MRP", [f"att_err_sigma_BR_{i}" for i in (1, 2, 3)])

    # ------------------------------------------------------------------
    def datasets(self):
        """Return {dataset name: DataFrame} built from the recorded channels."""
        first = next(iter(self.recorders.values()))
        time_s = first.times() * macros.NANO2SEC
        frames = {INTERNAL: {"time_s": time_s}, NAVIGATION: {"time_s": time_s}}

        for ch in self.channels:
            data = np.asarray(getattr(self.recorders[ch.rec_key], ch.field), dtype=float)
            if data.ndim == 1:
                data = data[:, None]
            if ch.index is not None:
                data = data[:, ch.index]
            data = data * ch.scale
            if data.shape != (len(time_s), len(ch.columns)):
                raise RuntimeError(f"{ch.rec_key}.{ch.field}: got shape {data.shape}, "
                                   f"expected {(len(time_s), len(ch.columns))}")
            for j, col in enumerate(ch.columns):
                frames[ch.dataset][col] = data[:, j]
            ch.stats = {"samples": data.shape[0], "min": data.min(),
                        "max": data.max(), "mean": data.mean()}

        return {name: pd.DataFrame(cols) for name, cols in frames.items()}
