"""
Parameters for the mini prototype.

All tunable numbers live here so the setup modules stay free of magic values.
Values describe a small (6U-class) LEO satellite with three reaction wheels,
a body-mounted solar panel, a battery and a constant bus load.
"""

import os

from Basilisk.utilities import macros

# --- Run / task settings -------------------------------------------------
NUM_ORBITS = 3                # number of orbits to simulate
DYN_STEP_S = 1.0              # dynamics + FSW task rate [s]
LOG_STEP_S = 10.0             # recorder sampling period [s]
EPOCH = "2026 MAR 20 12:00:00.000 (UTC)"   # SPICE start epoch
RNG_SEED = 42                 # simpleNav noise seed (repeatable runs)

# Module execution order within the task (higher runs first). Ephemerides feed
# gravity, the spacecraft writes the current state, nav reads that state, FSW
# reads nav, and the wheel effector picks up the new command for the next step.
# The atmosphere runs before the spacecraft so drag sees a written density.
# Unlisted modules (eclipse, sensors, thermal, power, recorders) run after these.
PRIORITY = {
    "spice": 200,
    "ephemConverter": 150,
    "atmosphere": 130,
    "spacecraft": 120,
    "simpleNav": 110,
    "guidance": 99,
    "mrpFeedback": 98,
    "rwMotorTorque": 97,
    "rwEffector": 96,
}

# --- Output --------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "mini_prototype")

# --- Orbit: 500 km near-circular sun-synchronous ------------------------
ALTITUDE_M = 500.0e3
ECCENTRICITY = 0.001
INCLINATION_DEG = 97.4
RAAN_DEG = 0.0                # with the March equinox epoch this gives a noon-midnight orbit
ARG_PERIAPSIS_DEG = 0.0
TRUE_ANOMALY_DEG = 0.0

# --- Spacecraft hub ------------------------------------------------------
SC_NAME = "miniSat"
SC_MASS_KG = 10.0
SC_INERTIA = [0.12, 0.0, 0.0,
              0.0, 0.10, 0.0,
              0.0, 0.0, 0.05]     # kg m^2 about the centre of mass
SIGMA_BN_INIT = [[0.1], [0.2], [-0.3]]       # initial attitude (MRP)
OMEGA_BN_B_INIT = [[0.001], [-0.002], [0.003]]  # initial body rate [rad/s]

# --- Reaction wheels (three orthogonal BCT RWP015, friction on) ---------
RW_TYPE = "BCT_RWP015"
RW_AXES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
RW_INIT_SPEED_RPM = [500.0, 750.0, 1000.0]
RW_BASE_POWER_W = 0.5         # electronics draw per wheel [W]
RW_MECH_TO_ELEC_EFF = -1.0    # negative = no regenerative braking

# --- Attitude control (sun pointing with mrpFeedback) -------------------
POINT_AXIS_B = [0, 0, 1]      # body axis to point at the Sun (= panel normal)
MRP_K = 0.064
MRP_P = 0.08                  # ~critically damped, ~0.4 rad/s bandwidth
MRP_KI = 0.05                 # integral feedback rejects wheel friction (negative = off)

# --- Power system --------------------------------------------------------
PANEL_NORMAL_B = POINT_AXIS_B
PANEL_AREA_M2 = 0.06
PANEL_EFFICIENCY = 0.28
BUS_LOAD_W = -6.0             # constant bus/payload sink (negative = draw)
# Battery: EcmBattery from the detailed_battery extension (equivalent-circuit Li-ion pack).
# Cell resistances, RC branch and OCV curve use the module defaults (NMC, 3.0-4.2 V).
BATTERY_N_SERIES = 2          # 2S2P x 3.0 Ah ~ 45 Wh, ~7.4 V nominal bus
BATTERY_N_PARALLEL = 2
BATTERY_CELL_AH = 3.0
BATTERY_INIT_SOC = 0.8
BATTERY_CUTOFF_CELL_V = 3.0   # BMS discharge cutoff per cell
BATTERY_MAX_CELL_V = 4.20     # CV charge limit per cell

# --- simpleNav noise (1-sigma Gauss-Markov diagonal + random-walk bounds)
# State order: position(3), velocity(3), attitude MRP(3), rate(3), sun vector(3), accumulated dV(3)
NAV_SIGMA = {
    "pos_m": 5.0,
    "vel_mps": 0.05,
    "att_mrp": 1.0e-4,                      # ~0.02 deg
    "rate_radps": 1.0e-5,
    "sun_rad": 1.0e-3,
    "dv_mps": 0.0,
}
NAV_WALK_BOUNDS = {
    "pos_m": 20.0,
    "vel_mps": 0.2,
    "att_mrp": 5.0e-4,
    "rate_radps": 5.0e-5,
    "sun_rad": 5.0e-3,
    "dv_mps": 0.0,
}


# --- Environment: atmosphere, drag, magnetic field ----------------------
# Constant space weather for MSIS (moderate activity)
SW_AP = 8.0                   # geomagnetic Ap index (all 3-hourly/daily slots)
SW_F107 = 110.0               # F10.7 solar flux (81-day average and daily)
DRAG_AREA_M2 = 0.06           # projected area (cannonball model)
DRAG_COEFF = 2.2
DRAG_COM_OFFSET_B = [0.0, 0.0, 0.01]   # centre of pressure offset -> drag torque [m]

# --- Thermal --------------------------------------------------------------
# sensorThermal: small payload sensor on the sun-facing (+Z) face
SENSOR_THERMAL = {
    "nHat_B": POINT_AXIS_B,
    "T_0": 20.0,              # initial temperature [C]
    "area_m2": 0.01,
    "absorptivity": 0.25,
    "emissivity": 0.8,
    "mass_kg": 0.1,
    "specific_heat": 890.0,   # aluminium [J/kg/K]
    "power_W": 1.0,           # electrical dissipation
}
# motorThermal: one per reaction wheel motor
MOTOR_THERMAL = {
    "T_0": 20.0,              # initial temperature [C]
    "ambient_C": 20.0,        # spacecraft internal temperature [C]
    "ambient_resistance": 5.0,  # [K/W]
    "heat_capacity": 50.0,    # [J/K]
    "efficiency": 0.7,        # electrical -> mechanical
}
# tempMeasurement: noisy readout of every temperature above
TEMP_MEAS_NOISE_C = 0.1
TEMP_MEAS_BIAS_C = 0.05
TEMP_MEAS_WALK_C = 0.5

# --- Attitude sensors -----------------------------------------------------
IMU_GYRO_NOISE = 1.0e-5       # [rad/s] 1-sigma
IMU_GYRO_WALK = 5.0e-5        # [rad/s] random-walk bound
IMU_GYRO_BIAS = [1.0e-5, -2.0e-5, 1.5e-5]   # [rad/s]
IMU_ACCEL_NOISE = 1.0e-5      # [m/s^2]
IMU_ACCEL_WALK = 5.0e-5
ST_NOISE_RAD = 1.0e-4         # star tracker 1-sigma per axis (~20 arcsec)
ST_WALK_RAD = 5.0e-4
MAG_NOISE_T = 1.0e-7          # magnetometer 1-sigma per axis (100 nT)
MAG_BIAS_T = [5.0e-8, -5.0e-8, 2.0e-8]
MAG_WALK_T = 5.0e-7
MAG_RANGE_T = 1.0e-4          # saturation +/-100 uT
CSS_NORMALS_B = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
CSS_FOV_DEG = 80.0            # half-angle field of view
CSS_NOISE = 0.002             # on the normalised cosine output


def dyn_step_ns():
    return macros.sec2nano(DYN_STEP_S)


def log_step_ns():
    return macros.sec2nano(LOG_STEP_S)
