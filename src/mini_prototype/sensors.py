"""
Attitude sensors: IMU (gyro + accelerometer), star tracker, three-axis
magnetometer (fed by WMM) and six coarse sun sensors, one per body face.

All sensor frames are aligned with the body frame, so their outputs can be
compared directly against the spacecraft truth state.
"""

import numpy as np

from Basilisk.simulation import coarseSunSensor, imuSensor, magnetometer, starTracker
from Basilisk.utilities import macros

import config


class AttitudeSensors:

    def __init__(self, sim, task_name, sc, env):
        eye3 = np.eye(3).tolist()

        # IMU
        self.imu = imuSensor.ImuSensor()
        self.imu.ModelTag = "imuSensor"
        self.imu.scStateInMsg.subscribeTo(sc.scStateOutMsg)
        self.imu.setBodyToPlatformDCM(0.0, 0.0, 0.0)
        self.imu.PMatrixGyro = (np.eye(3) * config.IMU_GYRO_NOISE).tolist()
        self.imu.setAMatrixGyro(eye3)
        self.imu.setWalkBoundsGyro([config.IMU_GYRO_WALK] * 3)
        self.imu.senRotBias = config.IMU_GYRO_BIAS
        self.imu.PMatrixAccel = (np.eye(3) * config.IMU_ACCEL_NOISE).tolist()
        self.imu.setAMatrixAccel(eye3)
        self.imu.setWalkBoundsAccel([config.IMU_ACCEL_WALK] * 3)
        self.imu.RNGSeed = config.RNG_SEED + 1
        sim.AddModelToTask(task_name, self.imu)

        # Star tracker
        self.star_tracker = starTracker.StarTracker()
        self.star_tracker.ModelTag = "starTracker"
        self.star_tracker.scStateInMsg.subscribeTo(sc.scStateOutMsg)
        self.star_tracker.dcm_CB = eye3
        self.star_tracker.PMatrix = (np.eye(3) * config.ST_NOISE_RAD).tolist()
        self.star_tracker.setAMatrix(eye3)
        self.star_tracker.setWalkBounds([config.ST_WALK_RAD] * 3)
        self.star_tracker.RNGSeed = config.RNG_SEED + 2
        sim.AddModelToTask(task_name, self.star_tracker)

        # Magnetometer
        self.magnetometer = magnetometer.Magnetometer()
        self.magnetometer.ModelTag = "magnetometer"
        self.magnetometer.stateInMsg.subscribeTo(sc.scStateOutMsg)
        self.magnetometer.magInMsg.subscribeTo(env.mag_field_msg)
        self.magnetometer.setBodyToSensorDCM(0.0, 0.0, 0.0)
        self.magnetometer.scaleFactor = 1.0
        self.magnetometer.senNoiseStd = [config.MAG_NOISE_T] * 3
        self.magnetometer.senBias = config.MAG_BIAS_T
        self.magnetometer.walkBounds = [config.MAG_WALK_T] * 3
        self.magnetometer.maxOutput = config.MAG_RANGE_T
        self.magnetometer.minOutput = -config.MAG_RANGE_T
        self.magnetometer.RNGSeed = config.RNG_SEED + 3
        sim.AddModelToTask(task_name, self.magnetometer)

        # Coarse sun sensors, collected in one constellation (only it runs in the task)
        self.css_list = []
        for i, n_hat in enumerate(config.CSS_NORMALS_B):
            css = coarseSunSensor.CoarseSunSensor()
            css.ModelTag = f"css{i}"
            css.nHat_B = n_hat
            css.fov = config.CSS_FOV_DEG * macros.D2R
            css.scaleFactor = 1.0
            css.maxOutput = 2.0
            css.minOutput = 0.0
            css.senNoiseStd = config.CSS_NOISE
            css.RNGSeed = config.RNG_SEED + 20 + i
            css.sunInMsg.subscribeTo(env.sun_msg)
            css.stateInMsg.subscribeTo(sc.scStateOutMsg)
            css.sunEclipseInMsg.subscribeTo(env.eclipse_msg)
            self.css_list.append(css)
        self.css_array = coarseSunSensor.CSSConstellation()
        self.css_array.ModelTag = "cssConstellation"
        self.css_array.sensorList = coarseSunSensor.CSSVector(self.css_list)
        sim.AddModelToTask(task_name, self.css_array)

    @property
    def num_css(self):
        return len(self.css_list)
