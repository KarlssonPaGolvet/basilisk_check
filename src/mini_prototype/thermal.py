"""
Thermal: a sun-facing payload sensor (sensorThermal), one motorThermal per
reaction wheel and a noisy tempMeasurement readout of every temperature.

Based on scenarioSensorThermal.py.
"""

from Basilisk.simulation import motorThermal, sensorThermal, tempMeasurement

import config


class Thermal:

    def __init__(self, sim, task_name, sc, env, rws):
        p = config.SENSOR_THERMAL
        self.sensor = sensorThermal.SensorThermal()
        self.sensor.ModelTag = "sensorThermal"
        self.sensor.nHat_B = p["nHat_B"]
        self.sensor.T_0 = p["T_0"]
        self.sensor.sensorArea = p["area_m2"]
        self.sensor.sensorAbsorptivity = p["absorptivity"]
        self.sensor.sensorEmissivity = p["emissivity"]
        self.sensor.sensorMass = p["mass_kg"]
        self.sensor.sensorSpecificHeat = p["specific_heat"]
        self.sensor.sensorPowerDraw = p["power_W"]
        self.sensor.sunInMsg.subscribeTo(env.sun_msg)
        self.sensor.stateInMsg.subscribeTo(sc.scStateOutMsg)
        self.sensor.sunEclipseInMsg.subscribeTo(env.eclipse_msg)
        sim.AddModelToTask(task_name, self.sensor)

        m = config.MOTOR_THERMAL
        self.motors = []
        for i in range(rws.num):
            mt = motorThermal.MotorThermal()
            mt.ModelTag = f"motorThermal{i}"
            mt.currentTemperature = m["T_0"]
            mt.ambientTemperature = m["ambient_C"]
            mt.ambientThermalResistance = m["ambient_resistance"]
            mt.motorHeatCapacity = m["heat_capacity"]
            mt.efficiency = m["efficiency"]
            mt.rwStateInMsg.subscribeTo(rws.wheel_msgs[i])
            sim.AddModelToTask(task_name, mt)
            self.motors.append(mt)

        # Noisy readouts: sensor first, then each motor
        self.measurements = []
        sources = [self.sensor.temperatureOutMsg] + [mt.temperatureOutMsg for mt in self.motors]
        for i, src in enumerate(sources):
            tm = tempMeasurement.TempMeasurement()
            tm.ModelTag = f"tempMeasurement{i}"
            tm.senNoiseStd = config.TEMP_MEAS_NOISE_C
            tm.senBias = config.TEMP_MEAS_BIAS_C
            tm.walkBounds = config.TEMP_MEAS_WALK_C
            tm.RNGSeed = config.RNG_SEED + 10 + i
            tm.tempInMsg.subscribeTo(src)
            sim.AddModelToTask(task_name, tm)
            self.measurements.append(tm)
