"""
Power system: solar panel (eclipse-aware), constant bus load, per-wheel
power draw and an equivalent-circuit battery (EcmBattery, from the
detailed_battery C++ extension) that integrates all power nodes and
models pack voltage and current.

Based on scenarioPowerDemo.py and scenarioAttitudeFeedbackRWPower.py.
"""

from Basilisk.simulation import ReactionWheelPower, simplePowerSink, simpleSolarPanel
from detailed_battery import ecmBattery

import config


class PowerSystem:

    def __init__(self, sim, task_name, sc, env, rws):
        self.panel = simpleSolarPanel.SimpleSolarPanel()
        self.panel.ModelTag = "solarPanel"
        self.panel.stateInMsg.subscribeTo(sc.scStateOutMsg)
        self.panel.sunInMsg.subscribeTo(env.sun_msg)
        self.panel.sunEclipseInMsg.subscribeTo(env.eclipse_msg)
        self.panel.setPanelParameters(config.PANEL_NORMAL_B, config.PANEL_AREA_M2, config.PANEL_EFFICIENCY)
        sim.AddModelToTask(task_name, self.panel)

        self.bus_load = simplePowerSink.SimplePowerSink()
        self.bus_load.ModelTag = "busLoad"
        self.bus_load.nodePowerOut = config.BUS_LOAD_W
        sim.AddModelToTask(task_name, self.bus_load)

        self.rw_power = []
        for i in range(rws.num):
            p = ReactionWheelPower.ReactionWheelPower()
            p.ModelTag = f"rwPower{i}"
            p.basePowerNeed = config.RW_BASE_POWER_W
            p.mechToElecEfficiency = config.RW_MECH_TO_ELEC_EFF
            p.rwStateInMsg.subscribeTo(rws.wheel_msgs[i])
            sim.AddModelToTask(task_name, p)
            self.rw_power.append(p)

        self.battery = ecmBattery.EcmBattery()
        self.battery.ModelTag = "battery"
        self.battery.nSeries = config.BATTERY_N_SERIES
        self.battery.nParallel = config.BATTERY_N_PARALLEL
        self.battery.cellCapacityAh = config.BATTERY_CELL_AH
        self.battery.initialSoc = config.BATTERY_INIT_SOC
        self.battery.cutoffCellVoltage = config.BATTERY_CUTOFF_CELL_V
        self.battery.maxCellVoltage = config.BATTERY_MAX_CELL_V
        self.battery.addPowerNodeToModel(self.panel.nodePowerOutMsg)
        self.battery.addPowerNodeToModel(self.bus_load.nodePowerOutMsg)
        for p in self.rw_power:
            self.battery.addPowerNodeToModel(p.nodePowerOutMsg)
        sim.AddModelToTask(task_name, self.battery)
