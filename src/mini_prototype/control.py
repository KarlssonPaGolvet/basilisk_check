"""
Flight software: point the solar panel (+Z body) at the Sun using the noisy
simpleNav solution, an MRP feedback law and the reaction wheels.

Based on scenarioAttitudeFeedbackRW.py (mrpFeedback + rwMotorTorque) and
scenarioSensorThermal.py (locationPointing at the Sun).
"""

from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import locationPointing, mrpFeedback, rwMotorTorque

import config


class SunPointingControl:

    def __init__(self, sim, task_name, nav, sun_ephem_msg, rws):
        # Guidance: align POINT_AXIS_B with the spacecraft-to-Sun vector
        self.guidance = locationPointing.locationPointing()
        self.guidance.ModelTag = "sunPointing"
        self.guidance.pHat_B = config.POINT_AXIS_B
        self.guidance.scAttInMsg.subscribeTo(nav.attOutMsg)
        self.guidance.scTransInMsg.subscribeTo(nav.transOutMsg)
        self.guidance.celBodyInMsg.subscribeTo(sun_ephem_msg)
        self.guidance.useBoresightRateDamping = 1  # also null the spin about the panel normal
        sim.AddModelToTask(task_name, self.guidance, config.PRIORITY["guidance"])

        # Vehicle configuration known to FSW
        vc_payload = messaging.VehicleConfigMsgPayload(ISCPntB_B=config.SC_INERTIA)
        self.vc_msg = messaging.VehicleConfigMsg().write(vc_payload)

        # Control law
        self.mrp = mrpFeedback.mrpFeedback()
        self.mrp.ModelTag = "mrpFeedback"
        self.mrp.guidInMsg.subscribeTo(self.guidance.attGuidOutMsg)
        self.mrp.vehConfigInMsg.subscribeTo(self.vc_msg)
        self.mrp.rwParamsInMsg.subscribeTo(rws.fsw_config_msg)
        self.mrp.rwSpeedsInMsg.subscribeTo(rws.speed_msg)
        self.mrp.K = config.MRP_K
        self.mrp.P = config.MRP_P
        self.mrp.Ki = config.MRP_KI
        self.mrp.integralLimit = 2.0 / self.mrp.Ki * 0.1
        sim.AddModelToTask(task_name, self.mrp, config.PRIORITY["mrpFeedback"])

        # Map body torque onto the wheels
        self.motor_torque = rwMotorTorque.rwMotorTorque()
        self.motor_torque.ModelTag = "rwMotorTorque"
        self.motor_torque.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        self.motor_torque.vehControlInMsg.subscribeTo(self.mrp.cmdTorqueOutMsg)
        self.motor_torque.rwParamsInMsg.subscribeTo(rws.fsw_config_msg)
        rws.effector.rwMotorCmdInMsg.subscribeTo(self.motor_torque.rwMotorTorqueOutMsg)
        sim.AddModelToTask(task_name, self.motor_torque, config.PRIORITY["rwMotorTorque"])

    @property
    def att_guid_msg(self):
        return self.guidance.attGuidOutMsg

    @property
    def motor_cmd_msg(self):
        return self.motor_torque.rwMotorTorqueOutMsg
