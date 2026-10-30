"""
Environment: Earth/Sun/Moon gravity, SPICE ephemerides, eclipse (Earth and
Moon as occulters), ephemeris converter, WMM magnetic field, MSIS
atmosphere and drag.

Based on scenarioPowerDemo.py, scenarioSensorThermal.py and scenarioDragDeorbit.py.
"""

from Basilisk.architecture import messaging
from Basilisk.simulation import (dragDynamicEffector, eclipse, ephemerisConverter, magneticFieldWMM,
                                 msisAtmosphere)
from Basilisk.utilities import simIncludeGravBody
from Basilisk.utilities.supportDataTools.dataFetcher import DataFile, get_path

import config

# Bodies converted to EphemerisMsg, in ephemOutMsgs order
EPHEM_BODIES = ["sun", "earth", "moon"]


class Environment:
    """Handles to the environment modules and their messages."""

    def __init__(self, sim, task_name, sc_object):
        # Gravity bodies: Earth is the central body, Sun and Moon act as third bodies
        self.grav_factory = simIncludeGravBody.gravBodyFactory()
        earth = self.grav_factory.createEarth()
        earth.isCentralBody = True
        self.grav_factory.createSun()
        self.grav_factory.createMoon()
        self.mu = earth.mu
        self.grav_factory.addBodiesTo(sc_object)

        # SPICE: planet states relative to Earth so nav outputs are geocentric.
        # epochInMsg=True also creates the epoch message used by WMM and MSIS.
        self.spice = self.grav_factory.createSpiceInterface(time=config.EPOCH, epochInMsg=True)
        self.spice.zeroBase = "Earth"
        sim.AddModelToTask(task_name, self.spice, config.PRIORITY["spice"])
        self.epoch_msg = self.grav_factory.epochMsg
        self.time_msg = self.spice.spiceTimeOutMsg

        body_names = list(self.grav_factory.gravBodies.keys())
        self.planet_msgs = {name: self.spice.planetStateOutMsgs[body_names.index(name)]
                            for name in ("earth", "sun", "moon")}
        self.earth_msg = self.planet_msgs["earth"]
        self.sun_msg = self.planet_msgs["sun"]
        self.moon_msg = self.planet_msgs["moon"]

        # Eclipse: shadow factor seen by the spacecraft (1 = full sun, 0 = umbra)
        self.eclipse = eclipse.Eclipse()
        self.eclipse.ModelTag = "eclipse"
        self.eclipse.addSpacecraftToModel(sc_object.scStateOutMsg)
        self.eclipse.addPlanetToModel(self.earth_msg)
        self.eclipse.addPlanetToModel(self.moon_msg)
        self.eclipse.sunInMsg.subscribeTo(self.sun_msg)
        sim.AddModelToTask(task_name, self.eclipse)
        self.eclipse_msg = self.eclipse.eclipseOutMsgs[0]

        # Ephemeris converter: SPICE states -> EphemerisMsg (position, velocity, attitude, rate)
        self.ephem = ephemerisConverter.EphemerisConverter()
        self.ephem.ModelTag = "ephemConverter"
        for name in EPHEM_BODIES:
            self.ephem.addSpiceInputMsg(self.planet_msgs[name])
        sim.AddModelToTask(task_name, self.ephem, config.PRIORITY["ephemConverter"])
        self.ephem_msgs = {name: self.ephem.ephemOutMsgs[i] for i, name in enumerate(EPHEM_BODIES)}
        self.sun_ephem_msg = self.ephem_msgs["sun"]

        # Magnetic field (WMM2025) at the spacecraft position
        self.mag_field = magneticFieldWMM.MagneticFieldWMM()
        self.mag_field.ModelTag = "WMM"
        self.mag_field.configureWMMFile(str(get_path(DataFile.MagneticFieldData.WMM)))
        self.mag_field.epochInMsg.subscribeTo(self.epoch_msg)
        self.mag_field.planetPosInMsg.subscribeTo(self.earth_msg)
        self.mag_field.addSpacecraftToModel(sc_object.scStateOutMsg)
        sim.AddModelToTask(task_name, self.mag_field)
        self.mag_field_msg = self.mag_field.envOutMsgs[0]

        # Atmosphere (NRLMSISE-00) with constant space weather
        self.atmo = msisAtmosphere.MsisAtmosphere()
        self.atmo.ModelTag = "msisAtmosphere"
        self.atmo.epochInMsg.subscribeTo(self.epoch_msg)
        self.atmo.planetPosInMsg.subscribeTo(self.earth_msg)
        self.atmo.addSpacecraftToModel(sc_object.scStateOutMsg)
        self.sw_msgs = []
        for i in range(23):
            # slots 0-20 are Ap indices, 21-22 are F10.7 values
            value = config.SW_AP if i < 21 else config.SW_F107
            self.sw_msgs.append(messaging.SwDataMsg().write(messaging.SwDataMsgPayload(dataValue=value)))
            self.atmo.swDataInMsgs[i].subscribeTo(self.sw_msgs[-1])
        sim.AddModelToTask(task_name, self.atmo, config.PRIORITY["atmosphere"])
        self.atmo_msg = self.atmo.envOutMsgs[0]

        # Drag on the hub (cannonball model, small CoP offset gives a drag torque)
        self.drag = dragDynamicEffector.DragDynamicEffector()
        self.drag.ModelTag = "dragEffector"
        self.drag.coreParams.projectedArea = config.DRAG_AREA_M2
        self.drag.coreParams.dragCoeff = config.DRAG_COEFF
        self.drag.coreParams.comOffset = config.DRAG_COM_OFFSET_B
        self.drag.atmoDensInMsg.subscribeTo(self.atmo_msg)
        sc_object.addDynamicEffector(self.drag)
        sim.AddModelToTask(task_name, self.drag, config.PRIORITY["atmosphere"])

    def unload(self):
        self.grav_factory.unloadSpiceKernels()
