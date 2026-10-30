/*
 detailed_battery - battery pack electrical telemetry message (Basilisk extension).

 Written by the EcmBattery module every time step. Every field describes the start of
 the step. Sign convention for current:
 packCurrent > 0 means the pack is DISCHARGING, < 0 means it is charging.
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double packVoltage;          //!< [V]  terminal voltage of the pack
    double packCurrent;          //!< [A]  pack current, + = discharge, - = charge
    double stateOfCharge;        //!< [-]  0..1, of the currently usable capacity
    double openCircuitVoltage;   //!< [V]  pack open-circuit voltage (no load)
    double polarizationVoltage;  //!< [V]  pack RC-branch voltage (slow recovery)
    double cellVoltage;          //!< [V]  average terminal voltage per series cell
    double internalResistance;   //!< [Ohm] pack ohmic resistance R0 (incl. faults)
    double capacityAh;           //!< [Ah] currently usable pack capacity (incl. faults)
    double powerDemand;          //!< [W]  power requested from the pack (+ = discharge)
    double powerDelivered;       //!< [W]  power actually delivered at the terminals
    double heatLoss;             //!< [W]  dissipation in R0 and R1 (I^2*R0 + V1^2/R1), always >= 0
    int32_t activeStrings;       //!< [-]  parallel strings still connected
    int32_t activeSeriesCells;   //!< [-]  series cells per string still active
    int32_t powerLimited;        //!< [-]  1 = discharge current capped by the BMS limit (less than demanded)
    int32_t underVoltage;        //!< [-]  1 = held at the cutoff limit (cell voltage = cutoffCellVoltage)
    int32_t chargeLimited;       //!< [-]  1 = charge current limited: held at maxCellVoltage (CV) or pack full
    int32_t depleted;            //!< [-]  1 = SoC at or below depletedSocThreshold, discharge stopped
} BatteryPackMsgPayload;

#ifdef __cplusplus
} // extern "C"
#endif
