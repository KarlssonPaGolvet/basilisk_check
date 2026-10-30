/*
 detailed_battery - equivalent-circuit Li-ion battery pack (Basilisk extension module).

 Drop-in replacement for Basilisk's simpleBattery: it inherits PowerStorageBase,
 so power nodes (solar panels, loads, reaction-wheel power) connect with
 addPowerNodeToModel() exactly as before, and batPowerOutMsg is still written.
 In addition it models the pack electrically and writes packOutMsg
 (BatteryPackMsgPayload): voltage, current, state of charge, resistance, flags.

 Electrical model (per cell, "Thevenin" first-order equivalent circuit):
     V_cell = OCV(SoC) - V1 - I_cell * R0
     dV1/dt = -V1 / (R1*C1) + I_cell / C1
 Pack: nSeries cells in series per string, nParallel strings in parallel.

 Faults (public members, may be changed at any time during a run, e.g. from a
 Basilisk event):
     failedStrings          open-circuit parallel strings (capacity loss)
     shortedCellsPerString  shorted series cells (voltage drop-off)
     resistanceFactor       multiplies R0 and R1 (ageing / resistance increase)
 */

#pragma once

#include <vector>

#include "simulation/power/_GeneralModuleFiles/powerStorageBase.h"
#include "architecture/messaging/messaging.h"
#include "BatteryPackMsgPayload.h"

/*! @brief Equivalent-circuit (OCV + R0 + one RC pair) Li-ion battery pack */
class EcmBattery : public PowerStorageBase
{
  public:
    EcmBattery();
    ~EcmBattery() = default;

    double ocvFromSoc(double soc) const;   //!< cell open-circuit voltage [V] at a state of charge
    double usableCapacityAh() const;      //!< pack capacity [Ah] with current faults applied
    double nominalCellVoltage() const;    //!< mean OCV over the table [V], used for energy bookkeeping
    double getStateOfCharge() const;      //!< current state of charge [-]

    // OCV curve: scalar methods keep the Python interface simple (no vector typemaps needed)
    void clearOcvTable();                          //!< remove all OCV points
    void addOcvPoint(double soc, double ocvCell);  //!< append a point; SoC must increase
    int getOcvPointCount() const;                  //!< number of OCV points

  private:
    void customReset(uint64_t CurrentClock) override;
    void evaluateBatteryModel(PowerStorageStatusMsgPayload* msg) override;
    void customWriteMessages(uint64_t CurrentClock) override;
    int activeStringsNow() const;
    int activeSeriesNow() const;

    std::vector<double> socBreakpoints; //!< [-] SoC points of the OCV table (increasing, 0..1)
    std::vector<double> ocvBreakpoints; //!< [V] cell OCV at each SoC point
    double soc = 0.0;             //!< [-] state of charge of each string (all strings equal)
    double v1Cell = 0.0;          //!< [V] RC-branch voltage per cell
    BatteryPackMsgPayload packMsgBuffer = {};
    bool evaluatedThisStep = false; //!< set by evaluateBatteryModel, cleared after each write

  public:
    // ---- pack layout and cell parameters ----
    int nSeries = 2;                    //!< [-] cells in series per string
    int nParallel = 2;                  //!< [-] strings in parallel
    double cellCapacityAh = 3.0;        //!< [Ah] capacity of one cell
    double cellR0 = 0.030;              //!< [Ohm] ohmic resistance of one cell
    double cellR1 = 0.015;              //!< [Ohm] RC-branch resistance of one cell
    double cellC1 = 2000.0;             //!< [F] RC-branch capacitance of one cell (tau = R1*C1)
    double initialSoc = 0.8;            //!< [-] state of charge at Reset
    double cutoffCellVoltage = 3.0;     //!< [V] BMS cutoff: discharge current is limited to keep cells at or above this
    double maxCellVoltage = 4.20;       //!< [V] CV charge limit: charge current is limited to keep cells at or below this
    double depletedSocThreshold = 1e-6; //!< [-] discharge stops (depleted) when SoC is at or below this

    // ---- faults (can be changed during a run) ----
    int failedStrings = 0;              //!< [-] open-circuit parallel strings
    int shortedCellsPerString = 0;      //!< [-] shorted series cells in every string
    double resistanceFactor = 1.0;      //!< [-] multiplier on R0 and R1

    Message<BatteryPackMsgPayload> packOutMsg; //!< electrical pack telemetry output
};
