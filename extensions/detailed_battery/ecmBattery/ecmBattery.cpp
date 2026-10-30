/*
 detailed_battery - equivalent-circuit Li-ion battery pack (Basilisk extension module).
 See ecmBattery.h for the model description.
 */

#include "ecmBattery.h"

#include <algorithm>
#include <cmath>

/*! Constructor: sets a typical NMC Li-ion OCV curve (18650-class cell). */
EcmBattery::EcmBattery()
{
    this->socBreakpoints = {0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00};
    this->ocvBreakpoints = {3.00, 3.30, 3.45, 3.55, 3.62, 3.67, 3.72, 3.79, 3.88, 3.97, 4.07, 4.20};
}

/*! Cell open-circuit voltage by linear interpolation in the OCV table.
 @param soc state of charge 0..1
 @return cell OCV [V]
 */
double EcmBattery::ocvFromSoc(double soc) const
{
    const auto& x = this->socBreakpoints;
    const auto& y = this->ocvBreakpoints;
    if (x.empty()) {
        return 0.0;
    }
    if (soc <= x.front()) {
        return y.front();
    }
    if (soc >= x.back()) {
        return y.back();
    }
    for (size_t i = 1; i < x.size(); i++) {
        if (soc <= x[i]) {
            double f = (soc - x[i - 1]) / (x[i] - x[i - 1]);
            return y[i - 1] + f * (y[i] - y[i - 1]);
        }
    }
    return y.back();
}

/*! Mean of the OCV curve (trapezoidal over SoC); converts Ah to Wh for bookkeeping. */
double EcmBattery::nominalCellVoltage() const
{
    const auto& x = this->socBreakpoints;
    const auto& y = this->ocvBreakpoints;
    if (x.size() < 2) {
        return y.empty() ? 0.0 : y.front();
    }
    double area = 0.0;
    for (size_t i = 1; i < x.size(); i++) {
        area += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
    }
    return area / (x.back() - x.front());
}

double EcmBattery::getStateOfCharge() const
{
    return this->soc;
}

void EcmBattery::clearOcvTable()
{
    this->socBreakpoints.clear();
    this->ocvBreakpoints.clear();
}

void EcmBattery::addOcvPoint(double socPoint, double ocvCell)
{
    this->socBreakpoints.push_back(socPoint);
    this->ocvBreakpoints.push_back(ocvCell);
}

int EcmBattery::getOcvPointCount() const
{
    return static_cast<int>(this->socBreakpoints.size());
}

int EcmBattery::activeStringsNow() const
{
    return std::max(0, this->nParallel - std::max(0, this->failedStrings));
}

int EcmBattery::activeSeriesNow() const
{
    return std::max(0, this->nSeries - std::max(0, this->shortedCellsPerString));
}

/*! Usable pack capacity with the current faults applied [Ah]. */
double EcmBattery::usableCapacityAh() const
{
    return this->activeStringsNow() * this->cellCapacityAh;
}

/*! Reset: validate parameters and initialise the electrical state. */
void EcmBattery::customReset(uint64_t CurrentClock [[maybe_unused]])
{
    if (this->nSeries < 1 || this->nParallel < 1) {
        this->bskLogger.bskError("EcmBattery: nSeries and nParallel must be at least 1.");
    }
    if (this->cellCapacityAh <= 0.0 || this->cellR0 < 0.0 || this->cellR1 < 0.0 || this->cellC1 <= 0.0) {
        this->bskLogger.bskError("EcmBattery: cell capacity and C1 must be positive, R0 and R1 non-negative.");
    }
    if (this->socBreakpoints.size() < 2 || this->socBreakpoints.size() != this->ocvBreakpoints.size()) {
        this->bskLogger.bskError("EcmBattery: socBreakpoints and ocvBreakpoints must have the same length (>= 2).");
    }
    for (size_t i = 1; i < this->socBreakpoints.size(); i++) {
        if (this->socBreakpoints[i] <= this->socBreakpoints[i - 1]) {
            this->bskLogger.bskError("EcmBattery: socBreakpoints must be strictly increasing.");
            break;
        }
    }
    // The table must span the full SoC range, or nominalCellVoltage() mis-scales the Wh capacity.
    if (!this->socBreakpoints.empty() &&
        (std::fabs(this->socBreakpoints.front()) > 1e-9 || std::fabs(this->socBreakpoints.back() - 1.0) > 1e-9)) {
        this->bskLogger.bskError("EcmBattery: the OCV table must start at SoC 0 and end at SoC 1.");
    }
    for (double ocv : this->ocvBreakpoints) {
        if (ocv <= 0.0) {
            this->bskLogger.bskError("EcmBattery: all OCV values must be positive.");
            break;
        }
    }
    if (this->initialSoc < 0.0 || this->initialSoc > 1.0) {
        this->bskLogger.bskError("EcmBattery: initialSoc must be between 0 and 1.");
    }
    this->soc = std::min(1.0, std::max(0.0, this->initialSoc));
    this->v1Cell = 0.0;
    this->packMsgBuffer = this->packOutMsg.zeroMsgPayload;
    this->evaluatedThisStep = false;
}

/*! One time step of the electrical model.
 currentPowerSum (from PowerStorageBase) is the sum of all power nodes:
 > 0 = net generation (charging), < 0 = net consumption (discharging).
 @param msg standard power-storage status message to fill
 */
void EcmBattery::evaluateBatteryModel(PowerStorageStatusMsgPayload* msg)
{
    this->evaluatedThisStep = true;
    const double dt = this->currentTimestep;
    const int nP = this->activeStringsNow();
    const int nS = this->activeSeriesNow();
    BatteryPackMsgPayload& out = this->packMsgBuffer;
    out = this->packOutMsg.zeroMsgPayload;
    out.activeStrings = nP;
    out.activeSeriesCells = nS;

    const double demand = -this->currentPowerSum;  // [W] + = discharge request
    out.powerDemand = demand;

    if (nP == 0 || nS == 0) {
        // Pack fully disconnected: nothing can be delivered or stored.
        out.depleted = 1;
        out.powerLimited = demand > 0.0 ? 1 : 0;
        msg->storageCapacity = 0.0;
        msg->storageLevel = 0.0;
        msg->currentNetPower = 0.0;
        return;
    }

    const double f = std::max(0.0, this->resistanceFactor);
    const double r0Pack = this->cellR0 * f * nS / nP;              // [Ohm]
    const double ocvPack = nS * this->ocvFromSoc(this->soc);        // [V]
    const double v1Pack = nS * this->v1Cell;                        // [V]
    const double e = ocvPack - v1Pack;                              // [V] source voltage behind R0

    // Solve demand = V*I = e*I - R0*I^2 for the pack current I (+ = discharge).
    // BMS limit: the discharge current is capped so the cell voltage never drops below
    // cutoffCellVoltage, i.e. I <= Imax = (e - nS*cutoff) / R0. Charging is not limited.
    const double vMinPack = nS * this->cutoffCellVoltage;  // [V]
    double current = 0.0;
    bool heldAtCutoff = false;
    if (r0Pack <= 1e-12) {
        current = (std::fabs(e) > 1e-12) ? demand / e : 0.0;
        if (current > 0.0 && e < vMinPack) {
            current = 0.0;  // no series resistance: the terminal voltage equals e
            heldAtCutoff = true;
        }
    } else {
        const double iMax = std::max(0.0, (e - vMinPack) / r0Pack);
        const double disc = e * e - 4.0 * r0Pack * demand;
        if (disc < 0.0) {
            current = iMax;
            heldAtCutoff = true;
        } else {
            current = (e - std::sqrt(disc)) / (2.0 * r0Pack);
            if (current > iMax) {
                current = iMax;
                heldAtCutoff = true;
            }
        }
    }
    out.powerLimited = heldAtCutoff ? 1 : 0;
    out.underVoltage = heldAtCutoff ? 1 : 0;

    // Charge and discharge limits.
    if (current < 0.0 && this->soc >= 1.0) {
        current = 0.0;
        out.chargeLimited = 1;
    }
    // Depleted: a discharge is requested and SoC is at or below depletedSocThreshold. A
    // threshold is needed because the BMS limit makes SoC approach 0 asymptotically when
    // OCV(0) equals the cutoff voltage, so SoC never reaches exactly 0.
    if (demand > 0.0 && this->soc <= this->depletedSocThreshold) {
        current = 0.0;
        out.depleted = 1;
        out.powerLimited = 1;
    }

    const double vPack = e - current * r0Pack;
    const double cellCurrent = current / nP;

    // Every packOutMsg field describes the start of the step (before the state update).
    out.packVoltage = vPack;
    out.packCurrent = current;
    out.openCircuitVoltage = ocvPack;
    out.polarizationVoltage = v1Pack;
    out.cellVoltage = vPack / nS;
    out.internalResistance = r0Pack;
    out.powerDelivered = vPack * current;
    out.stateOfCharge = this->soc;
    out.capacityAh = this->usableCapacityAh();
    // Dissipation in R0 plus in R1 (V1^2/R1 per cell, nS*nP cells), not V1*I which also
    // contains the energy flowing into and out of C1.
    const double r1Cell = this->cellR1 * f;
    out.heatLoss = current * current * r0Pack;
    if (r1Cell > 0.0) {
        out.heatLoss += nS * nP * this->v1Cell * this->v1Cell / r1Cell;
    }

    // Integrate the state over the step.
    if (dt > 0.0) {
        this->soc -= cellCurrent * dt / (this->cellCapacityAh * 3600.0);
        this->soc = std::min(1.0, std::max(0.0, this->soc));
        const double tau = r1Cell * this->cellC1;
        if (tau > 1e-12) {
            const double a = std::exp(-dt / tau);
            this->v1Cell = this->v1Cell * a + r1Cell * cellCurrent * (1.0 - a);
        } else {
            this->v1Cell = 0.0;
        }
    }

    // Standard PowerStorageStatus message, now with the EFFECTIVE capacity.
    const double vNom = nS * this->nominalCellVoltage();
    const double capacityWs = this->usableCapacityAh() * 3600.0 * vNom;
    this->storedCharge = this->soc * capacityWs;
    msg->storageCapacity = capacityWs;
    msg->storageLevel = this->storedCharge;
    msg->currentNetPower = -out.powerDelivered;
}

/*! Write the electrical pack message alongside the standard status message.
 If the base class skipped evaluateBatteryModel (unwritten or missing node messages) it
 zeroes batPowerOutMsg, so packOutMsg is zeroed too instead of repeating the last step.
 */
void EcmBattery::customWriteMessages(uint64_t CurrentClock)
{
    if (!this->evaluatedThisStep) {
        this->packMsgBuffer = this->packOutMsg.zeroMsgPayload;
    }
    this->packOutMsg.write(&this->packMsgBuffer, this->moduleID, CurrentClock);
    this->evaluatedThisStep = false;
}
