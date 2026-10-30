# detailed_battery — equivalent-circuit battery for Basilisk (C++ extension)

A C++ Basilisk module, `EcmBattery`, built as a **Basilisk Extension with bsk-sdk 2.12.1**,
against the pip-installed `bsk==2.12.1`. No Basilisk source build is needed.

It is a drop-in replacement for `simpleBattery`: it inherits `PowerStorageBase`, so power
nodes (solar panel, loads, wheel power) connect with `addPowerNodeToModel()` as before,
and `batPowerOutMsg` is still written. In addition it writes **`packOutMsg`**
(`BatteryPackMsgPayload`) with pack voltage, current, state of charge, open-circuit
voltage, RC (polarisation) voltage, cell voltage, internal resistance, usable capacity,
demanded/delivered power, heat loss and flags (power-limited, under-voltage,
charge-limited, depleted). Every `packOutMsg` field describes the start of the time step.

## Model

Per cell, a first-order Thevenin equivalent circuit:

    V_cell = OCV(SoC) − V1 − I_cell·R0          dV1/dt = −V1/(R1·C1) + I_cell/C1

The pack is `nSeries` cells per string × `nParallel` strings. Each step the current is
solved from the requested power, P = V·I (quadratic), so voltage sags with load.

A BMS current limit keeps every cell at or above `cutoffCellVoltage`: the discharge current
is capped at `Imax = (E − nSeries·cutoffCellVoltage) / R0` (zero if that is negative), where
E is the open-circuit voltage minus the RC voltage. If the request needs more than `Imax`
(or cannot be met at all), the pack delivers `Imax` and sets both `powerLimited` and
`underVoltage` (= "held at the cutoff limit"). Charging is not limited.
Charging stops at 100 % (`chargeLimited`). Discharging stops when depleted:
`depleted` = SoC at or below `depletedSocThreshold` (default 1e-6) while a discharge is
requested. A threshold is used because, with the default OCV table (OCV at 0 % equals the
3.0 V cutoff), the BMS limit makes SoC approach 0 asymptotically.

`heatLoss` is the dissipation in the resistors: `I²·R0 + nS·nP·V1²/R1` (always ≥ 0).

Defaults: 2S2P, 3.0 Ah cells (≈ 43 Wh), R0 = 30 mΩ, R1 = 15 mΩ, C1 = 2000 F (τ = 30 s),
NMC-type OCV curve 3.0–4.2 V, start at 80 % SoC.

## Faults (change at any time, e.g. from a Basilisk event)

| Member | Meaning | Effect |
|---|---|---|
| `failedStrings` | open-circuit parallel strings | capacity and current capacity drop; higher cell current |
| `shortedCellsPerString` | shorted series cells | pack voltage drop-off |
| `resistanceFactor` | multiplier on R0 and R1 | larger voltage sag, more heat (ageing) |

`batPowerOutMsg.storageCapacity` reports the **effective** capacity after faults, so a
capacity fault is visible in standard telemetry (unlike `simpleBattery`).

OCV curve: `clearOcvTable()`, `addOcvPoint(soc, ocvCell)`. SoC must be strictly increasing,
start at 0 and end at 1, and every OCV value must be positive (checked at Reset).

## Known limitations

- **Standard message energy balance.** `batPowerOutMsg.storageLevel` is
  SoC × usable Ah × mean OCV, while `currentNetPower` is the terminal power. They do not
  balance exactly: the change in `storageLevel` over a step differs from
  `currentNetPower·dt` by the internal losses and the OCV-vs-mean-voltage difference.
  `storageLevel / storageCapacity` is still the exact SoC.
- **No CC/CV charging.** Charging has no voltage limit (constant-voltage phase) and no
  maximum charge current: the pack accepts the full charge power until SoC reaches 1.0,
  then the current drops to zero (`chargeLimited`). Charge that would overshoot 1.0 in the
  last step is clipped.
- **Keep the source messages referenced.** The battery reads the power-node messages by
  reference and does not keep their Python objects alive. Keep every module whose
  `nodePowerOutMsg` is connected (and the battery itself) referenced for the whole run.

## Build and install (Windows, PowerShell, `bsk-env` active, from the repo root)

Requires a C++ compiler (Visual Studio 2019 or 2022 with the C++ workload).

```powershell
pip install bsk-sdk==2.12.1 scikit-build-core build "swig==4.4.1" cmake ninja numpy
$env:CMAKE_GENERATOR = "Visual Studio 16 2019"   # compiler is not on PATH; let CMake use VS directly
python -m build --wheel --no-isolation -o extension-dist "-Cbuild-dir=extension-build" extensions\detailed_battery
python -m pip install (Get-ChildItem extension-dist\detailed_battery-*.whl).FullName
python -c "from detailed_battery import ecmBattery; print(ecmBattery.EcmBattery())"
```

Rebuild after a C++ change: run the `build` and `pip install` lines again
(add `--force-reinstall` to the install line).

`extension-build/` and `extension-dist/` are listed in the repo `.gitignore`.

## Check

```powershell
python extensions\detailed_battery\tests\check_ecm_battery.py
```

Runs three Basilisk simulations and prints PASS/FAIL per check:

- **Faults** (5 W load): resistance ×2 (30 min), a failed string (60 min), a shorted cell
  (90 min) and a 300 W overload (120 min). Checks coulomb counting, the BMS limit during
  the overload, RC recovery after it and `heatLoss ≥ 0`. Saves `results\ecm_battery_check.png`.
- **Charging** (+20 W source from SoC 0.98): negative current, `chargeLimited`, SoC ≤ 1.
- **Depletion** (5 W load from SoC 0.01): `depleted`, current to zero, SoC ≥ 0.

## Using it in the mini prototype (power.py)

```python
from detailed_battery import ecmBattery

battery = ecmBattery.EcmBattery()
battery.ModelTag = "battery"
battery.nSeries, battery.nParallel = 2, 2
battery.cellCapacityAh = 3.0              # 2S2P x 3 Ah ≈ 43 Wh
battery.initialSoc = 0.8                  # replaces storedCharge_Init / storageCapacity
for node in (solarPanel, busLoad, *rwPowerNodes):
    battery.addPowerNodeToModel(node.nodePowerOutMsg)
sim.AddModelToTask(taskName, battery)     # same run order as the old battery

packLog = battery.packOutMsg.recorder(recordInterval)   # voltage, current, SoC, ...
```

Existing recorders on `battery.batPowerOutMsg` keep working. Keep `battery` referenced
for the whole run (return it from `build()`), like every other module.
