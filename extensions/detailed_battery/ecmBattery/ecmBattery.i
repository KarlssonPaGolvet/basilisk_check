/*
 detailed_battery - SWIG interface for the EcmBattery extension module.
 Pattern follows bsk-sdk's custom-atm-extension example and Basilisk's simpleBattery.i.
 */

%module ecmBattery

%include "architecture/utilities/bskException.swg"
%default_bsk_exception();

%{
    #include "ecmBattery.h"
%}

%pythoncode %{
from Basilisk.architecture.swig_common_model import *
%}
%include "swig_conly_data.i"
%include "std_vector.i"
%include "std_string.i"

// Use %import (not %include) for Basilisk's SysModel so this module's class
// inherits from Basilisk's cSysModel.SysModel and AddModelToTask() accepts it.
%import "sys_model.i"

// Message payloads used by the base class and by this module.
%include "architecture/msgPayloadDefC/PowerNodeUsageMsgPayload.h"
struct PowerNodeUsageMsg_C;
%include "architecture/msgPayloadDefC/PowerStorageStatusMsgPayload.h"
struct PowerStorageStatusMsg_C;
%include "BatteryPackMsgPayload.h"
struct BatteryPackMsg_C;

// Intermediate base class (not exposed by a Basilisk module we can %import): wrap locally.
%include "simulation/power/_GeneralModuleFiles/powerStorageBase.h"
%include "ecmBattery.h"

%pythoncode %{
import sys
protectAllClasses(sys.modules[__name__])
%}
