"""Detailed battery extension for Basilisk.

Usage:
    from detailed_battery import ecmBattery, messaging
    bat = ecmBattery.EcmBattery()
"""

import sys

from ._bsk_compatibility import check_basilisk_compatibility as _check_basilisk

# Refuse to load native code against a different Basilisk version or ABI.
_check_basilisk()
del _check_basilisk

from Basilisk.architecture import cSysModel as _cSysModel

# The SWIG wrapper does `import cSysModel` at load time; register the short name first.
sys.modules.setdefault("cSysModel", _cSysModel)

# Import generated message bindings BEFORE the module wrapper, so
# EcmBattery.packOutMsg gets its Python methods (recorder(), read(), ...).
from . import messaging  # noqa: E402
from . import ecmBattery  # noqa: E402

__all__ = ["ecmBattery", "messaging"]
