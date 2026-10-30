# =============================================================================
# Detailed battery extension for Basilisk (built with bsk-sdk).
# Boilerplate copied from bsk-sdk v2.12.1 examples/custom-atm-extension.
# =============================================================================
cmake_minimum_required(VERSION 3.26)
project(detailed_battery LANGUAGES C CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Locate bsk-sdk from the active Python environment
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
    "import bsk_sdk; print(bsk_sdk.cmake_config_dir(), end='')"
  OUTPUT_VARIABLE bsk_sdk_dir
  RESULT_VARIABLE rc
)
if(NOT rc EQUAL 0 OR bsk_sdk_dir STREQUAL "")
  message(FATAL_ERROR "bsk-sdk not found (is it installed in this Python environment?)")
endif()
file(TO_CMAKE_PATH "${bsk_sdk_dir}" bsk_sdk_dir)

set(bsk-sdk_DIR "${bsk_sdk_dir}")
find_package(bsk-sdk CONFIG REQUIRED)

# Where scikit-build-core places wheel outputs
set(EXTENSION_PKG_DIR "${SKBUILD_PLATLIB_DIR}/detailed_battery")

# Exact Basilisk version + ABI check, run at import time before native code loads
bsk_add_extension_compatibility_guard(
  EXTENSION_NAME "detailed_battery"
  OUTPUT_DIR "${EXTENSION_PKG_DIR}"
)

# =============================================================================
# Extension-specific configuration
# =============================================================================

# C++ module: equivalent-circuit battery pack
bsk_add_swig_module(
  TARGET ecmBattery
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/ecmBattery/ecmBattery.i"
  SOURCES   "${CMAKE_CURRENT_SOURCE_DIR}/ecmBattery/ecmBattery.cpp"
  INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/ecmBattery"
    "${CMAKE_CURRENT_SOURCE_DIR}/messages"
  OUTPUT_DIR "${EXTENSION_PKG_DIR}"
)

# Python bindings (Message, Reader, Recorder) for the extension's own message
bsk_generate_messages(
  GENERATE_C_INTERFACE
  OUTPUT_DIR "${EXTENSION_PKG_DIR}/messaging"
  MSG_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/messages/BatteryPackMsgPayload.h"
)
