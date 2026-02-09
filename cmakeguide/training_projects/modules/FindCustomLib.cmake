# Custom Find Module for Hypothetical Library
# FindCustomLib.cmake

# This module defines:
# CUSTOMLIB_FOUND - True if the library is found
# CUSTOMLIB_INCLUDE_DIRS - Include directories
# CUSTOMLIB_LIBRARIES - Libraries to link
# CUSTOMLIB_VERSION - Version of the library

# Search for the library
find_path(CUSTOMLIB_INCLUDE_DIR
    NAMES customlib.h
    PATHS /usr/local/include /opt/local/include
)

find_library(CUSTOMLIB_LIBRARY
    NAMES customlib
    PATHS /usr/local/lib /opt/local/lib
)

# Extract version from header (if available)
if(CUSTOMLIB_INCLUDE_DIR AND EXISTS "${CUSTOMLIB_INCLUDE_DIR}/customlib_version.h")
    file(READ "${CUSTOMLIB_INCLUDE_DIR}/customlib_version.h" CUSTOMLIB_VERSION_CONTENT)
    string(REGEX MATCH "VERSION_MAJOR ([0-9]+)" _ ${CUSTOMLIB_VERSION_CONTENT})
    set(CUSTOMLIB_VERSION_MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "VERSION_MINOR ([0-9]+)" _ ${CUSTOMLIB_VERSION_CONTENT})
    set(CUSTOMLIB_VERSION_MINOR ${CMAKE_MATCH_1})
    set(CUSTOMLIB_VERSION "${CUSTOMLIB_VERSION_MAJOR}.${CUSTOMLIB_VERSION_MINOR}")
endif()

# Handle the QUIETLY and REQUIRED arguments and set CUSTOMLIB_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CustomLib
    REQUIRED_VARS CUSTOMLIB_LIBRARY CUSTOMLIB_INCLUDE_DIR
    VERSION_VAR CUSTOMLIB_VERSION
)

# Create imported target if found
if(CUSTOMLIB_FOUND AND NOT TARGET CustomLib::CustomLib)
    add_library(CustomLib::CustomLib UNKNOWN IMPORTED)
    set_target_properties(CustomLib::CustomLib PROPERTIES
        IMPORTED_LOCATION "${CUSTOMLIB_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUSTOMLIB_INCLUDE_DIR}"
    )
endif()

# Hide internal variables from GUI
mark_as_advanced(CUSTOMLIB_INCLUDE_DIR CUSTOMLIB_LIBRARY)