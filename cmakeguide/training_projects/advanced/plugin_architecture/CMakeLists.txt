cmake_minimum_required(VERSION 3.15)
project(PluginArchitecture)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create the main executable
add_executable(plugin_demo 
    src/main.cpp
)

target_include_directories(plugin_demo PRIVATE include)

# Create shared libraries for plugins
add_library(hello_plugin SHARED plugins/hello_plugin.cpp)
set_target_properties(hello_plugin PROPERTIES PREFIX "")

target_include_directories(hello_plugin PRIVATE include)

add_library(upper_plugin SHARED plugins/uppercase_plugin.cpp)
set_target_properties(upper_plugin PROPERTIES PREFIX "")

target_include_directories(upper_plugin PRIVATE include)

# Installation
include(GNUInstallDirs)
install(TARGETS plugin_demo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install plugins to a specific directory
install(TARGETS hello_plugin upper_plugin
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/plugins
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/plugins
)