cmake_minimum_required(VERSION 3.10)
project(MultiFileProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create the utility library
add_library(utils
    src/string_utils.cpp
    src/math_utils.cpp
)

# Set include directories
target_include_directories(utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Create the executable
add_executable(multifile_app src/main.cpp)

# Link the library
target_link_libraries(multifile_app PRIVATE utils)