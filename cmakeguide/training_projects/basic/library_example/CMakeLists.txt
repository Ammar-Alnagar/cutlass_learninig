cmake_minimum_required(VERSION 3.10)
project(MathLibrary)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create the library
add_library(mathlib 
    src/mathlib.cpp
)

# Set include directories for the library
target_include_directories(mathlib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Create the executable
add_executable(calculator src/main.cpp)

# Link the library to the executable
target_link_libraries(calculator PRIVATE mathlib)