# CMakeLists.txt for Module 01: Layout Algebra Exercises
# Build individual exercises to practice layout algebra concepts

cmake_minimum_required(VERSION 3.18)
project(Module01_Exercises LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Find CuTe/CUTLASS - adjust these paths based on your installation
find_path(CUTLASS_INCLUDE_DIR NAMES cutlass/cutlass.h
    PATHS
        ${CMAKE_SOURCE_DIR}/../../cutlass/include
        /usr/local/cutlass/include
        $ENV{CUTLASS_PATH}/include
    DOC "CUTLASS include directory"
)

find_path(CUTE_INCLUDE_DIR NAMES cute/layout.hpp
    PATHS
        ${CMAKE_SOURCE_DIR}/../../cutlass/include
        /usr/local/cutlass/include
        $ENV{CUTLASS_PATH}/include
    DOC "CuTe include directory"
)

if(NOT CUTLASS_INCLUDE_DIR)
    message(FATAL_ERROR "CUTLASS not found. Please set CUTLASS_PATH or install CUTLASS.")
endif()

if(NOT CUTE_INCLUDE_DIR)
    message(FATAL_ERROR "CuTe not found. Please ensure CUTLASS includes CuTe headers.")
endif()

include_directories(
    ${CUTLASS_INCLUDE_DIR}
    ${CUTE_INCLUDE_DIR}
)

# Exercise 1: Basic Layout Creation
add_executable(ex01_basic_layout ex01_basic_layout.cu)
set_target_properties(ex01_basic_layout PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 2: Row-Major vs Column-Major
add_executable(ex02_rowmajor_colmajor ex02_rowmajor_colmajor.cu)
set_target_properties(ex02_rowmajor_colmajor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 3: Custom Strides and Padding
add_executable(ex03_custom_strides ex03_custom_strides.cu)
set_target_properties(ex03_custom_strides PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 4: Layout Composition
add_executable(ex04_layout_composition ex04_layout_composition.cu)
set_target_properties(ex04_layout_composition PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 5: Coordinate to Offset Mapping Challenge
add_executable(ex05_offset_mapping_challenge ex05_offset_mapping_challenge.cu)
set_target_properties(ex05_offset_mapping_challenge PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 6: Hierarchical Layouts
add_executable(ex06_hierarchical_layouts ex06_hierarchical_layouts.cu)
set_target_properties(ex06_hierarchical_layouts PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 7: Layout Transformation Practice
add_executable(ex07_layout_transformation ex07_layout_transformation.cu)
set_target_properties(ex07_layout_transformation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 8: Debug Layouts with cute::print
add_executable(ex08_debug_with_print ex08_debug_with_print.cu)
set_target_properties(ex08_debug_with_print PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "Module 01 Exercises configured successfully")
message(STATUS "CUTLASS include dir: ${CUTLASS_INCLUDE_DIR}")
message(STATUS "CuTe include dir: ${CUTE_INCLUDE_DIR}")
