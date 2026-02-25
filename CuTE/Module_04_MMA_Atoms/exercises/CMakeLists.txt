# CMakeLists.txt for Module 04: MMA Atoms Exercises
# Build individual exercises to practice MMA atom concepts

cmake_minimum_required(VERSION 3.18)
project(Module04_Exercises LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Find CuTe/CUTLASS
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

# Exercise 1: MMA Atom Basics
add_executable(ex01_mma_atom_basics ex01_mma_atom_basics.cu)
set_target_properties(ex01_mma_atom_basics PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 2: Tensor Core Operation Simulation
add_executable(ex02_tensor_core_sim ex02_tensor_core_sim.cu)
set_target_properties(ex02_tensor_core_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 3: Thread to Tensor Core Mapping
add_executable(ex03_thread_tensor_mapping ex03_thread_tensor_mapping.cu)
set_target_properties(ex03_thread_tensor_mapping PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 4: Accumulator Management
add_executable(ex04_accumulator_management ex04_accumulator_management.cu)
set_target_properties(ex04_accumulator_management PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 5: Mixed Precision MMA
add_executable(ex05_mixed_precision_mma ex05_mixed_precision_mma.cu)
set_target_properties(ex05_mixed_precision_mma PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 6: GEMM with MMA Atoms
add_executable(ex06_gemm_with_mma ex06_gemm_with_mma.cu)
set_target_properties(ex06_gemm_with_mma PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 7: MMA Atom Configurations
add_executable(ex07_mma_configurations ex07_mma_configurations.cu)
set_target_properties(ex07_mma_configurations PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 8: Warp-Level Matrix Multiply
add_executable(ex08_warp_level_mma ex08_warp_level_mma.cu)
set_target_properties(ex08_warp_level_mma PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "Module 04 Exercises configured successfully")
