# CMakeLists.txt for Module 05: Shared Memory Swizzling Exercises

cmake_minimum_required(VERSION 3.18)
project(Module05_Exercises LANGUAGES CUDA CXX)

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

# Exercise 1: Shared Memory Basics
add_executable(ex01_shared_memory_basics ex01_shared_memory_basics.cu)
set_target_properties(ex01_shared_memory_basics PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 2: Bank Conflict Analysis
add_executable(ex02_bank_conflict_analysis ex02_bank_conflict_analysis.cu)
set_target_properties(ex02_bank_conflict_analysis PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 3: Padding for Conflict Avoidance
add_executable(ex03_padding_conflict_avoidance ex03_padding_conflict_avoidance.cu)
set_target_properties(ex03_padding_conflict_avoidance PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 4: Swizzling Fundamentals
add_executable(ex04_swizzling_fundamentals ex04_swizzling_fundamentals.cu)
set_target_properties(ex04_swizzling_fundamentals PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 5: XOR-Based Swizzling
add_executable(ex05_xor_swizzling ex05_xor_swizzling.cu)
set_target_properties(ex05_xor_swizzling PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 6: Shared Memory Layouts for GEMM
add_executable(ex06_smem_layouts_gemm ex06_smem_layouts_gemm.cu)
set_target_properties(ex06_smem_layouts_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 7: Swizzle Pattern Design
add_executable(ex07_swizzle_pattern_design ex07_swizzle_pattern_design.cu)
set_target_properties(ex07_swizzle_pattern_design PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 8: Bank Conflict-Free Matrix Transpose
add_executable(ex08_conflict_free_transpose ex08_conflict_free_transpose.cu)
set_target_properties(ex08_conflict_free_transpose PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "Module 05 Exercises configured successfully")
