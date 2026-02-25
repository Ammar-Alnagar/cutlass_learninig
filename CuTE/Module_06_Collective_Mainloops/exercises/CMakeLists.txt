# CMakeLists.txt for Module 06: Collective Mainloops Exercises

cmake_minimum_required(VERSION 3.18)
project(Module06_Exercises LANGUAGES CUDA CXX)

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

# Exercise 1: Producer-Consumer Pipeline
add_executable(ex01_producer_consumer ex01_producer_consumer.cu)
set_target_properties(ex01_producer_consumer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 2: Collective Copy Operations
add_executable(ex02_collective_copy ex02_collective_copy.cu)
set_target_properties(ex02_collective_copy PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 3: Multi-Stage Pipeline
add_executable(ex03_multi_stage_pipeline ex03_multi_stage_pipeline.cu)
set_target_properties(ex03_multi_stage_pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 4: Thread Block Cooperation
add_executable(ex04_thread_block_cooperation ex04_thread_block_cooperation.cu)
set_target_properties(ex04_thread_block_cooperation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 5: Mainloop Scheduling
add_executable(ex05_mainloop_scheduling ex05_mainloop_scheduling.cu)
set_target_properties(ex05_mainloop_scheduling PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 6: Double Buffering
add_executable(ex06_double_buffering ex06_double_buffering.cu)
set_target_properties(ex06_double_buffering PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 7: Complete GEMM Mainloop
add_executable(ex07_complete_gemm_mainloop ex07_complete_gemm_mainloop.cu)
set_target_properties(ex07_complete_gemm_mainloop PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 8: Performance Optimization
add_executable(ex08_performance_optimization ex08_performance_optimization.cu)
set_target_properties(ex08_performance_optimization PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "Module 06 Exercises configured successfully")
