# CMakeLists.txt for Module 03: Tiled Copy Exercises
# Build individual exercises to practice tiled copy concepts

cmake_minimum_required(VERSION 3.18)
project(Module03_Exercises LANGUAGES CUDA CXX)

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

# Exercise 1: Tiled Copy Basics
add_executable(ex01_tiled_copy_basics ex01_tiled_copy_basics.cu)
set_target_properties(ex01_tiled_copy_basics PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 2: Vectorized Loads
add_executable(ex02_vectorized_loads ex02_vectorized_loads.cu)
set_target_properties(ex02_vectorized_loads PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 3: Thread Cooperation
add_executable(ex03_thread_cooperation ex03_thread_cooperation.cu)
set_target_properties(ex03_thread_cooperation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 4: Global to Shared Memory Copy
add_executable(ex04_gmem_to_smem ex04_gmem_to_smem.cu)
set_target_properties(ex04_gmem_to_smem PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 5: Copy Atom and Tiled Copy
add_executable(ex05_copy_atom ex05_copy_atom.cu)
set_target_properties(ex05_copy_atom PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 6: Coalescing Strategies
add_executable(ex06_coalescing_strategies ex06_coalescing_strategies.cu)
set_target_properties(ex06_coalescing_strategies PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 7: Tiled Copy for Matrix Transpose
add_executable(ex07_matrix_transpose_copy ex07_matrix_transpose_copy.cu)
set_target_properties(ex07_matrix_transpose_copy PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 8: Async Copy with cp.async
add_executable(ex08_async_copy ex08_async_copy.cu)
set_target_properties(ex08_async_copy PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "Module 03 Exercises configured successfully")
