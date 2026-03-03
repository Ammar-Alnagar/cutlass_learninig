# Project 11: INT8 GEMM - Standalone Build
cmake_minimum_required(VERSION 3.20)
project(project_11_int8_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

find_package(CUDAToolkit REQUIRED)

set(CUTLASS_PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../cutlass/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../cutlass/include"
)
find_path(CUTLASS_INCLUDE_DIR NAMES cute/tensor.hpp PATHS ${CUTLASS_PATHS})

if(NOT CUTLASS_INCLUDE_DIR)
    message(FATAL_ERROR "CUTLASS not found.")
endif()

add_executable(project_11_int8_gemm int8_gemm.cu)

target_include_directories(project_11_int8_gemm PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(project_11_int8_gemm PRIVATE CUDA::cudart)
