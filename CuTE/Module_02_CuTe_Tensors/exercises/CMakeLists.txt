# CMakeLists.txt for Module 02: CuTe Tensors Exercises
# Build individual exercises to practice CuTe tensor concepts

cmake_minimum_required(VERSION 3.18)
project(Module02_Exercises LANGUAGES CUDA CXX)

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

# Exercise 1: Tensor Creation from Raw Pointers
add_executable(ex01_tensor_creation ex01_tensor_creation.cu)
set_target_properties(ex01_tensor_creation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 2: Tensor Access Patterns
add_executable(ex02_tensor_access ex02_tensor_access.cu)
set_target_properties(ex02_tensor_access PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 3: Tensor Slicing Operations
add_executable(ex03_tensor_slicing ex03_tensor_slicing.cu)
set_target_properties(ex03_tensor_slicing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 4: Tensor Transpose and View
add_executable(ex04_tensor_transpose ex04_tensor_transpose.cu)
set_target_properties(ex04_tensor_transpose PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 5: Tensor Composition with Layouts
add_executable(ex05_tensor_composition ex05_tensor_composition.cu)
set_target_properties(ex05_tensor_composition PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 6: Multi-dimensional Tensors
add_executable(ex06_multidim_tensors ex06_multidim_tensors.cu)
set_target_properties(ex06_multidim_tensors PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 7: Tensor Memory Spaces
add_executable(ex07_tensor_memory_spaces ex07_tensor_memory_spaces.cu)
set_target_properties(ex07_tensor_memory_spaces PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Exercise 8: Tensor Broadcasting
add_executable(ex08_tensor_broadcasting ex08_tensor_broadcasting.cu)
set_target_properties(ex08_tensor_broadcasting PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "Module 02 Exercises configured successfully")
message(STATUS "CUTLASS include dir: ${CUTLASS_INCLUDE_DIR}")
message(STATUS "CuTe include dir: ${CUTE_INCLUDE_DIR}")
