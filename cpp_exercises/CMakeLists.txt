cmake_minimum_required(VERSION 3.20)
project(cpp_exercises LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture - adjust for your GPU
set(CMAKE_CUDA_ARCHITECTURES 89)

# Sanitizer flags for DEBUG exercises
set(SANITIZER_FLAGS "-fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${SANITIZER_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${SANITIZER_FLAGS}")

# Find all exercise files
file(GLOB_RECURSE PHASE0_CPPS "phase0_foundations/*.cpp")
file(GLOB_RECURSE PHASE1_CPPS "phase1_ownership/*.cpp")
file(GLOB_RECURSE PHASE2_CPPS "phase2_undefined_behavior/*.cpp")
file(GLOB_RECURSE PHASE3_CPPS "phase3_templates/*.cpp")
file(GLOB_RECURSE PHASE4_CPPS "phase4_atomics_memory_model/*.cpp")
file(GLOB_RECURSE PHASE5_CPPS "phase5_concurrency/*.cpp")
file(GLOB_RECURSE PHASE6_CPPS "phase6_modern_patterns/*.cpp")
file(GLOB_RECURSE PHASE7_CPPS "phase7_interview_sim/*.cpp")

file(GLOB_RECURSE PHASE0_CUS "phase0_foundations/*.cu")
file(GLOB_RECURSE PHASE1_CUS "phase1_ownership/*.cu")
file(GLOB_RECURSE PHASE5_CUS "phase5_concurrency/*.cu")
file(GLOB_RECURSE PHASE6_CUS "phase6_modern_patterns/*.cu")
file(GLOB_RECURSE PHASE7_CUS "phase7_interview_sim/*.cu")

# Helper function to create exercise targets
function(add_exercise_target src_file)
    get_filename_component(base_name ${src_file} NAME_WE)
    get_filename_component(src_dir ${src_file} DIRECTORY)
    get_filename_component(phase_dir ${src_dir} DIRECTORY)
    get_filename_component(phase_name ${phase_dir} NAME)
    
    set(target_name "${phase_name}_${base_name}")
    
    if(src_file MATCHES "\\.cu$")
        add_executable(${target_name} ${src_file})
        set_target_properties(${target_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    else()
        add_executable(${target_name} ${src_file})
    endif()
    
    target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR})
    
    # Link pthread for threading exercises
    find_package(Threads REQUIRED)
    target_link_libraries(${target_name} PRIVATE Threads::Threads)
endfunction()

# Create targets for all exercise files
foreach(src ${PHASE0_CPPS} ${PHASE1_CPPS} ${PHASE2_CPPS} ${PHASE3_CPPS} 
              ${PHASE4_CPPS} ${PHASE5_CPPS} ${PHASE6_CPPS} ${PHASE7_CPPS})
    add_exercise_target(${src})
endforeach()

foreach(src ${PHASE0_CUS} ${PHASE1_CUS} ${PHASE5_CUS} ${PHASE6_CUS} ${PHASE7_CUS})
    add_exercise_target(${src})
endforeach()

# Custom targets for building all exercises
add_custom_target(build_all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    COMMENT "Building all exercises"
)

# Helper target to list all available exercises
add_custom_target(list_exercises
    COMMAND ${CMAKE_COMMAND} -E echo "Available exercise targets:"
    COMMAND ${CMAKE_COMMAND} -E echo "  phase0: lvalue/rvalue foundations"
    COMMAND ${CMAKE_COMMAND} -E echo "  phase1: ownership and move semantics"
    COMMAND ${CMAKE_COMMAND} -E echo "  phase2: undefined behavior (DEBUG format)"
    COMMAND ${CMAKE_COMMAND} -E echo "  phase3: templates and type traits"
    COMMAND ${CMAKE_COMMAND} -E echo "  phase4: atomics and memory model"
    COMMAND ${CMAKE_COMMAND} -E echo "  phase5: concurrency patterns"
    COMMAND ${CMAKE_COMMAND} -E echo "  phase6: modern C++ patterns"
    COMMAND ${CMAKE_COMMAND} -E echo "  phase7: interview simulations"
    COMMENT "Listing exercise categories"
)
