# Utility functions for CMake projects
# Utils.cmake

# Function to create a version header
function(create_version_header target_name)
    set(version_file "${CMAKE_CURRENT_BINARY_DIR}/generated/${target_name}_version.h")
    
    # Create the generated directory
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
    
    # Generate the version header
    file(WRITE "${version_file}" 
"#pragma once

#define ${target_name}_VERSION_MAJOR ${PROJECT_VERSION_MAJOR}
#define ${target_name}_VERSION_MINOR ${PROJECT_VERSION_MINOR}
#define ${target_name}_VERSION_PATCH ${PROJECT_VERSION_PATCH}
#define ${target_name}_VERSION \"${PROJECT_VERSION}\"
")
    
    # Add the generated directory to the target's include directories
    target_include_directories(${target_name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
endfunction()

# Function to add an executable with associated tests
function(add_executable_with_tests exe_name)
    # Parse arguments
    set(single_args "")
    set(multi_args SOURCES TEST_SOURCES)
    cmake_parse_arguments(ARG "" "${single_args}" "${multi_args}" ${ARGN})
    
    # Create the executable
    add_executable(${exe_name} ${ARG_SOURCES})
    
    # If tests are provided, create a test executable
    if(DEFINED ARG_TEST_SOURCES AND BUILD_TESTS)
        enable_testing()
        
        set(test_exe_name test_${exe_name})
        add_executable(${test_exe_name} ${ARG_TEST_SOURCES})
        
        # Link the main executable's library if it exists, or link the sources
        target_sources(${exe_name} INTERFACE ${ARG_SOURCES})
        
        add_test(
            NAME ${test_exe_name}
            COMMAND ${test_exe_name}
        )
    endif()
endfunction()

# Function to set common compiler warnings
function(set_common_warnings target_name)
    target_compile_options(${target_name}
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
            $<$<CXX_COMPILER_ID:MSVC>:/W4>
    )
endfunction()

# Macro to add sanitizers for debugging builds
macro(enable_sanitizers target_name)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_options(${target_name} PRIVATE -fsanitize=address,undefined)
            target_link_options(${target_name} PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
endmacro()