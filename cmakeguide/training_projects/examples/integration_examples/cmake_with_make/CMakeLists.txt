cmake_minimum_required(VERSION 3.15)
project(CMakeWithMakeIntegration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create a simple executable
add_executable(main_app main.cpp)

# Add a custom target that calls a Makefile
add_custom_target(run-external-make
    COMMAND ${CMAKE_MAKE_PROGRAM} -f ${CMAKE_CURRENT_SOURCE_DIR}/external.mk
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running external Makefile"
)

# Make the main target depend on the external make target
add_dependencies(main_app run-external-make)

# Add a custom command for documentation generation
find_program(DOXYGEN_EXECUTABLE doxygen)
if(DOXYGEN_EXECUTABLE)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        @ONLY
    )

    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Add a custom target for code formatting
find_program(CLANG_FORMAT_EXECUTABLE clang-format)
if(CLANG_FORMAT_EXECUTABLE)
    file(GLOB_RECURSE SOURCES "*.cpp" "*.h" "*.hpp")
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXECUTABLE} -i ${SOURCES}
        COMMENT "Formatting source files"
    )
endif()