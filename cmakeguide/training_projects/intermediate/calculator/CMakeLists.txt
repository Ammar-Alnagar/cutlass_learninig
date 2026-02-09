cmake_minimum_required(VERSION 3.15)
project(AdvancedCalculator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create the calculator library
add_library(calclib
    src/calculator.cpp
)

target_include_directories(calclib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Create the main executable
add_executable(calculator src/main.cpp)
target_link_libraries(calculator PRIVATE calclib)

# Enable testing
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    
    # Create the test executable
    add_executable(test_calculator tests/test_calculator.cpp)
    target_link_libraries(test_calculator PRIVATE calclib)
    
    # Add the test
    add_test(
        NAME test_calculator
        COMMAND test_calculator
    )
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS calculator calclib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)