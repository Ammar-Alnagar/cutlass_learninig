cmake_minimum_required(VERSION 3.15)
project(LoggerLibrary)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Threads package (needed for mutex)
find_package(Threads REQUIRED)

# Create the logger library
add_library(loglib
    src/logger.cpp
)

target_include_directories(loglib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(loglib
    PRIVATE
        Threads::Threads
)

# Create the main executable
add_executable(logger_app src/main.cpp)
target_link_libraries(logger_app PRIVATE loglib)

# Enable testing
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    
    # Create the test executable
    add_executable(test_logger tests/test_logger.cpp)
    target_link_libraries(test_logger PRIVATE loglib)
    
    # Add the test
    add_test(
        NAME test_logger
        COMMAND test_logger
    )
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS logger_app loglib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)