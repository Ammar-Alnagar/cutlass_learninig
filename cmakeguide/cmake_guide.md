# Comprehensive CMake Guide

## Table of Contents
1. [Introduction to CMake](#introduction-to-cmake)
2. [Basic Project Setup](#basic-project-setup)
3. [Variables and Properties](#variables-and-properties)
4. [Targets and Dependencies](#targets-and-dependencies)
5. [Finding Packages](#finding-packages)
6. [Conditional Logic](#conditional-logic)
7. [Cross-Platform Development](#cross-platform-development)
8. [Best Practices](#best-practices)
9. [Advanced Topics](#advanced-topics)

## Introduction to CMake

CMake is a cross-platform build system generator that creates native build files for your platform (Makefiles on Unix, Visual Studio projects on Windows, etc.). It's not a build system itself but rather a tool that generates build systems.

### Why Use CMake?
- Cross-platform compatibility
- Modern C++ support
- Dependency management
- Integration with IDEs
- Extensive community support

## Basic Project Setup

### Minimum CMake Project

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyProject VERSION 1.0.0 LANGUAGES CXX C)

add_executable(my_app src/main.cpp)
```

### Understanding the Components

1. **cmake_minimum_required**: Sets the minimum CMake version required
2. **project**: Defines the project name, version, and languages
3. **add_executable**: Creates an executable target

### Common Project Structure

```
my_project/
├── CMakeLists.txt          # Root CMake file
├── src/                    # Source files
│   ├── main.cpp
│   └── lib/
├── include/                # Header files
│   └── myproject/
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── external/               # Third-party dependencies
```

### CMakeLists.txt for Structured Project

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create library
add_library(mylib
    src/lib/mylib.cpp
)

target_include_directories(mylib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Create executable
add_executable(my_app
    src/main.cpp
)

target_link_libraries(my_app PRIVATE mylib)

# Install targets
install(TARGETS my_app DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
```

## Variables and Properties

### Setting Variables

```cmake
# Set a variable
set(MY_VAR "value")

# Set a cache variable (visible in GUI)
set(MY_CACHE_VAR "value" CACHE STRING "Description")

# Unset a variable
unset(MY_VAR)
```

### Common Variables

```cmake
# Project information
${PROJECT_NAME}
${PROJECT_VERSION}
${PROJECT_SOURCE_DIR}
${PROJECT_BINARY_DIR}

# System information
${CMAKE_SYSTEM_NAME}
${CMAKE_SYSTEM_PROCESSOR}
${CMAKE_HOST_SYSTEM_NAME}

# Compiler information
${CMAKE_C_COMPILER}
${CMAKE_CXX_COMPILER}
${CMAKE_BUILD_TYPE}

# Useful paths
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_BINARY_DIR}
```

### Properties

```cmake
# Set properties on targets
set_target_properties(my_target PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Get properties
get_property(VAL TARGET my_target PROPERTY CXX_STANDARD)
```

## Targets and Dependencies

### Executable Target

```cmake
add_executable(my_app
    src/main.cpp
    src/utils.cpp
)
```

### Library Targets

```cmake
# Static library
add_library(mystatic STATIC
    src/lib.cpp
)

# Shared library
add_library(myshared SHARED
    src/lib.cpp
)

# Interface library (header-only)
add_library(myinterface INTERFACE)
target_include_directories(myinterface INTERFACE include/)
```

### Linking Libraries

```cmake
# Link to target
target_link_libraries(my_app PRIVATE mylib)
target_link_libraries(my_app PUBLIC mylib)
target_link_libraries(my_app INTERFACE mylib)

# Private: Only my_app needs mylib
# Public: Both my_app and targets that link to my_app need mylib
# Interface: Only targets that link to my_app need mylib
```

### Include Directories

```cmake
target_include_directories(my_target
    PRIVATE   include/internal
    PUBLIC    include/public
    INTERFACE include/interface
)
```

## Finding Packages

### Using find_package

```cmake
# Find a package
find_package(Boost REQUIRED COMPONENTS system filesystem)

# Check if found
if(Boost_FOUND)
    target_link_libraries(my_app PRIVATE ${Boost_LIBRARIES})
    target_include_directories(my_app PRIVATE ${Boost_INCLUDE_DIRS})
endif()

# Modern way using imported targets
find_package(Threads REQUIRED)
target_link_libraries(my_app PRIVATE Threads::Threads)
```

### Creating Custom Find Modules

Create `cmake/FindMyPackage.cmake`:

```cmake
find_path(MYPACKAGE_INCLUDE_DIR
    NAMES mypackage.h
    PATHS /usr/local/include
          /opt/mypackage/include
)

find_library(MYPACKAGE_LIBRARY
    NAMES mypackage
    PATHS /usr/local/lib
          /opt/mypackage/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    MyPackage DEFAULT_MSG
    MYPACKAGE_LIBRARY MYPACKAGE_INCLUDE_DIR
)

if(MYPACKAGE_FOUND)
    set(MYPACKAGE_LIBRARIES ${MYPACKAGE_LIBRARY})
    set(MYPACKAGE_INCLUDE_DIRS ${MYPACKAGE_INCLUDE_DIR})
endif()
```

## Conditional Logic

### Platform-Specific Code

```cmake
if(WIN32)
    target_compile_definitions(my_app PRIVATE WIN32_PLATFORM)
elseif(APPLE)
    target_compile_definitions(my_app PRIVATE APPLE_PLATFORM)
elseif(UNIX)
    target_compile_definitions(my_app PRIVATE UNIX_PLATFORM)
endif()
```

### Compiler-Specific Options

```cmake
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(my_app PRIVATE -Wall -Wextra)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(my_app PRIVATE -Wall -Wextra)
elseif(MSVC)
    target_compile_options(my_app PRIVATE /W4)
endif()
```

### Build Type Conditions

```cmake
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Or use target-specific options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(my_app PRIVATE DEBUG_MODE)
endif()
```

## Cross-Platform Development

### Toolchain Files

Create `toolchain-arm.cmake`:

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)

set(CMAKE_FIND_ROOT_PATH /usr/arm-linux-gnueabihf)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

Use with: `cmake -DCMAKE_TOOLCHAIN_FILE=toolchain-arm.cmake ..`

### Architecture Detection

```cmake
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCHITECTURE "x64")
else()
    set(ARCHITECTURE "x86")
endif()
```

## Best Practices

### Modern CMake

```cmake
# Use modern CMake (3.0+)
cmake_minimum_required(VERSION 3.15)

# Use target_* commands instead of global ones
target_compile_features(mylib PUBLIC cxx_std_17)
target_include_directories(mylib PUBLIC include/)
target_compile_definitions(mylib PRIVATE MYLIB_BUILDING)

# Use imported targets when available
find_package(Threads REQUIRED)
target_link_libraries(mylib PUBLIC Threads::Threads)
```

### Version Management

```cmake
project(MyProject VERSION 1.0.0)

# Semantic versioning
set(PROJECT_VERSION_MAJOR 1)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)

# Generate version header
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/version.h.in"
    "${PROJECT_BINARY_DIR}/generated/version.h"
)
```

### Testing Support

```cmake
option(BUILD_TESTING "Build tests" ON)

if(BUILD_TESTING)
    enable_testing()
    
    add_executable(test_mylib test/test_mylib.cpp)
    target_link_libraries(test_mylib PRIVATE mylib)
    
    add_test(NAME test_mylib COMMAND test_mylib)
endif()
```

### Installation

```cmake
include(GNUInstallDirs)

install(TARGETS my_app
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets for other projects
install(EXPORT my_targets
    FILE MyConfig.cmake
    NAMESPACE My::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/My
)
```

## Advanced Topics

### Custom Commands and Targets

```cmake
# Custom command
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated.cpp
    COMMAND code_generator --input input.def --output ${CMAKE_CURRENT_BINARY_DIR}/generated.cpp
    DEPENDS input.def
    COMMENT "Generating source code..."
)

# Custom target
add_custom_target(generate_code ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated.cpp
)
```

### Package Configuration

```cmake
# Create config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MyConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/MyConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/My
)

# Install config
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/MyConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/My
)
```

### Subdirectories

```cmake
# Include subdirectories
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tools)

# Pass variables to subdirectories
set(MY_VARIABLE "value" PARENT_SCOPE)
```

### Generator Expressions

```cmake
# Conditional compilation flags
target_compile_options(my_app PRIVATE
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O3>
    $<$<BOOL:${ENABLE_SANITIZER}>:-fsanitize=address>
)

# Conditional linking
target_link_libraries(my_app PRIVATE
    $<$<PLATFORM_ID:Linux>:pthread>
    $<$<PLATFORM_ID:Windows>:ws2_32>
)
```

## Common Pitfalls to Avoid

1. Use `target_*` commands instead of global ones
2. Don't hardcode paths - use variables and find modules
3. Always specify visibility (PRIVATE/PUBLIC/INTERFACE) for target commands
4. Use `CMAKE_CURRENT_SOURCE_DIR` and `CMAKE_CURRENT_BINARY_DIR` instead of relative paths
5. Don't use `add_definitions()` - use `target_compile_definitions()` instead
6. Be careful with `CMAKE_BUILD_TYPE` - it's ignored in multi-config generators
7. Always check if packages are found before using them
8. Use semantic versioning for your projects

## Useful Resources

- [Official CMake Documentation](https://cmake.org/documentation/)
- [Effective CMake](https://github.com/scivision/effective-cmake)
- [Modern CMake](https://cliutils.gitlab.io/modern-cmake/)
- [CMake Best Practices](https://pabloariasal.github.io/2018/02/19/its-time-to-do-cmake-right/)