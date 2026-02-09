# Comprehensive Hands-On CMake and Make Tutorial

## Table of Contents
1. [Introduction and Prerequisites](#introduction-and-prerequisites)
2. [Setting Up Your Environment](#setting-up-your-environment)
3. [Basic CMake Projects](#basic-cmake-projects)
4. [Intermediate CMake Concepts](#intermediate-cmake-concepts)
5. [Advanced CMake Features](#advanced-cmake-features)
6. [Working with Make Directly](#working-with-make-directly)
7. [Integration Examples](#integration-examples)
8. [Troubleshooting and Debugging](#troubleshooting-and-debugging)
9. [Best Practices](#best-practices)
10. [Real-World Projects](#real-world-projects)

## Introduction and Prerequisites

This hands-on tutorial will guide you through practical exercises to master CMake and Make build systems. You'll learn through building real projects, encountering and solving common problems, and understanding the underlying concepts.

### What You'll Learn
- Creating and configuring CMake projects from scratch
- Managing dependencies and libraries
- Cross-platform development techniques
- Advanced Makefile creation and maintenance
- Integration strategies for complex projects
- Debugging and troubleshooting build issues
- Best practices for maintainable build systems

### Learning Approach
Each section includes:
- Theory and concepts
- Step-by-step practical exercises
- Common pitfalls and solutions
- Real-world examples

## Setting Up Your Environment

Before starting, ensure you have the necessary tools installed:

```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build

# On CentOS/RHEL/Fedora
sudo dnf install gcc gcc-c++ make cmake ninja-build

# On macOS with Homebrew
brew install cmake ninja

# Verify installations
cmake --version
gcc --version
make --version
ninja --version  # Alternative to make
```

### Exercise 1: Verify Your Setup
Create a simple test to verify your environment:

```bash
mkdir ~/cmake-test && cd ~/cmake-test
cat > hello.c << EOF
#include <stdio.h>
int main() {
    printf("Environment verified!\\n");
    return 0;
}
EOF

gcc hello.c -o hello
./hello

# Test CMake
cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.10)
project(TestEnv)
add_executable(hello_cmake hello.c)
EOF

mkdir build && cd build
cmake ..
make
./hello_cmake
```

Expected output: `Environment verified!`

## Basic CMake Projects

### Exercise 2: Hello World with CMake

Create your first CMake project:

```bash
mkdir ~/hello-cmake && cd ~/hello-cmake
mkdir src
```

**File: src/main.cpp**
```cpp
#include <iostream>

int main() {
    std::cout << "Hello, CMake World!" << std::endl;
    return 0;
}
```

**File: CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.10)
project(HelloCmake VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(hello src/main.cpp)
```

**Build Instructions:**
```bash
mkdir build && cd build
cmake ..
make
./hello
```

Expected output: `Hello, CMake World!`

### Exercise 3: Adding a Library

Extend the project to include a custom library:

```bash
mkdir ~/hello-cmake-lib && cd ~/hello-cmake-lib
mkdir -p include/mylib src
```

**File: include/mylib/greeter.hpp**
```cpp
#ifndef MYLIB_GREETER_HPP
#define MYLIB_GREETER_HPP

#include <string>

namespace mylib {
    class Greeter {
    public:
        explicit Greeter(const std::string& name);
        std::string greet() const;
        
    private:
        std::string name_;
    };
}

#endif
```

**File: src/greeter.cpp**
```cpp
#include "mylib/greeter.hpp"

namespace mylib {
    Greeter::Greeter(const std::string& name) : name_(name) {}
    
    std::string Greeter::greet() const {
        return "Hello, " + name_ + "!";
    }
}
```

**File: src/main.cpp**
```cpp
#include "mylib/greeter.hpp"
#include <iostream>

int main() {
    mylib::Greeter greeter("CMake Learner");
    std::cout << greeter.greet() << std::endl;
    return 0;
}
```

**File: CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.10)
project(HelloCmakeLib VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create a library
add_library(mylib src/greeter.cpp)

# Specify include directories for the library
target_include_directories(mylib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Create executable
add_executable(hello src/main.cpp)

# Link the library to the executable
target_link_libraries(hello PRIVATE mylib)
```

**Build Instructions:**
```bash
mkdir build && cd build
cmake ..
make
./hello
```

Expected output: `Hello, CMake Learner!`

### Exercise 4: Understanding CMake Variables

Learn about important CMake variables:

**File: cmake-variables-demo/CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.10)
project(VariablesDemo)

# Print important variables
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

# Print system information
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")

add_executable(demo demo.cpp)
```

**File: cmake-variables-demo/demo.cpp**
```cpp
#include <iostream>

int main() {
    std::cout << "Variable demo" << std::endl;
    return 0;
}
```

## Intermediate CMake Concepts

### Exercise 5: Multi-Directory Project with Tests

Create a more complex project structure:

```bash
mkdir ~/calculator-project && cd ~/calculator-project
mkdir -p include/calculator src tests
```

**File: CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.15)
project(Calculator VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add subdirectories
add_subdirectory(src)

# Enable testing
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
include(GNUInstallDirs)
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

**File: src/CMakeLists.txt**
```cmake
# Create static library
add_library(calclib
    calculator.cpp
)

target_include_directories(calclib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Create executable
add_executable(calc main.cpp)
target_link_libraries(calc PRIVATE calclib)

# Installation
install(TARGETS calc calclib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
```

**File: include/calculator/core.hpp**
```cpp
#ifndef CALCULATOR_CORE_HPP
#define CALCULATOR_CORE_HPP

namespace calc {
    double add(double a, double b);
    double subtract(double a, double b);
    double multiply(double a, double b);
    double divide(double a, double b);
}

#endif
```

**File: src/calculator.cpp**
```cpp
#include "calculator/core.hpp"
#include <stdexcept>

namespace calc {
    double add(double a, double b) {
        return a + b;
    }

    double subtract(double a, double b) {
        return a - b;
    }

    double multiply(double a, double b) {
        return a * b;
    }

    double divide(double a, double b) {
        if (b == 0) {
            throw std::domain_error("Division by zero");
        }
        return a / b;
    }
}
```

**File: src/main.cpp**
```cpp
#include "calculator/core.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: calc <num1> <operator> <num2>\n";
        return 1;
    }

    double a = std::stod(argv[1]);
    std::string op = argv[2];
    double b = std::stod(argv[3]);
    double result;

    if (op == "+") {
        result = calc::add(a, b);
    } else if (op == "-") {
        result = calc::subtract(a, b);
    } else if (op == "*") {
        result = calc::multiply(a, b);
    } else if (op == "/") {
        try {
            result = calc::divide(a, b);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Unknown operator: " << op << std::endl;
        return 1;
    }

    std::cout << result << std::endl;
    return 0;
}
```

**File: tests/CMakeLists.txt**
```cmake
add_executable(test_calc test_calculator.cpp)
target_link_libraries(test_calc PRIVATE calclib)

add_test(
    NAME test_calc
    COMMAND test_calc
)
```

**File: tests/test_calculator.cpp**
```cpp
#include "calculator/core.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>

bool test_add() {
    double result = calc::add(2.0, 3.0);
    bool success = std::abs(result - 5.0) < 1e-10;
    std::cout << "add(2, 3) = " << result << " (expected: 5) "
              << (success ? "PASS" : "FAIL") << std::endl;
    return success;
}

bool test_divide_by_zero() {
    bool exception_thrown = false;
    try {
        calc::divide(5.0, 0.0);
    } catch (const std::domain_error&) {
        exception_thrown = true;
    }
    std::cout << "divide by zero throws exception: "
              << (exception_thrown ? "PASS" : "FAIL") << std::endl;
    return exception_thrown;
}

int main() {
    int failed = 0;

    if (!test_add()) failed++;
    if (!test_divide_by_zero()) failed++;

    std::cout << "\nTests run: 2, Failed: " << failed << std::endl;
    return failed > 0 ? 1 : 0;
}
```

**Build and Test Instructions:**
```bash
cd ~/calculator-project
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make
./calc 10 + 5
./test_calc
ctest -V
```

Expected outputs:
- `./calc 10 + 5` → `15`
- `./test_calc` → Shows test results
- `ctest -V` → Verbose test output

### Exercise 6: Using External Dependencies

Learn to work with external libraries:

```bash
mkdir ~/external-deps-demo && cd ~/external-deps-demo
```

**File: CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.15)
project(ExternalDepsDemo VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Threads package (always available)
find_package(Threads REQUIRED)

# Create executable
add_executable(thread_demo main.cpp)

# Link against Threads
target_link_libraries(thread_demo PRIVATE Threads::Threads)

# Option for optional packages
option(USE_BOOST "Use Boost libraries" OFF)
if(USE_BOOST)
    find_package(Boost 1.65.0)
    if(Boost_FOUND)
        target_include_directories(thread_demo PRIVATE ${Boost_INCLUDE_DIRS})
        target_link_libraries(thread_demo PRIVATE ${Boost_LIBRARIES})
        target_compile_definitions(thread_demo PRIVATE BOOST_AVAILABLE)
    else()
        message(WARNING "Boost not found, disabling Boost features")
    endif()
endif()
```

**File: main.cpp**
```cpp
#include <iostream>
#include <thread>
#include <chrono>

void worker_thread(int id) {
    std::cout << "Thread " << id << " starting..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Thread " << id << " finished." << std::endl;
}

int main() {
    std::cout << "Starting thread demo..." << std::endl;
    
    std::thread t1(worker_thread, 1);
    std::thread t2(worker_thread, 2);
    
    t1.join();
    t2.join();
    
    std::cout << "All threads completed." << std::endl;
    
#ifdef BOOST_AVAILABLE
    std::cout << "Boost is available in this build." << std::endl;
#endif
    
    return 0;
}
```

**Build Instructions:**
```bash
cd ~/external-deps-demo
mkdir build && cd build
cmake ..
make
./thread_demo

# Try with Boost (if available)
cmake .. -DUSE_BOOST=ON
make
```

## Advanced CMake Features

### Exercise 7: Cross-Platform Development

Create a project that works across different platforms:

```bash
mkdir ~/cross-platform-demo && cd ~/cross-platform-demo
mkdir -p include platform src
```

**File: CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.15)
project(CrossPlatformDemo VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect platform
if(WIN32)
    set(PLATFORM_NAME "Windows")
    set(PLATFORM_DEFINE "PLATFORM_WINDOWS")
elseif(APPLE)
    set(PLATFORM_NAME "macOS")
    set(PLATFORM_DEFINE "PLATFORM_MACOS")
elseif(UNIX)
    set(PLATFORM_NAME "Linux")
    set(PLATFORM_DEFINE "PLATFORM_LINUX")
else()
    set(PLATFORM_NAME "Unknown")
    set(PLATFORM_DEFINE "PLATFORM_UNKNOWN")
endif()

message(STATUS "Building for platform: ${PLATFORM_NAME}")

# Configure header file
configure_file(
    "${PROJECT_SOURCE_DIR}/platform/config.h.in"
    "${PROJECT_BINARY_DIR}/generated/config.h"
    @ONLY
)

# Create library
add_library(platform_lib
    src/platform.cpp
)

target_include_directories(platform_lib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
)

# Create executable
add_executable(cross_platform_app src/main.cpp)
target_link_libraries(cross_platform_app PRIVATE platform_lib)
```

**File: platform/config.h.in**
```cpp
#pragma once

#define PLATFORM_NAME "@PLATFORM_NAME@"
#define PLATFORM_DEFINE @PLATFORM_DEFINE@

// Version information
#define VERSION_MAJOR @PROJECT_VERSION_MAJOR@
#define VERSION_MINOR @PROJECT_VERSION_MINOR@
#define VERSION_PATCH @PROJECT_VERSION_PATCH@
#define VERSION_STRING "@PROJECT_VERSION@"
```

**File: include/platform_detector.hpp**
```cpp
#ifndef PLATFORM_DETECTOR_HPP
#define PLATFORM_DETECTOR_HPP

#include "config.h"  // Generated by CMake

class PlatformDetector {
public:
    static const char* getPlatformName();
    static const char* getVersionString();
};

#endif
```

**File: src/platform.cpp**
```cpp
#include "platform_detector.hpp"

const char* PlatformDetector::getPlatformName() {
    return PLATFORM_NAME;
}

const char* PlatformDetector::getVersionString() {
    return VERSION_STRING;
}
```

**File: src/main.cpp**
```cpp
#include "platform_detector.hpp"
#include <iostream>

int main() {
    std::cout << "Cross-platform demo" << std::endl;
    std::cout << "Platform: " << PlatformDetector::getPlatformName() << std::endl;
    std::cout << "Version: " << PlatformDetector::getVersionString() << std::endl;
    
#if defined(_WIN32)
    std::cout << "Compiled for Windows" << std::endl;
#elif defined(__APPLE__)
    std::cout << "Compiled for macOS" << std::endl;
#elif defined(__linux__)
    std::cout << "Compiled for Linux" << std::endl;
#else
    std::cout << "Compiled for unknown platform" << std::endl;
#endif
    
    return 0;
}
```

**Build Instructions:**
```bash
cd ~/cross-platform-demo
mkdir build && cd build
cmake ..
make
./cross_platform_app
```

### Exercise 8: Custom CMake Modules

Create reusable CMake modules:

```bash
mkdir ~/custom-module-demo && cd ~/custom-module-demo
mkdir -p cmake modules src
```

**File: cmake/FindCustomLib.cmake**
```cmake
# Custom module to find a hypothetical library
# This would typically be named Find<PackageName>.cmake

# Search for the library
find_path(CUSTOMLIB_INCLUDE_DIR
    NAMES customlib.h
    PATHS /usr/local/include /opt/local/include
)

find_library(CUSTOMLIB_LIBRARY
    NAMES customlib
    PATHS /usr/local/lib /opt/local/lib
)

# Handle the QUIETLY and REQUIRED arguments and set CUSTOMLIB_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CustomLib
    DEFAULT_MSG
    CUSTOMLIB_LIBRARY
    CUSTOMLIB_INCLUDE_DIR
)

# Create imported target if found
if(CUSTOMLIB_FOUND AND NOT TARGET CustomLib::CustomLib)
    add_library(CustomLib::CustomLib UNKNOWN IMPORTED)
    set_target_properties(CustomLib::CustomLib PROPERTIES
        IMPORTED_LOCATION "${CUSTOMLIB_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUSTOMLIB_INCLUDE_DIR}"
    )
endif()

# Hide internal variables from GUI
mark_as_advanced(CUSTOMLIB_INCLUDE_DIR CUSTOMLIB_LIBRARY)
```

**File: modules/MyCustomModule.cmake**
```cmake
# Custom module with helper functions

# Function to create a version header
function(create_version_header target_name)
    set(version_file "${CMAKE_CURRENT_BINARY_DIR}/generated/${target_name}_version.h")
    
    configure_file(
        "${CMAKE_SOURCE_DIR}/templates/version.h.in"
        "${version_file}"
        @ONLY
    )
    
    target_include_directories(${target_name}
        PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/generated
    )
endfunction()

# Macro to add an executable with automatic test
macro(add_executable_with_test exe_name src_file test_src)
    add_executable(${exe_name} ${src_file})
    
    if(BUILD_TESTS)
        add_executable(test_${exe_name} ${test_src})
        add_test(NAME test_${exe_name} COMMAND test_${exe_name})
    endif()
endmacro()
```

**File: templates/version.h.in**
```cpp
#pragma once

#define @TARGET_NAME@_VERSION_MAJOR @PROJECT_VERSION_MAJOR@
#define @TARGET_NAME@_VERSION_MINOR @PROJECT_VERSION_MINOR@
#define @TARGET_NAME@_VERSION_PATCH @PROJECT_VERSION_PATCH@
#define @TARGET_NAME@_VERSION "@PROJECT_VERSION@"
```

**File: CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.15)
project(CustomModuleDemo VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add custom module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules")

# Include custom module
include(MyCustomModule)

# Use custom function
add_executable(version_demo src/version_demo.cpp)
create_version_header(version_demo)

# Use custom macro
add_executable_with_test(simple_demo 
    src/simple_demo.cpp 
    src/test_simple_demo.cpp)
```

**File: src/version_demo.cpp**
```cpp
#include <iostream>
#include "version_demo_version.h"  // Generated by create_version_header

int main() {
    std::cout << "Version demo" << std::endl;
    std::cout << "Version: " << VERSION_DEMO_VERSION << std::endl;
    return 0;
}
```

**File: src/simple_demo.cpp**
```cpp
int add(int a, int b) {
    return a + b;
}
```

**File: src/test_simple_demo.cpp**
```cpp
#include <cassert>

extern int add(int a, int b);

int main() {
    assert(add(2, 3) == 5);
    assert(add(-1, 1) == 0);
    return 0;
}
```

## Working with Make Directly

### Exercise 9: Writing a Basic Makefile

Create a simple Makefile for a C project:

```bash
mkdir ~/basic-make-demo && cd ~/basic-make-demo
mkdir -p src include
```

**File: include/utils.h**
```c
#ifndef UTILS_H
#define UTILS_H

void print_message(const char* msg);
int add_numbers(int a, int b);

#endif
```

**File: src/utils.c**
```c
#include <stdio.h>
#include "utils.h"

void print_message(const char* msg) {
    printf("%s\n", msg);
}

int add_numbers(int a, int b) {
    return a + b;
}
```

**File: src/main.c**
```c
#include "utils.h"

int main() {
    print_message("Hello from Make!");
    int result = add_numbers(5, 7);
    printf("5 + 7 = %d\n", result);
    return 0;
}
```

**File: Makefile**
```makefile
# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2
CPPFLAGS = -Iinclude
LDFLAGS =
LDLIBS =

# Directory settings
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# Target executable
TARGET = $(BINDIR)/app

# Default target
.PHONY: all
all: $(TARGET)

# Create directories
$(OBJDIR):
	@mkdir -p $@

$(BINDIR):
	@mkdir -p $@

# Compile rule
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	@echo "CC $<"
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS) | $(BINDIR)
	@echo "LINK $@"
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(OBJDIR) $(BINDIR)

# Run the application
.PHONY: run
run: $(TARGET)
	@echo "Running application..."
	./$(TARGET)

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all  - Build the application (default)"
	@echo "  clean - Remove build artifacts"
	@echo "  run  - Build and run the application"
	@echo "  help - Show this help message"
```

**Build Instructions:**
```bash
cd ~/basic-make-demo
make
make run
```

### Exercise 10: Advanced Makefile with Automatic Dependencies

Enhance the Makefile with automatic dependency tracking:

**File: advanced-make-demo/Makefile**
```makefile
# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2 -g
CPPFLAGS = -Iinclude
LDFLAGS =
LDLIBS =

# Directory settings
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Find all source files
SOURCES := $(wildcard $(SRCDIR)/*.c)
OBJECTS := $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
DEPENDS := $(OBJECTS:.o=.d)

# Target executable
TARGET = $(BINDIR)/advanced_app

# Enable automatic dependency generation
CFLAGS += -MMD -MP

# Default target
.PHONY: all
all: $(TARGET)

# Create directories
$(OBJDIR) $(BINDIR):
	@mkdir -p $@

# Compile rule with automatic dependency tracking
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	@echo "Compiling $<"
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS) | $(BINDIR)
	@echo "Linking $@"
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Include auto-generated dependencies
-include $(DEPENDS)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(OBJDIR) $(BINDIR)

# Rebuild everything
.PHONY: rebuild
rebuild: clean all

# Run the application
.PHONY: run
run: $(TARGET)
	@echo "Running application..."
	./$(TARGET)

# Install target
PREFIX ?= /usr/local
.PHONY: install
install: $(TARGET)
	@echo "Installing to $(PREFIX)/bin"
	@install -d $(PREFIX)/bin
	@install $(TARGET) $(PREFIX)/bin/

# Print variables for debugging
print-%:
	@echo $* = $($*)

# Help target
.PHONY: help
help:
	@echo "Advanced Makefile targets:"
	@echo "  all      - Build the application (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  rebuild  - Clean and rebuild everything"
	@echo "  run      - Build and run the application"
	@echo "  install  - Install the executable to PREFIX (default: /usr/local)"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  PREFIX   - Installation prefix (default: /usr/local)"
	@echo "  CC       - C compiler (default: gcc)"
	@echo "  CFLAGS   - Compiler flags"
	@echo "  CPPFLAGS - Preprocessor flags"
	@echo ""
	@echo "Debug variables with 'make print-VARIABLE_NAME'"
```

**Build Instructions:**
```bash
cd ~/advanced-make-demo
make
make run
make print-CFLAGS
```

### Exercise 11: Multi-Configuration Makefile

Create a Makefile that supports different build configurations:

**File: multiconfig-make-demo/Makefile**
```makefile
# Configuration-specific settings
ifeq ($(CONFIG),debug)
    CFLAGS += -g -O0 -DDEBUG
    LDFLAGS += -g
    BUILD_DIR_SUFFIX = -debug
else ifeq ($(CONFIG),release)
    CFLAGS += -O3 -DNDEBUG
    BUILD_DIR_SUFFIX = -release
else ifeq ($(CONFIG),profile)
    CFLAGS += -pg -g
    LDFLAGS += -pg
    BUILD_DIR_SUFFIX = -profile
else
    # Default to debug
    CONFIG = debug
    CFLAGS += -g -O0 -DDEBUG
    LDFLAGS += -g
    BUILD_DIR_SUFFIX = -debug
endif

# Set directories based on configuration
OBJDIR = obj$(BUILD_DIR_SUFFIX)
BINDIR = bin$(BUILD_DIR_SUFFIX)

# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -std=c11
CPPFLAGS = -Iinclude
LDFLAGS =
LDLIBS =

# Directory settings
SRCDIR = src

# Find all source files
SOURCES := $(wildcard $(SRCDIR)/*.c)
OBJECTS := $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
DEPENDS := $(OBJECTS:.o=.d)

# Target executable
TARGET = $(BINDIR)/multiconfig_app

# Enable automatic dependency generation
CFLAGS += -MMD -MP

# Default target
.PHONY: all
all: $(TARGET)

# Create directories
$(OBJDIR) $(BINDIR):
	@mkdir -p $@

# Compile rule with automatic dependency tracking
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	@echo "Compiling $< (Config: $(CONFIG))"
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS) | $(BINDIR)
	@echo "Linking $@ (Config: $(CONFIG))"
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Include auto-generated dependencies
-include $(DEPENDS)

# Clean build artifacts for current config
.PHONY: clean
clean:
	@echo "Cleaning build artifacts for $(CONFIG) configuration..."
	@rm -rf $(OBJDIR) $(BINDIR)

# Clean all configurations
.PHONY: clean-all
clean-all:
	@echo "Cleaning all configurations..."
	@rm -rf obj*-debug obj*-release obj*-profile
	@rm -rf bin*-debug bin*-release bin*-profile

# Rebuild current configuration
.PHONY: rebuild
rebuild: clean all

# Run the application
.PHONY: run
run: $(TARGET)
	@echo "Running $(CONFIG) configuration..."
	./$(TARGET)

# Help target
.PHONY: help
help:
	@echo "Multi-configuration Makefile:"
	@echo "  Usage: make [CONFIG=config] [target]"
	@echo "  Config options: debug (default), release, profile"
	@echo ""
	@echo "  Targets:"
	@echo "    all       - Build with current configuration"
	@echo "    clean     - Clean current configuration"
	@echo "    clean-all - Clean all configurations"
	@echo "    rebuild   - Clean and rebuild current config"
	@echo "    run       - Build and run current config"
	@echo "    help      - Show this help message"
	@echo ""
	@echo "  Examples:"
	@echo "    make                    # Build debug config (default)"
	@echo "    make CONFIG=release     # Build release config"
	@echo "    make CONFIG=profile run # Build and run profile config"
```

**Build with different configurations:**
```bash
cd ~/multiconfig-make-demo

# Build debug version (default)
make

# Build release version
make CONFIG=release

# Build and run profile version
make CONFIG=profile run

# Clean all configurations
make clean-all
```

## Integration Examples

### Exercise 12: CMake with Custom Makefile Integration

Integrate CMake with custom Makefiles for specific tasks:

**File: integration-demo/CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.15)
project(IntegrationDemo)

set(CMAKE_CXX_STANDARD 17)

# Add executable
add_executable(app src/main.cpp)

# Add custom target that calls Make
add_custom_target(run-make
    COMMAND ${CMAKE_MAKE_PROGRAM} -f ${CMAKE_CURRENT_SOURCE_DIR}/custom.mk
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running custom Makefile"
)

# Add custom target for documentation
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

# Add custom target for formatting
find_program(CLANG_FORMAT_EXECUTABLE clang-format)
if(CLANG_FORMAT_EXECUTABLE)
    file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.h" "include/*.hpp")
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXECUTABLE} -i ${SOURCES}
        COMMENT "Formatting source files"
    )
endif()
```

**File: integration-demo/custom.mk**
```makefile
# Custom Makefile that can be called from CMake
.PHONY: custom-task validate setup

custom-task:
	@echo "Running custom task from standalone Makefile"
	@echo "Current directory: $(PWD)"

validate:
	@echo "Validating project structure..."
	@test -d src || (echo "Error: src directory missing" && exit 1)
	@test -d include || (echo "Error: include directory missing" && exit 1)
	@echo "Validation passed!"

setup:
	@echo "Setting up development environment..."
	@if [ ! -d build ]; then mkdir build; fi
	@echo "Setup complete!"
```

**File: integration-demo/src/main.cpp**
```cpp
#include <iostream>

int main() {
    std::cout << "Integrated CMake + Make example" << std::endl;
    return 0;
}
```

**File: integration-demo/Doxyfile.in**
```
PROJECT_NAME = "Integration Example"
OUTPUT_DIRECTORY = @CMAKE_CURRENT_BINARY_DIR@/docs
INPUT = @CMAKE_CURRENT_SOURCE_DIR@/src @CMAKE_CURRENT_SOURCE_DIR@/include
GENERATE_HTML = YES
GENERATE_LATEX = NO
```

**Build and Run Instructions:**
```bash
cd ~/integration-demo
mkdir build && cd build
cmake ..
make
make run-make
make validate  # This would run the validation from custom.mk
```

## Troubleshooting and Debugging

### Exercise 13: Debugging Common CMake Issues

Practice identifying and fixing common CMake problems:

**File: troubleshooting-demo/CMakeLists.txt** (with intentional errors)
```cmake
cmake_minimum_required(VERSION 3.10)
project(TroubleShoot VERSION 1.0.0)

# Intentional error: using old-style commands
set(SOURCES src/main.cpp src/helper.cpp)
add_executable(trouble ${SOURCES})

# Another issue: missing include directory
target_link_libraries(trouble helper_lib)  # helper_lib doesn't exist

# Yet another issue: wrong path
include_directories(/nonexistent/path)  # This path doesn't exist
```

**Exercise Instructions:**
1. Try to build this project and identify the errors
2. Fix the issues using modern CMake practices:

**Corrected Version:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(TroubleShoot VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)

add_executable(trouble
    src/main.cpp
    src/helper.cpp
)

# Only link libraries that actually exist
# target_link_libraries(trouble existing_lib)  # Uncomment when you have the lib
```

### Exercise 14: Debugging Makefile Issues

Common Makefile problems and solutions:

**Problem 1: Missing Dependencies**
```makefile
# WRONG - no header dependency
program: main.o utils.o
	gcc -o program main.o utils.o

main.o: main.c  # Missing header dependency!
	gcc -c main.c -o main.o
```

**Solution:**
```makefile
# CORRECT - explicit header dependency
program: main.o utils.o
	gcc -o program main.o utils.o

main.o: main.c common.h  # Explicit header dependency
	gcc -c main.c -o main.o
```

**Problem 2: Parallel Build Issues**
```makefile
# WRONG - missing intermediate target dependency
%.o: %.c
	gcc -c $< -o $@

all: program1 program2  # These might race to create .o files
```

**Solution:**
```makefile
# CORRECT - proper dependencies
objects = main.o utils.o

program1: $(objects)
	gcc -o program1 $(objects)

program2: $(objects)
	gcc -o program2 $(objects)

%.o: %.c
	gcc -c $< -o $@
```

### Exercise 15: Debugging Strategies

Learn effective debugging techniques:

#### CMake Debugging
1. **Use message() statements**:
```cmake
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
```

2. **Check CMake cache**:
```bash
# View all cache variables
cmake -LA

# View advanced variables too
cmake -LAH

# Modify cache variables
cmake -DVAR_NAME=VALUE ..
```

3. **Use GUI tools**:
```bash
# Interactive CMake configuration
ccmake ..
# Or with GUI
cmake-gui ..
```

#### Make Debugging
1. **Use dry-run**:
```bash
# See what commands would be executed without running them
make -n

# Even more verbose
make -n -B  # Force considering targets out of date
```

2. **Increase verbosity**:
```bash
# Show all commands
make V=1

# More detailed output
make --debug=b  # Basic debugging
make --debug=i  # Show implicit rules
make --debug=v  # Verbose (combines most other flags)
```

3. **Print variable values**:
```bash
# Print value of a variable
make print-CFLAGS
```

## Best Practices

### CMake Best Practices

1. **Always specify minimum CMake version**:
```cmake
cmake_minimum_required(VERSION 3.15)  # Use a recent version
```

2. **Use modern CMake** - prefer `target_*` commands:
```cmake
# GOOD
target_include_directories(my_target PRIVATE include/)
target_compile_features(my_target PRIVATE cxx_std_17)
target_link_libraries(my_target PRIVATE other_target)

# BAD
include_directories(include/)
set(CMAKE_CXX_STANDARD 17)
link_libraries(other_lib)
```

3. **Set C++ standard properly**:
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

4. **Organize with subdirectories**:
```cmake
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tools)
```

5. **Use proper installation paths**:
```cmake
include(GNUInstallDirs)
install(TARGETS my_target
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
```

### Make Best Practices

1. **Use pattern rules**:
```makefile
# GOOD
%.o: %.c
	$(CC) -c $< -o $@

# AVOID repetitive rules
# main.o: main.c
# 	$(CC) -c main.c -o main.o
# utils.o: utils.c
# 	$(CC) -c utils.c -o utils.o
```

2. **Define variables**:
```makefile
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2
```

3. **Automatic dependency generation**:
```makefile
CFLAGS += -MMD -MP
-include $(DEPENDS)
```

4. **Phony targets**:
```makefile
.PHONY: clean all install
clean:
	rm -f *.o program
```

5. **Silent builds**:
```makefile
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@echo "Compiling $<"
	$(CC) -c $< -o $@
```

## Real-World Projects

### Exercise 16: Complete CMake Project

Create a complete project that demonstrates all concepts:

```bash
mkdir ~/complete-project && cd ~/complete-project
mkdir -p include/myproject src tests docs cmake
```

**File: CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject VERSION 1.0.0 DESCRIPTION "A complete CMake example project")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add custom module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_DOCS "Build documentation" OFF)
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)

# Find dependencies
find_package(Threads REQUIRED)

# Add library
add_subdirectory(src)

# Add tests if enabled
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Documentation
if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/include)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs)
    set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
    
    doxygen_add_docs(docs
        ${DOXYGEN_INPUT_DIR}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Generate docs"
    )
endif()

# Coverage
if(ENABLE_COVERAGE)
    target_compile_options(myproject_lib PRIVATE --coverage)
    target_link_options(myproject_lib PRIVATE --coverage)
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS myproject_exe myproject_lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Packaging
set(CPACK_PACKAGE_VENDOR "Your Name")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A complete CMake example project")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${PROJECT_SOURCE_DIR}/README.md")

include(CPack)
```

**File: src/CMakeLists.txt**
```cmake
# Create library
add_library(myproject_lib
    myproject.cpp
)

target_include_directories(myproject_lib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(myproject_lib
    PRIVATE
        Threads::Threads
)

# Create executable
add_executable(myproject_exe
    main.cpp
)

target_link_libraries(myproject_exe
    PRIVATE
        myproject_lib
)
```

**File: include/myproject/core.hpp**
```cpp
#ifndef MYPROJECT_CORE_HPP
#define MYPROJECT_CORE_HPP

#include <string>
#include <memory>

namespace myproject {
    class Calculator {
    public:
        Calculator();
        ~Calculator();
        
        double add(double a, double b);
        double multiply(double a, double b);
        
    private:
        class Impl;
        std::unique_ptr<Impl> pimpl;
    };
}

#endif
```

**File: src/myproject.cpp**
```cpp
#include "myproject/core.hpp"
#include <thread>

namespace myproject {
    class Calculator::Impl {
    public:
        // Implementation details
        int thread_id = 0;
    };
    
    Calculator::Calculator() : pimpl(std::make_unique<Impl>()) {
        pimpl->thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    }
    
    Calculator::~Calculator() = default;
    
    double Calculator::add(double a, double b) {
        return a + b;
    }
    
    double Calculator::multiply(double a, double b) {
        return a * b;
    }
}
```

**File: src/main.cpp**
```cpp
#include "myproject/core.hpp"
#include <iostream>

int main() {
    myproject::Calculator calc;
    
    std::cout << "Calculator demo:" << std::endl;
    std::cout << "2 + 3 = " << calc.add(2, 3) << std::endl;
    std::cout << "4 * 5 = " << calc.multiply(4, 5) << std::endl;
    
    return 0;
}
```

**File: tests/CMakeLists.txt**
```cmake
add_executable(test_myproject
    test_calculator.cpp
)

target_link_libraries(test_myproject
    PRIVATE
        myproject_lib
)

add_test(
    NAME test_calculator
    COMMAND test_myproject
)
```

**File: tests/test_calculator.cpp**
```cpp
#include "myproject/core.hpp"
#include <cassert>
#include <iostream>

int main() {
    myproject::Calculator calc;
    
    // Test addition
    assert(calc.add(2, 3) == 5);
    assert(calc.add(-1, 1) == 0);
    std::cout << "Addition tests passed" << std::endl;
    
    // Test multiplication
    assert(calc.multiply(3, 4) == 12);
    assert(calc.multiply(-2, 3) == -6);
    std::cout << "Multiplication tests passed" << std::endl;
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
```

**Build Instructions:**
```bash
cd ~/complete-project
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make
./myproject_exe
make test
```

## Conclusion

This comprehensive hands-on tutorial covered:

1. **Basic CMake usage** - Creating simple projects with executables and libraries
2. **Intermediate concepts** - Multi-directory projects, testing, installation
3. **Advanced topics** - Cross-platform development, external dependencies, custom modules
4. **Pure Make usage** - Complex Makefiles for when CMake isn't appropriate
5. **Integration techniques** - Combining CMake and Make for complex workflows
6. **Troubleshooting** - Identifying and fixing common build issues
7. **Best practices** - Following industry standards for maintainable build systems

### Key Takeaways

- **Start simple**: Begin with basic CMake projects and gradually add complexity
- **Use modern CMake**: Prefer `target_*` commands over global ones
- **Think in terms of targets**: Organize your build around logical targets
- **Handle dependencies properly**: Use proper dependency management in both CMake and Make
- **Test your builds**: Always verify that your build system works correctly
- **Document your build system**: Comment complex build logic for future maintainers

### Next Steps

1. Practice with your own projects - apply these concepts to real code
2. Explore advanced CMake features like cross-compilation and packaging
3. Learn about build system generators like Ninja
4. Study build systems of popular open-source projects
5. Experiment with CI/CD integration for automated builds

Remember that mastering build systems takes practice. Start with simple projects and gradually work your way up to more complex scenarios. The key is to understand the underlying concepts and apply them consistently.