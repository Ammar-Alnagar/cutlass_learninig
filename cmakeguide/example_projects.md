# CMake and Make Example Projects

This document provides complete, runnable example projects that demonstrate CMake and Make concepts from the hands-on tutorial.

## Project 1: Basic CMake Project

### Directory Structure
```
basic_cmake_project/
├── CMakeLists.txt
├── include/
│   └── hello.hpp
├── src/
│   ├── hello.cpp
│   └── main.cpp
└── README.md
```

### Files

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(BasicHello VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create library
add_library(hello_lib
    src/hello.cpp
)

target_include_directories(hello_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Create executable
add_executable(basic_hello
    src/main.cpp
)

target_link_libraries(basic_hello
    PRIVATE
        hello_lib
)
```

**include/hello.hpp:**
```cpp
#ifndef HELLO_HPP
#define HELLO_HPP

#include <string>

class Greeter {
public:
    explicit Greeter(const std::string& name);
    std::string greet() const;
    void setName(const std::string& name);
    std::string getName() const;

private:
    std::string name_;
};

#endif
```

**src/hello.cpp:**
```cpp
#include "hello.hpp"

Greeter::Greeter(const std::string& name) : name_(name) {}

std::string Greeter::greet() const {
    return "Hello, " + name_ + "!";
}

void Greeter::setName(const std::string& name) {
    name_ = name;
}

std::string Greeter::getName() const {
    return name_;
}
```

**src/main.cpp:**
```cpp
#include "hello.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string name = (argc > 1) ? argv[1] : "World";
    
    Greeter greeter(name);
    std::cout << greeter.greet() << std::endl;
    
    return 0;
}
```

**README.md:**
```markdown
# Basic CMake Project

This is a simple CMake project to demonstrate basic concepts.

## Building

```bash
mkdir build
cd build
cmake ..
make
./basic_hello "Your Name"
```
```

## Project 2: Advanced CMake with Tests

### Directory Structure
```
advanced_cmake_project/
├── CMakeLists.txt
├── include/
│   └── calculator/
│       └── math.hpp
├── src/
│   ├── CMakeLists.txt
│   ├── calculator.cpp
│   └── main.cpp
├── tests/
│   ├── CMakeLists.txt
│   └── test_calculator.cpp
├── cmake/
│   └── version.h.in
└── README.md
```

### Files

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(Calculator VERSION 1.0.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate version header
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/version.h.in"
    "${PROJECT_BINARY_DIR}/generated/version.h"
    @ONLY
)

# Add subdirectories
add_subdirectory(src)

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

**cmake/version.h.in:**
```cpp
#pragma once

#define CALCULATOR_VERSION_MAJOR @PROJECT_VERSION_MAJOR@
#define CALCULATOR_VERSION_MINOR @PROJECT_VERSION_MINOR@
#define CALCULATOR_VERSION_PATCH @PROJECT_VERSION_PATCH@
#define CALCULATOR_VERSION "@PROJECT_VERSION@"
```

**src/CMakeLists.txt:**
```cmake
# Create static library
add_library(calclib
    calculator.cpp
)

target_include_directories(calclib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_BINARY_DIR}/generated
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

**include/calculator/math.hpp:**
```cpp
#ifndef CALCULATOR_MATH_HPP
#define CALCULATOR_MATH_HPP

namespace calc {
    double add(double a, double b);
    double subtract(double a, double b);
    double multiply(double a, double b);
    double divide(double a, double b);
}

#endif
```

**src/calculator.cpp:**
```cpp
#include "calculator/math.hpp"
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

**src/main.cpp:**
```cpp
#include "calculator/math.hpp"
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

**tests/CMakeLists.txt:**
```cmake
add_executable(test_calc test_calculator.cpp)
target_link_libraries(test_calc PRIVATE calclib)
add_test(NAME test_calc COMMAND test_calc)
```

**tests/test_calculator.cpp:**
```cpp
#include "calculator/math.hpp"
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

## Project 3: Pure Make Project

### Directory Structure
```
pure_make_project/
├── Makefile
├── include/
│   └── common.h
├── src/
│   ├── common.c
│   └── main.c
└── README.md
```

### Files

**Makefile:**
```makefile
# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2 -g
CPPFLAGS = -Iinclude
LDFLAGS =
LDLIBS =

# Directory settings
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

# Find all source files
SOURCES := $(wildcard $(SRCDIR)/*.c)
OBJECTS := $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
DEPENDS := $(OBJECTS:.o=.d)

# Target executable
TARGET = $(BINDIR)/make_app

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

# Help target
.PHONY: help
help:
	@echo "Makefile targets:"
	@echo "  all      - Build the application (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  rebuild  - Clean and rebuild everything"
	@echo "  run      - Build and run the application"
	@echo "  install  - Install the executable to PREFIX (default: /usr/local)"
	@echo "  help     - Show this help message"
```

**include/common.h:**
```c
#ifndef COMMON_H
#define COMMON_H

extern void print_message(const char* msg);

#endif
```

**src/common.c:**
```c
#include <stdio.h>
#include "common.h"

void print_message(const char* msg) {
    printf("%s\n", msg);
}
```

**src/main.c:**
```c
#include "common.h"

int main() {
    print_message("Hello from pure Make example!");
    return 0;
}
```

## Project 4: Multi-Configuration Make Project

### Directory Structure
```
multi_config_make_project/
├── Makefile
├── Makefile.configs
├── include/
│   └── app.h
├── src/
│   └── main.c
└── README.md
```

### Files

**Makefile.configs:**
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
```

**Makefile:**
```makefile
# Include configuration settings
include Makefile.configs

# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -std=c11
CPPFLAGS = -Iinclude
LDFLAGS =
LDLIBS =

# Directory settings
SRCDIR = src
INCDIR = include

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

**include/app.h:**
```c
#ifndef APP_H
#define APP_H

void print_config_info();

#endif
```

**src/main.c:**
```c
#include <stdio.h>
#include "app.h"

int main() {
    printf("Multi-configuration Make example\n");
    print_config_info();
    return 0;
}

void print_config_info() {
#ifdef DEBUG
    printf("Built in DEBUG mode\n");
#elif defined(PROFILE)
    printf("Built in PROFILE mode\n");
#else
    printf("Built in RELEASE mode\n");
#endif
}
```

## How to Use These Examples

1. **Clone or copy** the example project you want to work with
2. **Navigate** to the project directory
3. **Follow the instructions** in the README.md file
4. **Experiment** with the code and build system to understand how it works

Each example demonstrates different aspects of CMake and Make, from basic usage to advanced features. Practice with these examples to gain hands-on experience with build systems.