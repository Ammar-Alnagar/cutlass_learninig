# Complete CMake + Make Example Project

This directory contains a complete example project demonstrating both CMake and Make in practical use.

## Project Structure
```
complete_example/
├── CMakeLists.txt              # Main CMake configuration
├── Makefile                    # Top-level Makefile for orchestration
├── src/
│   ├── CMakeLists.txt          # CMake for source directory
│   ├── main.cpp               # Main application
│   ├── calculator.cpp         # Calculator implementation
│   └── utils.cpp              # Utility functions
├── include/
│   ├── calculator.hpp         # Calculator interface
│   └── utils.hpp              # Utility functions interface
├── tests/
│   ├── CMakeLists.txt         # CMake for tests
│   ├── test_calculator.cpp    # Calculator tests
│   └── test_utils.cpp         # Utils tests
├── docs/
│   └── Doxyfile.in            # Template for documentation
├── scripts/
│   ├── build.sh               # Build script
│   └── run_tests.sh           # Test runner script
└── external/                  # External dependencies
    └── README.md              # External deps info
```

## Files

### Main CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.15)
project(CompleteExample VERSION 1.0.0 DESCRIPTION "Complete CMake + Make example")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_DOCS "Build documentation" OFF)

# Add main source directory
add_subdirectory(src)

# Add tests if enabled
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Documentation
if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    if(DOXYGEN_FOUND)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
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
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS calculator_app
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

### Top-Level Makefile
```makefile
# Top-level Makefile for Complete Example Project
SHELL := /bin/bash

# Configuration
BUILD_DIR ?= build
BUILD_TYPE ?= RelWithDebInfo
INSTALL_PREFIX ?= /usr/local
PARALLEL_JOBS ?= 4

# Tools
CMAKE ?= cmake
CTEST ?= ctest
GIT ?= git

# Targets
.PHONY: all cmake-build build clean test install format docs help

# Default target
all: build

# Configure with CMake
cmake-build:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(CMAKE) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX) \
		..

# Build using CMake
build: cmake-build
	$(CMAKE) --build $(BUILD_DIR) --parallel $(PARALLEL_JOBS)

# Clean build directory
clean:
	rm -rf $(BUILD_DIR)

# Run tests
test: build
	cd $(BUILD_DIR) && $(CTEST) --verbose

# Install
install: build
	$(CMAKE) --build $(BUILD_DIR) --target install

# Format code
format:
	find src include tests -name '*.cpp' -o -name '*.hpp' -o -name '*.h' | xargs clang-format -i

# Build documentation
docs:
	cd $(BUILD_DIR) && $(CMAKE) --build . --target docs

# Run static analysis (if available)
static-analysis: build
	if command -v cppcheck >/dev/null 2>&1; then \
		cppcheck --enable=all --inconclusive --force src/; \
	fi

# Package
package: build
	$(CMAKE) --build $(BUILD_DIR) --target package

# Print configuration
config:
	@echo "Build configuration:"
	@echo "  BUILD_DIR = $(BUILD_DIR)"
	@echo "  BUILD_TYPE = $(BUILD_TYPE)"
	@echo "  INSTALL_PREFIX = $(INSTALL_PREFIX)"
	@echo "  PARALLEL_JOBS = $(PARALLEL_JOBS)"

# Help
help:
	@echo "Complete Example Project Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make [TARGET] [VARIABLE=VALUE]"
	@echo ""
	@echo "Targets:"
	@echo "  all             - Build the project (default)"
	@echo "  build           - Build the project"
	@echo "  cmake-build     - Configure with CMake only"
	@echo "  clean           - Remove build directory"
	@echo "  test            - Build and run tests"
	@echo "  install         - Build and install"
	@echo "  format          - Format source code"
	@echo "  docs            - Build documentation"
	@echo "  package         - Create distribution package"
	@echo "  static-analysis - Run static analysis tools"
	@echo "  config          - Print configuration"
	@echo "  help            - Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  BUILD_DIR       - Build directory (default: build)"
	@echo "  BUILD_TYPE      - Build type (default: RelWithDebInfo)"
	@echo "  INSTALL_PREFIX  - Installation prefix (default: /usr/local)"
	@echo "  PARALLEL_JOBS   - Parallel build jobs (default: 4)"

# Debug variable values
print-%:
	@echo $* = $($*)
```

### src/CMakeLists.txt
```cmake
# Create the main executable
add_executable(calculator_app
    main.cpp
    calculator.cpp
    utils.cpp
)

# Create a library
add_library(calclib
    calculator.cpp
    utils.cpp
)

# Set properties for the library
set_target_properties(calclib PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# Include directories
target_include_directories(calclib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

# Link libraries to executable
target_link_libraries(calculator_app PRIVATE calclib)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(calclib PRIVATE -Wall -Wextra -Wpedantic)
endif()
```

### include/calculator.hpp
```cpp
#ifndef CALCULATOR_HPP
#define CALCULATOR_HPP

namespace calc {
    class Calculator {
    public:
        Calculator();
        ~Calculator();
        
        double add(double a, double b);
        double subtract(double a, double b);
        double multiply(double a, double b);
        double divide(double a, double b);
        
        // Get number of operations performed
        int getOperationCount() const;
        
    private:
        int m_operationCount;
    };
}

#endif
```

### include/utils.hpp
```cpp
#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>

namespace utils {
    // Check if a string represents a valid number
    bool isValidNumber(const std::string& str);
    
    // Convert string to double with error checking
    bool stringToDouble(const std::string& str, double& result);
    
    // Get platform information
    std::string getPlatformInfo();
    
    // Simple logging function
    void logMessage(const std::string& msg);
}

#endif
```

### src/main.cpp
```cpp
#include "calculator.hpp"
#include "utils.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <num1> <operator> <num2>\n";
        std::cerr << "Example: " << argv[0] << " 5 + 3\n";
        return 1;
    }

    // Parse arguments
    double a, b;
    if (!utils::stringToDouble(argv[1], a)) {
        std::cerr << "Invalid number: " << argv[1] << std::endl;
        return 1;
    }
    
    std::string op = argv[2];
    
    if (!utils::stringToDouble(argv[3], b)) {
        std::cerr << "Invalid number: " << argv[3] << std::endl;
        return 1;
    }

    // Perform calculation
    calc::Calculator calc;
    double result;
    
    try {
        if (op == "+") {
            result = calc.add(a, b);
        } else if (op == "-") {
            result = calc.subtract(a, b);
        } else if (op == "*") {
            result = calc.multiply(a, b);
        } else if (op == "/") {
            result = calc.divide(a, b);
        } else {
            std::cerr << "Unknown operator: " << op << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << result << std::endl;
    std::cout << "Total operations performed: " << calc.getOperationCount() << std::endl;
    
    return 0;
}
```

### src/calculator.cpp
```cpp
#include "calculator.hpp"
#include <stdexcept>

namespace calc {
    Calculator::Calculator() : m_operationCount(0) {}
    
    Calculator::~Calculator() {}
    
    double Calculator::add(double a, double b) {
        m_operationCount++;
        return a + b;
    }
    
    double Calculator::subtract(double a, double b) {
        m_operationCount++;
        return a - b;
    }
    
    double Calculator::multiply(double a, double b) {
        m_operationCount++;
        return a * b;
    }
    
    double Calculator::divide(double a, double b) {
        m_operationCount++;
        if (b == 0) {
            throw std::domain_error("Division by zero");
        }
        return a / b;
    }
    
    int Calculator::getOperationCount() const {
        return m_operationCount;
    }
}
```

### src/utils.cpp
```cpp
#include "utils.hpp"
#include <sstream>
#include <cctype>

#ifdef _WIN32
#include <windows.h>
#define PLATFORM_NAME "Windows"
#elif __APPLE__
#include <sys/param.h>
#include <sys/sysctl.h>
#define PLATFORM_NAME "macOS"
#elif __linux__
#define PLATFORM_NAME "Linux"
#else
#define PLATFORM_NAME "Unknown"
#endif

namespace utils {
    bool isValidNumber(const std::string& str) {
        if (str.empty()) return false;
        
        size_t start = 0;
        if (str[0] == '-' || str[0] == '+') start = 1;
        
        bool decimalPointFound = false;
        for (size_t i = start; i < str.length(); i++) {
            if (str[i] == '.') {
                if (decimalPointFound) return false;  // Multiple decimal points
                decimalPointFound = true;
            } else if (!std::isdigit(str[i])) {
                return false;  // Invalid character
            }
        }
        
        return true;
    }
    
    bool stringToDouble(const std::string& str, double& result) {
        if (!isValidNumber(str)) {
            return false;
        }
        
        std::stringstream ss(str);
        ss >> result;
        
        // Check if entire string was consumed
        return ss.eof();
    }
    
    std::string getPlatformInfo() {
        return PLATFORM_NAME;
    }
    
    void logMessage(const std::string& msg) {
        std::cout << "[LOG] " << msg << std::endl;
    }
}
```

### tests/CMakeLists.txt
```cmake
# Test executable for calculator
add_executable(test_calculator
    test_calculator.cpp
)

target_link_libraries(test_calculator PRIVATE calclib)

add_test(
    NAME test_calculator
    COMMAND test_calculator
)

# Test executable for utils
add_executable(test_utils
    test_utils.cpp
)

target_link_libraries(test_utils PRIVATE calclib)

add_test(
    NAME test_utils
    COMMAND test_utils
)
```

### tests/test_calculator.cpp
```cpp
#include "calculator.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>

bool testAdd() {
    calc::Calculator calc;
    double result = calc.add(2.5, 3.7);
    bool success = std::abs(result - 6.2) < 1e-10;
    std::cout << "add(2.5, 3.7) = " << result << " (expected: 6.2) " 
              << (success ? "PASS" : "FAIL") << std::endl;
    return success;
}

bool testSubtract() {
    calc::Calculator calc;
    double result = calc.subtract(10.0, 3.0);
    bool success = std::abs(result - 7.0) < 1e-10;
    std::cout << "subtract(10.0, 3.0) = " << result << " (expected: 7.0) " 
              << (success ? "PASS" : "FAIL") << std::endl;
    return success;
}

bool testMultiply() {
    calc::Calculator calc;
    double result = calc.multiply(4.0, 5.0);
    bool success = std::abs(result - 20.0) < 1e-10;
    std::cout << "multiply(4.0, 5.0) = " << result << " (expected: 20.0) " 
              << (success ? "PASS" : "FAIL") << std::endl;
    return success;
}

bool testDivide() {
    calc::Calculator calc;
    double result = calc.divide(15.0, 3.0);
    bool success = std::abs(result - 5.0) < 1e-10;
    std::cout << "divide(15.0, 3.0) = " << result << " (expected: 5.0) " 
              << (success ? "PASS" : "FAIL") << std::endl;
    return success;
}

bool testDivideByZero() {
    calc::Calculator calc;
    bool exceptionThrown = false;
    try {
        calc.divide(5.0, 0.0);
    } catch (const std::domain_error&) {
        exceptionThrown = true;
    }
    std::cout << "divide by zero throws exception: " 
              << (exceptionThrown ? "PASS" : "FAIL") << std::endl;
    return exceptionThrown;
}

bool testOperationCount() {
    calc::Calculator calc;
    calc.add(1, 2);
    calc.multiply(3, 4);
    calc.divide(10, 2);
    bool success = calc.getOperationCount() == 3;
    std::cout << "operation count after 3 ops: " << calc.getOperationCount() 
              << " (expected: 3) " << (success ? "PASS" : "FAIL") << std::endl;
    return success;
}

int main() {
    int failed = 0;
    
    if (!testAdd()) failed++;
    if (!testSubtract()) failed++;
    if (!testMultiply()) failed++;
    if (!testDivide()) failed++;
    if (!testDivideByZero()) failed++;
    if (!testOperationCount()) failed++;
    
    std::cout << "\nTests run: 6, Failed: " << failed << std::endl;
    return failed > 0 ? 1 : 0;
}
```

### tests/test_utils.cpp
```cpp
#include "utils.hpp"
#include <iostream>

bool testIsValidNumber() {
    bool pass = true;
    
    // Valid numbers
    pass &= utils::isValidNumber("123");
    pass &= utils::isValidNumber("-456");
    pass &= utils::isValidNumber("+789");
    pass &= utils::isValidNumber("3.14159");
    pass &= utils::isValidNumber("-2.5");
    
    // Invalid numbers
    pass &= !utils::isValidNumber("");
    pass &= !utils::isValidNumber("abc");
    pass &= !utils::isValidNumber("12.34.56");
    pass &= !utils::isValidNumber("12a34");
    
    std::cout << "isValidNumber tests: " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

bool testStringToDouble() {
    double result;
    
    bool pass = true;
    pass &= utils::stringToDouble("123.45", result) && std::abs(result - 123.45) < 1e-10;
    pass &= utils::stringToDouble("-67.89", result) && std::abs(result - (-67.89)) < 1e-10;
    pass &= !utils::stringToDouble("invalid", result);
    
    std::cout << "stringToDouble tests: " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

bool testGetPlatformInfo() {
    std::string platform = utils::getPlatformInfo();
    bool pass = !platform.empty();
    std::cout << "getPlatformInfo: " << platform << " " 
              << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

int main() {
    int failed = 0;
    
    if (!testIsValidNumber()) failed++;
    if (!testStringToDouble()) failed++;
    if (!testGetPlatformInfo()) failed++;
    
    std::cout << "\nUtils tests run: 3, Failed: " << failed << std::endl;
    return failed > 0 ? 1 : 0;
}
```

### docs/Doxyfile.in
```
# Doxyfile for Complete Example Project
PROJECT_NAME = "Complete Example Project"
PROJECT_NUMBER = "@PROJECT_VERSION@"
OUTPUT_DIRECTORY = @CMAKE_CURRENT_BINARY_DIR@/docs
INPUT = @CMAKE_CURRENT_SOURCE_DIR@/include @CMAKE_CURRENT_SOURCE_DIR@/src
RECURSIVE = YES
GENERATE_HTML = YES
GENERATE_LATEX = NO
QUIET = YES
WARNINGS = YES
WARN_IF_UNDOCUMENTED = NO
EXTRACT_ALL = NO
EXTRACT_PRIVATE = NO
EXTRACT_STATIC = NO
CLASS_DIAGRAMS = NO
HAVE_DOT = NO
```

### scripts/build.sh
```bash
#!/bin/bash

# Build script for Complete Example Project

set -e  # Exit on any error

BUILD_DIR=${1:-build}
BUILD_TYPE=${2:-RelWithDebInfo}

echo "Building Complete Example Project..."
echo "Build directory: $BUILD_DIR"
echo "Build type: $BUILD_TYPE"

# Create build directory
mkdir -p "$BUILD_DIR"

# Configure with CMake
echo "Configuring with CMake..."
cd "$BUILD_DIR"
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..

# Build
echo "Building..."
make -j$(nproc)

echo "Build completed successfully!"
echo "Executable location: $BUILD_DIR/calculator_app"
```

### scripts/run_tests.sh
```bash
#!/bin/bash

# Test runner script for Complete Example Project

set -e  # Exit on any error

BUILD_DIR=${1:-build}

echo "Running tests for Complete Example Project..."
echo "Build directory: $BUILD_DIR"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory '$BUILD_DIR' does not exist."
    echo "Run build.sh first."
    exit 1
fi

cd "$BUILD_DIR"

# Run tests with CTest
echo "Running tests with CTest..."
ctest --verbose

echo "Tests completed!"
```

## How to Use This Example

1. **Build the project:**
   ```bash
   make
   # or
   mkdir build && cd build && cmake .. && make
   ```

2. **Run the application:**
   ```bash
   ./build/calculator_app 10 + 5
   # Output: 15
   ```

3. **Run tests:**
   ```bash
   make test
   # or
   cd build && ctest --verbose
   ```

4. **Clean build:**
   ```bash
   make clean
   ```

5. **Install:**
   ```bash
   make install
   ```

This complete example demonstrates:
- Proper CMake project structure
- Modern CMake practices
- Integration with Make for top-level orchestration
- Testing setup
- Documentation generation
- Cross-platform considerations
- Professional build system organization