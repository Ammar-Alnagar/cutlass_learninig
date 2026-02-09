# CMake and Make Hands-On Exercises

This document provides practical exercises to reinforce your understanding of CMake and Make build systems. Each exercise builds on the concepts learned in the main tutorial.

## Exercise 1: Basic CMake Project

**Objective**: Create a simple CMake project with one executable.

**Steps**:
1. Create a directory called `exercise1`
2. Inside this directory, create a file called `main.cpp` with a simple "Hello, World!" program
3. Create a `CMakeLists.txt` file that:
   - Sets the minimum CMake version to 3.10
   - Defines a project called "Exercise1"
   - Sets C++ standard to 17
   - Creates an executable called "hello" from main.cpp
4. Build the project in a separate build directory

**Solution**:
```bash
mkdir exercise1 && cd exercise1
cat > main.cpp << EOF
#include <iostream>

int main() {
    std::cout << "Hello, CMake!" << std::endl;
    return 0;
}
EOF

cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.10)
project(Exercise1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(hello main.cpp)
EOF

mkdir build && cd build
cmake ..
make
./hello
```

## Exercise 2: Adding a Library

**Objective**: Extend the previous project to include a custom library.

**Steps**:
1. Create directories `src` and `include`
2. Move `main.cpp` to `src/`
3. Create a header file `include/greeter.hpp` with a simple class
4. Create a source file `src/greeter.cpp` that implements the class
5. Update `CMakeLists.txt` to:
   - Create a library from `src/greeter.cpp`
   - Set include directories properly
   - Link the library to the executable

**Solution**:
```bash
mkdir -p exercise2/{src,include}
cd exercise2

cat > include/greeter.hpp << EOF
#ifndef GREETER_HPP
#define GREETER_HPP

#include <string>

class Greeter {
public:
    explicit Greeter(const std::string& name);
    std::string greet() const;
    
private:
    std::string name_;
};

#endif
EOF

cat > src/greeter.cpp << EOF
#include "greeter.hpp"

Greeter::Greeter(const std::string& name) : name_(name) {}

std::string Greeter::greet() const {
    return "Hello, " + name_ + "!";
}
EOF

cat > src/main.cpp << EOF
#include "greeter.hpp"
#include <iostream>

int main() {
    Greeter greeter("CMake Learner");
    std::cout << greeter.greet() << std::endl;
    return 0;
}
EOF

cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.10)
project(Exercise2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(greeter_lib src/greeter.cpp)

target_include_directories(greeter_lib
    PUBLIC
        \$<BUILD_INTERFACE:\${CMAKE_CURRENT_SOURCE_DIR}/include>
        \$<INSTALL_INTERFACE:include>
)

add_executable(hello src/main.cpp)
target_link_libraries(hello PRIVATE greeter_lib)
EOF

mkdir build && cd build
cmake ..
make
./hello
```

## Exercise 3: Multi-Directory Project with Tests

**Objective**: Create a project with multiple directories and unit tests.

**Steps**:
1. Create a calculator project with `include/calculator`, `src`, and `tests` directories
2. Implement basic arithmetic operations in the library
3. Create a main application that uses the calculator
4. Write unit tests for the calculator functions
5. Set up CMake to build the library, executable, and tests

**Solution**:
```bash
mkdir -p exercise3/{include/calculator,src,tests}
cd exercise3

cat > include/calculator/math.hpp << EOF
#ifndef CALCULATOR_MATH_HPP
#define CALCULATOR_MATH_HPP

namespace calc {
    double add(double a, double b);
    double subtract(double a, double b);
    double multiply(double a, double b);
    double divide(double a, double b);
}

#endif
EOF

cat > src/calculator.cpp << EOF
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
EOF

cat > src/main.cpp << EOF
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
EOF

cat > tests/test_calculator.cpp << EOF
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
EOF

cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.15)
project(CalculatorExercise)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create library
add_library(calclib
    src/calculator.cpp
)

target_include_directories(calclib
    PUBLIC
        \$<BUILD_INTERFACE:\${PROJECT_SOURCE_DIR}/include>
        \$<INSTALL_INTERFACE:include>
)

# Create executable
add_executable(calc src/main.cpp)
target_link_libraries(calc PRIVATE calclib)

# Enable testing
enable_testing()

# Create test executable
add_executable(test_calc tests/test_calculator.cpp)
target_link_libraries(test_calc PRIVATE calclib)

# Add test to CTest
add_test(
    NAME test_calc
    COMMAND test_calc
)
EOF

mkdir build && cd build
cmake .. -DBUILD_TESTING=ON
make
./calc 10 + 5
./test_calc
ctest -V
```

## Exercise 4: Working with External Dependencies

**Objective**: Use an external library (simulated) in your CMake project.

**Steps**:
1. Create a mock external library directory
2. Update CMakeLists.txt to find and use the external library
3. Modify your code to use the external library functionality

**Solution**:
```bash
mkdir -p exercise4/{include/calculator,src,external}
cd exercise4

# Copy previous files
cp ../exercise3/include/calculator/math.hpp include/calculator/
cp ../exercise3/src/calculator.cpp src/
cp ../exercise3/src/main.cpp src/

cat > external/logger.hpp << EOF
#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <string>

class Logger {
public:
    enum Level { DEBUG, INFO, WARN, ERROR };

    static void log(Level level, const std::string& message) {
        std::string level_str;
        switch(level) {
            case DEBUG: level_str = "DEBUG"; break;
            case INFO:  level_str = "INFO";  break;
            case WARN:  level_str = "WARN";  break;
            case ERROR: level_str = "ERROR"; break;
        }
        std::cout << "[" << level_str << "] " << message << std::endl;
    }

    static void info(const std::string& message) { log(INFO, message); }
    static void warn(const std::string& message) { log(WARN, message); }
    static void error(const std::string& message) { log(ERROR, message); }
    static void debug(const std::string& message) { log(DEBUG, message); }
};
#endif
EOF

cat > src/calculator.cpp << EOF
#include "calculator/math.hpp"
#include "../external/logger.hpp"
#include <stdexcept>

namespace calc {
    double add(double a, double b) {
        Logger::debug("Performing addition: " + std::to_string(a) + " + " + std::to_string(b));
        double result = a + b;
        Logger::info("Addition result: " + std::to_string(result));
        return result;
    }

    double subtract(double a, double b) {
        Logger::debug("Performing subtraction: " + std::to_string(a) + " - " + std::to_string(b));
        double result = a - b;
        Logger::info("Subtraction result: " + std::to_string(result));
        return result;
    }

    double multiply(double a, double b) {
        Logger::debug("Performing multiplication: " + std::to_string(a) + " * " + std::to_string(b));
        double result = a * b;
        Logger::info("Multiplication result: " + std::to_string(result));
        return result;
    }

    double divide(double a, double b) {
        Logger::debug("Performing division: " + std::to_string(a) + " / " + std::to_string(b));
        if (b == 0) {
            Logger::error("Division by zero attempted!");
            throw std::domain_error("Division by zero");
        }
        double result = a / b;
        Logger::info("Division result: " + std::to_string(result));
        return result;
    }
}
EOF

cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.15)
project(ExternalDepExercise)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create library
add_library(calclib
    src/calculator.cpp
)

target_include_directories(calclib
    PUBLIC
        \$<BUILD_INTERFACE:\${PROJECT_SOURCE_DIR}/include>
        \$<INSTALL_INTERFACE:include>
    PRIVATE
        \${PROJECT_SOURCE_DIR}/external
)

# Create executable
add_executable(calc src/main.cpp)
target_link_libraries(calc PRIVATE calclib)

# Enable testing
enable_testing()

# Create test executable
add_executable(test_calc tests/test_calculator.cpp)
target_link_libraries(test_calc PRIVATE calclib)
target_include_directories(test_calc PRIVATE external)

# Add test to CTest
add_test(
    NAME test_calc
    COMMAND test_calc
)
EOF

# Copy test file from previous exercise
mkdir tests
cp ../exercise3/tests/test_calculator.cpp tests/

mkdir build && cd build
cmake ..
make
./calc 10 + 5
```

## Exercise 5: Writing a Makefile from Scratch

**Objective**: Create a Makefile for a multi-file C project without CMake.

**Steps**:
1. Create a project with multiple source files
2. Write a Makefile that compiles all source files
3. Include targets for cleaning, installing, and running

**Solution**:
```bash
mkdir exercise5 && cd exercise5
mkdir -p src include

cat > include/utils.h << EOF
#ifndef UTILS_H
#define UTILS_H

void print_message(const char* msg);
int add_numbers(int a, int b);

#endif
EOF

cat > src/utils.c << EOF
#include <stdio.h>
#include "utils.h"

void print_message(const char* msg) {
    printf("%s\n", msg);
}

int add_numbers(int a, int b) {
    return a + b;
}
EOF

cat > src/main.c << EOF
#include "utils.h"

int main() {
    print_message("Hello from Makefile!");
    int result = add_numbers(5, 7);
    printf("Result: %d\n", result);
    return 0;
}
EOF

cat > Makefile << 'EOF'
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

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
DEPENDS = $(OBJECTS:.o=.d)

# Target executable
TARGET = $(BINDIR)/app

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
	@echo "Available targets:"
	@echo "  all    - Build the application (default)"
	@echo "  clean  - Remove build artifacts"
	@echo "  run    - Build and run the application"
	@echo "  install - Install the executable to PREFIX (default: /usr/local)"
	@echo "  help   - Show this help message"
EOF

make
make run
make help
```

## Exercise 6: Advanced CMake Features

**Objective**: Implement advanced CMake features like custom targets, packaging, and installation.

**Steps**:
1. Create a project with custom build targets
2. Add packaging support with CPack
3. Implement installation rules

**Solution**:
```bash
mkdir -p exercise6/{include/mylib,src,tools}
cd exercise6

cat > include/mylib/core.hpp << EOF
#ifndef MYLIB_CORE_HPP
#define MYLIB_CORE_HPP

#include <string>

namespace mylib {
    class Processor {
    public:
        Processor();
        ~Processor();
        
        std::string process(const std::string& input);
        
    private:
        int id_;
    };
}

#endif
EOF

cat > src/core.cpp << EOF
#include "mylib/core.hpp"
#include <thread>
#include <sstream>

namespace mylib {
    Processor::Processor() {
        id_ = std::hash<std::thread::id>{}(std::this_thread::get_id());
    }
    
    Processor::~Processor() = default;
    
    std::string Processor::process(const std::string& input) {
        std::ostringstream oss;
        oss << "Processed(" << id_ << "): " << input;
        return oss.str();
    }
}
EOF

cat > src/main.cpp << EOF
#include "mylib/core.hpp"
#include <iostream>

int main() {
    mylib::Processor proc;
    std::cout << proc.process("Hello, Advanced CMake!") << std::endl;
    return 0;
}
EOF

cat > tools/generate_version.cpp << EOF
#include <fstream>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: generate_version <output_file>" << std::endl;
        return 1;
    }
    
    std::ofstream file(argv[1]);
    file << "#define APP_VERSION \"1.0.0\"" << std::endl;
    file << "#define BUILD_DATE \"" << __DATE__ << "\"" << std::endl;
    file.close();
    
    std::cout << "Version file generated: " << argv[1] << std::endl;
    return 0;
}
EOF

cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.15)
project(AdvancedExercise VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create library
add_library(mylib
    src/core.cpp
)

target_include_directories(mylib
    PUBLIC
        \$<BUILD_INTERFACE:\${PROJECT_SOURCE_DIR}/include>
        \$<INSTALL_INTERFACE:include>
)

# Create main executable
add_executable(app src/main.cpp)
target_link_libraries(app PRIVATE mylib)

# Create tool executable
add_executable(generate_version tools/generate_version.cpp)

# Custom target to generate version header
add_custom_target(generate-version-header
    COMMAND generate_version "\${CMAKE_CURRENT_BINARY_DIR}/generated/version.h"
    WORKING_DIRECTORY "\${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating version header"
    VERBATIM
)

# Make main executable depend on version header
add_dependencies(app generate-version-header)

# Add include directory for generated files
target_include_directories(app PRIVATE "\${CMAKE_CURRENT_BINARY_DIR}/generated")

# Installation
include(GNUInstallDirs)
install(TARGETS app mylib
    RUNTIME DESTINATION \${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION \${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION \${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION \${CMAKE_INSTALL_INCLUDEDIR}
)

# Packaging
set(CPACK_PACKAGE_VENDOR "Your Name")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Advanced CMake exercise")
set(CPACK_PACKAGE_VERSION_MAJOR \${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR \${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH \${PROJECT_VERSION_PATCH})
set(CPACK_RESOURCE_FILE_LICENSE "\${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "\${PROJECT_SOURCE_DIR}/README.md")

include(CPack)
EOF

# Create dummy license and readme files
touch LICENSE README.md

mkdir build && cd build
cmake ..
make
./app
make package  # This creates a package if CPack is available
```

## Exercise 7: Troubleshooting Common Issues

**Objective**: Practice identifying and fixing common build system problems.

**Scenario 1**: Header not found error
```bash
# Problem: fatal error: myheader.h: No such file or directory
# Solution: Add include directory to target
target_include_directories(target_name PRIVATE include_dir)
```

**Scenario 2**: Undefined reference error
```bash
# Problem: linker error about undefined symbols
# Solution: Link required libraries
target_link_libraries(target_name PRIVATE required_library)
```

**Scenario 3**: Outdated builds
```bash
# Problem: Changes not reflected in output
# Solution: Ensure proper dependencies are set
# In CMake: Use target_* commands properly
# In Make: Use automatic dependency generation (-MMD -MP)
```

## Exercise 8: Real-World Integration

**Objective**: Integrate CMake with external tools and systems.

**Steps**:
1. Create a project that generates compile_commands.json
2. Add a custom target for code formatting
3. Integrate with a documentation generator

**Solution**:
```bash
mkdir exercise8 && cd exercise8
mkdir -p src include

cat > include/app.hpp << EOF
#ifndef APP_HPP
#define APP_HPP

#include <string>

class Application {
public:
    Application(const std::string& name);
    void run();
    
private:
    std::string name_;
};

#endif
EOF

cat > src/app.cpp << EOF
#include "app.hpp"
#include <iostream>

Application::Application(const std::string& name) : name_(name) {}

void Application::run() {
    std::cout << "Running application: " << name_ << std::endl;
}
EOF

cat > src/main.cpp << EOF
#include "app.hpp"

int main() {
    Application app("RealWorldIntegration");
    app.run();
    return 0;
}
EOF

cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.15)
project(RealWorldExercise)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json for IDEs/editors
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Create library
add_library(app_lib
    src/app.cpp
)

target_include_directories(app_lib
    PUBLIC
        \$<BUILD_INTERFACE:\${PROJECT_SOURCE_DIR}/include>
        \$<INSTALL_INTERFACE:include>
)

# Create executable
add_executable(real_world_app src/main.cpp)
target_link_libraries(real_world_app PRIVATE app_lib)

# Code formatting target
find_program(CLANG_FORMAT_EXECUTABLE clang-format)
if(CLANG_FORMAT_EXECUTABLE)
    file(GLOB_RECURSE ALL_SOURCES "src/*.cpp" "include/*.hpp")
    add_custom_target(format
        COMMAND \${CLANG_FORMAT_EXECUTABLE} -i \${ALL_SOURCES}
        COMMENT "Formatting source files"
    )
endif()

# Documentation target
find_program(DOXYGEN_EXECUTABLE doxygen)
if(DOXYGEN_EXECUTABLE)
    configure_file(
        \${PROJECT_SOURCE_DIR}/Doxyfile.in
        \${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        @ONLY
    )

    add_custom_target(docs
        COMMAND \${DOXYGEN_EXECUTABLE} \${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY \${PROJECT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()
EOF

cat > Doxyfile.in << EOF
PROJECT_NAME = "Real World Integration Example"
OUTPUT_DIRECTORY = @CMAKE_CURRENT_BINARY_DIR@/docs
INPUT = @PROJECT_SOURCE_DIR@/src @PROJECT_SOURCE_DIR@/include
GENERATE_HTML = YES
GENERATE_LATEX = NO
QUIET = YES
EOF

mkdir build && cd build
cmake ..
make
./real_world_app

# Check if compile_commands.json was generated
ls -la compile_commands.json

# Try running format target (if clang-format is available)
make format 2>/dev/null || echo "clang-format not available, skipping format target"
```

## Exercise 9: Cross-Platform Development

**Objective**: Create a project that builds on multiple platforms.

**Solution**:
```bash
mkdir -p exercise9/{include,cross_platform,src}
cd exercise9

cat > include/platform.hpp << EOF
#ifndef PLATFORM_HPP
#define PLATFORM_HPP

#include <string>

class Platform {
public:
    static std::string getName();
    static std::string getArchitecture();
};

#endif
EOF

cat > cross_platform/config.h.in << EOF
#pragma once

#cmakedefine PLATFORM_WINDOWS
#cmakedefine PLATFORM_MACOS
#cmakedefine PLATFORM_LINUX

#define APP_VERSION "@PROJECT_VERSION@"
EOF

cat > src/platform.cpp << EOF
#include "platform.hpp"
#include "config.h"  // Generated by CMake

std::string Platform::getName() {
#ifdef PLATFORM_WINDOWS
    return "Windows";
#elif defined(PLATFORM_MACOS)
    return "macOS";
#elif defined(PLATFORM_LINUX)
    return "Linux";
#else
    return "Unknown";
#endif
}

std::string Platform::getArchitecture() {
    #ifdef _WIN64
        return "x64";
    #elif _WIN32
        return "x86";
    #elif __APPLE__
        #include <TargetConditionals.h>
        #if TARGET_CPU_ARM64
            return "arm64";
        #else
            return "x86_64";
        #endif
    #elif __linux__
        #ifdef __aarch64__
            return "arm64";
        #elif __x86_64__
            return "x86_64";
        #else
            return "unknown";
        #endif
    #else
        return "unknown";
    #endif
}
EOF

cat > src/main.cpp << EOF
#include "platform.hpp"
#include <iostream>

int main() {
    std::cout << "Cross-platform demo" << std::endl;
    std::cout << "Platform: " << Platform::getName() << std::endl;
    std::cout << "Architecture: " << Platform::getArchitecture() << std::endl;
    
    return 0;
}
EOF

cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.15)
project(CrossPlatformExercise VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect platform
if(WIN32)
    set(PLATFORM_WINDOWS ON)
elseif(APPLE)
    set(PLATFORM_MACOS ON)
elseif(UNIX)
    set(PLATFORM_LINUX ON)
endif()

# Configure header file
configure_file(
    "\${PROJECT_SOURCE_DIR}/cross_platform/config.h.in"
    "\${PROJECT_BINARY_DIR}/generated/config.h"
    @ONLY
)

# Create library
add_library(platform_lib
    src/platform.cpp
)

target_include_directories(platform_lib
    PUBLIC
        \$<BUILD_INTERFACE:\${PROJECT_SOURCE_DIR}/include>
        \$<BUILD_INTERFACE:\${PROJECT_BINARY_DIR}/generated>
        \$<INSTALL_INTERFACE:include>
)

# Create executable
add_executable(cross_platform_app src/main.cpp)
target_link_libraries(cross_platform_app PRIVATE platform_lib)

# Installation
include(GNUInstallDirs)
install(TARGETS cross_platform_app platform_lib
    RUNTIME DESTINATION \${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION \${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION \${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION \${CMAKE_INSTALL_INCLUDEDIR}
)
EOF

mkdir build && cd build
cmake ..
make
./cross_platform_app
```

## Exercise 10: Complete Project Integration

**Objective**: Combine all learned concepts into a comprehensive project.

**Solution**:
```bash
mkdir -p complete_project/{include/toolkit,src,tests,docs,scripts}
cd complete_project

cat > include/toolkit/data_processor.hpp << EOF
#ifndef DATA_PROCESSOR_HPP
#define DATA_PROCESSOR_HPP

#include <vector>
#include <string>

namespace toolkit {
    class DataProcessor {
    public:
        DataProcessor();
        ~DataProcessor();
        
        std::vector<double> process(const std::vector<double>& input);
        std::string getStatus() const;
        
    private:
        int process_count_;
    };
}

#endif
EOF

cat > src/data_processor.cpp << EOF
#include "toolkit/data_processor.hpp"
#include <algorithm>
#include <numeric>

namespace toolkit {
    DataProcessor::DataProcessor() : process_count_(0) {}
    
    DataProcessor::~DataProcessor() = default;
    
    std::vector<double> DataProcessor::process(const std::vector<double>& input) {
        std::vector<double> result = input;
        
        // Normalize values to range [0, 1]
        if (!result.empty()) {
            double min_val = *std::min_element(result.begin(), result.end());
            double max_val = *std::max_element(result.begin(), result.end());
            
            if (max_val != min_val) {
                for (auto& val : result) {
                    val = (val - min_val) / (max_val - min_val);
                }
            }
        }
        
        process_count_++;
        return result;
    }
    
    std::string DataProcessor::getStatus() const {
        return "Processed " + std::to_string(process_count_) + " datasets";
    }
}
EOF

cat > src/main.cpp << EOF
#include "toolkit/data_processor.hpp"
#include <iostream>
#include <vector>

int main() {
    toolkit::DataProcessor processor;
    
    std::vector<double> data = {1.0, 5.0, 3.0, 9.0, 2.0};
    std::cout << "Original data: ";
    for (double val : data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    auto processed = processor.process(data);
    std::cout << "Processed data: ";
    for (double val : processed) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    std::cout << processor.getStatus() << std::endl;
    
    return 0;
}
EOF

cat > tests/test_data_processor.cpp << EOF
#include "toolkit/data_processor.hpp"
#include <iostream>
#include <vector>
#include <cmath>

bool test_normalization() {
    toolkit::DataProcessor processor;
    std::vector<double> input = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto result = processor.process(input);
    
    // Check that min is 0 and max is 1
    double min_val = *std::min_element(result.begin(), result.end());
    double max_val = *std::max_element(result.begin(), result.end());
    
    bool success = (std::abs(min_val - 0.0) < 1e-10) && (std::abs(max_val - 1.0) < 1e-10);
    std::cout << "Normalization test: " << (success ? "PASS" : "FAIL") << std::endl;
    return success;
}

bool test_empty_input() {
    toolkit::DataProcessor processor;
    std::vector<double> input = {};
    auto result = processor.process(input);
    
    bool success = result.empty();
    std::cout << "Empty input test: " << (success ? "PASS" : "FAIL") << std::endl;
    return success;
}

int main() {
    int failed = 0;
    
    if (!test_normalization()) failed++;
    if (!test_empty_input()) failed++;
    
    std::cout << "\nTests run: 2, Failed: " << failed << std::endl;
    return failed > 0 ? 1 : 0;
}
EOF

cat > scripts/build_debug.sh << 'EOF'
#!/bin/bash
set -e

echo "Building in debug mode..."

if [ ! -d "build_debug" ]; then
    mkdir build_debug
fi

cd build_debug
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)

echo "Debug build completed!"
EOF

cat > scripts/build_release.sh << 'EOF'
#!/bin/bash
set -e

echo "Building in release mode..."

if [ ! -d "build_release" ]; then
    mkdir build_release
fi

cd build_release
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
make -j$(nproc)

echo "Release build completed!"
EOF

chmod +x scripts/*.sh

cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.15)
project(CompleteProject VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_TESTS "Build tests" ON)

# Create library
add_library(toolkit
    src/data_processor.cpp
)

target_include_directories(toolkit
    PUBLIC
        \$<BUILD_INTERFACE:\${PROJECT_SOURCE_DIR}/include>
        \$<INSTALL_INTERFACE:include>
)

# Create executable
add_executable(complete_app src/main.cpp)
target_link_libraries(complete_app PRIVATE toolkit)

# Tests
if(BUILD_TESTS)
    enable_testing()
    
    add_executable(test_toolkit tests/test_data_processor.cpp)
    target_link_libraries(test_toolkit PRIVATE toolkit)
    
    add_test(
        NAME test_data_processor
        COMMAND test_toolkit
    )
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS complete_app toolkit
    RUNTIME DESTINATION \${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION \${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION \${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION \${CMAKE_INSTALL_INCLUDEDIR}
)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Packaging
set(CPACK_PACKAGE_VENDOR "Your Name")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Complete CMake and Make tutorial project")
set(CPACK_PACKAGE_VERSION_MAJOR \${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR \${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH \${PROJECT_VERSION_PATCH})

include(CPack)
EOF

# Build and test the complete project
./scripts/build_debug.sh
cd build_debug
./complete_app
./test_toolkit
ctest -V

echo "All exercises completed successfully!"
```

## Summary

These exercises progressively build your skills in CMake and Make:

1. **Basic projects** - Learn fundamental CMake syntax
2. **Libraries** - Understand target-based design
3. **Testing** - Implement proper testing infrastructure
4. **Dependencies** - Manage external libraries
5. **Makefiles** - Master direct Make usage
6. **Advanced features** - Use custom targets and packaging
7. **Troubleshooting** - Identify and fix common issues
8. **Integration** - Connect with external tools
9. **Cross-platform** - Handle multiple platforms
10. **Complete project** - Combine all concepts

Practice these exercises multiple times to solidify your understanding. Each exercise builds on the previous ones, creating a comprehensive foundation in build systems.