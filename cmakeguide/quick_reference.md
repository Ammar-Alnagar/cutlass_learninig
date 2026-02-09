# CMake and Make Quick Reference Guide

## CMake Basics

### Minimum CMake Project
```cmake
cmake_minimum_required(VERSION 3.15)
project(ProjectName)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(myapp main.cpp)
```

### Creating Targets
```cmake
# Executable
add_executable(myapp main.cpp)

# Static library
add_library(mystatic STATIC lib.cpp)

# Shared library
add_library(myshared SHARED lib.cpp)
```

### Target Properties
```cmake
# Include directories
target_include_directories(mytarget
    PUBLIC  include_dir    # For target and users
    PRIVATE src_dir       # For target only
)

# Compile features
target_compile_features(mytarget
    PRIVATE cxx_std_17
)

# Link libraries
target_link_libraries(mytarget
    PRIVATE other_target
    PUBLIC  other_lib
)
```

### Variables
```cmake
# Set variables
set(MY_VAR "value")

# Access variables
${MY_VAR}

# Built-in variables
${CMAKE_SOURCE_DIR}      # Source directory
${CMAKE_BINARY_DIR}      # Build directory
${PROJECT_SOURCE_DIR}    # Project source directory
${PROJECT_BINARY_DIR}    # Project build directory
```

### Conditions
```cmake
if(CONDITION)
    # do something
elseif(OTHER_CONDITION)
    # do something else
else()
    # default
endif()

# Platform detection
if(WIN32)
    # Windows specific
elseif(APPLE)
    # macOS specific
elseif(UNIX)
    # Linux specific
endif()
```

### Loops
```cmake
foreach(item IN ITEMS item1 item2 item3)
    # do something with ${item}
endforeach()

while(CONDITION)
    # loop body
endwhile()
```

## Make Basics

### Basic Makefile Structure
```makefile
# Variables
CC = gcc
CFLAGS = -Wall -Wextra -std=c11
CPPFLAGS = -Iinclude
LDFLAGS =
LDLIBS =

# Targets
target: dependencies
	commands

# Phony targets (not files)
.PHONY: clean all install
clean:
	rm -f *.o program
```

### Pattern Rules
```makefile
# Compile all .c files to .o
%.o: %.c
	$(CC) -c $< -o $@

# $@ = target name, $< = first prerequisite, $^ = all prerequisites
```

### Automatic Dependencies
```makefile
# Enable automatic dependency generation
CFLAGS += -MMD -MP

# Include generated dependencies
-include $(DEPENDS)
```

### Functions
```makefile
# Wildcard
SOURCES := $(wildcard src/*.c)

# Substitute
OBJECTS := $(SOURCES:src/%.c=obj/%.o)

# Shell command
TIMESTAMP := $(shell date +%s)
```

## Common Commands

### CMake Commands
```bash
# Configure project
cmake -B build -S .

# Build project
cmake --build build

# Run tests
ctest --test-dir build

# Install
cmake --install build

# View cache
cmake -LAH build

# Interactive configuration
ccmake build
```

### Make Commands
```bash
# Build default target
make

# Build specific target
make target_name

# Build with multiple jobs
make -j4

# Clean build
make clean

# Show commands without executing
make -n

# Increase verbosity
make V=1

# Print variable value
make print-VAR_NAME
```

## Best Practices

### CMake Best Practices
```cmake
# 1. Use modern CMake (3.15+)
cmake_minimum_required(VERSION 3.15)

# 2. Use target_* commands instead of global ones
target_include_directories(mytarget PRIVATE include/)
# Instead of: include_directories(include/)

# 3. Set C++ standard properly
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 4. Use proper scope for properties
target_compile_features(mytarget PRIVATE cxx_std_17)

# 5. Organize with subdirectories
add_subdirectory(src)
add_subdirectory(tests)
```

### Make Best Practices
```makefile
# 1. Use variables for configuration
CC = gcc
CFLAGS = -Wall -Wextra -std=c11

# 2. Use pattern rules for compilation
%.o: %.c
	$(CC) -c $< -o $@

# 3. Include automatic dependencies
CFLAGS += -MMD -MP
-include $(DEPENDS)

# 4. Mark phony targets
.PHONY: clean all install

# 5. Use silent commands with echo
%.o: %.c
	@echo "CC $<"
	$(CC) -c $< -o $@
```

## Common Patterns

### Multi-Directory CMake Project
```cmake
# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src)
add_subdirectory(libs)
add_subdirectory(tests)

# src/CMakeLists.txt
add_library(mylib
    source1.cpp
    source2.cpp
)

target_include_directories(mylib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE mylib)
```

### Advanced Makefile Pattern
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

# Find sources
SOURCES := $(wildcard $(SRCDIR)/*.c)
OBJECTS := $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
DEPENDS := $(OBJECTS:.o=.d)

# Target
TARGET = $(BINDIR)/myapp

# Enable automatic dependencies
CFLAGS += -MMD -MP

.PHONY: all clean
all: $(TARGET)

$(OBJDIR):
	@mkdir -p $@

$(BINDIR):
	@mkdir -p $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	@echo "CC $<"
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS) | $(BINDIR)
	@echo "LINK $@"
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Include dependencies
-include $(DEPENDS)
```

## Troubleshooting

### Common CMake Issues
```cmake
# 1. Header not found
# Solution: Add include directory
target_include_directories(target PRIVATE include_dir)

# 2. Undefined references
# Solution: Link required libraries
target_link_libraries(target PRIVATE required_lib)

# 3. Outdated builds
# Solution: Ensure proper dependencies
# Use target_* commands properly

# 4. Wrong C++ standard
# Solution: Set standard properly
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

### Common Make Issues
```makefile
# 1. Missing header dependencies
# Solution: Use automatic dependency generation
CFLAGS += -MMD -MP
-include $(DEPENDS)

# 2. Parallel build problems
# Solution: Proper dependencies
program: main.o utils.o  # Explicit dependencies

# 3. Variables not expanding
# Solution: Check syntax
GOOD := immediate_assignment
BAD = deferred_assignment
```

## Debugging Commands

### CMake Debugging
```bash
# Verbose configuration
cmake --debug-output ..

# Print cache variables
cmake -LAH build

# Print specific variable
cmake -LA build | grep VAR_NAME

# Interactive configuration
ccmake build
```

### Make Debugging
```bash
# Dry run (show commands without executing)
make -n

# Verbose output
make --debug=v

# Print variable
make print-VAR_NAME

# Increase verbosity
make V=1
```

## Integration Patterns

### CMake with Custom Commands
```cmake
# Custom target that runs Make
add_custom_target(run-custom-make
    COMMAND ${CMAKE_MAKE_PROGRAM} -f custom.mk
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Custom command for preprocessing
add_custom_command(
    OUTPUT processed_file.cpp
    COMMAND preprocess_script < input_file.cpp > processed_file.cpp
    DEPENDS input_file.cpp
)
```

### Make Calling CMake
```makefile
build_dir:
	mkdir -p build

cmake-build: build_dir
	cd build && cmake .. && make

clean-cmake:
	rm -rf build
```

This quick reference guide provides essential information for working with CMake and Make. Keep it handy for quick lookups of syntax and common patterns.