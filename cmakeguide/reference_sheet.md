# CMake and Make Quick Reference Sheet

## Table of Contents
1. [CMake Commands Reference](#cmake-commands-reference)
2. [CMake Variables Reference](#cmake-variables-reference)
3. [CMake Best Practices](#cmake-best-practices)
4. [Make Commands Reference](#make-commands-reference)
5. [Make Variables and Functions](#make-variables-and-functions)
6. [Common Patterns](#common-patterns)
7. [Troubleshooting Tips](#troubleshooting-tips)

## CMake Commands Reference

### Project Setup
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject VERSION 1.0.0 LANGUAGES CXX C)
```

### Creating Targets
```cmake
# Executable
add_executable(my_app src/main.cpp)

# Static library
add_library(mylib STATIC src/lib.cpp)

# Shared library
add_library(mylib SHARED src/lib.cpp)

# Interface library (header-only)
add_library(myinterface INTERFACE)
```

### Target Properties
```cmake
# Include directories
target_include_directories(my_target
    PRIVATE   include/internal
    PUBLIC    include/public
    INTERFACE include/interface
)

# Compile features
target_compile_features(my_target PUBLIC cxx_std_17)

# Compile definitions
target_compile_definitions(my_target PRIVATE MY_DEFINE)

# Compile options
target_compile_options(my_target PRIVATE -Wall)

# Link libraries
target_link_libraries(my_target PRIVATE other_target)
```

### Finding Packages
```cmake
# Find package with components
find_package(Boost REQUIRED COMPONENTS system filesystem)

# Find package with imported targets
find_package(Threads REQUIRED)
target_link_libraries(my_target PRIVATE Threads::Threads)

# Check if found
if(Boost_FOUND)
    target_link_libraries(my_target PRIVATE ${Boost_LIBRARIES})
endif()
```

### Control Flow
```cmake
# Platform checks
if(WIN32)
    # Windows-specific code
elseif(APPLE)
    # macOS-specific code
elseif(UNIX)
    # Unix-specific code
endif()

# Compiler checks
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(my_target PRIVATE -Wall)
endif()

# Build type checks
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(my_target PRIVATE DEBUG_MODE)
endif()
```

### Installation
```cmake
include(GNUInstallDirs)

install(TARGETS my_target
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

## CMake Variables Reference

### Project Information
- `${PROJECT_NAME}` - Name of the current project
- `${PROJECT_VERSION}` - Version of the current project
- `${PROJECT_SOURCE_DIR}` - Root source directory of the current project
- `${PROJECT_BINARY_DIR}` - Root binary directory of the current project

### System Information
- `${CMAKE_SYSTEM_NAME}` - Operating system name (Linux, Windows, Darwin, etc.)
- `${CMAKE_SYSTEM_PROCESSOR}` - Processor architecture
- `${CMAKE_HOST_SYSTEM_NAME}` - Host system name
- `${CMAKE_SIZEOF_VOID_P}` - Size of pointer in bytes (4 or 8)

### Compiler Information
- `${CMAKE_C_COMPILER}` - Path to C compiler
- `${CMAKE_CXX_COMPILER}` - Path to C++ compiler
- `${CMAKE_BUILD_TYPE}` - Build type (Debug, Release, etc.)
- `${CMAKE_GENERATOR}` - Generator used to create build files

### Directory Paths
- `${CMAKE_CURRENT_SOURCE_DIR}` - Current source directory
- `${CMAKE_CURRENT_BINARY_DIR}` - Current binary directory
- `${CMAKE_SOURCE_DIR}` - Root source directory
- `${CMAKE_BINARY_DIR}` - Root binary directory

### Standard Variables
- `${CMAKE_MODULE_PATH}` - List of directories to search for CMake modules
- `${CMAKE_PREFIX_PATH}` - List of prefixes to search for CMake packages
- `${CMAKE_INSTALL_PREFIX}` - Installation prefix

## CMake Best Practices

### Modern CMake (3.0+)
```cmake
# Use modern CMake
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

# Generate version header
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/version.h.in"
    "${PROJECT_BINARY_DIR}/generated/version.h"
    @ONLY
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

### Conditional Compilation
```cmake
# Use generator expressions for conditional compilation
target_compile_options(my_app PRIVATE
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O3>
    $<$<BOOL:${ENABLE_SANITIZER}>:-fsanitize=address>
)
```

## Make Commands Reference

### Basic Syntax
```
target: prerequisites
[TAB] recipe
```

### Common Targets
```makefile
# Phony targets (don't correspond to files)
.PHONY: all clean install test

all: myapp

clean:
	rm -f *.o myapp

install: myapp
	cp myapp /usr/local/bin/

test: myapp
	./myapp --test
```

### Variables
```makefile
# Assignment types
VAR1 = value          # Recursive expansion
VAR2 := value         # Simple expansion
VAR3 ?= value         # Conditional assignment
VAR4 += value         # Append

# Built-in variables
CC = gcc              # C compiler
CXX = g++             # C++ compiler
RM = rm -f            # Remove command
CFLAGS = -Wall        # C compiler flags
CXXFLAGS = -Wall      # C++ compiler flags
LDFLAGS =             # Linker flags
LDLIBS =              # Libraries to link
```

### Automatic Variables
```makefile
$@  # Name of the target
$<  # Name of the first prerequisite
$^  # Names of all prerequisites
$?  # Names of prerequisites newer than target
$*  # Stem in pattern rule
```

### Pattern Rules
```makefile
# Compile C files
%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

# Compile C++ files
%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

# Link executable
%: %.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
```

## Make Variables and Functions

### String Functions
```makefile
# Substitute text
$(subst old,new,text)

# Pattern substitution
$(patsubst %.c,%.o,file1.c file2.c)  # -> file1.o file2.o

# Strip whitespace
$(strip   a b c   )

# Filter by pattern
$(filter %.c %.h,*.c *.h *.o)  # -> *.c *.h

# Filter out by pattern
$(filter-out %.c,*.c *.h *.o)  # -> *.h *.o

# Find substring
$(findstring a,abc)  # -> a
```

### File Functions
```makefile
# Find files matching pattern
$(wildcard *.c)

# Get directory parts
$(dir src/file.c inc/util.h)  # -> src/ inc/

# Get file parts
$(notdir src/file.c inc/util.h)  # -> file.c util.h

# Get suffixes
$(suffix src/file.c inc/util.h)  # -> .c .h

# Get basenames
$(basename src/file.c inc/util.h)  # -> src/file inc/util

# Add suffix/prefix
$(addsuffix .o,file1 file2)  # -> file1.o file2.o
$(addprefix src/,file.c)  # -> src/file.c
```

### Shell and Other Functions
```makefile
# Execute shell command
DATE = $(shell date +%Y%m%d)

# Iterate over list
$(foreach var,list,text)

# Call parameterized text
$(call func,arg1,arg2)

# Evaluate text as makefile
$(eval ...)

# Conditional evaluation
$(if condition,then-part,else-part)
```

## Common Patterns

### CMake: Library with Headers
```cmake
add_library(mylib
    src/impl.cpp
)

target_include_directories(mylib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        src/
)
```

### CMake: Executable linking library
```cmake
add_executable(myapp
    src/main.cpp
)

target_link_libraries(myapp PRIVATE mylib)
```

### CMake: Finding and using a package
```cmake
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

target_link_libraries(myapp PRIVATE
    Threads::Threads
    OpenSSL::SSL
    OpenSSL::Crypto
)
```

### Make: Object files from source
```makefile
SOURCES = $(wildcard src/*.c)
OBJECTS = $(SOURCES:.c=.o)

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@
```

### Make: Directory creation
```makefile
OBJDIR = obj
$(OBJDIR):
	mkdir -p $@

# Create object files in subdirectory
$(OBJDIR)/%.o: src/%.c | $(OBJDIR)
	$(CC) -c $(CFLAGS) $< -o $@
```

### Make: Conditional compilation
```makefile
# Debug vs Release
ifeq ($(DEBUG), 1)
    CFLAGS += -g -DDEBUG
else
    CFLAGS += -O2 -DNDEBUG
endif

# Platform-specific
ifeq ($(OS), Windows_NT)
    RM = del
    CC = gcc.exe
else
    RM = rm -f
    CC = gcc
endif
```

## Troubleshooting Tips

### CMake Common Issues

**Issue**: "Could not find a package configuration file"
- **Solution**: Check `CMAKE_PREFIX_PATH` or `CMAKE_MODULE_PATH`
- **Check**: `find_package(PackageName PATHS /path/to/package NO_DEFAULT_PATH)`

**Issue**: "Target has no SOURCES property"
- **Solution**: Ensure you're adding sources to the target correctly
- **Check**: `target_sources(target_name PRIVATE source.cpp)`

**Issue**: "Include directory not found"
- **Solution**: Use `target_include_directories()` with correct scope
- **Check**: `target_include_directories(target PUBLIC include/)`

**Issue**: "Link error: undefined reference"
- **Solution**: Check linking order and visibility
- **Check**: `target_link_libraries(target PRIVATE dep)`

### Make Common Issues

**Issue**: "Missing separator" error
- **Solution**: Ensure recipes are indented with tabs, not spaces

**Issue**: "Nothing to be done for 'target'" 
- **Solution**: Check if target file is newer than prerequisites

**Issue**: "Variable not expanded correctly"
- **Solution**: Use `:=` for immediate expansion instead of `=`

**Issue**: "Pattern rule not matching"
- **Solution**: Ensure pattern matches exactly, check file extensions

### Debugging Commands

**CMake Debugging**:
```bash
# Verbose output
cmake --debug-output ..

# Trace variable values
cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..

# List variables
cmake -LAH  # List cached variables
```

**Make Debugging**:
```bash
# Verbose output
make VERBOSE=1

# Print variable values
make print-CFLAGS

# Dry run
make -n

# Print database
make -p
```

### Performance Tips

**CMake Performance**:
- Use `target_*` commands instead of global ones
- Minimize `find_package` calls
- Use `ExternalProject_Add` for external dependencies
- Enable ccache if available

**Make Performance**:
- Use `-jN` for parallel builds
- Use `.NOTPARALLEL` for targets that can't be parallelized
- Use pattern rules efficiently
- Minimize shell invocations in recipes